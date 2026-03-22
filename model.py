import itertools
import math
from dataclasses import dataclass
from functools import partial
from random import randrange
from typing import Callable

import torch
from torch import Tensor, cat, nn
import torch.nn.functional as F

from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from mlps import (
    LowRankLinear,
    NobleLinear,
    create_mlp as _create_mlp,
    _get_activation_spec,
    _resolve_low_rank_config,
)

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pragma: no cover
    causal_conv1d_fn = None

# FlexAttention compatibility import. Not all environments expose this module.
try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
except ImportError:  # pragma: no cover
    BlockMask = None
    flex_attention = None
    _compiled_flex_attention = None
else:
    try:
        _compiled_flex_attention = torch.compile(flex_attention)
    except Exception:  # pragma: no cover
        _compiled_flex_attention = None


def _get_flex_attention():
    return _compiled_flex_attention or flex_attention

# -----------------------------------------------------------------------------
# FlexAttention kernel options for different GPU architectures
# GPUs with limited shared memory need reduced num_stages/block sizes in backward pass.
# -----------------------------------------------------------------------------

_flex_attention_kernel_options = None


def set_flex_attention_kernel_options(gpu_arch: str | None):
    global _flex_attention_kernel_options
    if gpu_arch in ("blackwell", "ampere"):
        _flex_attention_kernel_options = {
            "num_stages": 1,
            "num_warps": 4,
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 32,
        }
    else:
        _flex_attention_kernel_options = None
    return _flex_attention_kernel_options


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def exists(v):
    return v is not None


def divisible_by(num, den):
    return (num % den) == 0


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def add(x, y):
    return x + y


def _set_param_metadata(param: nn.Parameter, label: str, *, lr_mul: float = 1.0, wd_mul: float = 0.0):
    param.label = label
    param.lr_mul = lr_mul
    param.wd_mul = wd_mul
    return param


@dataclass(frozen=True)
class CanonSettings:
    enabled: bool = False
    set: str = "ABCD"
    first_n: int = 0
    last_n: int = 0
    layers: tuple[int, ...] = ()
    xsa_last_n: int = 0
    xsa_learnable_gate: bool = False
    xsa_gate_init: float = 2.0
    boundary_delta_enabled: bool = False
    boundary_delta_first_n: int = 0
    boundary_delta_gate_vector: bool = False
    boundary_delta_gate_init: float = -4.0
    use_resid_mix: bool = False
    smear_mode: str = "ramen"
    skip_topology: str = "ramen"
    bigram_vocab_size: int = 0
    bigram_dim: int = 0
    kernel: int = 4
    bias: bool = False
    activation: bool = False
    residual: bool = True
    delta_gate: bool = False
    delta_gate_init: float = -4.0
    use_fast_conv1d: bool = True

    @classmethod
    def from_config(cls, canon_config: dict | None, *, num_layers: int) -> "CanonSettings":
        canon_config = canon_config or {}
        layers = tuple(
            int(layer) for layer in canon_config.get("layers", ()) if str(layer).strip()
        )
        layers = tuple(dict.fromkeys(layers))
        first_n = int(canon_config.get("first_n", 0))
        last_n = int(canon_config.get("last_n", 0))
        boundary_delta_first_n = int(canon_config.get("boundary_delta_first_n", 0))
        smear_mode = str(canon_config.get("smear_mode", "ramen")).lower()
        skip_topology = str(canon_config.get("skip_topology", "ramen")).lower()

        if smear_mode not in {"ramen", "canon_vector"}:
            raise ValueError("canon_config.smear_mode must be 'ramen' or 'canon_vector'")
        if skip_topology not in {"ramen", "canon_unet"}:
            raise ValueError("canon_config.skip_topology must be 'ramen' or 'canon_unet'")
        if first_n < 0 or last_n < 0:
            raise ValueError("canon_config.first_n and canon_config.last_n must be >= 0")
        if first_n > num_layers or last_n > num_layers:
            raise ValueError("canon_config.first_n/last_n cannot exceed model_config.num_layers")
        if boundary_delta_first_n < 0 or boundary_delta_first_n > num_layers:
            raise ValueError(
                "canon_config.boundary_delta_first_n must be between 0 and model_config.num_layers"
            )
        if any(layer < 1 or layer > num_layers for layer in layers):
            raise ValueError(
                f"canon_config.layers must use 1-based indices within [1, {num_layers}], got {layers}"
            )

        enabled = bool(canon_config.get("enabled", False))
        return cls(
            enabled=enabled,
            set=str(canon_config.get("set", "ABCD")).upper(),
            first_n=first_n,
            last_n=last_n,
            layers=layers,
            xsa_last_n=int(canon_config.get("xsa_last_n", 0)),
            xsa_learnable_gate=bool(canon_config.get("xsa_learnable_gate", False)),
            xsa_gate_init=float(canon_config.get("xsa_gate_init", 2.0)),
            boundary_delta_enabled=bool(canon_config.get("boundary_delta_enabled", False)),
            boundary_delta_first_n=boundary_delta_first_n,
            boundary_delta_gate_vector=bool(
                canon_config.get("boundary_delta_gate_vector", False)
            ),
            boundary_delta_gate_init=float(
                canon_config.get("boundary_delta_gate_init", -4.0)
            ),
            use_resid_mix=enabled and bool(canon_config.get("use_resid_mix", False)),
            smear_mode=smear_mode,
            skip_topology=skip_topology,
            bigram_vocab_size=int(canon_config.get("bigram_vocab_size", 0)),
            bigram_dim=int(canon_config.get("bigram_dim", 0)),
            kernel=int(canon_config.get("kernel", 4)),
            bias=bool(canon_config.get("bias", False)),
            activation=bool(canon_config.get("activation", False)),
            residual=bool(canon_config.get("residual", True)),
            delta_gate=bool(canon_config.get("delta_gate", False)),
            delta_gate_init=float(canon_config.get("delta_gate_init", -4.0)),
            use_fast_conv1d=bool(canon_config.get("use_fast_conv1d", True)),
        )

    @property
    def explicit_layers(self) -> frozenset[int]:
        return frozenset(layer - 1 for layer in self.layers)

    def layer_kwargs(self) -> dict:
        return dict(
            kernel=self.kernel,
            bias=self.bias,
            activation=self.activation,
            residual=self.residual,
            delta_gate=self.delta_gate,
            delta_gate_init=self.delta_gate_init,
            use_fast_conv1d=self.use_fast_conv1d,
        )

    def boundary_delta_gate_shape(self, dim: int) -> tuple[int, ...]:
        return (dim,) if self.boundary_delta_gate_vector else (1,)

    def use_layer_hooks(self, layer_idx: int, num_layers: int) -> bool:
        if not self.enabled:
            return False
        if self.layers:
            return layer_idx in self.explicit_layers
        if self.first_n > 0 or self.last_n > 0:
            return layer_idx < self.first_n or layer_idx >= num_layers - self.last_n
        return True

    def layer_set_for(self, layer_idx: int, num_layers: int) -> str:
        return self.set if self.use_layer_hooks(layer_idx, num_layers) else ""

    def uses_xsa(self, layer_idx: int, num_layers: int) -> bool:
        return self.enabled and layer_idx >= max(0, num_layers - self.xsa_last_n)

    def uses_boundary_delta(self, layer_idx: int) -> bool:
        return self.enabled and self.boundary_delta_enabled and layer_idx < self.boundary_delta_first_n


def _build_canon_layer(dim: int, canon_settings: CanonSettings) -> "CanonLayer":
    return CanonLayer(dim, **canon_settings.layer_kwargs())


def sinkhorn_log(logits, num_iters=10, tau=0.05):
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.full(
        (n,), -math.log(n), device=logits.device, dtype=logits.dtype
    )

    u = torch.zeros(n, device=Z.device, dtype=Z.dtype)
    v = torch.zeros(n, device=Z.device, dtype=Z.dtype)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(1), dim=0)

    return torch.exp(Z + u.unsqueeze(1) + v.unsqueeze(0)) * n


def zeropower_via_newtonschulz(X, steps=5, eps=1e-7, coeffs=(3.0, -3.2, 1.2)):
    a, b, c = coeffs

    X = X / (X.norm() + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T

    return X


def orthostochastic_project(logits, ns_steps=5, ns_eps=1e-7, ns_coeffs=(3.0, -3.2, 1.2)):
    O = zeropower_via_newtonschulz(logits, steps=ns_steps, eps=ns_eps, coeffs=ns_coeffs)
    return O.square()


# -------------------------------------------------------------------------
# KromHC helpers (Kronecker-product doubly stochastic factor matrices)
# Ref: https://arxiv.org/abs/2601.21579
# -------------------------------------------------------------------------

_kromhc_perm_mats_2x2: dict[str, torch.Tensor] = {}
_kromhc_perm_mats_general: dict[tuple, torch.Tensor] = {}


def get_2x2_perm_matrices(device="cpu"):
    """Returns the two 2x2 permutation matrices: identity and swap. Shape: (2, 2, 2)."""
    return torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )


def factorize_into_twos(n: int):
    """Factorize *n* into a product of 2s. Only power-of-2 *n* is fully supported."""
    if n == 1:
        return []
    factors = []
    remaining = n
    while remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    if remaining > 1:
        factors.append(remaining)
    return factors


def get_all_permutations(n: int):
    """Generate all n! permutation matrices, returned as shape (n!, n, n)."""
    assert n >= 1, "n must be a positive integer"
    perms = list(itertools.permutations(range(n)))
    index = torch.tensor(perms, dtype=torch.long, device="cpu")
    eye = torch.eye(n, dtype=torch.float32, device="cpu")
    return eye[index]


def get_cached_2x2_perms(device):
    """Get cached 2x2 permutation matrices for *device*."""
    dev_key = str(device)
    if dev_key not in _kromhc_perm_mats_2x2:
        _kromhc_perm_mats_2x2[dev_key] = get_2x2_perm_matrices(device)
    return _kromhc_perm_mats_2x2[dev_key]


# -------------------------------------------------------------------------
# Residual stream connections
# Adapted from:
# https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
# -------------------------------------------------------------------------


def get_expand_reduce_stream_functions(
    num_streams, add_stream_embed=False, dim=None, disable=False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), (
            "`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning "
            "an expansion function with stream embeddings added"
        )
        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        expand_fn = _RepeatExpand(num_streams)

    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn


class _RepeatExpand(nn.Module):
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = num_streams

    def forward(self, residuals):
        return repeat(residuals, "b ... -> (b s) ...", s=self.num_streams)


def get_init_and_expand_reduce_stream_functions(
    num_streams, num_fracs=1, dim=None, add_stream_embed=False, disable=None
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


def build_residual_connection_fns(residual_connection_config: dict, model_dim: int):
    """Resolve residual mode and return residual helper fns and initializer."""
    residual_connection_config = residual_connection_config or {}
    residual_connection_mode = residual_connection_config.get("mode", "standard").lower()
    if residual_connection_config.get("disable", False):
        residual_connection_mode = "standard"
    if residual_connection_mode == "residual":
        residual_connection_mode = "standard"
    if residual_connection_mode not in {"standard", "hc", "mhc", "kromhc"}:
        raise ValueError(
            f"Unsupported residual_connection.mode={residual_connection_mode!r}. "
            "Expected one of: standard, hc, mhc, kromhc."
        )

    if residual_connection_mode == "standard":
        return (
            residual_connection_mode,
            nn.Identity(),
            nn.Identity(),
            None,
        )

    residual_num_streams = int(residual_connection_config.get("num_streams", 4))
    residual_num_fracs = int(residual_connection_config.get("num_fracs", 1))
    if residual_num_streams < 1:
        raise ValueError("residual_connection.num_streams must be >= 1")
    if residual_num_fracs < 1:
        raise ValueError("residual_connection.num_fracs must be >= 1")

    if residual_connection_mode == "kromhc":
        if residual_num_streams < 2 or (residual_num_streams & (residual_num_streams - 1)) != 0:
            raise ValueError(
                "residual_connection.num_streams must be a power of 2 and >= 2 "
                f"for kromhc mode, got {residual_num_streams}"
            )

        residual_expand, residual_reduce = get_expand_reduce_stream_functions(
            residual_num_streams
        )
        init_residual_connection = partial(
            KromHC, residual_num_streams, num_fracs=residual_num_fracs
        )
        init_residual_connection = partial(init_residual_connection, dim=model_dim)

        return (
            residual_connection_mode,
            residual_expand,
            residual_reduce,
            init_residual_connection,
        )

    if residual_connection_mode == "mhc" and residual_num_fracs != 1:
        raise ValueError("residual_connection.num_fracs must be 1 when mode='mhc'")

    residual_kwargs = dict(
        tanh=residual_connection_config.get("tanh", True),
        num_fracs=residual_num_fracs,
        mhc=residual_connection_mode == "mhc",
        sinkhorn_iters=residual_connection_config.get("sinkhorn_iters", 10),
        sinkhorn_tau=residual_connection_config.get("sinkhorn_tau", 0.05),
        mhc_h_res_proj=residual_connection_config.get("mhc_h_res_proj", "sinkhorn"),
        ns_steps=residual_connection_config.get("ns_steps", 5),
        ns_eps=residual_connection_config.get("ns_eps", 1e-7),
        ns_coeffs=tuple(residual_connection_config.get("ns_coeffs", (3.0, -3.2, 1.2))),
        mhc_residual_identity_mix=residual_connection_config.get(
            "mhc_residual_identity_mix", False
        ),
        mhc_residual_alpha=residual_connection_config.get(
            "mhc_residual_alpha", 0.01
        ),
    )

    residual_disable = residual_connection_config.get("disable", None)
    init_residual_connection, residual_expand, residual_reduce = (
        get_init_and_expand_reduce_stream_functions(
            residual_num_streams,
            num_fracs=residual_num_fracs,
            disable=residual_disable,
        )
    )
    init_residual_connection = partial(
        init_residual_connection,
        dim=model_dim,
        **residual_kwargs,
    )

    return (
        residual_connection_mode,
        residual_expand,
        residual_reduce,
        init_residual_connection,
    )


# norms


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


class Residual(nn.Module):
    def __init__(
        self,
        *args,
        branch=None,
        residual_transform=None,
        **kwargs,
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = torch.utils._pytree.tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return torch.utils._pytree.tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


class HyperConnections(nn.Module):
    """Residual stream connection module from mHC-hyper-connections."""

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch=None,
        layer_index=None,
        tanh=True,
        channel_first=False,
        dropout=0.0,
        residual_transform=None,
        add_branch_out_to_residual=True,
        num_input_views=1,
        depth_residual_fn=add,
        num_fracs=1,
        mhc=False,
        sinkhorn_iters=10,
        sinkhorn_tau=0.05,
        mhc_h_res_proj="sinkhorn",
        ns_steps=5,
        ns_eps=1e-7,
        ns_coeffs=(3.0, -3.2, 1.2),
        mhc_residual_identity_mix=False,
        mhc_residual_alpha=0.01,
    ):
        super().__init__()

        self.branch = branch
        self.act = nn.Tanh() if tanh else nn.Identity()
        self.has_fracs = num_fracs > 1
        self.num_fracs = num_fracs
        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")
        self.norm = RMSNorm(dim // num_fracs)

        assert num_residual_streams > 0, "`num_residual_streams` must be greater than 0"
        self.num_residual_streams = num_residual_streams
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )

        assert divisible_by(dim, num_fracs), (
            f"feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})"
        )
        dim //= num_fracs

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        init_alpha0 = torch.zeros((num_residual_streams_fracs, num_input_views_fracs))
        init_alpha0[init_residual_index, :] = 1.0
        self.static_alpha = nn.Parameter(
            cat((init_alpha0, torch.eye(num_residual_streams_fracs)), dim=1)
        )
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(dim, num_residual_streams_fracs + num_input_views_fracs)
        )
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            self.static_beta = nn.Parameter(torch.ones(num_residual_streams_fracs))

            dynamic_beta_shape = (dim,) if num_fracs == 1 else (dim, num_fracs)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)
        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

        self.mhc = mhc
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.mhc_h_res_proj = mhc_h_res_proj
        self.ns_steps = ns_steps
        self.ns_eps = ns_eps
        self.ns_coeffs = ns_coeffs
        self.mhc_residual_identity_mix = mhc_residual_identity_mix

        if mhc:
            assert num_fracs == 1, "mhc currently requires num_fracs = 1"
            assert num_input_views == 1, "mhc currently requires num_input_views = 1"
            assert mhc_h_res_proj in ("sinkhorn", "orthostochastic"), (
                "mhc_h_res_proj must be 'sinkhorn' or 'orthostochastic'"
            )

            H_res_init = torch.full((num_residual_streams, num_residual_streams), -8.0)
            H_res_init.fill_diagonal_(0.0)
            self.H_res_logits = nn.Parameter(H_res_init)

            H_pre_init = torch.full((num_residual_streams,), -8.0)
            H_pre_init[init_residual_index] = 0.0
            self.H_pre_logits = nn.Parameter(H_pre_init)

            if add_branch_out_to_residual:
                self.H_post_logits = nn.Parameter(torch.zeros(num_residual_streams))

            if mhc_residual_identity_mix:
                alpha_clamped = max(1e-4, min(1 - 1e-4, mhc_residual_alpha))
                alpha_logit_init = math.log(alpha_clamped / (1 - alpha_clamped))
                self.H_res_alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))

    def width_connection(self, residuals):
        residual_dtype = residuals.dtype
        streams = self.num_residual_streams
        residuals_mixed_source = None
        if self.mhc:
            residuals_mixed_source = self.residual_transform(residuals)

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        if self.mhc:
            if self.channel_first:
                residuals_mixed_source = rearrange(residuals_mixed_source, "b d ... -> b ... d")

            residuals_mixed_source = self.split_fracs(residuals_mixed_source)
            residuals_mixed_source = rearrange(
                residuals_mixed_source, "(b s) ... d -> b ... s d", s=streams
            )
            residuals_mixed_source = residuals_mixed_source.to(residual_dtype)

            if self.mhc_h_res_proj == "orthostochastic":
                S = orthostochastic_project(
                    self.H_res_logits,
                    ns_steps=self.ns_steps,
                    ns_eps=self.ns_eps,
                    ns_coeffs=self.ns_coeffs,
                ).to(residual_dtype)
            else:
                S = sinkhorn_log(self.H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau).to(
                    residual_dtype
                )

            if self.mhc_residual_identity_mix:
                alpha = torch.sigmoid(self.H_res_alpha_logit)
                I = torch.eye(streams, device=S.device, dtype=S.dtype)
                H_res = (1 - alpha) * I + alpha * S
            else:
                H_res = S

            H_pre = F.softmax(self.H_pre_logits, dim=-1).to(residual_dtype)
            H_post = None
            if self.add_branch_out_to_residual:
                H_post = F.softmax(self.H_post_logits, dim=-1).to(residual_dtype)

            residuals_mixed = einsum(
                H_res, residuals_mixed_source, "s t, ... s d -> ... t d"
            )
            branch_input = einsum(H_pre, residuals, "s, ... s d -> ... d")

            if self.channel_first:
                branch_input = rearrange(branch_input, "b ... d -> b d ...")

            branch_input = self.merge_fracs(branch_input)
            residuals_out = rearrange(residuals_mixed, "b ... s d -> (b s) ... d")
            residuals_out = self.merge_fracs(residuals_out)

            if self.channel_first:
                residuals_out = rearrange(residuals_out, "b ... d -> b d ...")

            return (
                branch_input,
                residuals_out,
                dict(beta=H_post, residuals_mixed=residuals_mixed),
            )

        normed = self.norm(residuals).to(self.dynamic_alpha_fn.dtype)
        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        static_alpha = rearrange(self.static_alpha, "(f s) d -> f s d", s=streams)
        alpha = (dynamic_alpha + static_alpha).to(residual_dtype)
        alpha = self.split_fracs(alpha)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = self.act(normed @ self.dynamic_beta_fn)
            if not self.has_fracs:
                dc_weight = rearrange(dc_weight, "... -> ... 1")

            dynamic_beta = dc_weight * self.dynamic_beta_scale
            static_beta = rearrange(self.static_beta, "... (s f) -> ... s f", s=streams).to(
                residual_dtype
            )
            beta = dynamic_beta + static_beta

        mix_h = einsum(alpha, residuals, "... f1 s f2 t, ... f1 s d -> ... f2 t d")
        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = (
                mix_h[..., : self.num_input_views, :],
                mix_h[..., self.num_input_views :, :],
            )
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        branch_input = self.merge_fracs(branch_input)
        residuals = rearrange(residuals, "b ... s d -> (b s) ... d")
        residuals = self.merge_fracs(residuals)

        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")

        residuals = self.residual_transform(residuals)
        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta, residuals_mixed=None):
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)
        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        if self.mhc:
            assert residuals_mixed is not None
            assert beta is not None
            beta = beta.to(branch_output.dtype)
            branch_to_streams = einsum(branch_output, beta, "b ... d, s -> b ... s d")
            output = residuals_mixed + branch_to_streams
            output = rearrange(output, "b ... s d -> (b s) ... d")
            output = self.merge_fracs(output)

            if self.channel_first:
                output = rearrange(output, "b ... d -> b d ...")

            return self.dropout(output)

        output = einsum(
            branch_output,
            beta.to(branch_output.dtype),
            "b ... f1 d, b ... f1 s f2 -> b ... f2 s d",
        )
        output = rearrange(output, "b ... s d -> (b s) ... d")
        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = torch.utils._pytree.tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return torch.utils._pytree.tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


HyperConnections.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)


class KromHC(nn.Module):
    """Kronecker Low-Rank Hyper-Connections (KromHC).

    The H_res matrix is a Kronecker product of small (2x2) doubly stochastic
    factor matrices, guaranteeing exact double stochasticity with O(n^2*C)
    parameters.  Ref: https://arxiv.org/abs/2601.21579
    """

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch=None,
        layer_index=None,
        channel_first=False,
        dropout=0.0,
        residual_transform=None,
        add_branch_out_to_residual=True,
        num_input_views=1,
        depth_residual_fn=add,
        num_fracs=1,
    ):
        super().__init__()

        self.branch = branch
        assert num_fracs >= 1
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")
        assert divisible_by(dim, num_fracs), (
            f"feature dimension ({dim}) must be divisible by `num_fracs` ({num_fracs})"
        )

        dim //= num_fracs

        assert num_residual_streams >= 2, "`num_residual_streams` must be at least 2"
        assert num_residual_streams & (num_residual_streams - 1) == 0, (
            f"`num_residual_streams` must be a power of 2, got {num_residual_streams}"
        )

        self.num_residual_streams = num_residual_streams
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        # width-norm over flattened streams
        self.norm = RMSNorm(dim * num_residual_streams_fracs)

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # Factorize num_residual_streams into 2s for Kronecker structure
        self.factors = factorize_into_twos(num_residual_streams)
        self.num_factors = len(self.factors)

        self.factor_perms: list[int] = []
        total_res_coeffs = 0
        for f in self.factors:
            num_perms = math.factorial(f)
            self.factor_perms.append(num_perms)
            total_res_coeffs += num_perms

        # Cache perm matrices for non-2 factors (future non-power-of-2 support)
        for f in self.factors:
            if f > 2 and (f, "cpu") not in _kromhc_perm_mats_general:
                _kromhc_perm_mats_general[(f, "cpu")] = get_all_permutations(f).to("cpu")

        # --- static alpha: pre-branch selector + Kronecker residual coefficients ---
        init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
        init_alpha0[init_residual_index, :] = 1.0

        init_alpha1 = torch.ones(total_res_coeffs * num_fracs) * -8
        coeff_idx = 0
        for num_perms in self.factor_perms:
            init_alpha1[coeff_idx] = 0.0  # identity perm of each factor
            coeff_idx += num_perms

        self.static_alpha = nn.Parameter(cat([init_alpha0.view(-1), init_alpha1], dim=-1))

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim * num_residual_streams,
                num_fracs * (total_res_coeffs + num_residual_streams * num_input_views),
            )
        )

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        self.total_res_coeffs = total_res_coeffs

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            beta_init = torch.ones(num_residual_streams_fracs) * -1.0
            beta_init[init_residual_index] = 1.0
            self.static_beta = nn.Parameter(beta_init)

            self.dynamic_beta_fn = nn.Parameter(
                torch.zeros(dim * num_residual_streams, num_fracs * num_residual_streams)
            )
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)
        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

    # -- Kronecker H_res builder ------------------------------------------

    def _get_factor_perms(self, factor_size, device):
        if factor_size == 2:
            return get_cached_2x2_perms(device)
        dev_key = str(device)
        if (factor_size, dev_key) not in _kromhc_perm_mats_general:
            _kromhc_perm_mats_general[(factor_size, dev_key)] = (
                get_all_permutations(factor_size).to(device)
            )
        return _kromhc_perm_mats_general[(factor_size, dev_key)]

    def _build_kronecker_hres(self, dynamic_coeffs, static_coeffs, device):
        """Build the H_res matrix via Kronecker product of learned 2x2 DS factors."""
        if len(self.factors) == 0:
            return dynamic_coeffs.new_ones(dynamic_coeffs.shape[:-1] + (1, 1))

        combined_coeffs = self.residual_scale * dynamic_coeffs + static_coeffs

        all_2x2 = all(f == 2 for f in self.factors)

        if all_2x2:
            batch_shape = combined_coeffs.shape[:-1]
            coeffs_reshaped = combined_coeffs.view(*batch_shape, self.num_factors, 2)
            weights = F.softmax(coeffs_reshaped, dim=-1)
            p = weights[..., 0]

            one_minus_p = 1.0 - p
            row0 = torch.stack([p, one_minus_p], dim=-1)
            row1 = torch.stack([one_minus_p, p], dim=-1)
            all_factor_matrices = torch.stack([row0, row1], dim=-2)

            result = all_factor_matrices[..., 0, :, :]
            for k in range(1, self.num_factors):
                mat = all_factor_matrices[..., k, :, :]
                result_exp = result.unsqueeze(-1).unsqueeze(-3)
                mat_exp = mat.unsqueeze(-4).unsqueeze(-2)
                kron = result_exp * mat_exp
                result = kron.reshape(
                    *batch_shape, result.shape[-2] * 2, result.shape[-1] * 2
                )
            return result
        else:
            # Fallback for non-2x2 factors
            factor_matrices = []
            coeff_idx = 0
            for factor_size, num_perms in zip(self.factors, self.factor_perms):
                factor_coeffs = combined_coeffs[..., coeff_idx : coeff_idx + num_perms]
                coeff_idx += num_perms
                perms = self._get_factor_perms(factor_size, device)
                w = F.softmax(factor_coeffs, dim=-1)
                U_k = einsum(w, perms, "... r, r i j -> ... i j")
                factor_matrices.append(U_k)

            result = factor_matrices[0]
            for mat in factor_matrices[1:]:
                result_exp = rearrange(result, "... a1 a2 -> ... a1 1 a2 1")
                mat_exp = rearrange(mat, "... b1 b2 -> ... 1 b1 1 b2")
                kron = result_exp * mat_exp
                result = rearrange(kron, "... a b c d -> ... (a b) (c d)")
            return result

    # -- width / depth / forward ------------------------------------------

    def width_connection(self, residuals):
        residual_dtype = residuals.dtype
        streams = self.num_residual_streams

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        # norm over flattened streams
        normed = rearrange(residuals, "b ... s d -> b ... (s d)", s=streams)
        normed = self.norm(normed)

        if self.add_branch_out_to_residual:
            fused_weights = cat([self.dynamic_alpha_fn, self.dynamic_beta_fn], dim=-1)
        else:
            fused_weights = self.dynamic_alpha_fn
        combined_weight = normed @ fused_weights

        alpha_size = self.dynamic_alpha_fn.shape[-1]
        wc_weight = combined_weight[..., :alpha_size]

        psize = self.num_input_views * streams
        dynamic_pre, dynamic_residual = wc_weight[..., :psize], wc_weight[..., psize:]
        static_pre, static_residual = self.static_alpha[:psize], self.static_alpha[psize:]

        device = combined_weight.device

        alpha_residual = self._build_kronecker_hres(dynamic_residual, static_residual, device)
        alpha_residual = self.split_fracs(alpha_residual)

        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre
        alpha_pre = rearrange(
            alpha_pre, "... (f s v) -> ... s f v", v=self.num_input_views, f=self.num_fracs
        )
        alpha_pre = alpha_pre.sigmoid()

        alpha = cat((alpha_pre, alpha_residual), dim=-1).to(residual_dtype)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = combined_weight[..., alpha_size:]
            dc_weight = rearrange(dc_weight, "... (s f) -> ... s f", s=streams)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, "... (s f) -> ... s f", s=streams)
            beta = dynamic_beta + static_beta
            beta = (beta.sigmoid() * 2).to(residual_dtype)

        mix_h = einsum(alpha, residuals, "... f1 s f2 t, ... f1 s d -> ... f2 t d")

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = (
                mix_h[..., : self.num_input_views, :],
                mix_h[..., self.num_input_views :, :],
            )
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        branch_input = self.merge_fracs(branch_input)
        residuals = rearrange(residuals, "b ... s d -> (b s) ... d")
        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        output = einsum(
            branch_output, beta.to(branch_output.dtype),
            "b ... f1 d, b ... f1 s f2 -> b ... f2 s d",
        )
        output = rearrange(output, "b ... s d -> (b s) ... d")
        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            return add_residual(branch_output)

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            (branch_out, *rest), tree_spec = torch.utils._pytree.tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return torch.utils._pytree.tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


KromHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
KromHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)


class StreamEmbed(nn.Module):
    def __init__(self, num_streams, dim, channel_first=False, expand_to_streams=False):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams
        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):
        if self.expand_to_streams:
            residuals = repeat(residuals, "b ... -> (b s) ...", s=self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, "(b s) d ... -> b ... s d", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, "b ... s d -> (b s) d ...", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "b ... s d -> (b s) ... d", s=self.num_streams)

        return residuals


class AttentionPoolReduceStream(nn.Module):
    def __init__(self, num_streams, dim, channel_first=False):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first
        self.to_attn_logits = nn.Linear(dim, dim, bias=False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):
        if self.channel_first:
            residuals = rearrange(residuals, "(b s) d ... -> b ... s d", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim=-2)
        residuals = reduce(residuals * attn, "b ... s d -> b ... d", "sum")

        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")
        return residuals


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    """Apply rotary position embeddings to input tensor"""
    assert cos.size(0) >= x_BTHD.size(-3)
    rotary_dim = cos.size(-1) * 2
    assert rotary_dim > 0 and rotary_dim <= x_BTHD.size(-1)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x_rot = x_BTHD[..., :rotary_dim]
    x_pass = x_BTHD[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    y = torch.cat((y1, y2), 3)
    if x_pass.numel() == 0:
        return y
    return torch.cat((y, x_pass), dim=-1)


class PositionalEmbedding(nn.Module):
    """Base class for positional embedding modules.
    Subclasses must provide cos/sin buffers and attn_scale."""

    def reset(self):
        raise NotImplementedError

    def apply(self, old_window: int, new_window: int):
        raise NotImplementedError


class CastedLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0
    ):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        # Zero init to match train_gpt.py @Grad62304977
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: Tensor):
        # Simplified version without FP8 for single GPU
        return F.linear(x, self.weight.type_as(x))


# YaRN implementation for dynamic RoPE adaptation (from train_gpt.py @classiclarryd)
class YarnPositionalEmbedding(PositionalEmbedding):
    """
    YaRN (Yet another RoPE extensioN) for dynamic window size adaptation.
    Allows extending context length during training by adjusting RoPE frequencies.
    """

    def __init__(
        self,
        rope_dim: int,
        max_seq_len: int,
        base_freq: float,
        block_size: int,
        initial_attn_scale: float = 0.1,
    ):
        super().__init__()
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len
        self.base_freq = base_freq
        self.block_size = block_size
        self.initial_attn_scale = initial_attn_scale
        self.reset()

    def reset(self):
        """Reset to initial state (called at start of training and after warmup)"""
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / self.base_freq) ** torch.linspace(
            0, 1, steps=self.rope_dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.rope_dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        # Inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan
        self.attn_scale = self.initial_attn_scale

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        """
        Apply YaRN interpolation when window size changes.
        This adjusts the RoPE frequencies to handle longer contexts.
        """
        rotations = self.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq = self.angular_freq * (
            scaling_factor + interpolation_weight * (1 - scaling_factor)
        )
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


class HalfRoPE(PositionalEmbedding):
    """Half-truncated RoPE without dynamic window adaptation.
    Based on legacy Rotary class from train_gpt_single_gpu.py."""

    def __init__(
        self,
        rope_dim: int,
        max_seq_len: int,
        base_freq: float = 1024,
        initial_attn_scale: float = 0.1,
    ):
        super().__init__()
        self.rope_dim = rope_dim
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / base_freq) ** torch.linspace(
            0, 1, steps=rope_dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(rope_dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.attn_scale = initial_attn_scale

    def reset(self):
        pass

    def apply(self, old_window: int, new_window: int):
        pass


class StandardRoPE(PositionalEmbedding):
    """Full-spectrum RoPE: all head_dim // 2 dimensions get non-zero frequencies."""

    def __init__(
        self,
        rope_dim: int,
        max_seq_len: int,
        base_freq: float = 1024,
        initial_attn_scale: float = 0.1,
    ):
        super().__init__()
        self.rope_dim = rope_dim
        angular_freq = (1 / base_freq) ** torch.linspace(
            0, 1, steps=rope_dim // 2, dtype=torch.float32
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.attn_scale = initial_attn_scale

    def reset(self):
        pass

    def apply(self, old_window: int, new_window: int):
        pass


class NoPositionalEmbedding(PositionalEmbedding):
    """No positional encoding (NoPE/NoPos).

    Sets cos=1, sin=0 so rotary() is a no-op. The model learns implicit
    positional information from the causal attention mask alone.

    Refs:
    - Kazemnejad et al., "The Impact of Positional Encoding on Length
      Generalization in Transformers", NeurIPS 2023 (arXiv:2305.19466)
    - Haviv et al., "Transformer Language Models without Positional
      Encodings Still Learn Positional Information", EMNLP 2022
      (arXiv:2203.16634)
    """

    def __init__(self, rope_dim: int, max_seq_len: int, initial_attn_scale: float = 0.1):
        super().__init__()
        self.rope_dim = rope_dim
        self.cos = nn.Buffer(
            torch.ones(max_seq_len, rope_dim // 2, dtype=torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            torch.zeros(max_seq_len, rope_dim // 2, dtype=torch.bfloat16), persistent=False
        )
        self.attn_scale = initial_attn_scale

    def reset(self):
        pass

    def apply(self, old_window: int, new_window: int):
        pass


def create_positional_embedding(rope_config: dict, head_dim: int, max_seq_len: int, block_size: int):
    """Factory for positional embedding modules."""
    rope_type = rope_config.get("type", "yarn")
    base_freq = rope_config.get("base_freq", 1024)
    initial_attn_scale = rope_config.get("initial_attn_scale", 0.1)
    rope_dim = int(rope_config.get("rope_dims", 0) or head_dim)
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2 != 0:
        raise ValueError(
            f"rope_config.rope_dims must be 0 or an even integer in [2, {head_dim}], got {rope_dim}"
        )
    if rope_type in {"yarn", "half_rope"} and rope_dim % 4 != 0:
        raise ValueError(f"rope_config.rope_dims must be divisible by 4 for rope type {rope_type!r}")

    if rope_type == "yarn":
        return YarnPositionalEmbedding(
            rope_dim, max_seq_len, base_freq, block_size, initial_attn_scale
        )
    if rope_type == "half_rope":
        return HalfRoPE(rope_dim, max_seq_len, base_freq, initial_attn_scale)
    if rope_type == "rope":
        return StandardRoPE(rope_dim, max_seq_len, base_freq, initial_attn_scale)
    if rope_type in ("none", "nope"):
        return NoPositionalEmbedding(rope_dim, max_seq_len, initial_attn_scale)

    supported = ["yarn", "half_rope", "rope", "none", "nope"]
    raise ValueError(f"Unsupported rope type: {rope_type!r}. Supported: {', '.join(supported)}")


class CanonLayer(nn.Module):
    """Depthwise causal convolution used for Canon A/B/C/D placements."""

    def __init__(
        self,
        dim: int,
        kernel: int = 4,
        *,
        bias: bool = False,
        activation: bool = False,
        residual: bool = True,
        delta_gate: bool = False,
        delta_gate_init: float = -4.0,
        use_fast_conv1d: bool = True,
    ):
        super().__init__()
        if kernel <= 0:
            raise ValueError(f"canon kernel must be positive, got {kernel}")
        self.kernel = kernel
        self.activation = activation
        self.residual = residual
        self.delta_gate = delta_gate
        self.use_fast_conv1d = bool(use_fast_conv1d) and causal_conv1d_fn is not None and kernel in (2, 3, 4)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, groups=dim, bias=bias, padding=kernel - 1)
        if delta_gate:
            self.delta_gate_logit = nn.Parameter(torch.tensor(float(delta_gate_init), dtype=torch.float32))
        else:
            self.register_parameter("delta_gate_logit", None)

    def forward(self, x: Tensor) -> Tensor:
        x_conv = x.transpose(1, 2).contiguous()
        weight = self.conv.weight
        bias = self.conv.bias
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        if self.use_fast_conv1d:
            y = causal_conv1d_fn(
                x=x_conv,
                weight=weight.squeeze(1),
                bias=bias,
                activation="silu" if self.activation else None,
            ).transpose(1, 2)
        else:
            y = F.conv1d(x_conv, weight, bias, groups=self.conv.groups, padding=self.kernel - 1)
            y = y[..., : x.size(1)].transpose(1, 2)
            if self.activation:
                y = F.silu(y)
        if self.delta_gate:
            gate = torch.sigmoid(self.delta_gate_logit.to(dtype=x.dtype))
            return x + gate * y if self.residual else gate * y
        return x + y if self.residual else y


class VectorSmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor, prev_state: Tensor | None = None) -> Tensor:
        gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        if prev_state is None:
            x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        else:
            x_prev = prev_state.view(1, 1, -1)
        return (1 - gate) * x + gate * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        layer_idx: int,
        gating_config: dict,
        value_embed_layers: list,
        value_embed_gate_scale: float,
        low_rank_config: dict | None = None,
        qk_gain_init: float = 1.0,
        canon_settings: CanonSettings | None = None,
        use_xsa: bool = False,
        use_canon_b: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dim = dim
        self.layer_idx = layer_idx
        self.value_embed_gate_scale = value_embed_gate_scale
        self.enable_gqa = num_kv_heads != num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"model_config.num_heads ({num_heads}) must be divisible by model_config.num_kv_heads ({num_kv_heads})"
            )
        hdim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.qkv_dim = dim + 2 * self.kv_dim
        self.group_size = num_heads // num_kv_heads
        low_rank_config = _resolve_low_rank_config(low_rank_config)
        self.low_rank_mode = low_rank_config["mode"]
        self.use_low_rank = low_rank_config["enabled"] and low_rank_config["apply_attention"]
        self.use_factorized = self.use_low_rank and self.low_rank_mode == "factorized"
        self.use_noble = self.use_low_rank and self.low_rank_mode == "noble"
        canon_settings = canon_settings or CanonSettings()

        assert hdim == dim, "num_heads * head_dim must equal model_dim"
        if self.use_low_rank and self.enable_gqa:
            raise ValueError("GQA is not implemented for low_rank_config.apply_attention=True in model.py")
        std = dim**-0.5
        bound = (3**0.5) * std  # improved init scale by @YouJiacheng

        self.low_rank_pairs: list[tuple[Tensor, Tensor]] = []

        # Merged QKVO weights (from train_gpt.py)
        # Layout: [Q, K, V, O] each of size (dim, hdim)
        if self.use_factorized:
            self.qkvo_w = LowRankLinear(
                in_features=dim,
                out_features=self.qkv_dim + dim,
                rank_ratio=low_rank_config["rank_ratio"],
                rank=low_rank_config["rank"],
                min_rank=low_rank_config["min_rank"],
                max_rank=low_rank_config["max_rank"],
                label="attn",
                lr_mul=1.0,
                wd_mul=1.0,
            )
            with torch.no_grad():
                self.qkvo_w.A.uniform_(-bound, bound)
                self.qkvo_w.B.uniform_(-bound, bound)
                self.qkvo_w.A[self.qkv_dim :].zero_()  # init O weights to zero
            self.low_rank_pairs = [(self.qkvo_w.A, self.qkvo_w.B)]
        else:
            self.qkvo_w = nn.Parameter(torch.empty(self.qkv_dim + dim, dim))
            self.qkvo_w.label = "attn"
            self.qkvo_w.lr_mul = 1.0
            self.qkvo_w.wd_mul = 1.0
            self.qkvo_w.muon_split_heads = num_heads  # for per-head orthogonalization
            with torch.no_grad():
                qkv_bound = 0.5 * bound if self.use_noble else bound
                self.qkvo_w[: self.qkv_dim].uniform_(-qkv_bound, qkv_bound)  # init QKV weights
                self.qkvo_w[self.qkv_dim :].zero_()  # init O weights to zero

        if self.use_noble:
            noble_kwargs = dict(
                rank_ratio=low_rank_config["rank_ratio"],
                rank=low_rank_config["rank"],
                min_rank=low_rank_config["min_rank"],
                max_rank=low_rank_config["max_rank"],
                label="attn",
                up_init_alpha=low_rank_config["noble_up_init_alpha"],
                lr_power=low_rank_config["noble_lr_power"],
                mix_lr_power=low_rank_config["noble_mix_lr_power"],
                freq_lr_mul=low_rank_config["noble_freq_lr_mul"],
                phase_lr_mul=low_rank_config["noble_phase_lr_mul"],
                freq_min=low_rank_config["noble_freq_min"],
                freq_max=low_rank_config["noble_freq_max"],
                phase_std=low_rank_config["noble_phase_std"],
            )
            self.qkv_noble = NobleLinear(
                in_features=dim,
                out_features=self.qkv_dim,
                **noble_kwargs,
            )
            self.o_noble = NobleLinear(
                in_features=dim,
                out_features=dim,
                **noble_kwargs,
            )
        else:
            self.qkv_noble = None
            self.o_noble = None

        self.q_gain = _set_param_metadata(
            nn.Parameter(torch.full((num_heads,), float(qk_gain_init), dtype=torch.float32)),
            "q_gain",
        )

        # Sparse gated attention (from train_gpt.py @classiclarryd)
        gate_input_dim = gating_config["gate_input_dim"]
        if gating_config.get("use_attn_gate", True):
            self.attn_gate = CastedLinear(gate_input_dim, num_heads)
            self.attn_gate.weight.label = "attn_gate"
        else:
            self.attn_gate = None

        # Value embedding gate (only on specific layers)
        if gating_config.get("use_value_embed_gate", True) and layer_idx in value_embed_layers:
            self.value_embed_gate = CastedLinear(gate_input_dim, num_kv_heads)
            self.value_embed_gate.weight.label = "value_embed_gate"
        else:
            self.value_embed_gate = None

        self.use_xsa = bool(use_xsa)
        if self.use_xsa and canon_settings.xsa_learnable_gate:
            self.xsa_gate = _set_param_metadata(
                nn.Parameter(torch.tensor(canon_settings.xsa_gate_init, dtype=torch.float32)),
                "xsa_gate",
            )
        else:
            self.register_parameter("xsa_gate", None)

        if use_canon_b:
            self.canon_b = _build_canon_layer(self.qkv_dim, canon_settings)
        else:
            self.canon_b = None

    def _project_qkv(self, x: Tensor, sa_lambda: Tensor):
        if self.use_factorized:
            qkv = self.qkvo_w(x)[:, :, : self.qkv_dim]
            return sa_lambda * qkv

        qkv = F.linear(x, self.qkvo_w[: self.qkv_dim].type_as(x))
        if self.qkv_noble is not None:
            qkv = qkv + self.qkv_noble(x)
        return sa_lambda * qkv

    def _project_o(self, x: Tensor, sa_lambda: Tensor):
        if self.use_factorized:
            o_A = self.qkvo_w.A[self.qkv_dim :, :].type_as(x)
            y = F.linear(x, self.qkvo_w.B.type_as(x))
            y = F.linear(y, o_A)
            return sa_lambda * y

        y = F.linear(x, self.qkvo_w[self.qkv_dim :].type_as(x))
        if self.o_noble is not None:
            y = y + self.o_noble(x)
        return sa_lambda * y

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        bsz, heads, seqlen, head_dim = y.shape
        y_grouped = y.reshape(bsz, self.num_kv_heads, self.group_size, seqlen, head_dim)
        v_norm = F.normalize(v, dim=-1).unsqueeze(2)
        proj = (y_grouped * v_norm).sum(dim=-1, keepdim=True) * v_norm
        return (y_grouped - proj).reshape(bsz, heads, seqlen, head_dim)

    def _apply_value_embed(self, v: Tensor, ve: Tensor | None, x: Tensor) -> Tensor:
        if ve is None:
            return v
        B, T = x.shape[:2]
        ve = ve.reshape(B, T, self.num_heads, self.head_dim)
        if self.enable_gqa:
            ve = ve.reshape(B, T, self.num_kv_heads, self.group_size, self.head_dim).mean(dim=3)
        if self.value_embed_gate is not None:
            ve_gate_out = self.value_embed_gate_scale * torch.sigmoid(
                self.value_embed_gate(x[..., : self.value_embed_gate.weight.size(-1)])
            ).reshape(B, T, self.num_kv_heads, 1)
            return v + ve_gate_out * ve
        return v + ve

    def _apply_xsa(self, y: Tensor, v: Tensor) -> Tensor:
        if not self.use_xsa:
            return y
        y_xsa = self._xsa_efficient(y, v)
        if self.xsa_gate is None:
            return y_xsa
        alpha = torch.sigmoid(self.xsa_gate.to(dtype=y.dtype))
        return y + alpha * (y_xsa - y)

    def forward(
        self,
        x: Tensor,
        ve: Tensor,
        sa_lambdas: Tensor,
        block_mask,
        cos: Tensor,
        sin: Tensor,
        attn_scale: float,
        docs: Tensor,
        key_offset: bool = False,
        kv_cache: dict | None = None,
        cache_docs: Tensor | None = None,
    ):
        B, T = x.size(0), x.size(1)
        # Apply sa_lambdas[0] to QKV weights (from train_gpt.py)
        qkv = self._project_qkv(x, sa_lambdas[0])
        if self.canon_b is not None:
            qkv = self.canon_b(qkv)
        q, k, v = qkv.split((self.dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # QK norm and RoPE
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # Key offset: shift keys forward for the stationary head dims (from train_gpt.py)
        # Enables 1-layer induction on long attention window layers
        if key_offset:
            k[:, 1:, :, self.head_dim // 4 : self.head_dim // 2] = k[
                :, :-1, :, self.head_dim // 4 : self.head_dim // 2
            ].clone()
            k[:, 1:, :, 3 * self.head_dim // 4 :] = k[:, :-1, :, 3 * self.head_dim // 4 :].clone()

        # Value embedding with gating (from train_gpt.py)
        v = self._apply_value_embed(v, ve, x)
        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)
        v_heads = v.transpose(1, 2)

        if kv_cache is not None and T == 1 and kv_cache.get("k") is not None:
            cache_k = kv_cache.get("k")
            cache_v = kv_cache.get("v")
            cache_len = 0 if cache_k is None else cache_k.shape[2]

            if key_offset and cache_len > 0:
                prev_k = cache_k[:, :, -1:, :]
                k_heads[:, :, :, self.head_dim // 4 : self.head_dim // 2] = prev_k[
                    :, :, :, self.head_dim // 4 : self.head_dim // 2
                ]
                k_heads[:, :, :, 3 * self.head_dim // 4 :] = prev_k[:, :, :, 3 * self.head_dim // 4 :]
            # Append current token cache to existing prefix.
            if cache_k is None:
                cat_k = k_heads
                cat_v = v_heads
            else:
                cat_k = torch.cat([cache_k, k_heads], dim=2)
                cat_v = torch.cat([cache_v, v_heads], dim=2)

            # Include current token for self-attention; allow only same-document positions.
            if cache_docs is None:
                cache_docs = torch.empty(0, dtype=torch.int32, device=x.device)
            current_doc = docs[0]
            full_docs = torch.cat([cache_docs.to(torch.int32), current_doc.view(1)])
            same_doc = full_docs == current_doc
            attn_mask = torch.full(
                (1, 1, 1, cat_k.size(2)),
                float("-inf"),
                device=x.device,
                dtype=q_heads.dtype,
            )
            attn_mask.masked_fill_(same_doc.view(1, 1, 1, -1), 0.0)
            y = F.scaled_dot_product_attention(
                q_heads,
                cat_k,
                cat_v,
                attn_mask=attn_mask,
                is_causal=False,
                scale=float(attn_scale),
                enable_gqa=self.enable_gqa,
            )
            y = self._apply_xsa(y, v_heads)

            # Attention gating (from train_gpt.py)
            if self.attn_gate is not None:
                y = y * torch.sigmoid(
                    self.attn_gate(x[..., : self.attn_gate.weight.size(-1)])
                ).transpose(1, 2).unsqueeze(-1)

            y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)

            # Output projection using merged weights with sa_lambdas[1]
            y = self._project_o(y, sa_lambdas[1])

            kv_cache["k"] = cat_k
            kv_cache["v"] = cat_v
            return y

        # Element-wise mask for FlexAttention
        def score_mod(score, b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            mask = causal_mask & document_mask
            return torch.where(mask, score, -float("inf"))

        # FlexAttention
        if block_mask is not None and _get_flex_attention() is not None:
            y = _get_flex_attention()(
                q_heads,
                k_heads,
                v_heads,
                block_mask=block_mask,
                scale=attn_scale,
                score_mod=score_mod,
                enable_gqa=self.enable_gqa,
                kernel_options=_flex_attention_kernel_options,
            )
        else:
            y = F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                attn_mask=None,
                is_causal=True,
                scale=float(attn_scale),
                enable_gqa=self.enable_gqa,
            )
        y = self._apply_xsa(y, v_heads)

        # Attention gating (from train_gpt.py)
        if self.attn_gate is not None:
            y = y * torch.sigmoid(
                self.attn_gate(x[..., : self.attn_gate.weight.size(-1)])
            ).transpose(1, 2).unsqueeze(-1)

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)

        # Output projection using merged weights with sa_lambdas[1]
        y = self._project_o(y, sa_lambdas[1])

        if kv_cache is not None and kv_cache.get("k") is None:
            kv_cache["k"] = k_heads
            kv_cache["v"] = v_heads
        return y


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        layer_idx: int,
        head_dim: int,
        skip_attention: bool,
        c_proj_lr_mul: float,
        mlp_std_scale: float,
        value_embed_gate_scale: float,
        gating_config: dict = None,
        value_embed_layers: list = None,
        activation: str = "relu_squared",
        ffn_dim: int | None = None,
        mlp_type: str = "default",
        mlp_kwargs: dict = None,
        low_rank_config: dict | None = None,
        residual_connection: nn.Module = None,
        qk_gain_init: float = 1.0,
        ln_scale: bool = False,
        canon_settings: CanonSettings | None = None,
        canon_set: str = "",
        use_xsa: bool = False,
        use_boundary_delta: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        gating_config = gating_config or {}
        value_embed_layers = value_embed_layers or []
        low_rank_config = low_rank_config or {}
        canon_settings = canon_settings or CanonSettings()
        self.canon_enabled = canon_settings.enabled
        self.use_resid_mix = self.canon_enabled and canon_settings.use_resid_mix
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

        # Skip attention of specific layers (e.g., layer 6 in train_gpt.py) by @YouJiacheng
        if not skip_attention:
            self.attn = CausalSelfAttention(
                dim,
                num_heads,
                num_kv_heads,
                max_seq_len,
                head_dim,
                layer_idx,
                gating_config,
                value_embed_layers,
                value_embed_gate_scale=value_embed_gate_scale,
                low_rank_config=low_rank_config,
                qk_gain_init=qk_gain_init,
                canon_settings=canon_settings,
                use_xsa=use_xsa,
                use_canon_b=self.canon_enabled and "B" in canon_set,
            )
        else:
            self.attn = None

        hidden_transform = None
        if self.canon_enabled and "D" in canon_set:
            hidden_dim = int(ffn_dim if ffn_dim is not None else 4 * dim)
            hidden_transform = _build_canon_layer(hidden_dim, canon_settings)

        # FFN via factory — mlp_type selects the variant.
        self.mlp = _create_mlp(
            mlp_type=mlp_type,
            dim=dim,
            c_proj_lr_mul=c_proj_lr_mul,
            std_scale=mlp_std_scale,
            activation=activation,
            ffn_dim=ffn_dim,
            low_rank_config=low_rank_config,
            hidden_transform=hidden_transform,
            **(mlp_kwargs or {}),
        )
        self.residual_connection = residual_connection
        self.attn_scale = None
        self.mlp_scale = None
        self.resid_mix = None
        self.boundary_delta_gate = None
        self.canon_a = None
        self.canon_c = None

        if self.canon_enabled:
            self.attn_scale = _set_param_metadata(
                nn.Parameter(torch.ones(dim, dtype=torch.float32)),
                "attn_scale",
            )
            self.mlp_scale = _set_param_metadata(
                nn.Parameter(torch.ones(dim, dtype=torch.float32)),
                "mlp_scale",
            )

            if self.use_resid_mix:
                self.resid_mix = _set_param_metadata(
                    nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float()),
                    "resid_mix",
                )

            if use_boundary_delta:
                self.boundary_delta_gate = _set_param_metadata(
                    nn.Parameter(
                        torch.full(
                            canon_settings.boundary_delta_gate_shape(dim),
                            canon_settings.boundary_delta_gate_init,
                            dtype=torch.float32,
                        )
                    ),
                    "boundary_delta_gate",
                )

            if "A" in canon_set:
                self.canon_a = _build_canon_layer(dim, canon_settings)
            if "C" in canon_set:
                self.canon_c = _build_canon_layer(dim, canon_settings)

    def _forward_impl(
        self,
        x: Tensor,
        ve: Tensor,
        sa_lambdas: Tensor,
        block_mask,
        cos: Tensor,
        sin: Tensor,
        attn_scale: float,
        docs: Tensor,
        key_offset: bool = False,
        kv_cache: dict | None = None,
        cache_docs: Tensor | None = None,
        x0: Tensor | None = None,
        return_delta: bool = False,
    ):
        residual = x
        if self.resid_mix is not None:
            if x0 is None:
                raise ValueError("Canon resid_mix requires x0 to be passed into Block.forward")
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.attn is not None:
            if self.canon_enabled:
                attn_input = norm(x) * self.ln_scale_factor
                if self.boundary_delta_gate is not None:
                    x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
                    gate = torch.sigmoid(self.boundary_delta_gate.to(dtype=x.dtype))
                    attn_input = attn_input + (x - x_prev) * gate.view(1, 1, -1)
                if self.canon_a is not None:
                    attn_input = self.canon_a(attn_input)
            else:
                attn_input = norm(x)

            attn_out = self.attn(
                attn_input,
                ve,
                sa_lambdas,
                block_mask,
                cos,
                sin,
                attn_scale,
                docs,
                key_offset,
                kv_cache=kv_cache,
                cache_docs=cache_docs,
            )
            if self.attn_scale is not None:
                attn_out = attn_out * self.attn_scale.to(dtype=x.dtype)[None, None, :]
            x = x + attn_out

        if self.canon_enabled:
            mlp_input = norm(x) * self.ln_scale_factor
            if self.canon_c is not None:
                mlp_input = self.canon_c(mlp_input)
        else:
            mlp_input = norm(x) if getattr(self.mlp, "needs_external_norm", True) else x

        mlp_out = self.mlp(mlp_input)
        if self.mlp_scale is not None:
            mlp_out = mlp_out * self.mlp_scale.to(dtype=x.dtype)[None, None, :]
        x = x + mlp_out
        return x - residual if return_delta else x

    def forward(
        self,
        x: Tensor,
        ve: Tensor,
        sa_lambdas: Tensor,
        block_mask,
        cos: Tensor,
        sin: Tensor,
        attn_scale: float,
        docs: Tensor,
        key_offset: bool = False,
        kv_cache: dict | None = None,
        cache_docs: Tensor | None = None,
        x0: Tensor | None = None,
    ):
        if self.residual_connection is not None:
            return self.residual_connection(
                x,
                ve,
                sa_lambdas,
                block_mask,
                cos,
                sin,
                attn_scale,
                docs,
                key_offset,
                kv_cache=kv_cache,
                cache_docs=cache_docs,
                x0=x0,
            )

        return self._forward_impl(
            x,
            ve,
            sa_lambdas,
            block_mask,
            cos,
            sin,
            attn_scale,
            docs,
            key_offset,
            kv_cache=kv_cache,
            cache_docs=cache_docs,
            x0=x0,
        )


class GPT(nn.Module):
    def __init__(
        self,
        model_config: dict,
        attention_config: dict,
        lambda_config: dict,
        lr_multipliers: dict,
        max_seq_len: int,
        attention_pattern_config: dict,
        gating_config: dict = None,
        skip_config: dict = None,
        rope_config: dict = None,
        embed_config: dict = None,
        canon_config: dict = None,
        low_rank_config: dict = None,
        residual_connection_config: dict = None,
        wd_multipliers: dict = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.attention_config = attention_config
        self.lambda_config = lambda_config
        self.attention_pattern_config = attention_pattern_config
        self.gating_config = gating_config or {}
        self.skip_config = skip_config or {}
        self.rope_config = rope_config or {}
        self.embed_config = embed_config or {}
        self.low_rank_config = _resolve_low_rank_config(low_rank_config)
        self.low_rank_pairs: list[tuple[Tensor, Tensor]] = []

        c_proj_lr_mul = lr_multipliers["c_proj"]
        mlp_init_std_scale = model_config["mlp_init_std_scale"]
        lm_head_init_std = model_config["lm_head_init_std"]
        embed_padding_multiple = model_config["embed_padding_multiple"]
        eos_token_id = model_config["eos_token_id"]
        value_embed_head_indices = model_config["value_embed_head_indices"]
        value_embed_mid_layer_count = model_config["value_embed_mid_layer_count"]
        value_embed_tail_indices = model_config["value_embed_tail_indices"]
        value_embed_gate_scale = model_config["value_embed_gate_scale"]
        skip_gate_scale = model_config["skip_gate_scale"]
        residual_first_layer_index = model_config["residual_first_layer_index"]
        logits_softcap_scale = model_config["logits_softcap_scale"]
        logits_softcap_shift = model_config["logits_softcap_shift"]
        logits_softcap_divisor = model_config["logits_softcap_divisor"]
        logits_softcap_mode = str(model_config.get("logits_softcap_mode", "sigmoid")).lower()
        logits_tanh_cap = float(model_config.get("logits_tanh_cap", 30.0))

        vocab_size = model_config["vocab_size"]
        num_layers = model_config["num_layers"]
        num_heads = model_config["num_heads"]
        num_kv_heads = int(model_config.get("num_kv_heads", num_heads))
        model_dim = model_config["model_dim"]
        head_dim = model_config["head_dim"]
        qk_gain_init = float(model_config.get("qk_gain_init", 1.0))
        ln_scale = bool(model_config.get("ln_scale", False))
        block_size = attention_config["block_size"]
        self.activation = model_config.get("activation", "relu_squared")
        self.ffn_dim = model_config.get("ffn_dim", None)
        mlp_type = model_config.get("mlp_type", "default")
        mlp_kwargs = model_config.get("mlp_kwargs", {})
        self._weight_tied_embeddings = self.embed_config.get("weight_tied", True)
        self._enable_embed_split = self.embed_config.get("enable_embed_split", True)

        # Validate and normalize activation configuration.
        is_glu, _ = _get_activation_spec(self.activation)
        if self.ffn_dim is not None:
            self.ffn_dim = int(self.ffn_dim)
        else:
            self.ffn_dim = 4 * model_dim
            if is_glu:
                # GLU-family FFNs use split hidden projection; use a smaller width
                # so parameter counts stay near legacy 2:1 linear projection ratio.
                self.ffn_dim = max(1, (8 * model_dim) // 3)

        # Vocab size rounded up for efficiency
        vocab_size_padded = next_multiple_of_n(vocab_size, n=embed_padding_multiple)
        self.vocab_size = vocab_size
        self.vocab_size_padded = vocab_size_padded
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.canon_settings = CanonSettings.from_config(canon_config, num_layers=num_layers)

        if num_heads % num_kv_heads != 0:
            raise ValueError("model_config.num_heads must be divisible by model_config.num_kv_heads")
        if self.canon_settings.enabled and mlp_type != "default":
            raise ValueError("canon_config.enabled=True currently requires model_config.mlp_type='default'")
        if logits_softcap_mode not in {"sigmoid", "tanh"}:
            raise ValueError(
                f"model_config.logits_softcap_mode must be 'sigmoid' or 'tanh', got {logits_softcap_mode!r}"
            )
        if logits_tanh_cap <= 0:
            raise ValueError("model_config.logits_tanh_cap must be > 0")

        # Positional embedding (YaRN, HalfRoPE, StandardRoPE, or none)
        self.pos_emb = create_positional_embedding(self.rope_config, head_dim, max_seq_len, block_size)

        # Smear gate: shift token embeddings forward (from train_gpt.py @classiclarryd)
        gate_input_dim = self.gating_config["gate_input_dim"]
        if self.canon_settings.enabled and self.canon_settings.smear_mode == "canon_vector":
            self.canon_smear = VectorSmearGate(model_dim)
            _set_param_metadata(
                self.canon_smear.gate,
                "smear_gate",
                lr_mul=lr_multipliers["smear_gate"],
            )
            self.smear_gate = None
        elif self.gating_config.get("use_smear_gate", True):
            self.smear_gate = CastedLinear(gate_input_dim, 1)
            self.smear_gate.weight.label = "smear_gate"
            self.smear_gate.weight.lr_mul = lr_multipliers["smear_gate"]
            self.canon_smear = None
        else:
            self.smear_gate = None
            self.canon_smear = None

        # Skip gate (from train_gpt.py)
        if self.canon_settings.skip_topology == "ramen" and self.gating_config.get("use_skip_gate", True):
            self.skip_gate = CastedLinear(gate_input_dim, 1)
            self.skip_gate.weight.label = "skip_gate"
            self.skip_gate.weight.lr_mul = lr_multipliers["skip_gate"]
        else:
            self.skip_gate = None

        # Token value embeddings (from train_gpt.py @KoszarskyB)
        self.value_embeds = nn.ModuleList(
            [
                nn.Embedding(vocab_size_padded, model_dim)
                for _ in range(attention_pattern_config["num_value_embeds"])
            ]
        )
        for embed in self.value_embeds:
            nn.init.zeros_(embed.weight)
            embed.weight.label = "value_embed"

        if self.canon_settings.enabled and self.canon_settings.bigram_vocab_size > 0:
            if self.canon_settings.bigram_dim <= 0:
                raise ValueError("canon_config.bigram_dim must be > 0 when bigram_vocab_size is enabled")
            self.bigram = BigramHashEmbedding(
                self.canon_settings.bigram_vocab_size,
                self.canon_settings.bigram_dim,
                model_dim,
            )
            _set_param_metadata(self.bigram.scale, "bigram_scale")
        else:
            self.bigram = None

        # Blocks with gating config (matching train_gpt.py)
        value_embed_layers = attention_pattern_config["value_embed_layers"]
        blocks = []
        for i in range(num_layers):
            blocks.append(
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    max_seq_len,
                    i,
                    head_dim,
                    skip_attention=i in attention_pattern_config["skip_attention_layers"],
                    gating_config=self.gating_config,
                    value_embed_layers=value_embed_layers,
                    c_proj_lr_mul=c_proj_lr_mul,
                    mlp_std_scale=mlp_init_std_scale,
                    value_embed_gate_scale=value_embed_gate_scale,
                    activation=self.activation,
                    ffn_dim=self.ffn_dim,
                    mlp_type=mlp_type,
                    low_rank_config=self.low_rank_config,
                    mlp_kwargs=mlp_kwargs,
                    qk_gain_init=qk_gain_init,
                    ln_scale=ln_scale,
                    canon_settings=self.canon_settings,
                    canon_set=self.canon_settings.layer_set_for(i, num_layers),
                    use_xsa=self.canon_settings.uses_xsa(i, num_layers),
                    use_boundary_delta=self.canon_settings.uses_boundary_delta(i),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        for block in self.blocks:
            if block.attn is not None:
                self.low_rank_pairs.extend(block.attn.low_rank_pairs)
            self.low_rank_pairs.extend(getattr(block.mlp, "low_rank_pairs", []))

        if self.canon_settings.enabled and self.canon_settings.skip_topology == "canon_unet":
            self.num_encoder_layers = num_layers // 2
            self.num_decoder_layers = num_layers - self.num_encoder_layers
            self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
            self.skip_weights = _set_param_metadata(
                nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)),
                "skip_weights",
            )
        else:
            self.num_encoder_layers = 0
            self.num_decoder_layers = 0
            self.num_skip_weights = 0
            self.register_parameter("skip_weights", None)

        # LM head with proper initialization
        self.lm_head = CastedLinear(model_dim, vocab_size_padded, use_fp8=False)
        nn.init.normal_(self.lm_head.weight, mean=0, std=lm_head_init_std)
        self.lm_head.weight.label = "lm_head"

        # Weight tying / untied embedding behavior.
        if self._weight_tied_embeddings:
            # Start with tied embedding. `create_embed()` can split later.
            self.embed = None  # Will use lm_head.weight
            self.split_embed = False
        else:
            # Start untied from step 0.
            self.embed = nn.Embedding(self.vocab_size_padded, self.model_dim)
            self.embed = self.embed.to(
                device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype
            )
            self.embed.weight.data.copy_(self.lm_head.weight.data)
            self.embed.weight.label = "embed"
            self.embed.weight.wd_mul = wd_multipliers["embed"]
            self.split_embed = True

        # x0_lambdas separated for different optimizer treatment
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.x0_lambdas.label = "x0_lambdas"
        self.x0_lambdas.lr_mul = lr_multipliers["x0_lambdas"]

        # Construct scalars parameter
        value_embed_layers_set = set(value_embed_layers)
        resid_init = lambda_config["resid_lambdas_init"]
        sa_init = lambda_config["sa_lambdas_init"]
        sa_init_no_ve = lambda_config["sa_lambdas_init_no_ve"]
        smear_init = lambda_config["smear_lambda_init"]
        backout_init = lambda_config["backout_lambda_init"]
        skip_lambda_init = lambda_config["skip_lambda_init"]

        self.scalars = nn.Parameter(
            torch.cat(
                [
                    resid_init * torch.ones(num_layers),  # resid_lambdas
                    *[
                        torch.tensor(sa_init if i in value_embed_layers_set else sa_init_no_ve)
                        for i in range(num_layers)
                    ],  # SA lambdas
                    torch.tensor([smear_init]),  # smear_lambda
                    torch.tensor([backout_init]),  # backout_lambda
                    torch.tensor([skip_lambda_init]),  # skip_lambda
                ]
            )
        )
        self.scalars.label = "scalars"
        self.scalars.lr_mul = lr_multipliers["scalars"]

        # Set learning rate and weight decay multipliers
        wd_multipliers = wd_multipliers or {}
        for param in self.value_embeds.parameters():
            param.lr_mul = lr_multipliers["value_embed"]
            param.wd_mul = wd_multipliers["value_embed"]
        self.lm_head.weight.wd_mul = wd_multipliers["head"]
        self.scalars.wd_mul = wd_multipliers["scalars"]
        self.x0_lambdas.wd_mul = wd_multipliers["x0_lambdas"]
        if self.smear_gate:
            self.smear_gate.weight.wd_mul = wd_multipliers["smear_gate"]
        if self.skip_gate:
            self.skip_gate.weight.wd_mul = wd_multipliers["skip_gate"]

        self._value_embed_head_indices = value_embed_head_indices
        self._value_embed_mid_layer_count = value_embed_mid_layer_count
        self._value_embed_tail_indices = value_embed_tail_indices
        self._eos_token_id = eos_token_id
        self._skip_gate_scale = skip_gate_scale
        self._residual_first_layer_index = residual_first_layer_index
        self._logits_softcap_scale = logits_softcap_scale
        self._logits_softcap_shift = logits_softcap_shift
        self._logits_softcap_divisor = logits_softcap_divisor
        self._logits_softcap_mode = logits_softcap_mode
        self._logits_tanh_cap = logits_tanh_cap
        self._model_wd_multipliers = wd_multipliers

        (
            self._residual_connection_mode,
            self._residual_connection_expand,
            self._residual_connection_reduce,
            self._residual_connection_init,
        ) = build_residual_connection_fns(
            residual_connection_config,
            model_dim,
        )

        if self.canon_settings.use_resid_mix and self._residual_connection_init is not None:
            raise ValueError("canon_config.use_resid_mix is not supported with residual_connection_config modes")

        if self._residual_connection_init is not None:
            for i, block in enumerate(self.blocks):
                block.residual_connection = self._residual_connection_init(
                    branch=partial(block._forward_impl, return_delta=True),
                    layer_index=i,
                )

    def create_embed(self):
        """Create separate embedding when weight tying is split"""
        if self.embed is None and self._weight_tied_embeddings and self._enable_embed_split:
            self.embed = nn.Embedding(self.vocab_size_padded, self.model_dim)
            # Move to correct device and dtype to match lm_head
            self.embed = self.embed.to(
                device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype
            )
            # Copy lm_head weights to embed
            self.embed.weight.data.copy_(self.lm_head.weight.data)
            self.embed.weight.label = "embed"
            # Set wd_mul to match train_gpt.py (150.0 for embed like lm_head)
            self.embed.weight.wd_mul = self._model_wd_multipliers["embed"]
        self.split_embed = True

    def create_kv_cache(self):
        return {
            "layers": [{"k": None, "v": None} for _ in range(self.num_layers)],
            "docs": torch.empty(0, dtype=torch.int32, device=self.lm_head.weight.device),
            "smear_state": None,
        }

    def create_blockmasks(self, docs: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = self.attention_config["block_size"]
        # docs passed in

        def document_causal(b, h, q_idx, kv_idx):
            _ = b, h  # unused but required by FlexAttention API
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            # Convert to float for argsort (CUDA doesn't support bool sorting)
            indices = (
                dense_blockmask.float()
                .argsort(dim=-1, descending=False, stable=True)
                .flip(-1)
                .to(torch.int32)
            )
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(docs) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(docs) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor):
            return BlockMask.from_kv_blocks(
                torch.clamp_max(
                    partial_kv_num_blocks,
                    torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
                ),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def _build_value_embeddings(self, input_seq: Tensor) -> list[Tensor | None]:
        num_layers = len(self.blocks)
        if len(self.value_embeds) == 0:
            return [None] * num_layers

        ve_computed = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = (
            [ve_computed[i] for i in self._value_embed_head_indices]
            + [None] * (num_layers - self._value_embed_mid_layer_count)
            + [ve_computed[i] for i in self._value_embed_tail_indices]
        )
        assert len(ve) == num_layers
        return ve

    def _build_attention_layout(
        self,
        input_seq: Tensor,
        *,
        is_decode: bool,
        cache_docs: Tensor | None,
        sliding_window_num_blocks: Tensor,
    ) -> tuple[Tensor, list, list[bool]]:
        requires_mask = [block.attn is not None for block in self.blocks]
        docs: Tensor
        if is_decode:
            assert cache_docs is not None
            current_doc = cache_docs[-1] + (input_seq[-1] == self._eos_token_id).to(cache_docs.dtype)
            docs = current_doc.view(1)
        else:
            docs = (input_seq == self._eos_token_id).cumsum(0)

        long_bm = short_bm = None
        if any(requires_mask) and not is_decode:
            long_bm, short_bm = self.create_blockmasks(docs, sliding_window_num_blocks)

        block_masks = []
        key_offsets = []
        for needs_mask, char in zip(requires_mask, self.attention_pattern_config["block_mask_pattern"]):
            if is_decode or not needs_mask:
                block_masks.append(None)
                key_offsets.append(False)
            elif char == "L":
                block_masks.append(long_bm)
                key_offsets.append(True)
            elif char == "S":
                block_masks.append(short_bm)
                key_offsets.append(False)
            elif char == "N":
                block_masks.append(None)
                key_offsets.append(False)
            else:
                raise ValueError(
                    f"Invalid block mask pattern character: {char}. Use 'L', 'S', or 'N'."
                )

        assert len(block_masks) == len(self.blocks)
        return docs, block_masks, key_offsets

    def _embed_tokens(self, input_seq: Tensor) -> Tensor:
        if self.split_embed and self.embed is not None:
            x = self.embed(input_seq)
        else:
            x = F.embedding(input_seq, self.lm_head.weight)
        if self.bigram is not None:
            x = x + self.bigram(input_seq)
        return x

    def _apply_input_smear(
        self,
        x: Tensor,
        cache_smear_state: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        smear_lambda = self.scalars[3 * self.num_layers]
        x_raw = x
        smear_state_to_cache = None

        if self.canon_smear is not None:
            x = norm(x[None])
            pre_smear_x = x
            prev_state = cache_smear_state if x.shape[1] == 1 and cache_smear_state is not None else None
            x = self.canon_smear(x, prev_state=prev_state)
            smear_state_to_cache = pre_smear_x[0, -1].detach()
            return x, x, smear_state_to_cache

        if self.smear_gate is not None:
            gate_width = self.smear_gate.weight.size(-1)
            if x.shape[0] == 1 and cache_smear_state is not None:
                smear_gate_out = torch.sigmoid(self.smear_gate(x[:, :gate_width]))
                x = x_raw + (smear_lambda * smear_gate_out) * cache_smear_state.view(1, -1)
            else:
                smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :gate_width]))
                x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
            smear_state_to_cache = x_raw[-1].detach()

        x = norm(x[None])
        return x, x, smear_state_to_cache

    def _apply_block(
        self,
        layer_idx: int,
        hidden: Tensor,
        *,
        x0: Tensor,
        ve: list[Tensor | None],
        sa_lambdas: Tensor,
        block_masks: list,
        cos: Tensor,
        sin: Tensor,
        attn_scale: float,
        docs: Tensor,
        key_offsets: list[bool],
        cache_layers: list[dict] | None,
        cache_docs: Tensor | None,
    ) -> Tensor:
        if not self.canon_settings.use_resid_mix:
            if layer_idx == self._residual_first_layer_index:
                hidden = (self.scalars[0] + self.x0_lambdas[0]) * hidden
            else:
                hidden = self.scalars[layer_idx] * hidden + self.x0_lambdas[layer_idx] * x0

        return self.blocks[layer_idx](
            hidden,
            ve[layer_idx],
            sa_lambdas[layer_idx],
            block_masks[layer_idx],
            cos,
            sin,
            attn_scale,
            docs,
            key_offsets[layer_idx],
            kv_cache=cache_layers[layer_idx] if cache_layers is not None else None,
            cache_docs=cache_docs,
            x0=x0,
        )

    def _run_canon_unet_blocks(self, x: Tensor, apply_block: Callable[[int, Tensor], Tensor]) -> Tensor:
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = apply_block(i, x)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            if skip_connections and self.skip_weights is not None:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip_connections.pop()
            x = apply_block(self.num_encoder_layers + i, x)
        return x

    def _run_ramen_blocks(
        self,
        x: Tensor,
        x0: Tensor,
        apply_block: Callable[[int, Tensor], Tensor],
        *,
        skip_in_layers: list[int],
        skip_out_layers: list[int],
        backout_layer: int,
        backout_lambda: Tensor,
        skip_lambda: Tensor,
    ) -> Tensor:
        skip_connections = []
        x_backout = None

        for i in range(len(self.blocks)):
            if i in skip_out_layers and skip_connections:
                if self.skip_gate is not None:
                    skip_gate_out = (
                        torch.sigmoid(skip_lambda)
                        * self._skip_gate_scale
                        * torch.sigmoid(self.skip_gate(x0[..., : self.skip_gate.weight.size(-1)]))
                    )
                    x = x + skip_gate_out * skip_connections.pop()
                else:
                    x = x + skip_connections.pop()

            x = apply_block(i, x)

            if i in skip_in_layers:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - backout_lambda * x_backout
        return x

    def _compute_mtp_loss(self, logits_flat: Tensor, target_seq: Tensor, mtp_weights: Tensor):
        """Compute multi-token prediction loss with weighted offsets.

        Stacks shifted targets (padded with ignore_index for invalid tail
        positions), expands logits, and computes the entire MTP loss as a
        single F.cross_entropy call for better torch.compile fusion.
        """
        num_offsets = mtp_weights.size(0)
        seq_len = logits_flat.size(0)

        # Stack shifted targets with -100 padding: (num_offsets, seq_len)
        shifted = [target_seq]
        for k in range(1, num_offsets):
            pad = target_seq.new_full((k,), -100)
            shifted.append(torch.cat([target_seq[k:], pad]))
        targets_stacked = torch.stack(shifted)

        # Expand logits to match: (num_offsets, seq_len, V) — expand is free
        logits_expanded = logits_flat.unsqueeze(0).expand(num_offsets, -1, -1)

        # Single batched cross-entropy with ignore_index for padded positions
        per_token_loss = F.cross_entropy(
            logits_expanded.reshape(-1, logits_expanded.size(-1)),
            targets_stacked.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(num_offsets, seq_len)

        # Weighted sum: sum over tokens, then dot with weights
        per_offset_loss = per_token_loss.sum(dim=1)
        return (mtp_weights * per_offset_loss).sum()

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor,
        sliding_window_num_blocks: Tensor,
        return_logits: bool = False,
        kv_cache: dict | None = None,
        mtp_weights: Tensor | None = None,
    ):
        assert input_seq.ndim == 1
        cache_layers = None
        cache_docs = None
        cache_smear_state = None
        is_decode = False
        if kv_cache is not None:
            if "layers" not in kv_cache:
                kv_cache["layers"] = [{"k": None, "v": None} for _ in range(self.num_layers)]
            cache_layers = kv_cache.get("layers")
            cache_docs = kv_cache.get("docs")
            cache_smear_state = kv_cache.get("smear_state")
            is_decode = (
                input_seq.numel() == 1
                and cache_docs is not None
                and cache_docs.numel() > 0
                and cache_layers is not None
                and len(cache_layers) == self.num_layers
            )

        skip_in_layers = self.skip_config.get("skip_in_layers", [])
        skip_out_layers = self.skip_config.get("skip_out_layers", [])
        backout_layer = self.skip_config.get("backout_layer", -1)

        ve = self._build_value_embeddings(input_seq)
        docs, block_masks, key_offsets = self._build_attention_layout(
            input_seq,
            is_decode=is_decode,
            cache_docs=cache_docs,
            sliding_window_num_blocks=sliding_window_num_blocks,
        )
        x, x0, smear_state_to_cache = self._apply_input_smear(
            self._embed_tokens(input_seq),
            cache_smear_state,
        )

        x = self._residual_connection_expand(x)
        x0 = self._residual_connection_expand(x0)

        sa_lambdas = self.scalars[self.num_layers : 3 * self.num_layers].view(-1, 2)
        backout_lambda = self.scalars[3 * self.num_layers + 1]
        skip_lambda = self.scalars[3 * self.num_layers + 2]

        cos, sin = self.pos_emb.cos, self.pos_emb.sin
        attn_scale = self.pos_emb.attn_scale

        apply_block = partial(
            self._apply_block,
            x0=x0,
            ve=ve,
            sa_lambdas=sa_lambdas,
            block_masks=block_masks,
            cos=cos,
            sin=sin,
            attn_scale=attn_scale,
            docs=docs,
            key_offsets=key_offsets,
            cache_layers=cache_layers,
            cache_docs=cache_docs,
        )

        if self.canon_settings.skip_topology == "canon_unet":
            x = self._run_canon_unet_blocks(x, apply_block)
        else:
            x = self._run_ramen_blocks(
                x,
                x0,
                apply_block,
                skip_in_layers=skip_in_layers,
                skip_out_layers=skip_out_layers,
                backout_layer=backout_layer,
                backout_lambda=backout_lambda,
                skip_lambda=skip_lambda,
            )

        x = self._residual_connection_reduce(x)
        x = norm(x)
        logits = self.lm_head(x)

        if self._logits_softcap_mode == "tanh":
            logits = self._logits_tanh_cap * torch.tanh(logits / self._logits_tanh_cap)
        else:
            # Updated softcap formula @classiclarryd
            # 23 * sigmoid((logits + 5) / 7.5)
            logits = self._logits_softcap_scale * torch.sigmoid(
                (logits + self._logits_softcap_shift) / self._logits_softcap_divisor
            )
        logits_for_loss = logits.float() if not self.training else logits

        if kv_cache is not None and is_decode:
            if cache_docs is None:
                kv_cache["docs"] = docs.to(dtype=torch.int32)
            else:
                kv_cache["docs"] = torch.cat([cache_docs, docs.to(dtype=torch.int32)])
        elif kv_cache is not None:
            kv_cache["docs"] = docs.to(dtype=torch.int32)
        if kv_cache is not None:
            kv_cache["smear_state"] = smear_state_to_cache

        # Language modeling loss
        if return_logits:
            return logits_for_loss[:, : self.vocab_size]

        if self.training:
            logits_flat = logits_for_loss.view(-1, logits_for_loss.size(-1))
            if mtp_weights is not None:
                return self._compute_mtp_loss(logits_flat, target_seq, mtp_weights)
            return F.cross_entropy(logits_flat, target_seq, reduction="sum")

        return F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            target_seq,
            reduction="mean",
        )
