import math
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
    create_mlp as _create_mlp,
    _get_activation_spec,
)

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


def _resolve_low_rank_config(low_rank_config: dict | None) -> dict:
    if low_rank_config is None or not isinstance(low_rank_config, dict):
        return {
            "enabled": False,
            "rank_ratio": 0.25,
            "rank": None,
            "min_rank": 1,
            "max_rank": None,
            "apply_attention": True,
            "apply_mlp": True,
        }
    return {
        "enabled": bool(low_rank_config.get("enabled", False)),
        "rank_ratio": float(low_rank_config.get("rank_ratio", 0.25)),
        "rank": low_rank_config.get("rank"),
        "min_rank": low_rank_config.get("min_rank", 1),
        "max_rank": low_rank_config.get("max_rank", None),
        "apply_attention": bool(low_rank_config.get("apply_attention", True)),
        "apply_mlp": bool(low_rank_config.get("apply_mlp", True)),
    }

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
    if residual_connection_mode not in {"standard", "hc", "mhc"}:
        raise ValueError(
            f"Unsupported residual_connection.mode={residual_connection_mode!r}. "
            "Expected one of: standard, hc, mhc."
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
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)


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
        head_dim: int,
        max_seq_len: int,
        base_freq: float,
        block_size: int,
        initial_attn_scale: float = 0.1,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base_freq = base_freq
        self.block_size = block_size
        self.initial_attn_scale = initial_attn_scale
        self.reset()

    def reset(self):
        """Reset to initial state (called at start of training and after warmup)"""
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / self.base_freq) ** torch.linspace(
            0, 1, steps=self.head_dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 4)])
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
        head_dim: int,
        max_seq_len: int,
        base_freq: float = 1024,
        initial_attn_scale: float = 0.1,
    ):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / base_freq) ** torch.linspace(
            0, 1, steps=head_dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim // 4)])
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
        head_dim: int,
        max_seq_len: int,
        base_freq: float = 1024,
        initial_attn_scale: float = 0.1,
    ):
        super().__init__()
        angular_freq = (1 / base_freq) ** torch.linspace(
            0, 1, steps=head_dim // 2, dtype=torch.float32
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
    """Identity positional embedding: cos=1, sin=0 so rotary() is a no-op."""

    def __init__(self, head_dim: int, max_seq_len: int, initial_attn_scale: float = 0.1):
        super().__init__()
        self.cos = nn.Buffer(
            torch.ones(max_seq_len, head_dim // 2, dtype=torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            torch.zeros(max_seq_len, head_dim // 2, dtype=torch.bfloat16), persistent=False
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

    if rope_type == "yarn":
        return YarnPositionalEmbedding(
            head_dim, max_seq_len, base_freq, block_size, initial_attn_scale
        )
    if rope_type == "half_rope":
        return HalfRoPE(head_dim, max_seq_len, base_freq, initial_attn_scale)
    if rope_type == "rope":
        return StandardRoPE(head_dim, max_seq_len, base_freq, initial_attn_scale)
    if rope_type == "none":
        return NoPositionalEmbedding(head_dim, max_seq_len, initial_attn_scale)

    supported = ["yarn", "half_rope", "rope", "none"]
    raise ValueError(f"Unsupported rope type: {rope_type!r}. Supported: {', '.join(supported)}")


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        layer_idx: int,
        gating_config: dict,
        value_embed_layers: list,
        value_embed_gate_scale: float,
        low_rank_config: dict | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.layer_idx = layer_idx
        self.value_embed_gate_scale = value_embed_gate_scale
        hdim = num_heads * head_dim
        low_rank_config = _resolve_low_rank_config(low_rank_config)

        assert hdim == dim, "num_heads * head_dim must equal model_dim"
        std = dim**-0.5
        bound = (3**0.5) * std  # improved init scale by @YouJiacheng

        self.low_rank_pairs: list[tuple[Tensor, Tensor]] = []

        # Merged QKVO weights (from train_gpt.py)
        # Layout: [Q, K, V, O] each of size (dim, hdim)
        if low_rank_config["enabled"] and low_rank_config["apply_attention"]:
            self.qkvo_w = LowRankLinear(
                in_features=hdim,
                out_features=dim * 4,
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
                self.qkvo_w.A[dim * 3 :].zero_()  # init O weights to zero
            self.low_rank_pairs = [(self.qkvo_w.A, self.qkvo_w.B)]
        else:
            self.qkvo_w = nn.Parameter(torch.empty(dim * 4, hdim))
            self.qkvo_w.label = "attn"
            with torch.no_grad():
                self.qkvo_w[: dim * 3].uniform_(-bound, bound)  # init QKV weights
                self.qkvo_w[dim * 3 :].zero_()  # init O weights to zero

        # Sparse gated attention (from train_gpt.py @classiclarryd)
        gate_input_dim = gating_config["gate_input_dim"]
        if gating_config.get("use_attn_gate", True):
            self.attn_gate = CastedLinear(gate_input_dim, num_heads)
            self.attn_gate.weight.label = "attn_gate"
        else:
            self.attn_gate = None

        # Value embedding gate (only on specific layers)
        if gating_config.get("use_value_embed_gate", True) and layer_idx in value_embed_layers:
            self.value_embed_gate = CastedLinear(gate_input_dim, num_heads)
            self.value_embed_gate.weight.label = "value_embed_gate"
        else:
            self.value_embed_gate = None

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
    ):
        B, T = x.size(0), x.size(1)
        # Apply sa_lambdas[0] to QKV weights (from train_gpt.py)
        if isinstance(self.qkvo_w, LowRankLinear):
            qkv = sa_lambdas[0] * self.qkvo_w(x)
            qkv = qkv[:, :, : self.dim * 3]
            q, k, v = qkv.view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        else:
            q, k, v = (
                F.linear(x, sa_lambdas[0] * self.qkvo_w[: self.dim * 3].type_as(x))
                .view(B, T, 3 * self.num_heads, self.head_dim)
                .chunk(3, dim=-2)
            )

        # QK norm and RoPE
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        # Key offset: shift keys forward for the stationary head dims (from train_gpt.py)
        # Enables 1-layer induction on long attention window layers
        if key_offset:
            k[:, 1:, :, self.head_dim // 4 : self.head_dim // 2] = k[
                :, :-1, :, self.head_dim // 4 : self.head_dim // 2
            ].clone()
            k[:, 1:, :, 3 * self.head_dim // 4 :] = k[:, :-1, :, 3 * self.head_dim // 4 :].clone()

        # Value embedding with gating (from train_gpt.py)
        if ve is not None and self.value_embed_gate is not None:
            ve_gate_out = self.value_embed_gate_scale * torch.sigmoid(
                self.value_embed_gate(x[..., : self.value_embed_gate.weight.size(-1)])
            ).view(B, T, self.num_heads, 1)
            v = v + ve_gate_out * ve.view_as(v)
        elif ve is not None:
            # Fallback to lambda-based mixing if no gate
            v = v + ve.view_as(v)

        # Element-wise mask for FlexAttention
        def score_mod(score, b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            mask = causal_mask & document_mask
            return torch.where(mask, score, -float("inf"))

        # FlexAttention
        y = _get_flex_attention()(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=attn_scale,
            score_mod=score_mod,
            kernel_options=_flex_attention_kernel_options,
        ).transpose(1, 2)

        # Attention gating (from train_gpt.py)
        if self.attn_gate is not None:
            y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate.weight.size(-1)])).view(
                B, T, self.num_heads, 1
            )

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)

        # Output projection using merged weights with sa_lambdas[1]
        if isinstance(self.qkvo_w, LowRankLinear):
            o_A = self.qkvo_w.A[self.dim * 3 :, :].type_as(y)
            y = F.linear(y, self.qkvo_w.B.type_as(y))
            y = F.linear(y, o_A)
        else:
            o_weight = self.qkvo_w[self.dim * 3 :].type_as(y)
            y = F.linear(y, sa_lambdas[1] * o_weight)
        if isinstance(self.qkvo_w, LowRankLinear):
            y = sa_lambdas[1] * y
        return y


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
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
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        gating_config = gating_config or {}
        value_embed_layers = value_embed_layers or []
        low_rank_config = low_rank_config or {}

        # Skip attention of specific layers (e.g., layer 6 in train_gpt.py) by @YouJiacheng
        if not skip_attention:
            self.attn = CausalSelfAttention(
                dim,
                num_heads,
                max_seq_len,
                head_dim,
                layer_idx,
                gating_config,
                value_embed_layers,
                value_embed_gate_scale=value_embed_gate_scale,
                low_rank_config=low_rank_config,
            )
        else:
            self.attn = None

        # FFN via factory — mlp_type selects the variant.
        self.mlp = _create_mlp(
            mlp_type=mlp_type,
            dim=dim,
            c_proj_lr_mul=c_proj_lr_mul,
            std_scale=mlp_std_scale,
            activation=activation,
            ffn_dim=ffn_dim,
            low_rank_config=low_rank_config,
            **(mlp_kwargs or {}),
        )
        self.residual_connection = residual_connection

    def _forward_standard(
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
    ):
        # Attention branch
        if self.attn is not None:
            attn_out = self.attn(
                norm(x), ve, sa_lambdas, block_mask, cos, sin, attn_scale, docs, key_offset
            )
            x = x + attn_out
        # MLP branch — variants with internal norms skip the external norm()
        mlp_input = norm(x) if getattr(self.mlp, "needs_external_norm", True) else x
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out
        return x

    def _forward_delta(
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
    ):
        residual = x

        # Attention branch
        if self.attn is not None:
            attn_out = self.attn(
                norm(x), ve, sa_lambdas, block_mask, cos, sin, attn_scale, docs, key_offset
            )
            x = x + attn_out

        # MLP branch — variants with internal norms skip the external norm()
        mlp_input = norm(x) if getattr(self.mlp, "needs_external_norm", True) else x
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out
        return x - residual

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
    ):
        if self.residual_connection is not None:
            return self.residual_connection(
                x, ve, sa_lambdas, block_mask, cos, sin, attn_scale, docs, key_offset
            )

        return self._forward_standard(
            x, ve, sa_lambdas, block_mask, cos, sin, attn_scale, docs, key_offset
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

        vocab_size = model_config["vocab_size"]
        num_layers = model_config["num_layers"]
        num_heads = model_config["num_heads"]
        model_dim = model_config["model_dim"]
        head_dim = model_config["head_dim"]
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

        # Positional embedding (YaRN, HalfRoPE, StandardRoPE, or none)
        self.pos_emb = create_positional_embedding(self.rope_config, head_dim, max_seq_len, block_size)

        # Smear gate: shift token embeddings forward (from train_gpt.py @classiclarryd)
        gate_input_dim = self.gating_config["gate_input_dim"]
        if self.gating_config.get("use_smear_gate", True):
            self.smear_gate = CastedLinear(gate_input_dim, 1)
            self.smear_gate.weight.label = "smear_gate"
            self.smear_gate.weight.lr_mul = lr_multipliers["smear_gate"]
        else:
            self.smear_gate = None

        # Skip gate (from train_gpt.py)
        if self.gating_config.get("use_skip_gate", True):
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

        # Blocks with gating config (matching train_gpt.py)
        value_embed_layers = attention_pattern_config["value_embed_layers"]
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
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
                )
                for i in range(num_layers)
            ]
        )

        for block in self.blocks:
            if block.attn is not None:
                self.low_rank_pairs.extend(block.attn.low_rank_pairs)
            self.low_rank_pairs.extend(getattr(block.mlp, "low_rank_pairs", []))

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

        if self._residual_connection_init is not None:
            for i, block in enumerate(self.blocks):
                block.residual_connection = self._residual_connection_init(
                    branch=block._forward_delta,
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

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        # Get skip/backout config
        skip_in_layers = self.skip_config["skip_in_layers"]
        skip_out_layers = self.skip_config["skip_out_layers"]
        backout_layer = self.skip_config["backout_layer"]

        # Build value embeddings pattern from config (from train_gpt.py)
        # train_gpt.py pattern: [ve[1], ve[2]] + [None] * (num_layers - 5) + [ve[0], ve[1], ve[2]]
        # 012 ... 012 structure by @YouJiacheng, improved on @leloykun's U-net structure
        # dropping first layer updates this to .12 ... 012
        ve_computed = [value_embed(input_seq) for value_embed in self.value_embeds]
        num_layers = len(self.blocks)
        ve = (
            [ve_computed[i] for i in self._value_embed_head_indices]
            + [None] * (num_layers - self._value_embed_mid_layer_count)
            + [ve_computed[i] for i in self._value_embed_tail_indices]
        )
        assert len(ve) == num_layers

        # Build block masks pattern and key_offset flags
        requires_mask = []
        any_requires_mask = False
        for blk in self.blocks:
            needs = blk.attn is not None
            requires_mask.append(needs)
            any_requires_mask = any_requires_mask or needs

        docs = (input_seq == self._eos_token_id).cumsum(0)

        if any_requires_mask:
            long_bm, short_bm = self.create_blockmasks(docs, sliding_window_num_blocks)
        block_masks = []
        key_offsets = []  # Key offset flags for each layer (True for long windows)
        for i, char in enumerate(self.attention_pattern_config["block_mask_pattern"]):
            if not requires_mask[i]:
                block_masks.append(None)
                key_offsets.append(False)
            elif char == "L":
                block_masks.append(long_bm)
                key_offsets.append(True)  # Apply key offset on long window layers
            elif char == "S":
                block_masks.append(short_bm)
                key_offsets.append(False)
            elif char == "N":
                block_masks.append(None)  # Skip attention
                key_offsets.append(False)
            else:
                raise ValueError(
                    f"Invalid block mask pattern character: {char}. Use 'L', 'S', or 'N'."
                )
        assert len(block_masks) == len(self.blocks)

        # Get embedding (weight-tied or separate)
        if self.split_embed and self.embed is not None:
            x = self.embed(input_seq)
        else:
            x = F.embedding(input_seq, self.lm_head.weight)

        # Smear gate: shift token embeddings forward (from train_gpt.py @classiclarryd)
        smear_lambda = self.scalars[3 * self.num_layers]
        if self.smear_gate is not None:
            smear_gate_out = smear_lambda * torch.sigmoid(
                self.smear_gate(x[1:, : self.smear_gate.weight.size(-1)])
            )
            x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])

        x = x0 = norm(x[None])
        x = self._residual_connection_expand(x)
        x0 = self._residual_connection_expand(x0)

        # Extract lambdas from scalars
        resid_lambdas = self.scalars[: self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[self.num_layers : 3 * self.num_layers].view(-1, 2)
        backout_lambda = self.scalars[3 * self.num_layers + 1]
        skip_lambda = self.scalars[3 * self.num_layers + 2]

        # Get RoPE values from positional embedding
        cos, sin = self.pos_emb.cos, self.pos_emb.sin
        attn_scale = self.pos_emb.attn_scale

        # Skip connections
        skip_connections = []
        x_backout = None

        for i in range(len(self.blocks)):
            # Apply skip connection from earlier layers
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

            # Apply residual mixing
            if i == self._residual_first_layer_index:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0

            # Forward through block with key_offset for long window layers
            x = self.blocks[i](
                x, ve[i], sa_lambdas[i], block_masks[i], cos, sin, attn_scale, docs, key_offsets[i]
            )

            # Save skip connection
            if i in skip_in_layers:
                skip_connections.append(x)

            # Save for backout
            if i == backout_layer:
                x_backout = x

        # Backout: subtract contribution from early layers
        if x_backout is not None:
            x = x - backout_lambda * x_backout

        x = self._residual_connection_reduce(x)
        x = norm(x)
        logits = self.lm_head(x)

        # Updated softcap formula @classiclarryd
        # 23 * sigmoid((logits + 5) / 7.5)
        logits = self._logits_softcap_scale * torch.sigmoid(
            (logits + self._logits_softcap_shift) / self._logits_softcap_divisor
        )
        logits_for_loss = logits.float() if not self.training else logits

        # Language modeling loss
        if self.training:
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="sum"
            )
        else:
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="mean"
            )

        return loss
