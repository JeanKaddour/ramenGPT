"""
MLP variants and factory for ramenGPT.

Provides:
  - DefaultMLP          (mlp_type="default")  — current 2-layer FF from model.py
  - StackedFeedforward  (mlp_type="ff")       — stacked residual FF with internal RMSNorm
  - NormalizedFeedforward (mlp_type="nff")     — L2-norm-based nGPT feedforward
  - ResidualNormedMLP   (mlp_type="residual_normed") — deep residual MLP (Wang et al.)

Factory: create_mlp(mlp_type, dim, c_proj_lr_mul, std_scale, activation, ffn_dim, **kwargs)
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization


# ---------------------------------------------------------------------------
# Helpers copied/adapted from x-mlps-pytorch
# ---------------------------------------------------------------------------


def _exists(v):
    return v is not None


def _default(v, d):
    return v if _exists(v) else d


def _l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)


def _infer_low_rank_rank(in_features, rank_ratio=0.25, rank=None, min_rank=1, max_rank=None):
    if rank is not None:
        inferred_rank = int(rank)
    else:
        inferred_rank = int(in_features * rank_ratio)

    inferred_rank = max(1, inferred_rank)

    if min_rank is not None:
        inferred_rank = max(inferred_rank, int(min_rank))
    if max_rank is not None:
        inferred_rank = min(inferred_rank, int(max_rank))

    return min(inferred_rank, int(in_features))


class LowRankLinear(nn.Module):
    """Low-rank linear projection with two factor matrices.

    The effective dense weight is ``A @ B`` where:
      - ``A`` is ``(out_features, rank)``
      - ``B`` is ``(rank, in_features)``

    Forward pass avoids materializing the dense matrix explicitly.
    """

    _is_low_rank_linear = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank_ratio: float = 0.25,
        rank: int | None = None,
        min_rank: int | None = 1,
        max_rank: int | None = None,
        label: str | None = None,
        lr_mul: float = 1.0,
        wd_mul: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = _infer_low_rank_rank(
            in_features=in_features,
            rank_ratio=rank_ratio,
            rank=rank,
            min_rank=min_rank,
            max_rank=max_rank,
        )

        self.A = nn.Parameter(torch.empty(out_features, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, in_features))

        init_std = (1.0 / max(1, in_features)) ** 0.5
        with torch.no_grad():
            self.A.normal_(0.0, init_std)
            self.B.normal_(0.0, init_std)

        if label is not None:
            self.A.label = label
            self.B.label = label
        self.A.lr_mul = lr_mul
        self.B.lr_mul = lr_mul
        self.A.wd_mul = wd_mul
        self.B.wd_mul = wd_mul

    def materialize(self):
        return self.A.float().matmul(self.B.float()).to(dtype=self.A.dtype)

    @property
    def weight(self):
        return self.materialize()

    def forward(self, x: Tensor):
        return F.linear(F.linear(x, self.B.type_as(x)), self.A.type_as(x))


def _resolve_low_rank_config(low_rank_config: dict | None) -> dict:
    if low_rank_config is None or not isinstance(low_rank_config, dict):
        return {
            "enabled": False,
            "rank_ratio": 0.25,
            "rank": None,
            "min_rank": 1,
            "max_rank": None,
            "apply_mlp": True,
        }
    return {
        "enabled": bool(low_rank_config.get("enabled", False)),
        "rank_ratio": float(low_rank_config.get("rank_ratio", 0.25)),
        "rank": low_rank_config.get("rank"),
        "min_rank": low_rank_config.get("min_rank", 1),
        "max_rank": low_rank_config.get("max_rank", None),
        "apply_mlp": bool(low_rank_config.get("apply_mlp", True)),
    }


# ---------------------------------------------------------------------------
# RMSNormWD — weight-decay friendly RMSNorm (renamed to avoid collision with
# model.py's `norm()` helper)
# ---------------------------------------------------------------------------


class RMSNormWD(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, x.shape[-1:], self.gamma + 1, eps=self.eps)


class LayerNormWD(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma + 1, eps=self.eps)


# ---------------------------------------------------------------------------
# L2Norm, NormLinear, Scale, Residual — from nff.py
# ---------------------------------------------------------------------------


class L2Norm(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return _l2norm(t, dim=self.dim)


class NormLinear(nn.Module):
    def __init__(self, dim, dim_out, norm_dim_in=True, parametrize=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.linear = nn.Linear(dim, dim_out, bias=False)
        self.parametrize = parametrize
        self.l2norm = L2Norm(dim=-1 if norm_dim_in else 0)
        if parametrize:
            register_parametrization(self.linear, "weight", self.l2norm)
        self.norm_weights_()

    @torch.no_grad()
    def norm_weights_(self):
        if self.parametrize:
            normed = self.weight
            original = self.linear.parametrizations.weight.original
            original.copy_(normed)
        else:
            self.weight.copy_(self.l2norm(self.weight))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Scale(nn.Module):
    def __init__(self, dim, init=1.0, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, dim: int, init: float, scale: float | None = None):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, _default(scale, dim**-0.5))

    def forward(self, x, **kwargs):
        residual = x
        out = self.fn(x, **kwargs)
        tuple_output = isinstance(out, tuple)
        if tuple_output:
            out, *rest = out
        out = _l2norm(out)
        out = _l2norm(residual.lerp(out, self.branch_scale().type_as(out)))
        if tuple_output:
            out = (out, *rest)
        return out


# ---------------------------------------------------------------------------
# Activation classes from x-mlps-pytorch/activations.py
# ---------------------------------------------------------------------------


class ReluSquared(nn.Module):
    def __init__(self, signed=False):
        super().__init__()
        self.signed = signed

    def forward(self, x):
        out = x.relu().square()
        if not self.signed:
            return out
        return out * x.sign()


class BSiLU(nn.Module):
    def __init__(self, alpha=1.67):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        a = self.alpha
        return (x + a) * x.sigmoid() - a / 2


class NeLU(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        a = self.alpha
        return -a / (1.0 + x.square())


class Sugar(nn.Module):
    def __init__(self, forward_fn: nn.Module, backward_fn: nn.Module):
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x):
        forward_out = self.forward_fn(x)
        if not x.requires_grad:
            return forward_out
        backward_out = self.backward_fn(x)
        soft = torch.where(x > 0, forward_out, backward_out)
        return soft + (forward_out - soft).detach()


def ReluNelu(alpha=0.05):
    return Sugar(nn.ReLU(), NeLU(alpha))


# ---------------------------------------------------------------------------
# Activation spec helpers (moved from model.py, re-exported)
# ---------------------------------------------------------------------------


def _normalize_activation_name(activation: str) -> str:
    if not isinstance(activation, str):
        raise TypeError(f"activation must be a string, got {type(activation)!r}")
    normalized = activation.strip().lower()
    if not normalized:
        raise ValueError("activation cannot be empty")
    return normalized


def _get_activation_spec(activation: str):
    activation = _normalize_activation_name(activation)
    if activation == "relu":
        return False, F.relu
    if activation == "gelu":
        return False, F.gelu
    if activation == "swish":
        return False, F.silu
    if activation == "silu":
        return False, F.silu
    if activation == "linear":
        return False, lambda x: x
    if activation == "identity":
        return False, lambda x: x
    if activation == "relu_squared":
        return False, lambda x: F.relu(x).square()
    if activation == "gelu_squared":
        return False, lambda x: F.gelu(x).square()
    if activation == "swish_squared":
        return False, lambda x: F.silu(x).square()
    if activation == "silu_squared":
        return False, lambda x: F.silu(x).square()
    if activation == "geglu":
        return True, F.gelu
    if activation == "swiglu":
        return True, F.silu
    # Module-based activations from x-mlps-pytorch
    if activation == "bsilu":
        return False, BSiLU()
    if activation == "nelu":
        return False, NeLU()
    if activation == "relu_nelu":
        return False, ReluNelu()

    supported = [
        "relu",
        "gelu",
        "swish",
        "silu",
        "linear",
        "identity",
        "relu_squared",
        "gelu_squared",
        "swish_squared",
        "silu_squared",
        "geglu",
        "swiglu",
        "bsilu",
        "nelu",
        "relu_nelu",
    ]
    raise ValueError(f"Unsupported activation: {activation}. Supported: {', '.join(supported)}")


# ---------------------------------------------------------------------------
# create_normed_mlp helper (from normed_mlp.py, needed by ResidualNormedMLP)
# ---------------------------------------------------------------------------


def create_normed_mlp(
    dim,
    depth,
    *,
    norm_fn=None,
    activation=nn.ReLU(),
    bias=False,
    activate_last=False,
):
    """Build a simple MLP where each layer has norm->linear->activation."""
    if norm_fn is None:
        norm_fn = RMSNormWD

    layers = []
    for i in range(1, depth + 1):
        is_last = i == depth
        layer_modules = [nn.Linear(dim, dim, bias=bias), norm_fn(dim)]
        if not is_last or activate_last:
            layer_modules.append(activation)
        layers.append(nn.Sequential(*layer_modules))

    return nn.ModuleList(layers)


# ---------------------------------------------------------------------------
# Helper: label & init parameters following ramenGPT conventions
# ---------------------------------------------------------------------------


def _label_mlp_params(module: nn.Module, c_proj_lr_mul: float):
    """Walk *module* and set `.label = "mlp"` on every 2-D parameter.

    For down-projection parameters (named ``*proj*``, ``*to_out*``, or
    ``*c_proj*``) also set `.lr_mul = c_proj_lr_mul`.

    Sets `.aro_transpose` for ARO symmetry-aware orientation:
    input-side params (c_fc, up, gate) get transposed; output-side
    params (c_proj, down_proj, to_out) do not.
    """
    for name, p in module.named_parameters():
        if p.ndim >= 2:
            p.label = "mlp"
            lname = name.lower()
            if "proj" in lname or "to_out" in lname or "c_proj" in lname:
                p.lr_mul = c_proj_lr_mul
            # ARO orientation: output-side (c_proj/down/to_out) = no transpose
            is_output_side = "c_proj" in lname or "down" in lname or "to_out" in lname
            p.aro_transpose = not is_output_side


# ===========================================================================
#  MLP variants
# ===========================================================================


class DefaultMLP(nn.Module):
    """Original 2-layer MLP from model.py (raw nn.Parameter with transposed layout)."""

    needs_external_norm = True

    def __init__(
        self,
        dim: int,
        c_proj_lr_mul: float,
        std_scale: float,
        activation: str = "relu_squared",
        ffn_dim: int | None = None,
        low_rank_config: dict | None = None,
    ):
        super().__init__()

        self.low_rank_config = _resolve_low_rank_config(low_rank_config)
        self.use_low_rank = self.low_rank_config["enabled"] and self.low_rank_config.get(
            "apply_mlp", True
        )
        self.low_rank_pairs: list[tuple[Tensor, Tensor]] = []

        self.is_glu, self.activation_fn = _get_activation_spec(activation)

        if ffn_dim is None:
            ffn_dim = 4 * dim
        if self.is_glu:
            c_fc_dim = 2 * int(ffn_dim)
        else:
            c_fc_dim = ffn_dim

        if self.use_low_rank:
            self.c_fc = LowRankLinear(
                in_features=dim,
                out_features=c_fc_dim,
                rank_ratio=self.low_rank_config["rank_ratio"],
                rank=self.low_rank_config["rank"],
                min_rank=self.low_rank_config["min_rank"],
                max_rank=self.low_rank_config["max_rank"],
                label="mlp",
                lr_mul=1.0,
                wd_mul=1.0,
            )
            self.c_proj = LowRankLinear(
                in_features=ffn_dim,
                out_features=dim,
                rank_ratio=self.low_rank_config["rank_ratio"],
                rank=self.low_rank_config["rank"],
                min_rank=self.low_rank_config["min_rank"],
                max_rank=self.low_rank_config["max_rank"],
                label="mlp",
                lr_mul=c_proj_lr_mul,
                wd_mul=1.0,
            )
            self.low_rank_pairs = [(self.c_fc.A, self.c_fc.B), (self.c_proj.A, self.c_proj.B)]
        else:
            self.c_fc = nn.Parameter(torch.empty(c_fc_dim, dim))
            self.c_proj = nn.Parameter(torch.empty(ffn_dim, dim))
            self.c_fc.label = "mlp"
            self.c_fc.aro_transpose = True  # input-side
            self.c_proj.label = "mlp"
            self.c_proj.lr_mul = c_proj_lr_mul
            self.c_proj.aro_transpose = False  # output-side
        self.ffn_dim = ffn_dim

        std = std_scale * (dim**-0.5)
        bound = (3**0.5) * std
        with torch.no_grad():
            if self.use_low_rank:
                # Scale the low-rank factors with the existing MLP width-aware variance.
                self.c_fc.A.uniform_(-bound, bound)
                self.c_fc.B.uniform_(-bound, bound)
                self.c_proj.A.uniform_(-bound, bound)
                self.c_proj.B.uniform_(-bound, bound)
                self.c_proj.B.zero_()
            else:
                self.c_fc.uniform_(-bound, bound)
                self.c_proj.zero_()

    def forward(self, x: Tensor):
        if self.use_low_rank:
            x = self.c_fc(x)
        else:
            x = F.linear(x, self.c_fc.type_as(x))
        if self.is_glu:
            x_gate, x_sig = x.chunk(2, dim=-1)
            x = self.activation_fn(x_gate) * x_sig
        else:
            x = self.activation_fn(x)
        if self.use_low_rank:
            x = self.c_proj(x)
        else:
            x = F.linear(x, self.c_proj.T.type_as(x))
        return x


# ---------------------------------------------------------------------------


class StackedFeedforward(nn.Module):
    """Stacked residual feedforward with internal RMSNorm (adapted from ff.py)."""

    needs_external_norm = False

    def __init__(
        self,
        dim: int,
        c_proj_lr_mul: float,
        std_scale: float,
        activation: str = "relu_squared",
        ffn_dim: int | None = None,
        *,
        depth: int = 1,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * dim
        ffn_dim = int(ffn_dim)

        is_glu, act_fn = _get_activation_spec(activation)
        act_module = _to_module(act_fn)

        layers = []
        for _ in range(depth):
            if is_glu:
                layer = nn.Sequential(
                    RMSNormWD(dim),
                    nn.Linear(dim, 2 * ffn_dim, bias=False),
                    _GLUActivation(act_fn),
                    nn.Linear(ffn_dim, dim, bias=False),
                )
            else:
                layer = nn.Sequential(
                    RMSNormWD(dim),
                    nn.Linear(dim, ffn_dim, bias=False),
                    act_module,
                    nn.Linear(ffn_dim, dim, bias=False),
                )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        # Label all 2D params and explicitly set lr_mul on down-projections
        _label_mlp_params(self, c_proj_lr_mul)
        with torch.no_grad():
            for layer in self.layers:
                down_proj = layer[-1]  # last Linear is the down-projection
                down_proj.weight.lr_mul = c_proj_lr_mul
                down_proj.weight.zero_()

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x) + x
        return x


class _WrappedActivation(nn.Module):
    """Wrap a plain function as an nn.Module for use in nn.Sequential."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _to_module(act_fn) -> nn.Module:
    """Return *act_fn* as an nn.Module, wrapping plain callables if needed."""
    if isinstance(act_fn, nn.Module):
        return act_fn
    return _WrappedActivation(act_fn)


class _GLUActivation(nn.Module):
    """GLU-style gated activation: split input in half, apply act to gate."""

    def __init__(self, act_fn):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, x):
        gate, value = x.chunk(2, dim=-1)
        return self.act_fn(gate) * value


# ---------------------------------------------------------------------------


class NormalizedFeedforward(nn.Module):
    """L2-norm-based nGPT feedforward (adapted from nff.py)."""

    needs_external_norm = False

    def __init__(
        self,
        dim: int,
        c_proj_lr_mul: float,
        std_scale: float,
        activation: str = "relu_squared",
        ffn_dim: int | None = None,
        *,
        depth: int = 1,
        alpha_init: float | None = None,
        s_hidden_init: float = 1.0,
        s_hidden_scale: float = 1.0,
        s_gate_init: float = 1.0,
        s_gate_scale: float = 1.0,
    ):
        super().__init__()
        expand_factor = (ffn_dim / dim) if ffn_dim is not None else 4.0
        alpha_init = _default(alpha_init, 1.0 / max(depth, 1))

        self.layers = nn.ModuleList()
        for _ in range(depth):
            ff = _nFeedforward(
                dim,
                expand_factor=expand_factor,
                s_hidden_init=s_hidden_init,
                s_hidden_scale=s_hidden_scale,
                s_gate_init=s_gate_init,
                s_gate_scale=s_gate_scale,
            )
            self.layers.append(
                Residual(ff, dim, alpha_init, dim**-0.5)
            )

        _label_mlp_params(self, c_proj_lr_mul)

    def forward(self, x: Tensor):
        x = _l2norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class _nFeedforward(nn.Module):
    """Single nGPT feedforward block (SiLU-gated, L2-norm weights)."""

    def __init__(
        self,
        dim,
        expand_factor=4.0,
        s_hidden_init=1.0,
        s_hidden_scale=1.0,
        s_gate_init=1.0,
        s_gate_scale=1.0,
    ):
        super().__init__()
        dim_inner = int(dim * expand_factor * 2 / 3)
        self.dim = dim
        self.to_hidden = NormLinear(dim, dim_inner)
        self.to_gate = NormLinear(dim, dim_inner)
        self.hidden_scale = Scale(dim_inner, s_hidden_init, s_hidden_scale)
        self.gate_scale = Scale(dim_inner, s_gate_init, s_gate_scale)
        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)
        hidden = hidden * self.hidden_scale().type_as(hidden)
        gate = gate * self.gate_scale().type_as(gate) * (self.dim**0.5)
        hidden = F.silu(gate) * hidden
        return self.to_out(hidden)


# ---------------------------------------------------------------------------


class ResidualNormedMLP(nn.Module):
    """Deep residual MLP with periodic residuals (Wang et al. arXiv:2503.14858)."""

    needs_external_norm = False

    def __init__(
        self,
        dim: int,
        c_proj_lr_mul: float,
        std_scale: float,
        activation: str = "relu_squared",
        ffn_dim: int | None = None,
        *,
        depth: int = 8,
        residual_every: int = 4,
    ):
        super().__init__()
        assert depth % residual_every == 0, "`depth` must be divisible by `residual_every`"

        is_glu, act_fn = _get_activation_spec(activation)
        if is_glu:
            raise ValueError(
                f"GLU activations ({activation!r}) are not supported with mlp_type='residual_normed' "
                f"because its dim→dim layers cannot do the gated split. "
                f"Use a non-gated activation (relu_squared, gelu, silu, etc.)."
            )
        act_module = _to_module(act_fn)

        blocks = []
        for _ in range(depth // residual_every):
            block = create_normed_mlp(
                dim=dim,
                depth=residual_every,
                norm_fn=RMSNormWD,
                activation=act_module,
                bias=False,
                activate_last=True,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNormWD(dim)

        # Label all 2D params and explicitly set lr_mul on last linear of each block
        _label_mlp_params(self, c_proj_lr_mul)
        with torch.no_grad():
            for block in self.blocks:
                # block is a ModuleList; last sub-layer's first element is nn.Linear
                last_layer_seq = block[-1]
                for m in last_layer_seq.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.lr_mul = c_proj_lr_mul
                        m.weight.zero_()
                        break

    def forward(self, x: Tensor):
        for block in self.blocks:
            residual = x
            for layer in block:
                x = layer(x)
            x = residual + x
        x = self.final_norm(x)
        return x


# ===========================================================================
#  Factory
# ===========================================================================

_MLP_REGISTRY = {
    "default": DefaultMLP,
    "ff": StackedFeedforward,
    "nff": NormalizedFeedforward,
    "residual_normed": ResidualNormedMLP,
}

MLP_TYPES = tuple(_MLP_REGISTRY.keys())


def create_mlp(
    mlp_type: str,
    dim: int,
    c_proj_lr_mul: float,
    std_scale: float,
    activation: str = "relu_squared",
    ffn_dim: int | None = None,
    low_rank_config: dict | None = None,
    **kwargs,
) -> nn.Module:
    """Build an MLP variant.

    Returns a module with ``forward(x) -> Tensor`` and a
    ``needs_external_norm: bool`` class attribute.
    """
    if mlp_type not in _MLP_REGISTRY:
        raise ValueError(
            f"Unknown mlp_type={mlp_type!r}. Choose from: {', '.join(MLP_TYPES)}"
        )
    if "low_rank_config" in kwargs:
        low_rank_config = kwargs.pop("low_rank_config")
    cls = _MLP_REGISTRY[mlp_type]
    if mlp_type == "default":
        kwargs["low_rank_config"] = low_rank_config
    return cls(
        dim=dim,
        c_proj_lr_mul=c_proj_lr_mul,
        std_scale=std_scale,
        activation=activation,
        ffn_dim=ffn_dim,
        **kwargs,
    )
