import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# FlexAttention compatibility import. Not all environments expose this module.
try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
except ImportError:  # pragma: no cover
    BlockMask = None
    flex_attention = None

# -----------------------------------------------------------------------------
# FlexAttention kernel options for different GPU architectures
# GPUs with limited shared memory need reduced num_stages/block sizes in backward pass.
# -----------------------------------------------------------------------------

_flex_attention_kernel_options = None

def set_flex_attention_kernel_options(gpu_arch: str | None):
    global _flex_attention_kernel_options
    if gpu_arch in ('blackwell', 'ampere'):
        _flex_attention_kernel_options = {
            'num_stages': 1,
            'num_warps': 4,
            'BLOCK_M': 64,
            'BLOCK_N': 64,
            'BLOCK_M1': 32,
            'BLOCK_N1': 32,
            'BLOCK_M2': 32,
            'BLOCK_N2': 32,
        }
    else:
        _flex_attention_kernel_options = None
    return _flex_attention_kernel_options

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


def _normalize_activation_name(activation: str) -> str:
    if not isinstance(activation, str):
        raise TypeError(f"activation must be a string, got {type(activation)!r}")
    normalized = activation.strip().lower()
    if not normalized:
        raise ValueError("activation cannot be empty")
    return normalized


def _get_activation_spec(activation: str):
    activation = _normalize_activation_name(activation)
    if activation == 'relu':
        return False, F.relu
    if activation == 'gelu':
        return False, F.gelu
    if activation == 'swish':
        return False, F.silu
    if activation == 'silu':
        return False, F.silu
    if activation == 'linear':
        return False, lambda x: x
    if activation == 'identity':
        return False, lambda x: x
    if activation == 'relu_squared':
        return False, lambda x: F.relu(x).square()
    if activation == 'gelu_squared':
        return False, lambda x: F.gelu(x).square()
    if activation == 'swish_squared':
        return False, lambda x: F.silu(x).square()
    if activation == 'silu_squared':
        return False, lambda x: F.silu(x).square()
    if activation == 'geglu':
        return True, F.gelu
    if activation == 'swiglu':
        return True, F.silu

    supported = [
        'relu', 'gelu', 'swish', 'silu', 'linear', 'identity',
        'relu_squared', 'gelu_squared', 'swish_squared', 'silu_squared',
        'geglu', 'swiglu',
    ]
    raise ValueError(f"Unsupported activation: {activation}. Supported: {', '.join(supported)}")

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    """Apply rotary position embeddings to input tensor"""
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, :x_BTHD.size(-3), None, :],
        sin[None, :x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
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
class Yarn(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) for dynamic window size adaptation.
    Allows extending context length during training by adjusting RoPE frequencies.
    """
    def __init__(self, head_dim: int, max_seq_len: int, base_freq: float, block_size: int):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base_freq = base_freq
        self.block_size = block_size
        self.reset()

    def reset(self):
        """Reset to initial state (called at start of training and after warmup)"""
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / self.base_freq) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        # Start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        """
        Apply YaRN interpolation when window size changes.
        This adjusts the RoPE frequencies to handle longer contexts.
        """
        rotations = self.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq = self.angular_freq * (scaling_factor + interpolation_weight * (1 - scaling_factor))
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim: int,
                 layer_idx: int, gating_config: dict, value_embed_layers: list,
                 value_embed_gate_scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.layer_idx = layer_idx
        self.value_embed_gate_scale = value_embed_gate_scale
        hdim = num_heads * head_dim

        assert hdim == dim, "num_heads * head_dim must equal model_dim"
        std = dim ** -0.5
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng

        # Merged QKVO weights (from train_gpt.py)
        # Layout: [Q, K, V, O] each of size (dim, hdim)
        self.qkvo_w = nn.Parameter(torch.empty(dim * 4, hdim))
        self.qkvo_w.label = 'attn'

        with torch.no_grad():
            self.qkvo_w[:dim * 3].uniform_(-bound, bound)  # init QKV weights
            self.qkvo_w[dim * 3:].zero_()  # init O weights to zero

        # Sparse gated attention (from train_gpt.py @classiclarryd)
        gate_input_dim = gating_config['gate_input_dim']
        if gating_config.get('use_attn_gate', True):
            self.attn_gate = CastedLinear(gate_input_dim, num_heads)
            self.attn_gate.weight.label = 'attn_gate'
        else:
            self.attn_gate = None

        # Value embedding gate (only on specific layers)
        if gating_config.get('use_value_embed_gate', True) and layer_idx in value_embed_layers:
            self.value_embed_gate = CastedLinear(gate_input_dim, num_heads)
            self.value_embed_gate.weight.label = 'value_embed_gate'
        else:
            self.value_embed_gate = None

    def forward(self, x: Tensor, ve: Tensor, sa_lambdas: Tensor, block_mask,
                cos: Tensor, sin: Tensor, attn_scale: float, docs: Tensor,
                key_offset: bool = False):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"

        # Apply sa_lambdas[0] to QKV weights (from train_gpt.py)
        q, k, v = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x)).view(
            B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

        # QK norm and RoPE
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        # Key offset: shift keys forward for the stationary head dims (from train_gpt.py)
        # Enables 1-layer induction on long attention window layers
        if key_offset:
            k[:, 1:, :, self.head_dim // 4:self.head_dim // 2] = k[:, :-1, :, self.head_dim // 4:self.head_dim // 2].clone()
            k[:, 1:, :, 3 * self.head_dim // 4:] = k[:, :-1, :, 3 * self.head_dim // 4:].clone()

        # Value embedding with gating (from train_gpt.py)
        if ve is not None and self.value_embed_gate is not None:
            ve_gate_out = self.value_embed_gate_scale * torch.sigmoid(
                self.value_embed_gate(x[..., :self.value_embed_gate.weight.size(-1)])
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
            return torch.where(mask, score, -float('inf'))

        # FlexAttention
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                          block_mask=block_mask, scale=attn_scale, score_mod=score_mod,
                          kernel_options=_flex_attention_kernel_options).transpose(1, 2)

        # Attention gating (from train_gpt.py)
        if self.attn_gate is not None:
            y = y * torch.sigmoid(
                self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])
            ).view(B, T, self.num_heads, 1)

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)

        # Output projection using merged weights with sa_lambdas[1]
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))
        return y


class MLP(nn.Module):
    """
    MLP block matching train_gpt.py structure exactly.
    Uses raw nn.Parameter with transposed layout and configurable activation.
    """
    def __init__(self, dim: int, c_proj_lr_mul: float, std_scale: float,
                 activation: str = 'relu_squared', ffn_dim: int | None = None):
        super().__init__()

        self.activation, self.activation_fn = _get_activation_spec(activation)
        self.activation_name = _normalize_activation_name(activation)

        if ffn_dim is None:
            ffn_dim = 4 * dim
        if self.activation:
            ffn_dim = int(ffn_dim)
            c_fc_dim = 2 * ffn_dim
        else:
            c_fc_dim = ffn_dim

        # Transposed layout to match attention weights (from train_gpt.py)
        self.c_fc = nn.Parameter(torch.empty(c_fc_dim, dim))
        self.c_proj = nn.Parameter(torch.empty(ffn_dim, dim))
        # Label all modules for explicit optimizer grouping
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        self.c_proj.lr_mul = c_proj_lr_mul  # Match train_gpt.py
        self.ffn_dim = ffn_dim

        std = std_scale * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.type_as(x))
        if self.activation:
            x_gate, x_sig = x.chunk(2, dim=-1)
            x = self.activation_fn(x_gate) * x_sig
        else:
            x = self.activation_fn(x)
        x = F.linear(x, self.c_proj.T.type_as(x))
        return x



class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int,
                 head_dim: int, skip_attention: bool,
                 c_proj_lr_mul: float, mlp_std_scale: float, value_embed_gate_scale: float,
                 gating_config: dict = None, value_embed_layers: list = None,
                 activation: str = 'relu_squared', ffn_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        gating_config = gating_config or {}
        value_embed_layers = value_embed_layers or []

        # Skip attention of specific layers (e.g., layer 6 in train_gpt.py) by @YouJiacheng
        if not skip_attention:
            self.attn = CausalSelfAttention(
                dim, num_heads, max_seq_len, head_dim,
                layer_idx, gating_config, value_embed_layers,
                value_embed_gate_scale=value_embed_gate_scale
            )
        else:
            self.attn = None

        # FFN activation and width are configurable per model_config.
        self.mlp = MLP(
            dim,
            c_proj_lr_mul=c_proj_lr_mul,
            std_scale=mlp_std_scale,
            activation=activation,
            ffn_dim=ffn_dim,
        )
        # Labels already set inside MLP.__init__()

    def forward(self, x: Tensor, ve: Tensor, sa_lambdas: Tensor, block_mask,
                cos: Tensor, sin: Tensor, attn_scale: float, docs: Tensor,
                key_offset: bool = False):
        # Attention branch
        if self.attn is not None:
            attn_out = self.attn(norm(x), ve, sa_lambdas, block_mask, cos, sin, attn_scale, docs, key_offset)
            x = x + attn_out
        # MLP branch
        mlp_out = self.mlp(norm(x))
        x = x + mlp_out
        return x

class GPT(nn.Module):
    def __init__(self, model_config: dict, attention_config: dict, lambda_config: dict,
                 lr_multipliers: dict, max_seq_len: int, attention_pattern_config: dict,
                 gating_config: dict = None, skip_config: dict = None,
                 rope_config: dict = None, embed_config: dict = None,
                 wd_multipliers: dict = None):
        super().__init__()
        self.model_config = model_config
        self.attention_config = attention_config
        self.lambda_config = lambda_config
        self.attention_pattern_config = attention_pattern_config
        self.gating_config = gating_config or {}
        self.skip_config = skip_config or {}
        self.rope_config = rope_config or {}
        self.embed_config = embed_config or {}

        c_proj_lr_mul = lr_multipliers['c_proj']
        mlp_init_std_scale = model_config['mlp_init_std_scale']
        lm_head_init_std = model_config['lm_head_init_std']
        embed_padding_multiple = model_config['embed_padding_multiple']
        eos_token_id = model_config['eos_token_id']
        value_embed_head_indices = model_config['value_embed_head_indices']
        value_embed_mid_layer_count = model_config['value_embed_mid_layer_count']
        value_embed_tail_indices = model_config['value_embed_tail_indices']
        value_embed_gate_scale = model_config['value_embed_gate_scale']
        skip_gate_scale = model_config['skip_gate_scale']
        residual_first_layer_index = model_config['residual_first_layer_index']
        logits_softcap_scale = model_config['logits_softcap_scale']
        logits_softcap_shift = model_config['logits_softcap_shift']
        logits_softcap_divisor = model_config['logits_softcap_divisor']

        vocab_size = model_config['vocab_size']
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        model_dim = model_config['model_dim']
        head_dim = model_config['head_dim']
        block_size = attention_config['block_size']
        self.activation = model_config.get('activation', 'relu_squared')
        self.ffn_dim = model_config.get('ffn_dim', None)

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

        # YaRN for dynamic RoPE adaptation (from train_gpt.py)
        base_freq = self.rope_config['base_freq']
        self.yarn = Yarn(head_dim, max_seq_len, base_freq, block_size)

        # Smear gate: shift token embeddings forward (from train_gpt.py @classiclarryd)
        gate_input_dim = self.gating_config['gate_input_dim']
        if self.gating_config.get('use_smear_gate', True):
            self.smear_gate = CastedLinear(gate_input_dim, 1)
            self.smear_gate.weight.label = 'smear_gate'
            self.smear_gate.weight.lr_mul = lr_multipliers['smear_gate']
        else:
            self.smear_gate = None

        # Skip gate (from train_gpt.py)
        if self.gating_config.get('use_skip_gate', True):
            self.skip_gate = CastedLinear(gate_input_dim, 1)
            self.skip_gate.weight.label = 'skip_gate'
            self.skip_gate.weight.lr_mul = lr_multipliers['skip_gate']
        else:
            self.skip_gate = None

        # Token value embeddings (from train_gpt.py @KoszarskyB)
        self.value_embeds = nn.ModuleList([
            nn.Embedding(vocab_size_padded, model_dim)
            for _ in range(attention_pattern_config['num_value_embeds'])
        ])
        for embed in self.value_embeds:
            nn.init.zeros_(embed.weight)
            embed.weight.label = 'value_embed'

        # Blocks with gating config (matching train_gpt.py)
        value_embed_layers = attention_pattern_config['value_embed_layers']
        self.blocks = nn.ModuleList([
            Block(
                model_dim,
                num_heads,
                max_seq_len,
                i,
                head_dim,
                skip_attention=i in attention_pattern_config['skip_attention_layers'],
                gating_config=self.gating_config,
                value_embed_layers=value_embed_layers,
                c_proj_lr_mul=c_proj_lr_mul,
                mlp_std_scale=mlp_init_std_scale,
                value_embed_gate_scale=value_embed_gate_scale,
                activation=self.activation,
                ffn_dim=self.ffn_dim,
            )
            for i in range(num_layers)
        ])

        # LM head with proper initialization
        self.lm_head = CastedLinear(model_dim, vocab_size_padded, use_fp8=False)
        nn.init.normal_(self.lm_head.weight, mean=0, std=lm_head_init_std)
        self.lm_head.weight.label = 'lm_head'

        # Weight-tied embedding: use lm_head.weight for embed initially
        # Separate embed created when split_embed is set to True
        self.embed = None  # Will use lm_head.weight
        self.split_embed = False

        # x0_lambdas separated for different optimizer treatment
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.x0_lambdas.label = 'x0_lambdas'
        self.x0_lambdas.lr_mul = lr_multipliers['x0_lambdas']

        # Construct scalars parameter
        value_embed_layers_set = set(value_embed_layers)
        resid_init = lambda_config['resid_lambdas_init']
        sa_init = lambda_config['sa_lambdas_init']
        sa_init_no_ve = lambda_config['sa_lambdas_init_no_ve']
        smear_init = lambda_config['smear_lambda_init']
        backout_init = lambda_config['backout_lambda_init']
        skip_lambda_init = lambda_config['skip_lambda_init']

        self.scalars = nn.Parameter(torch.cat([
            resid_init * torch.ones(num_layers),  # resid_lambdas
            *[torch.tensor(sa_init if i in value_embed_layers_set else sa_init_no_ve)
              for i in range(num_layers)],  # SA lambdas
            torch.tensor([smear_init]),  # smear_lambda
            torch.tensor([backout_init]),  # backout_lambda
            torch.tensor([skip_lambda_init]),  # skip_lambda
        ]))
        self.scalars.label = 'scalars'
        self.scalars.lr_mul = lr_multipliers['scalars']

        # Set learning rate and weight decay multipliers
        wd_multipliers = wd_multipliers or {}
        for param in self.value_embeds.parameters():
            param.lr_mul = lr_multipliers['value_embed']
            param.wd_mul = wd_multipliers['value_embed']
        self.lm_head.weight.wd_mul = wd_multipliers['head']
        self.scalars.wd_mul = wd_multipliers['scalars']
        self.x0_lambdas.wd_mul = wd_multipliers['x0_lambdas']
        if self.smear_gate:
            self.smear_gate.weight.wd_mul = wd_multipliers['smear_gate']
        if self.skip_gate:
            self.skip_gate.weight.wd_mul = wd_multipliers['skip_gate']

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

    def create_embed(self):
        """Create separate embedding when weight tying is split"""
        if self.embed is None:
            self.embed = nn.Embedding(self.vocab_size_padded, self.model_dim)
            # Move to correct device and dtype to match lm_head
            self.embed = self.embed.to(device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype)
            # Copy lm_head weights to embed
            self.embed.weight.data.copy_(self.lm_head.weight.data)
            self.embed.weight.label = 'embed'
            # Set wd_mul to match train_gpt.py (150.0 for embed like lm_head)
            self.embed.weight.wd_mul = self._model_wd_multipliers['embed']
        self.split_embed = True

    def create_blockmasks(self, docs: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = self.attention_config['block_size']
        # docs passed in

        def document_causal(b, h, q_idx, kv_idx):
            _ = b, h  # unused but required by FlexAttention API
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            # Convert to float for argsort (CUDA doesn't support bool sorting)
            indices = dense_blockmask.float().argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
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
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
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
        skip_in_layers = self.skip_config['skip_in_layers']
        skip_out_layers = self.skip_config['skip_out_layers']
        backout_layer = self.skip_config['backout_layer']

        # Build value embeddings pattern from config (from train_gpt.py)
        # train_gpt.py pattern: [ve[1], ve[2]] + [None] * (num_layers - 5) + [ve[0], ve[1], ve[2]]
        # 012 ... 012 structure by @YouJiacheng, improved on @leloykun's U-net structure
        # dropping first layer updates this to .12 ... 012
        ve_computed = [value_embed(input_seq) for value_embed in self.value_embeds]
        num_layers = len(self.blocks)
        ve = [ve_computed[i] for i in self._value_embed_head_indices] + \
             [None] * (num_layers - self._value_embed_mid_layer_count) + \
             [ve_computed[i] for i in self._value_embed_tail_indices]
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
        for i, char in enumerate(self.attention_pattern_config['block_mask_pattern']):
            if not requires_mask[i]:
                block_masks.append(None)
                key_offsets.append(False)
            elif char == 'L':
                block_masks.append(long_bm)
                key_offsets.append(True)  # Apply key offset on long window layers
            elif char == 'S':
                block_masks.append(short_bm)
                key_offsets.append(False)
            elif char == 'N':
                block_masks.append(None)  # Skip attention
                key_offsets.append(False)
            else:
                raise ValueError(f"Invalid block mask pattern character: {char}. Use 'L', 'S', or 'N'.")
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
                self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)])
            )
            x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])

        x = x0 = norm(x[None])

        # Extract lambdas from scalars
        resid_lambdas = self.scalars[:self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[self.num_layers:3 * self.num_layers].view(-1, 2)
        backout_lambda = self.scalars[3 * self.num_layers + 1]
        skip_lambda = self.scalars[3 * self.num_layers + 2]

        # Get RoPE values from Yarn
        cos, sin = self.yarn.cos, self.yarn.sin
        attn_scale = self.yarn.attn_scale

        # Skip connections
        skip_connections = []
        x_backout = None

        for i in range(len(self.blocks)):
            # Apply skip connection from earlier layers
            if i in skip_out_layers and skip_connections:
                if self.skip_gate is not None:
                    skip_gate_out = torch.sigmoid(skip_lambda) * self._skip_gate_scale * torch.sigmoid(
                        self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)])
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
            x = self.blocks[i](x, ve[i], sa_lambdas[i], block_masks[i], cos, sin, attn_scale, docs, key_offsets[i])

            # Save skip connection
            if i in skip_in_layers:
                skip_connections.append(x)

            # Save for backout
            if i == backout_layer:
                x_backout = x

        # Backout: subtract contribution from early layers
        if x_backout is not None:
            x = x - backout_lambda * x_backout

        x = norm(x)
        logits = self.lm_head(x)

        # Updated softcap formula @classiclarryd
        # 23 * sigmoid((logits + 5) / 7.5)
        logits = self._logits_softcap_scale * torch.sigmoid((logits + self._logits_softcap_shift) / self._logits_softcap_divisor)
        logits_for_loss = logits.float() if not self.training else logits

        # Language modeling loss
        if self.training:
            loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)),
                                   target_seq, reduction='sum')
        else:
            loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)),
                                   target_seq, reduction='mean')

        return loss
