# 🍜 ramenGPT

**Train small GPT models on a single GPU**: Modded [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) w/o distributed training.

![header](assets/header.jpg)

## Quickstart
On a single GPU machine (eg. 30/40/5090 or A/H100), you can run

```bash
uv venv
uv sync
uv run run.py
```

## Implemented Methods

### Optimizers (`optimizers.py`)

| Method | Paper/Source |
|---|---|
| **Muon** (NorMuon) — momentum orthogonalized by Newton-Schulz with variance reduction | [arxiv.org/abs/2510.05491](https://arxiv.org/abs/2510.05491) |
| **Polar Express** — compiled sign-method orthogonalization used inside Muon | [arxiv.org/abs/2505.16932](https://arxiv.org/abs/2505.16932) |
| **LITE** — Muon with flat-direction dynamics enhancement | [arxiv.org/abs/2602.22681](https://arxiv.org/abs/2602.22681) |
| **BAM** — Balanced Axis Momentum; replaces Newton-Schulz with Sinkhorn normalization | [github.com/knightron0/bam](https://github.com/knightron0/bam) |
| **ARO-Sinkhorn** — adaptively rotated optimization with Sinkhorn normalization | [arxiv.org/abs/2602.09006](https://arxiv.org/abs/2602.09006) |
| **Spectron** — optimizer for low-rank matrix factor pairs using polar express + power iteration | [arxiv.org/abs/2602.12429](https://arxiv.org/abs/2602.12429) |

### Architecture (`model.py`, `mlps.py`)

| Method | Paper/Source |
|---|---|
| **FlexAttention** — PyTorch native block-sparse attention | [PyTorch blog](https://pytorch.org/blog/flexattention/) |
| **YaRN RoPE** — dynamic context-length adaptation for rotary embeddings | [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071) |
| **Long-short sliding window attention** — alternating window sizes across layers | [Gemma 2](https://arxiv.org/abs/2408.00118) |
| **Logit softcapping** — sigmoid-bounded output logits | [Gemma 2](https://arxiv.org/abs/2408.00118) |
| **HyperConnections / mHC** — dynamic multi-stream residual routing with manifold constraints | [arxiv.org/abs/2409.19606](https://arxiv.org/abs/2409.19606) |
| **KromHC** — Kronecker-product manifold-constrained hyper-connections with doubly stochastic factors | [arxiv.org/abs/2601.21579](https://arxiv.org/abs/2601.21579) |
| **Normalized feedforward (nFF)** — L2-norm-based nGPT feedforward | [arxiv.org/abs/2410.01131](https://arxiv.org/abs/2410.01131) |
| **Deep residual MLP** — periodic residuals (Wang et al.) | [arxiv.org/abs/2503.14858](https://arxiv.org/abs/2503.14858) |
| **NOBLE / CosNet low-rank branches** — nonlinear additive low-rank branches for attention and MLP projections | [noble.md](noble.md) |

Various other tricks (QK norm, value embeddings, merged QKVO, sparse gated attention, smear/skip gates, key offset, BOS-aligned batching, batch/window size scheduling, embedding split) from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

## NOBLE / CosNet

NOBLE is integrated into the existing `low_rank_config` path. In `mode="noble"`, the model keeps the main dense projection weights and adds a nonlinear low-rank CosNet branch on top. The current implementation covers attention `QKV` and `O`, plus the default 2-layer MLP (`mlp_type="default"`).

Enable it in a config override like this:

```python
from config.base import *

low_rank_config.update(
    dict(
        enabled=True,
        mode="noble",
        rank=64,
    )
)
```

Relevant knobs in `config/base.py`:

```python
low_rank_config = dict(
    enabled=False,
    mode="factorized",  # "factorized" or "noble"
    rank_ratio=0.25,
    rank=None,
    min_rank=1,
    max_rank=None,
    apply_attention=True,
    apply_mlp=True,
    noble_up_init_alpha=0.01,
    noble_lr_power=0.3,
    noble_mix_lr_power=0.45,
    noble_freq_lr_mul=3.0,
    noble_phase_lr_mul=5.0,
    noble_freq_min=0.8,
    noble_freq_max=1.2,
    noble_phase_std=0.1,
)
```

Current constraints:

- `mode="noble"` does not support `optimizer_config["matrix_optimizer"] = "spectron"`.
- Noble MLP branches are only implemented for `model_config["mlp_type"] = "default"`. For other MLP types, set `low_rank_config["apply_mlp"] = False`.
- W&B run names and tags include the low-rank mode, so Noble runs show up as `lrk-noble`.

Quick smoke test:

```python
from config.base import *

low_rank_config.update(
    dict(
        enabled=True,
        mode="noble",
        rank=64,
    )
)
```

Save that as `/tmp/noble_smoke_config.py`, then run:

```bash
uv run run.py --config /tmp/noble_smoke_config.py --early_stop_steps 10 --seed 123
```

# References

* [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
* [nanoGPT](https://github.com/karpathy/nanoGPT)
* [mHC-manifold-constrained-hyper-connections](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)
