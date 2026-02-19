# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ramenGPT trains small GPT models on a single GPU. It is a non-distributed fork of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), preserving the same architecture patterns (FlexAttention, YaRN RoPE, value embeddings, skip connections, gating) but adapted for single-GPU training with gradient accumulation.

## Commands

```bash
# Setup
uv venv && uv sync

# Training (default config)
uv run run.py --config config/base.py

# Smoke test (quick validation)
uv run run.py --config config/base.py --early_stop_steps 10 --seed 123

# Download data (first 9 FineWeb10B shards; auto-triggered on first run)
uv run python data/cached_fineweb10B.py 9

# CLI overrides
uv run run.py --activation swiglu --ffn_dim 512 --checkpoint_every 100
```

No test suite exists. Validate via smoke runs with `--early_stop_steps` and `--seed` for reproducibility.

## Architecture

**Entrypoint flow:** `run.py` (CLI + GPU setup) -> `utils.py` (env config, Triton patches) -> `train.py:run_training()` (training loop)

**Core files:**
- `model.py` — GPT model: `CausalSelfAttention` (merged QKVO weights, FlexAttention, sparse gating), `MLP` (configurable activation/gating), `Block`, `GPT` (YaRN RoPE, smear/skip gates, value embeddings, logit softcapping)
- `train.py` — `TrainingManager` (batch/window/embed schedules), data loading (`BOSFinder`, `DataPreloader`), training loop with gradient accumulation, validation, checkpointing
- `optimizers.py` — `Adam`, `NorMuon` (Muon with polar express orthogonalization + variance reduction), `BAM` (SinkNorm-based), `AROSinkhorn` (learned rotation + Sinkhorn normalization)
- `config/base.py` — All hyperparameters in a single flat Python module with dicts: `model_config`, `optimizer_config`, `training_config`, `data_config`, etc.
- `utils.py` — GPU architecture detection, Triton shared memory patches (Blackwell workaround), torch compilation flags

**Key design patterns:**
- Parameters carry `.label` and `.lr_mul`/`.wd_mul` attributes for per-parameter optimizer grouping
- Three-optimizer split: Adam (head/embed), matrix optimizer (NorMuon/BAM/ARO for hidden layers), scalar Adam (lambdas/scalars)
- Stepped schedules for batch size, window size, and embed/lm_head weight untying (at `split_frac` of training)
- Scalar optimizer freezes for N steps after schedule transitions
- Config is a plain Python module loaded via `importlib`; override via CLI args or env vars (`DATA_PATH`, `BATCH_SIZE_MULTIPLE`)

**Data format:** Pre-tokenized FineWeb shards (`.bin` files with 256-int32 header + uint16 tokens). BOS-aligned batching assembles sequences starting at document boundaries.

## Validation Workflow

After every newly introduced method or refactor, run a 10-step smoke test to verify nothing breaks:

```bash
uv run run.py --config config/base.py --early_stop_steps 10 --seed 123
```

Always keep WandB online by default for training runs (including smoke checks), and use offline mode only when explicitly requested or required.

## Coding Style

- PEP 8 with 4-space indentation, `snake_case` for config keys
- Black (line-length 100) and isort configured in `pyproject.toml` but not enforced via CI
- Use `uv run` for all script execution
