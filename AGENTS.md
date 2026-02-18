# Repository Guidelines

## Project Structure & Module Organization
`ramenGPT` is a small single-GPU training project. Core code lives at the repository root:
- `run.py`: CLI entrypoint and startup orchestration.
- `train.py`: main training loop, optimizer/scheduler setup, checkpointing, and logging.
- `model.py`: model architecture and tensor utilities.
- `utils.py`: runtime setup, config loading, and environment helpers.
- `optimizers.py`: optimizer implementations used by training.
- `config/base.py`: default training configuration.
- `data/`: dataset scripts for FineWeb and cached token shards.
- `assets/`, `logs/`, `wandb/`: generated output, experiment artifacts, and visual assets.

## Build, Test, and Development Commands
- Use `uv run` for all Python script execution in this repo (for consistent environments and dependency resolution).
- `uv venv`
  - Create a local UV environment.
- `uv sync`
  - Install dependencies from `pyproject.toml`.
- `uv run run.py --help`
  - Show current CLI options (`--config`, `--seed`, `--early_stop_steps`, etc.).
- `uv run run.py --config config/base.py`
  - Start a default training run.
- `uv run run.py --config config/base.py --early_stop_steps 10 --seed 123`
  - Quick smoke check for local validation.
- `uv run python data/cached_fineweb10B.py 9`
  - Download the first 9 FineWeb10B shards for fast local setup.

## Coding Style & Naming Conventions
- Follow PEP 8 style with 4-space indentation.
- Use `snake_case` for config dict keys and function/module names.
- Keep variables/config keys explicit (`train_seq_len`, `num_iterations`, `wandb_project`).
- Prefer small, composable helper functions and preserve existing guard-style assertions for runtime safety.
- If adding formatting/linting tooling, document it here before adoption (none is currently enforced).

## Testing Guidelines
- There is no dedicated unit-test suite yet.
- Validation is currently practical via smoke runs and deterministic reproducibility checks:
  - `--early_stop_steps` for bounded runs.
  - `--seed` to verify repeatability.
- For experiments, keep quick checks on tiny settings before long runs.
- Treat `wandb` usage as optional in local checks; prefer online runs by default and only use offline mode when explicitly requested or when offline access is required.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative summaries (e.g., `Update README.md...`, `Refactor training loop...`).
- Recommended format: imperative subject line (<=72 chars), optional longer body for design rationale.
- PRs should include:
  - Summary of change and impact.
  - Commands used to validate (`uv run run.py ...` args and dataset mode).
  - Any config changes and affected output paths.
  - Reproducibility notes (seed, GPU type, config file).

## Security & Configuration Tips
- Never commit large datasets, local checkpoints, API keys, or `.venv`.
- Keep experiment data in ignored paths (`data/*`, `wandb/`, `logs/`, `checkpoints/`) unless explicitly versioned for a reason.
- Store sensitive tokens (HF/W&B) in environment variables or local secret stores, not files.
