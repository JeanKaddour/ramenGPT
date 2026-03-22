"""Thin Modal wrapper for running ramenGPT training in the cloud.

Usage:
    # One-time data setup
    uv run modal run modal_app.py -- hydrate --num-shards 9

    # Single training run
    uv run modal run modal_app.py -- train --args --config config/base.py --seed 42

    # Parallel sweep from shell script
    uv run modal run modal_app.py -- train --script run_diversity_sweep.sh
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "ramengpt"
FUNCTION_TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours
PYTHON_VERSION = "3.10"
DATA_VOLUME_NAME = "ramengpt-data"
DATA_MOUNT = "/root/ramengpt/data/fineweb10B"
WORKDIR = "/root/ramengpt"

# ---------------------------------------------------------------------------
# WandB secrets
# ---------------------------------------------------------------------------

WANDB_ENV_KEYS = ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE")


def _collect_wandb_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in WANDB_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            env[key] = val
    return env


_wandb_env = _collect_wandb_env()
_wandb_secret = modal.Secret.from_dict(_wandb_env) if _wandb_env else None

# ---------------------------------------------------------------------------
# App, image, volumes
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "torch",
        "datasets",
        "tiktoken",
        "numpy",
        "huggingface-hub",
        "kernels",
        "einops",
        "setuptools",
        "wandb",
    )
    .add_local_dir(
        ".",
        remote_path=WORKDIR,
        ignore=[".venv", "wandb", "logs", "data/fineweb10B", "__pycache__", ".git"],
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RunResult(TypedDict):
    label: str
    artifacts: dict[str, bytes]


def _stream_subprocess(
    command: list[str],
    label: str,
    cwd: str,
    env: dict[str, str] | None = None,
) -> None:
    prefix = f"[{label}] "
    print(f"{prefix}starting: {' '.join(command)}", flush=True)

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_newline = True
    try:
        assert process.stdout is not None
        for line in process.stdout:
            last_newline = line.endswith("\n")
            sys.stdout.write(f"{prefix}{line}" if last_newline else f"{prefix}{line}\n")
            sys.stdout.flush()
    finally:
        if process.stdout is not None:
            process.stdout.close()

    rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command)

    if not last_newline:
        sys.stdout.write("\n")
        sys.stdout.flush()
    print(f"{prefix}completed", flush=True)


def _collect_log_artifacts(workdir: str) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    logs_dir = os.path.join(workdir, "logs")
    if not os.path.isdir(logs_dir):
        return artifacts
    for entry in os.listdir(logs_dir):
        log_file = os.path.join(logs_dir, entry, "train.log")
        if os.path.isfile(log_file):
            key = f"logs/{entry}/train.log"
            with open(log_file, "rb") as f:
                artifacts[key] = f.read()
    return artifacts


def _write_artifacts(save_dir: Path, artifacts: dict[str, bytes]) -> None:
    for filename, payload in artifacts.items():
        target = save_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print(f"  Saved {target}")


# ---------------------------------------------------------------------------
# Shell script parser
# ---------------------------------------------------------------------------


def parse_experiment_script(script_path: str) -> list[list[str]]:
    """Parse a shell script and extract ``run.py`` invocations as arg lists."""
    with open(script_path) as f:
        lines = f.readlines()

    # Resolve simple variable assignments: VAR=value or VAR="value"
    variables: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(\w+)=(.*)$", stripped)
        if m and not stripped.startswith("export "):
            var_name = m.group(1)
            var_value = m.group(2).strip().strip('"').strip("'")
            # Expand previously defined variables within the value
            for k, v in variables.items():
                var_value = var_value.replace(f"${k}", v).replace(f"${{{k}}}", v)
            variables[var_name] = var_value

    runs: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        # Strip trailing redirects: 2>&1, | tail -3, etc.
        stripped = re.sub(r"\s*2>&1.*$", "", stripped)
        stripped = re.sub(r"\s*\|.*$", "", stripped)

        # Match: uv run run.py ... OR uv run python run.py ...
        m = re.match(r"^uv run (?:python )?run\.py\s+(.*)", stripped)
        if not m:
            continue

        arg_string = m.group(1)
        # Expand $VAR and ${VAR}
        for var_name, var_value in variables.items():
            arg_string = arg_string.replace(f"${{{var_name}}}", var_value)
            arg_string = arg_string.replace(f"${var_name}", var_value)

        # Warn and skip if the args reference a /tmp config (heredoc-generated)
        if "/tmp/" in arg_string:
            print(
                f"Warning: skipping run with /tmp config (heredocs not supported): "
                f"run.py {arg_string}"
            )
            continue

        runs.append(shlex.split(arg_string))

    return runs


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100",
    timeout=FUNCTION_TIMEOUT_SECONDS,
    volumes={DATA_MOUNT: data_volume},
    env={"RAMENGPT_RUN_CONTEXT": "modal"},
    secrets=[_wandb_secret] if _wandb_secret is not None else [],
)
def run_training_remote(
    run_args: list[str],
    run_label: str = "run",
    wandb_group: str | None = None,
) -> RunResult:
    """Run one ``run.py`` invocation remotely and return log artifacts."""
    command = [sys.executable, "-u", "run.py"] + run_args
    env = os.environ.copy()
    if wandb_group:
        env["WANDB_RUN_GROUP"] = wandb_group

    _stream_subprocess(command, run_label, cwd=WORKDIR, env=env)

    artifacts = _collect_log_artifacts(WORKDIR)
    return {"label": run_label, "artifacts": artifacts}


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=30 * 60,
)
def hydrate_data(num_shards: int = 9) -> None:
    """Download FineWeb10B shards from HuggingFace into the data volume."""
    from huggingface_hub import hf_hub_download

    target_dir = "/data"
    os.makedirs(target_dir, exist_ok=True)

    existing = set(os.listdir(target_dir)) if os.path.isdir(target_dir) else set()

    def _download(fname: str) -> None:
        if fname in existing:
            print(f"  {fname} already present, skipping")
            return
        print(f"  Downloading {fname} ...")
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=target_dir,
        )

    _download("fineweb_val_000000.bin")
    for i in range(1, num_shards + 1):
        _download(f"fineweb_train_{i:06d}.bin")

    data_volume.commit()
    print(f"Data volume hydrated: 1 val + {num_shards} train shards")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(*cli_args: str) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="uv run modal run modal_app.py --",
        description="Run ramenGPT training on Modal",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- hydrate ---
    hydrate_p = subparsers.add_parser("hydrate", help="Download FineWeb data to Modal volume")
    hydrate_p.add_argument(
        "--num-shards", type=int, default=9, help="Number of train shards (default: 9)"
    )

    # --- train ---
    train_p = subparsers.add_parser("train", help="Run training experiments")
    source = train_p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--script", type=str, help="Shell script containing uv run run.py invocations"
    )
    source.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass directly to run.py (single run)",
    )
    train_p.add_argument("--gpu", type=str, default="A100", help="Modal GPU type (default: A100)")
    train_p.add_argument(
        "--save-dir",
        type=str,
        default="modal_artifacts",
        help="Local directory for returned artifacts (default: modal_artifacts)",
    )

    args = parser.parse_args(cli_args)

    # --- hydrate command ---
    if args.command == "hydrate":
        print(f"Hydrating data volume with {args.num_shards} shards ...")
        hydrate_data.remote(num_shards=args.num_shards)
        print("Done.")
        return

    # --- train command ---
    save_dir = Path(args.save_dir)

    if args.script:
        runs = parse_experiment_script(args.script)
        if not runs:
            raise SystemExit(f"No `uv run run.py` invocations found in {args.script}")
        script_stem = Path(args.script).stem
        labeled_runs = [(ra, f"{script_stem}/{i + 1}") for i, ra in enumerate(runs)]
    else:
        labeled_runs = [(args.args, "single")]

    # WandB group for multi-run batches
    wandb_group: str | None = None
    if len(labeled_runs) > 1:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        wandb_group = f"modal-{stamp}-{uuid.uuid4().hex[:8]}"
        print(f"WandB group: {wandb_group}")

    # Choose the right function handle (allow GPU override)
    fn = run_training_remote
    if args.gpu != "A100":
        fn = run_training_remote.with_options(gpu=args.gpu)

    started_at = time.time()

    if len(labeled_runs) == 1:
        run_args, label = labeled_runs[0]
        print(f"Dispatching single run: run.py {' '.join(run_args)}", flush=True)
        result = fn.remote(run_args=run_args, run_label=label, wandb_group=wandb_group)
        _write_artifacts(save_dir, result["artifacts"])
    else:
        print(f"Dispatching {len(labeled_runs)} runs in parallel ...", flush=True)
        results = list(
            fn.starmap(
                [(ra, label, wandb_group) for ra, label in labeled_runs],
                return_exceptions=True,
            )
        )

        failures: list[str] = []
        for (_, label), result in zip(labeled_runs, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"  {label}: {result}")
                continue
            _write_artifacts(save_dir / label, result["artifacts"])
            print(f"Collected artifacts for {label}", flush=True)

        if failures:
            raise RuntimeError("Some runs failed:\n" + "\n".join(failures))

    elapsed = time.time() - started_at
    print(f"Completed in {elapsed:.1f}s")
