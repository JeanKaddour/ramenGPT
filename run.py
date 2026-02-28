import argparse
import glob
import os
import random
import subprocess
import sys
import uuid
import warnings

import numpy as np
import torch

from utils import (
    configure_torch_runtime,
    load_config,
    patch_triton_shared_memory,
    setup_gpu_environment,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.py",
        help="Path to configuration file (default: config/base.py)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--early_stop_steps",
        type=int,
        default=None,
        help="Stop training after this many steps (does not affect LR/batch schedules)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=[
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
        ],
        help="Activation function to use (overrides config)",
    )
    parser.add_argument(
        "--ffn_dim",
        type=int,
        default=None,
        help="FFN hidden width (overrides config; default auto from activation)",
    )
    parser.add_argument(
        "--mlp_type",
        type=str,
        default=None,
        choices=["default", "ff", "nff", "residual_normed"],
        help="MLP variant to use (overrides config)",
    )
    parser.add_argument(
        "--residual_connection_mode",
        type=str,
        default=None,
        choices=["standard", "hc", "mhc", "kromhc", "residual"],
        help="Residual connection mode to use (overrides config)",
    )
    parser.add_argument(
        "--residual_connection_num_streams",
        type=int,
        default=None,
        help="Residual stream count for hc/mhc modes (overrides config)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=None,
        help="Save checkpoints every K steps (overrides training_config.checkpoint_every; 0 disables periodic saves)",
    )
    return parser.parse_args()


def _has_files(patterns):
    if not patterns:
        return False
    if isinstance(patterns, str):
        return len(glob.glob(patterns)) > 0
    return any(len(glob.glob(pat)) > 0 for pat in patterns)


def _ensure_fineweb10b_cached_data(config):
    data_config = getattr(config, "data_config", None)
    if not isinstance(data_config, dict):
        return

    train_files = data_config.get("train_files", "")
    val_files = data_config.get("val_files", "")
    if (not isinstance(train_files, str) or "fineweb10B" not in train_files) and (
        not isinstance(val_files, str) or "fineweb10B" not in val_files
    ):
        return

    if _has_files(train_files) and _has_files(val_files):
        return

    script_path = os.path.join(os.path.dirname(__file__), "data", "cached_fineweb10B.py")
    if not os.path.exists(script_path):
        print(
            f"Warning: expected cached data script not found at {script_path}; skipping auto-download"
        )
        return

    print("Auto-downloading FineWeb10B cached shards (9 chunks)...")
    subprocess.run([sys.executable, script_path, "9"], check=True)


def _apply_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def _apply_relaxed_compile_from_config(config):
    """Apply relaxed compile default from config unless user already set env var."""
    if os.environ.get("RAMENGPT_RELAXED_COMPILE"):
        return

    relaxed_compile = False
    compilation_config = getattr(config, "compilation_config", {})
    if isinstance(compilation_config, dict):
        relaxed_compile = bool(compilation_config.get("relaxed_compile", False))

    if relaxed_compile:
        os.environ["RAMENGPT_RELAXED_COMPILE"] = "1"
        print("  -> Relaxed compile enabled by config: compilation_config.relaxed_compile=True")


def main():
    args = _parse_args()

    if args.config is None:
        raise SystemExit("No config path provided")
    config = load_config(args.config)
    _apply_relaxed_compile_from_config(config)

    detected_gpu_info = setup_gpu_environment()
    # Patch Triton shared memory reporting before any compilation paths are used.
    patch_triton_shared_memory(detected_gpu_info)

    warnings.filterwarnings("ignore", message="The fast path is not available")

    # Import torch only after environment tuning above.
    import torch

    torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems

    configure_torch_runtime(detected_gpu_info)

    if args.seed is not None:
        _apply_seed(args.seed)

    assert torch.cuda.is_available()
    torch.cuda.set_device(torch.device("cuda:0"))

    _ensure_fineweb10b_cached_data(config)

    from train import run_training

    run_training(config, args, "", detected_gpu_info, uuid.uuid4())


if __name__ == "__main__":
    main()
