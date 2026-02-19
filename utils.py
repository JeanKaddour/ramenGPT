"""Shared utilities for ramenGPT."""

from __future__ import annotations

import importlib.util
import os
import subprocess


def _is_relaxed_compile_enabled() -> bool:
    return os.environ.get("RAMENGPT_RELAXED_COMPILE", "").lower() in {"1", "true", "on", "yes"}


def detect_gpu_architecture():
    """Detect GPU architecture before importing torch."""
    gpu_info = {
        "name": "unknown",
        "compute_capability": None,
        "architecture": "unknown",
        "shared_memory_per_block": None,
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split(",")
                gpu_info["name"] = parts[0].strip() if len(parts) > 0 else "unknown"
                if len(parts) > 1:
                    cc = parts[1].strip()
                    gpu_info["compute_capability"] = cc

                    major = int(cc.split(".")[0]) if "." in cc else int(cc)
                    if major >= 10:
                        gpu_info["architecture"] = "blackwell"
                    elif major >= 9:
                        gpu_info["architecture"] = "hopper"
                    elif major >= 8:
                        if cc.startswith("8.9"):
                            gpu_info["architecture"] = "ada"
                        else:
                            gpu_info["architecture"] = "ampere"
                    elif major >= 7:
                        gpu_info["architecture"] = "volta_turing"
                    else:
                        gpu_info["architecture"] = "legacy"
    except Exception as exc:
        print(f"Warning: Could not detect GPU via nvidia-smi: {exc}")

    return gpu_info


def setup_gpu_environment():
    """Set environment before torch/triton import."""
    gpu_info = detect_gpu_architecture()
    relaxed_compile = _is_relaxed_compile_enabled()
    print(
        f"Detected GPU: {gpu_info['name']} (compute capability: {gpu_info['compute_capability']}, "
        f"architecture: {gpu_info['architecture']})"
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["FLA_CONV_BACKEND"] = "triton"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_cache"
    os.environ["PYTORCH_DISABLE_CUDA_GRAPHS"] = "1"

    if gpu_info["architecture"] == "ampere":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
        os.environ["TRITON_MAX_SHARED_MEMORY"] = "100000"
        if relaxed_compile:
            os.environ["TRITON_NUM_STAGES"] = "2"
            os.environ["TRITON_AUTOTUNE"] = "1"
            print("  -> Configured for Ampere architecture with relaxed compile settings")
        else:
            os.environ["TRITON_AUTOTUNE"] = "0"
            os.environ["TRITON_NUM_STAGES"] = "1"
            os.environ["TORCH_INDUCTOR_FORCE_DISABLE_CACHES"] = "1"
            os.environ["TRITON_DEFAULT_NUM_STAGES"] = "1"
            os.environ["TRITON_DEFAULT_NUM_WARPS"] = "4"
            os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "EXHAUSTIVE"
            os.environ["TORCH_INDUCTOR_DEFAULT_NUM_STAGES"] = "1"
            os.environ["INDUCTOR_TRITON_NUM_STAGES"] = "1"
            print("  -> Configured for Ampere architecture (RTX 30 series)")

    elif gpu_info["architecture"] == "ada":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
        os.environ["TRITON_MAX_SHARED_MEMORY"] = "164000"
        if relaxed_compile:
            print("  -> Configured for Ada architecture with relaxed compile settings")
        else:
            print("  -> Configured for Ada Lovelace architecture (RTX 40 series)")

    elif gpu_info["architecture"] == "blackwell":
        if gpu_info["compute_capability"]:
            os.environ["TORCH_CUDA_ARCH_LIST"] = gpu_info["compute_capability"]
        os.environ["TRITON_MAX_SHARED_MEMORY"] = "228000"
        if relaxed_compile:
            print("  -> Configured for Blackwell architecture with relaxed compile settings")
        else:
            os.environ["TRITON_AUTOTUNE"] = "0"
            os.environ["TRITON_NUM_STAGES"] = "1"
            os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "EXHAUSTIVE"
            os.environ["TORCH_INDUCTOR_DEFAULT_NUM_STAGES"] = "1"
            os.environ["INDUCTOR_TRITON_NUM_STAGES"] = "1"
            os.environ["TORCH_INDUCTOR_FORCE_DISABLE_CACHES"] = "1"
            os.environ["TRITON_DEFAULT_NUM_STAGES"] = "1"
            os.environ["TRITON_DEFAULT_NUM_WARPS"] = "4"
            print("  -> Configured for Blackwell architecture (RTX 50 series)")

    elif gpu_info["architecture"] == "hopper":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
        os.environ["TRITON_MAX_SHARED_MEMORY"] = "228000"
        print("  -> Configured for Hopper architecture (H100, etc.)")
    else:
        print("  -> Using auto-detection for unknown architecture")

    return gpu_info


def configure_torch_runtime(gpu_info: dict):
    """Set conservative torch compilation flags before training."""
    import torch
    import torch._dynamo
    import torch._inductor.config as inductor_config

    relaxed_compile = _is_relaxed_compile_enabled()

    inductor_config.max_autotune = False
    inductor_config.autotune_local_cache = False
    inductor_config.autotune_remote_cache = False
    inductor_config.triton.unique_kernel_names = True
    inductor_config.triton.descriptive_names = False
    inductor_config.triton.cudagraphs = False

    if gpu_info.get("architecture") in ("blackwell", "ampere"):
        if relaxed_compile:
            inductor_config.max_autotune = True
            inductor_config.autotune_local_cache = True
            inductor_config.autotune_remote_cache = True
            inductor_config.fallback_random = False
            inductor_config.force_disable_caches = False
            inductor_config.compile_threads = os.cpu_count() or 4
            if hasattr(inductor_config, "triton"):
                if hasattr(inductor_config.triton, "num_stages"):
                    inductor_config.triton.num_stages = 2
                if hasattr(inductor_config.triton, "num_warps"):
                    inductor_config.triton.num_warps = 8
            print(
                f"  -> Applied relaxed inductor settings for {gpu_info.get('architecture')}"
            )
        else:
            inductor_config.fallback_random = True
            inductor_config.force_disable_caches = True
            inductor_config.compile_threads = 1
            if hasattr(inductor_config, "worker_start_method"):
                inductor_config.worker_start_method = "fork"
            if hasattr(inductor_config, "triton"):
                if hasattr(inductor_config.triton, "num_stages"):
                    inductor_config.triton.num_stages = 1
                if hasattr(inductor_config.triton, "num_warps"):
                    inductor_config.triton.num_warps = 4
            print(f"  -> Applied conservative inductor settings for {gpu_info.get('architecture')}")

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 256
    if gpu_info.get("architecture") in ("ampere", "blackwell"):
        torch._dynamo.config.assume_static_by_default = True
        if hasattr(torch._dynamo.config, "automatic_dynamic_shapes"):
            torch._dynamo.config.automatic_dynamic_shapes = False


def patch_triton_shared_memory(gpu_info: dict):
    """Work around Triton blackwell shared-memory reporting bugs."""
    if gpu_info.get("architecture") != "blackwell":
        return

    blackwell_shared_mem = 228000
    try:
        from triton.runtime import driver

        original_get_device_properties = driver.active.utils.get_device_properties

        def patched_get_device_properties(device_id):
            props = original_get_device_properties(device_id)
            if props.get("max_shared_mem", 0) < 200000:
                props = dict(props)
                props["max_shared_mem"] = blackwell_shared_mem
            return props

        driver.active.utils.get_device_properties = patched_get_device_properties

        import triton.compiler.compiler as triton_compiler

        def patched_max_shared_mem(device):
            props = patched_get_device_properties(device)
            return props["max_shared_mem"]

        triton_compiler.max_shared_mem = patched_max_shared_mem
        print(
            f"  -> Patched Triton shared memory detection for Blackwell ({blackwell_shared_mem} bytes)"
        )
    except Exception as exc:
        print(f"  -> Warning: Could not patch Triton shared memory: {exc}")


def load_config(config_path):
    """Load configuration from a Python module path."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module
