import copy
from functools import lru_cache
import glob
from itertools import cycle
import math
import os
from pathlib import Path
import sys
import threading
import time

import torch
from torch import Tensor, nn
import torch.nn.utils as nn_utils

from model import GPT, next_multiple_of_n, set_flex_attention_kernel_options
from optimizers import create_optimizer
import wandb


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=True
        )  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


BOS_ID = 50256


class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents
    def __init__(self, tokens: Tensor, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens = tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # Only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (
                (tokens[:4_000_000] == BOS_ID)
                .nonzero(as_tuple=True)[0]
                .to(torch.int64)
                .cpu()
                .numpy()
            )
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (
                (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            )
        self.i = 0
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (
            (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        )
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        # If quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter == 5:
            self.get()
        n = len(self.bos_idx)
        starts = []
        ends = []

        idx = self.i
        cur_len = 0
        while cur_len <= num_tokens_local:
            if idx >= n:
                raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
            cur = self.bos_idx[idx]
            starts.append(cur)
            end = min(
                self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                cur + max_seq_len,
                cur + num_tokens_local - cur_len + 1,
            )
            ends.append(end)
            cur_len += end - cur
            idx += 1

        assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter += 1
        return starts, ends


class DataPreloader:
    # Helper for asynchronously loading next shard and indexing BOS tokens
    def __init__(self, file_iter):
        self.file_iter = file_iter
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data


class TrainingManager:
    """
    Manages training schedule, optimizers, and model state changes.
    Aligned with train_gpt.py training loop structure.
    """

    def __init__(self, model: nn.Module, config: dict, print_fn=print):
        self.model = model
        self.config = config
        self.print_fn = print_fn

        # Extract config sections
        self.training_config = config.get("training_config", {})
        self.batch_schedule_config = config.get("batch_schedule_config", {})
        self.window_schedule_config = config.get("window_schedule_config", {})
        self.embed_config = config.get("embed_config", {})
        self.optimizer_config = config.get("optimizer_config", {})
        self.data_config = config.get("data_config", {})
        self.train_seq_len = self.data_config.get(
            "train_micro_batch_tokens", self.data_config.get("train_seq_len")
        )

        # Training parameters
        self.num_iterations = self.training_config.get("num_iterations", 1845)
        self.num_scheduled_iterations = self.training_config.get(
            "num_scheduled_iterations", self.num_iterations
        )

        split_frac = self.embed_config.get("split_frac", 2 / 3)
        self.split_step = math.ceil(split_frac * self.num_scheduled_iterations)

        # Current state
        self.current_batch_size_idx = 0
        self.current_window_size_idx = 0
        self.current_window_size = self.window_schedule_config.get("schedule", [3])[0]
        self.embed_split_done = False
        self.scalar_freeze_countdown = 0

        # Optimizers (set later)
        self.adam_opt = None
        self.scalar_opt = None
        self.muon_opt = None

        # Track transitions for scalar freeze
        self.freeze_steps = self.optimizer_config.get("freeze_scalars_on_transition", 40)

    def get_batch_size(self, step: int) -> int:
        """Get batch size (grad accum steps) for current step using stepped schedule."""
        schedule_type = self.batch_schedule_config.get("schedule_type", "stepped")

        if schedule_type == "stepped":
            batch_sizes = self.batch_schedule_config.get("batch_sizes", [8, 16, 24])
            base_tokens = self.batch_schedule_config.get("base_tokens")
            transitions = self.batch_schedule_config.get("transitions", [1 / 3, 2 / 3])
            progress = step / self.num_scheduled_iterations

            for i, trans in enumerate(transitions):
                if progress < trans:
                    return self._resolve_grad_accum_steps(batch_sizes[i], base_tokens)
            return self._resolve_grad_accum_steps(batch_sizes[-1], base_tokens)
        else:
            # Legacy linear schedule
            return self._get_linear_batch_size(step)

    def _resolve_grad_accum_steps(self, batch_units: int, base_tokens: int | None) -> int:
        if base_tokens is None or not self.train_seq_len:
            return batch_units
        target_tokens = batch_units * base_tokens
        if target_tokens % self.train_seq_len != 0:
            self.print_fn(
                f"Warning: target_tokens ({target_tokens}) not divisible by train_seq_len "
                f"({self.train_seq_len}); rounding grad_accum_steps."
            )
        grad_accum = max(1, int(round(target_tokens / self.train_seq_len)))
        return grad_accum

    def _get_linear_batch_size(self, step: int) -> int:
        """Legacy linear batch size ramp."""
        progress = step / self.num_iterations
        initial = self.batch_schedule_config.get("initial_grad_accum_steps", 8)
        final = self.batch_schedule_config.get("final_grad_accum_steps", 24)
        warmup_frac = self.batch_schedule_config.get("warmup_frac", 0.5)

        if progress >= warmup_frac:
            return final
        return int(round(initial + (final - initial) * (progress / warmup_frac)))

    def get_window_size(self, step: int) -> int:
        """Get window size for current step using stepped schedule."""
        schedule = self.window_schedule_config.get("schedule", [3, 7, 11])
        final_ws = self.window_schedule_config.get("final_ws", 13)
        post_yarn = self.window_schedule_config.get("post_yarn_extension", 20)
        transitions = self.window_schedule_config.get("transitions", [1 / 3, 2 / 3])

        # Post-training extension phase
        if step >= self.num_scheduled_iterations:
            return post_yarn

        progress = step / self.num_scheduled_iterations

        # Use stepped schedule during main training
        for i, trans in enumerate(transitions):
            if progress < trans:
                return schedule[i]

        # After all transitions, use final window size
        return final_ws

    def get_window_size_blocks(self, step: int, block_size: int = 128) -> Tensor:
        """Get window size in blocks for FlexAttention."""
        ws = self.get_window_size(step)
        return torch.tensor(ws * block_size // block_size, dtype=torch.int32, device="cuda")

    def advance_schedule(self, step: int):
        """
        Update schedules and model state for the given step.
        Called at the beginning of each training step.
        """
        # Check for batch size transitions
        old_batch_idx = self.current_batch_size_idx
        batch_sizes = self.batch_schedule_config.get("batch_sizes", [8, 16, 24])
        transitions = self.batch_schedule_config.get("transitions", [1 / 3, 2 / 3])
        progress = step / self.num_scheduled_iterations

        for i, trans in enumerate(transitions):
            if progress >= trans:
                self.current_batch_size_idx = i + 1

        if self.current_batch_size_idx != old_batch_idx:
            resolved = self.get_batch_size(step)
            self.print_fn(
                f"Step {step}: Batch size transition -> {batch_sizes[self.current_batch_size_idx]} "
                f"(grad_accum_steps={resolved})"
            )
            self.scalar_freeze_countdown = self.freeze_steps

        # Check for window size transitions
        old_ws = self.current_window_size
        new_ws = self.get_window_size(step)

        if new_ws != old_ws:
            self.print_fn(f"Step {step}: Window size transition {old_ws} -> {new_ws}")
            # Call positional embedding update on window size change
            if hasattr(self.model, "pos_emb") and self.model.pos_emb is not None:
                self.model.pos_emb.apply(old_ws, new_ws)
            self.current_window_size = new_ws
            self.scalar_freeze_countdown = self.freeze_steps

        # Check for embed/lm_head split
        if not self.embed_split_done and step == self.split_step:
            if hasattr(self.model, "create_embed"):
                self.print_fn(
                    f"Step {step}: Splitting embed from lm_head (split_step={self.split_step})"
                )
                self.model.create_embed()
                self.embed_split_done = True
                # Add new embed to Adam optimizer and copy state from lm_head
                if (
                    self.adam_opt is not None
                    and hasattr(self.model, "embed")
                    and self.model.embed is not None
                ):
                    self._copy_lm_to_embed()

        # Decrement scalar freeze countdown
        if self.scalar_freeze_countdown > 0:
            self.scalar_freeze_countdown -= 1

    def set_optimizers(self, adam_opt, scalar_opt=None, muon_opt=None):
        """Set optimizer references for management."""
        self.adam_opt = adam_opt
        self.scalar_opt = scalar_opt
        self.muon_opt = muon_opt

    def _copy_lm_to_embed(self):
        """
        Copy lm_head optimizer state to embed (from train_gpt.py).
        Called at 2/3 of training when embed splits from lm_head.
        """
        if self.adam_opt is None:
            return

        # Resolve lm_head parameter from model first, then fall back to labels.
        lm_head = None
        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            lm_head = self.model.lm_head.weight
        if lm_head is None and hasattr(self.model, "named_parameters"):
            for name, p in self.model.named_parameters():
                if name.endswith("lm_head.weight"):
                    lm_head = p
                    break
        if lm_head is None:
            for group in self.adam_opt.param_groups:
                for p in group["params"]:
                    if getattr(p, "label", None) == "lm_head":
                        lm_head = p
                        break
                if lm_head is not None:
                    break

        if lm_head is None:
            self.print_fn("Warning: Could not find lm_head for state copying")
            return

        lm_head_group = None
        for group in self.adam_opt.param_groups:
            if lm_head in group["params"]:
                lm_head_group = group
                break
        if lm_head_group is None:
            self.print_fn("Warning: Found lm_head parameter, but not in Adam param groups")
            return

        embed = self.model.embed.weight
        # If resuming from a checkpoint after split, this param group may already exist.
        for group in self.adam_opt.param_groups:
            if embed in group["params"]:
                return

        # Add embed to optimizer with same config as lm_head
        lr = lm_head_group.get("lr", self.optimizer_config.get("adam", {}).get("lr", 0.008))
        betas = lm_head_group.get(
            "betas", self.optimizer_config.get("adam", {}).get("betas", (0.65, 0.95))
        )
        eps = lm_head_group.get("eps", self.optimizer_config.get("adam", {}).get("eps", 1e-8))
        weight_decay = lm_head_group.get(
            "weight_decay", self.optimizer_config.get("adam", {}).get("weight_decay", 0.005)
        )
        self.adam_opt.add_param_group(
            {
                "params": [embed],
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "initial_lr": lr,
            }
        )

        # Copy optimizer state from lm_head to embed
        if lm_head in self.adam_opt.state:
            lm_head_state = self.adam_opt.state[lm_head]
            embed_state = self.adam_opt.state[embed] = {}
            embed_state["step"] = lm_head_state.get("step", 0)
            if "exp_avg" in lm_head_state:
                embed_state["exp_avg"] = lm_head_state["exp_avg"].clone()
            if "exp_avg_sq" in lm_head_state:
                embed_state["exp_avg_sq"] = lm_head_state["exp_avg_sq"].clone()
            self.print_fn(
                f"Copied optimizer state from lm_head to embed (step={embed_state['step']})"
            )

    def step_optimizers(self, step: int):
        """Step optimizers according to schedule."""
        # Muon always steps
        if self.muon_opt is not None:
            self.muon_opt.step()
            self.muon_opt.zero_grad(set_to_none=True)

        if self.adam_opt is not None:
            self.adam_opt.step()
            self.adam_opt.zero_grad(set_to_none=True)

        # Scalar optimizer steps if not frozen
        if self.scalar_opt is not None:
            if self.scalar_freeze_countdown > 0:
                self.scalar_opt.zero_grad(set_to_none=True)
            else:
                self.scalar_opt.step()
                self.scalar_opt.zero_grad(set_to_none=True)


# -----------------------------------------------------------------------------
# Simple data loader for single GPU


def single_gpu_data_generator(
    filename_pattern: str, num_tokens: int, max_seq_len: int, align_to_bos: bool = True
):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No data shards matched pattern: {filename_pattern}")
    file_iter = cycle(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, quickload=True)
        preloader = DataPreloader(file_iter)
        preloader.start()
    else:
        pos = 0
    while True:
        if align_to_bos:
            try:
                start_idxs, end_idxs = finder.next_batch(num_tokens, max_seq_len)
            except StopIteration:
                tokens, finder = preloader.get()
                preloader.start()
                continue
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            buf = tokens[pos : pos + num_tokens + 1]
            pos += num_tokens
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        yield inputs, targets


# batch size schedule: gradually increase gradient accumulation steps (legacy fallback)
def get_grad_accum_steps(step: int, training_config=None, batch_schedule_config=None):
    if training_config is None:
        raise RuntimeError(
            "get_grad_accum_steps requires an explicit training_config when called outside run_training."
        )
    if batch_schedule_config is None:
        raise RuntimeError(
            "get_grad_accum_steps requires an explicit batch_schedule_config when called outside run_training."
        )

    x = step / training_config["num_iterations"]  # progress in training
    assert 0 <= x <= 1

    initial_steps = batch_schedule_config["initial_grad_accum_steps"]
    final_steps = batch_schedule_config["final_grad_accum_steps"]
    warmup_frac = batch_schedule_config["warmup_frac"]
    schedule_type = batch_schedule_config["schedule_type"]

    if x >= warmup_frac:
        # After warmup, use final batch size
        return final_steps

    # During warmup, interpolate based on schedule type
    warmup_progress = x / warmup_frac

    if schedule_type == "linear":
        # Linear interpolation
        steps = initial_steps + (final_steps - initial_steps) * warmup_progress
    elif schedule_type == "cosine":
        # Cosine schedule (smooth transition)
        steps = (
            initial_steps
            + (final_steps - initial_steps) * (1 - math.cos(warmup_progress * math.pi)) / 2
        )
    elif schedule_type == "exponential":
        # Exponential growth
        log_ratio = math.log(final_steps / initial_steps)
        steps = initial_steps * math.exp(log_ratio * warmup_progress)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    # Round to nearest integer
    return int(round(steps))


# -----------------------------------------------------------------------------
# Main training code


def run_training(config, args, code: str, detected_gpu_info: dict, run_id):
    model_config = config.model_config
    model_type = model_config.get("model_type", "gpt")
    data_config = config.data_config
    training_config = config.training_config
    optimizer_config = config.optimizer_config
    batch_schedule_config = config.batch_schedule_config
    warmup_config = config.warmup_config
    logging_config = config.logging_config
    compilation_config = getattr(config, "compilation_config", {"compile_model": False})
    lr_scheduler_config = getattr(config, "lr_scheduler_config", None)
    train_micro_batch_tokens = data_config.get(
        "train_micro_batch_tokens", data_config["train_seq_len"]
    )

    # Match train_gpt.py: allow DATA_PATH to prefix data file globs.
    data_path = os.environ.get("DATA_PATH", ".")
    data_config["train_files"] = os.path.join(data_path, data_config["train_files"])
    data_config["val_files"] = os.path.join(data_path, data_config["val_files"])

    attention_config = getattr(config, "attention_config", None)
    attention_pattern_config = getattr(config, "attention_pattern_config", None)
    lambda_config = getattr(config, "lambda_config", None)
    lr_multipliers = optimizer_config.get("lr_multipliers", {})
    wd_multipliers = optimizer_config.get("wd_multipliers", {})

    # New config sections for train_gpt.py alignment
    gating_config = getattr(config, "gating_config", None)
    skip_config = getattr(config, "skip_config", None)
    rope_config = getattr(config, "rope_config", None)
    embed_config = getattr(config, "embed_config", None)
    window_schedule_config = getattr(config, "window_schedule_config", None)

    # Extract gradient clipping configuration from training config
    if attention_config is None:
        attention_config = {}

    # Apply command-line overrides for activation function
    if args.activation is not None:
        model_config["activation"] = args.activation
        print(f"Override: activation = {args.activation}")

    if args.ffn_dim is not None:
        if args.ffn_dim < 1:
            raise ValueError("--ffn_dim must be >= 1")
        model_config["ffn_dim"] = args.ffn_dim
        print(f"Override: ffn_dim = {args.ffn_dim}")

    if getattr(args, "mlp_type", None) is not None:
        model_config["mlp_type"] = args.mlp_type
        print(f"Override: mlp_type = {args.mlp_type}")

    if args.checkpoint_every is not None:
        if args.checkpoint_every < 0:
            raise ValueError("--checkpoint_every must be >= 0")
        training_config["checkpoint_every"] = args.checkpoint_every
        print(f"Override: checkpoint_every = {args.checkpoint_every}")

    # Configure model backend features tied to GPU architecture.
    set_flex_attention_kernel_options(detected_gpu_info.get("architecture"))

    # -----------------------------------------------------------------------------
    #    Construct model and optimizer
    # -----------------------------------------------------------------------------
    max_seq_len = max(
        data_config["train_seq_len"], data_config["val_seq_len"], train_micro_batch_tokens
    )

    # Create model with new config parameters
    model: nn.Module = GPT(
        model_config=model_config,
        attention_config=attention_config,
        lambda_config=lambda_config,
        lr_multipliers=lr_multipliers,
        max_seq_len=max_seq_len,
        attention_pattern_config=attention_pattern_config,
        gating_config=gating_config,
        skip_config=skip_config,
        rope_config=rope_config,
        embed_config=embed_config,
        wd_multipliers=wd_multipliers,
    ).cuda()
    # Convert all weights to bfloat16 (from train_gpt.py)
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()

    # Create TrainingManager
    training_manager_config = {
        "training_config": training_config,
        "batch_schedule_config": batch_schedule_config,
        "window_schedule_config": window_schedule_config or {},
        "embed_config": embed_config or {},
        "data_config": data_config,
        "optimizer_config": optimizer_config,
    }
    training_manager = TrainingManager(model, training_manager_config, print_fn=print)

    # Setup optimizers based on configuration
    optimizer_state = create_optimizer(model, optimizer_config, print_fn=print)
    optimizer1 = optimizer_state["adam_opt"]
    optimizer2 = optimizer_state["matrix_opt"]
    scalar_opt = optimizer_state["scalar_opt"]
    matrix_optimizer_type = optimizer_state["matrix_optimizer_type"]
    optimizers = optimizer_state["optimizers"]
    hidden_matrix_params = optimizer_state["hidden_matrix_params"]
    embed_params = optimizer_state["embed_params"]
    scalar_params = optimizer_state["scalar_params"]
    x0_lambda_params = optimizer_state["x0_lambda_params"]
    head_params = optimizer_state["head_params"]

    # Connect optimizers to TrainingManager
    training_manager.set_optimizers(
        adam_opt=optimizer1, scalar_opt=scalar_opt, muon_opt=optimizer2
    )

    # Print gradient clipping configuration
    grad_clip_norm = training_config.get("grad_clip_norm", None)  # None means no clipping

    if grad_clip_norm is not None:
        print(f"Gradient clipping enabled: max_norm = {grad_clip_norm}")
    else:
        print("Gradient clipping: disabled")

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Initialize wandb after model creation
    if logging_config["use_wandb"]:
        # Generate descriptive run name with model type and key parameters
        model_size_info = ""
        if model_type == "gpt":
            model_size_info = f"{model_config['num_layers']}L_{model_config['model_dim']}D_{model_config['num_heads']}H"
        elif model_type in ["mamba", "mamba2"]:
            model_size_info = f"{model_config['num_layers']}L_{model_config['hidden_size']}D"
        elif model_type in ["gla", "retnet", "linear_attn", "rwkv6", "hgrn"]:
            model_size_info = f"{model_config['num_hidden_layers']}L_{model_config['hidden_size']}D_{model_config['num_heads']}H"

        # Get sequence length and batch size info
        seq_len_info = f"seq{data_config['train_seq_len']}"
        micro_batch_tokens = data_config.get(
            "train_micro_batch_tokens", data_config["train_seq_len"]
        )
        grad_accum_initial = batch_schedule_config.get("initial_grad_accum_steps", 1)
        grad_accum_final = batch_schedule_config.get("final_grad_accum_steps", grad_accum_initial)
        train_seq_len = max(1, data_config["train_seq_len"])
        micro_batch_size = data_config.get(
            "batch_size_multiple", micro_batch_tokens // train_seq_len
        )
        effective_batch_min_tokens = micro_batch_tokens * grad_accum_initial
        effective_batch_max_tokens = micro_batch_tokens * grad_accum_final
        batch_info = (
            f"bs{micro_batch_size}-grad{grad_accum_initial}-{grad_accum_final}"
            f"_eff{effective_batch_min_tokens}-{effective_batch_max_tokens}"
        )

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_info = (
            f"{total_params // 1_000_000}M"
            if total_params >= 1_000_000
            else f"{total_params // 1_000}K"
        )

        # Get activation function info
        activation_name = model_config.get("activation", "relu_squared")
        # Shorten common activation names for the run name
        activation_short = {
            "relu_squared": "relu2",
            "gelu": "gelu",
            "swish": "swish",
            "silu": "silu",
            "geglu": "geglu",
            "swiglu": "swiglu",
        }.get(activation_name, activation_name)
        activation_info = f"_{activation_short}"

        # Generate automatic run name if not specified
        if logging_config["wandb_run_name"] is None:
            wandb_run_name = f"{model_type}_{model_size_info}_{param_info}_{seq_len_info}{activation_info}_{batch_info}"
        else:
            wandb_run_name = logging_config["wandb_run_name"]

        # Use a single project for all runs
        wandb_project = "modded-nanogpt-comparison"  # Override individual project names

        # Create tags for easy filtering
        tags = [
            model_type,  # Model architecture
            f"seq_{data_config['train_seq_len']}",  # Sequence length
            f"params_{param_info}",  # Total parameters in human-readable format
            f"act_{activation_short}",  # Activation function
        ]

        # Add optimizer info to tags
        if optimizer2 is not None:
            tags.append("muon_adam")
        else:
            tags.append("adam_only")

        wandb_config = {
            **model_config,
            **data_config,
            **training_config,
            **optimizer_config,
            **batch_schedule_config,
            "config_file": args.config,
            "batch_size_multiple": micro_batch_size,
            "micro_batch_size": micro_batch_size,
            "grad_accum_initial": grad_accum_initial,
            "grad_accum_final": grad_accum_final,
            "effective_batch_min_tokens": effective_batch_min_tokens,
            "effective_batch_max_tokens": effective_batch_max_tokens,
            "total_params": total_params,
            "early_stop_steps": args.early_stop_steps,
            "activation": activation_name,  # Explicitly track activation function
        }

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config,
            id=str(run_id),
            tags=tags,
        )

    # logging + run output
    output_dir = os.path.join("logs", str(run_id))
    os.makedirs(output_dir, exist_ok=True)
    logfile = os.path.join(output_dir, "train.log")
    print(f"Logging to: {logfile}")

    def print_log(s, console=True):
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

    print_log("=" * 100)
    # log information about the hardware/software environment this is running on
    print_log(f"Running Python {sys.version}")
    print_log(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print_log(f"Configuration: {args.config}")
    print_log(f"Model type: {model_type}")
    if args.early_stop_steps is not None:
        print_log(f"Early stopping enabled: will stop after {args.early_stop_steps} steps")

    def nvidia_smi():
        import subprocess

        return subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout

    print_log(nvidia_smi())
    print_log("=" * 100)

    # learning rate scheduler: supports multiple scheduler types
    def get_lr(step: int):
        num_iterations = training_config["num_iterations"]
        num_scheduled_iterations = training_config.get("num_scheduled_iterations", num_iterations)

        # Extension phase
        if step > num_scheduled_iterations:
            return 0.1

        # Use lr_scheduler_config if available, otherwise fall back to legacy config
        if lr_scheduler_config is not None:
            scheduler_type = lr_scheduler_config.get("scheduler_type", "linear")
            warmup_steps = lr_scheduler_config.get("warmup_steps", 0)
            use_linear_warmup = lr_scheduler_config.get("use_linear_warmup", True)

            # Determine cooldown steps
            cooldown_steps = lr_scheduler_config.get("cooldown_steps")
            if cooldown_steps is None:
                cooldown_frac = training_config.get("cooldown_frac", 0.0)
                cooldown_steps = int(num_iterations * cooldown_frac)

            # Main training steps (excluding warmup and cooldown)
            main_steps = num_iterations - warmup_steps - cooldown_steps

            # Initialize lr_multiplier
            lr_multiplier = 1.0

            # Handle warmup phase
            if step < warmup_steps and warmup_steps > 0:
                if use_linear_warmup:
                    # Linear warmup
                    lr_multiplier = step / warmup_steps
                else:
                    # Cosine warmup
                    lr_multiplier = 0.5 * (1 + math.cos(math.pi * (1 - step / warmup_steps)))
                return lr_multiplier

            # Adjust step for main phase (after warmup)
            adjusted_step = step - warmup_steps

            # Handle main training phase
            if adjusted_step < main_steps:
                progress = adjusted_step / main_steps if main_steps > 0 else 1.0

                if scheduler_type == "cosine":
                    # Cosine annealing
                    min_lr_ratio = lr_scheduler_config.get("cosine_min_lr_ratio", 0.1)
                    lr_multiplier = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                        1 + math.cos(math.pi * progress)
                    )

                elif scheduler_type == "cosine_with_restarts":
                    # Cosine annealing with warm restarts
                    num_cycles = lr_scheduler_config.get("cosine_restart_cycles", 1)
                    decay_factor = lr_scheduler_config.get("cosine_restart_decay", 1.0)
                    min_lr_ratio = lr_scheduler_config.get("cosine_min_lr_ratio", 0.1)

                    # Determine which cycle we're in
                    cycle_length = main_steps / num_cycles
                    cycle = int(adjusted_step / cycle_length)
                    cycle_progress = (adjusted_step % cycle_length) / cycle_length

                    # Apply decay to max LR for this cycle
                    cycle_max = decay_factor**cycle
                    cycle_min = min_lr_ratio * cycle_max

                    # Cosine annealing within the cycle
                    lr_multiplier = cycle_min + (cycle_max - cycle_min) * 0.5 * (
                        1 + math.cos(math.pi * cycle_progress)
                    )

                elif scheduler_type == "exponential":
                    # Exponential decay
                    gamma = lr_scheduler_config.get("exponential_gamma", 0.95)
                    lr_multiplier = gamma**adjusted_step

                elif scheduler_type == "linear":
                    # Linear decay (stable phase)
                    lr_multiplier = 1.0

                else:
                    # Default to no decay
                    lr_multiplier = 1.0

            # Handle cooldown phase
            elif cooldown_steps > 0:
                cooldown_progress = (adjusted_step - main_steps) / cooldown_steps
                final_lr_ratio = lr_scheduler_config.get("final_lr_ratio", 0.1)

                if scheduler_type == "cosine" or scheduler_type == "cosine_with_restarts":
                    # Continue cosine decay in cooldown
                    min_lr_ratio = lr_scheduler_config.get("cosine_min_lr_ratio", 0.1)
                    # Get the LR at the end of main phase
                    if scheduler_type == "cosine":
                        main_end_lr = min_lr_ratio
                    else:
                        # For restarts, use the final cycle's minimum
                        num_cycles = lr_scheduler_config.get("cosine_restart_cycles", 1)
                        decay_factor = lr_scheduler_config.get("cosine_restart_decay", 1.0)
                        main_end_lr = min_lr_ratio * (decay_factor ** (num_cycles - 1))

                    # Linear decay from main_end_lr to final_lr_ratio during cooldown
                    lr_multiplier = (
                        main_end_lr * (1 - cooldown_progress) + final_lr_ratio * cooldown_progress
                    )

                else:
                    # Linear cooldown for other schedulers
                    lr_multiplier = (
                        1 - cooldown_progress
                    ) * 1.0 + cooldown_progress * final_lr_ratio
            else:
                # Beyond training steps
                lr_multiplier = lr_scheduler_config.get("final_lr_ratio", 0.1)

            return lr_multiplier

        else:
            # Legacy behavior: use warmup_config
            # Match train_gpt.py intent: increase LR when batch size increases
            lr_max = 1.0
            x = step / num_scheduled_iterations  # progress in training

            # Batch size schedule scaling (config-driven, not hardcoded thirds).
            if batch_schedule_config is not None:
                batch_sizes = batch_schedule_config.get("batch_sizes")
                transitions = batch_schedule_config.get("transitions")
                if batch_sizes and len(batch_sizes) > 0:
                    current_idx = 0
                    if transitions:
                        for i, trans in enumerate(transitions):
                            if x >= trans:
                                current_idx = min(i + 1, len(batch_sizes) - 1)
                    base_batch = batch_sizes[0]
                    current_batch = batch_sizes[current_idx]
                    if base_batch > 0 and current_batch > base_batch:
                        scale = current_batch / base_batch
                        # Keep LR increases softer for tiny runs to avoid sharp loss jitter.
                        exponent = 0.45 if scale < 2.0 else 0.4
                        lr_max = scale**exponent
            else:
                # Fallback to previous default when no batch schedule exists.
                if x > 1 / 3:
                    lr_max = 1.52  # (16/8)**0.6
                if x > 2 / 3:
                    lr_max = 1.73  # (24/8)**0.5

            lr_warmup_steps = warmup_config.get("lr_warmup_steps", 0)

            # Learning rate warmup phase
            if lr_warmup_steps > 0 and step < lr_warmup_steps:
                # Linear warmup from 0 to 1
                return step / lr_warmup_steps
            # Stable phase
            elif x < 1 - training_config["cooldown_frac"]:
                return lr_max
            # Cooldown phase
            else:
                w = (1 - x) / training_config["cooldown_frac"]
                return lr_max * w + (1 - w) * 0.1

    # attention window size schedule (only for GPT)
    if model_type == "gpt":
        block_size = attention_config.get("block_size", 128)

        @lru_cache(maxsize=32)
        def get_window_size_blocks_helper(window_size: int):
            return torch.tensor(window_size, dtype=torch.int32, pin_memory=True).cuda(
                non_blocking=True
            )

        def get_window_size_blocks(step: int):
            # Use TrainingManager's stepped schedule if window_schedule_config is available
            if window_schedule_config is not None:
                ws = training_manager.get_window_size(step)
                return get_window_size_blocks_helper(ws)
            else:
                # Legacy: linearly increase the block-wise sliding window size over training
                x = step / training_config["num_iterations"]
                assert 0 <= x <= 1
                max_window_size = attention_config["max_window_size"]
                window_size = next_multiple_of_n(max_window_size * x, n=block_size)
                return get_window_size_blocks_helper(window_size // block_size)

        def get_grad_accum_steps_managed(step: int):
            # Use TrainingManager's stepped schedule if available
            if window_schedule_config is not None:
                return training_manager.get_batch_size(step)
            else:
                return get_grad_accum_steps(
                    step,
                    training_config=training_config,
                    batch_schedule_config=batch_schedule_config,
                )

    else:
        # SSM models don't use window size blocks
        def get_window_size_blocks(step: int):
            return None

        def get_grad_accum_steps_managed(step: int):
            return get_grad_accum_steps(
                step,
                training_config=training_config,
                batch_schedule_config=batch_schedule_config,
            )

    ########################################
    #            Warmup kernels            #
    ########################################

    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    warmup_steps = warmup_config["warmup_steps"]
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
    )
    # Use smaller sequence length for warmup if specified
    warmup_seq_len = warmup_config.get("warmup_seq_len") or data_config["train_seq_len"]
    print_log(f"Running warmup with sequence length: {warmup_seq_len}")
    for _ in range(warmup_steps):
        inputs = targets = torch.randint(
            0, model_config["vocab_size"], size=(warmup_seq_len,), device="cuda"
        )
        if model_type == "gpt":
            loss = model(inputs.to(torch.int32), targets, get_window_size_blocks(0))
        else:
            # SSM models expect different input format
            outputs = model(
                input_ids=inputs[None, :].to(torch.int64), labels=targets[None, :].to(torch.int64)
            )
            loss = outputs.loss
        loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state
    print_log("Warmup complete, model state reset")

    # Compile model after warmup based on configuration
    compile_model = compilation_config.get("compile_model", False)
    relaxed_compile = os.environ.get("RAMENGPT_RELAXED_COMPILE", "").lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    compile_mode = compilation_config.get("compile_mode", "default")

    if not compile_model:
        print_log("Model compilation disabled by configuration")
    elif model_type in ["mamba", "mamba2"]:
        print_log("Skipping compilation for Mamba models (known to cause recompilation overhead)")
    else:
        print_log("Compiling model...")

        try:
            # Check if block size is compatible with compiled FlexAttention
            if model_type == "gpt" and attention_config.get("block_size", 128) == 64:
                print_log(
                    "Warning: Block size 64 may not be compatible with compiled FlexAttention"
                )
                print_log(
                    "If compilation fails, consider using block_size=128 in attention_config for compatibility"
                )

            # For GPUs with shared memory constraints, use default mode
            # max-autotune is too slow for first run; default mode works well after our fixes
            if detected_gpu_info.get("architecture") in ("ampere", "blackwell") and not relaxed_compile:
                compile_mode = "default"
                print_log(
                    f"  -> Using compile mode '{compile_mode}' for {detected_gpu_info.get('architecture')} architecture"
                )
            else:
                compile_mode = compile_mode
                if not relaxed_compile:
                    print_log(f"  -> Using compile mode '{compile_mode}' for non-conservative path")

            # Try compilation with appropriate settings
            model: nn.Module = torch.compile(
                model, backend="inductor", dynamic=False, mode=compile_mode
            )
            print_log("Model compilation complete")
        except Exception as e:
            error_str = str(e)
            print_log(f"Warning: Model compilation failed with error: {e}")
            if "Q and KV block size must be divisible by BLOCK_M and BLOCK_N" in error_str:
                print_log("This is due to FlexAttention block size requirements in compiled mode")
                print_log("Consider using block_size=128 in attention_config for compatibility")
            elif "No valid triton configs" in error_str or "out of resource" in error_str.lower():
                print_log("This is due to Triton shared memory limits on your GPU")
                print_log("The model will run in eager mode (slower but functional)")
            print_log(
                "Continuing without compilation - performance will be slower but training will work"
            )
            # Model remains uncompiled

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = single_gpu_data_generator(
        data_config["train_files"],
        train_micro_batch_tokens,
        data_config["train_seq_len"],
        align_to_bos=True,
    )
    training_time_s = 0
    train_tokens_processed = (
        0  # Track train-only tokens processed for clear throughput comparisons.
    )
    total_tokens_processed = 0  # Track total tokens processed including validation.
    # Zero-align training time and tokens to the first completed train step for clean cross-run x-axes.
    first_train_step_time_s = None
    first_train_tokens_processed = None
    prev_train_step_time_s = None
    # begin training
    train_steps = training_config["num_iterations"]
    # Limit iterations to early_stop_steps if specified
    max_steps = (
        min(train_steps + 1, args.early_stop_steps + 1)
        if args.early_stop_steps is not None
        else train_steps + 1
    )

    # Log learning rate scheduler information
    if lr_scheduler_config is not None:
        scheduler_type = lr_scheduler_config.get("scheduler_type", "linear")
        warmup_steps = lr_scheduler_config.get("warmup_steps", 0)
        cooldown_steps = lr_scheduler_config.get("cooldown_steps")
        if cooldown_steps is None:
            cooldown_frac = training_config.get("cooldown_frac", 0.0)
            cooldown_steps = int(train_steps * cooldown_frac)
        main_steps = train_steps - warmup_steps - cooldown_steps

        print_log(f"Learning rate scheduler: {scheduler_type}")

        if scheduler_type == "cosine":
            min_lr_ratio = lr_scheduler_config.get("cosine_min_lr_ratio", 0.1)
            print_log(f"  - Cosine annealing with min LR ratio: {min_lr_ratio}")
        elif scheduler_type == "cosine_with_restarts":
            num_cycles = lr_scheduler_config.get("cosine_restart_cycles", 1)
            decay_factor = lr_scheduler_config.get("cosine_restart_decay", 1.0)
            min_lr_ratio = lr_scheduler_config.get("cosine_min_lr_ratio", 0.1)
            print_log(f"  - Cosine with {num_cycles} restart cycle(s)")
            print_log(f"  - Restart decay factor: {decay_factor}, min LR ratio: {min_lr_ratio}")
        elif scheduler_type == "exponential":
            gamma = lr_scheduler_config.get("exponential_gamma", 0.95)
            print_log(f"  - Exponential decay with gamma: {gamma}")

        if warmup_steps > 0:
            use_linear_warmup = lr_scheduler_config.get("use_linear_warmup", True)
            warmup_type = "linear" if use_linear_warmup else "cosine"
            print_log(f"  - Warmup: {warmup_steps} steps ({warmup_type})")

        print_log(f"  - Main phase: {main_steps} steps")

        if cooldown_steps > 0:
            final_lr_ratio = lr_scheduler_config.get("final_lr_ratio", 0.1)
            print_log(f"  - Cooldown: {cooldown_steps} steps (final LR ratio: {final_lr_ratio})")

        # Log phase boundaries
        if warmup_steps > 0 or cooldown_steps > 0:
            phases = []
            if warmup_steps > 0:
                phases.append(f"Warmup: 0-{warmup_steps-1}")
            phases.append(f"Main: {warmup_steps}-{warmup_steps+main_steps-1}")
            if cooldown_steps > 0:
                phases.append(f"Cooldown: {warmup_steps+main_steps}-{train_steps-1}")
            print_log(f"  - Phase boundaries: {', '.join(phases)}")
    else:
        # Legacy behavior
        lr_warmup_steps = warmup_config.get("lr_warmup_steps", 0)
        if lr_warmup_steps > 0:
            print_log(
                f"Learning rate warmup: enabled for {lr_warmup_steps} steps (linearly increasing from 0 to 1)"
            )
            cooldown_start = int(train_steps * (1 - training_config["cooldown_frac"]))
            print_log(f"  - Warmup phase: steps 0-{lr_warmup_steps-1}")
            print_log(f"  - Stable phase: steps {lr_warmup_steps}-{cooldown_start-1}")
            print_log(f"  - Cooldown phase: steps {cooldown_start}-{train_steps}")
        else:
            print_log("Learning rate warmup: disabled")

    # Prepare checkpoint directory.
    # Periodic checkpoints write under output_dir/checkpoints/<timestamp>.
    # Final-only checkpoints keep legacy checkpoint_root/date/run_id behavior.
    ckpt_every = int(training_config.get("checkpoint_every", 0) or 0)
    ckpt_dir = None
    if ckpt_every > 0:
        ckpt_timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(output_dir, "checkpoints", ckpt_timestamp)
        os.makedirs(ckpt_dir, exist_ok=True)
        print_log(f"Periodic checkpointing enabled: every {ckpt_every} step(s) -> {ckpt_dir}")
    elif training_config.get("save_checkpoint", False):
        ckpt_root = training_config.get("checkpoint_root", "checkpoints")
        date_str = time.strftime("%Y-%m-%d")
        ckpt_dir = os.path.join(ckpt_root, date_str, str(run_id))
        os.makedirs(ckpt_dir, exist_ok=True)

    # Start timing right before the training/validation loop.
    # This avoids counting compile/setup work in train_time_s.
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(max_steps):
        # Advance training schedules (batch size, window size, embed split)
        training_manager.advance_schedule(step)

        # Check for early stopping
        early_stop = args.early_stop_steps is not None and step >= args.early_stop_steps
        last_step = (step == train_steps) or early_stop

        # --------------- VALIDATION SECTION -----------------
        if last_step or (
            training_config["val_loss_every"] > 0
            and (step == 0 or step % training_config["val_loss_every"] == 0)
        ):
            # stop the clock
            torch.cuda.synchronize()
            training_time_s += time.perf_counter() - t0
            model.eval()
            # For validation, we don't need gradient accumulation
            val_batch_size = data_config["val_seq_len"]
            assert data_config["val_tokens"] % val_batch_size == 0
            val_steps = data_config["val_tokens"] // val_batch_size
            val_loader = single_gpu_data_generator(
                data_config["val_files"],
                data_config["val_seq_len"],
                data_config["val_seq_len"],
                align_to_bos=False,
            )
            val_loss = 0
            val_tokens_this_step = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    if model_type == "gpt":
                        val_loss += model(inputs, targets, get_window_size_blocks(step))
                    else:
                        outputs = model(
                            input_ids=inputs[None, :].to(torch.int64),
                            labels=targets[None, :].to(torch.int64),
                        )
                        val_loss += outputs.loss.item()
                    val_tokens_this_step += data_config["val_seq_len"]
            val_loss /= val_steps
            total_tokens_processed += val_tokens_this_step
            del val_loader

            print_log(
                f"step:{step}/{train_steps} val_loss:{val_loss:.4f} grad_accum:{get_grad_accum_steps_managed(step)} train_time:{training_time_s:.2f}s step_avg:{training_time_s/max(step, 1):.2f}s",
                console=True,
            )

            # Calculate tokens per second based on total training time
            tokens_per_sec = train_tokens_processed / max(training_time_s, 1e-6)

            # Log to wandb
            if logging_config["use_wandb"]:
                if first_train_step_time_s is None:
                    train_time_s_zero = 0.0
                    train_tokens_processed_for_plot = 0
                else:
                    train_time_s_zero = max(0.0, training_time_s - first_train_step_time_s)
                    train_tokens_processed_for_plot = max(
                        0,
                        train_tokens_processed - (first_train_tokens_processed or 0),
                    )
                log_dict = {
                    "val_loss": val_loss,
                    "train_time_s": train_time_s_zero,
                    "train_time_s_abs": training_time_s,
                    "step_avg_s": training_time_s / max(step, 1),
                    "learning_rate": optimizers[0].param_groups[0]["lr"],
                    "muon_lr": optimizer2.param_groups[0]["lr"] if optimizer2 is not None else None,
                    "muon_momentum": (
                        optimizer2.param_groups[0]["momentum"] if optimizer2 is not None else None
                    ),
                    "window_size_blocks": (
                        get_window_size_blocks(step).item() if model_type == "gpt" else None
                    ),
                    "window_size": (
                        training_manager.get_window_size(step) if model_type == "gpt" else None
                    ),
                    "grad_accum_steps": get_grad_accum_steps_managed(step),
                    "effective_batch_size": get_grad_accum_steps_managed(step)
                    * train_micro_batch_tokens,
                    "tokens_processed": train_tokens_processed_for_plot,
                    "total_tokens_processed": total_tokens_processed,
                    "train_tokens_processed_abs": train_tokens_processed,
                    "tokens_per_sec": tokens_per_sec,
                    "val_tokens_this_step": val_tokens_this_step,
                }
                log_dict["train_time_s_zero"] = train_time_s_zero
                if first_train_tokens_processed is not None:
                    log_dict["tokens_processed_zero"] = train_tokens_processed_for_plot

                wandb.log(log_dict, step=step)

            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            # Log early stopping message if applicable
            if args.early_stop_steps is not None and step >= args.early_stop_steps:
                print_log(
                    f"Early stopping at step {step} (requested: {args.early_stop_steps})",
                    console=True,
                )
            if training_config["save_checkpoint"]:
                log = dict(
                    step=step,
                    code=code,
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers],
                )
                save_path = os.path.join(ckpt_dir or output_dir, f"state_step{step:06d}.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(log, save_path)
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        # Gradient accumulation to match effective batch size
        grad_accum_steps = get_grad_accum_steps_managed(step)
        train_loss_accum = 0.0
        train_tokens_this_step = 0

        for micro_step in range(grad_accum_steps):
            inputs, targets = next(train_loader)
            if model_type == "gpt":
                loss = model(inputs, targets, get_window_size_blocks(step))
                # Match train_gpt.py: loss is summed over tokens, scale only by grad_accum_steps
                # so the gradient corresponds to the total token loss per iteration.
                train_loss_accum += loss.detach()
                loss = loss / grad_accum_steps
            else:
                outputs = model(
                    input_ids=inputs[None, :].to(torch.int64),
                    labels=targets[None, :].to(torch.int64),
                )
                train_loss_accum += outputs.loss.detach()
                loss = outputs.loss / grad_accum_steps
            loss.backward()
            train_tokens_this_step += train_micro_batch_tokens
        train_tokens_processed += train_tokens_this_step
        total_tokens_processed += train_tokens_this_step

        # Calculate gradient norms before clipping (for logging)
        grad_norm_before = None
        if logging_config["use_wandb"] or grad_clip_norm is not None:
            grad_norm_dict = {}

            with torch.no_grad():
                for param_group_name, params in [
                    ("hidden_matrix", hidden_matrix_params),
                    ("embed", embed_params),
                    ("scalar", scalar_params),
                    ("head", head_params),
                ]:
                    total_norm = 0.0
                    for p in params:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm_dict[f"grad_norm_{param_group_name}"] = total_norm**0.5
                # Calculate total gradient norm for clipping
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm_before = total_norm**0.5
                grad_norm_dict["grad_norm_total"] = grad_norm_before

        # Apply gradient clipping if configured
        grad_norm_after = None
        if grad_clip_norm is not None:
            # Clip gradients by L2 norm
            grad_norm_after = nn_utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if logging_config["use_wandb"]:
                grad_norm_dict["grad_norm_clipped"] = grad_norm_after
                grad_norm_dict["grad_clip_ratio"] = (
                    min(1.0, grad_clip_norm / grad_norm_before) if grad_norm_before > 0 else 1.0
                )

        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)

        if optimizer2 is not None:
            # Apply momentum warmup and cooldown for matrix optimizer (NorMuon or AROSinkhorn)
            # Read from the active optimizer's config dict
            matrix_optimizer_type = matrix_optimizer_type or optimizer_config.get(
                "matrix_optimizer", "muon"
            )
            default_momentum_cfg = {
                "momentum": 0.95,
                "momentum_min": 0.85,
                "momentum_warmup_frac": 0.10,
                "momentum_cooldown_frac": 0.10,
            }
            matrix_cfg = optimizer_config.get(matrix_optimizer_type, default_momentum_cfg)
            momentum_max = matrix_cfg.get("momentum", 0.95)
            momentum_min = matrix_cfg.get("momentum_min", 0.85)
            num_iterations = training_config["num_iterations"]

            # Compute warmup and cooldown steps from fractions
            warmup_steps = int(matrix_cfg.get("momentum_warmup_frac", 0.15) * num_iterations)
            cooldown_steps = int(matrix_cfg.get("momentum_cooldown_frac", 0.025) * num_iterations)
            cooldown_start = num_iterations - cooldown_steps

            # Compute momentum with warmup and cooldown
            if step < warmup_steps:
                # Warmup phase: linearly increase from min to max
                frac = step / warmup_steps if warmup_steps > 0 else 1.0
                momentum = momentum_min + frac * (momentum_max - momentum_min)
            elif step > cooldown_start:
                # Cooldown phase: linearly decrease from max to min
                frac = (step - cooldown_start) / cooldown_steps if cooldown_steps > 0 else 1.0
                momentum = momentum_max - frac * (momentum_max - momentum_min)
            else:
                # Stable phase: use max momentum
                momentum = momentum_max

            for group in optimizer2.param_groups:
                group["momentum"] = momentum

        # Step optimizers using TrainingManager (Adam + Muon + optional scalar optimizer)
        training_manager.step_optimizers(step)

        # Periodic checkpointing
        if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
            log = dict(
                step=step + 1,
                code=code,
                model=model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
            )
            save_path = os.path.join(ckpt_dir or output_dir, f"state_step{step + 1:06d}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(log, save_path)

        # logging
        approx_training_time_s = training_time_s + (time.perf_counter() - t0)
        effective_batch_size = grad_accum_steps * train_micro_batch_tokens
        train_loss_per_token = train_loss_accum.item() / max(effective_batch_size, 1)
        lr_mult = get_lr(step)

        # Determine if we should show lr_mult in logging
        if lr_scheduler_config is not None:
            warmup_steps = lr_scheduler_config.get("warmup_steps", 0)
            show_lr_info = warmup_steps > 0 and step < warmup_steps
        else:
            warmup_steps = warmup_config.get("lr_warmup_steps", 0)
            show_lr_info = warmup_steps > 0 and step < warmup_steps

        lr_info = f" lr_mult:{lr_mult:.3f}" if show_lr_info else ""
        print_log(
            f"step:{step+1}/{train_steps} train_loss:{train_loss_per_token:.4f}{lr_info} grad_accum:{grad_accum_steps} batch_size:{effective_batch_size} train_time:{approx_training_time_s:.2f}s step_avg:{approx_training_time_s/(step + 1):.2f}s",
            console=True,
        )

        # Calculate tokens per second for training steps too
        approx_training_time_s = training_time_s + (time.perf_counter() - t0)
        tokens_per_sec = train_tokens_processed / max(approx_training_time_s, 1e-6)
        if first_train_step_time_s is None:
            first_train_step_time_s = approx_training_time_s
            first_train_tokens_processed = train_tokens_processed
        train_time_s_zero = max(0.0, approx_training_time_s - first_train_step_time_s)
        train_tokens_processed_for_plot = max(
            0, train_tokens_processed - (first_train_tokens_processed or 0)
        )
        train_step_time_s = (
            0.0
            if prev_train_step_time_s is None
            else max(0.0, approx_training_time_s - prev_train_step_time_s)
        )
        prev_train_step_time_s = approx_training_time_s

        # Log training metrics to wandb
        if logging_config["use_wandb"]:
            log_dict = {
                "train_loss": train_loss_per_token,
                "train_time_s": train_time_s_zero,
                "train_time_s_abs": approx_training_time_s,
                "train_time_s_zero": train_time_s_zero,
                "train_step_time_s": train_step_time_s,
                "step_avg_s": approx_training_time_s / (step + 1),
                "learning_rate": optimizers[0].param_groups[0]["lr"],
                "muon_lr": optimizer2.param_groups[0]["lr"] if optimizer2 is not None else None,
                "muon_momentum": (
                    optimizer2.param_groups[0]["momentum"] if optimizer2 is not None else None
                ),
                "window_size_blocks": (
                    get_window_size_blocks(step).item() if model_type == "gpt" else None
                ),
                "window_size": (
                    training_manager.get_window_size(step) if model_type == "gpt" else None
                ),
                "lr_multiplier": get_lr(step),
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": effective_batch_size,
                "tokens_processed": train_tokens_processed_for_plot,
                "total_tokens_processed": total_tokens_processed,
                "train_tokens_processed_abs": train_tokens_processed,
                "train_tokens_this_step": train_tokens_this_step,
                "tokens_per_sec": tokens_per_sec,
            }

            log_dict.update(grad_norm_dict)

            wandb.log(log_dict, step=step + 1)

    peak_memory = torch.cuda.max_memory_allocated() // 1024 // 1024
    reserved_memory = torch.cuda.max_memory_reserved() // 1024 // 1024
    print_log(
        f"peak memory allocated: {peak_memory} MiB reserved: {reserved_memory} MiB", console=True
    )

    # Calculate final tokens per second
    final_tokens_per_sec = train_tokens_processed / max(training_time_s, 1e-6)

    # Log final metrics and finish wandb
    if logging_config["use_wandb"]:
        wandb.log(
            {
                "peak_memory_MiB": peak_memory,
                "reserved_memory_MiB": reserved_memory,
                "final_train_tokens_processed": train_tokens_processed,
                "final_total_tokens_processed": total_tokens_processed,
                "final_tokens_per_sec": final_tokens_per_sec,
            }
        )
        wandb.finish()
