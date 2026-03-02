"""Lightweight post-training text sampling for validation checks."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

DEFAULT_VALIDATION_INFERENCE = dict(
    num_samples=3,
    prompt_tokens=128,
    max_new_tokens=80,
    temperature=0.8,
    top_k=40,
    seed=42,
    stop_on_eos=True,
)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    generator: torch.Generator,
) -> int:
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.numel())
        top_vals, top_idx = torch.topk(logits, k)
        probs = F.softmax(top_vals / temperature, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=generator).item()
        return int(top_idx[choice].item())

    probs = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


class Engine:
    """Single-purpose autoregressive sampler using model KV cache."""

    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def generate_one(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        seed: int,
        window_size_blocks: torch.Tensor,
        eos_token_id: int,
        stop_on_eos: bool,
    ) -> List[int]:
        if not prompt_tokens or max_tokens <= 0:
            return []

        device = next(self.model.parameters()).device
        create_cache = getattr(self.model, "create_kv_cache", None)
        if create_cache is None:
            return []
        cache = create_cache()
        prompt = torch.tensor(prompt_tokens, dtype=torch.int32, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        out: List[int] = []
        vocab_size = int(self.model.model_config["vocab_size"])

        # Prefill once; cache receives full K/V for incremental decoding.
        _ = self.model(
            prompt,
            prompt,
            window_size_blocks,
            return_logits=True,
            kv_cache=cache,
        )

        current_token = prompt[-1:].contiguous()
        while len(out) < max_tokens:
            logits = self.model(
                current_token,
                current_token,
                window_size_blocks,
                return_logits=True,
                kv_cache=cache,
            )
            next_token = _sample_next_token(
                logits[-1, :vocab_size].float(),
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            out.append(next_token)
            current_token = torch.tensor([next_token], dtype=torch.int32, device=device)
            if stop_on_eos and next_token == eos_token_id:
                break
        return out

    def generate_batch(
        self,
        prompt_tokens: List[int],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        seed: int,
        window_size_blocks: torch.Tensor,
        eos_token_id: int,
        stop_on_eos: bool,
    ) -> List[List[int]]:
        if num_samples <= 0:
            return []

        results = []
        for sample_idx in range(int(num_samples)):
            sample_seed = int(seed) + sample_idx if seed is not None else None
            sample = self.generate_one(
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=sample_seed,
                window_size_blocks=window_size_blocks,
                eos_token_id=eos_token_id,
                stop_on_eos=stop_on_eos,
            )
            results.append(sample)
        return results


def _resolve_inference_config(config: Optional[dict], data_seq_len: int) -> Dict[str, object]:
    inference_cfg = dict(DEFAULT_VALIDATION_INFERENCE)
    if isinstance(config, dict):
        inference_cfg.update(config)

    prompt_tokens = int(max(1, inference_cfg.get("prompt_tokens", 128)))
    prompt_tokens = min(prompt_tokens, max(1, data_seq_len - 1))
    max_new_tokens = int(max(1, inference_cfg.get("max_new_tokens", 80)))
    max_new_tokens = min(max_new_tokens, max(1, data_seq_len - prompt_tokens))

    return {
        **inference_cfg,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
    }


def _as_int(value, fallback: int) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return fallback
    return ivalue


def run_validation_generation(
    model: torch.nn.Module,
    tokenizer,
    val_loader,
    window_size_blocks: torch.Tensor,
    data_seq_len: int,
    config: Optional[dict] = None,
) -> List[Dict[str, object]]:
    """Generate compact text examples on validation data for human inspection."""

    if data_seq_len <= 1:
        return []

    cfg = _resolve_inference_config(config, data_seq_len)
    num_samples = max(1, _as_int(cfg.get("num_samples"), 3))
    prompt_tokens = _as_int(cfg.get("prompt_tokens"), data_seq_len // 2)
    max_new_tokens = max(1, _as_int(cfg.get("max_new_tokens"), 80))
    temperature = float(cfg.get("temperature", 0.8))
    top_k = cfg.get("top_k")
    top_k = None if top_k is None else max(0, int(top_k))
    seed = cfg.get("seed", 42)
    stop_on_eos = bool(cfg.get("stop_on_eos", True))

    if num_samples <= 0 or max_new_tokens <= 0:
        return []

    engine = Engine(model)
    eos_token_id = int(model.model_config.get("eos_token_id", 50256))
    was_training = model.training
    model.eval()

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for sample_idx in range(num_samples):
            prompt, _ = next(val_loader)
            prompt_slice = prompt[:prompt_tokens].tolist()
            if not prompt_slice:
                continue

            generated = engine.generate_batch(
                prompt_slice,
                num_samples=1,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=_as_int(seed, 42) + sample_idx,
                window_size_blocks=window_size_blocks,
                eos_token_id=eos_token_id,
                stop_on_eos=stop_on_eos,
            )[0]

            if stop_on_eos and generated and generated[-1] == eos_token_id:
                generated = generated[:-1]

            rows.append(
                {
                    "prompt": tokenizer.decode(prompt_slice),
                    "generated": tokenizer.decode(generated),
                    "prompt_len": len(prompt_slice),
                    "generated_len": len(generated),
                }
            )

    if was_training:
        model.train()

    return rows
