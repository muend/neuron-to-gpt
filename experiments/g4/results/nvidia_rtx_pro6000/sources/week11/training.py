"""Training primitives for Week 11; reuse our verified Week 10 architecture."""

from contextlib import nullcontext
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from week10.model import GPT, GPTConfig


class DataLoaderLite:
    """Read a cyclic token stream, including its tail, with next-token targets.

    Wrapping treats the text as repeated documents. The tokenizer appends EOT
    so a document boundary is explicit. No incomplete tail is silently dropped.
    State is CPU-only; next_batch transfers just the requested batch.
    """

    def __init__(self, tokens, batch_size, sequence_length):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long).cpu()
        if self.tokens.ndim != 1 or len(self.tokens) < 2:
            raise ValueError("At least two tokens in a one-dimensional stream are required")
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("Batch size and sequence length must be positive")
        self.batch_size, self.sequence_length = batch_size, sequence_length
        self.tokens_seen = 0

    def next_batch(self, device="cpu"):
        count = self.batch_size * self.sequence_length
        positions = (torch.arange(count + 1) + self.tokens_seen) % len(self.tokens)
        window = self.tokens[positions]
        self.tokens_seen += count
        shape = (self.batch_size, self.sequence_length)
        return window[:-1].reshape(shape).to(device), window[1:].reshape(shape).to(device)


def learning_rate(step, warmup_steps, total_steps, max_lr=6e-4, min_lr=6e-5):
    """Zero-based update index; last planned update reaches the LR floor."""
    if not 0 < warmup_steps < total_steps - 1:
        raise ValueError("Require 0 < warmup_steps < total_steps - 1")
    if step < 0 or not 0 <= min_lr <= max_lr or not math.isfinite(max_lr):
        raise ValueError("Invalid step or learning-rate bounds")
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / (total_steps - 1 - warmup_steps))
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


def configure_optimizer(model, device, lr=6e-4, weight_decay=0.1):
    # parameters() deduplicates the tied embedding/output weight.
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim == 2 else no_decay).append(parameter)
    groups = [dict(params=decay, weight_decay=weight_decay),
              dict(params=no_decay, weight_decay=0.0)]
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                  fused=device.type == "cuda")
    audit = {"decay_tensors": len(decay), "decay_parameters": sum(p.numel() for p in decay),
             "no_decay_tensors": len(no_decay),
             "no_decay_parameters": sum(p.numel() for p in no_decay),
             "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": weight_decay,
             "fused": device.type == "cuda"}
    return optimizer, audit


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type,
                          dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def attention_context(force_flash):
    if not force_flash:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    # Disallow math/memory-efficient fallback: a successful measurement proves
    # the flash backend ran (including during compiled execution).
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION)


def make_scaler(device, precision):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")


def optimizer_update(model, optimizer, loader, device, *, accumulation_steps=1,
                     precision="fp32", scaler=None, force_flash=False):
    """One update: average equal-token microbatches, unscale, clip once, step."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    optimizer.zero_grad(set_to_none=True)
    mean_loss = torch.zeros((), device=device)
    with attention_context(force_flash):
        for _ in range(accumulation_steps):
            x, y = loader.next_batch(device)
            with autocast_context(device, precision):
                _, loss = model(x, y)
                loss = loss / accumulation_steps
            mean_loss += loss.detach()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        # Return value is the global norm BEFORE clipping; logged every update.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite = bool(torch.isfinite(norm).item() and torch.isfinite(mean_loss).item())
        if not finite and (scaler is None or not scaler.is_enabled()):
            raise FloatingPointError("Nonfinite loss/gradient in an unscaled update")
        if scaler is not None:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
        else:
            optimizer.step()
            skipped = False
    return {"loss": mean_loss.item(), "gradient_norm_before_clip": norm.item(),
            "skipped_update": skipped, "finite": finite}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def set_tf32(enabled):
    # A fresh process for each CLI run avoids mixing this API with the newer
    # backend fp32_precision API. Apply the same setting to every stage.
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    torch.backends.cudnn.allow_tf32 = enabled


def config_for(tiny=False, padded=False, attention="eager"):
    return GPTConfig(block_size=32 if tiny else 1024,
                     vocab_size=(384 if padded else 257) if tiny else (50304 if padded else 50257),
                     n_layer=2 if tiny else 12, n_head=2 if tiny else 12,
                     n_embd=64 if tiny else 768, attn_implementation=attention)
