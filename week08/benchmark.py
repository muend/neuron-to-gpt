"""Measure this runtime instead of guessing T4/A100 training duration."""

import argparse
import json
import time

import torch

from config import ModelConfig, PRESETS
from data import Corpus, corpus_path
from models import make_model
from train import amp_context, resolve_device, resolve_precision, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="t4")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("auto", "none", "fp16", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    settings = PRESETS[args.preset].copy()
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("batch-size must be positive")
        settings["batch_size"] = args.batch_size
    corpus = Corpus(corpus_path("en"), settings["block_size"], device)
    config = ModelConfig(len(corpus.tokenizer), **{k: v for k, v in settings.items()
                                               if k not in ("batch_size", "steps", "learning_rate")})
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024.)
    generator = torch.Generator().manual_seed(1337)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def update():
        x, y = corpus.get_batch("train", settings["batch_size"], generator)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(precision):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=precision != "fp16")
        scaler.step(optimizer)
        scaler.update()
    for _ in range(10):
        update()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(50):
        update()
    synchronize(device)
    seconds_per_step = (time.perf_counter() - start) / 50
    print(json.dumps({"preset": args.preset, "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                      "precision": precision, "seconds_per_step": seconds_per_step,
                      "training_only_minutes_per_language": seconds_per_step * settings["steps"] / 60,
                      "peak_allocated_GB": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
                      "note": "Warmup 10, measured 50 updates. Estimate excludes evaluation, checkpoints and sampling; "
                              "one small probe cannot guarantee future runtime performance."}, indent=2))


if __name__ == "__main__":
    main()
