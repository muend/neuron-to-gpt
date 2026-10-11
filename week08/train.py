"""Train/evaluate a staged GPT; save resumable checkpoints and measured results."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import torch

from config import ModelConfig, PRESETS, STAGES
from data import Corpus, HERE, corpus_path
from models import make_model


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select a Colab GPU runtime, or use cpu for smoke tests.")
    return name


def resolve_precision(precision, device):
    if precision == "auto":
        return ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "none"
    if precision != "none" and device != "cuda":
        raise ValueError("Mixed precision is supported here only on CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; choose fp16")
    return precision


def amp_context(precision):
    if precision == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model, corpus, batch_size, eval_batches):
    previous_mode = model.training
    model.eval()
    try:
        results = {}
        for split, seed in (("train", 2026), ("val", 2027)):
            generator = torch.Generator().manual_seed(seed)
            total = 0.
            for _ in range(eval_batches):
                x, y = corpus.get_batch(split, batch_size, generator)
                # FP32 evaluation makes the reported validation metric consistent.
                _, loss = model(x, y)
                total += loss.item()
            results[split] = total / eval_batches
        return results
    finally:
        model.train(previous_mode)


def plot_history(history, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for split, style in (("train", "--"), ("val", "-")):
        ax.plot([row["step"] for row in history], [row[split] for row in history], style, label=split)
    ax.set(xlabel="Optimizer step", ylabel="Sampled NLL (nats/character)", title="Train / validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


def fit(config, corpus, run_dir, *, steps, batch_size=32, learning_rate=.003,
        eval_batches=200, eval_interval=500, seed=1337, precision="none", resume=False,
        sample_tokens=600, preset="custom"):
    run_dir = Path(run_dir)
    if config.vocab_size != len(corpus.tokenizer) or config.block_size != corpus.block_size:
        raise ValueError("Model vocabulary/context must match the loaded corpus settings")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = str(corpus.device)
    precision = resolve_precision(precision, device)
    critical = {"batch_size": batch_size, "learning_rate": learning_rate,
                "seed": seed, "precision": precision, "eval_batches": eval_batches}
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists() and not resume:
        raise ValueError(f"Run already exists: {run_dir}. Use --resume or a new output directory.")
    if resume and not last_path.exists():
        raise ValueError(f"No checkpoint to resume in {run_dir}")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024.)
    generator = torch.Generator().manual_seed(seed + 1)
    start, history, training_seconds, elapsed_seconds = 0, [], 0., 0.
    best_val = math.inf
    skipped_updates = 0
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["model_config"] != config.to_dict() or saved["training_config"] != critical:
            raise ValueError("Model/critical training settings changed; use a new run directory")
        if saved["sha256"] != corpus.sha256 or saved["chars"] != corpus.tokenizer.chars:
            raise ValueError("Corpus or vocabulary changed; refusing to resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["batch_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, history, best_val = saved["step"], saved["history"], saved["best_val"]
        training_seconds, elapsed_seconds = saved["training_seconds"], saved["elapsed_seconds"]
        skipped_updates = saved["skipped_updates"]
        print(f"Resuming {run_dir.name} at step {start}", flush=True)
    if steps < start:
        raise ValueError("Requested total steps are below the saved step")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"stage={config.stage}, parameters={parameters:,}, device={device}, precision={precision}", flush=True)
    print(f"config={config.to_dict()}", flush=True)
    session_start = time.perf_counter()
    synchronize(device)
    segment_start = time.perf_counter()

    def save(step):
        state = {"model_config": config.to_dict(), "training_config": critical,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "chars": corpus.tokenizer.chars,
                 "sha256": corpus.sha256, "step": step, "history": history,
                 "best_val": best_val, "batch_rng": generator.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                 "training_seconds": training_seconds,
                 "skipped_updates": skipped_updates,
                 "elapsed_seconds": elapsed_seconds + time.perf_counter() - session_start}
        atomic_checkpoint(last_path, state)
        return state

    model.train()
    for step in range(start + 1, steps + 1):
        x, y = corpus.get_batch("train", batch_size, generator)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(precision):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # GradScaler detects FP16 overflow and safely skips that update.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=precision != "fp16")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped_updates += int(scaler.get_scale() < previous_scale)
        if step % eval_interval == 0 or step == steps:
            synchronize(device)
            training_seconds += time.perf_counter() - segment_start
            scores = evaluate(model, corpus, batch_size, eval_batches)
            history.append({"step": step, **scores})
            improved = scores["val"] < best_val
            best_val = min(best_val, scores["val"])
            state = save(step)
            if improved:
                atomic_checkpoint(best_path, state)
            atomic_json(run_dir / "history.json", history)
            print(f"step {step}/{steps}: train={scores['train']:.4f}, val={scores['val']:.4f}, "
                  f"training seconds={training_seconds:.1f}; saved last.pt", flush=True)
            synchronize(device)
            segment_start = time.perf_counter()
    if not history:
        raise ValueError("No training/evaluation was performed")
    synchronize(device)
    elapsed_seconds += time.perf_counter() - session_start
    result = {"status": "smoke_only" if preset == "smoke" else "measured",
              "model_config": config.to_dict(), "training_config": critical,
              "steps": steps, "eval_batches": eval_batches,
              "skipped_updates": skipped_updates, "optimizer_updates": steps - skipped_updates,
              "final_train_nll": history[-1]["train"], "final_val_nll": history[-1]["val"],
              "best_val_nll": best_val, "parameters": parameters,
              "training_seconds": training_seconds, "elapsed_seconds": elapsed_seconds,
              "device": device, "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
              "torch_version": str(torch.__version__), "corpus": corpus.metadata, "preset": preset}
    # Sample the best validation checkpoint. The table still reports FINAL loss.
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    sample_generator = torch.Generator(device=device).manual_seed(seed)
    prompt = torch.tensor([[corpus.tokenizer.stoi.get("\n", 0)]], device=device)
    sample = corpus.tokenizer.decode(model.generate(prompt, sample_tokens, sample_generator)[0].tolist())
    (run_dir / "sample.txt").write_text(sample, encoding="utf-8")
    result["sample_checkpoint_step"] = best["step"]
    atomic_json(run_dir / "result.json", result)
    plot_history(history, run_dir)
    print(f"FINAL val NLL={result['final_val_nll']:.4f}; best={best_val:.4f}; "
          f"wall time excluding generation/plot={elapsed_seconds:.1f}s\nSample from best checkpoint:\n{sample}", flush=True)
    return result


def build_settings(args, vocab_size):
    preset = PRESETS[args.preset].copy()
    batch_size = args.batch_size or preset.pop("batch_size")
    preset.pop("batch_size", None)
    steps = args.steps or preset.pop("steps")
    preset.pop("steps", None)
    learning_rate = preset.pop("learning_rate")
    config = ModelConfig(vocab_size, stage=args.stage, **preset)
    return config, {"steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
                    "eval_batches": args.eval_batches, "eval_interval": args.eval_interval,
                    "precision": args.precision, "resume": args.resume,
                    "sample_tokens": args.sample_tokens, "preset": args.preset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "tr"), default="en")
    parser.add_argument("--preset", choices=PRESETS, default="t4")
    parser.add_argument("--stage", choices=STAGES, default="layernorm")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "none", "fp16", "bf16"), default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--sample-tokens", type=int, default=600)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for value in (args.steps, args.batch_size, args.eval_batches, args.eval_interval, args.sample_tokens):
        if value is not None and value < 1:
            parser.error("Sizes, steps and intervals must be positive")
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    corpus = Corpus(corpus_path(args.language), PRESETS[args.preset]["block_size"], device)
    config, settings = build_settings(args, len(corpus.tokenizer))
    fit(config, corpus, args.output_root / f"{args.language}_{args.preset}_{args.stage}", **settings)


if __name__ == "__main__":
    main()
