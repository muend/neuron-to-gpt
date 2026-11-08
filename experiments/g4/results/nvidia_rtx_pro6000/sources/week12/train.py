"""Time-bounded GPT-2 training with held-out loss, samples and resumable state."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import shutil
import time

import torch

try:
    from .common import (GPT, GPTConfig, HERE, autocast_context, choose_device, choose_precision,
                         configure_optimizer, environment, make_scaler, optimizer_update,
                         read_json, save_json, set_tf32, sha256, source_hashes, synchronize)
    from .data import ShardLoader
except ImportError:
    from common import (GPT, GPTConfig, HERE, autocast_context, choose_device, choose_precision,
                        configure_optimizer, environment, make_scaler, optimizer_update,
                        read_json, save_json, set_tf32, sha256, source_hashes, synchronize)
    from data import ShardLoader


def timed_learning_rate(step, warmup_steps, elapsed, budget, warmup_finished=None):
    if step < warmup_steps:
        return 6e-4 * (step + 1) / warmup_steps
    start = warmup_finished if warmup_finished is not None else 0.0
    progress = min(1.0, max(0.0, (elapsed - start) / max(1e-6, budget - start)))
    return 6e-5 + (6e-4 - 6e-5) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def validation_loss(model, loader, device, precision, batches):
    was_training = model.training
    model.eval()
    loader.reset()
    losses = []
    try:
        for _ in range(batches):
            try:
                x, y = loader.next_batch(device)
            except StopIteration:
                break
            with autocast_context(device, precision):
                loss = model(x, y)[1]
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss is nonfinite")
            losses.append(loss.item())
    finally:
        model.train(was_training)
    if not losses:
        raise ValueError("Validation shard must hold at least B*T+1 tokens")
    return dict(loss=sum(losses) / len(losses), batches=len(losses),
                tokens=len(losses) * loader.batch_size * loader.sequence_length,
                available_tokens=loader.total_tokens)


@torch.no_grad()
def sample_text(model, encoding, device, precision, prompt, new_tokens=64):
    was_training = model.training
    model.eval()
    tokens = torch.tensor([encoding.encode_ordinary(prompt)], device=device)
    generator = torch.Generator(device=device).manual_seed(42)
    try:
        for _ in range(new_tokens):
            with autocast_context(device, precision):
                logits = model(tokens[:, -model.config.block_size:])[0][:, -1, :50257].float()
            # Padding rows are implementation details, not decodable GPT-2 IDs.
            values, indices = torch.topk(logits / 0.8, min(50, logits.shape[-1]), dim=-1)
            choice = torch.multinomial(values.softmax(-1), 1, generator=generator)
            token = indices.gather(-1, choice)
            tokens = torch.cat((tokens, token), dim=1)
            if token.item() == encoding.eot_token:
                break
        return encoding.decode(tokens[0].tolist())
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, scaler, loader, evidence, backup_dir=None):
    temporary = path.with_suffix(".pt.tmp")
    payload = dict(model_config=asdict(model.config), model=model.state_dict(),
                   optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                   loader_tokens_seen=loader.tokens_seen, evidence=evidence,
                   rng_cpu=torch.get_rng_state(),
                   rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(payload, temporary)
    temporary.replace(path)
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for source in (path, path.parent / "training.json", path.parent / "data_manifest.json"):
            target = backup_dir / source.name
            staging = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, staging)
            staging.replace(target)


def run(args):
    import tiktoken
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    precision = choose_precision(device)
    set_tf32(device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8)
    torch.manual_seed(1337)
    config = GPTConfig(vocab_size=50304, attn_implementation="sdpa",
                       block_size=128 if args.tiny else 1024,
                       n_layer=2 if args.tiny else 12, n_head=2 if args.tiny else 12,
                       n_embd=32 if args.tiny else 768)
    if args.sequence_length > config.block_size:
        raise ValueError("Requested training context exceeds model positions")
    raw = GPT(config).to(device).train()
    optimizer, optimizer_audit = configure_optimizer(raw, device)
    scaler = make_scaler(device, precision)
    loader = ShardLoader(args.data_dir, "train", args.batch_size, args.sequence_length)
    val_loader = ShardLoader(args.data_dir, "val", args.batch_size, args.sequence_length)
    encoding = tiktoken.get_encoding("gpt2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "manifest.json"
    provenance = dict(source_sha256=source_hashes(), data_manifest_sha256=sha256(manifest_path))
    settings = dict(batch_size=args.batch_size, sequence_length=args.sequence_length,
                    accumulation_steps=args.accumulation_steps, precision=precision,
                    compile=args.compile, warmup_steps=args.warmup_steps,
                    val_batches=args.val_batches, sample_tokens=args.sample_tokens)
    if args.resume:
        # Only load our own checkpoint, never an arbitrary third-party pickle.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        evidence = checkpoint["evidence"]
        if (evidence["provenance"] != provenance or evidence["settings"] != settings
                or checkpoint["model_config"] != asdict(config)):
            raise ValueError("Resume source/data/config differs; restore original settings and sources")
        raw.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        loader.tokens_seen = checkpoint["loader_tokens_seen"]
        torch.set_rng_state(checkpoint["rng_cpu"])
        if device.type == "cuda" and checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        del checkpoint
        evidence["resumes"] = evidence.get("resumes", 0) + 1
    else:
        if (args.output_dir / "training.json").exists() or (args.output_dir / "last.pt").exists():
            raise ValueError("Output already has training; use --resume or choose a new directory")
        evidence = dict(status="running", smoke_only=args.tiny, environment=environment(device),
                        provenance=provenance, settings=settings, model_config=asdict(config),
                        parameters=raw.num_parameters(), optimizer=optimizer_audit,
                        successful_updates=0, attempted_updates=0, skipped_updates=0,
                        tokens_seen=0, elapsed_wall_seconds=0.0, optimizer_seconds=0.0,
                        warmup_finished_seconds=None, history=[], validation=[], samples=[],
                        effective_batch_tokens=args.batch_size * args.sequence_length * args.accumulation_steps,
                        timing_scope="wall budget includes first compile, validation, samples, checkpoint IO; "
                                     "optimizer_seconds covers synchronized updates including first compile",
                        resume_policy="exact loader/optimizer/scaler/RNG state; schedule follows total elapsed wall time")
    save_json(args.output_dir / "data_manifest.json", read_json(manifest_path))
    evidence["budget_seconds"] = args.minutes * 60
    evidence["status"] = "running"
    model = torch.compile(raw, fullgraph=True) if args.compile else raw
    prior_elapsed = evidence["elapsed_wall_seconds"]
    start = time.perf_counter()

    def elapsed():
        return prior_elapsed + time.perf_counter() - start

    def persist():
        evidence["elapsed_wall_seconds"] = elapsed()
        evidence["tokens_seen"] = loader.tokens_seen
        save_json(args.output_dir / "training.json", evidence)

    def evaluate():
        result = validation_loss(raw, val_loader, device, precision, args.val_batches)
        result.update(step=evidence["successful_updates"], elapsed_seconds=elapsed())
        evidence["validation"].append(result)
        prompt = "The children walked into the"
        sample = dict(step=result["step"], prompt=prompt, seed=42, temperature=0.8, top_k=50,
                      text=sample_text(raw, encoding, device, precision, prompt, args.sample_tokens))
        evidence["samples"].append(sample)
        print(f"VAL | step {result['step']} | loss {result['loss']:.6f} | tokens {result['tokens']}", flush=True)
        print("SAMPLE |", sample["text"], flush=True)
        persist()

    try:
        if not evidence["validation"]:
            evaluate()
        while elapsed() < args.minutes * 60:
            step = evidence["successful_updates"]
            if args.max_steps and step >= args.max_steps:
                break
            lr = timed_learning_rate(step, args.warmup_steps, elapsed(), args.minutes * 60,
                                     evidence["warmup_finished_seconds"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            synchronize(device)
            update_start = time.perf_counter()
            row = optimizer_update(model, optimizer, loader, device,
                                   accumulation_steps=args.accumulation_steps,
                                   precision=precision, scaler=scaler)
            synchronize(device)
            duration = time.perf_counter() - update_start
            evidence["attempted_updates"] += 1
            evidence["optimizer_seconds"] += duration
            if row["skipped_update"]:
                evidence["skipped_updates"] += 1
            else:
                evidence["successful_updates"] += 1
            step = evidence["successful_updates"]
            if step == args.warmup_steps and evidence["warmup_finished_seconds"] is None:
                evidence["warmup_finished_seconds"] = elapsed()
            row.update(step=step, attempt=evidence["attempted_updates"], learning_rate=lr,
                       milliseconds=duration * 1000, tokens_per_second=evidence["effective_batch_tokens"] / duration,
                       tokens_seen=loader.tokens_seen, elapsed_seconds=elapsed())
            evidence["history"].append(row)
            if evidence["attempted_updates"] % args.log_every == 0 or evidence["attempted_updates"] == 1:
                print(f"step {step} | loss {row['loss']:.5f} | norm {row['gradient_norm_before_clip']:.3f} "
                      f"| lr {lr:.7f} | {duration*1000:.1f} ms | {row['tokens_per_second']:.0f} tok/s "
                      f"| seen {loader.tokens_seen:,} | elapsed {elapsed()/60:.1f} min", flush=True)
                persist()
            if evidence["skipped_updates"] > 100 and step == 0:
                raise FloatingPointError("More than 100 FP16 updates skipped without progress")
            if not row["skipped_update"] and step % args.eval_every == 0:
                evaluate()
            if not row["skipped_update"] and step % args.checkpoint_every == 0:
                persist()
                save_checkpoint(args.output_dir / "last.pt", raw, optimizer, scaler, loader,
                                evidence, args.backup_dir)
                print("Checkpoint saved |", step, flush=True)
        if not evidence["history"]:
            raise RuntimeError("Time budget ended before an optimizer update")
        if evidence["validation"][-1]["step"] != evidence["successful_updates"]:
            evaluate()
        evidence["status"] = "complete"
        evidence["stop_reason"] = "max_steps" if args.max_steps and evidence["successful_updates"] >= args.max_steps else "time_budget"
        persist()
        save_checkpoint(args.output_dir / "last.pt", raw, optimizer, scaler, loader,
                        evidence, args.backup_dir)
        persist()
    except BaseException as error:
        evidence["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
        persist()
        raise
    finally:
        try:
            from .report import make_report
        except ImportError:
            from report import make_report
        make_report(args.output_dir)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data/fineweb20m")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/l4")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (not math.isfinite(args.minutes) or args.minutes <= 0 or args.max_steps < 0
            or min(args.batch_size, args.sequence_length, args.accumulation_steps, args.warmup_steps,
                   args.eval_every, args.val_batches, args.sample_tokens, args.checkpoint_every,
                   args.log_every, args.threads) < 1):
        parser.error("Use positive sizes/intervals/time and a nonnegative step cap")
    run(args)


if __name__ == "__main__":
    main()
