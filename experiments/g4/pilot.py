"""Bounded continuation of our L4 GPT-2, with fresh data and fixed validation."""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from week12.common import (FINEWEB_REVISION, GPT, GPTConfig, GPT2_REVISION,
                           choose_device, choose_precision, configure_optimizer,
                           environment, make_scaler, optimizer_update, read_json,
                           save_json, set_tf32, sha256, source_hashes, synchronize)
from week12.data import ShardLoader, ShardWriter
from week12.train import sample_text, save_checkpoint, validation_loss
import torch

BASE_SHA = "3151b14452b1a62ef7e0722acdd3071025700da2881c3950994536e3aedc97f3"
BASE_MANIFEST = ROOT / "week12/results/nvidia_l4/data_manifest.json"


def sources():
    return dict(source_hashes(), **{"experiments/g4/pilot.py": sha256(__file__)})


def remaining_seconds(minutes, deadline=None, now=None):
    budget = minutes * 60
    if deadline:
        limit = datetime.fromisoformat(deadline)
        if limit.tzinfo is None:
            raise ValueError("Deadline needs an explicit timezone")
        current = now or datetime.now(limit.tzinfo)
        budget = min(budget, (limit - current).total_seconds())
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Training deadline has passed or budget is invalid")
    return budget


def pilot_lr(step, elapsed, budget, warmup=30, peak=3e-4, floor=3e-5):
    ramp = min(1.0, (step + 1) / warmup)
    progress = min(1.0, max(0.0, elapsed / budget))
    return ramp * (floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2)


def resume_evidence(checkpoint, settings, provenance):
    """Validate state and align interrupted microbatches to committed AdamW steps.

    BF16 has no skipped scaler updates and this model has zero dropout and no
    mutable training buffers. Uncommitted microbatches can therefore be read
    again. AdamW tensor step counters must all agree before accepting a save.
    """
    evidence = copy.deepcopy(checkpoint["evidence"])
    if (evidence["settings"] != settings or evidence["smoke_only"]
            or evidence["provenance"]["base_checkpoint_sha256"] != BASE_SHA
            or evidence["provenance"]["data_manifest_sha256"] != provenance["data_manifest_sha256"]):
        raise ValueError("Resume settings, base checkpoint or dataset differs")
    previous_sources = evidence["provenance"]["source_sha256"]
    for name in set(previous_sources) | set(provenance["source_sha256"]):
        if name != "experiments/g4/pilot.py" and previous_sources.get(name) != provenance["source_sha256"].get(name):
            raise ValueError(f"Resume core source differs: {name}")
    if settings["precision"] != "bf16" or checkpoint["model_config"]["dropout"] != 0:
        raise ValueError("Interrupted-state alignment requires BF16 and zero dropout")
    states = checkpoint["optimizer"]["state"].values()
    counters = {float(state["step"]) for state in states if "step" in state}
    if len(counters) != 1:
        raise ValueError("AdamW step counters disagree; use an earlier complete checkpoint")
    total_steps = counters.pop()
    if not math.isfinite(total_steps) or not total_steps.is_integer():
        raise ValueError("Invalid AdamW step counter")
    committed = int(total_steps) - evidence["base_updates"]
    batch_tokens = evidence["effective_batch_tokens"]
    committed_tokens = committed * batch_tokens
    uncommitted = checkpoint["loader_tokens_seen"] - committed_tokens
    if committed < 0 or not 0 <= uncommitted <= batch_tokens:
        raise ValueError("Loader state is inconsistent with committed optimizer steps")
    if abs(committed - evidence["successful_updates"]) > 1:
        raise ValueError("Optimizer and logged step counts differ by more than one update")
    audit = dict(previous_successful_updates=evidence["successful_updates"],
                 committed_updates=committed, replayed_uncommitted_tokens=uncommitted,
                 previous_source_sha256=previous_sources,
                 previous_budget_seconds=evidence["budget_seconds"], previous_deadline=evidence.get("deadline"))
    evidence["successful_updates"] = committed
    evidence["attempted_updates"] = committed  # BF16: every completed step is successful.
    evidence["tokens_seen"] = committed_tokens
    evidence.setdefault("resumes", []).append(audit)
    evidence["provenance"] = provenance
    evidence.pop("error", None)
    return evidence, committed_tokens


def prepare_fresh(documents, encoding, directory, *, old_manifest, val_reader,
                  train_tokens, seconds=240, shard_size=1_000_000):
    """Exclude every baseline document; retain the byte-identical old val shard.

    The time bound is checked between documents; a blocked network read can
    exceed it. An interrupted incomplete directory is never reused silently.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Fresh data directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    seen = set(old_manifest["document_sha256"]["train"])
    seen.update(old_manifest["document_sha256"]["val"])
    hashes, duplicates = [], 0
    writer = ShardWriter(directory, "train", shard_size)
    start = time.monotonic()
    for text in documents:
        if time.monotonic() - start >= seconds:
            break
        text = text.strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        tokens = encoding.encode_ordinary(text) + [encoding.eot_token]
        left = train_tokens - writer.total
        if len(tokens) > left:
            tokens = tokens[:left]
            tokens[-1] = encoding.eot_token
        writer.append(tokens)
        hashes.append(digest)
        if len(writer.shards) and len(writer.shards) % 10 == 0 and writer.used < len(tokens):
            print(f"Fresh data: {writer.total:,} tokens", flush=True)
        if writer.total >= train_tokens:
            break
    writer.flush()
    if writer.total < 2:
        raise ValueError("No fresh training data received within preparation budget")
    for entry in old_manifest["splits"]["val"]["shards"]:
        name = entry["filename"]
        if Path(name).name != name or "\\" in name:
            raise ValueError("Invalid validation shard filename")
        val_reader(name, directory / name)
        if sha256(directory / name) != entry["sha256"]:
            raise ValueError("Validation must match the original L4 shard")
    manifest = dict(status="complete", tokenizer="gpt2", dtype="uint16", eot_token=50256,
                    source=dict(dataset="HuggingFaceFW/fineweb-edu", config="sample-10BT",
                                revision=FINEWEB_REVISION, baseline_manifest_sha256=sha256(BASE_MANIFEST)),
                    requested_tokens=dict(train=train_tokens, val=old_manifest["splits"]["val"]["tokens"]),
                    document_sha256=dict(train=hashes, val=old_manifest["document_sha256"]["val"]),
                    duplicates_removed=duplicates, shard_size=shard_size,
                    preparation_seconds=time.monotonic()-start,
                    split_policy="fresh train excludes all baseline train/val exact document hashes; original val retained",
                    splits=dict(train=dict(tokens=writer.total, documents=len(hashes), shards=writer.shards),
                                val=old_manifest["splits"]["val"]))
    save_json(directory / "manifest.json", manifest)
    print(f"Ready: {writer.total:,} fresh tokens; original validation preserved", flush=True)
    return manifest


def prepare(args):
    import tiktoken
    from datasets import load_dataset
    old = read_json(BASE_MANIFEST)
    if (args.data / "manifest.json").exists():
        manifest = read_json(args.data / "manifest.json")
        if manifest["source"].get("baseline_manifest_sha256") != sha256(BASE_MANIFEST):
            raise ValueError("Existing data has different baseline provenance")
        for split in ("train", "val"):
            ShardLoader(args.data, split, 1, 1)
        print("Verified existing fresh-data shards", flush=True)
        return
    # Explicit pinned files avoid dataset-wide discovery of unrelated configs.
    prefix = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{FINEWEB_REVISION}"
    urls = [f"{prefix}/sample/10BT/{i:03d}_00000.parquet" for i in range(4)]
    print("Opening pinned FineWeb stream (CPU preparation; GPU idle)", flush=True)
    stream = load_dataset("parquet", data_files={"train": urls}, split="train",
                          streaming=True, columns=["text"])
    with zipfile.ZipFile(args.tokens_zip) as archive:
        def restore(name, target):
            matches = [item for item in archive.infolist() if Path(item.filename).name == name]
            if len(matches) != 1 or matches[0].file_size > 10_000_000:
                raise ValueError("Missing/ambiguous/oversized validation entry")
            with archive.open(matches[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return prepare_fresh((row["text"] for row in stream), tiktoken.get_encoding("gpt2"),
                             args.data, old_manifest=old, val_reader=restore,
                             train_tokens=args.train_tokens, seconds=args.prepare_seconds)


def backup(output, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("last.pt", "best.pt", "training.json", "data_manifest.json", "updates.jsonl"):
        source = output / name
        if source.exists():
            target = destination / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(target)


def train(args):
    import tiktoken
    budget = remaining_seconds(args.minutes, args.deadline)
    started = time.perf_counter()
    prior_elapsed = 0.0
    elapsed = lambda: prior_elapsed + time.perf_counter() - started
    if (args.output / "training.json").exists() and not args.resume:
        raise ValueError("Choose a new run directory; pilot never overwrites an existing run")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.resume or args.checkpoint
    checkpoint_digest = sha256(checkpoint_path)
    if not args.resume and checkpoint_digest != BASE_SHA:
        raise ValueError("Expected our verified L4 checkpoint, not OpenAI weights or another run")
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    device = choose_device("cuda")
    precision = choose_precision(device)
    if precision != "bf16":
        raise ValueError("This G4 pilot requires CUDA BF16 support")
    set_tf32(True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = GPTConfig(**checkpoint["model_config"])
    raw = GPT(config).to(device).train()
    raw.load_state_dict(checkpoint["model"])
    optimizer, audit = configure_optimizer(raw, device, lr=3e-4)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = make_scaler(device, precision)
    base = checkpoint["evidence"]
    loader = ShardLoader(args.data, "train", args.batch_size, args.context)
    # Keep B/T/target count identical to the L4 validation protocol.
    val = ShardLoader(args.data, "val", 4, 256)
    encoding = tiktoken.get_encoding("gpt2")
    batch_tokens = args.batch_size * args.context * args.accumulation
    manifest = read_json(args.data / "manifest.json")
    if manifest["source"].get("baseline_manifest_sha256") != sha256(BASE_MANIFEST):
        raise ValueError("Fresh-data manifest must derive from the verified L4 validation split")
    baseline_manifest = read_json(BASE_MANIFEST)
    if manifest["splits"]["val"] != baseline_manifest["splits"]["val"]:
        raise ValueError("Validation shard metadata must exactly match the L4 run")
    excluded = set().union(*map(set, baseline_manifest["document_sha256"].values()))
    if excluded.intersection(manifest["document_sha256"]["train"]):
        raise ValueError("Fresh training data contains a baseline document")
    if args.context > config.block_size:
        raise ValueError("Context exceeds learned position table")
    evidence = dict(status="running", smoke_only=False, environment=environment(device),
                    provenance=dict(source_sha256=sources(), base_checkpoint_sha256=BASE_SHA,
                                    data_manifest_sha256=sha256(args.data / "manifest.json")),
                    base_updates=base["successful_updates"], base_tokens=base["tokens_seen"],
                    successful_updates=0, attempted_updates=0, tokens_seen=0,
                    settings=dict(batch_size=args.batch_size, context=args.context,
                                  accumulation=args.accumulation, precision=precision, compile=not args.no_compile,
                                  peak_lr=3e-4, floor_lr=3e-5, warmup_updates=30,
                                  validation="B4 T256 first32768 targets, original L4 shard"),
                    optimizer=audit, effective_batch_tokens=batch_tokens,
                    budget_seconds=budget, deadline=args.deadline,
                    timing_scope="90-minute cap includes loading, compilation, validation and periodic IO; final save/test extra",
                    validation=[], samples=[], history=[], optimizer_seconds=0.0)
    if args.resume:
        evidence, loader.tokens_seen = resume_evidence(checkpoint, evidence["settings"], evidence["provenance"])
        if not (args.output / "best.pt").exists():
            raise ValueError("Resume needs the original best.pt in the same output directory")
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["rng_cpu"])
        if checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        prior_elapsed = evidence["elapsed_wall_seconds"]
        budget = min(args.minutes * 60, prior_elapsed + remaining_seconds(args.minutes, args.deadline))
        if budget <= prior_elapsed:
            raise ValueError("Requested total budget has already been consumed")
        evidence["resumes"][-1].update(checkpoint_sha256=checkpoint_digest,
                                       requested_total_budget_seconds=budget)
        evidence.update(status="running", environment=environment(device),
                        budget_seconds=budget, deadline=args.deadline)
        print(f"RESUMED step {evidence['successful_updates']} | {loader.tokens_seen:,} tokens "
              f"| {prior_elapsed/60:.2f}/{budget/60:.2f} min | deadline {args.deadline}", flush=True)
    del checkpoint
    save_json(args.output / "data_manifest.json", manifest)
    model = torch.compile(raw, fullgraph=True) if not args.no_compile else raw
    best_loss = evidence.get("best_validation_loss", float("inf"))

    def persist():
        evidence.update(elapsed_wall_seconds=elapsed(), tokens_seen=loader.tokens_seen)
        save_json(args.output / "training.json", evidence)

    def evaluate():
        nonlocal best_loss
        row = validation_loss(raw, val, device, precision, 32)
        row.update(step=evidence["successful_updates"], elapsed_seconds=elapsed())
        evidence["validation"].append(row)
        evidence["samples"].append(dict(step=row["step"], text=sample_text(
            raw, encoding, device, precision, "The children walked into the", 48)))
        print(f"VAL step {row['step']} | loss {row['loss']:.6f} | {elapsed()/60:.1f} min", flush=True)
        print(evidence["samples"][-1]["text"], flush=True)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            evidence.update(best_validation_loss=best_loss, best_step=row["step"],
                            best_tokens_seen=loader.tokens_seen, tokens_seen=loader.tokens_seen,
                            elapsed_wall_seconds=elapsed())
            temporary = args.output / "best.pt.tmp"
            torch.save(dict(model_config=asdict(config), model=raw.state_dict(),
                            evidence=dict(evidence)), temporary)
            temporary.replace(args.output / "best.pt")
        persist()

    next_eval, next_save = elapsed() + 300.0, elapsed() + 600.0
    try:
        evaluate()
        with (args.output / "updates.jsonl").open("a", encoding="utf-8", buffering=1) as log:
            while elapsed() < budget:
                step = evidence["successful_updates"]
                lr = pilot_lr(step, elapsed(), budget)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                synchronize(device)
                update_start = time.perf_counter()
                row = optimizer_update(model, optimizer, loader, device,
                                       accumulation_steps=args.accumulation, precision=precision, scaler=scaler)
                synchronize(device)
                duration = time.perf_counter() - update_start
                evidence["attempted_updates"] += 1
                evidence["successful_updates"] += int(not row["skipped_update"])
                evidence["optimizer_seconds"] += duration
                row.update(step=evidence["successful_updates"], learning_rate=lr,
                           elapsed_seconds=elapsed(), tokens_seen=loader.tokens_seen,
                           milliseconds=duration*1000, tokens_per_second=batch_tokens/duration)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if evidence["attempted_updates"] % 10 == 0 or evidence["attempted_updates"] == 1:
                    evidence["history"].append(row)
                    print(f"step {row['step']} | loss {row['loss']:.4f} | {row['tokens_per_second']:.0f} tok/s "
                          f"| lr {lr:.7f} | {elapsed()/60:.1f}/{budget/60:.1f} min", flush=True)
                    persist()
                if elapsed() >= next_eval:
                    evaluate()
                    next_eval = elapsed() + 300
                if elapsed() >= next_save:
                    persist()
                    save_checkpoint(args.output / "last.pt", raw, optimizer, scaler, loader, evidence)
                    backup(args.output, args.backup)
                    next_save = elapsed() + 600
        evaluate()
        evidence.update(status="complete", stop_reason="time_budget_or_deadline")
    except BaseException as error:
        evidence.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        persist()
        save_checkpoint(args.output / "last.pt", raw, optimizer, scaler, loader, evidence)
        backup(args.output, args.backup)
    print("Training saved. Evaluate best.pt selected solely by validation loss.", flush=True)


def finish(args):
    from week12.evaluate import prepare_hellaswag, evaluate_model, HELLASWAG_REVISION
    import tiktoken
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    device = choose_device("cuda")
    torch.set_num_threads(4)
    set_tf32(False)
    cache = prepare_hellaswag(args.data / "hellaswag", 2000)
    examples = read_json(cache)["examples"]
    encoding = tiktoken.get_encoding("gpt2")
    evaluation_id = hashlib.sha256((sha256(args.output / "best.pt") +
        json.dumps(sources(), sort_keys=True)).encode()).hexdigest()[:16]
    evaluation_dir = args.output / "evaluations" / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in ("own", "openai"):
        if task == "own":
            path = args.output / "best.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = GPT(GPTConfig(**state["model_config"]))
            model.load_state_dict(state["model"])
            identity = dict(checkpoint_sha256=sha256(path), selected_step=state["evidence"]["successful_updates"])
            del state
        else:
            model = GPT.from_pretrained(revision=GPT2_REVISION,
                                        cache_dir=ROOT / "week12/.cache/gpt2")
            identity = dict(model_id="openai-community/gpt2", model_revision=GPT2_REVISION)
        for block in model.transformer.h:
            block.attn.attn_implementation = "sdpa"
        model.to(device).eval()
        for offset, label in ((0, "course1000"), (1000, "heldout1000")):
            selection = examples[offset:offset+1000]
            provenance = dict(identity, model_name=task, source_sha256=sources(),
                              dataset_revision=HELLASWAG_REVISION, environment=environment(device),
                              selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
                              selection=f"validation indices [{offset},{offset+1000})", example_count=1000,
                              scoring="ending-only mean NLL, common 50257 vocabulary, FP32, TF32 off")
            results[f"{task}_{label}"] = evaluate_model(model, selection, encoding, device, provenance,
                evaluation_dir / f"hellaswag_{task}_{label}.json")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    record = read_json(args.output / "training.json")
    vals = record["validation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([v["elapsed_seconds"]/60 for v in vals], [v["loss"] for v in vals], "o-")
    axes[0].axhline(4.984965160489082, linestyle="--", color="gray", label="L4 final")
    axes[0].set(xlabel="G4 minutes", ylabel="Validation loss (fixed T256)")
    axes[0].legend()
    names = ["L4/course", "G4/course", "GPT2/course", "G4/heldout", "GPT2/heldout"]
    scores = [27.6] + [results[k]["accuracy_norm"]*100 for k in
        ("own_course1000", "openai_course1000", "own_heldout1000", "openai_heldout1000")]
    bars = axes[1].bar(names, scores)
    axes[1].bar_label(bars, fmt="%.1f%%")
    axes[1].axhline(25, color="gray", linestyle="--")
    axes[1].set(ylabel="HellaSwag accuracy (%)", ylim=(0, max(scores)+10))
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(args.output / "summary.png", dpi=160)
    summary = dict(base_validation_loss=4.984965160489082, best_validation_loss=record["best_validation_loss"],
                   best_step=record["best_step"], additional_tokens=record["tokens_seen"],
                   selected_checkpoint_additional_tokens=record["best_tokens_seen"],
                   total_tokens_including_L4=record["base_tokens"]+record["tokens_seen"],
                   training_minutes=record["elapsed_wall_seconds"]/60,
                   optimizer_tokens_per_second=record["tokens_seen"]/max(record["optimizer_seconds"], 1e-9),
                   wall_tokens_per_second=record["tokens_seen"]/max(record["elapsed_wall_seconds"], 1e-9),
                   fresh_corpus_tokens=read_json(args.output / "data_manifest.json")["splits"]["train"]["tokens"],
                   evaluation_directory=f"evaluations/{evaluation_id}",
                   hellaswag_percent=dict(zip(names, scores)),
                   interpretation="Pilot; new heldout1000 is primary comparison. A point-estimate lead alone does not establish superiority.")
    save_json(args.output / "summary.json", summary)
    backup(args.output, args.backup)
    evaluation_backup = args.backup / "evaluations" / evaluation_id
    evaluation_backup.mkdir(parents=True, exist_ok=True)
    for path in evaluation_dir.glob("hellaswag_*.json"):
        shutil.copyfile(path, evaluation_backup / path.name)
    for name in ("summary.json", "summary.png"):
        shutil.copyfile(args.output / name, args.backup / name)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("prepare", "train", "finish"))
    parser.add_argument("--data", type=Path, default=Path("experiments/g4/data/fresh200m"))
    parser.add_argument("--output", type=Path, default=Path("experiments/g4/outputs/pilot90"))
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume G4 last.pt; --minutes is TOTAL G4 run time")
    parser.add_argument("--tokens-zip", type=Path)
    parser.add_argument("--train-tokens", type=int, default=200_000_000)
    parser.add_argument("--prepare-seconds", type=float, default=240)
    parser.add_argument("--minutes", type=float, default=90)
    parser.add_argument("--deadline", help="ISO timestamp with timezone; stops new steps before this time")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if (min(args.batch_size, args.context, args.accumulation, args.train_tokens) < 1
            or not math.isfinite(args.prepare_seconds) or args.prepare_seconds <= 0):
        parser.error("Sizes and preparation budget must be positive")
    if args.task == "prepare" and not args.tokens_zip:
        parser.error("prepare requires --tokens-zip from our L4 backup")
    if args.task in ("train", "finish") and not args.backup:
        parser.error("train/finish require a separate --backup directory")
    if args.task == "train" and not (args.checkpoint or args.resume):
        parser.error("train requires --checkpoint from L4 or --resume from G4")
    if args.task in ("train", "finish"):
        forbidden = {args.output.resolve()}
        if args.checkpoint:
            forbidden.add(args.checkpoint.parent.resolve())
        if args.backup.resolve() in forbidden:
            parser.error("Backup must be separate from local output and original checkpoint folder")
    {"prepare": prepare, "train": train, "finish": finish}[args.task](args)
    if args.task == "prepare":
        # Bounded HF streams may retain network workers during interpreter
        # shutdown. Writers and ZIP are closed and the manifest is saved.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
