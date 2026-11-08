"""Completion-only HellaSwag scoring on the same pinned validation examples."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

try:
    from .common import (GPT, GPTConfig, GPT2_REVISION, HELLASWAG_REVISION, HERE,
                         choose_device, environment, read_json, save_json, set_tf32,
                         sha256, source_hashes, synchronize)
except ImportError:
    from common import (GPT, GPTConfig, GPT2_REVISION, HELLASWAG_REVISION, HERE,
                        choose_device, environment, read_json, save_json, set_tf32,
                        sha256, source_hashes, synchronize)


def prepare_hellaswag(directory, limit):
    path = Path(directory) / f"validation_first_{limit}.json"
    if path.exists():
        result = read_json(path)
        if result["revision"] != HELLASWAG_REVISION or result["limit"] != limit:
            raise ValueError("HellaSwag cache provenance mismatch")
        return path
    from datasets import load_dataset
    dataset = load_dataset("Rowan/hellaswag", split="validation", revision=HELLASWAG_REVISION)
    if limit > len(dataset):
        raise ValueError("Requested more HellaSwag examples than validation contains")
    examples = []
    for index in range(limit):
        row = dataset[index]
        examples.append(dict(index=index, ind=int(row["ind"]), ctx=row["ctx"],
                             endings=list(row["endings"]), label=int(row["label"])))
    save_json(path, dict(dataset="Rowan/hellaswag", revision=HELLASWAG_REVISION,
                         split="validation", limit=limit, selection="first N in dataset order",
                         examples=examples))
    print("Prepared HellaSwag:", path, flush=True)
    return path


def render_example(example, encoding, max_length=1024):
    context = encoding.encode_ordinary(example["ctx"])
    endings = [encoding.encode_ordinary(" " + ending) for ending in example["endings"]]
    if not context or len(endings) != 4 or any(not ending for ending in endings):
        raise ValueError("Each example needs context and four nonempty tokenized endings")
    if max(map(len, endings)) >= max_length:
        raise ValueError("An ending alone exceeds the context limit")
    keep = max_length - max(map(len, endings))
    trimmed = len(context) > keep
    context = context[-keep:]
    lengths = [len(context) + len(ending) for ending in endings]
    tokens = torch.zeros((4, max(lengths)), dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, ending in enumerate(endings):
        row = context + ending
        tokens[index, :len(row)] = torch.tensor(row)
        mask[index, len(context):len(row)] = True
    return tokens, mask, trimmed


def completion_scores(logits, tokens, completion_mask):
    """The first ending token is predicted from the last context token.

    Score only ending target positions. All padded positions and context
    losses are excluded. Compare on the common 50,257-ID GPT-2 vocabulary,
    removing our model's 47 implementation-only padding classes.
    """
    targets = tokens[:, 1:]
    mask = completion_mask[:, 1:]
    if logits.shape[:2] != targets.shape or not torch.all(mask.sum(-1) > 0):
        raise ValueError("Invalid shifted shapes or empty completion mask")
    losses = F.cross_entropy(logits[..., :50257].float().reshape(-1, min(50257, logits.shape[-1])),
                             targets.reshape(-1), reduction="none").reshape_as(targets)
    totals = (losses * mask).sum(-1)
    means = totals / mask.sum(-1)
    if not torch.isfinite(totals).all() or not torch.isfinite(means).all():
        raise FloatingPointError("Nonfinite HellaSwag scores")
    return totals, means


def wilson_interval(correct, total):
    z = 1.959963984540054
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


@torch.no_grad()
def evaluate_model(model, examples, encoding, device, provenance, output, resume=True):
    model.eval()
    evidence = dict(provenance, status="running", processed=0, correct_sum=0, correct_norm=0,
                    elapsed_seconds=0.0, predictions=[])
    if resume and output.exists():
        previous = read_json(output)
        if any(previous.get(key) != value for key, value in provenance.items()):
            raise ValueError("Evaluation output has different model/data/scoring provenance")
        evidence = previous
    if evidence["processed"] > len(examples):
        raise ValueError("Saved progress exceeds evaluation selection")
    start, prior_elapsed = time.perf_counter(), evidence["elapsed_seconds"]
    for example in examples[evidence["processed"]:]:
        tokens, mask, trimmed = render_example(example, encoding, model.config.block_size)
        tokens, mask = tokens.to(device), mask.to(device)
        # Both models use FP32 with TF32 disabled, no compile or AMP.
        logits = model(tokens[:, :-1])[0]
        totals, means = completion_scores(logits, tokens, mask)
        predicted_sum, predicted_norm = totals.argmin().item(), means.argmin().item()
        label = int(example["label"])
        if label not in range(4):
            raise ValueError("HellaSwag labels must be 0..3")
        evidence["correct_sum"] += int(predicted_sum == label)
        evidence["correct_norm"] += int(predicted_norm == label)
        evidence["processed"] += 1
        evidence["predictions"].append(dict(index=example["index"], ind=example["ind"], label=label,
                                            predicted_sum=predicted_sum, predicted_norm=predicted_norm,
                                            sum_nll=totals.tolist(), mean_nll=means.tolist(),
                                            context_left_trimmed=trimmed))
        if evidence["processed"] % 50 == 0:
            synchronize(device)
            evidence["elapsed_seconds"] = prior_elapsed + time.perf_counter() - start
            save_json(output, evidence)
            print(f"HellaSwag {provenance['model_name']} | {evidence['processed']}/{len(examples)} "
                  f"| normalized correct {evidence['correct_norm']} "
                  f"| acc {evidence['correct_norm']/evidence['processed']:.4%}", flush=True)
    synchronize(device)
    evidence.update(status="complete", elapsed_seconds=prior_elapsed + time.perf_counter() - start,
                    accuracy=evidence["correct_sum"] / len(examples),
                    accuracy_norm=evidence["correct_norm"] / len(examples),
                    accuracy_norm_wilson95=wilson_interval(evidence["correct_norm"], len(examples)))
    save_json(output, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("prepare", "own", "openai", "both"), default="both")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--cache-dir", type=Path, default=HERE / "data/hellaswag")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/l4")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.limit <= 10042 or args.threads < 1:
        parser.error("Use 1..10042 examples and positive thread count")
    cache = prepare_hellaswag(args.cache_dir, args.limit)
    if args.task == "prepare":
        return
    import tiktoken
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    set_tf32(False)
    examples = read_json(cache)["examples"]
    selection_hash = hashlib.sha256(json.dumps(examples, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    base = dict(environment=environment(device), source_sha256=source_hashes(),
                dataset="Rowan/hellaswag", dataset_revision=HELLASWAG_REVISION,
                selection_sha256=selection_hash, example_count=len(examples),
                selection="first N validation examples in pinned dataset order",
                scoring="context + separately encoded space-prefixed ending; no cleanup; "
                        "ending-only sum NLL and token-mean NLL; common 50257 vocabulary; FP32; TF32 off",
                precision="fp32", tf32=False)
    encoding = tiktoken.get_encoding("gpt2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = ("own", "openai") if args.task == "both" else (args.task,)
    try:
        for task in tasks:
            if task == "own":
                checkpoint_path = args.checkpoint or args.output_dir / "last.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if checkpoint["evidence"]["smoke_only"]:
                    raise ValueError("A tiny wiring checkpoint is not a full GPT-2 benchmark")
                model = GPT(GPTConfig(**checkpoint["model_config"]))
                model.load_state_dict(checkpoint["model"])
                provenance = dict(base, model_name="own", checkpoint_sha256=sha256(checkpoint_path),
                                  training_successful_updates=checkpoint["evidence"]["successful_updates"],
                                  training_source_sha256=checkpoint["evidence"]["provenance"]["source_sha256"])
                del checkpoint
            else:
                model = GPT.from_pretrained(revision=GPT2_REVISION,
                                            cache_dir=HERE / ".cache/gpt2")
                for block in model.transformer.h:
                    block.attn.attn_implementation = "sdpa"
                provenance = dict(base, model_name="openai", model_id="openai-community/gpt2",
                                  model_revision=GPT2_REVISION)
            model = model.to(device).eval()
            evaluate_model(model, examples, encoding, device, provenance,
                           args.output_dir / f"hellaswag_{task}.json")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        try:
            from .report import make_report
        except ImportError:
            from report import make_report
        make_report(args.output_dir)


if __name__ == "__main__":
    main()
