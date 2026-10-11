"""Validate and import the owner's Colab export without importing raw corpora."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from PIL import Image
import torch

from config import ModelConfig, PRESETS
from data import HERE, Corpus, corpus_path
from models import make_model

RUNS = ("en_a100_layernorm", "tr_a100_layernorm")
ASSETS = ("history.json", "loss_curve.png", "result.json", "sample.txt")
EXPECTED = {f"{run}/{asset}" for run in RUNS for asset in ASSETS} | {
    "submission_report.md", "corpus_manifest.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_archive(path, check_local=True):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(EXPECTED) and {i.filename for i in entries} == EXPECTED,
                "Unexpected, missing or duplicate archive entries; review before importing")
        require(sum(i.file_size for i in entries) <= 20_000_000, "Export unexpectedly exceeds 20 MB")
        require(all(i.file_size <= 5_000_000 for i in entries), "Export member unexpectedly exceeds 5 MB")
        require(archive.testzip() is None, "ZIP integrity check failed")
        raw = {name: archive.read(name) for name in EXPECTED}
    manifest = json.loads(raw["corpus_manifest.json"])
    require(manifest.get("selection_id") == "nazim_hikmet", "Turkish collection is not the selected author")
    require(manifest.get("sources") and all(s.get("author_id") == "nazim_hikmet" for s in manifest["sources"]),
            "Unexpected Turkish source authors")
    checks, records = [], {}
    for run in RUNS:
        result = json.loads(raw[f"{run}/result.json"])
        history = json.loads(raw[f"{run}/history.json"])
        require(result["status"] == "measured" and result["preset"] == "a100", f"{run}: not a full A100 run")
        require(result["steps"] == result["optimizer_updates"] == 5000 and result["skipped_updates"] == 0,
                f"{run}: incomplete or skipped optimizer updates")
        require(result["device"] == "cuda" and "A100" in result["hardware"], f"{run}: inconsistent GPU metadata")
        require(result["training_config"]["precision"] == "bf16", f"{run}: unexpected precision")
        require(result["eval_batches"] == result["training_config"]["eval_batches"] == 200,
                f"{run}: inconsistent evaluation budget")
        expected_training = {"batch_size": 64, "learning_rate": .0003, "seed": 1337,
                             "precision": "bf16", "eval_batches": 200}
        require(result["training_config"] == expected_training, f"{run}: changed training settings")
        config = ModelConfig(**result["model_config"])
        preset = PRESETS["a100"]
        for key in ("n_embd", "n_head", "n_layer", "block_size", "dropout", "initialize_small"):
            require(getattr(config, key) == preset[key], f"{run}: mismatched {key}")
        count = sum(p.numel() for p in make_model(config).parameters())
        require(count == result["parameters"], f"{run}: parameter count does not match the code")
        require([row["step"] for row in history] == list(range(500, 5001, 500)), f"{run}: history has missing steps")
        require(all(math.isfinite(row[key]) and row[key] >= 0 for row in history for key in ("train", "val")),
                f"{run}: nonfinite or negative history loss")
        for field, key in (("final_train_nll", "train"), ("final_val_nll", "val")):
            require(math.isclose(result[field], history[-1][key], rel_tol=0, abs_tol=1e-12),
                    f"{run}: final loss differs from history")
        best = min(history, key=lambda row: row["val"])
        require(math.isclose(result["best_val_nll"], best["val"], rel_tol=0, abs_tol=1e-12),
                f"{run}: best loss differs from history")
        require(result["sample_checkpoint_step"] == best["step"], f"{run}: sample is not labeled with the best step")
        require(0 < result["training_seconds"] <= result["elapsed_seconds"] and math.isfinite(result["elapsed_seconds"]),
                f"{run}: invalid elapsed/training timing")
        meta = result["corpus"]
        require(config.vocab_size == meta["vocabulary"], f"{run}: vocabulary mismatch")
        require(meta["split_sizes"]["train"] == int(meta["characters"] * .9) and
                sum(meta["split_sizes"].values()) == meta["characters"], f"{run}: invalid 90/10 split")
        sample = raw[f"{run}/sample.txt"].decode("utf-8")
        require(len(sample) == 601, f"{run}: unexpected generated sample length")
        with Image.open(io.BytesIO(raw[f"{run}/loss_curve.png"])) as picture:
            require(picture.format == "PNG", f"{run}: invalid plot format")
            picture.verify()
        if check_local:
            corpus = Corpus(corpus_path(run[:2]), config.block_size)
            require(corpus.metadata == meta, f"{run}: local corpus checksum/metadata differ")
            require(set(sample).issubset(corpus.tokenizer.chars), f"{run}: sample contains unknown characters")
        records[run] = result
        checks.append({"run": run, "parameter_count": count, "history_checkpoints": len(history),
                       "best_step": best["step"], "corpus_sha256": meta["sha256"],
                       "local_corpus_verified": check_local})
    tr = records["tr_a100_layernorm"]["corpus"]
    require(manifest["turkish_sha256"] == tr["sha256"] and manifest["bytes"] == tr["bytes"],
            "Turkish provenance differs from the trained corpus")
    left, right = (records[run] for run in RUNS)
    require({k: v for k, v in left["model_config"].items() if k != "vocab_size"} ==
            {k: v for k, v in right["model_config"].items() if k != "vocab_size"},
            "Language model architectures differ beyond vocabulary size")
    report = {"assessment": "share_with_noted_caveats", "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "verified": checks, "total_training_seconds": sum(r["training_seconds"] for r in records.values()),
              "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in records.values()),
              "caveats": ["Metrics are user-returned Colab results, not independently rerun GPU evaluation.",
                          "No checkpoints were included in the export; loss and sample claims were checked against metadata/history.",
                          "Loss plots must also be visually inspected against the numerical histories.",
                          "English and Turkish NLL values do not directly rank language quality.",
                          "The scaled GPU run changes model, context, iterations and precision relative to CPU ablations.",
                          "Turkish OCR noise and the contiguous genre/period split remain limitations.",
                          "Elapsed seconds exclude generation/plotting and preprocessing; they are not the whole Colab session."]}
    return raw, report


def import_archive(path, destination, check_local=True):
    raw, report = validate_archive(path, check_local)
    # Do not publish the private provenance manifest. Keep only result assets and the report.
    outputs = {name: content for name, content in raw.items() if name != "corpus_manifest.json"}
    outputs["validation.json"] = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    destination = Path(destination)
    # Preflight every target before writing anything; identical re-imports are safe.
    for name, content in outputs.items():
        target = destination / name
        require(not target.exists() or target.read_bytes() == content, f"Existing result differs: {target}")
    for name, content in outputs.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=HERE / "results/gpu")
    parser.add_argument("--skip-local-corpus-check", action="store_true", help="Validate metadata only when private corpora are unavailable")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = import_archive(args.archive, args.destination, not args.skip_local_corpus_check)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
