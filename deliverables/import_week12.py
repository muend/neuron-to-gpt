"""Validate and import a Week 12 evidence ZIP without executing archived code."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"week10/model.py", "week11/training.py", "week12/common.py",
           "week12/data.py", "week12/evaluate.py", "week12/make_colab_notebook.py",
           "week12/prepare_data.py", "week12/report.py", "week12/test_pipeline.py",
           "week12/train.py", "week12/requirements.txt"}
RESULTS = {"training.json", "data_manifest.json", "hellaswag_own.json",
           "hellaswag_openai.json", "training_curves.png", "submission_report.md",
           "TESLIM_NOTU.txt", "source_manifest.json", "checkpoint_manifest.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def import_results(archive_path):
    expected = RESULTS | {"sources/" + name for name in SOURCES}
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and {row.filename for row in entries} == expected,
                "Unexpected or duplicate ZIP entries")
        require(sum(row.file_size for row in entries) < 150_000_000, "ZIP exceeds evidence size limit")
        for row in entries:
            require(not row.is_dir() and not stat.S_ISLNK(row.external_attr >> 16)
                    and row.file_size < 50_000_000, "Invalid evidence entry")
        raw = {name: archive.read(name) for name in expected}
    sources = json.loads(raw["source_manifest.json"])
    require(set(sources) == SOURCES, "Unexpected source manifest")
    for name, expected_hash in sources.items():
        require(digest(raw["sources/" + name]) == expected_hash, f"Archived source checksum mismatch: {name}")
    local_matches = {name: (ROOT / name).exists() and digest((ROOT / name).read_bytes()) == checksum
                     for name, checksum in sources.items()}
    # A later notebook output-forwarding fix changes the generator but none of
    # the actual model/training/scoring implementation used by this run.
    require(all(match for name, match in local_matches.items()
                if name != "week12/make_colab_notebook.py"), "Core source differs from local implementation")
    training = json.loads(raw["training.json"])
    data = json.loads(raw["data_manifest.json"])
    checkpoint = json.loads(raw["checkpoint_manifest.json"])
    own, reference = [json.loads(raw[f"hellaswag_{name}.json"]) for name in ("own", "openai")]
    require(training["status"] == "complete" and not training["smoke_only"], "Training is not complete/full-size")
    require(training["parameters"] == 124_475_904, "Expected padded GPT-2 124M architecture")
    config = training["model_config"]
    require(all(config[key] == value for key, value in dict(n_layer=12, n_head=12, n_embd=768,
                                                          vocab_size=50304, block_size=1024).items()), "Unexpected architecture")
    require(training["provenance"]["source_sha256"] == sources, "Training source hashes differ")
    require(training["provenance"]["data_manifest_sha256"] == digest(raw["data_manifest.json"]), "Data manifest differs")
    require(training["environment"]["hardware"] == "NVIDIA L4" and training["environment"]["device"] == "cuda", "Expected L4 CUDA run")
    history = training["history"]
    require(len(history) == training["attempted_updates"], "History count differs")
    successes = sum(not row["skipped_update"] for row in history)
    require(successes == training["successful_updates"] == 19586, "Unexpected successful update count")
    require(training["skipped_updates"] == 0 and all(row["finite"] for row in history), "Skipped/nonfinite update")
    effective_tokens = training["effective_batch_tokens"]
    require(effective_tokens == 4096 and len(history) * effective_tokens == training["tokens_seen"], "Token accounting differs")
    for index, row in enumerate(history, 1):
        require(row["step"] == index and row["tokens_seen"] == index * effective_tokens, "Invalid step/token sequence")
        require(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm_before_clip"]), "Nonfinite recorded metric")
        require(math.isclose(row["tokens_per_second"], effective_tokens / (row["milliseconds"] / 1000), rel_tol=1e-8), "Throughput calculation differs")
    require(math.isclose(sum(row["milliseconds"] / 1000 for row in history), training["optimizer_seconds"], rel_tol=1e-8), "Optimizer timing sum differs")
    validation = training["validation"]
    require(validation[0]["step"] == 0 and validation[-1]["step"] == successes, "Missing initial/final validation")
    require(all(row["tokens"] == 32768 and math.isfinite(row["loss"]) for row in validation), "Validation sizes/metrics differ")
    require(data["status"] == "complete" and data["tokenizer"] == "gpt2", "Incomplete/wrong data tokenizer")
    require(not set(data["document_sha256"]["train"]) & set(data["document_sha256"]["val"]), "Train/validation documents overlap")
    for split, count in (("train", 20_000_000), ("val", 131_072)):
        require(data["splits"][split]["tokens"] == count
                and sum(row["tokens"] for row in data["splits"][split]["shards"]) == count, "Shard token accounting differs")
    for result in (own, reference):
        require(result["status"] == "complete" and result["processed"] == result["example_count"] == 1000, "HellaSwag incomplete")
        require(result["source_sha256"] == sources, "Evaluation source differs")
        predictions = result["predictions"]
        require(len(predictions) == 1000 and [row["index"] for row in predictions] == list(range(1000)), "Invalid HellaSwag indices")
        for row in predictions:
            require(row["label"] in range(4) and len(row["mean_nll"]) == len(row["sum_nll"]) == 4, "Invalid prediction record")
            require(all(math.isfinite(value) for value in row["mean_nll"] + row["sum_nll"]), "Nonfinite score")
            require(row["predicted_norm"] == min(range(4), key=lambda i: row["mean_nll"][i])
                    and row["predicted_sum"] == min(range(4), key=lambda i: row["sum_nll"][i]), "Prediction is not minimum NLL")
        for count_key, prediction_key, rate_key in (("correct_norm", "predicted_norm", "accuracy_norm"),
                                                     ("correct_sum", "predicted_sum", "accuracy")):
            count = sum(row[prediction_key] == row["label"] for row in predictions)
            require(count == result[count_key] and math.isclose(result[rate_key], count / 1000), "Accuracy differs from predictions")
    for key in ("selection_sha256", "example_count", "dataset_revision", "scoring", "precision", "tf32"):
        require(own[key] == reference[key], f"Comparison settings differ: {key}")
    require([(r["ind"], r["label"]) for r in own["predictions"]] == [(r["ind"], r["label"]) for r in reference["predictions"]], "Examples/labels differ")
    require(own["correct_norm"] == 276 and reference["correct_norm"] == 336, "Scores differ from reported run")
    require(own["checkpoint_sha256"] == checkpoint["sha256"] and checkpoint["successful_updates"] == successes,
            "Checkpoint identity differs")
    require(raw["training_curves.png"].startswith(b"\x89PNG\r\n\x1a\n"), "Invalid plot")
    target = ROOT / "week12/results/nvidia_l4"
    for name, content in raw.items():
        destination = target / name
        require(not destination.exists() or destination.read_bytes() == content, f"Conflicting imported result: {name}")
    for name, content in raw.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    checks = dict(status="passed", archive_sha256=digest(Path(archive_path).read_bytes()),
                  local_source_matches=local_matches, successful_updates=successes,
                  tokens_seen=training["tokens_seen"], own_correct_norm=276, openai_correct_norm=336,
                  gap_percentage_points=6.0,
                  scope="ZIP integrity, exact archived sources, core local sources, manifest/accounting, "
                        "finite metrics, prediction argmins/counts, matching examples/settings/checkpoint digest. "
                        "No GPU rerun; checkpoint weights and token shard bytes are held separately in Drive.")
    (target / "import_validation.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    delivery = ROOT / "deliverables/hafta12_l4_teslim.zip"
    if delivery.resolve() != Path(archive_path).resolve():
        require(not delivery.exists() or delivery.read_bytes() == Path(archive_path).read_bytes(), "Conflicting delivery ZIP")
        shutil.copyfile(archive_path, delivery)
    (ROOT / "deliverables/hafta12_teslim_notu.txt").write_bytes(raw["TESLIM_NOTU.txt"])
    print(f"Verified/imported {target}")
    print(f"Wall minutes: {training['elapsed_wall_seconds']/60:.4f}; optimizer tok/s: "
          f"{training['tokens_seen']/training['optimizer_seconds']:.2f}; wall tok/s: "
          f"{training['tokens_seen']/training['elapsed_wall_seconds']:.2f}")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path.home() / "Downloads/hafta12_l4_teslim.zip")
    import_results(parser.parse_args().archive)
