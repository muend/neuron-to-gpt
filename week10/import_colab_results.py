"""Import only reviewed metric/plot artifacts from this week's Colab ZIP."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

HERE = Path(__file__).resolve().parent
EXPECTED = {
    "pretrained_metrics.json", "overfit_metrics.json", "weights.tsv",
    "position_embeddings.png", "overfit_loss.png", "sample_en.txt", "sample_tr.txt",
    "submission_report.md", "source_manifest.json",
}


def import_results(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != len(EXPECTED):
            raise ValueError("Expected exactly the nine exported Week 10 result files")
        data = {}
        run_names = set()
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if (len(name.parts) != 2 or name.parts[0] not in {"a100", "t4"}
                    or name.name not in EXPECTED or "\\" in entry.filename
                    or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.file_size > 5_000_000):
                raise ValueError(f"Unexpected result entry: {entry.filename}")
            if name.name in data:
                raise ValueError("Duplicate result filename")
            data[name.name] = archive.read(entry)
            run_names.add(name.parts[0])
    if len(run_names) != 1 or set(data) != EXPECTED:
        raise ValueError("ZIP must contain one complete GPU run")
    run_name = run_names.pop()
    manifest = json.loads(data["source_manifest.json"])
    expected_sources = {"model.py", "run_experiments.py", "test_gpt2.py", "requirements.txt"}
    if set(manifest.get("source_hashes", {})) != expected_sources:
        raise ValueError("Missing source hashes in the Colab manifest")
    for name, digest in manifest["source_hashes"].items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Colab source differs from local source: {name}")
    pretrained = json.loads(data["pretrained_metrics.json"])
    overfit = json.loads(data["overfit_metrics.json"])
    if (pretrained.get("status") != "completed"
            or pretrained.get("unique_parameters") != 124439808
            or not pretrained.get("logit_parity", {}).get("all_passed")
            or overfit.get("status") != "completed"
            or overfit.get("evidence") != "full_gpt2_124m"
            or overfit.get("parameters") != 124439808
            or not overfit.get("overfit_passed")
            or overfit.get("steps_requested") != 300
            or overfit.get("steps_completed") != 300
            or overfit.get("final_loss", 1) >= 0.1
            or overfit.get("final_token_accuracy", 0) <= 0.99):
        raise ValueError("ZIP does not establish all required completed 124M experiments")
    for metrics in (pretrained, overfit):
        hardware = metrics.get("environment", {}).get("hardware", "")
        if ("A100" if run_name == "a100" else "T4") not in hardware:
            raise ValueError("GPU folder does not match recorded hardware")
    if len(data["position_embeddings.png"]) < 100 or len(data["overfit_loss.png"]) < 100:
        raise ValueError("Missing plot content")
    target = Path(destination).resolve() / run_name
    for name, raw in data.items():
        path = target / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Conflicting result already exists: {path}")
    target.mkdir(parents=True, exist_ok=True)
    for name, raw in data.items():
        (target / name).write_bytes(raw)
    print(f"Imported {run_name}: 124M, 300 iterations, final NLL {overfit['final_loss']:.6f}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", default=HERE / "results", type=Path)
    args = parser.parse_args()
    import_results(args.archive, args.destination)
