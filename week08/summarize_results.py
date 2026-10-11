"""Collect measured ablations and language runs into a compact submission report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.output_root.rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path.parent.relative_to(args.output_root), result))
    lines = ["# Week 8 measured results", "", "Final losses are sampled NLL in nats per character.", "",
             "| Run | Status | Parameters | Steps | Final val NLL | Best val NLL | Training seconds | Hardware |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for name, result in rows:
        lines.append(f"| {name.as_posix()} | {result['status']} | {result['parameters']:,} | {result['steps']} "
                     f"| {result['final_val_nll']:.4f} | {result['best_val_nll']:.4f} "
                     f"| {result['training_seconds']:.1f} | {result['hardware']} |")
    # Describe the actual training inputs in the results, not whichever local files exist now.
    corpora = {}
    for name, result in rows:
        corpora[str(name.as_posix())] = result["corpus"]
    lines += ["", "## Training data recorded in each result", "", "```json", json.dumps(corpora, indent=2), "```", "",
              "## Generated text (best validation checkpoints)", ""]
    for name, result in rows:
        sample = args.output_root / name / "sample.txt"
        if sample.exists() and result["preset"] != "ablation":
            lines += [f"### {name.as_posix()} ({result['status']})", "", "```text",
                      sample.read_text(encoding="utf-8"), "```", ""]
    lines += ["## Interpretation limits", "",
              "Do not rank English vs Turkish losses as language quality: vocabularies, character distributions, "
              "genres, cleaning and held-out segments differ. Same-language ablations use fixed windows. "
              "Smoke runs verify wiring only; they are not final training evidence. "
              "Elapsed time is recorded separately in each result.json; training seconds exclude validation and I/O.", ""]
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / "submission_report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)
    print("\n".join(lines[:len(rows) + 6]))


if __name__ == "__main__":
    main()
