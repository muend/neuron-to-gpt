"""Measure all six architectural stages using one fixed training protocol."""

import argparse
import csv
import json
from pathlib import Path

import torch

from config import ModelConfig, STAGES
from data import Corpus, HERE, corpus_path
from train import fit, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.eval_batches) < 1:
        parser.error("Steps and eval batches must be positive")
    torch.set_num_threads(1)
    corpus = Corpus(corpus_path("en"), 32, resolve_device(args.device))
    destination = args.output_root / "ablations"
    results = []
    for stage in STAGES:
        run_dir = destination / stage
        config = ModelConfig(len(corpus.tokenizer), stage=stage, n_embd=64, n_head=4,
                             n_layer=1, block_size=32, dropout=0, initialize_small=False)
        result = fit(config, corpus, run_dir, steps=args.steps, eval_batches=args.eval_batches,
                     resume=args.resume and (run_dir / "last.pt").exists(), sample_tokens=200,
                     precision="none", preset="ablation")
        results.append({"stage": stage, "parameters": result["parameters"],
                        "val_nll": result["final_val_nll"], "training_seconds": result["training_seconds"],
                        "steps": result["steps"]})
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print("\nSIX-STAGE COMPARISON (final validation loss)")
    for row in results:
        print(f"{row['stage']:12s} | {row['parameters']:7,d} params | val={row['val_nll']:.4f} "
              f"| training={row['training_seconds']:.1f}s")
    previous = json.loads((HERE.parent / "week07" / "outputs" / "results.json").read_text())
    print(f"Historical Week 7: bigram={previous['results']['bigram']['val']:.4f}; "
          f"single_head={previous['results']['single_head']['val']:.4f}")
    print("New runs use vectorized batches and clipping; report reruns and historical values separately.")


if __name__ == "__main__":
    main()
