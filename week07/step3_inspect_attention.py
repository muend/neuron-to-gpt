"""Inspect the saved attention model without running training again."""

import argparse
from pathlib import Path

import torch

from data import CharacterTokenizer
from models import SingleHeadLanguageModel
from run_experiments import inspect_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "single_head.pt")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error("Checkpoint not found. Run week07/run_experiments.py first.")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    tokenizer = CharacterTokenizer("".join(checkpoint["chars"]))
    model = SingleHeadLanguageModel(len(tokenizer), config["block_size"],
                                    config["embedding_size"], config["head_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Embedding channels: {config['embedding_size']}; head channels: {config['head_size']}")
    inspect_attention(model, tokenizer, "cpu", config["block_size"])


if __name__ == "__main__":
    main()
