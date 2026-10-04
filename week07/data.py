"""Character tokenization and contiguous Tiny Shakespeare batches."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import torch

SOURCE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).resolve().parent / "data" / "input.txt"


class CharacterTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, indices) -> str:
        return "".join(self.itos[int(index)] for index in indices)

    def __len__(self) -> int:
        return len(self.chars)


def load_data(path: Path = DATA_PATH):
    if not path.exists():
        request = urllib.request.Request(SOURCE, headers={"User-Agent": "neuron-to-gpt-week07"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        # Decode before caching so a failed response cannot become valid input.
        raw.decode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    tokenizer = CharacterTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    boundary = int(0.9 * len(encoded))
    splits = {"train": encoded[:boundary], "val": encoded[boundary:]}
    return tokenizer, splits, hashlib.sha256(raw).hexdigest()


def get_batch(splits, split: str, batch_size: int, block_size: int,
              generator: torch.Generator, device="cpu"):
    data = splits[split]
    if batch_size < 1 or block_size < 1 or len(data) <= block_size:
        raise ValueError("Positive batch/context sizes and at least context+1 tokens are required")
    # The last target is at start + block_size, not start + block_size - 1.
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    tokenizer, splits, digest = load_data()
    print(f"Characters: {sum(len(part) for part in splits.values()):,}")
    print(f"Vocabulary: {len(tokenizer)}")
    print(f"Train: {len(splits['train']):,}; val: {len(splits['val']):,}")
    print(f"SHA256: {digest}")
    x, y = get_batch(splits, "train", 4, 8, torch.Generator().manual_seed(1337))
    print(f"Batch: X={tuple(x.shape)}, Y={tuple(y.shape)}")
    print(f"X[0]: {tokenizer.decode(x[0])!r}")
    print(f"Y[0]: {tokenizer.decode(y[0])!r}")
