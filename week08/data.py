"""Reproducible character corpora and vectorized, split-safe batching."""

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


class CharacterTokenizer:
    def __init__(self, text=None, *, chars=None):
        self.chars = sorted(set(text)) if chars is None else list(chars)
        if not self.chars or len(set(self.chars)) != len(self.chars):
            raise ValueError("A nonempty, unique character vocabulary is required")
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        return "".join(self.chars[int(i)] for i in indices)

    def __len__(self):
        return len(self.chars)


class Corpus:
    def __init__(self, path, block_size, device="cpu"):
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        self.tokenizer = CharacterTokenizer(text)
        encoded = torch.tensor(self.tokenizer.encode(text), dtype=torch.long, device=device)
        boundary = int(len(encoded) * .9)
        self.splits = {"train": encoded[:boundary], "val": encoded[boundary:]}
        if min(map(len, self.splits.values())) <= block_size:
            raise ValueError("Both splits need more than block_size characters")
        self.block_size = block_size
        self.device = torch.device(device)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.metadata = {"sha256": self.sha256, "bytes": len(raw), "characters": len(text),
                         "vocabulary": len(self.tokenizer),
                         "split_sizes": {k: len(v) for k, v in self.splits.items()}}

    def get_batch(self, split, batch_size, generator):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        data = self.splits[split]
        # Draw starts on CPU so every architecture receives the same windows.
        starts = torch.randint(len(data) - self.block_size, (batch_size,), generator=generator)
        offsets = starts.to(self.device)[:, None] + torch.arange(self.block_size, device=self.device)
        return data[offsets], data[offsets + 1]


def validate_turkish_corpus(directory=None):
    directory = Path(directory) if directory is not None else HERE / "data"
    path, manifest_path = directory / "turkish.txt", directory / "corpus_manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise ValueError("Nazim Hikmet corpus is not ready. Supply source files and run prepare_data.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection_id") != "nazim_hikmet":
        raise ValueError("Rejected Turkish corpus: it is not the user-selected Nazim Hikmet collection")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("turkish_sha256"):
        raise ValueError("Turkish corpus checksum differs from its provenance manifest")
    if not manifest.get("sources") or any(source.get("author_id") != "nazim_hikmet"
                                         for source in manifest["sources"]):
        raise ValueError("Turkish corpus provenance must contain only the selected author's sources")
    return path


def corpus_path(language):
    if language == "tr":
        return validate_turkish_corpus()
    if language != "en":
        raise ValueError("language must be en or tr")
    return HERE / "data" / "shakespeare.txt"


def describe_data():
    result = {}
    for language in ("en", "tr"):
        try:
            path = corpus_path(language)
        except ValueError as error:
            result[language] = {"status": "not_ready", "reason": str(error)}
            continue
        if path.exists():
            result[language] = Corpus(path, 8).metadata
    return result


if __name__ == "__main__":
    print(json.dumps(describe_data(), indent=2))
