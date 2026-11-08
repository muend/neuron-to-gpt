"""Tokenize document-disjoint splits and read NPY shards across boundaries."""

import bisect
import hashlib
from pathlib import Path

import numpy as np
import torch

try:
    from .common import read_json, save_json, sha256
except ImportError:
    from common import read_json, save_json, sha256


class ShardWriter:
    def __init__(self, directory, split, shard_size):
        if shard_size < 2:
            raise ValueError("shard_size must be at least two")
        self.directory, self.split, self.shard_size = Path(directory), split, shard_size
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.used, self.total, self.shards = 0, 0, []

    def append(self, tokens):
        values = np.asarray(tokens, dtype=np.int64)
        if values.ndim != 1 or not len(values) or values.min() < 0 or values.max() >= 50257:
            raise ValueError("Expected nonempty GPT-2 token IDs in [0,50257)")
        offset = 0
        while offset < len(values):
            count = min(len(values) - offset, self.shard_size - self.used)
            self.buffer[self.used:self.used + count] = values[offset:offset + count]
            self.used += count
            self.total += count
            offset += count
            if self.used == self.shard_size:
                self.flush()

    def flush(self):
        if not self.used:
            return
        filename = f"{self.split}_{len(self.shards):05d}.npy"
        path = self.directory / filename
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, self.buffer[:self.used], allow_pickle=False)
        temporary.replace(path)
        self.shards.append(dict(filename=filename, tokens=self.used, sha256=sha256(path)))
        self.used = 0


def prepare_documents(documents, encoding, directory, *, train_tokens, val_tokens,
                      shard_size=1_000_000, source=None):
    """First unique documents fill val; subsequent unique documents fill train.

    A boundary document is truncated (with EOT) and its remainder discarded,
    never assigned to the other split. Exact stripped-text duplicates are
    removed across both splits. This is not near-duplicate decontamination.
    """
    if min(train_tokens, val_tokens) < 2:
        raise ValueError("Both token budgets must be at least two")
    directory = Path(directory)
    writers = {split: ShardWriter(directory, split, shard_size)
               for split in ("train", "val")}
    budgets = dict(train=train_tokens, val=val_tokens)
    seen, doc_hashes, duplicates = set(), dict(train=[], val=[]), 0
    progress = 0
    for text in documents:
        text = text.strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        split = "val" if writers["val"].total < val_tokens else "train"
        remaining = budgets[split] - writers[split].total
        tokens = encoding.encode_ordinary(text) + [encoding.eot_token]
        if len(tokens) > remaining:
            tokens = tokens[:remaining]
            tokens[-1] = encoding.eot_token
        writers[split].append(tokens)
        doc_hashes[split].append(digest)
        total = sum(writer.total for writer in writers.values())
        if total // 1_000_000 > progress:
            progress = total // 1_000_000
            print(f"Prepared {total:,} tokens | train {writers['train'].total:,} "
                  f"| val {writers['val'].total:,}", flush=True)
        if writers["train"].total >= train_tokens:
            break
    for writer in writers.values():
        writer.flush()
    if any(writers[split].total != budget for split, budget in budgets.items()):
        raise ValueError("Source exhausted before requested budgets; reduce budgets or supply more documents")
    manifest = dict(status="complete", tokenizer="gpt2", eot_token=encoding.eot_token,
                    dtype="uint16", source=source, requested_tokens=budgets,
                    shard_size=shard_size, duplicates_removed=duplicates,
                    document_sha256=doc_hashes,
                    split_policy="document-disjoint; exact-text dedup; first unique docs val, then train; "
                                 "boundary document remainder discarded, EOT retained",
                    splits={split: dict(tokens=writer.total, documents=len(doc_hashes[split]),
                                        shards=writer.shards) for split, writer in writers.items()})
    save_json(directory / "manifest.json", manifest)
    return manifest


class ShardLoader:
    """Memory-map shards and join consecutive ranges without dropping tails.

    Training wraps over the full training stream. Validation is finite and
    never wraps: only complete fixed-shape batches are returned. Its residual
    tail (<B*T+1) is excluded explicitly from that evaluation.
    """
    def __init__(self, directory, split, batch_size, sequence_length, *, cyclic=None):
        if split not in ("train", "val") or min(batch_size, sequence_length) < 1:
            raise ValueError("Invalid split/batch/context")
        directory = Path(directory)
        manifest = read_json(directory / "manifest.json")
        if manifest.get("status") != "complete" or manifest.get("tokenizer") != "gpt2":
            raise ValueError("A complete GPT-2 data manifest is required")
        if set(manifest["document_sha256"]["train"]) & set(manifest["document_sha256"]["val"]):
            raise ValueError("Train/validation documents overlap")
        self.arrays, self.ends, length = [], [], 0
        for entry in manifest["splits"][split]["shards"]:
            filename = entry["filename"]
            if Path(filename).name != filename or "\\" in filename:
                raise ValueError("Invalid shard filename")
            path = directory / filename
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {filename}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (array.ndim != 1 or array.dtype != np.uint16 or len(array) != entry["tokens"]
                    or not len(array) or int(array.max()) >= 50257):
                raise ValueError(f"Invalid token shard: {filename}")
            self.arrays.append(array)
            length += len(array)
            self.ends.append(length)
        if length != manifest["splits"][split]["tokens"] or length < 2:
            raise ValueError("Manifest token count mismatch")
        self.total_tokens = length
        self.batch_size, self.sequence_length = batch_size, sequence_length
        self.cyclic = split == "train" if cyclic is None else cyclic
        self.tokens_seen = 0

    def reset(self):
        self.tokens_seen = 0

    def next_batch(self, device="cpu"):
        count = self.batch_size * self.sequence_length
        if not self.cyclic and self.tokens_seen + count + 1 > self.total_tokens:
            raise StopIteration
        start, remaining, chunks = self.tokens_seen % self.total_tokens, count + 1, []
        while remaining:
            index = bisect.bisect_right(self.ends, start)
            previous = self.ends[index - 1] if index else 0
            available = min(remaining, self.ends[index] - start)
            chunks.append(self.arrays[index][start - previous:start - previous + available])
            remaining -= available
            start = (start + available) % self.total_tokens
        buffer = torch.from_numpy(np.concatenate(chunks).astype(np.int64))
        self.tokens_seen += count
        shape = self.batch_size, self.sequence_length
        return buffer[:-1].reshape(shape).to(device), buffer[1:].reshape(shape).to(device)
