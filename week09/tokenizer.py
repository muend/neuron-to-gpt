"""Educational byte-level BPE, trained separately from a language model.

The first 256 IDs represent raw bytes, so new Unicode text needs no unknown
token. Learned IDs concatenate earlier tokens. Encoding always follows the
learned merge order; it does not choose the currently most frequent pair.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import regex


# OpenAI tiktoken_ext/openai_public.py: cl100k_base(). This is the optimized
# equivalent of the GPT-4 pattern used in Karpathy's lecture. "regex", unlike
# Python's "re", supports Unicode properties and possessive quantifiers.
GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)


def get_stats(ids: Iterable[int], counts: dict | None = None) -> dict[tuple[int, int], int]:
    """Count overlapping adjacent pairs, preserving first-appearance order."""
    ids = list(ids) if not isinstance(ids, (list, tuple)) else ids
    stats = Counter(zip(ids, ids[1:]))
    if counts is None:
        return dict(stats)
    for pair, frequency in stats.items():
        counts[pair] = counts.get(pair, 0) + frequency
    return counts


def merge(ids: Iterable[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace nonoverlapping instances of pair, from left to right."""
    ids = list(ids) if not isinstance(ids, (list, tuple)) else ids
    result = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and ids[index] == pair[0] and ids[index + 1] == pair[1]:
            result.append(new_id)
            index += 2
        else:
            result.append(ids[index])
            index += 1
    return result


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class BasicTokenizer:
    """Byte BPE over a complete text, allowing merges across word boundaries."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge_history: list[dict] = []

    def split(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [text] if text else []

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Reset and train until vocab_size is reached, or raise if exhausted.

        Equal frequencies use the first pair encountered in corpus order.
        Regex chunks are counted with multiplicity, so recurring words need
        only one mutable sequence without changing frequencies or tie order.
        Requested size includes the 256 byte tokens.
        """
        if not _is_integer(vocab_size) or vocab_size < 256:
            raise ValueError("vocab_size must be an integer of at least 256")
        chunks = self.split(text)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merge_history = []
        weighted = Counter(tuple(chunk.encode("utf-8")) for chunk in chunks)
        sequences = [(list(chunk), count) for chunk, count in weighted.items()]
        for new_id in range(256, vocab_size):
            counts: dict[tuple[int, int], int] = {}
            for ids, weight in sequences:
                for pair, frequency in get_stats(ids).items():
                    counts[pair] = counts.get(pair, 0) + frequency * weight
            if not counts:
                raise ValueError(
                    f"Corpus exhausted at vocabulary size {len(self.vocab)}; "
                    f"cannot reach requested size {vocab_size}"
                )
            pair = max(counts, key=counts.get)
            frequency = counts[pair]
            sequences = [(merge(ids, pair, new_id), weight) for ids, weight in sequences]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merge_history.append({"id": new_id, "pair": list(pair), "count": frequency})
            if verbose:
                print(f"{new_id}: {pair} -> {self.vocab[new_id]!r}, pair frequency={frequency}")

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            # The smallest learned ID is the earliest training merge. Applying
            # it everywhere first reproduces the training merge hierarchy.
            pairs = get_stats(ids)
            pair = min(pairs, key=lambda candidate: self.merges.get(candidate, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        result = []
        # Cache recurring regex pieces only for this call, avoiding a persistent
        # cache that could retain arbitrary user text or become stale on train.
        cache: dict[str, list[int]] = {}
        for chunk in self.split(text):
            if chunk not in cache:
                cache[chunk] = self._encode_chunk(list(chunk.encode("utf-8")))
            result.extend(cache[chunk])
        return result

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        parts = []
        for token_id in ids:
            if not _is_integer(token_id) or token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id!r}")
            parts.append(self.vocab[token_id])
        return b"".join(parts)

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Concatenate bytes first, then decode; isolated tokens may be invalid.

        Replacement (U+FFFD) is useful for arbitrary generated sequences.
        errors='strict' exposes invalid UTF-8 through UnicodeDecodeError.
        """
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        payload = {
            "format": "week09-byte-bpe",
            "version": 1,
            "type": "regex" if isinstance(self, RegexTokenizer) else "basic",
            "pattern": self.pattern if isinstance(self, RegexTokenizer) else None,
            "merges": [
                {"pair": list(pair), "id": token_id}
                for pair, token_id in self.merges.items()
            ],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BasicTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tokenizer model must be a JSON object")
        if payload.get("format") != "week09-byte-bpe" or payload.get("version") != 1:
            raise ValueError("Unsupported tokenizer model format or version")
        kind = payload.get("type")
        if kind == "basic" and payload.get("pattern") is None:
            tokenizer = BasicTokenizer()
        elif kind == "regex" and isinstance(payload.get("pattern"), str):
            tokenizer = RegexTokenizer(payload["pattern"])
        else:
            raise ValueError("Invalid tokenizer type or pattern")
        if cls is not BasicTokenizer and not isinstance(tokenizer, cls):
            raise ValueError(f"Model type {kind!r} does not match {cls.__name__}")
        entries = payload.get("merges")
        if not isinstance(entries, list):
            raise ValueError("Model merges must be a list")
        for expected_id, entry in enumerate(entries, start=256):
            if not isinstance(entry, dict):
                raise ValueError("Each merge must be an object")
            pair = entry.get("pair")
            token_id = entry.get("id")
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not _is_integer(value) or value not in tokenizer.vocab for value in pair)
                or not _is_integer(token_id)
                or token_id != expected_id
                or tuple(pair) in tokenizer.merges
            ):
                raise ValueError(f"Invalid merge at expected token ID {expected_id}")
            left, right = pair
            tokenizer.merges[(left, right)] = token_id
            tokenizer.vocab[token_id] = tokenizer.vocab[left] + tokenizer.vocab[right]
        return tokenizer


class RegexTokenizer(BasicTokenizer):
    """BPE within pretokenized chunks, never across regex boundaries."""

    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN) -> None:
        super().__init__()
        self.pattern = pattern
        self._compiled_pattern = regex.compile(pattern)

    def split(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        chunks = []
        position = 0
        for match in self._compiled_pattern.finditer(text):
            if match.start() != position or match.end() == match.start():
                raise ValueError("Regex pattern must cover text without gaps or empty matches")
            chunks.append(match.group())
            position = match.end()
        if position != len(text):
            raise ValueError("Regex pattern does not cover the entire text")
        return chunks
