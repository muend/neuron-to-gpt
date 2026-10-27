"""A Python implementation of the assignment's pretrained cl100k_base BPE.

Only construction uses tiktoken: its learned byte strings and ranks are loaded
from ``Encoding._mergeable_ranks``. That attribute is private and has no public
replacement for reconstructing merge parents; incompatible metadata raises an
explicit error. Encoding and decoding below run our own Python implementation.

``cl100k_base`` is the encoding requested by the historical GPT-4 exercise. It
is not a claim about the tokenizer used by every current OpenAI model.

References for merge recovery and the single-byte rank permutation:
https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py
https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache

import regex

try:
    from .tokenizer import GPT4_SPLIT_PATTERN, get_stats, merge
except ImportError:  # Allow: python week09/run_assignment.py
    from tokenizer import GPT4_SPLIT_PATTERN, get_stats, merge


def _split_at_rank(
    token: bytes, ranks: Mapping[bytes, int], stop_rank: int
) -> list[bytes]:
    """Apply older merges only, leaving a token's two original parents."""
    parts = [bytes([value]) for value in token]
    while len(parts) > 1:
        best_index = None
        best_rank = stop_rank
        for index, (left, right) in enumerate(zip(parts, parts[1:])):
            rank = ranks.get(left + right)
            if rank is not None and rank < best_rank:
                best_index, best_rank = index, rank
        if best_index is None:
            return parts
        parts[best_index : best_index + 2] = [
            parts[best_index] + parts[best_index + 1]
        ]
    return parts


def recover_merges(ranks: Mapping[bytes, int]) -> dict[tuple[int, int], int]:
    """Recover token-ID pairs without treating raw byte values as token IDs."""
    merges = {}
    for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
        if len(token) == 1:
            continue
        parents = _split_at_rank(token, ranks, rank)
        if len(parents) != 2:
            raise ValueError(f"Cannot recover two merge parents for rank {rank}")
        left, right = (ranks[parent] for parent in parents)
        if max(left, right) >= rank:
            raise ValueError(f"Merge rank {rank} does not follow its parents")
        merges[left, right] = rank
    return merges


@lru_cache(maxsize=1)
def _load_cl100k_metadata() -> tuple[
    dict[bytes, int], dict[tuple[int, int], int], dict[str, int]
]:
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError("Install week09/requirements.txt to load cl100k_base") from exc

    reference = tiktoken.get_encoding("cl100k_base")
    ranks = getattr(reference, "_mergeable_ranks", None)
    special_tokens = getattr(reference, "_special_tokens", None)
    if not isinstance(ranks, dict) or not isinstance(special_tokens, dict):
        raise RuntimeError("This tiktoken version does not expose the expected rank metadata")
    if (
        len(ranks) != 100256
        or any(not isinstance(token, bytes) or not token for token in ranks)
        or set(ranks.values()) != set(range(100256))
        or {ranks.get(bytes([value])) for value in range(256)} != set(range(256))
    ):
        raise RuntimeError("Unexpected cl100k_base ranks or single-byte permutation")
    ranks = dict(ranks)
    return ranks, recover_merges(ranks), dict(special_tokens)


class GPT4Tokenizer:
    """Pretrained cl100k_base with our own regex and byte-pair merging.

    ``encode`` and ``encode_ordinary`` both treat special-token spellings as
    literal text, matching tiktoken's ``encode_ordinary``. This differs from
    tiktoken's default ``encode``, which rejects those spellings. Recognition of
    special strings is deliberately outside the ordinary-text assignment.
    Decoding accepts both mergeable and the known special-token IDs.
    """

    encoding_name = "cl100k_base"
    pattern = GPT4_SPLIT_PATTERN

    def __init__(self) -> None:
        ranks, merges, special_tokens = _load_cl100k_metadata()
        self.merges = dict(merges)
        # Vocab stores original UTF-8 bytes, so decoding implicitly reverses
        # the raw-byte permutation used when initializing each encoded chunk.
        self.vocab = {rank: token for token, rank in ranks.items()}
        self.byte_shuffle = {value: ranks[bytes([value])] for value in range(256)}
        self.inverse_byte_shuffle = {rank: value for value, rank in self.byte_shuffle.items()}
        self.special_tokens = dict(special_tokens)
        self._special_bytes = {
            token_id: spelling.encode("utf-8") for spelling, token_id in special_tokens.items()
        }
        self._compiled_pattern = regex.compile(self.pattern)

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        ids = [self.byte_shuffle[value] for value in chunk]
        while len(ids) > 1:
            pairs = get_stats(ids)
            pair = min(pairs, key=lambda candidate: self.merges.get(candidate, float("inf")))
            token_id = self.merges.get(pair)
            if token_id is None:
                break
            ids = merge(ids, pair, token_id)
        return ids

    def encode(self, text: str) -> list[int]:
        """Return ordinary-text token IDs without calling tiktoken's encoder."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # Like tiktoken, repair isolated surrogate code points before UTF-8.
        # Valid surrogate pairs become their scalar value; lone ones become U+FFFD.
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        result = []
        for match in self._compiled_pattern.finditer(text):
            result.extend(self._encode_chunk(match.group().encode("utf-8")))
        return result

    def encode_ordinary(self, text: str) -> list[int]:
        return self.encode(text)

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Join token bytes before UTF-8 decoding; incomplete sequences replace by default."""
        parts = []
        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            token_bytes = self.vocab.get(token_id)
            if token_bytes is None:
                token_bytes = self._special_bytes.get(token_id)
            if token_bytes is None:
                raise ValueError(f"Unknown cl100k_base token ID: {token_id}")
            parts.append(token_bytes)
        return b"".join(parts).decode("utf-8", errors=errors)
