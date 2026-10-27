"""Behavioral checks for byte BPE and the independent cl100k_base encoder.

Run from the repository root with:
    python -m unittest discover -s week09 -p "test_*.py"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import tempfile
import unittest

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parent / ".tiktoken_cache"))

import regex
import tiktoken

from tokenizer import BasicTokenizer, GPT4_SPLIT_PATTERN, RegexTokenizer, get_stats, merge
from gpt4_tokenizer import GPT4Tokenizer, recover_merges


def model_with_merges(entries: list[tuple[tuple[int, int], int]]) -> BasicTokenizer:
    """Build a small known hierarchy to expose incorrect encoding strategies."""
    tokenizer = BasicTokenizer()
    for pair, token_id in entries:
        tokenizer.merges[pair] = token_id
        tokenizer.vocab[token_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
    return tokenizer


class ByteBPETests(unittest.TestCase):
    def test_pair_count_overlaps_but_replacement_does_not(self):
        self.assertEqual(get_stats([97] * 5), {(97, 97): 4})
        self.assertEqual(merge([97] * 5, (97, 97), 256), [256, 256, 97])
        self.assertEqual(merge([], (97, 97), 256), [])
        self.assertEqual(merge([97], (97, 97), 256), [97])

    def test_train_ties_follow_first_occurrence(self):
        tokenizer = BasicTokenizer()
        tokenizer.train("abab", 258)
        self.assertEqual(tokenizer.merges, {(97, 98): 256, (256, 256): 257})
        self.assertEqual(tokenizer.vocab[257], b"abab")
        self.assertEqual(tokenizer.encode("ababa"), [257, 97])

    def test_encode_uses_training_rank_not_current_pair_frequency(self):
        tokenizer = model_with_merges([((97, 98), 256), ((98, 99), 257)])
        self.assertEqual(tokenizer.encode("abcbcbc"), [256, 99, 257, 257])

    def test_encode_is_not_longest_vocab_prefix_matching(self):
        tokenizer = model_with_merges([
            ((97, 98), 256), ((99, 100), 257), ((256, 99), 258),
        ])
        self.assertEqual(tokenizer.vocab[258], b"abc")
        # "cd" merges before "ab" can become "abc".
        self.assertEqual(tokenizer.encode("abcd"), [256, 257])

    def test_training_is_repeatable_and_resets_old_merges(self):
        first, second = BasicTokenizer(), BasicTokenizer()
        for tokenizer in (first, second):
            tokenizer.train("abcabc", 259)
        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.encode("abcabc"), [258])
        first.train("xyxy", 257)
        self.assertEqual(first.merges, {(120, 121): 256})
        self.assertEqual(len(first.vocab), 257)

    def test_byte_fallback_roundtrips_unseen_turkish_emoji_and_controls(self):
        text = "Çığ, ılık göl; şüphe. İstanbul 🙂👩🏽‍💻\n\r\t\0 e\u0301 𐍈"
        for tokenizer in (BasicTokenizer(), RegexTokenizer()):
            with self.subTest(tokenizer=type(tokenizer).__name__):
                tokenizer.train("abab", 257)
                ids = tokenizer.encode(text)
                self.assertEqual(tokenizer.decode(ids, errors="strict"), text)
                self.assertEqual(tokenizer.decode_bytes(ids), text.encode("utf-8"))

    def test_decode_joins_bytes_before_utf8_decoding(self):
        tokenizer = BasicTokenizer()
        self.assertEqual(tokenizer.decode([0xC3, 0xA7]), "ç")
        self.assertEqual(tokenizer.decode([0xC3]), "\ufffd")
        self.assertEqual(tokenizer.decode([0xC3], errors="ignore"), "")
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([0xC3], errors="strict")

    def test_unknown_ids_and_invalid_vocab_sizes_are_rejected(self):
        tokenizer = BasicTokenizer()
        for token_id in (-1, 256, 97.0, True):
            with self.subTest(token_id=token_id), self.assertRaises(ValueError):
                tokenizer.decode([token_id])
        for size in (0, 255, 256.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                tokenizer.train("abc", size)

    def test_empty_input_and_exhausted_training_are_explicit(self):
        for tokenizer in (BasicTokenizer(), RegexTokenizer()):
            with self.subTest(tokenizer=type(tokenizer).__name__):
                tokenizer.train("", 256)
                self.assertEqual(tokenizer.encode(""), [])
                self.assertEqual(tokenizer.decode([]), "")
                with self.assertRaises(ValueError):
                    tokenizer.train("a", 257)
                self.assertEqual(len(tokenizer.vocab), 256)

    def test_save_load_preserves_type_pattern_ids_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            for tokenizer in (BasicTokenizer(), RegexTokenizer()):
                with self.subTest(tokenizer=type(tokenizer).__name__):
                    tokenizer.train("abab", 258)
                    path = Path(directory) / f"{type(tokenizer).__name__}.json"
                    tokenizer.save(path)
                    restored = BasicTokenizer.load(path)
                    self.assertIs(type(restored), type(tokenizer))
                    self.assertEqual(restored.merges, tokenizer.merges)
                    self.assertEqual(restored.vocab, tokenizer.vocab)
                    self.assertEqual(restored.encode("ababa Çığ 🙂"), tokenizer.encode("ababa Çığ 🙂"))
                    self.assertEqual(restored.decode(restored.encode("Çığ 🙂")), "Çığ 🙂")
                    if isinstance(tokenizer, RegexTokenizer):
                        self.assertEqual(restored.pattern, tokenizer.pattern)

    def test_load_rejects_forward_reference_and_duplicate_ids(self):
        tokenizer = BasicTokenizer()
        tokenizer.train("abab", 258)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            tokenizer.save(path)
            original = json.loads(path.read_text(encoding="utf-8"))
            for bad_merges in (
                [{"pair": [256, 97], "id": 256}],
                [{"pair": [97, 98], "id": 257}],
                [{"pair": [97, 98], "id": 256}, {"pair": [97, 98], "id": 257}],
            ):
                with self.subTest(merges=bad_merges):
                    changed = dict(original, merges=bad_merges)
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        BasicTokenizer.load(path)


class RegexBPETests(unittest.TestCase):
    def test_apostrophes_expose_english_contraction_boundary(self):
        self.assertEqual(
            RegexTokenizer().split("Ankara'dan İstanbul'da"),
            ["Ankara", "'d", "an", " İstanbul", "'d", "a"],
        )

    def test_regex_training_cannot_merge_across_pretoken_boundaries(self):
        basic, split = BasicTokenizer(), RegexTokenizer()
        for tokenizer in (basic, split):
            tokenizer.train("ab! ab! ab!", 258)
        self.assertIn(b"ab!", basic.vocab.values())
        self.assertNotIn(b"ab!", split.vocab.values())
        self.assertNotIn(b"! ", split.vocab.values())
        self.assertEqual(split.encode("ab!"), [256, ord("!")])

    def test_pretokenization_covers_unicode_and_whitespace_without_loss(self):
        texts = (
            "", "\r\n\n  \t ", "1234567", "İstanbul’dan, Ankara'dan!",
            "e\u0301 \u0307abc 🙂👩🏽‍💻 中文\0\u200d", "  a\n\t\r\n b  ",
        )
        for text in texts:
            with self.subTest(text=text):
                pieces = RegexTokenizer().split(text)
                self.assertEqual("".join(pieces), text)
                self.assertTrue(all(pieces))
                self.assertEqual(pieces, regex.findall(GPT4_SPLIT_PATTERN, text))
        self.assertEqual(RegexTokenizer().split("1234567"), ["123", "456", "7"])

    def test_custom_pattern_cannot_silently_drop_text(self):
        with self.assertRaises(ValueError):
            RegexTokenizer(r"\p{L}+").encode("ab!")
        with self.assertRaises(ValueError):
            RegexTokenizer(r".*?").encode("ab")


class MergeRecoveryTests(unittest.TestCase):
    def test_recovered_parents_use_ranks_instead_of_raw_byte_values(self):
        ranks = {b"abc": 11, b"bc": 10, b"a": 7, b"b": 3, b"c": 1, b"ab": 9}
        self.assertEqual(recover_merges(ranks), {(7, 3): 9, (3, 1): 10, (9, 1): 11})

    def test_inconsistent_rank_metadata_fails_instead_of_guessing_parents(self):
        with self.assertRaises(ValueError):
            recover_merges({b"a": 0, b"b": 1, b"c": 2, b"abc": 3})


class GPT4ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = tiktoken.get_encoding("cl100k_base")
        cls.tokenizer = GPT4Tokenizer()

    def test_exact_ordinary_ids_on_diverse_text(self):
        texts = (
            "", "hello world", "aaaaaaa abcabc abcdefghijklmnopqrstuvwxyz",
            "Ankara'dan İstanbul'da Çığ, ılık göl; şüphe.",
            "I'm we've he'll she'd DON'T can't It's that's YOU'RE",
            "İ I ı i e\u0301 é \u0307abc \u200d\0\x01",
            "1234567890 3.141592 -12 +34 00000000",
            "🙂👩🏽‍💻🏳️‍🌈 中文 日本語 한국어 العربية Русский 𐍈",
            " \t  \n\r\n\n\tend   ",
            "<|endoftext|> <|fim_prefix|> <|fim_middle|> <|fim_suffix|> <|endofprompt|>",
            "def f(x):\n    return x + 1\n# Türkçe\n",
        )
        for text in texts:
            with self.subTest(text=text):
                expected = self.reference.encode_ordinary(text)
                actual = self.tokenizer.encode(text)
                self.assertEqual(actual, expected)
                self.assertEqual(self.tokenizer.decode(actual, errors="strict"), text)

    def test_all_raw_byte_token_ids_decode_consistently(self):
        # Token IDs 0..255 are permuted bytes in cl100k_base, unlike our trained BPE.
        for token_id in range(256):
            with self.subTest(token_id=token_id):
                self.assertEqual(self.tokenizer.decode([token_id]), self.reference.decode([token_id]))

    def test_exact_ids_for_deterministic_mixed_unicode_examples(self):
        rng = random.Random(20261026)
        alphabet = "abcABC'’_  \n\t\r0123456789çğıöşüİé\u0301中文🙂\u200d\0"
        for index in range(40):
            text = "".join(rng.choice(alphabet) for _ in range(80))
            with self.subTest(example=index):
                self.assertEqual(self.tokenizer.encode(text), self.reference.encode_ordinary(text))

    def test_isolated_surrogates_follow_reference_replacement(self):
        for text in ("a\ud800b", "\udc00", "\ud83d\ude42"):
            with self.subTest(text=repr(text)):
                expected = self.reference.encode_ordinary(text)
                self.assertEqual(self.tokenizer.encode(text), expected)
                self.assertEqual(self.tokenizer.decode(expected), self.reference.decode(expected))

    def test_invalid_utf8_replacement_and_strict_policy(self):
        invalid_id = self.reference._mergeable_ranks[b"\xc3"]
        self.assertEqual(self.tokenizer.decode([invalid_id]), "\ufffd")
        with self.assertRaises(UnicodeDecodeError):
            self.tokenizer.decode([invalid_id], errors="strict")


if __name__ == "__main__":
    unittest.main()
