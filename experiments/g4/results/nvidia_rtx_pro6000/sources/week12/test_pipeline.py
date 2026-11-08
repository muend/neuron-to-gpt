"""Independent small checks; no network, pretrained weights or GPU required."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

try:
    from .common import GPT, GPTConfig, configure_optimizer, make_scaler, optimizer_update, read_json
    from .data import ShardLoader, prepare_documents
    from .evaluate import completion_scores, evaluate_model, render_example, wilson_interval
    from .train import save_checkpoint, timed_learning_rate, validation_loss
except ImportError:
    from common import GPT, GPTConfig, configure_optimizer, make_scaler, optimizer_update, read_json
    from data import ShardLoader, prepare_documents
    from evaluate import completion_scores, evaluate_model, render_example, wilson_interval
    from train import save_checkpoint, timed_learning_rate, validation_loss


class FakeEncoding:
    eot_token = 50256

    def encode_ordinary(self, text):
        return list(text.encode("utf-8"))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.encoding = FakeEncoding()

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return prepare_documents([f"Unique document {i} with a long enough body." for i in range(20)],
                                 self.encoding, self.directory, train_tokens=23,
                                 val_tokens=9, shard_size=5, source={"test": True})

    def test_shard_boundaries_targets_tail_and_multiple_epochs(self):
        manifest = self.prepare()
        stream = np.concatenate([np.load(self.directory / row["filename"])
                                 for row in manifest["splits"]["train"]["shards"]])
        self.assertEqual(len(manifest["splits"]["val"]["shards"]), 2)
        self.assertEqual(len(stream), 23)
        self.assertEqual(stream[-1], 50256)
        loader = ShardLoader(self.directory, "train", 2, 4)
        for offset in (0, 8, 16, 24, 32, 40):
            x, y = loader.next_batch()
            self.assertEqual(x.flatten().tolist(), [int(stream[(offset + i) % 23]) for i in range(8)])
            self.assertEqual(y.flatten().tolist(), [int(stream[(offset + i + 1) % 23]) for i in range(8)])
        self.assertEqual(loader.tokens_seen, 48)

    def test_validation_is_finite_repeatable_and_does_not_wrap(self):
        self.prepare()
        loader = ShardLoader(self.directory, "val", 1, 4)
        first = loader.next_batch()
        loader.next_batch()
        with self.assertRaises(StopIteration):
            loader.next_batch()
        loader.reset()
        torch.testing.assert_close(first[0], loader.next_batch()[0])
        torch.manual_seed(42)
        model = GPT(GPTConfig(block_size=4, vocab_size=50304, n_layer=1, n_head=1, n_embd=8))
        model.train()
        a = validation_loss(model, loader, torch.device("cpu"), "fp32", 10)
        b = validation_loss(model, loader, torch.device("cpu"), "fp32", 10)
        self.assertEqual(a, b)
        self.assertEqual(a["tokens"], 8)
        self.assertTrue(model.training)

    def test_duplicate_documents_are_removed_before_split_and_boundaries_end_in_eot(self):
        documents = ["duplicate", "duplicate", "another document", "last long document"]
        manifest = prepare_documents(documents, self.encoding, self.directory,
                                     train_tokens=15, val_tokens=10, shard_size=100)
        self.assertEqual(manifest["duplicates_removed"], 1)
        self.assertFalse(set(manifest["document_sha256"]["train"]) & set(manifest["document_sha256"]["val"]))
        expected = hashlib.sha256(b"duplicate").hexdigest()
        self.assertEqual(manifest["document_sha256"]["val"], [expected])
        for split in ("train", "val"):
            self.assertEqual(np.load(self.directory / manifest["splits"][split]["shards"][-1]["filename"])[-1], 50256)

    def test_corrupt_shard_is_rejected_before_training(self):
        manifest = self.prepare()
        path = self.directory / manifest["splits"]["train"]["shards"][0]["filename"]
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "checksum"):
            ShardLoader(self.directory, "train", 1, 4)

    def test_hellaswag_mask_excludes_context_and_padding_and_normalizes_lengths(self):
        # Different lengths cause sum-NLL to favor answer 0, mean-NLL answer 1.
        lengths, nll = [1, 2, 3, 4], [0.5, 0.3, 0.4, 0.6]
        tokens = torch.zeros((4, 6), dtype=torch.long)
        mask = torch.zeros((4, 6), dtype=torch.bool)
        logits = torch.full((4, 5, 3), -10.0)
        for i, (length, loss) in enumerate(zip(lengths, nll)):
            tokens[i, 2:2 + length] = 1
            mask[i, 2:2 + length] = True
            probability = math.exp(-loss)
            distribution = torch.tensor([(1 - probability)/2, probability, (1 - probability)/2])
            logits[i, 1:1 + length] = distribution.log()
        totals, means = completion_scores(logits, tokens, mask)
        torch.testing.assert_close(totals, torch.tensor([a*b for a, b in zip(lengths, nll)]))
        torch.testing.assert_close(means, torch.tensor(nll))
        self.assertEqual(totals.argmin().item(), 0)
        self.assertEqual(means.argmin().item(), 1)
        changed = logits.clone()
        changed[:, 0, :] = torch.tensor([30.0, -30.0, 0.0])
        for index, length in enumerate(lengths):
            changed[index, 1 + length:, :] = 500.0
        actual = completion_scores(changed, tokens, mask)
        torch.testing.assert_close(totals, actual[0])
        torch.testing.assert_close(means, actual[1])

    def test_render_keeps_ending_boundary_and_trims_context_equally(self):
        example = dict(ctx="abcdefghijkl", endings=["x", "yy", "zzz", "wwww"])
        tokens, mask, trimmed = render_example(example, self.encoding, max_length=10)
        self.assertTrue(trimmed)
        self.assertEqual(mask.sum(-1).tolist(), [2, 3, 4, 5])
        for row in range(4):
            self.assertEqual(tokens[row, :5].tolist(), list(b"hijkl"))
            self.assertFalse(mask[row, :5].any())
            self.assertTrue(mask[row, 5])

    def test_evaluation_progress_resume_and_matching_selection(self):
        torch.manual_seed(42)
        model = GPT(GPTConfig(block_size=16, vocab_size=128, n_layer=1, n_head=1, n_embd=8))
        examples = [dict(index=i, ind=i, ctx="hi", endings=["a", "b", "c", "d"], label=i%4) for i in range(3)]
        output = self.directory / "evaluation.json"
        provenance = dict(model_name="fixture", selection_sha256="test")
        first = evaluate_model(model, examples, self.encoding, torch.device("cpu"), provenance, output)
        second = evaluate_model(model, examples, self.encoding, torch.device("cpu"), provenance, output)
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(second["processed"], 3)
        self.assertEqual(read_json(output)["status"], "complete")
        with self.assertRaisesRegex(ValueError, "provenance"):
            evaluate_model(model, examples, self.encoding, torch.device("cpu"),
                           dict(provenance, selection_sha256="different"), output)

    def test_time_schedule_warmup_floor_and_interval(self):
        self.assertAlmostEqual(timed_learning_rate(0, 4, 0, 100), 6e-4/4)
        self.assertAlmostEqual(timed_learning_rate(3, 4, 10, 100), 6e-4)
        self.assertAlmostEqual(timed_learning_rate(4, 4, 10, 100, 10), 6e-4)
        self.assertAlmostEqual(timed_learning_rate(100, 4, 100, 100, 10), 6e-5)
        lower, upper = wilson_interval(250, 1000)
        self.assertLess(lower, 0.25)
        self.assertGreater(upper, 0.25)

    def test_checkpoint_restores_optimizer_loader_and_next_update(self):
        self.prepare()
        device = torch.device("cpu")
        config = GPTConfig(block_size=4, vocab_size=50304, n_layer=1, n_head=1, n_embd=8)
        first = GPT(config)
        optimizer, _ = configure_optimizer(first, device)
        scaler = make_scaler(device, "fp32")
        loader = ShardLoader(self.directory, "train", 1, 4)
        optimizer_update(first, optimizer, loader, device, scaler=scaler)
        path = self.directory / "last.pt"
        save_checkpoint(path, first, optimizer, scaler, loader, {"test": True})
        state = torch.load(path, weights_only=True, map_location="cpu")
        second = GPT(GPTConfig(**state["model_config"]))
        second.load_state_dict(state["model"])
        other_optimizer, _ = configure_optimizer(second, device)
        other_optimizer.load_state_dict(state["optimizer"])
        other_loader = ShardLoader(self.directory, "train", 1, 4)
        other_loader.tokens_seen = state["loader_tokens_seen"]
        a = optimizer_update(first, optimizer, loader, device)
        b = optimizer_update(second, other_optimizer, other_loader, device)
        self.assertEqual(loader.tokens_seen, other_loader.tokens_seen)
        self.assertAlmostEqual(a["loss"], b["loss"], places=7)
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
