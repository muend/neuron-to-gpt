"""Small independent checks of training semantics; no downloads or GPU needed."""

from copy import deepcopy
from dataclasses import replace
import math
import unittest

import torch

try:
    from .training import (GPT, GPTConfig, DataLoaderLite, configure_optimizer,
                           learning_rate, optimizer_update)
    from .run_experiments import create_model, default_settings
except ImportError:
    from training import (GPT, GPTConfig, DataLoaderLite, configure_optimizer,
                          learning_rate, optimizer_update)
    from run_experiments import create_model, default_settings


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.config = GPTConfig(block_size=16, vocab_size=31, n_layer=2, n_head=2, n_embd=32)

    def test_loader_shift_tail_and_multiple_wraps(self):
        loader = DataLoaderLite(list(range(11)), 2, 3)
        for offset in (0, 6, 12, 18):
            x, y = loader.next_batch()
            self.assertEqual(x.flatten().tolist(), [(offset + i) % 11 for i in range(6)])
            self.assertEqual(y.flatten().tolist(), [(offset + i + 1) % 11 for i in range(6)])
        self.assertEqual(loader.tokens_seen, 24)
        # A corpus shorter than one batch must still return a complete batch.
        x, y = DataLoaderLite([1, 2], 2, 4).next_batch()
        self.assertEqual(x.tolist(), [[1, 2, 1, 2], [1, 2, 1, 2]])
        self.assertEqual(y.tolist(), [[2, 1, 2, 1], [2, 1, 2, 1]])

    def test_accumulated_gradients_and_update_match_one_large_batch(self):
        large = GPT(self.config)
        micro = deepcopy(large)
        first, _ = configure_optimizer(large, torch.device("cpu"))
        second, _ = configure_optimizer(micro, torch.device("cpu"))
        tokens = torch.randint(31, (97,))
        direct = optimizer_update(large, first, DataLoaderLite(tokens, 4, 8), torch.device("cpu"))
        accumulated = optimizer_update(micro, second, DataLoaderLite(tokens, 1, 8),
                                       torch.device("cpu"), accumulation_steps=4)
        self.assertAlmostEqual(direct["loss"], accumulated["loss"], places=6)
        self.assertAlmostEqual(direct["gradient_norm_before_clip"],
                               accumulated["gradient_norm_before_clip"], places=5)
        for (name, a), (_, b) in zip(large.named_parameters(), micro.named_parameters()):
            with self.subTest(name=name):
                # Check the gradients themselves, not only Adam's update signs.
                torch.testing.assert_close(a.grad, b.grad, atol=3e-7, rtol=3e-4)
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-4)
        clipped = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in large.parameters()]))
        self.assertLessEqual(clipped.item(), 1.000001)

    def test_optimizer_partitions_unique_parameters_and_excludes_layernorm_bias(self):
        model = GPT(self.config)
        optimizer, audit = configure_optimizer(model, torch.device("cpu"))
        decayed, other = optimizer.param_groups
        self.assertTrue(all(p.ndim == 2 for p in decayed["params"]))
        self.assertTrue(all(p.ndim < 2 for p in other["params"]))
        all_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
                         model.num_parameters())
        self.assertIn(id(model.transformer.wte.weight), [id(p) for p in decayed["params"]])
        self.assertNotIn(id(model.transformer.ln_f.weight), [id(p) for p in decayed["params"]])
        self.assertEqual(audit["decay_parameters"] + audit["no_decay_parameters"], model.num_parameters())
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))

    def test_schedule_warmup_peak_floor_and_monotonicity(self):
        rates = [learning_rate(step, 4, 20) for step in range(23)]
        self.assertAlmostEqual(rates[0], 6e-4 / 4)
        self.assertAlmostEqual(rates[3], 6e-4)
        self.assertEqual(rates[3], rates[4])
        self.assertAlmostEqual(rates[19], 6e-5)
        self.assertAlmostEqual(rates[22], 6e-5)
        self.assertTrue(all(a <= b for a, b in zip(rates[:3], rates[1:4])))
        self.assertTrue(all(a >= b for a, b in zip(rates[4:-1], rates[5:])))
        for values in ((0, 0, 20), (0, 19, 20), (-1, 4, 20)):
            with self.assertRaises(ValueError):
                learning_rate(*values)

    def test_full_size_counts_and_weight_decay_audit_on_meta(self):
        with torch.device("meta"):
            tied = GPT(GPTConfig())
            untied = GPT(GPTConfig(weight_tying=False))
            padded = GPT(GPTConfig(vocab_size=50304))
        self.assertEqual(tied.num_parameters(), 124_439_808)
        self.assertEqual(untied.num_parameters() - tied.num_parameters(), 38_597_376)
        self.assertEqual(padded.num_parameters(), 124_475_904)
        _, audit = configure_optimizer(tied, torch.device("cpu"))
        self.assertEqual(audit["decay_parameters"], 124_318_464)
        self.assertEqual(audit["no_decay_parameters"], 121_344)

    def test_initialization_scales_both_residual_branches(self):
        model = GPT(replace(self.config, n_embd=128))
        expected = 0.02 / math.sqrt(2 * self.config.n_layer)
        for block in model.transformer.h:
            for layer in (block.attn.c_proj, block.mlp.c_proj):
                self.assertAlmostEqual(layer.weight.std().item(), expected, delta=expected * 0.04)
                self.assertEqual(layer.bias.count_nonzero().item(), 0)
        self.assertAlmostEqual(model.transformer.wte.weight.std().item(), 0.02, delta=0.001)

    def test_vocab_padding_preserves_all_existing_weights_and_tying(self):
        original = create_model(True, default_settings(), torch.device("cpu"))
        padded = create_model(True, dict(default_settings(), padded=True), torch.device("cpu"))
        for name, parameter in original.named_parameters():
            actual = dict(padded.named_parameters())[name]
            if name == "transformer.wte.weight":
                actual = actual[:original.config.vocab_size]
            torch.testing.assert_close(parameter, actual, atol=0, rtol=0)
        self.assertIs(padded.lm_head.weight, padded.transformer.wte.weight)
        self.assertEqual(padded.config.vocab_size, 384)


if __name__ == "__main__":
    unittest.main()
