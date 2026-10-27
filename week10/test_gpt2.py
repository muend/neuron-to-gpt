"""Behavioral GPT-2 checks; all models here are tiny or use meta tensors.

No pretrained weights, internet connection, GPU or 124M-parameter training run
is required. Run from the repository root:
    python -m unittest discover -s week10 -p "test_*.py"
"""

from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from .model import GPT, GPTConfig
except ImportError:
    from model import GPT, GPTConfig


def tiny_config(**overrides):
    settings = dict(block_size=12, vocab_size=67, n_layer=2, n_head=4,
                    n_embd=32, dropout=0.0)
    settings.update(overrides)
    return GPTConfig(**settings)


def hf_reference(config):
    """Construct an independent HF model locally; never call from_pretrained."""
    reference_config = GPT2Config(
        vocab_size=config.vocab_size, n_positions=config.block_size,
        n_ctx=config.block_size, n_layer=config.n_layer, n_head=config.n_head,
        n_embd=config.n_embd, resid_pdrop=config.dropout,
        embd_pdrop=config.dropout, attn_pdrop=config.dropout,
        activation_function="gelu_new", layer_norm_epsilon=1e-5,
        tie_word_embeddings=config.weight_tying, use_cache=False,
    )
    reference_config._attn_implementation = "eager"
    return GPT2LMHeadModel(reference_config).eval()


class GPTArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(1337)

    def test_gpt2_unique_parameter_count_and_tying_without_allocation(self):
        with torch.device("meta"):
            tied = GPT(GPTConfig())
            untied = GPT(GPTConfig(weight_tying=False))
        self.assertEqual(tied.num_parameters(), 124_439_808)
        self.assertEqual(untied.num_parameters(), 163_037_184)
        self.assertEqual(untied.num_parameters() - tied.num_parameters(), 50_257 * 768)
        self.assertIs(tied.lm_head.weight, tied.transformer.wte.weight)
        self.assertIsNot(untied.lm_head.weight, untied.transformer.wte.weight)

    def test_forward_shapes_and_loss_target_alignment(self):
        model = GPT(tiny_config()).eval()
        stream = torch.randint(67, (2, 10))
        x, targets = stream[:, :-1], stream[:, 1:]
        logits, loss = model(x, targets)
        self.assertEqual(tuple(logits.shape), (2, 9, 67))
        self.assertIsNone(model(x)[1])
        expected = F.cross_entropy(logits.reshape(-1, 67), targets.reshape(-1))
        torch.testing.assert_close(loss, expected)
        self.assertTrue(torch.isfinite(loss))

    def test_future_tokens_do_not_change_prefix_logits(self):
        model = GPT(tiny_config()).eval()
        original = torch.randint(67, (2, 9))
        changed = original.clone()
        changed[:, 4:] = (changed[:, 4:] + 17) % 67
        with torch.no_grad():
            before = model(original)[0]
            after = model(changed)[0]
        torch.testing.assert_close(before[:, :4], after[:, :4], atol=1e-6, rtol=1e-6)
        self.assertGreater((before[:, 4:] - after[:, 4:]).abs().max().item(), 1e-4)

    def test_all_unique_parameters_receive_finite_gradients(self):
        model = GPT(tiny_config())
        x, targets = torch.randint(67, (2, 8)), torch.randint(67, (2, 8))
        model(x, targets)[1].backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.transformer.wte.weight.grad.abs().sum().item(), 0)

    def test_sdpa_and_explicit_attention_match_forward_and_gradients(self):
        config = tiny_config()
        eager = GPT(config)
        sdpa = GPT(replace(config, attn_implementation="sdpa"))
        sdpa.load_state_dict(eager.state_dict())
        x, targets = torch.randint(67, (2, 8)), torch.randint(67, (2, 8))
        first_logits, first_loss = eager(x, targets)
        second_logits, second_loss = sdpa(x, targets)
        torch.testing.assert_close(first_logits, second_logits, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(first_loss, second_loss, atol=1e-6, rtol=1e-6)
        first_loss.backward()
        second_loss.backward()
        for (name, first), (_, second) in zip(eager.named_parameters(), sdpa.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(first.grad, second.grad, atol=3e-6, rtol=3e-4)

    def test_initial_loss_is_near_uniform_prediction_for_small_logits(self):
        config = tiny_config()
        model = GPT(config).eval()
        x, targets = torch.randint(67, (4, 12)), torch.randint(67, (4, 12))
        loss = model(x, targets)[1].item()
        self.assertLess(abs(loss - math.log(config.vocab_size)), 0.15)

    def test_tiny_model_can_memorize_one_fixed_batch(self):
        # This is a graph/optimizer smoke test, not evidence of 124M training.
        model = GPT(tiny_config(block_size=8, vocab_size=31, n_layer=1,
                                n_head=2, n_embd=24))
        stream = torch.arange(1, 10).unsqueeze(0)
        x, targets = stream[:, :-1], stream[:, 1:]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0)
        initial = model(x, targets)[1].item()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, targets)[1]
            loss.backward()
            optimizer.step()
        final = model(x, targets)[1].item()
        self.assertLess(final, 0.08)
        self.assertLess(final, initial / 20)

    def test_invalid_configuration_fails_before_model_construction(self):
        cases = (
            {"block_size": 0}, {"vocab_size": -1}, {"n_layer": True},
            {"n_head": 0}, {"n_embd": 31}, {"dropout": -0.1},
            {"dropout": 1.0}, {"dropout": float("nan")},
            {"weight_tying": 1}, {"attn_implementation": "unknown"},
        )
        for kwargs in cases:
            with self.subTest(config=kwargs), self.assertRaises((ValueError, TypeError)):
                tiny_config(**kwargs)

    def test_invalid_input_dimensions_dtypes_and_target_shapes_are_rejected(self):
        model = GPT(tiny_config())
        for x in (torch.zeros(3, dtype=torch.long), torch.zeros((0, 2), dtype=torch.long),
                  torch.zeros((1, 0), dtype=torch.long), torch.zeros((1, 13), dtype=torch.long),
                  torch.zeros((1, 2), dtype=torch.float32)):
            with self.subTest(shape=tuple(x.shape), dtype=x.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    model(x)
        with self.assertRaises(ValueError):
            model(torch.ones((1, 4), dtype=torch.long), torch.ones((1, 3), dtype=torch.long))


class HFWeightImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(20261026)
        self.config = tiny_config()
        self.reference = hf_reference(self.config)

    def test_import_matches_independent_hf_logits_and_shifted_language_loss(self):
        model = GPT(self.config).eval()
        model.load_hf_weights(self.reference)
        stream = torch.randint(67, (2, 10))
        with torch.no_grad():
            our_logits, our_loss = model(stream[:, :-1], stream[:, 1:])
            # HF applies the shift internally; our forward uses explicit targets.
            hf_output = self.reference(stream, labels=stream, use_cache=False)
        torch.testing.assert_close(our_logits, hf_output.logits[:, :-1], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(our_loss, hf_output.loss, atol=2e-6, rtol=2e-6)

    def test_every_conv1d_projection_is_transposed_including_square_matrix(self):
        model = GPT(self.config)
        audit = model.load_hf_weights(self.reference)
        self.assertEqual(len(audit["transposed_tensors"]), 4 * self.config.n_layer)
        for our_block, hf_block in zip(model.transformer.h, self.reference.transformer.h):
            for our_layer, hf_layer in (
                (our_block.attn.c_attn, hf_block.attn.c_attn),
                (our_block.attn.c_proj, hf_block.attn.c_proj),
                (our_block.mlp.c_fc, hf_block.mlp.c_fc),
                (our_block.mlp.c_proj, hf_block.mlp.c_proj),
            ):
                torch.testing.assert_close(our_layer.weight, hf_layer.weight.t(), atol=0, rtol=0)
                torch.testing.assert_close(our_layer.bias, hf_layer.bias, atol=0, rtol=0)
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)

    def test_incompatible_configuration_is_rejected(self):
        model = GPT(tiny_config(n_head=2))
        with self.assertRaises(ValueError):
            model.load_hf_weights(self.reference)

    def test_bad_state_shape_is_rejected_without_partial_copy(self):
        model = GPT(self.config)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        bad = dict(self.reference.state_dict())
        bad["transformer.h.1.mlp.c_fc.weight"] = bad["transformer.h.1.mlp.c_fc.weight"][:-1]
        malformed = SimpleNamespace(config=self.reference.config, state_dict=lambda: bad)
        with self.assertRaises(ValueError):
            model.load_hf_weights(malformed)
        for name, actual in model.state_dict().items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual, before[name], atol=0, rtol=0)

    def test_unexpected_state_keys_and_distinct_tied_weights_are_rejected(self):
        for kind in ("unexpected_key", "distinct_embeddings"):
            with self.subTest(kind=kind):
                bad = dict(self.reference.state_dict())
                if kind == "unexpected_key":
                    bad["surprise.weight"] = torch.ones(1)
                else:
                    bad["lm_head.weight"] = bad["lm_head.weight"].clone() + 1
                malformed = SimpleNamespace(config=self.reference.config, state_dict=lambda: bad)
                with self.assertRaises(ValueError):
                    GPT(self.config).load_hf_weights(malformed)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.model = GPT(tiny_config(block_size=5, n_layer=1)).eval()
        self.prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    def test_top_k_one_matches_greedy_generation_with_context_cropping(self):
        expected = self.prompt.clone()
        with torch.no_grad():
            for _ in range(8):
                next_id = self.model(expected[:, -5:])[0][:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, next_id), dim=1)
        actual = self.model.generate(self.prompt, max_new_tokens=8, top_k=1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[:, :3], self.prompt))

    def test_seeded_sampling_is_repeatable_and_remains_inside_top_k(self):
        first = self.model.generate(self.prompt, 8, temperature=0.8, top_k=4,
                                    generator=torch.Generator().manual_seed(123))
        second = self.model.generate(self.prompt, 8, temperature=0.8, top_k=4,
                                     generator=torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(first, second))
        with torch.no_grad():
            for index in range(self.prompt.shape[1], first.shape[1]):
                logits = self.model(first[:, max(0, index - 5):index])[0][:, -1]
                allowed = logits.topk(4, dim=-1).indices[0].tolist()
                self.assertIn(first[0, index].item(), allowed)

    def test_generation_restores_training_mode_and_zero_steps_preserves_prompt(self):
        self.model.train()
        generated = self.model.generate(self.prompt, 2, top_k=1)
        self.assertTrue(self.model.training)
        self.assertEqual(generated.shape[1], 5)
        self.assertTrue(torch.equal(self.model.generate(self.prompt, 0), self.prompt))

    def test_invalid_sampling_parameters_are_rejected(self):
        for kwargs in ({"max_new_tokens": -1}, {"max_new_tokens": True},
                       {"max_new_tokens": 1, "temperature": 0},
                       {"max_new_tokens": 1, "temperature": float("inf")},
                       {"max_new_tokens": 1, "top_k": 0},
                       {"max_new_tokens": 1, "top_k": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.model.generate(self.prompt, **kwargs)


if __name__ == "__main__":
    unittest.main()
