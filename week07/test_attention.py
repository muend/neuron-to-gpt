"""Small deterministic checks for data, attention, gradients and generation."""

import unittest

import torch

from data import CharacterTokenizer, get_batch
from models import AttentionHead, BigramLanguageModel, SingleHeadLanguageModel
from step2_attention_math import causal_averages


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_tokenizer_roundtrip(self):
        text = "Ab a!\n"
        tokenizer = CharacterTokenizer(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.chars, sorted(set(text)))

    def test_shifted_batches_and_split(self):
        splits = {"train": torch.arange(100), "val": torch.arange(100, 150)}
        for split in splits:
            x, y = get_batch(splits, split, 16, 8, torch.Generator().manual_seed(42))
            self.assertEqual(tuple(x.shape), (16, 8))
            self.assertTrue(torch.equal(y, x + 1))
            self.assertGreaterEqual(x.min().item(), splits[split].min().item())
            self.assertLessEqual(y.max().item(), splits[split].max().item())

    def test_three_averages(self):
        x = torch.randn(4, 8, 3)
        loop, multiplied, softmax_average, matrix, softmax_weights = causal_averages(x)
        self.assertTrue(torch.allclose(loop, multiplied, atol=1e-6))
        self.assertTrue(torch.allclose(loop, softmax_average, atol=1e-6))
        self.assertTrue(torch.allclose(matrix, softmax_weights))

    def test_mask_and_weight_sums(self):
        head = AttentionHead(12, 8, 8)
        out, weights = head(torch.randn(2, 5, 12), return_weights=True)
        self.assertEqual(tuple(out.shape), (2, 5, 8))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 5)))
        self.assertEqual(torch.count_nonzero(torch.triu(weights, 1)).item(), 0)
        self.assertTrue(torch.equal(weights[:, 0, 0], torch.ones(2)))

    def test_future_tokens_cannot_change_past_logits(self):
        model = SingleHeadLanguageModel(7, 8, 12, 8)
        x = torch.randint(7, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 7
        first, _ = model(x)
        second, _ = model(changed)
        self.assertTrue(torch.allclose(first[:, :4], second[:, :4]))

    def test_all_parameters_receive_gradients(self):
        model = SingleHeadLanguageModel(7, 8, 12, 8)
        _, loss = model(torch.randint(7, (2, 8)), torch.randint(7, (2, 8)))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_generation_crops_long_prompt_and_restores_mode(self):
        for model in (BigramLanguageModel(7, 4), SingleHeadLanguageModel(7, 4, 12, 8)):
            model.train()
            prompt = torch.zeros((1, 10), dtype=torch.long)
            generated = model.generate(prompt, 12, torch.Generator().manual_seed(42))
            self.assertEqual(tuple(generated.shape), (1, 22))
            self.assertTrue(torch.equal(generated[:, :10], prompt))
            self.assertTrue(model.training)

    def test_oversized_context_rejected(self):
        model = SingleHeadLanguageModel(7, 4, 12, 8)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 5), dtype=torch.long))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
