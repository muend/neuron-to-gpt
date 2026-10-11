"""Unit and integration tests for the decoder and restart-safe training pipeline."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from config import ModelConfig, STAGES
from data import CharacterTokenizer, Corpus, validate_turkish_corpus
from models import make_model, FeedForward, TransformerBlock
from prepare_data import normalize_text, compile_turkish
from train import fit, evaluate


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def config(self, stage="layernorm", dropout=0):
        return ModelConfig(11, stage=stage, n_embd=16, n_head=4,
                           n_layer=1, block_size=8, dropout=dropout, initialize_small=False)

    def test_all_six_stages_shapes_and_gradients(self):
        x, y = torch.randint(11, (2, 8)), torch.randint(11, (2, 8))
        for stage in STAGES:
            model = make_model(self.config(stage))
            logits, loss = model(x, y)
            self.assertEqual(tuple(logits.shape), (2, 8, 11), stage)
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (stage, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (stage, name))

    def test_future_independence_all_stages(self):
        x = torch.randint(11, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 11
        for stage in STAGES:
            model = make_model(self.config(stage)).eval()
            first, _ = model(x)
            second, _ = model(changed)
            self.assertTrue(torch.allclose(first[:, :4], second[:, :4], atol=1e-6), stage)

    def test_multihead_width(self):
        model = make_model(self.config("multi_head"))
        attention = model.blocks[0]
        x = torch.randn(2, 8, 16)
        self.assertEqual(len(attention.heads), 4)
        self.assertEqual(tuple(attention.heads[0](x).shape), (2, 8, 4))
        self.assertEqual(tuple(attention(x).shape), (2, 8, 16))

    def test_feedforward_is_positionwise(self):
        layer = FeedForward(16, 0)
        x = torch.randn(2, 8, 16)
        changed = x.clone()
        changed[:, 4:] += 100
        self.assertTrue(torch.equal(layer(x)[:, :4], layer(changed)[:, :4]))

    def test_residual_identity_path(self):
        block = TransformerBlock(self.config("residual"))
        for parameter in block.parameters():
            parameter.data.zero_()
        x = torch.randn(2, 8, 16, requires_grad=True)
        self.assertTrue(torch.equal(block(x), x))
        block(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_layernorm_axes_and_no_running_statistics(self):
        model = make_model(self.config())
        norm = model.blocks[0].norm1
        x = torch.randn(2, 8, 16) * 2 + 3
        result = norm(x)
        self.assertTrue(torch.allclose(result.mean(-1), torch.zeros(2, 8), atol=1e-6))
        self.assertTrue(torch.allclose(result.var(-1, unbiased=False), torch.ones(2, 8), atol=1e-4))
        self.assertEqual(list(norm.buffers()), [])
        norm.eval()
        self.assertTrue(torch.equal(result, norm(x)))
        changed = x.clone()
        changed[1] += 100
        self.assertTrue(torch.equal(norm(x)[0], norm(changed)[0]))

    def test_dropout_and_generation_mode(self):
        model = make_model(self.config(dropout=.4))
        x = torch.randint(11, (2, 8))
        model.train()
        self.assertFalse(torch.equal(model(x)[0], model(x)[0]))
        model.eval()
        self.assertTrue(torch.equal(model(x)[0], model(x)[0]))
        model.train()
        prompt = torch.zeros((1, 12), dtype=torch.long)
        result = model.generate(prompt, 10, torch.Generator().manual_seed(42))
        self.assertEqual(tuple(result.shape), (1, 22))
        self.assertTrue(model.training)

    def test_bad_config_and_oversized_input(self):
        with self.assertRaises(ValueError):
            ModelConfig(11, n_embd=15, n_head=4)
        with self.assertRaises(ValueError):
            make_model(self.config())(torch.zeros((1, 9), dtype=torch.long))

    def test_turkish_roundtrip_and_verse_preservation(self):
        text = "Çığ, şüphe: İı Öö Üü!\n"
        tokenizer = CharacterTokenizer(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        verse = "\ufeffİlk dize  \r\nİkinci dize\r\n\r\nSon dize\r\n"
        self.assertEqual(normalize_text(verse), "İlk dize\nİkinci dize\n\nSon dize\n")

    def test_selected_corpus_provenance_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            target = root / "compiled"
            source = {"file": "fixture.txt", "title": "Synthetic test fixture, not an author's work",
                      "genre": "test", "author_id": "nazim_hikmet", "source_reference": "unit test",
                      "permission_basis": "synthetic test text"}
            selection = {"selection_id": "nazim_hikmet", "sources": [source]}
            with self.assertRaises(ValueError):
                compile_turkish(sources, target)
            (sources / "fixture.txt").write_text("Synthetic line\n\nAnother line\n", encoding="utf-8", newline="\n")
            (sources / "sources.json").write_text(json.dumps(selection), encoding="utf-8")
            manifest = compile_turkish(sources, target)
            self.assertFalse(manifest["near_1mb_target_met"])
            self.assertEqual(validate_turkish_corpus(target), target / "turkish.txt")
            self.assertEqual((target / "turkish.txt").read_text(), "Synthetic line\n\nAnother line\n")
            # Complete-text duplicates must be rejected, not silently repeated to pad the corpus.
            selection["sources"].append(source)
            (sources / "sources.json").write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaises(ValueError):
                compile_turkish(sources, root / "duplicates")
            manifest["selection_id"] = "rejected_previous_selection"
            (target / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_turkish_corpus(target)
            manifest["selection_id"] = "nazim_hikmet"
            (target / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (target / "turkish.txt").write_text("Changed input", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_turkish_corpus(target)

    def test_vectorized_batches_and_evaluation_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("abcdefghijk\n" * 100, encoding="utf-8", newline="\n")
            corpus = Corpus(path, 8)
            generator = torch.Generator().manual_seed(1)
            x, y = corpus.get_batch("val", 4, generator)
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            self.assertEqual(corpus.metadata["split_sizes"]["val"], 120)
            model = make_model(ModelConfig(len(corpus.tokenizer), n_embd=16, n_head=4,
                               n_layer=1, block_size=8, dropout=.2))
            rng = torch.get_rng_state().clone()
            self.assertEqual(evaluate(model, corpus, 4, 2), evaluate(model, corpus, 4, 2))
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path = root / "input.txt"
            path.write_text("abcdefghij\n" * 100, encoding="utf-8", newline="\n")
            corpus = Corpus(path, 8)
            config = self.config(dropout=.2)
            kwargs = dict(batch_size=4, eval_batches=2, eval_interval=2, sample_tokens=5, preset="smoke")
            fit(config, corpus, root / "full", steps=4, **kwargs)
            fit(config, corpus, root / "resumed", steps=2, **kwargs)
            fit(config, corpus, root / "resumed", steps=4, resume=True, **kwargs)
            full = torch.load(root / "full" / "last.pt", weights_only=True)
            resumed = torch.load(root / "resumed" / "last.pt", weights_only=True)
            for name in full["model"]:
                self.assertTrue(torch.equal(full["model"][name], resumed["model"][name]), name)
            self.assertEqual(full["history"], resumed["history"])
            with self.assertRaises(ValueError):
                fit(config, corpus, root / "resumed", steps=4, resume=True, batch_size=8,
                    eval_batches=2, eval_interval=2, sample_tokens=5)
            path.write_text("jihgfedcba\n" * 100, encoding="utf-8", newline="\n")
            changed_corpus = Corpus(path, 8)
            with self.assertRaises(ValueError):
                fit(config, changed_corpus, root / "resumed", steps=4, resume=True, **kwargs)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
