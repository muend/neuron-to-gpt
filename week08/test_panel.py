"""Panel regression tests; tiny random checkpoints test mechanics, not model quality."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from config import ModelConfig
from models import make_model
from panel_runtime import MODELS, PanelService, SamplingSettings, architecture_rows


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results, self.weights = self.root / "results", self.root / "weights"
        self.chars = sorted("\n abç")
        self.config = ModelConfig(len(self.chars), n_embd=8, n_head=2, n_layer=1, block_size=4, dropout=.2)
        torch.manual_seed(8)
        self.model = make_model(self.config)
        self.saved = {"model_config": self.config.to_dict(), "model": self.model.state_dict(),
                      "chars": self.chars, "sha256": "a" * 64, "step": 4500}
        for key, (_, run, _) in MODELS.items():
            directory = self.results / run
            directory.mkdir(parents=True)
            (directory / "result.json").write_text(json.dumps({
                "model_config": self.config.to_dict(), "corpus": {"sha256": "a" * 64, "vocabulary": len(self.chars), "characters": 200},
                "parameters": sum(p.numel() for p in self.model.parameters()), "sample_checkpoint_step": 4500,
                "final_val_nll": 2., "final_train_nll": 1.8, "best_val_nll": 1.9,
                "training_seconds": 60., "elapsed_seconds": 80.}), encoding="utf-8")
            (directory / "history.json").write_text('[{"step":4500,"train":1.8,"val":1.9}]', encoding="utf-8")
            (directory / "sample.txt").write_text(f"Recorded {key} sample", encoding="utf-8")
        self.write_checkpoint()

    def write_checkpoint(self):
        path = self.weights / MODELS["en"][1] / "best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.saved, path)
        return path

    def service(self, with_weights=True):
        return PanelService(self.weights if with_weights else None, results_root=self.results, device="cpu")

    def test_snapshot_is_explicitly_not_live(self):
        service = self.service(False)
        text, notice = service.saved_sample("en")
        self.assertEqual(text, "Recorded en sample")
        self.assertIn("Not live generation", notice)
        self.assertFalse(service.has_checkpoint("en"))
        with self.assertRaisesRegex(ValueError, "needs the owner's"):
            service.load_model("en")

    def test_unknown_selection_and_missing_second_checkpoint(self):
        service = self.service()
        self.assertTrue(service.has_checkpoint("en"))
        self.assertFalse(service.has_checkpoint("tr"))
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            service.record("../data")

    def test_matching_checkpoint_is_cached_eval_and_rng_preserved(self):
        service = self.service()
        before = torch.get_rng_state().clone()
        first, tokenizer = service.load_model("en")
        second, _ = service.load_model("en")
        self.assertIs(first, second)
        self.assertFalse(first.training)
        self.assertEqual(tokenizer.chars, self.chars)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_wrong_corpus_step_and_architecture_rejected(self):
        for field, replacement, message in (("sha256", "b" * 64, "checksum"), ("step", 5000, "step"),
                                             ("model_config", {**self.config.to_dict(), "dropout": .1}, "architecture")):
            original = self.saved[field]
            self.saved[field] = replacement
            self.write_checkpoint()
            with self.assertRaisesRegex(ValueError, message):
                self.service().load_model("en")
            self.saved[field] = original

    def test_wrong_vocabulary_rejected(self):
        self.saved["chars"] = list(reversed(self.chars))
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.service().load_model("en")

    def test_nonfinite_weights_rejected(self):
        self.saved["model"]["lm_head.weight"][0, 0] = float("nan")
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.service().load_model("en")

    def test_repeatable_stream_crops_context_preserves_prompt(self):
        service = self.service()
        prompt = "ab ç\n" * 10
        settings = SamplingSettings(20, .8, 2, 123)
        first = list(service.stream("en", prompt, settings))
        second = list(service.stream("en", prompt, settings))
        self.assertEqual(first[-1][0], second[-1][0])
        self.assertTrue(first[-1][0].startswith(prompt))
        self.assertEqual(len(first[-1][0]), len(prompt) + 20)
        self.assertIn("live CPU inference", first[-1][1])
        self.assertIn("Complete", first[-1][1])
        self.assertEqual(len(first), 2)

    def test_empty_prompt_and_all_character_sampling(self):
        final = list(self.service().stream("en", "", SamplingSettings(3, 1., 0, 5)))[-1][0]
        self.assertTrue(final.startswith("\n"))
        self.assertEqual(len(final), 4)

    def test_unknown_prompt_characters_rejected_without_silent_replacement(self):
        with self.assertRaisesRegex(ValueError, "outside this model's vocabulary"):
            list(self.service().stream("en", "xyz", SamplingSettings(2)))

    def test_bad_sampling_ranges(self):
        for change in ({"temperature": float("nan")}, {"temperature": 0}, {"characters": 1001},
                       {"top_k": -1}, {"seed": -1}, {"characters": 1.5}):
            with self.assertRaises(ValueError):
                SamplingSettings(**change)

    def test_architecture_table_reads_measured_rows(self):
        rows = architecture_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0][2], 2.4942)
        self.assertEqual(rows[-1][2], 1.8673)

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Optional Gradio dependencies not installed")
    def test_ui_builds_without_checkpoints_and_sharing_needs_auth(self):
        from panel import build_panel, launch_panel
        service = PanelService(device="cpu")
        demo = build_panel(service)
        self.addCleanup(demo.close)
        self.assertEqual(demo.title, "Neuron to GPT | Week 08")
        buttons = [c for c in demo.config["components"] if c["type"] == "button"]
        generate = next(c for c in buttons if c["props"].get("value") == "Generate continuation")
        self.assertFalse(generate["props"]["interactive"])
        with self.assertRaisesRegex(ValueError, "requires authentication"):
            launch_panel(share=True)
        with self.assertRaisesRegex(ValueError, "requires authentication"):
            launch_panel(share=True, auth=("viewer", ""))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
