"""Regression tests for safe export import and source-faithful reporting."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from import_gpu_results import ASSETS, RUNS, import_archive, validate_archive
import summarize_results

HERE = Path(__file__).resolve().parent


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.entries = {f"{run}/{asset}": (HERE / "results/gpu" / run / asset).read_bytes()
                        for run in RUNS for asset in ASSETS}
        self.entries["submission_report.md"] = b"# Returned report\n"
        tr = json.loads(self.entries["tr_a100_layernorm/result.json"])["corpus"]
        # Test provenance is synthetic; no private manifest is required by the suite.
        self.entries["corpus_manifest.json"] = json.dumps({
            "selection_id": "nazim_hikmet", "sources": [{"author_id": "nazim_hikmet"}],
            "turkish_sha256": tr["sha256"], "bytes": tr["bytes"]}).encode()

    def archive(self):
        path = self.root / "export.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in self.entries.items():
                archive.writestr(name, content)
        return path

    def change_json(self, name, change):
        record = json.loads(self.entries[name])
        change(record)
        self.entries[name] = json.dumps(record).encode()

    def test_verified_metadata_without_private_corpora(self):
        _, report = validate_archive(self.archive(), check_local=False)
        self.assertEqual([r["best_step"] for r in report["verified"]], [4500, 5000])
        self.assertTrue(all(not r["local_corpus_verified"] for r in report["verified"]))
        self.assertEqual(report["assessment"], "share_with_noted_caveats")

    def test_unknown_or_traversal_entry_rejected(self):
        self.entries["../escape.txt"] = b"untrusted"
        with self.assertRaisesRegex(ValueError, "archive entries"):
            validate_archive(self.archive(), check_local=False)
        self.assertFalse((self.root.parent / "escape.txt").exists())

    def test_missing_entry_rejected(self):
        del self.entries["en_a100_layernorm/sample.txt"]
        with self.assertRaisesRegex(ValueError, "archive entries"):
            validate_archive(self.archive(), check_local=False)

    def test_final_loss_must_match_history(self):
        self.change_json("en_a100_layernorm/result.json", lambda r: r.update(final_val_nll=1.0))
        with self.assertRaisesRegex(ValueError, "final loss differs"):
            validate_archive(self.archive(), check_local=False)

    def test_parameter_count_recomputed(self):
        self.change_json("en_a100_layernorm/result.json", lambda r: r.update(parameters=1))
        with self.assertRaisesRegex(ValueError, "parameter count"):
            validate_archive(self.archive(), check_local=False)

    def test_history_steps_required(self):
        self.change_json("en_a100_layernorm/history.json", lambda r: r.pop())
        with self.assertRaisesRegex(ValueError, "missing steps"):
            validate_archive(self.archive(), check_local=False)

    def test_wrong_selected_author_rejected(self):
        self.change_json("corpus_manifest.json", lambda r: r.update(selection_id="another_author"))
        with self.assertRaisesRegex(ValueError, "selected author"):
            validate_archive(self.archive(), check_local=False)

    def test_provenance_hash_must_match_training(self):
        self.change_json("corpus_manifest.json", lambda r: r.update(turkish_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            validate_archive(self.archive(), check_local=False)

    def test_import_is_idempotent_and_keeps_manifest_private(self):
        archive, destination = self.archive(), self.root / "imported"
        first = import_archive(archive, destination, check_local=False)
        second = import_archive(archive, destination, check_local=False)
        self.assertEqual(first, second)
        self.assertFalse((destination / "corpus_manifest.json").exists())
        for run in RUNS:
            for asset in ASSETS:
                self.assertEqual((destination / run / asset).read_bytes(), self.entries[f"{run}/{asset}"])

    def test_conflicting_target_preflight_prevents_partial_import(self):
        destination = self.root / "imported"
        destination.mkdir()
        (destination / "submission_report.md").write_text("Owner's existing report", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Existing result differs"):
            import_archive(self.archive(), destination, check_local=False)
        self.assertEqual([p.name for p in destination.iterdir()], ["submission_report.md"])

    def test_report_uses_saved_metadata_without_local_text(self):
        destination = self.root / "report_inputs"
        destination.mkdir()
        run = destination / RUNS[0]
        run.mkdir()
        (run / "result.json").write_bytes(self.entries[f"{RUNS[0]}/result.json"])
        with patch("sys.argv", ["summarize_results.py", "--output-root", str(destination)]):
            with contextlib.redirect_stdout(io.StringIO()):
                summarize_results.main()
        report = (destination / "submission_report.md").read_text(encoding="utf-8")
        metadata = json.loads(self.entries[f"{RUNS[0]}/result.json"])["corpus"]
        self.assertIn(metadata["sha256"], report)
        self.assertIn('"characters": 1115394', report)
        self.assertIn("1.4723", report)
        self.assertNotIn("not_ready", report)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
