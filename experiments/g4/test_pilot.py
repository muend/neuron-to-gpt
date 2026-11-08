"""Data leakage, exact validation retention and deadline checks for the pilot."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.g4.pilot import prepare_fresh, remaining_seconds, pilot_lr
from week12.data import ShardLoader, ShardWriter


class Encoding:
    eot_token = 50256

    def encode_ordinary(self, text):
        return [ord(char) % 100 for char in text]


class PilotTests(unittest.TestCase):
    def test_old_documents_excluded_and_validation_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = ShardWriter(root / "old", "val", 16)
            writer.append([1, 2, 3, 4, 50256])
            writer.flush()
            digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
            old = dict(document_sha256=dict(train=[digest("old train")], val=[digest("old val")]),
                       splits=dict(val=dict(tokens=writer.total, documents=1, shards=writer.shards)))
            manifest = prepare_fresh([" old train ", "old val", "abc", "abc", "defgh"],
                Encoding(), root / "new", old_manifest=old,
                val_reader=lambda name, target: shutil.copyfile(root / "old" / name, target),
                train_tokens=8, seconds=60, shard_size=3)
            self.assertEqual(manifest["splits"]["train"]["tokens"], 8)
            self.assertEqual(manifest["duplicates_removed"], 3)
            self.assertEqual(manifest["document_sha256"]["train"], [digest("abc"), digest("defgh")])
            self.assertEqual(manifest["splits"]["val"], old["splits"]["val"])
            loader = ShardLoader(root / "new", "train", 1, 7)
            x, y = loader.next_batch()
            self.assertEqual(y[0, -1].item(), 50256)
            self.assertEqual(x[0, 3].item(), 50256)
            del loader  # Release NPY memory maps before Windows temp cleanup.

    def test_partial_budget_records_actual_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = dict(document_sha256=dict(train=[], val=[]), splits=dict(val=dict(tokens=0, shards=[])))
            manifest = prepare_fresh(["abc"], Encoding(), root / "new", old_manifest=old,
                val_reader=lambda *_: None, train_tokens=100, seconds=60)
            self.assertEqual(manifest["splits"]["train"]["tokens"], 4)
            self.assertEqual(manifest["requested_tokens"]["train"], 100)

    def test_deadline_caps_ninety_minutes(self):
        now = datetime(2026, 11, 8, 14, 30, tzinfo=timezone.utc)
        self.assertEqual(remaining_seconds(90, "2026-11-08T18:48:00+03:00", now), 78*60)
        self.assertEqual(remaining_seconds(90), 90*60)
        with self.assertRaises(ValueError):
            remaining_seconds(90, "2026-11-08T17:00:00+03:00", now)
        with self.assertRaises(ValueError):
            remaining_seconds(90, "2026-11-08T18:48:00", now)

    def test_lr_warmup_and_floor(self):
        self.assertAlmostEqual(pilot_lr(0, 0, 5400), 3e-4/30)
        self.assertAlmostEqual(pilot_lr(29, 0, 5400), 3e-4)
        self.assertAlmostEqual(pilot_lr(1000, 5400, 5400), 3e-5)
        self.assertAlmostEqual(pilot_lr(1000, 6000, 5400), 3e-5)


if __name__ == "__main__":
    unittest.main()
