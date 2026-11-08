"""Data leakage, exact validation retention and deadline checks for the pilot."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import copy
import torch
from experiments.g4.pilot import BASE_SHA, prepare_fresh, remaining_seconds, pilot_lr, resume_evidence
from week12.data import ShardLoader, ShardWriter


class Encoding:
    eot_token = 50256

    def encode_ordinary(self, text):
        return [ord(char) % 100 for char in text]


class PilotTests(unittest.TestCase):
    def resume_fixture(self):
        settings = dict(precision="bf16")
        provenance = dict(base_checkpoint_sha256=BASE_SHA, data_manifest_sha256="data",
                          source_sha256={"week10/model.py": "model", "experiments/g4/pilot.py": "old"})
        checkpoint = dict(model_config=dict(dropout=0), loader_tokens_seen=72,
                          optimizer=dict(state={0: dict(step=torch.tensor(19588.)),
                                                1: dict(step=torch.tensor(19588.))}),
                          evidence=dict(settings=settings, smoke_only=False, provenance=provenance,
                                        base_updates=19586, successful_updates=1, effective_batch_tokens=32,
                                        attempted_updates=1, budget_seconds=3372, deadline="old"))
        new_provenance = copy.deepcopy(provenance)
        new_provenance["source_sha256"]["experiments/g4/pilot.py"] = "new"
        return checkpoint, settings, new_provenance

    def test_resume_replays_uncommitted_microbatches_and_counts_committed_step(self):
        checkpoint, settings, provenance = self.resume_fixture()
        evidence, tokens = resume_evidence(checkpoint, settings, provenance)
        self.assertEqual(tokens, 64)
        self.assertEqual(evidence["successful_updates"], 2)
        self.assertEqual(evidence["resumes"][-1]["replayed_uncommitted_tokens"], 8)
        self.assertEqual(evidence["resumes"][-1]["previous_source_sha256"]["experiments/g4/pilot.py"], "old")
        self.assertEqual(checkpoint["evidence"]["successful_updates"], 1)

    def test_resume_rejects_different_core_source_and_inconsistent_steps(self):
        checkpoint, settings, provenance = self.resume_fixture()
        bad = copy.deepcopy(provenance)
        bad["source_sha256"]["week10/model.py"] = "different"
        with self.assertRaisesRegex(ValueError, "core source"):
            resume_evidence(checkpoint, settings, bad)
        bad = copy.deepcopy(provenance)
        bad["data_manifest_sha256"] = "different"
        with self.assertRaises(ValueError):
            resume_evidence(checkpoint, settings, bad)
        checkpoint["optimizer"]["state"][1]["step"] += 1
        with self.assertRaisesRegex(ValueError, "counters disagree"):
            resume_evidence(checkpoint, settings, provenance)

    def test_adamw_restored_state_matches_uninterrupted_next_update(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95))
        x, target = torch.randn(4, 3), torch.randn(4, 2)
        def update(net, opt):
            opt.zero_grad(set_to_none=True)
            (net(x) - target).square().mean().backward()
            opt.step()
        for _ in range(3):
            update(model, optimizer)
        checkpoint, settings, provenance = self.resume_fixture()
        checkpoint["optimizer"] = copy.deepcopy(optimizer.state_dict())
        checkpoint["evidence"].update(base_updates=1, successful_updates=2)
        checkpoint["loader_tokens_seen"] = 64
        evidence, tokens = resume_evidence(checkpoint, settings, provenance)
        restored = torch.nn.Linear(3, 2)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        update(model, optimizer)
        update(restored, restored_optimizer)
        self.assertEqual((evidence["successful_updates"], tokens), (2, 64))
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(restored_optimizer.state.values(), optimizer.state.values()):
            torch.testing.assert_close(actual["exp_avg"], expected["exp_avg"], rtol=0, atol=0)

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
