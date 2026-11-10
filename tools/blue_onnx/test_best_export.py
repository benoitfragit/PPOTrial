"""End-to-end tests for BEST selection and immutable checkpoint export."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_best import export_best
from test_export import Fixture


class BestExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "best"
        self.output = Path(self.temp.name) / "onnx"

    def make_champion(self, relative: str, *,
                      evaluated: bool = True,
                      mismatch: bool = False) -> Path:
        folder = self.root / relative
        folder.mkdir(parents=True)
        v10 = Fixture()
        if mismatch:
            v10.rng = np.random.default_rng(991)
        v10.actor()
        v7 = Fixture()
        v7.training()
        actor_path = folder / "blue_patrol_generalized_v10.txt"
        training_path = folder / "blue_patrol_generalized_training_v7.txt"
        actor_path.write_text(v10.text(), encoding="utf-8")
        training_path.write_text(v7.text(), encoding="utf-8")
        manifest = {
            "gate": folder.name,
            "evaluatedForGate": evaluated,
            "updates": 42,
            "sha256": {
                actor_path.name: hashlib.sha256(actor_path.read_bytes()).hexdigest(),
                training_path.name:
                    hashlib.sha256(training_path.read_bytes()).hexdigest(),
            },
        }
        (folder / "champion.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return folder

    def test_automatic_discovers_all_scenario_namespaces(self):
        sources = (
            "4v4",
            "zones-on/2v2",
            "objective-randomized/3v3",
            "objective-randomized/zones-on/1v0",
        )
        before = {}
        for name in sources:
            folder = self.make_champion(name)
            before[name] = [
                p.read_bytes() for p in sorted(folder.glob("*.txt"))
            ]
        selected = export_best(self.root, self.output)
        self.assertEqual(sorted(selected), sorted(sources))
        for name in sources:
            location = self.output / name
            self.assertTrue((location / "blue_actor.onnx").is_file())
            self.assertTrue((location / "blue_critic.onnx").is_file())
            self.assertTrue((location / "manifest.json").is_file())
            self.assertEqual(
                before[name],
                [p.read_bytes() for p in sorted((self.root / name).glob("*.txt"))],
            )

    def test_manual_selects_only_requested_variant(self):
        self.make_champion("4v4")
        self.make_champion("objective-randomized/zones-on/4v4")
        result = export_best(
            self.root, self.output, gate="4v4",
            variant="randomized-zones-on"
        )
        self.assertEqual(result, ["objective-randomized/zones-on/4v4"])
        self.assertFalse((self.output / "4v4").exists())
        self.assertTrue(
            (self.output / result[0] / "blue_actor.onnx").is_file()
        )

    def test_missing_or_unevaluated_gate_is_rejected_manually(self):
        with self.assertRaisesRegex(ValueError, "Missing evaluated BEST"):
            export_best(self.root, self.output, gate="4v4")
        self.make_champion("4v4", evaluated=False)
        with self.assertRaisesRegex(ValueError, "not yet evaluated"):
            export_best(self.root, self.output, gate="4v4")
        self.assertEqual(export_best(self.root, self.output), [])

    def test_corrupt_hash_fails_before_any_export(self):
        folder = self.make_champion("4v4")
        manifest_file = folder / "champion.json"
        metadata = json.loads(manifest_file.read_text(encoding="utf-8"))
        metadata["sha256"]["blue_patrol_generalized_v10.txt"] = "0" * 64
        manifest_file.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            export_best(self.root, self.output, gate="4v4")
        self.assertFalse(self.output.exists())

    def test_mismatched_actor_is_rejected_even_with_valid_hashes(self):
        self.make_champion("4v4", mismatch=True)
        with self.assertRaises(AssertionError):
            export_best(self.root, self.output, gate="4v4")
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_gate_and_variant(self):
        self.make_champion("4v4")
        with self.assertRaisesRegex(ValueError, "Unknown BEST gate"):
            export_best(self.root, self.output, gate="../../etc")
        with self.assertRaisesRegex(ValueError, "Unknown BEST scenario"):
            export_best(self.root, self.output, gate="4v4",
                        variant="not-a-variant")


if __name__ == "__main__":
    unittest.main()
