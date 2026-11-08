"""Tests for one canonical MAPPO schedule and CI max_team_size selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import progressive_gate as gate
import dispatch_next_gate as dispatcher


ROOT = Path(__file__).resolve().parents[3]
GATES = ROOT / "tools/ci/adaptive_curriculum/gates.json"


class CurriculumSelectionTests(unittest.TestCase):
    def test_single_schedule_includes_exact_fourteen_gates(self):
        source = GATES.read_bytes()
        document = json.loads(source)
        self.assertEqual(document["formatVersion"], 2)
        self.assertEqual(document["contractId"], "triangular-v1")
        expected = ["1v0", "2v0", "1v1", "2v1", "2v2"]
        for blue in (3, 4):
            expected.extend(f"{blue}v{red}" for red in range(blue + 1))
        self.assertEqual([s["id"] for s in gate.validate_gates(document)], expected)
        self.assertEqual(len(document["gates"]), 14)
        self.assertTrue(all(s["red"] <= s["blue"] for s in document["gates"]))
        self.assertTrue(all(not s["hostileZones"] for s in document["gates"]))

    def test_only_team_size_selects_two_three_four(self):
        document = json.loads(GATES.read_bytes())
        for size, count in ((2, 5), (3, 9), (4, 14)):
            with self.subTest(max_team_size=size):
                selected = gate.select_curriculum(document, size)
                self.assertEqual(len(selected), count)
                self.assertEqual(selected[-1]["id"], f"{size}v{size}")
                self.assertEqual(selected, document["gates"][:count])
        for invalid in (1, 5, "4", True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                gate.select_curriculum(document, invalid)

    def test_invalid_gate_order_and_red_majority_are_rejected(self):
        document = json.loads(GATES.read_bytes())
        for blue, red in ((1, 2), (2, 3), (3, 4)):
            copy = json.loads(json.dumps(document))
            copy["gates"][-1] = dict(copy["gates"][-1],
                                      id=f"{blue}v{red}", blue=blue, red=red)
            with self.assertRaisesRegex(ValueError, "triangular"):
                gate.select_curriculum(copy, 4)
        copy = json.loads(json.dumps(document))
        copy["contractId"] = "unknown"
        with self.assertRaises(ValueError):
            gate.select_curriculum(copy, 4)

    def test_resume_must_match_selected_size(self):
        # The selected size is recorded in checkpoint + result, independent
        # of the fingerprint of the same canonical schedule file.
        config = GATES.read_bytes()
        fingerprint = hashlib.sha256(config).hexdigest()
        for size in (2, 3, 4):
            stages = gate.select_curriculum(json.loads(config), size)
            current, following = stages[-2:]
            result = dict(gate=current["id"], passed=True,
                          autoAdvance=True, gateConfigSha256=fingerprint,
                          nextGate=following["id"],
                          nextThresholdPercent=following["thresholdPercent"],
                          nextMaxSeconds=following["maxSeconds"],
                          rehearsalProfile="balanced_v1")
            if size != 2:
                result["maxTeamSize"] = size
            inputs = dispatcher.next_dispatch(result, config, "420001")
            self.assertEqual(inputs["progressive_gate"], following["id"])
            self.assertEqual(inputs.get("max_team_size"),
                             None if size == 2 else str(size))
            if size != 2:
                with self.assertRaises(ValueError):
                    dispatcher.next_dispatch(dict(result, maxTeamSize=2),
                                             config, "420001")

    def test_ci_workflows_offer_one_size_control(self):
        for workflow in ("mappo-training-windows.yml",
                         "mappo-restart-progressive.yml"):
            content = (ROOT / ".github/workflows" / workflow).read_text(
                encoding="utf-8")
            self.assertEqual(content.count("      max_team_size:"), 1)
            for label in ('        default: "2"',
                          '          - "2"', '          - "3"', '          - "4"'):
                self.assertIn(label, content)
            self.assertNotIn("balanced_3v3", content)
            self.assertNotIn("balanced_4v4", content)

    def test_gui_uses_same_selected_two_team_prefix(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import local_mappo_gui_curriculum as gui
        stages, index, validated = gui.load_production_stages()
        expected = gate.select_curriculum(json.loads(GATES.read_bytes()), 2)
        self.assertEqual(stages, expected)
        self.assertEqual(index, 4)
        self.assertEqual(validated, ["1v0", "2v0", "1v1", "2v1"])


if __name__ == "__main__":
    unittest.main()
