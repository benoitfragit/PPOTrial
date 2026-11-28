"""Deterministic equivalence tests for the common curriculum configuration client."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import common_plan
import progressive_gate
import rehearsal_profiles


class CommonPlanTests(unittest.TestCase):
    """Historical gate policies must translate without becoming a runtime engine."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.gates_file = Path(__file__).with_name("gates.json")
        cls.document = json.loads(cls.gates_file.read_text(encoding="utf-8"))

    def test_legacy_plan_preserves_gates_thresholds_and_rehearsal_period(self) -> None:
        plan = common_plan.build_plan(
            self.gates_file, 2, "legacy", None, False, "fixed")
        gates = progressive_gate.select_curriculum(self.document, 2)
        self.assertEqual(plan["schema"], "ppokit.curriculum/1")
        self.assertEqual(plan["rehearsal_every_blocks"],
                         self.document["rehearsalEveryBlocks"])
        self.assertEqual([stage["name"] for stage in plan["stages"]],
                         [gate["id"] for gate in gates])
        for stage, gate in zip(plan["stages"], gates):
            self.assertEqual(stage["payload"], {
                "blue": gate["blue"], "red": gate["red"],
                "hostileZones": False, "objectiveLayout": "fixed"})
            self.assertEqual(stage["promotion"]["threshold"],
                             gate["thresholdPercent"] / 100.0)
            self.assertEqual(stage["max_seconds"], gate["maxSeconds"])

    def test_weighted_plan_matches_the_historical_profile_schedule(self) -> None:
        plan = common_plan.build_plan(
            self.gates_file, 3, "balanced_v1", 87.5, True, "randomized")
        gates = progressive_gate.select_curriculum(self.document, 3)
        validated: list[str] = []
        for index, (stage, gate) in enumerate(zip(plan["stages"], gates)):
            length = rehearsal_profiles.weighted_cycle_length(
                gates, index, validated, "balanced_v1")
            expected = Counter(
                rehearsal_profiles.select_weighted_stage(
                    gates, index, block, validated, "balanced_v1")["id"]
                for block in range(length))
            self.assertEqual(stage["rehearsal_weights"], dict(expected))
            self.assertEqual(stage["promotion"]["threshold"], 0.875)
            self.assertTrue(stage["payload"]["hostileZones"])
            self.assertEqual(stage["payload"]["objectiveLayout"], "randomized")
            validated.append(gate["id"])


if __name__ == "__main__":
    unittest.main()
