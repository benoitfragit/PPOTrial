"""Deterministic contract tests for the read-only validation runner."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inference_validation as validation


def make_log(blue: int, red: int, episodes: int = 2,
             seed_base: int = 2004242, success: float = 1.0) -> str:
    rows = [
        (f"[MAPPO_EVAL_EPISODE] update=0 campaign=1 case={i + 1} "
         f"seed={seed_base + i} steps=310 P={blue} N={red} "
         "blue_alive=1 red_alive=0 objective_alive=1 terminal=reds_cleared")
        for i in range(episodes)
    ]
    rows.append(
        f"[BLUE MAPPO V10] Evaluation #1 P={blue} N={red} "
        f"score=88.5 success={success} blue-survival=1 "
        "red-neutralization=1 objective-survival=1")
    return "\n".join(rows)


class InferenceValidationTests(unittest.TestCase):
    def test_exact_five_gates_and_no_excluded_gate(self):
        self.assertEqual([v[0] for v in validation.selected_gates("all")],
                         ["1v0", "2v0", "1v1", "2v1", "2v2"])
        with self.assertRaises(ValueError):
            validation.selected_gates("1v2")

    def test_complete_readonly_inference_evaluation(self):
        result = validation.evaluation_result(
            make_log(2, 2), ("2v2", 2, 2), 2004242, 2)
        self.assertEqual(result["success_percent"], 100)
        self.assertEqual(result["score"], 88.5)
        self.assertEqual({case["seed"] for case in result["cases"]},
                         {2004242, 2004243})

    def test_reject_wrong_gate_and_duplicate_campaign(self):
        with self.assertRaisesRegex(ValueError, "P/N"):
            validation.evaluation_result(
                make_log(2, 1), ("1v1", 1, 1), 2004242, 2)
        with self.assertRaisesRegex(ValueError, "one completed campaign"):
            validation.evaluation_result(
                make_log(1, 0) + "\n" + make_log(1, 0),
                ("1v0", 1, 0), 2004242, 2)

    def test_reject_missing_duplicate_or_wrong_seeds(self):
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            validation.evaluation_result(
                make_log(1, 0, episodes=1),
                ("1v0", 1, 0), 2004242, 2)
        with self.assertRaisesRegex(ValueError, "seeds"):
            validation.evaluation_result(
                make_log(1, 0, seed_base=1004242),
                ("1v0", 1, 0), 2004242, 2)

    def test_temporary_scenario_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_file = root / "training.json"
            scenario_file = root / "scenario.json"
            training = {"evaluation": {"episodes": 8, "seed": 4242}}
            scenario = {"blueDefense": {
                "blueTeam": {"activeCount": 4},
                "redTeam": {"activeCount": 4},
                "restrictedZones": {"enabled": True},
                "restrictedZoneSpawn": {"enabledProbability": 0.75}}}
            training_bytes = json.dumps(training).encode()
            scenario_bytes = json.dumps(scenario).encode()
            with (patch.object(validation, "TRAINING_CONFIG", training_file),
                  patch.object(validation, "SCENARIO_CONFIG", scenario_file)):
                validation.configure(("2v1", 2, 1), 20, 2004242,
                                     training_bytes, scenario_bytes)
                self.assertEqual(json.loads(training_file.read_text())["evaluation"],
                                 {"episodes": 20, "seed": 2004242})
                configured = json.loads(scenario_file.read_text())["blueDefense"]
                self.assertEqual(configured["blueTeam"]["activeCount"], 2)
                self.assertEqual(configured["redTeam"]["activeCount"], 1)
                self.assertFalse(configured["restrictedZones"]["enabled"])
                self.assertEqual(
                    configured["restrictedZoneSpawn"]["enabledProbability"], 0.0)

    def test_model_grade_does_not_claim_ci_failure(self):
        cases = [validation.evaluation_result(
            make_log(2, 2, success=0.8), ("2v2", 2, 2), 2004242, 2)]
        result = validation.render_summary(cases, "deadbeef", 41948019589, 90)
        self.assertIn("Overall assessment: FAIL", result)
        self.assertIn("not a CI infrastructure failure", result)


if __name__ == "__main__":
    unittest.main()
