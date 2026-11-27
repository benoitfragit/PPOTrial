"""Deterministic contract tests for the read-only validation runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import blue_checkpoint_fixtures as fixtures
import inference_validation as validation


NAMES = ["objective_survived", "mission_success", "timeout", "blue_survival",
         "red_neutralization", "leader_patrol_quality", "formation_quality", "score"]


def make_report(episodes: int = 2, seed_base: int = 2004242, success: float = 1.0,
                suite: str = "inference") -> dict:
    """A ppokit-run suite report as `eval` writes it (aggregates and episodes)."""
    cases = [{"seed": seed_base + i, "return": 12.5, "length": 310, "terminated": True,
              "metrics": [1.0, success, 0.0, 1.0, 1.0, 0.9, 0.8, 88.5]}
             for i in range(episodes)]
    return {"suite": suite, "passed": True,
            "aggregates": {"score.mean": 88.5, "mission_success.rate": success,
                           "blue_survival.mean": 1.0, "red_neutralization.mean": 1.0,
                           "objective_survived.rate": 1.0},
            "episodes": cases}


class InferenceValidationTests(unittest.TestCase):
    def test_exact_five_gates_and_no_excluded_gate(self):
        self.assertEqual([v[0] for v in validation.selected_gates("all")],
                         ["1v0", "2v0", "1v1", "2v1", "2v2"])
        with self.assertRaises(ValueError):
            validation.selected_gates("1v2")

    def test_evaluation_profiles_select_only_requested_existing_gates(self):
        self.assertEqual(
            [gate[0] for gate in validation.selected_gates("patrol")],
            ["1v0", "2v0"])
        self.assertEqual(
            [gate[0] for gate in validation.selected_gates("combat")],
            ["1v1", "2v1", "2v2"])
        self.assertEqual(
            [gate[0] for gate in validation.selected_gates("2v2")],
            ["2v2"])

    def test_metric_names_follow_task_json(self):
        task = {"metrics": {name: {"kind": "number"} for name in NAMES}}
        self.assertEqual(validation.metric_names(task), NAMES)

    def test_complete_readonly_inference_evaluation(self):
        result = validation.evaluation_result(
            make_report(), NAMES, ("2v2", 2, 2), 2004242, 2)
        self.assertEqual(result["success_percent"], 100)
        self.assertEqual(result["score"], 88.5)
        self.assertEqual([case["seed"] for case in result["cases"]], [2004242, 2004243])
        self.assertEqual({case["termination"] for case in result["cases"]}, {"success"})

    def test_case_outcomes_from_metrics(self):
        self.assertEqual(validation.case_termination({"timeout": 1.0, "mission_success": 0.0}), "timeout")
        self.assertEqual(validation.case_termination({"timeout": 0.0, "mission_success": 1.0}), "success")
        self.assertEqual(validation.case_termination({"timeout": 0.0, "mission_success": 0.0}), "failure")

    def test_reject_other_suite_missing_or_wrong_seeds(self):
        with self.assertRaisesRegex(ValueError, "suite"):
            validation.evaluation_result(
                make_report(suite="validation"), NAMES, ("1v0", 1, 0), 2004242, 2)
        with self.assertRaisesRegex(ValueError, "expected 2 episodes"):
            validation.evaluation_result(
                make_report(episodes=1), NAMES, ("1v0", 1, 0), 2004242, 2)
        with self.assertRaisesRegex(ValueError, "seeds"):
            validation.evaluation_result(
                make_report(seed_base=1004242), NAMES, ("1v0", 1, 0), 2004242, 2)

    def test_reject_metrics_that_do_not_match_the_task(self):
        report = make_report()
        report["episodes"][0]["metrics"].pop()
        with self.assertRaisesRegex(ValueError, "task.json"):
            validation.evaluation_result(report, NAMES, ("1v0", 1, 0), 2004242, 2)

    def test_temporary_scenario_and_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario_file = root / "scenario.json"
            suite_file = root / "eval" / "inference.json"
            scenario = {"blueDefense": {
                "blueTeam": {"activeCount": 4},
                "redTeam": {"activeCount": 4},
                "restrictedZones": {"enabled": True},
                "restrictedZoneSpawn": {"enabledProbability": 0.75}}}
            with (patch.object(validation, "SCENARIO_CONFIG", scenario_file),
                  patch.object(validation, "SUITE_FILE", suite_file)):
                validation.configure(("2v1", 2, 1), 20, 2004242, json.dumps(scenario).encode())
                configured = json.loads(scenario_file.read_text())["blueDefense"]
                self.assertEqual(configured["blueTeam"]["activeCount"], 2)
                self.assertEqual(configured["redTeam"]["activeCount"], 1)
                self.assertFalse(configured["restrictedZones"]["enabled"])
                self.assertEqual(
                    configured["restrictedZoneSpawn"]["enabledProbability"], 0.0)
                suite = json.loads(suite_file.read_text())
                self.assertEqual(suite["seeds"], {"first": 2004242, "count": 20})
                self.assertEqual(suite["episodes"], 1)

    def test_readonly_run_restores_the_project(self):
        for name, text in (("policy artifact", fixtures.policy(2, 0)),
                           ("former V10 actor", fixtures.former_policy(2, 0))):
            with self.subTest(actor=name):
                self.run_readonly(text)

    def test_training_or_foreign_actor_is_refused(self):
        for text in (fixtures.training(12), "BLUE_TEAM_MAPPO_V10 10 9 test\n"):
            with self.subTest(text=text[:40]):
                with self.assertRaises(ValueError):
                    self.run_readonly(text)

    def run_readonly(self, actor_text: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "ppokit-run"
            runner.write_bytes(b"not executed - subprocess is mocked")
            project = root / "BluePatrol"
            model_root = project / "checkpoints"
            actor = model_root / "prod/policy.ppokit"
            actor.parent.mkdir(parents=True)
            actor.write_text(actor_text, encoding="utf-8")
            scenario_file = project / "scenario.json"
            scenario_file.write_text(json.dumps({
                "blueDefense": {
                    "blueTeam": {"activeCount": 4},
                    "redTeam": {"activeCount": 4},
                    "restrictedZones": {"enabled": True},
                    "restrictedZoneSpawn": {"enabledProbability": 0.75}}}))
            task_file = project / "task.json"
            task_file.write_text(json.dumps({"metrics": {name: {} for name in NAMES}}))
            suite_file = project / "eval" / "inference.json"
            report_file = model_root / "evaluation_inference.json"
            scenario_original = scenario_file.read_bytes()
            output = root / "results"
            args = argparse.Namespace(
                ppokit_run=str(runner), output=str(output),
                source_run_id=41948019589, gate="2v0", episodes=20,
                seed_base=2004242, threshold=90.0, timeout_seconds=1200)

            def evaluate(command, **_):
                # ppokit-run eval writes its report next to the checkpoints.
                self.assertEqual(command[1:3], ["eval", "--project"])
                self.assertEqual(command[3], str(project / "project.json"))
                self.assertIn("--policy", command)
                report_file.write_text(json.dumps(make_report(episodes=20)))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

            with (patch.object(validation, "PROJECT", project),
                  patch.object(validation, "MODEL_ROOT", model_root),
                  patch.object(validation, "SCENARIO_CONFIG", scenario_file),
                  patch.object(validation, "TASK", task_file),
                  patch.object(validation, "SUITE_FILE", suite_file),
                  patch.object(validation, "REPORT", report_file),
                  patch.object(validation.subprocess, "run", side_effect=evaluate)):
                self.assertEqual(validation.run(args), 0)
            self.assertEqual(len(json.loads(
                (output / "results.json").read_text())["results"][0]["cases"]), 20)
            self.assertTrue((output / "2v0.json").is_file())
            self.assertEqual(scenario_file.read_bytes(), scenario_original)
            self.assertFalse(suite_file.exists())
            self.assertFalse(report_file.exists())
            self.assertEqual(actor.read_text(), actor_text)

    def test_model_grade_does_not_claim_ci_failure(self):
        cases = [validation.evaluation_result(
            make_report(success=0.8), NAMES, ("2v2", 2, 2), 2004242, 2)]
        result = validation.render_summary(cases, "deadbeef", 41948019589, 90)
        self.assertIn("Overall assessment: FAIL", result)
        self.assertIn("not a CI infrastructure failure", result)


if __name__ == "__main__":
    unittest.main()
