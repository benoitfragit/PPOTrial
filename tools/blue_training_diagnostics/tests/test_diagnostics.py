"""Deterministic tests for the dependency-free Blue diagnostic engine."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from blue_diagnostics.analysis import analyze, compatibility, paired_evaluation_comparison
from blue_diagnostics.credentials import git_credential_token
from blue_diagnostics.github_client import GitHubApiError, _safe_extract
from blue_diagnostics.models import Repository
from blue_diagnostics.report import write_comparison_report, write_single_report
from blue_diagnostics.telemetry import REQUIRED_TELEMETRY_FIELDS, load_run


class BlueDiagnosticsTests(unittest.TestCase):
    def test_repository_remote_formats(self):
        self.assertEqual(
            Repository.parse("https://github.com/benoitfragit/PPOTrial.git").full_name,
            "benoitfragit/PPOTrial",
        )
        self.assertEqual(
            Repository.parse("git@github.com:benoitfragit/PPOTrial.git").full_name,
            "benoitfragit/PPOTrial",
        )
        with self.assertRaises(ValueError):
            Repository.parse("not-a-repository")
        with self.assertRaises(ValueError):
            Repository.parse("owner/repo\nhost=attacker")

    @mock.patch("blue_diagnostics.credentials.subprocess.run")
    def test_git_credential_manager_secret_stays_in_memory(self, run):
        run.return_value = mock.Mock(
            returncode=0,
            stdout="protocol=https\nhost=github.com\nusername=user\npassword=secret-value\n",
            stderr="",
        )
        token = git_credential_token(Repository("benoitfragit", "PPOTrial"))
        self.assertEqual(token, "secret-value")
        request = run.call_args[1]["input"]
        self.assertIn("host=github.com", request)
        self.assertNotIn("secret-value", request)

    def test_complete_run_analysis_and_html_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run-42"
            self._write_run(run_root, 42, "v3", "4", "4")
            analysis = analyze(load_run(run_root))
            self.assertEqual(analysis.metrics["end_update"], 12)
            self.assertEqual(analysis.metrics["completed_updates"], 2)
            self.assertEqual(analysis.metrics["optimizer_updates"], 1)
            self.assertAlmostEqual(analysis.metrics["approx_kl_latest"], 0.014)
            self.assertAlmostEqual(analysis.metrics["blue_survival_rate"], 0.5)
            self.assertEqual(analysis.metrics["loss_sources"]["low_speed"], 1)
            self.assertEqual(analysis.metrics["patrol"]["episodes"], 1)
            self.assertEqual(analysis.metrics["tolerance_by_role"]["samples"], 2)
            self.assertEqual(analysis.metrics["reward_by_role"]["samples"], 1)
            self.assertAlmostEqual(analysis.metrics["optimizer_ms_mean"], 1450.0)
            self.assertEqual(analysis.metrics["loss_diagnostics"]["low_speed"], 1)
            report = write_single_report(analysis, run_root / "report.html")
            content = report.read_text(encoding="utf-8")
            self.assertIn("Diagnostic du run 42", content)
            self.assertIn("Santé numérique", content)
            self.assertIn("Patrouille et formation", content)
            self.assertIn("Récompenses par rôle", content)
            self.assertIn("<svg", content)
            self.assertNotIn("nan", content.lower())

    def test_comparison_rejects_different_evaluation_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_root = root / "run-1"
            right_root = root / "run-2"
            self._write_run(left_root, 1, "v3", "4", "4")
            self._write_run(right_root, 2, "v2", "3", "1")
            left = analyze(load_run(left_root))
            right = analyze(load_run(right_root))
            comparable, reasons = compatibility(left, right)
            self.assertFalse(comparable)
            self.assertTrue(any("curriculum" in reason for reason in reasons))
            self.assertTrue(any("P/N" in reason for reason in reasons))
            report = write_comparison_report(left, right, root / "compare.html")
            content = report.read_text(encoding="utf-8")
            self.assertIn("Comparaison descriptive uniquement", content)
            self.assertIn("Contexte différent", content)

    def test_paired_comparison_uses_common_seed_case_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_root = root / "run-10"
            right_root = root / "run-11"
            self._write_run(left_root, 10, "v3", "4", "4")
            self._write_run(right_root, 11, "v3", "4", "4")
            paired = paired_evaluation_comparison(
                analyze(load_run(left_root)), analyze(load_run(right_root))
            )
            self.assertEqual(paired["count"], 1)
            self.assertEqual(paired["terminal_changes"], 0)
            self.assertAlmostEqual(paired["blue_alive_delta_mean"], 0.0)


    def test_ci_generator_uses_existing_run_evidence_and_summary(self):
        from blue_diagnostics.ci_report import generate_ci_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runner"
            self._write_run(run_root, 123, "v3", "1", "0")
            csv_source = next(run_root.rglob("mappo-training.csv"))
            summary = root / "step-summary.md"
            output = run_root / "diagnostic.html"
            checkpoint = next(run_root.rglob("blue_patrol_generalized_training_v6.txt"))
            report = generate_ci_report(
                run_root, csv_source, output, summary,
                {"run_number": 123, "id": 20260123}, (checkpoint,),
            )
            self.assertEqual(report, output)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Diagnostic du run 123", text)
            self.assertIn("Patrouille et formation", text)
            self.assertIn("<svg", text)
            summary_text = summary.read_text(encoding="utf-8")
            self.assertIn("blue-mappo-diagnostic", summary_text)
            self.assertIn("Dernier update PPO", summary_text)
            self.assertIn("Violations de suivi", summary_text)

    def test_ci_generator_gracefully_renders_failed_run_without_csv(self):
        from blue_diagnostics.ci_report import generate_ci_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = generate_ci_report(
                root, root / "missing.csv", root / "diagnostic.html",
                root / "summary.md", {"run_number": 99},
            )
            self.assertTrue(report.is_file())
            self.assertIn("Télémétrie CSV absente",
                          report.read_text(encoding="utf-8"))

    def test_safe_zip_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "artifact.zip"
            with zipfile.ZipFile(str(archive), "w") as output:
                output.writestr("../outside.txt", "forbidden")
            with self.assertRaises(GitHubApiError):
                _safe_extract(archive, root / "destination")
            self.assertFalse((root / "outside.txt").exists())

    def _write_run(
        self, run_root: Path, run_number: int, curriculum: str, blue: str, red: str
    ) -> None:
        telemetry = run_root / "artifacts" / "blue-mappo-telemetry-1"
        telemetry.mkdir(parents=True)
        (run_root / "run.json").write_text(
            json.dumps(
                {
                    "id": 1000 + run_number,
                    "run_number": run_number,
                    "head_branch": "master",
                    "head_sha": "abcdef0123456789",
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2026-11-01T10:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        (telemetry / "run-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "inputs": {
                        "scenario": "blue-defense",
                        "curriculum_enabled": True,
                        "curriculum_version": curriculum,
                    },
                }
            ),
            encoding="utf-8",
        )
        csv_path = telemetry / "mappo-training.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_TELEMETRY_FIELDS)
            writer.writeheader()
            for elapsed, update, score in ((0, 10, 35), (30, 11, 45), (60, 12, 55)):
                writer.writerow(
                    {
                        "elapsed_seconds": elapsed,
                        "updates": update,
                        "rollout_steps": 0,
                        "rollout_target": 2048,
                        "evaluating": 0,
                        "evaluation_progress": 1,
                        "evaluation_cases_done": 2,
                        "evaluation_cases_total": 2,
                        "evaluation_steps": 400,
                        "evaluation_steps_per_second": 100,
                        "training": 1,
                        "eval_completed": 1,
                        "latest_score": score,
                        "best_score": score,
                        "eval_success_rate": 0.5,
                        "instant_reward": update / 10.0,
                    }
                )
        (telemetry / "mappo-training.log").write_text(
            "Evaluation #2 P={} N={} score=55\n".format(blue, red)
            + "[MAPPO_EVAL_EPISODE] update=12 campaign=2 case=1 seed=4242 "
            "steps=400 P={} N={} blue_alive=2 red_alive=0 objective_alive=1 "
            "leader_alive=1 terminal=reds_cleared blue_1=alive blue_2=low_speed\n".format(blue, red)
            + "[MAPPO_EVAL_LOSS] update=12 campaign=2 case=1 seed=4242 blue=blue_2 "
            "source=low_speed speed_before=4 speed=1 accel_cmd=-0.5 low_speed=1 "
            "fuel_empty=0 fuel_remaining=80 fuel_capacity=100\n"
            + "[MAPPO_PATROL_EPISODE] phase=training P={} N={} steps=400 "
            "distance_m=100 qualified_m=60 circuit_m=100 resets=1 reset_cause=collision "
            "tracking_violations=2 spacing_applicable=1 spacing_margin_m=120\n".format(blue, red)
            + "[MAPPO_PATROL_AGENT] blue=1 role=leader valid=1 "
            "position_tolerances=2 speed_tolerances=3 heading_tolerances=4\n"
            + "[MAPPO_PATROL_AGENT] blue=2 role=teammate valid=1 "
            "position_tolerances=4 speed_tolerances=5 heading_tolerances=6\n"
            + "[MAPPO_ROLE_REWARD] role=leader agent_steps=400 "
            "[Patrol tracking]=-0.1 [Formation keeping]=0.2 [Control effort]=-0.01\n"
            + "[MAPPO_STATUS] elapsed_s=60 updates=12 rollout=0/2048 evaluation=0 "
            "eval_cases=2/2 eval_steps_per_second=100\n"
            + "[MAPPO_TIMING] optimizer=done update=12 critic_baselines_ms=400 "
            "advantages_ms=50 ppo_ms=1000 total_ms=1450\n"
            + "[MAPPO_UPDATE] update=12 policy_loss=-0.1 value_loss=0.8 "
            "entropy=2.1 approx_kl=0.014 clip_fraction=0.2 "
            "explained_variance=0.45 advantage_abs_mean=1.2 "
            "actor_derivative_l1=3 critic_derivative_l1=4 "
            "log_std_gradient_l1=5 action_std_mean=0.61 learnable=20 "
            "critic_rows=24 total_ms=1450\n",
            encoding="utf-8",
        )
        checkpoint = (
            run_root
            / "artifacts"
            / "blue-mappo-checkpoints-2"
            / "blue_patrol_generalized_training_v6.txt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("BLUE_TEAM_TRAINING_V6 6 9 fixture\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

