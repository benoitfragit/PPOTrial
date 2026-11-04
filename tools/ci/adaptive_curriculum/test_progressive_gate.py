"""Tests for independent, outcome-based GitHub MAPPO training gates."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import progressive_gate as gate
import dispatch_next_gate as dispatcher


def sample_gates() -> dict:
    return {"formatVersion": 1, "rehearsalEveryBlocks": 8, "gates": [
        {"id": "1v0", "blue": 1, "red": 0, "hostileZones": False,
         "thresholdPercent": 75, "maxSeconds": 14400, "autoAdvance": True},
        {"id": "1v1", "blue": 1, "red": 1, "hostileZones": False,
         "thresholdPercent": 80, "maxSeconds": 12000, "autoAdvance": True},
        {"id": "4v4", "blue": 4, "red": 4, "hostileZones": True,
         "thresholdPercent": 85, "maxSeconds": 14400, "autoAdvance": False},
    ]}


class ProgressiveGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        previous = Path.cwd()
        os.chdir(temp.name)
        self.addCleanup(os.chdir, previous)
        gate.existing.write_json(gate.GATES_FILE, sample_gates())
        gate.existing.write_json(gate.existing.TRAINING_FILE,
                                 {"training": {"maxUpdates": 300,
                                               "checkpointEvery": 10,
                                               "saveEveryUpdates": 1}})
        gate.existing.write_json(gate.existing.SCENARIO_FILE,
                                 {"blueDefense": {
                                     "blueTeam": {"activeCount": 4},
                                     "redTeam": {"activeCount": 4},
                                     "restrictedZones": {"enabled": True},
                                     "restrictedZoneSpawn": {
                                         "enabledProbability": 0.7},
                                 }})
        self.training_bytes = gate.existing.TRAINING_FILE.read_bytes()
        self.scenario_bytes = gate.existing.SCENARIO_FILE.read_bytes()
        self.binary = Path("PPO.exe")
        self.binary.write_text("simulated executable", encoding="utf-8")

    def args(self, **overrides: object) -> SimpleNamespace:
        kwargs = dict(binary=str(self.binary), gate="1v0",
                      threshold_percent=75.0, max_seconds=120,
                      block_updates=5, resume=False, auto_advance=True)
        kwargs.update(overrides)
        return SimpleNamespace(**kwargs)

    def checkpoint(self, number: int) -> None:
        gate.existing.LATEST.parent.mkdir(parents=True, exist_ok=True)
        gate.existing.LATEST.write_text(
            f"BLUE_TEAM_TRAINING_V7 7 10 200 UNIFIED_PHYSICAL_GUIDANCE_V1\n"
            f"UPDATES {number}\n",
            encoding="utf-8")
        gate.champion.LATEST_ACTOR.write_text(
            f"BLUE_TEAM_MAPPO_V10 10 10 200 UNIFIED_PHYSICAL_GUIDANCE_V1\n"
            f"ACTOR_UPDATE {number}\n",
            encoding="utf-8")

    def evaluation(self, stage: dict, success: float,
                   score: float = 55.0) -> str:
        return (
            f"[BLUE MAPPO V10] Saved actor and training checkpoint at "
            f"update {gate.existing.read_updates()}\n"
            f"[BLUE MAPPO V10] Starting deterministic P={stage['blue']} "
            f"N={stage['red']} evaluation after update "
            f"{gate.existing.read_updates()} (8 cases)\n"
            f"[BLUE MAPPO V10] Evaluation #1 P={stage['blue']} "
            f"N={stage['red']} score={score} success={success}\n")

    def test_schema_has_no_fixed_gate_update_limit(self) -> None:
        stages = gate.validate_gates(sample_gates())
        self.assertEqual([s["id"] for s in stages], ["1v0", "1v1", "4v4"])
        self.assertTrue(all("updates" not in s for s in stages))
        bad = sample_gates()
        bad["gates"][0]["updates"] = 50
        with self.assertRaisesRegex(ValueError, "fields"):
            gate.validate_gates(bad)
        bad = sample_gates()
        bad["gates"][1]["thresholdPercent"] = 101
        with self.assertRaisesRegex(ValueError, "dimensions"):
            gate.validate_gates(bad)

    def test_rehearsal_does_not_limit_updates_and_rotates_prior_gates(self) -> None:
        stages = gate.validate_gates(sample_gates())
        ids = [gate.select_training_stage(stages, 2, k, 8)["id"]
               for k in range(32)]
        self.assertEqual([ids[k] for k in (7, 15, 23, 31)],
                         ["1v0", "1v1", "1v0", "1v1"])
        self.assertTrue(all(ids[k] == "4v4"
                            for k in range(32) if (k + 1) % 8 != 0))
        self.assertEqual(
            gate.select_training_stage(stages, 0, 99999, 8)["id"], "1v0")
        with self.assertRaisesRegex(ValueError, "rehearsal"):
            gate.select_training_stage(stages, 1, -1, 8)

    def test_campaign_matching_not_score_or_other_gate(self) -> None:
        p = Path("evaluation.log")
        p.write_text(
            "Evaluation #1 P=4 N=4 score=99 success=1\n"
            "Evaluation #1 P=1 N=0 score=80 success=0.50\n"
            "Evaluation #2 P=1 N=0 score=95 success=0.875 "
            "blue-survival=1\n", encoding="utf-8")
        self.assertEqual(gate.matching_success(p, 1, 0), 87.5)
        self.assertEqual(gate.matching_success(p, 1, 1), None)

    def test_pass_and_next_gate_keep_same_optimizer_updates(self) -> None:
        runs = []
        def simulate(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> str:
            runs.append(stage["id"])
            target = json.loads(
                gate.existing.TRAINING_FILE.read_text(encoding="utf-8")
            )["training"]["maxUpdates"]
            self.checkpoint(target)
            rate = 0.875
            log_path.write_text(self.evaluation(stage, rate), encoding="utf-8")
            return "evaluation"

        with patch.object(gate.existing, "run_stage", side_effect=simulate):
            first = gate.run_gate(self.args())
            self.assertTrue(first["passed"])
            self.assertEqual(gate.existing.read_updates(), 5)
            result = json.loads(gate.RESULT_FILE.read_text())
            self.assertEqual(result["nextGate"], "1v1")
            self.assertEqual(result["nextThresholdPercent"], 80)
            self.assertEqual(result["nextMaxSeconds"], 12000)
            second = gate.run_gate(self.args(
                gate="1v1", threshold_percent=80, resume=True))
            self.assertTrue(second["passed"])
        self.assertEqual(runs, ["1v0", "1v1"])
        self.assertEqual(gate.existing.read_updates(), 10)
        self.assertEqual(gate.existing.TRAINING_FILE.read_bytes(),
                         self.training_bytes)
        self.assertEqual(gate.existing.SCENARIO_FILE.read_bytes(),
                         self.scenario_bytes)

    def test_failure_continues_same_gate_with_fresh_run_budget(self) -> None:
        clock = [0.0]
        attempted = []
        def no_pass(binary: Path, stage: dict, seconds: int,
                    log_path: Path) -> str:
            attempted.append(stage["id"])
            self.checkpoint(gate.existing.read_updates() + 5)
            log_path.write_text(self.evaluation(stage, 0.25, score=99))
            clock[0] = 45.0
            return "no pass"

        with patch.object(gate.time, "monotonic", side_effect=lambda: clock[0]):
            with patch.object(gate.existing, "run_stage", side_effect=no_pass):
                first = gate.run_gate(self.args(max_seconds=45))
            self.assertFalse(first["passed"])
            self.assertEqual(first["gate"], "1v0")
            self.assertEqual(gate.existing.read_updates(), 5)
            clock[0] = 0.0

            def now_pass(binary: Path, stage: dict, seconds: int,
                         log_path: Path) -> str:
                attempted.append(stage["id"])
                self.checkpoint(gate.existing.read_updates() + 5)
                log_path.write_text(self.evaluation(stage, 0.875))
                clock[0] = 45.0
                return "pass"

            with patch.object(gate.existing, "run_stage", side_effect=now_pass):
                second = gate.run_gate(self.args(max_seconds=45, resume=True))
        self.assertEqual(attempted, ["1v0", "1v0"])
        self.assertTrue(second["passed"])
        self.assertEqual(gate.existing.read_updates(), 10)

    def test_explicit_skip_preserves_optimizer_without_fake_success(self) -> None:
        stages = gate.validate_gates(sample_gates())
        cfg_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "Later gates"):
            gate.initialize_state(stages, "4v4", False, cfg_hash)
        with self.assertRaisesRegex(ValueError, "requires complete"):
            gate.initialize_state(stages, "1v0", True, cfg_hash)
        self.checkpoint(240)
        # Old checkpoints had no validatedGates; they enforced sequential order.
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": cfg_hash,
            "gateIndex": 0, "gate": "1v0", "passed": True,
            "elapsedSeconds": 3600.0, "blockIndex": 2,
            "lastCheckpointUpdates": 240, "bestSuccessPercent": 87.5,
            "lastSuccessPercent": 87.5,
        })
        state = gate.initialize_state(stages, "4v4", True, cfg_hash)
        self.assertEqual(state["gate"], "4v4")
        self.assertEqual(state["gateIndex"], 2)
        self.assertEqual(state["validatedGates"], ["1v0"])
        self.assertEqual(state["lastCheckpointUpdates"], 240)
        self.assertEqual(gate.existing.read_updates(), 240)
        self.assertFalse(state["passed"])
        self.assertEqual(state["bestSuccessPercent"], 0.0)
        self.assertEqual(state["blockIndex"], 0)
        rehearsed = gate.select_training_stage(
            stages, 2, 7, 8, state["validatedGates"])
        self.assertEqual(rehearsed["id"], "1v0")
        gate.existing.write_json(gate.STATE_FILE, state)
        with self.assertRaisesRegex(ValueError, "Backward"):
            gate.initialize_state(stages, "1v1", True, cfg_hash)

    def test_skip_unfinished_gate_does_not_certify_it(self) -> None:
        stages = gate.validate_gates(sample_gates())
        cfg_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        self.checkpoint(245)
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": cfg_hash,
            "gateIndex": 1, "gate": "1v1", "passed": False,
            "elapsedSeconds": 60.0, "blockIndex": 3,
            "lastCheckpointUpdates": 245, "bestSuccessPercent": 20.0,
            "lastSuccessPercent": 20.0, "validatedGates": ["1v0"],
        })
        state = gate.initialize_state(stages, "4v4", True, cfg_hash)
        self.assertEqual(state["validatedGates"], ["1v0"])
        self.assertEqual(state["gate"], "4v4")
        gate.existing.write_json(gate.STATE_FILE, state)
        with self.assertRaisesRegex(ValueError, "history"):
            state["validatedGates"] = ["1v0", "4v4"]
            gate.existing.write_json(gate.STATE_FILE, state)
            gate.initialize_state(stages, "4v4", True, cfg_hash)

    def test_rehearsal_candidate_never_overwrites_four_v_four_production(self) -> None:
        stages = gate.validate_gates(sample_gates())
        config_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        self.checkpoint(15)
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": config_hash,
            "gateIndex": 2, "gate": "4v4", "passed": False,
            "elapsedSeconds": 40.0, "blockIndex": 7,
            "lastCheckpointUpdates": 15, "bestSuccessPercent": 0.0,
            "lastSuccessPercent": None,
        })
        gate.existing.PROD.parent.mkdir(parents=True, exist_ok=True)
        gate.existing.PROD.write_text("promoted reference")
        gate.existing.PROD_SCORE.write_text("reference score")
        clock = [0.0]
        observed = []

        def simulate(binary: Path, selected: dict, seconds: int,
                     log_path: Path) -> str:
            observed.append(selected["id"])
            target = json.loads(
                gate.existing.TRAINING_FILE.read_text()
            )["training"]["maxUpdates"]
            self.checkpoint(target)
            gate.existing.PROD.write_text("candidate " + selected["id"])
            gate.existing.PROD_SCORE.write_text("candidate score")
            log_path.write_text(self.evaluation(selected, 1.0, score=100))
            clock[0] = 45.0
            return "evaluation"

        with patch.object(gate.time, "monotonic",
                          side_effect=lambda: clock[0]):
            with patch.object(gate.existing, "run_stage", side_effect=simulate):
                first = gate.run_gate(self.args(
                    gate="4v4", threshold_percent=85, resume=True,
                    max_seconds=45))
                self.assertFalse(first["passed"])
                self.assertEqual(first["blockIndex"], 8)
                self.assertEqual(first["lastCheckpointUpdates"], 20)
                self.assertEqual(
                    gate.existing.PROD.read_text(), "promoted reference")
                self.assertEqual(
                    gate.existing.PROD_SCORE.read_text(), "reference score")
                clock[0] = 0.0
                second = gate.run_gate(self.args(
                    gate="4v4", threshold_percent=85, resume=True,
                    max_seconds=45))
        self.assertEqual(observed, ["1v0", "4v4"])
        self.assertTrue(second["passed"])
        self.assertEqual(gate.existing.read_updates(), 25)
        # The pre-existing 4v4 production model is not replaced by a
        # separate curriculum run. Its gate-local BEST is persisted instead.
        self.assertEqual(gate.existing.PROD.read_text(), "promoted reference")
        self.assertEqual(
            (gate.champion.ROOT / "4v4" /
             gate.champion.ACTOR_NAME).read_bytes(),
            gate.champion.LATEST_ACTOR.read_bytes())
        self.assertEqual(gate.existing.TRAINING_FILE.read_bytes(),
                         self.training_bytes)
        self.assertEqual(gate.existing.SCENARIO_FILE.read_bytes(),
                         self.scenario_bytes)

    def test_no_new_optimizer_update_keeps_current_gate_resumable(self) -> None:
        stages = gate.validate_gates(sample_gates())
        config_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        self.checkpoint(40)
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": config_hash,
            "gateIndex": 0, "gate": "1v0", "passed": False,
            "elapsedSeconds": 14000.0, "blockIndex": 2,
            "lastCheckpointUpdates": 40, "bestSuccessPercent": 0.0,
            "lastSuccessPercent": None,
        })
        clock = [0.0]

        def expired(binary: Path, stage: dict, seconds: int,
                    log_path: Path) -> None:
            log_path.write_text("")
            clock[0] = 45.0

        with patch.object(gate.time, "monotonic",
                          side_effect=lambda: clock[0]):
            with patch.object(gate.existing, "run_stage", side_effect=expired):
                state = gate.run_gate(self.args(max_seconds=45, resume=True))
        self.assertFalse(state["passed"])
        self.assertEqual(state["blockIndex"], 2)
        self.assertEqual(state["lastCheckpointUpdates"], 40)
        self.assertEqual(gate.existing.read_updates(), 40)
        self.assertIsNone(json.loads(
            gate.RESULT_FILE.read_text())["lastSuccessPercent"])
        self.assertEqual(
            gate.initialize_state(stages, "1v0", True, config_hash)["gate"], "1v0")


    def test_best_full_checkpoint_survives_later_degraded_update(self) -> None:
        """A 25%/100 score cannot replace an 87.5%/55 evaluated pair."""
        clock = [0.0]
        evaluations = []

        def simulate(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> str:
            target = json.loads(
                gate.existing.TRAINING_FILE.read_text()
            )["training"]["maxUpdates"]
            self.checkpoint(target)
            success, score = [(0.875, 55), (0.25, 100)][len(evaluations)]
            evaluations.append(target)
            log_path.write_text(self.evaluation(stage, success, score))
            if len(evaluations) == 2:
                clock[0] = 45.0
            return "evaluation"

        with patch.object(gate.time, "monotonic", side_effect=lambda: clock[0]):
            with patch.object(gate.existing, "run_stage", side_effect=simulate):
                state = gate.run_gate(self.args(
                    block_updates=20, threshold_percent=95, max_seconds=45))
        self.assertEqual(evaluations, [10, 20])
        self.assertEqual(gate.existing.read_updates(), 20)
        record = gate.champion.read("1v0")
        self.assertEqual(record["updates"], 10)
        self.assertEqual(record["successPercent"], 87.5)
        self.assertEqual(record["score"], 55)
        self.assertEqual(state["bestSuccessPercent"], 87.5)
        self.assertIn("UPDATES 10", (
            gate.champion.ROOT / "1v0" /
            gate.champion.TRAINING_NAME).read_text())

    def test_gate_transition_restores_exact_previous_champion_pair(self) -> None:
        self.checkpoint(100)
        evaluated = {"successPercent": 87.5, "score": 80.0}
        gate.champion.promote("1v0", evaluated, 100)
        # Simulate additional, worse training with no new promotion.
        self.checkpoint(120)
        config_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": config_hash,
            "gateIndex": 0, "gate": "1v0", "passed": True,
            "elapsedSeconds": 100.0, "blockIndex": 2,
            "lastCheckpointUpdates": 100, "bestSuccessPercent": 87.5,
            "lastSuccessPercent": 87.5, "validatedGates": ["1v0"],
            "championRequired": True,
        })
        stages = gate.validate_gates(sample_gates())
        next_state = gate.initialize_state(stages, "1v1", True, config_hash)
        self.assertEqual(next_state["lastCheckpointUpdates"], 100)
        self.assertEqual(gate.existing.read_updates(), 100)
        self.assertEqual((gate.champion.LATEST_ACTOR).read_bytes(), (
            gate.champion.ROOT / "1v0" /
            gate.champion.ACTOR_NAME).read_bytes())
        seed = gate.champion.read("1v1")
        self.assertFalse(seed["evaluatedForGate"])
        self.assertEqual(seed["originGate"], "1v0")
        self.assertEqual(seed["updates"], 100)
        self.assertEqual(gate.champion.read("1v0")["successPercent"], 87.5)

    def test_corrupt_champion_fails_closed_and_preserves_latest(self) -> None:
        self.checkpoint(30)
        gate.champion.promote(
            "1v0", {"successPercent": 100.0, "score": 90.0}, 30)
        (gate.champion.ROOT / "1v0" /
         gate.champion.TRAINING_NAME).write_text("truncated checkpoint")
        with self.assertRaisesRegex(ValueError, "champion"):
            gate.champion.restore_latest("1v0")
        self.assertEqual(gate.existing.read_updates(), 30)

    def test_incomplete_evaluation_cannot_promote(self) -> None:
        path = Path("incomplete.log")
        path.write_text(
            "[BLUE MAPPO V10] Saved actor and training checkpoint at update 40\n"
            "[BLUE MAPPO V10] Starting deterministic P=1 N=0 "
            "evaluation after update 40 (8 cases)\n"
            "[BLUE MAPPO V10] Evaluation #1 P=1 N=0 score=80 success=1\n")
        self.assertIsNone(gate.champion.evaluation_at_update(path, 1, 0, 50))
        self.assertIsNone(gate.champion.evaluation_at_update(path, 2, 0, 40))
        self.assertEqual(
            gate.champion.evaluation_at_update(path, 1, 0, 40),
            {"score": 80.0, "successPercent": 100.0})

    def test_legacy_gate_pass_migrates_final_evaluated_not_lost_best(self) -> None:
        # Real-world legacy pattern: historical 100%, final passing 87.5%.
        # The lost historical 100% must not be fabricated or cause restart.
        self.checkpoint(240)
        config_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": config_hash,
            "gateIndex": 0, "gate": "1v0", "passed": True,
            "elapsedSeconds": 100.0, "blockIndex": 5,
            "lastCheckpointUpdates": 240, "bestSuccessPercent": 100.0,
            "lastSuccessPercent": 87.5, "validatedGates": ["1v0"],
        })
        state = gate.initialize_state(
            gate.validate_gates(sample_gates()), "1v1", True, config_hash)
        verified = gate.champion.read("1v0")
        self.assertEqual(verified["successPercent"], 87.5)
        self.assertEqual(verified["originGate"], "legacy-v1-gate-pass")
        self.assertEqual(verified["updates"], 240)
        self.assertEqual(gate.champion.read("1v1")["originGate"], "1v0")
        self.assertEqual(gate.existing.read_updates(), 240)
        self.assertEqual(state["gate"], "1v1")

    def test_promote_requires_completed_matching_v7_save(self) -> None:
        stage = sample_gates()["gates"][0]
        self.checkpoint(40)
        path = Path("evaluated.log")
        full = self.evaluation(stage, 0.875)
        path.write_text(full.replace(
            "[BLUE MAPPO V10] Saved actor and training checkpoint at update 40\n",
            ""), encoding="utf-8")
        self.assertIsNone(gate.champion.evaluation_at_update(path, 1, 0, 40))
        path.write_text(full.replace("checkpoint at update 40",
                                     "checkpoint at update 39"), encoding="utf-8")
        self.assertIsNone(gate.champion.evaluation_at_update(path, 1, 0, 40))
        path.write_text(full, encoding="utf-8")
        self.assertEqual(gate.champion.evaluation_at_update(path, 1, 0, 40),
                         {"score": 55.0, "successPercent": 87.5})

    def test_interleaved_evaluation_start_uses_verified_save(self) -> None:
        self.checkpoint(40)
        path = Path("interleaved.log")
        path.write_text(
            "[BLUE MAPPO V10] Saved actor and training checkpoint at update 40\n"
            "[BLUE MAPPO V10] Starting deterministic P=1 N=0 "
            "evaluation after upd[BLUE PATROL] Episode\n"
            "ate 40 (8 cases)\n"
            "[BLUE MAPPO V10] Evaluation #1 P=1 N=0 score=75 success=0.875\n",
            encoding="utf-8")
        self.assertEqual(gate.champion.evaluation_at_update(path, 1, 0, 40),
                         {"score": 75.0, "successPercent": 87.5})

    def test_legacy_import_rejects_latest_newer_than_passing_update(self) -> None:
        self.checkpoint(250)
        config_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": config_hash,
            "gateIndex": 0, "gate": "1v0", "passed": True,
            "elapsedSeconds": 100.0, "blockIndex": 5,
            "lastCheckpointUpdates": 240, "bestSuccessPercent": 87.5,
            "lastSuccessPercent": 87.5, "validatedGates": ["1v0"],
        })
        with self.assertRaisesRegex(ValueError, "does not match its recorded"):
            gate.initialize_state(
                gate.validate_gates(sample_gates()), "1v1",
                True, config_hash)
        self.assertIsNone(gate.champion.read("1v0"))
        self.assertEqual(gate.existing.read_updates(), 250)

    def test_resume_passes_from_existing_evaluated_best_without_updates(self) -> None:
        self.checkpoint(110)
        gate.champion.promote(
            "1v0", {"successPercent": 87.5, "score": 90.0}, 110)
        self.checkpoint(120)  # Newer LATEST must not become the champion.
        config_hash = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": config_hash,
            "gateIndex": 0, "gate": "1v0", "passed": False,
            "elapsedSeconds": 120.0, "blockIndex": 5,
            "lastCheckpointUpdates": 120, "bestSuccessPercent": 87.5,
            "lastSuccessPercent": 25.0, "validatedGates": [],
            "championRequired": True,
        })
        with patch.object(gate.existing, "run_stage") as no_training:
            state = gate.run_gate(self.args(
                gate="1v0", threshold_percent=85, resume=True))
        no_training.assert_not_called()
        self.assertTrue(state["passed"])
        self.assertEqual(gate.existing.read_updates(), 120)
        self.assertEqual(gate.champion.read("1v0")["updates"], 110)
        self.assertEqual(json.loads(gate.RESULT_FILE.read_text())["nextGate"],
                         "1v1")

    def test_both_workflows_publish_and_restart_from_best_pairs(self) -> None:
        root = Path(__file__).resolve().parents[3] / ".github/workflows"
        train = (root / "mappo-training-windows.yml").read_text(encoding="utf-8")
        restart = (root / "mappo-restart-progressive.yml").read_text(
            encoding="utf-8")
        self.assertIn("resources/model/BluePatrol/best/**", train)
        self.assertIn("resources/model/BluePatrol/best/**", restart)
        self.assertIn("champion.restore_latest(source_gate)", restart)
        self.assertIn("champion.seed(start_gate, source_gate)", restart)

    def test_workflow_resume_powershell_argument_splatting(self) -> None:
        """Run 32 regression: optional flags must remain whole argv tokens."""
        workflow = (Path(__file__).resolve().parents[3] /
                    ".github/workflows/mappo-training-windows.yml")
        content = workflow.read_text(encoding="utf-8")
        step = content.split(
            "      - name: Train one progressive MAPPO gate (time- and success-bounded)\n",
            1)[1].split("      - name: Validate resumable checkpoint\n", 1)[0]
        self.assertIn("$pythonArgs = @(", step)
        self.assertIn("& python @pythonArgs", step)
        self.assertIn('$pythonArgs += "--resume"', step)
        self.assertIn('$pythonArgs += "--auto-advance"', step)
        self.assertIn('$pythonArgs += "--per-gate-thresholds"', step)
        self.assertNotIn("@resume @advance", step)
        powershell = shutil.which("pwsh")
        if powershell is None:
            return  # Structural checks still run without PowerShell.
        block = textwrap.dedent(step.split("        run: |\n", 1)[1].split(
            "          if ($LASTEXITCODE", 1)[0])
        cases = (
            ("", "false", "true", ()),
            ("41690261169", "false", "true", ("--resume",)),
            ("41690261169", "true", "true", ("--resume", "--auto-advance")),
            ("", "false", "false", ("--per-gate-thresholds",)),
        )
        for resume_id, auto_advance, same_threshold, flags in cases:
            with self.subTest(resume=resume_id, auto_advance=auto_advance,
                              uniform=same_threshold):
                environment = os.environ.copy()
                environment.update({
                    "GATE": "1v0", "SUCCESS_PERCENT": "75",
                    "MAX_SECONDS": "3600", "RESUME_RUN_ID": resume_id,
                    "AUTO_ADVANCE": auto_advance,
                    "SAME_THRESHOLD_ALL_GATES": same_threshold,
                })
                script = (
                    "function python { $script:captured = @($args) }\n"
                    + block
                    + "\nWrite-Output ('ARGV=' + "
                    "(ConvertTo-Json -InputObject $script:captured -Compress))"
                )
                completed = subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive",
                     "-Command", script],
                    env=environment, capture_output=True, text=True,
                    timeout=20, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                output = completed.stdout.rsplit("ARGV=", 1)[1].strip()
                self.assertEqual(json.loads(output), [
                    "-u", "tools/ci/adaptive_curriculum/progressive_gate.py",
                    "--gate", "1v0", "--threshold-percent", "75",
                    "--max-seconds", "3600", "--block-updates", "20", *flags])

    def test_gate_dispatch_accepts_windows_and_linux_line_endings(self) -> None:
        """Run 33: identical gates must dispatch despite checkout CRLF/LF."""
        config_lf = gate.GATES_FILE.read_bytes().replace(b"\r\n", b"\n")
        config_crlf = config_lf.replace(b"\n", b"\r\n")
        self.assertNotEqual(
            hashlib.sha256(config_lf).hexdigest(),
            hashlib.sha256(config_crlf).hexdigest())
        result = {
            "gateConfigSha256": hashlib.sha256(config_crlf).hexdigest(),
            "gate": "1v0", "passed": True, "autoAdvance": True,
            "nextGate": "1v1", "nextThresholdPercent": 80,
            "nextMaxSeconds": 12000,
        }
        payload = dispatcher.next_dispatch(result, config_lf, "41703576142")
        self.assertEqual(payload["progressive_gate"], "1v1")
        self.assertEqual(payload["resume_run_id"], "41703576142")
        result["gateConfigSha256"] = hashlib.sha256(config_lf).hexdigest()
        self.assertEqual(
            dispatcher.next_dispatch(result, config_crlf, "41703576142"),
            payload)

        # Different valid gate definitions must still fail closed.
        changed = json.loads(config_lf)
        changed["rehearsalEveryBlocks"] += 1
        changed_bytes = json.dumps(changed).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            dispatcher.next_dispatch(result, changed_bytes, "41703576142")
        result["gateConfigSha256"] = None
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            dispatcher.next_dispatch(result, config_lf, "41703576142")


    def test_workflow_gate_choices_match_configuration(self) -> None:
        root = Path(__file__).resolve().parents[3]
        prog = (root / ".github/workflows/mappo-training-windows.yml"
                ).read_text(encoding="utf-8")
        blue = (root / ".github/workflows/mappo-blue-defense-windows.yml"
                ).read_text(encoding="utf-8")
        expected = [item["id"] for item in json.loads(
            (root / "tools/ci/adaptive_curriculum/gates.json").read_text()
        )["gates"]]
        choices = prog.split("      progressive_gate:\n", 1)[1].split(
            "      success_percent:\n", 1)[0]
        self.assertIn("type: choice", choices)
        self.assertEqual(re.findall(r"^          - (\dv\d+)$",
                                    choices, re.MULTILINE), expected)
        self.assertNotIn("blue-patrol", prog + blue)
        self.assertNotIn("scenario:\n", prog.split("permissions:", 1)[0])
        self.assertNotIn("progressive_gate:", blue)
        self.assertIn("Train staged MAPPO curriculum", blue)
        self.assertNotIn("Train staged MAPPO curriculum", prog)
        self.assertIn("Train one progressive MAPPO gate", prog)
        self.assertNotIn("Train one progressive MAPPO gate", blue)

    def test_both_workflows_cache_complete_runtime_and_verify_libtorch(self) -> None:
        root = Path(__file__).resolve().parents[3] / ".github/workflows"
        for name in ("mappo-training-windows.yml",
                     "mappo-blue-defense-windows.yml"):
            with self.subTest(workflow=name):
                contents = (root / name).read_text(encoding="utf-8")
                self.assertIn("actions/cache/restore@v4", contents)
                self.assertIn("actions/cache/save@v4", contents)
                self.assertIn("build/release/Main/Release", contents)
                self.assertIn("torch_cpu.dll", contents)
                self.assertIn("c10.dll", contents)
                self.assertIn("& $exe --example blue-defense --check-example",
                              contents)
                self.assertIn("if: steps.runtime_probe.outputs.usable != 'true'",
                              contents)
                self.assertIn("libtorch291cpu-${{ github.sha }}", contents)

    def test_auto_advance_is_explicit_and_validated(self) -> None:
        config = gate.GATES_FILE.read_bytes()
        digest = hashlib.sha256(config).hexdigest()
        result = {
            "gateConfigSha256": digest,
            "gate": "1v0", "passed": True, "autoAdvance": True,
            "nextGate": "1v1", "nextThresholdPercent": 80,
            "nextMaxSeconds": 12000,
        }
        payload = dispatcher.next_dispatch(result, config, "987654")
        self.assertEqual(payload["progressive_gate"], "1v1")
        self.assertEqual(payload["resume_run_id"], "987654")
        self.assertNotIn("scenario", payload)
        self.assertNotIn("curriculum_enabled", payload)
        self.assertEqual(payload["success_percent"], "80")
        result["autoAdvance"] = False
        self.assertIsNone(dispatcher.next_dispatch(result, config, "987654"))
        result["autoAdvance"] = True
        result["nextGate"] = "4v4"
        with self.assertRaisesRegex(ValueError, "handoff"):
            dispatcher.next_dispatch(result, config, "987654")
        result["nextGate"] = "1v1"
        result["gateConfigSha256"] = "corrupted"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            dispatcher.next_dispatch(result, config, "987654")


    def test_run_51_fixed_threshold_pass_dispatches_run_52(self) -> None:
        """85% over all gates must not depend on a workflow_run event."""
        config = gate.GATES_FILE.read_bytes()
        digest = hashlib.sha256(config).hexdigest()
        state = {
            "gateConfigSha256": digest, "gate": "1v0", "gateIndex": 0,
            "passed": True, "lastCheckpointUpdates": 2560,
            "validatedGates": ["1v0"],
            "restartCycle": {
                "schemaVersion": 1, "bootstrapRunId": 41798655431,
                "sourceRunId": 41768154448, "sameThresholdAllGates": True,
                "successPercent": 85.0, "autoAdvance": True,
            },
        }
        result = {
            "gateConfigSha256": digest, "gate": "1v0", "passed": True,
            "optimizerUpdates": 2560, "autoAdvance": False,
            "thresholdPercent": 85.0, "nextGate": "1v1",
            "nextThresholdPercent": 80, "nextMaxSeconds": 12000,
        }
        inputs = dispatcher.next_dispatch_from_checkpoint(
            result, state, config, "41798688051")
        self.assertEqual(inputs, {
            "resume_run_id": "41798688051", "progressive_gate": "1v1",
            "success_percent": "85", "max_seconds": "12000",
            "save_every_updates": "1", "auto_advance": "false",
            "same_threshold_all_gates": "true",
        })
        # A failed gate remains resumable; it does not dispatch a new run.
        state["passed"] = result["passed"] = False
        state["validatedGates"] = []
        self.assertIsNone(dispatcher.next_dispatch_from_checkpoint(
            result, state, config, "41798688051"))
        # Unchecked automatic continuation stays disabled.
        state["restartCycle"]["autoAdvance"] = False
        state["passed"] = result["passed"] = True
        state["validatedGates"] = ["1v0"]
        self.assertIsNone(dispatcher.next_dispatch_from_checkpoint(
            result, state, config, "41798688051"))

    def test_default_uniform_threshold_is_preserved_across_gates(self) -> None:
        """85% on 1v0 must remain 85% on 1v1 and 4v4, not 80%/75%."""
        config = gate.GATES_FILE.read_bytes()
        digest = hashlib.sha256(config).hexdigest()
        for current, nxt, config_next in (
                ("1v0", "1v1", 80), ("1v1", "4v4", 85)):
            with self.subTest(gate=current):
                state = {
                    "gateConfigSha256": digest, "gate": current,
                    "gateIndex": ["1v0", "1v1", "4v4"].index(current),
                    "lastCheckpointUpdates": 50,
                    "passed": True,
                    "validatedGates": ["1v0"] if current == "1v0"
                    else ["1v0", "1v1"],
                    "sameThresholdAllGates": True,
                    "uniformThresholdPercent": 85.0,
                }
                result = {
                    "gateConfigSha256": digest, "gate": current,
                    "passed": True, "autoAdvance": True,
                    "optimizerUpdates": 50,
                    "thresholdPercent": 85.0,
                    "sameThresholdAllGates": True,
                    "uniformThresholdPercent": 85.0,
                    "nextGate": nxt, "nextThresholdPercent": config_next,
                    "nextMaxSeconds": 14400 if nxt == "4v4" else 12000,
                }
                payload = dispatcher.next_dispatch_from_checkpoint(
                    result, state, config, "34567")
                self.assertEqual(payload["success_percent"], "85")
                self.assertEqual(payload["same_threshold_all_gates"], "true")
                self.assertEqual(payload["progressive_gate"], nxt)
                self.assertEqual(payload["resume_run_id"], "34567")

    def test_uniform_threshold_rejects_corrupt_handoff(self) -> None:
        config = gate.GATES_FILE.read_bytes()
        digest = hashlib.sha256(config).hexdigest()
        state = {
            "gateConfigSha256": digest, "gate": "1v0", "gateIndex": 0,
            "passed": True, "lastCheckpointUpdates": 10,
            "validatedGates": ["1v0"],
            "sameThresholdAllGates": True, "uniformThresholdPercent": 85.0,
        }
        result = {
            "gateConfigSha256": digest, "gate": "1v0", "passed": True,
            "optimizerUpdates": 10, "autoAdvance": True,
            "thresholdPercent": 85.0,
            "sameThresholdAllGates": True,
            "uniformThresholdPercent": 85.0,
            "nextGate": "1v1", "nextThresholdPercent": 80,
            "nextMaxSeconds": 12000,
        }
        for bad in (
                {"uniformThresholdPercent": 80.0},
                {"sameThresholdAllGates": False},
                {"thresholdPercent": 75.0}):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "uniform threshold"):
                    dispatcher.next_dispatch_from_checkpoint(
                        {**result, **bad}, state, config, "34567")

    def test_workflow_uniform_choice_defaults_to_true_and_92(self) -> None:
        root = Path(__file__).resolve().parents[3] / ".github/workflows"
        train = (root / "mappo-training-windows.yml").read_text(encoding="utf-8")
        restart = (root / "mappo-restart-progressive.yml").read_text(encoding="utf-8")
        choice = train.split("      same_threshold_all_gates:\n", 1)[1].split(
            "      auto_advance:\n", 1)[0]
        self.assertIn("type: boolean", choice)
        self.assertIn("default: true", choice)
        threshold = train.split("      success_percent:\n", 1)[1].split(
            "      same_threshold_all_gates:\n", 1)[0]
        self.assertIn("default: 92", threshold)
        self.assertIn("inputs[same_threshold_all_gates]", restart)
        self.assertIn("SAME_THRESHOLD_ALL_GATES", train)

    def test_auto_dispatch_rejects_mixed_cycle_and_checkpoint(self) -> None:
        config = gate.GATES_FILE.read_bytes()
        digest = hashlib.sha256(config).hexdigest()
        state = {
            "gateConfigSha256": digest, "gate": "1v0", "gateIndex": 0,
            "passed": True, "lastCheckpointUpdates": 60,
            "validatedGates": ["1v0"],
            "restartCycle": {
                "schemaVersion": 1, "bootstrapRunId": 10,
                "sourceRunId": 9, "sameThresholdAllGates": True,
                "successPercent": 85.0, "autoAdvance": True,
            },
        }
        result = {
            "gateConfigSha256": digest, "gate": "1v0", "passed": True,
            "optimizerUpdates": 60, "autoAdvance": False,
            "thresholdPercent": 85.0, "nextGate": "1v1",
            "nextThresholdPercent": 80, "nextMaxSeconds": 12000,
        }
        for field, value in (
            ("optimizerUpdates", 61), ("thresholdPercent", 75.0),
            ("autoAdvance", True), ("gate", "4v4")):
            with self.subTest(field=field):
                altered = dict(result, **{field: value})
                with self.assertRaises(ValueError):
                    dispatcher.next_dispatch_from_checkpoint(
                        altered, state, config, "41798688051")
        corrupted = dict(state, gateIndex=1)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            dispatcher.next_dispatch_from_checkpoint(
                result, corrupted, config, "41798688051")
        corrupted = dict(state, restartCycle={
            **state["restartCycle"], "successPercent": 101})
        with self.assertRaisesRegex(ValueError, "threshold"):
            dispatcher.next_dispatch_from_checkpoint(
                result, corrupted, config, "41798688051")

    def test_standard_dispatch_still_uses_next_gate_threshold(self) -> None:
        config = gate.GATES_FILE.read_bytes()
        digest = hashlib.sha256(config).hexdigest()
        state = {
            "gateConfigSha256": digest, "gate": "1v0", "gateIndex": 0,
            "passed": True, "lastCheckpointUpdates": 60,
            "validatedGates": ["1v0"],
        }
        result = {
            "gateConfigSha256": digest, "gate": "1v0", "passed": True,
            "optimizerUpdates": 60, "autoAdvance": True,
            "thresholdPercent": 75, "nextGate": "1v1",
            "nextThresholdPercent": 80, "nextMaxSeconds": 12000,
        }
        inputs = dispatcher.next_dispatch_from_checkpoint(
            result, state, config, "12345")
        self.assertEqual(inputs["success_percent"], "80")
        self.assertEqual(inputs["auto_advance"], "true")
        self.assertEqual(inputs["same_threshold_all_gates"], "false")
        state["restartCycle"] = {
            "schemaVersion": 1, "bootstrapRunId": 10,
            "sourceRunId": 9, "sameThresholdAllGates": False,
            "successPercent": 75.0, "autoAdvance": True,
        }
        self.assertEqual(dispatcher.next_dispatch_from_checkpoint(
            result, state, config, "12345"), inputs)

    def test_per_gate_threshold_changes_after_restart(self) -> None:
        """The bootstrap target must not constrain later per-gate overrides."""
        config = gate.GATES_FILE.read_bytes()
        digest = hashlib.sha256(config).hexdigest()
        state = {
            "gateConfigSha256": digest, "gate": "1v1", "gateIndex": 1,
            "passed": True, "lastCheckpointUpdates": 120,
            "validatedGates": ["1v0", "1v1"],
            "restartCycle": {
                "schemaVersion": 1, "bootstrapRunId": 11,
                "sourceRunId": 10, "sameThresholdAllGates": False,
                "successPercent": 75, "autoAdvance": True,
            },
        }
        result = {
            "gateConfigSha256": digest, "gate": "1v1", "passed": True,
            "optimizerUpdates": 120, "autoAdvance": True,
            "thresholdPercent": 80, "nextGate": "4v4",
            "nextThresholdPercent": 85, "nextMaxSeconds": 14400,
        }
        inputs = dispatcher.next_dispatch_from_checkpoint(
            result, state, config, "123456")
        self.assertEqual(inputs["progressive_gate"], "4v4")
        self.assertEqual(inputs["success_percent"], "85")
        self.assertEqual(inputs["resume_run_id"], "123456")


    def test_production_gates_prune_unrealistic_combinations(self) -> None:
        root = Path(__file__).resolve().parents[3]
        document = json.loads((root / "tools/ci/adaptive_curriculum/gates.json"
                               ).read_text(encoding="utf-8"))
        expected = ["1v0", "2v0", "3v0", "4v0", "1v1", "2v1",
                    "4v1", "2v2", "3v2", "3v3", "4v3", "4v4"]
        self.assertEqual([stage["id"] for stage in document["gates"]],
                         expected)
        self.assertEqual(document["rehearsalEveryBlocks"], 8)
        self.assertEqual((document["gates"][6]["blue"],
                          document["gates"][6]["red"]), (4, 1))
        self.assertEqual(expected[7], "2v2")
        restart = (root / ".github/workflows/mappo-restart-progressive.yml"
                   ).read_text(encoding="utf-8")
        choices = restart.split("      start_gate:\n", 1)[1].split(
            "      success_percent:\n", 1)[0]
        self.assertEqual(re.findall(r"^          - (\dv\d+)$",
                                    choices, re.MULTILINE), expected)
        training = (root / ".github/workflows/mappo-training-windows.yml"
                    ).read_text(encoding="utf-8")
        training_choices = training.split("      progressive_gate:\n", 1)[1].split(
            "      success_percent:\n", 1)[0]
        self.assertEqual(re.findall(r"^          - (\dv\d+)$",
                                    training_choices, re.MULTILINE), expected)

    def test_twenty_evaluation_cases_use_distinct_holdout_seeds(self) -> None:
        root = Path(__file__).resolve().parents[3]
        settings = json.loads((root / "resources/settings/Training/"
                               "BluePatrol.json").read_text(encoding="utf-8"))
        # Existing independent BlueDefense workflow must keep its defaults.
        self.assertEqual(settings["evaluation"]["episodes"], 8)
        workflow = (root / ".github/workflows/mappo-training-windows.yml"
                    ).read_text(encoding="utf-8")
        self.assertIn('-Name episodes -Value 20 -Force', workflow)
        self.assertIn('-Name seed -Value 1004242 -Force', workflow)
        eval_seeds = set(range(1004242, 1004242 + 20))
        self.assertEqual(len(eval_seeds), 20)
        self.assertNotIn(settings["training"]["seed"], eval_seeds)
        self.assertTrue(eval_seeds.isdisjoint(set(range(4242, 4250))))
        # 92% cannot be represented exactly by 20 cases: require 19/20 = 95%.
        self.assertLess(18 / 20 * 100, 92)
        self.assertGreaterEqual(19 / 20 * 100, 92)

    def test_source_commit_fingerprint_permits_retired_best_gate_only(self) -> None:
        legacy = sample_gates()
        previous = {"id": "4v1", "blue": 4, "red": 1,
                    "hostileZones": False, "thresholdPercent": 92,
                    "maxSeconds": 14400, "autoAdvance": True}
        legacy["gates"].insert(-1, previous)
        source = json.dumps(legacy).encode("utf-8")
        digest = hashlib.sha256(source).hexdigest()
        parsed = gate.validate_source_gate_fingerprint(source, digest, "4v1")
        self.assertIn("4v1", [stage["id"] for stage in parsed])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            gate.validate_source_gate_fingerprint(source, "bad", "4v1")
        with self.assertRaisesRegex(ValueError, "absent"):
            gate.validate_source_gate_fingerprint(source, digest, "2v2")
        damaged = source.replace(b'"4v1"', b'"3v1"')
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            gate.validate_source_gate_fingerprint(damaged, digest, "4v1")

    def test_restart_bootstrap_uses_run_62_best_not_cancelled_run_63(self) -> None:
        root = Path(__file__).resolve().parents[3]
        restart = (root / ".github/workflows/mappo-restart-progressive.yml"
                   ).read_text(encoding="utf-8")
        self.assertIn('default: "41865674608"', restart)
        self.assertIn('default: "2v2"', restart)
        self.assertNotIn("41866496491", restart)
        self.assertIn("champion.restore_latest(source_gate)", restart)
        self.assertIn("champion.seed(start_gate, source_gate)", restart)
        self.assertIn("validate_source_gate_fingerprint(", restart)
        self.assertIn("gateConfigSha256=hashlib.sha256(gates_bytes).hexdigest()",
                      restart)
        self.assertIn("validatedGates=rehearsal_history", restart)
        self.assertIn("evaluated 4v1 BEST pair", restart)


    def test_rehearsal_preserves_previous_relevant_gates(self) -> None:
        root = Path(__file__).resolve().parents[3]
        stages = json.loads((root / "tools/ci/adaptive_curriculum/gates.json"
                             ).read_text(encoding="utf-8"))["gates"]
        target = "2v2"
        names = [stage["id"] for stage in stages]
        history = ["1v0", "2v0", "3v0", "4v0", "1v1", "2v1",
                   "3v1", "4v1"]
        retained = [stage_id for stage_id in names[:names.index(target)]
                    if stage_id in history]
        self.assertEqual(retained, ["1v0", "2v0", "3v0", "4v0",
                                    "1v1", "2v1", "4v1"])
        with patch.object(gate.champion, "read",
                          return_value={"evaluatedForGate": True}):
            self.assertEqual(gate.validated_rehearsal_history(
                stages, target, history), retained)
        self.assertEqual(
            gate.select_training_stage(stages, names.index(target),
                                       7, 8, retained)["id"], "1v0")
        restart = (root / ".github/workflows/mappo-restart-progressive.yml"
                   ).read_text(encoding="utf-8")
        self.assertIn("rehearsal_history = validated_rehearsal_history(", restart)
        self.assertIn("source_names.index(gate_id)", restart)
        self.assertIn("validatedGates=rehearsal_history", restart)

    def test_replay_history_requires_an_evaluated_best_for_each_gate(self) -> None:
        stages = gate.validate_gates(sample_gates())
        with patch.object(gate.champion, "read",
                          return_value={"evaluatedForGate": True}) as read:
            self.assertEqual(
                gate.validated_rehearsal_history(
                    stages, "4v4", ["1v0", "1v1", "4v1"]),
                ["1v0", "1v1"])
            self.assertEqual(read.call_count, 2)
        with patch.object(gate.champion, "read", return_value=None):
            with self.assertRaisesRegex(ValueError, "Missing evaluated BEST"):
                gate.validated_rehearsal_history(stages, "4v4", ["1v0"])
        with patch.object(gate.champion, "read",
                          return_value={"evaluatedForGate": False}):
            with self.assertRaisesRegex(ValueError, "Missing evaluated BEST"):
                gate.validated_rehearsal_history(stages, "4v4", ["1v0"])
        with self.assertRaisesRegex(ValueError, "validation list"):
            gate.validated_rehearsal_history(stages, "4v4", ["1v0", "1v0"])
        with self.assertRaisesRegex(ValueError, "Unknown resumed gate"):
            gate.validated_rehearsal_history(stages, "unknown", [])

    def test_only_training_workflow_owns_automatic_gate_dispatch(self) -> None:
        root = Path(__file__).resolve().parents[3] / ".github/workflows"
        training = (root / "mappo-training-windows.yml").read_text(
            encoding="utf-8")
        restart = (root / "mappo-restart-progressive.yml").read_text(
            encoding="utf-8")
        self.assertIn("needs.train.result == 'success' && github.run_attempt == 1",
                      training)
        self.assertIn("name: blue-mappo-checkpoints\n          path: build/mappo-checkpoint",
                      training)
        self.assertNotIn("workflow_run:", restart)
        self.assertNotIn("continue_same_threshold:", restart)
        self.assertIn("same_threshold_all_gates:", restart)


if __name__ == "__main__":
    unittest.main()
