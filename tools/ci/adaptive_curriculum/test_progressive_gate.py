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
            f"BLUE_TEAM_TRAINING_V7 7 10 200\nUPDATES {number}\n",
            encoding="utf-8")

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
            log_path.write_text(
                f"Evaluation #1 P={stage['blue']} N={stage['red']} "
                f"score=55 success={rate}\n", encoding="utf-8")
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
        self.assertEqual(gate.existing.read_updates(), 6)
        self.assertEqual(gate.existing.TRAINING_FILE.read_bytes(),
                         self.training_bytes)
        self.assertEqual(gate.existing.SCENARIO_FILE.read_bytes(),
                         self.scenario_bytes)

    def test_resumed_new_gate_probes_at_one_update(self) -> None:
        """Only one update is needed to test a restored checkpoint on a new gate."""
        digest = hashlib.sha256(gate.GATES_FILE.read_bytes()).hexdigest()
        self.checkpoint(2560)
        gate.existing.write_json(gate.STATE_FILE, {
            "formatVersion": 1, "gateConfigSha256": digest,
            "gateIndex": 0, "gate": "1v0", "passed": True,
            "elapsedSeconds": 50.0, "blockIndex": 10,
            "lastCheckpointUpdates": 2560, "bestSuccessPercent": 87.5,
            "lastSuccessPercent": 87.5, "validatedGates": ["1v0"],
        })
        checked = []

        def simulate(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> str:
            self.assertEqual(gate.existing.read_updates(), 2560)
            self.assertEqual(stage["id"], "1v1")
            target = json.loads(
                gate.existing.TRAINING_FILE.read_text()
            )["training"]["maxUpdates"]
            checked.append(target)
            self.checkpoint(target)
            log_path.write_text(
                "Evaluation #1 P=1 N=1 score=74 success=0.875\n",
                encoding="utf-8")
            return "evaluation"

        with patch.object(gate.existing, "run_stage", side_effect=simulate):
            state = gate.run_gate(self.args(
                gate="1v1", threshold_percent=85, resume=True,
                block_updates=20))
        self.assertTrue(state["passed"])
        self.assertEqual(checked, [2561])
        self.assertEqual(state["lastCheckpointUpdates"], 2561)
        self.assertEqual(gate.existing.read_updates(), 2561)
        self.assertEqual(state["validatedGates"], ["1v0", "1v1"])

    def test_blocks_end_at_each_evaluation_checkpoint(self) -> None:
        """Do not pass an old actor after later optimizer steps worsened it."""
        updates = []

        def simulate(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> str:
            current = gate.existing.read_updates()
            target = json.loads(
                gate.existing.TRAINING_FILE.read_text()
            )["training"]["maxUpdates"]
            updates.append((current, target))
            self.checkpoint(target)
            success = 0.875 if len(updates) == 3 else 0.25
            log_path.write_text(
                f"Evaluation #1 P=1 N=0 score=80 success={success}\n",
                encoding="utf-8")
            return "evaluation"

        with patch.object(gate.existing, "run_stage", side_effect=simulate):
            state = gate.run_gate(self.args(
                threshold_percent=85, block_updates=20))
        self.assertEqual(updates, [(0, 10), (10, 20), (20, 30)])
        self.assertTrue(state["passed"])
        self.assertEqual(gate.existing.read_updates(), 30)
        self.assertEqual(state["lastSuccessPercent"], 87.5)

    def test_checkpoint_evaluation_cadence_must_be_valid(self) -> None:
        settings = json.loads(
            gate.existing.TRAINING_FILE.read_text(encoding="utf-8"))
        settings["training"]["checkpointEvery"] = 0
        gate.existing.write_json(gate.existing.TRAINING_FILE, settings)
        with self.assertRaisesRegex(ValueError, "checkpointEvery"):
            gate.run_gate(self.args())
        self.assertFalse(gate.STATE_FILE.is_file())

    def test_failure_continues_same_gate_with_fresh_run_budget(self) -> None:
        clock = [0.0]
        attempted = []
        def no_pass(binary: Path, stage: dict, seconds: int,
                    log_path: Path) -> str:
            attempted.append(stage["id"])
            self.checkpoint(gate.existing.read_updates() + 5)
            log_path.write_text(
                "Evaluation #1 P=1 N=0 score=99 success=0.25\n")
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
                log_path.write_text(
                    "Evaluation #1 P=1 N=0 score=55 success=0.875\n")
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
            log_path.write_text(
                f"Evaluation #1 P={selected['blue']} N={selected['red']} "
                "score=100 success=1\n")
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
        self.assertEqual(gate.existing.PROD.read_text(), "candidate 4v4")
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
        self.assertNotIn("@resume @advance", step)
        powershell = shutil.which("pwsh")
        if powershell is None:
            return  # Structural checks still run without PowerShell.
        block = textwrap.dedent(step.split("        run: |\n", 1)[1].split(
            "          if ($LASTEXITCODE", 1)[0])
        cases = (
            ("", "false", ()),
            ("41690261169", "false", ("--resume",)),
            ("41690261169", "true", ("--resume", "--auto-advance")),
        )
        for resume_id, auto_advance, flags in cases:
            with self.subTest(resume=resume_id, auto_advance=auto_advance):
                environment = os.environ.copy()
                environment.update({
                    "GATE": "1v0", "SUCCESS_PERCENT": "75",
                    "MAX_SECONDS": "3600", "RESUME_RUN_ID": resume_id,
                    "AUTO_ADVANCE": auto_advance,
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
