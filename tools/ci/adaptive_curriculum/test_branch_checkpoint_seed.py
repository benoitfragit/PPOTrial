"""Regression tests for opt-in, run-free branch checkpoint CI seeding."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import branch_checkpoint_seed as branch
import dispatch_next_gate as dispatcher


ROOT = Path(__file__).resolve().parents[3]
ACTOR = "BLUE_TEAM_MAPPO_V10 10 10 295 UNIFIED_PHYSICAL_GUIDANCE_V1\n"
TRAINING = "BLUE_TEAM_TRAINING_V7 7 10 295 UNIFIED_PHYSICAL_GUIDANCE_V1\n"
HASHES = {"latestActor": "a" * 64, "latestTraining": "b" * 64,
          "prodActor": "c" * 64}


class BranchCheckpointSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "latest").mkdir()
        (self.root / "prod").mkdir()
        self.actor = self.root / "latest" / branch.ACTOR_NAME
        self.training = self.root / "latest" / branch.TRAINING_NAME
        self.prod = self.root / "prod" / branch.ACTOR_NAME
        self.actor.write_text(ACTOR + "actor weights\n", encoding="utf-8")
        self.training.write_text(TRAINING + "UPDATES 1791\noptimizer\n",
                                 encoding="utf-8")
        self.prod.write_text(ACTOR + "promoted weights\n", encoding="utf-8")

    def test_accepts_distinct_prod_and_complete_latest_seed(self) -> None:
        result = branch.validate_branch_checkpoints(self.root)
        self.assertEqual(result["updates"], 1791)
        self.assertNotEqual(result["sha256"]["latestActor"],
                            result["sha256"]["prodActor"])
        for key, path in (("latestActor", self.actor),
                          ("latestTraining", self.training),
                          ("prodActor", self.prod)):
            self.assertEqual(result["sha256"][key],
                             hashlib.sha256(path.read_bytes()).hexdigest())
        provenance = branch.branch_source_provenance(
            42498140103, "1" * 40, result["sha256"])
        self.assertEqual(provenance["sourceMode"], "branch")
        self.assertIsNone(provenance["sourceRunId"])
        self.assertEqual(provenance["sourceBranchSha"], "1" * 40)

    def test_missing_prod_fails_closed(self) -> None:
        self.prod.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            branch.validate_branch_checkpoints(self.root)

    def test_missing_v7_fails_closed(self) -> None:
        self.training.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            branch.validate_branch_checkpoints(self.root)

    def test_invalid_optimizer_contract_fails_closed(self) -> None:
        self.training.write_text("BLUE_TEAM_TRAINING_V6 6 9\nUPDATES 1\n")
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            branch.validate_branch_checkpoints(self.root)

    def test_zero_completed_updates_fails_closed(self) -> None:
        self.training.write_text(TRAINING + "UPDATES 0\n")
        with self.assertRaisesRegex(ValueError, "no completed PPO"):
            branch.validate_branch_checkpoints(self.root)

    def test_mismatched_prod_architecture_fails_closed(self) -> None:
        self.prod.write_text(ACTOR.replace("295", "296"))
        with self.assertRaisesRegex(ValueError, "architectures differ"):
            branch.validate_branch_checkpoints(self.root)

    def test_invalid_provenance_fails_closed(self) -> None:
        for branch_sha, hashes in (("bad", HASHES),
                                   ("1" * 40, {"latestActor": "a" * 64}),
                                   ("1" * 40, dict(HASHES, prodActor="bad"))):
            with self.subTest(sha=branch_sha, hashes=hashes):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    branch.branch_source_provenance(123, branch_sha, hashes)

    def test_branch_cycle_dispatches_without_fabricated_source_run(self) -> None:
        gates_bytes = (ROOT / "tools/ci/adaptive_curriculum/gates.json"
                       ).read_bytes()
        stages = json.loads(gates_bytes)["gates"]
        current, following = stages[0], stages[1]
        cycle = {
            "schemaVersion": 1, "bootstrapRunId": 42,
            **branch.branch_source_provenance(42, "1" * 40, HASHES),
            "sameThresholdAllGates": True, "successPercent": 85.0,
            "autoAdvance": True, "rehearsalProfile": "balanced_v1",
        }
        digest = hashlib.sha256(gates_bytes).hexdigest()
        state = {
            "gateConfigSha256": digest, "gate": current["id"],
            "gateIndex": 0, "passed": True,
            "lastCheckpointUpdates": 1792, "validatedGates": [current["id"]],
            "rehearsalProfile": "balanced_v1", "restartCycle": cycle,
        }
        result = {
            "gateConfigSha256": digest, "gate": current["id"],
            "passed": True, "optimizerUpdates": 1792,
            "autoAdvance": False, "thresholdPercent": 85.0,
            "nextGate": following["id"],
            "nextThresholdPercent": following["thresholdPercent"],
            "nextMaxSeconds": following["maxSeconds"],
            "rehearsalProfile": "balanced_v1",
        }
        next_inputs = dispatcher.next_dispatch_from_checkpoint(
            result, state, gates_bytes, "9999")
        self.assertEqual(next_inputs["resume_run_id"], "9999")
        self.assertEqual(next_inputs["progressive_gate"], following["id"])
        self.assertEqual(next_inputs["success_percent"], "85")
        invalid = dict(state, restartCycle=dict(cycle, sourceBranchSha="bad"))
        with self.assertRaisesRegex(ValueError, "provenance"):
            dispatcher.next_dispatch_from_checkpoint(
                result, invalid, gates_bytes, "9999")
        invalid = dict(state, restartCycle=dict(cycle, sourceRunId=42))
        with self.assertRaisesRegex(ValueError, "provenance"):
            dispatcher.next_dispatch_from_checkpoint(
                result, invalid, gates_bytes, "9999")

    def test_workflow_branch_bootstrap_3v3_uses_checkout_without_run(self) -> None:
        """Execute the exact embedded bootstrap against an isolated branch tree."""
        restart = (ROOT / ".github/workflows/mappo-restart-progressive.yml"
                   ).read_text(encoding="utf-8").splitlines()
        start = restart.index("          python - <<'PY'") + 1
        end = restart.index("          PY", start)
        script = "\n".join(
            line[10:] if line.startswith("          ") else line
            for line in restart[start:end])
        compile(script, "<branch-bootstrap>", "exec")
        with tempfile.TemporaryDirectory() as folder:
            sandbox = Path(folder)
            model = sandbox / "resources/model/BluePatrol"
            (model / "latest").mkdir(parents=True)
            (model / "prod").mkdir()
            actor = model / "latest" / branch.ACTOR_NAME
            training = model / "latest" / branch.TRAINING_NAME
            prod = model / "prod" / branch.ACTOR_NAME
            actor.write_text(ACTOR + "actor weights\n", encoding="utf-8")
            training.write_text(TRAINING + "UPDATES 1791\noptimizer\n",
                                encoding="utf-8")
            prod.write_text(ACTOR + "production weights\n", encoding="utf-8")
            source_bytes = actor.read_bytes()
            production_bytes = prod.read_bytes()
            gates = sandbox / "tools/ci/adaptive_curriculum/gates.json"
            gates.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / "tools/ci/adaptive_curriculum/gates.json", gates)
            summary = sandbox / "summary.md"
            env = dict(os.environ, **{
                "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "SOURCE_RUN_ID": "branch",
                "GITHUB_SHA": "a" * 40,
                "GITHUB_RUN_ID": "123456",
                "GITHUB_REPOSITORY": "benoitfragit/PPOTrial",
                "GITHUB_STEP_SUMMARY": str(summary),
                "START_GATE": "3v3",
                "MAX_TEAM_SIZE": "4",
                "SCENARIO_VARIANT": "randomized_zones",
                "SUCCESS_PERCENT": "85",
                "SAME_THRESHOLD": "true",
                "REHEARSAL_PROFILE": "balanced_v1",
                "AUTO_ADVANCE": "true",
                "RETRAIN_SELECTED_GATE": "false",
            })
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=sandbox, env=env, capture_output=True, text=True,
                timeout=20, check=False)
            self.assertEqual(completed.returncode, 0,
                             completed.stdout + completed.stderr)
            checkpoint = json.loads((model / "adaptive-curriculum-progress.json"
                                     ).read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["gate"], "3v3")
            self.assertEqual(checkpoint["maxTeamSize"], 4)
            self.assertEqual(checkpoint["lastCheckpointUpdates"], 1791)
            self.assertFalse(checkpoint["passed"])
            self.assertEqual(checkpoint["validatedGates"], [])
            self.assertFalse(checkpoint["championRequired"])
            self.assertEqual(checkpoint["rehearsalProfile"], "balanced_v1")
            self.assertTrue(checkpoint["hostileZones"])
            self.assertEqual(checkpoint["objectiveLayout"], "randomized")
            cycle = checkpoint["restartCycle"]
            self.assertEqual(cycle["sourceMode"], "branch")
            self.assertIsNone(cycle["sourceRunId"])
            self.assertEqual(cycle["sourceBranchSha"], "a" * 40)
            self.assertEqual(cycle["sourceCheckpointSha256"]["latestActor"],
                             hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(cycle["sourceCheckpointSha256"]["prodActor"],
                             hashlib.sha256(production_bytes).hexdigest())
            self.assertEqual(actor.read_bytes(), source_bytes)
            self.assertEqual(prod.read_bytes(), production_bytes)
            self.assertIn("branch LATEST V7 + PROD V10", summary.read_text())

    def test_existing_restart_source_selector_preserves_ten_inputs(self) -> None:
        restart = (ROOT / ".github/workflows/mappo-restart-progressive.yml"
                   ).read_text(encoding="utf-8")
        training = (ROOT / ".github/workflows/mappo-training-windows.yml"
                    ).read_text(encoding="utf-8")
        self.assertEqual(restart.count("      source_run_id:"), 1)
        self.assertEqual(restart.count("      use_branch_checkpoints:"), 0)
        self.assertIn("inputs.source_run_id != 'branch'", restart)
        self.assertIn("get('SOURCE_RUN_ID', '').strip().lower() == 'branch'",
                      restart)
        self.assertIn('default: "41865674608"', restart)
        # GitHub workflow_dispatch permits at most ten independent inputs.
        selection = restart.split("    inputs:\n", 1)[1].split("\npermissions:", 1)[0]
        self.assertEqual(sum(line.startswith("      ") and line.rstrip().endswith(":")
                             and not line.startswith("        ")
                             for line in selection.splitlines()), 10)
        self.assertIn("branch_checkpoint_seed", restart)
        self.assertIn("inputs[resume_run_id]=${GITHUB_RUN_ID}", restart)
        self.assertNotIn("      use_branch_checkpoints:", training)


if __name__ == "__main__":
    unittest.main()
