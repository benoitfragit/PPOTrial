"""Regression tests for opt-in, run-free branch checkpoint CI seeding."""
from __future__ import annotations

import hashlib
import json
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

    def test_existing_restart_workflow_switch_is_opt_in(self) -> None:
        restart = (ROOT / ".github/workflows/mappo-restart-progressive.yml"
                   ).read_text(encoding="utf-8")
        training = (ROOT / ".github/workflows/mappo-training-windows.yml"
                    ).read_text(encoding="utf-8")
        self.assertEqual(restart.count("      use_branch_checkpoints:"), 1)
        self.assertIn('        default: false', restart)
        self.assertIn("inputs.use_branch_checkpoints != true", restart)
        self.assertIn("USE_BRANCH_CHECKPOINTS: ${{ inputs.use_branch_checkpoints }}", restart)
        self.assertIn("branch_checkpoint_seed", restart)
        self.assertIn("inputs[resume_run_id]=${GITHUB_RUN_ID}", restart)
        self.assertNotIn("      use_branch_checkpoints:", training)


if __name__ == "__main__":
    unittest.main()
