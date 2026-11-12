"""Deterministic CI regression for warm-starting a different scenario variant."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci.adaptive_curriculum import champion_checkpoints as champion  # noqa: E402

ACTOR = "BLUE_TEAM_MAPPO_V10 10 10 295 UNIFIED_PHYSICAL_GUIDANCE_V1\n"
TRAINING = "BLUE_TEAM_TRAINING_V7 7 10 295 UNIFIED_PHYSICAL_GUIDANCE_V1\n"


class RestartCrossVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "resources/model/BluePatrol"
        latest = self.root / "latest"
        latest.mkdir(parents=True)
        (self.root / "prod").mkdir()
        self.actor = latest / champion.ACTOR_NAME
        self.training = latest / champion.TRAINING_NAME
        self.prod = self.root / "prod" / champion.ACTOR_NAME
        self.actor.write_text(ACTOR + "later candidate\n", encoding="utf-8")
        self.training.write_text(
            TRAINING + "UPDATES 1910\nlater optimizer\n", encoding="utf-8")
        self.prod.write_text(ACTOR + "production actor\n", encoding="utf-8")
        self.original_prod = self.prod.read_bytes()
        self.source_best = self.root / "best/4v4"
        self.source_best.mkdir(parents=True)
        self.best_actor = ACTOR.encode() + b"evaluated actor\n"
        self.best_training = TRAINING.encode() + b"UPDATES 1900\nevaluated optimizer\n"
        (self.source_best / champion.ACTOR_NAME).write_bytes(self.best_actor)
        (self.source_best / champion.TRAINING_NAME).write_bytes(self.best_training)
        self.manifest = {
            "version": 1, "gate": "4v4", "originGate": "4v3",
            "evaluatedForGate": True, "updates": 1900,
            "successPercent": 90.0, "score": 80.0,
            "sha256": {
                champion.ACTOR_NAME: hashlib.sha256(self.best_actor).hexdigest(),
                champion.TRAINING_NAME: hashlib.sha256(self.best_training).hexdigest(),
            },
        }
        self.write_manifest()
        self.gates_bytes = (
            ROOT / "tools/ci/adaptive_curriculum/gates.json").read_bytes()
        gate_path = self.workspace / "tools/ci/adaptive_curriculum/gates.json"
        gate_path.parent.mkdir(parents=True)
        gate_path.write_bytes(self.gates_bytes)
        self.source_state = {
            "formatVersion": 1, "gate": "4v4", "passed": True,
            "gateConfigSha256": hashlib.sha256(self.gates_bytes).hexdigest(),
            "lastCheckpointUpdates": 1910, "validatedGates": ["4v4"],
            "championRequired": True,
        }
        self.progress = self.root / "adaptive-curriculum-progress.json"
        self.write_state()

    def write_manifest(self) -> None:
        (self.source_best / "champion.json").write_text(
            json.dumps(self.manifest), encoding="utf-8")

    def write_state(self) -> None:
        self.progress.write_text(json.dumps(self.source_state), encoding="utf-8")

    def bootstrap(self, target_variant: str) -> dict:
        workflow = (ROOT / ".github/workflows/mappo-restart-progressive.yml"
                    ).read_text(encoding="utf-8").splitlines()
        start = workflow.index("          python - <<'PY'") + 1
        end = workflow.index("          PY", start)
        script = "\n".join(
            line[10:] if line.startswith("          ") else line
            for line in workflow[start:end])
        compile(script, "<restart-bootstrap>", "exec")

        def gh_api(command: list[str], **kwargs: object) -> str:
            self.assertEqual(command[:2], ["gh", "api"])
            if "/actions/runs/" in command[2]:
                return json.dumps({"head_sha": "a" * 40})
            if "/contents/tools/ci/adaptive_curriculum/gates.json?" in command[2]:
                return json.dumps({
                    "encoding": "base64",
                    "content": base64.b64encode(self.gates_bytes).decode("ascii"),
                })
            raise AssertionError("Unexpected source provenance lookup")

        env = {
            "SOURCE_RUN_ID": "42639806607",
            "GITHUB_RUN_ID": "12345678",
            "GITHUB_SHA": "b" * 40,
            "GITHUB_REPOSITORY": "benoitfragit/PPOTrial",
            "GITHUB_STEP_SUMMARY": str(self.workspace / "summary.md"),
            "START_GATE": "4v4", "MAX_TEAM_SIZE": "4",
            "SCENARIO_VARIANT": target_variant,
            "SUCCESS_PERCENT": "85", "SAME_THRESHOLD": "true",
            "REHEARSAL_PROFILE": "balanced_v1",
            "AUTO_ADVANCE": "false", "RETRAIN_SELECTED_GATE": "true",
        }
        old_cwd = Path.cwd()
        old_namespace = champion.ROOT
        try:
            os.chdir(self.workspace)
            champion.ROOT = Path("resources/model/BluePatrol/best")
            with (mock.patch.dict(os.environ, env),
                  mock.patch.object(subprocess, "check_output",
                                    side_effect=gh_api),
                  contextlib.redirect_stdout(io.StringIO())):
                exec(script, {"__name__": "__bootstrap__"})
        finally:
            champion.ROOT = old_namespace
            os.chdir(old_cwd)
        return json.loads(self.progress.read_text(encoding="utf-8"))

    def assert_new_campaign(self, variant: str, target: Path) -> None:
        preserved = {p.name: p.read_bytes() for p in self.source_best.iterdir()}
        state = self.bootstrap(variant)
        self.assertFalse(state["passed"])
        self.assertEqual(state["validatedGates"], [])
        self.assertEqual(state["lastCheckpointUpdates"], 1900)
        self.assertEqual(state["rehearsalProfile"], "balanced_v1")
        self.assertEqual(state["restartCycle"]["sourceRunId"], 42639806607)
        self.assertTrue(state["championRequired"])
        self.assertEqual(self.actor.read_bytes(), self.best_actor)
        self.assertEqual(self.training.read_bytes(), self.best_training)
        self.assertEqual(self.prod.read_bytes(), self.original_prod)
        self.assertEqual({p.name: p.read_bytes() for p in self.source_best.iterdir()},
                         preserved)
        seed = json.loads((target / "champion.json").read_text(encoding="utf-8"))
        self.assertFalse(seed["evaluatedForGate"])
        self.assertIsNone(seed["successPercent"])
        self.assertEqual((target / champion.ACTOR_NAME).read_bytes(),
                         self.best_actor)
        self.assertEqual((target / champion.TRAINING_NAME).read_bytes(),
                         self.best_training)

    def test_fixed_to_randomized_no_zones(self) -> None:
        self.assert_new_campaign(
            "randomized_no_zones", self.root / "best/objective-randomized/4v4")

    def test_fixed_to_fixed_zones(self) -> None:
        self.assert_new_campaign(
            "fixed_zones", self.root / "best/zones-on/4v4")

    def test_fixed_to_randomized_zones(self) -> None:
        self.assert_new_campaign(
            "randomized_zones", self.root / "best/objective-randomized/zones-on/4v4")

    def test_randomized_to_zones_preserves_source_best(self) -> None:
        new_source = self.root / "best/objective-randomized/4v4"
        new_source.parent.mkdir(parents=True)
        self.source_best.rename(new_source)
        self.source_best = new_source
        self.source_state["objectiveLayout"] = "randomized"
        self.write_state()
        self.assert_new_campaign(
            "randomized_zones", self.root / "best/objective-randomized/zones-on/4v4")

    def test_randomized_to_fixed_preserves_source_best(self) -> None:
        new_source = self.root / "best/objective-randomized/4v4"
        new_source.parent.mkdir(parents=True)
        self.source_best.rename(new_source)
        self.source_best = new_source
        self.source_state["objectiveLayout"] = "randomized"
        self.write_state()
        self.assert_new_campaign("fixed_no_zones", self.root / "best/4v4")

    def test_missing_or_unevaluated_best_is_rejected(self) -> None:
        self.manifest.update(evaluatedForGate=False,
                             successPercent=None, score=None)
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, "evaluated source BEST"):
            self.bootstrap("randomized_no_zones")
        self.assertEqual(json.loads(self.progress.read_text()), self.source_state)
        self.manifest.update(evaluatedForGate=True,
                             successPercent=90.0, score=80.0)
        self.write_manifest()
        (self.source_best / champion.ACTOR_NAME).unlink()
        with self.assertRaisesRegex(ValueError, "Missing mandatory champion pair"):
            self.bootstrap("randomized_no_zones")
        self.assertEqual(json.loads(self.progress.read_text()), self.source_state)

    def test_same_variant_preserves_existing_certified_best(self) -> None:
        state = self.bootstrap("fixed_no_zones")
        self.assertNotIn("objectiveLayout", state)
        self.assertFalse(state["passed"])
        self.assertEqual(state["lastCheckpointUpdates"], 1900)
        self.assertTrue(json.loads(
            (self.source_best / "champion.json").read_text())["evaluatedForGate"])
        self.assertEqual(self.prod.read_bytes(), self.original_prod)


if __name__ == "__main__":
    unittest.main()
