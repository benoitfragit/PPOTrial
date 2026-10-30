"""Regression tests for staged MAPPO orchestration; no simulator build required."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mappo_curriculum as curriculum


def test_stages(first_updates: int = 3) -> dict:
    return {
        "formatVersion": 1,
        "stages": [
            {"id": "hippodrome", "blue": 1, "red": 0,
             "hostileZones": False, "updates": first_updates},
            {"id": "reference-4v4", "blue": 4, "red": 4,
             "hostileZones": True, "updates": 2},
        ],
    }


class CurriculumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.previous = Path.cwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, self.previous)
        curriculum.TRAINING_FILE.parent.mkdir(parents=True)
        curriculum.SCENARIO_FILE.parent.mkdir(parents=True)
        self.training = {
            "training": {"maxUpdates": 300, "saveEveryUpdates": 1},
            "evaluation": {"episodes": 8},
        }
        self.scenario = {
            "blueDefense": {
                "blueTeam": {"activeCount": 4},
                "redTeam": {"activeCount": 4},
                "restrictedZones": {"enabled": True},
                "restrictedZoneSpawn": {"enabledProbability": 0.75},
            }
        }
        curriculum.write_json(curriculum.TRAINING_FILE, self.training)
        curriculum.write_json(curriculum.SCENARIO_FILE, self.scenario)
        self.original_training = curriculum.TRAINING_FILE.read_bytes()
        self.original_scenario = curriculum.SCENARIO_FILE.read_bytes()
        curriculum.PROD.parent.mkdir(parents=True)
        curriculum.PROD.write_text("baseline 4v4", encoding="utf-8")
        curriculum.PROD_SCORE.write_text("baseline score", encoding="utf-8")
        self.binary = Path("PPO.exe")
        self.binary.write_text("test stub", encoding="utf-8")
        self.curriculum_path = Path("curriculum.json")
        curriculum.write_json(self.curriculum_path, test_stages())

    def args(self) -> SimpleNamespace:
        return SimpleNamespace(binary=str(self.binary),
                               curriculum=str(self.curriculum_path),
                               max_seconds=240, fresh_start=False)

    def fake_checkpoint(self, updates: int) -> None:
        curriculum.LATEST.parent.mkdir(parents=True, exist_ok=True)
        curriculum.LATEST.write_text(
            f"BLUE_TEAM_TRAINING_V6 6 8 200\nUPDATES {updates}\n",
            encoding="utf-8",
        )

    def test_validates_exact_stage_schema_and_counts(self) -> None:
        self.assertEqual(len(curriculum.validate_curriculum(test_stages())), 2)
        invalid = test_stages()
        invalid["stages"][0]["blue"] = True
        with self.assertRaisesRegex(ValueError, "Invalid counts"):
            curriculum.validate_curriculum(invalid)
        invalid = test_stages()
        invalid["stages"][0]["activeCount"] = 1
        with self.assertRaisesRegex(ValueError, "Each stage"):
            curriculum.validate_curriculum(invalid)
        invalid = test_stages()
        invalid["stages"][-1]["blue"] = 3
        with self.assertRaisesRegex(ValueError, "Final stage"):
            curriculum.validate_curriculum(invalid)

    def test_fresh_start_excludes_legacy_blue_seed_only(self) -> None:
        for path in curriculum.FRESH_SEEDS:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("legacy blue seed")
        red = Path("resources/model/RLMission/prod/model.pt")
        red.parent.mkdir(parents=True, exist_ok=True)
        red.write_text("unchanged red model")
        curriculum.require_clean_fresh_start()
        self.assertTrue(all(not path.exists()
                            for path in curriculum.FRESH_SEEDS))
        self.assertEqual(red.read_text(), "unchanged red model")

    def test_fresh_start_rejects_existing_curriculum_progress(self) -> None:
        curriculum.write_json(curriculum.PROGRESS, {"nextStage": 1})
        with self.assertRaisesRegex(ValueError, "progress"):
            curriculum.require_clean_fresh_start()

    def test_checkpoint_counter_is_authoritative(self) -> None:
        self.assertEqual(curriculum.read_updates(), 0)
        self.fake_checkpoint(17)
        self.assertEqual(curriculum.read_updates(), 17)
        curriculum.LATEST.write_text("WRONG\nUPDATES 17\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            curriculum.read_updates()

    def test_configures_scenario_without_mutating_baseline(self) -> None:
        stage = test_stages()["stages"][0]
        curriculum.apply_stage(stage, 25, self.training, self.scenario)
        configured = json.loads(curriculum.SCENARIO_FILE.read_text())
        updated_training = json.loads(curriculum.TRAINING_FILE.read_text())
        self.assertEqual(configured["blueDefense"]["redTeam"]["activeCount"], 0)
        self.assertFalse(configured["blueDefense"]["restrictedZones"]["enabled"])
        self.assertEqual(
            configured["blueDefense"]["restrictedZoneSpawn"]["enabledProbability"],
            0,
        )
        self.assertEqual(updated_training["training"]["maxUpdates"], 25)
        self.assertEqual(self.scenario["blueDefense"]["redTeam"]["activeCount"], 4)

    def test_full_progress_and_production_isolation(self) -> None:
        previous_production = []

        def simulate(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> str:
            self.assertGreater(seconds, 0)
            current = json.loads(curriculum.TRAINING_FILE.read_text())
            previous_production.append(curriculum.PROD.read_text()
                                       if curriculum.PROD.exists() else None)
            self.assertEqual(current["training"]["saveEveryUpdates"], 1)
            self.fake_checkpoint(current["training"]["maxUpdates"])
            curriculum.PROD.write_text("candidate " + stage["id"])
            curriculum.PROD_SCORE.write_text("candidate score")
            return "Evaluation #1 P=4 N=4 score=75"

        with patch.object(curriculum, "run_stage", side_effect=simulate):
            self.assertEqual(curriculum.run(self.args()), 0)
        self.assertEqual(previous_production, [None, "baseline 4v4"])
        self.assertEqual(curriculum.read_updates(), 5)
        self.assertEqual(json.loads(curriculum.PROGRESS.read_text())["nextStage"], 2)
        self.assertEqual(curriculum.PROD.read_text(), "candidate reference-4v4")
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(),
                         self.original_training)
        self.assertEqual(curriculum.SCENARIO_FILE.read_bytes(),
                         self.original_scenario)
        candidate = (curriculum.OUTPUT / "curriculum/01-hippodrome" /
                     curriculum.PROD.name)
        self.assertEqual(candidate.read_text(), "candidate hippodrome")

    def test_partial_progress_can_resume_same_level(self) -> None:
        calls = []

        def simulate(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> None:
            calls.append(stage["id"])
            target = json.loads(
                curriculum.TRAINING_FILE.read_text())["training"]["maxUpdates"]
            current = curriculum.read_updates()
            self.fake_checkpoint(min(target, current + 1))
            return None

        with patch.object(curriculum, "run_stage", side_effect=simulate):
            self.assertEqual(curriculum.run(self.args()), 0)
            self.assertEqual(curriculum.read_updates(), 1)
            self.assertEqual(json.loads(curriculum.PROGRESS.read_text())[
                "nextStage"], 0)
            self.assertEqual(curriculum.run(self.args()), 0)
        self.assertEqual(calls, ["hippodrome", "hippodrome"])
        self.assertEqual(curriculum.read_updates(), 2)

    def test_failure_restores_configuration_and_reference_model(self) -> None:
        def fail(binary: Path, stage: dict, seconds: int,
                 log_path: Path) -> None:
            curriculum.PROD.write_text("bad intermediate candidate")
            raise RuntimeError("simulated failure")

        with patch.object(curriculum, "run_stage", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                curriculum.run(self.args())
        self.assertEqual(curriculum.PROD.read_text(), "baseline 4v4")
        self.assertEqual(curriculum.PROD_SCORE.read_text(), "baseline score")
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(),
                         self.original_training)
        self.assertEqual(curriculum.SCENARIO_FILE.read_bytes(),
                         self.original_scenario)


if __name__ == "__main__":
    unittest.main()
