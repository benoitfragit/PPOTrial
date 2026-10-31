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

    def test_rejects_progress_ahead_of_checkpoint(self) -> None:
        config = json.loads(self.curriculum_path.read_text())
        curriculum.write_json(curriculum.PROGRESS, {
            "formatVersion": 1,
            "curriculumHash": curriculum.fingerprint(config),
            "baselineUpdates": 0,
            "nextStage": 1,
        })
        self.fake_checkpoint(2)  # Stage one requires three updates.
        with self.assertRaisesRegex(ValueError, "behind completed"):
            curriculum.run(self.args())

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


    def v2_config(self) -> dict:
        return {
            "formatVersion": 2,
            "totalUpdates": 15,
            "rehearsal": {"primaryUpdates": 2, "priorUpdates": 1},
            "stages": test_stages(first_updates=3)["stages"][:-1] + [
                {"id": "reference-4v4", "blue": 4, "red": 4,
                 "hostileZones": True, "updates": 12},
            ],
        }

    def test_v2_repository_policy_is_1000_updates_and_opt_in(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        v2_file = (repository / "resources/settings/Training/"
                   "BluePatrolCurriculumV2.json")
        v1_file = (repository / "resources/settings/Training/"
                   "BluePatrolCurriculum.json")
        v2 = json.loads(v2_file.read_text(encoding="utf-8"))
        stages, primary, prior = curriculum.validate_curriculum_v2(v2)
        self.assertEqual(v2["totalUpdates"], 1000)
        self.assertEqual(len(stages), 9)
        self.assertEqual(sum(item["updates"] for item in stages), 1000)
        self.assertEqual(stages[-1]["updates"], 400)
        self.assertEqual((primary, prior), (24, 6))
        self.assertEqual(json.loads(v1_file.read_text())["formatVersion"], 1)

    def test_v2_rehearsal_only_uses_prior_stages(self) -> None:
        cfg = self.v2_config()
        stages, primary, prior = curriculum.validate_curriculum_v2(cfg)
        first, first_end = curriculum.choose_rehearsal_block(
            stages, 0, 0, 3, 0, primary, prior)
        self.assertEqual((first["id"], first_end), ("hippodrome", 3))
        sequence = []
        updates = 3
        while updates < 15:
            selected, next_update = curriculum.choose_rehearsal_block(
                stages, 1, 3, 15, updates, primary, prior)
            sequence.append((selected["id"], updates, next_update))
            self.assertGreater(next_update, updates)
            self.assertLessEqual(next_update, 15)
            updates = next_update
        self.assertEqual(
            [x[0] for x in sequence],
            ["reference-4v4", "hippodrome", "reference-4v4",
             "hippodrome", "reference-4v4", "hippodrome",
             "reference-4v4", "reference-4v4"])
        # Other earlier phases rotate round-robin, never referencing future.
        multi = [
            {"id": str(i), "blue": 1, "red": 0,
             "hostileZones": False, "updates": 20}
            for i in range(5)
        ]
        seen = [
            curriculum.choose_rehearsal_block(
                multi, 4, 100, 220, 102 + cycle * 3,
                2, 1)[0]["id"]
            for cycle in range(4)
        ]
        self.assertEqual(seen, ["0", "1", "2", "3"])

    def test_v2_all_phases_rehearse_and_final_is_mostly_four_v_four(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        file = (repository / "resources/settings/Training/"
                "BluePatrolCurriculumV2.json")
        stages, primary, prior = curriculum.validate_curriculum_v2(
            json.loads(file.read_text(encoding="utf-8")))
        first_update = 0
        for index, stage in enumerate(stages):
            end = first_update + stage["updates"]
            updates = first_update
            primary_count = 0
            earlier_ids = set()
            while updates < end:
                active, next_update = curriculum.choose_rehearsal_block(
                    stages, index, first_update, end,
                    updates, primary, prior)
                self.assertGreater(next_update, updates)
                self.assertLessEqual(next_update, end)
                if active["id"] == stage["id"]:
                    primary_count += next_update - updates
                else:
                    self.assertIn(active, stages[:index])
                    earlier_ids.add(active["id"])
                updates = next_update
            self.assertEqual(updates, end)
            if index == 0:
                self.assertEqual(primary_count, stage["updates"])
            else:
                self.assertGreaterEqual(
                    primary_count / stage["updates"], 0.75)
                self.assertGreater(len(earlier_ids), 0)
            if index == len(stages) - 1:
                self.assertEqual(len(earlier_ids), 8)
                self.assertGreaterEqual(primary_count, 320)
            first_update = end
        self.assertEqual(first_update, 1000)

    def test_v2_invalid_config_and_v1_progress_are_rejected(self) -> None:
        cfg = self.v2_config()
        cfg["totalUpdates"] = 14
        with self.assertRaisesRegex(ValueError, "sum"):
            curriculum.validate_curriculum_v2(cfg)
        cfg["totalUpdates"] = 15
        cfg["rehearsal"]["priorUpdates"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            curriculum.validate_curriculum_v2(cfg)
        cfg = self.v2_config()
        curriculum.write_json(self.curriculum_path, cfg)
        legacy = {
            "formatVersion": 1, "curriculumHash": "old-v1",
            "baselineUpdates": 0, "nextStage": 1,
        }
        curriculum.write_json(curriculum.PROGRESS, legacy)
        self.fake_checkpoint(4)
        with self.assertRaisesRegex(ValueError, "Incompatible V1"):
            curriculum.run(self.args())
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(),
                         self.original_training)

    def test_v2_fresh_100_percent_succeeds_and_preserves_production(self) -> None:
        curriculum.write_json(self.curriculum_path, self.v2_config())
        selected_scenarios = []

        def simulate(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> str:
            self.assertGreaterEqual(seconds, 20)
            current = json.loads(curriculum.TRAINING_FILE.read_text())
            selected_scenarios.append(stage["id"])
            self.fake_checkpoint(current["training"]["maxUpdates"])
            curriculum.PROD.write_text("candidate " + stage["id"])
            curriculum.PROD_SCORE.write_text("candidate score")
            return "Evaluation #1 P=4 N=4 score=75"

        with patch.object(curriculum, "run_stage", side_effect=simulate):
            self.assertEqual(curriculum.run(self.args()), 0)
        self.assertEqual(curriculum.read_updates(), 15)
        self.assertEqual(
            json.loads(curriculum.PROGRESS.read_text())["formatVersion"], 2)
        self.assertEqual(
            json.loads(curriculum.PROGRESS.read_text())["nextStage"], 2)
        self.assertEqual(selected_scenarios.count("hippodrome"), 4)
        self.assertEqual(selected_scenarios.count("reference-4v4"), 5)
        self.assertEqual(curriculum.PROD.read_text(), "candidate reference-4v4")
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(),
                         self.original_training)
        self.assertEqual(curriculum.SCENARIO_FILE.read_bytes(),
                         self.original_scenario)

    def test_v2_partial_block_keeps_training_within_same_run(self) -> None:
        # Regression for run 41192579957: 494 -> 499 stopped before the
        # phase's update 500 despite most of the budget remaining.
        cfg = {
            "formatVersion": 2, "totalUpdates": 8,
            "rehearsal": {"primaryUpdates": 2, "priorUpdates": 1},
            "stages": [
                {"id": "hippodrome", "blue": 1, "red": 0,
                 "hostileZones": False, "updates": 6},
                {"id": "reference-4v4", "blue": 4, "red": 4,
                 "hostileZones": True, "updates": 2},
            ],
        }
        curriculum.write_json(self.curriculum_path, cfg)
        curriculum.write_json(curriculum.PROGRESS, {
            "formatVersion": 2,
            "curriculumHash": curriculum.fingerprint(cfg),
            "baselineUpdates": 494, "nextStage": 0,
            "targetUpdates": 8,
        })
        self.fake_checkpoint(494)
        seen = []

        def partial(binary: Path, stage: dict, seconds: int,
                    log_path: Path) -> None:
            current = curriculum.read_updates()
            target = json.loads(
                curriculum.TRAINING_FILE.read_text())["training"]["maxUpdates"]
            self.assertEqual(seconds, self.args().max_seconds)
            next_update = min(target, current + 5)
            seen.append((stage["id"], current, next_update))
            self.fake_checkpoint(next_update)
            return None

        with patch.object(curriculum.time, "monotonic", return_value=0.0):
            with patch.object(curriculum, "run_stage", side_effect=partial):
                self.assertEqual(curriculum.run(self.args()), 0)
        self.assertEqual(seen, [
            ("hippodrome", 494, 499),
            ("hippodrome", 499, 500),
            ("reference-4v4", 500, 502),
        ])
        self.assertEqual(curriculum.read_updates(), 502)
        self.assertEqual(
            json.loads(curriculum.PROGRESS.read_text())["nextStage"], 2)
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(),
                         self.original_training)
        self.assertEqual(curriculum.SCENARIO_FILE.read_bytes(),
                         self.original_scenario)

    def test_v2_resume_same_block_after_time_budget_exhaustion(self) -> None:
        curriculum.write_json(self.curriculum_path, self.v2_config())
        seen = []
        clock = [0.0]

        def partial(binary: Path, stage: dict, seconds: int,
                    log_path: Path) -> None:
            seen.append(stage["id"])
            self.fake_checkpoint(curriculum.read_updates() + 1)
            clock[0] = 240.0
            return None

        with patch.object(curriculum.time, "monotonic",
                          side_effect=lambda: clock[0]):
            with patch.object(curriculum, "run_stage", side_effect=partial):
                self.assertEqual(curriculum.run(self.args()), 0)
            self.assertEqual(curriculum.read_updates(), 1)
            self.assertEqual(
                json.loads(curriculum.PROGRESS.read_text())["nextStage"], 0)

            clock[0] = 0.0

            def complete(binary: Path, stage: dict, seconds: int,
                         log_path: Path) -> None:
                seen.append(stage["id"])
                target = json.loads(
                    curriculum.TRAINING_FILE.read_text())["training"]["maxUpdates"]
                self.fake_checkpoint(target)
                return None

            with patch.object(curriculum, "run_stage", side_effect=complete):
                self.assertEqual(curriculum.run(self.args()), 0)
        self.assertEqual(seen[0:2], ["hippodrome", "hippodrome"])
        self.assertEqual(curriculum.read_updates(), 15)
        self.assertEqual(
            json.loads(curriculum.PROGRESS.read_text())["nextStage"], 2)


    def test_v3_run16_extension_preserves_optimizer_counter(self) -> None:
        cfg = {
            "formatVersion": 3, "totalUpdates": 12,
            "resumeFromV2Hash": "verified-v2-run16",
            "minimumResumeUpdate": 796,
            "rehearsal": {"primaryUpdates": 2, "priorUpdates": 1},
            "stages": [
                {"id": "combat-4v4", "blue": 4, "red": 4,
                 "hostileZones": True, "updates": 8},
                {"id": "reference-4v4", "blue": 4, "red": 4,
                 "hostileZones": True, "updates": 4},
            ],
        }
        curriculum.write_json(self.curriculum_path, cfg)
        curriculum.write_json(curriculum.PROGRESS, {
            "formatVersion": 2, "curriculumHash": "verified-v2-run16",
            "baselineUpdates": 0, "nextStage": 8, "targetUpdates": 1000,
        })
        self.fake_checkpoint(796)
        clock = [0.0]

        def partial(binary: Path, stage: dict, seconds: int,
                    log_path: Path) -> None:
            self.assertEqual(curriculum.read_updates(), 796)
            self.fake_checkpoint(797)
            clock[0] = 240.0
            return None

        with patch.object(curriculum.time, "monotonic",
                          side_effect=lambda: clock[0]):
            with patch.object(curriculum, "run_stage", side_effect=partial):
                self.assertEqual(curriculum.run(self.args()), 0)
        progress = json.loads(curriculum.PROGRESS.read_text())
        self.assertEqual(progress["formatVersion"], 3)
        self.assertEqual(progress["baselineUpdates"], 796)
        self.assertEqual(progress["targetUpdates"], 12)
        self.assertEqual(progress["nextStage"], 0)
        self.assertEqual(curriculum.read_updates(), 797)
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(),
                         self.original_training)

    def test_v3_refuses_unknown_or_older_checkpoint(self) -> None:
        cfg = {
            "formatVersion": 3, "totalUpdates": 2,
            "resumeFromV2Hash": "verified-v2-run16",
            "minimumResumeUpdate": 796,
            "rehearsal": {"primaryUpdates": 1, "priorUpdates": 1},
            "stages": [
                {"id": "reference-4v4", "blue": 4, "red": 4,
                 "hostileZones": True, "updates": 2},
            ],
        }
        curriculum.write_json(self.curriculum_path, cfg)
        progress = {
            "formatVersion": 2, "curriculumHash": "wrong-v2",
            "baselineUpdates": 0, "nextStage": 8, "targetUpdates": 1000,
        }
        curriculum.write_json(curriculum.PROGRESS, progress)
        self.fake_checkpoint(796)
        with self.assertRaisesRegex(ValueError, "verified run-16"):
            curriculum.run(self.args())
        self.assertEqual(json.loads(curriculum.PROGRESS.read_text()), progress)
        progress["curriculumHash"] = "verified-v2-run16"
        curriculum.write_json(curriculum.PROGRESS, progress)
        self.fake_checkpoint(795)
        with self.assertRaisesRegex(ValueError, "verified run-16"):
            curriculum.run(self.args())

    def test_v3_never_silently_restarts_without_run16_progress(self) -> None:
        cfg = {
            "formatVersion": 3, "totalUpdates": 2,
            "resumeFromV2Hash": "verified-v2-run16",
            "minimumResumeUpdate": 796,
            "rehearsal": {"primaryUpdates": 1, "priorUpdates": 1},
            "stages": [
                {"id": "reference-4v4", "blue": 4, "red": 4,
                 "hostileZones": True, "updates": 2},
            ],
        }
        curriculum.write_json(self.curriculum_path, cfg)
        self.fake_checkpoint(796)
        with self.assertRaisesRegex(ValueError, "never start"):
            curriculum.run(self.args())
        self.assertEqual(curriculum.read_updates(), 796)
        self.assertFalse(curriculum.PROGRESS.exists())

    def test_v3_repository_budget_and_combat_rehearsal(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        cfg = json.loads((repository / "resources/settings/Training/"
                          "BluePatrolCurriculumV3.json").read_text())
        stages, primary, prior = curriculum.validate_curriculum_v2(cfg)
        self.assertEqual(cfg["totalUpdates"], 1200)
        self.assertEqual(sum(stage["updates"] for stage in stages), 1200)
        self.assertEqual(stages[0]["id"], "01-protect-combat-4v4")
        self.assertEqual((stages[-1]["blue"], stages[-1]["red"]), (4, 4))
        self.assertEqual((primary, prior), (24, 6))
        self.assertGreaterEqual(sum(s["updates"] for s in stages
                                    if s["red"] > 0), 700)


if __name__ == "__main__":
    unittest.main()
