"""Regression tests for staged MAPPO orchestration; no simulator build required."""
from __future__ import annotations

import json
import os
import tempfile
import re
import sys
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


class WatchdogTests(unittest.TestCase):
    def test_chatty_process_and_matching_evaluation_are_preserved(self):
        import mappo_process_watchdog as watchdog
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = [sys.executable, "-u", "-c",
                "print('Evaluation #1 P=2 N=2 score=85'); print('ready')"]
            result = watchdog.run(
                command, "2v2", 2.0, root / "training.log",
                root / "combined.log",
                re.compile(r"Evaluation #\d+ P=\d+ N=\d+ score=[0-9.]+"),
                lambda: 15, watchdog.WatchdogLimits(
                    heartbeat_seconds=0.05, quiet_warning_seconds=0.1,
                    stalled_update_seconds=5.0, budget_grace_seconds=0.3,
                    shutdown_seconds=1.0))
            self.assertEqual(result, "Evaluation #1 P=2 N=2 score=85")
            self.assertIn("ready", (root / "training.log").read_text())
            self.assertEqual(json.loads((root / "watchdog.json").read_text())[
                "status"], "completed")

    def test_silent_but_living_process_is_not_killed(self):
        import mappo_process_watchdog as watchdog
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            watchdog.run(
                [sys.executable, "-u", "-c", "import time; time.sleep(0.35)"],
                "2v0", 0.7, root / "training.log",
                root / "combined.log", re.compile("Evaluation"),
                lambda: 20, watchdog.WatchdogLimits(
                    heartbeat_seconds=0.05, quiet_warning_seconds=0.1,
                    stalled_update_seconds=3.0, budget_grace_seconds=0.3,
                    shutdown_seconds=1.0))
            snapshot = json.loads((root / "watchdog.json").read_text())
            self.assertEqual(snapshot["status"], "completed")
            self.assertGreater(snapshot["heartbeatCount"], 0)
            self.assertGreater(snapshot["lastLogAgeSeconds"], 0.1)

    def test_timeout_is_bounded_and_cannot_be_certified(self):
        import mappo_process_watchdog as watchdog
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "budget_timeout"):
                watchdog.run(
                    [sys.executable, "-u", "-c", "import time; time.sleep(10)"],
                    "4v4", 0.15, root / "training.log",
                    root / "combined.log", re.compile("Evaluation"),
                    lambda: 50, watchdog.WatchdogLimits(
                        heartbeat_seconds=0.05, quiet_warning_seconds=0.1,
                        stalled_update_seconds=8.0, budget_grace_seconds=0.1,
                        shutdown_seconds=1.0))
            state = json.loads((root / "watchdog.json").read_text())
            self.assertEqual(state["status"], "incomplete")
            self.assertEqual(state["stopReason"], "budget_timeout")

    def test_explicit_mappo_error_is_never_swallowed(self):
        import mappo_process_watchdog as watchdog
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "mappo_error"):
                watchdog.run(
                    [sys.executable, "-u", "-c",
                     "print('[MAPPO_ERROR] invalid training buffer')"],
                    "1v0", 1.0, root / "training.log",
                    root / "combined.log", re.compile("Evaluation"),
                    lambda: 2, watchdog.WatchdogLimits(
                        heartbeat_seconds=0.05, quiet_warning_seconds=0.2,
                        stalled_update_seconds=3.0, budget_grace_seconds=0.2,
                        shutdown_seconds=1.0))
            self.assertIn("[MAPPO_ERROR]",
                          (root / "training.log").read_text())


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
            f"BLUE_TEAM_TRAINING_V7 7 10 200\nUPDATES {updates}\n",
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
        archived = Path("resources/model/BluePatrol/prod/blue_patrol_generalized_v8.txt")
        archived.write_text("preserved previous candidate")
        curriculum.require_clean_fresh_start()
        self.assertTrue(all(not path.exists()
                            for path in curriculum.FRESH_SEEDS))
        self.assertEqual(red.read_text(), "unchanged red model")
        self.assertEqual(archived.read_text(), "preserved previous candidate")

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


    def v3_config(self) -> dict:
        cfg = test_stages()
        cfg.update(formatVersion=3, totalUpdates=5, observationVersion=10,
                   initialization="scratch",
                   rehearsal={"primaryUpdates": 2, "priorUpdates": 1})
        curriculum.write_json(self.curriculum_path, cfg)
        return cfg

    def test_v3_starts_at_zero_and_resumes_only_its_own_progress(self) -> None:
        cfg = self.v3_config()
        clock = [0.0]
        observed = []

        def partial(binary: Path, stage: dict, seconds: int,
                    log_path: Path) -> None:
            observed.append(curriculum.read_updates())
            self.fake_checkpoint(1)
            clock[0] = 240.0

        args = self.args()
        args.fresh_start = True
        with patch.object(curriculum.time, "monotonic", side_effect=lambda: clock[0]):
            with patch.object(curriculum, "run_stage", side_effect=partial):
                self.assertEqual(curriculum.run(args), 0)
        self.assertEqual(observed, [0])
        progress = json.loads(curriculum.PROGRESS.read_text())
        self.assertEqual(progress["formatVersion"], 3)
        self.assertEqual(progress["baselineUpdates"], 0)
        self.assertEqual(progress["curriculumHash"], curriculum.fingerprint(cfg))
        self.assertEqual(curriculum.read_updates(), 1)

        def finish(binary: Path, stage: dict, seconds: int, log_path: Path) -> None:
            observed.append(curriculum.read_updates())
            training = json.loads(curriculum.TRAINING_FILE.read_text())
            self.fake_checkpoint(training["training"]["maxUpdates"])

        with patch.object(curriculum, "run_stage", side_effect=finish):
            self.assertEqual(curriculum.run(self.args()), 0)
        self.assertEqual(observed[1], 1)
        self.assertEqual(curriculum.read_updates(), 5)
        self.assertEqual(json.loads(curriculum.PROGRESS.read_text())["nextStage"], 2)
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(), self.original_training)
        self.assertEqual(curriculum.SCENARIO_FILE.read_bytes(), self.original_scenario)

    def test_v3_rejects_old_schema_and_previous_curriculum(self) -> None:
        self.v3_config()
        self.fake_checkpoint(796)
        curriculum.LATEST.write_text("BLUE_TEAM_TRAINING_V7 7 9 200\nUPDATES 796\n")
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            curriculum.run(self.args())
        self.assertFalse(curriculum.PROGRESS.exists())
        self.fake_checkpoint(796)
        progress = {"formatVersion": 2, "curriculumHash": "old-run16",
                    "baselineUpdates": 0, "nextStage": 8, "targetUpdates": 1000}
        curriculum.write_json(curriculum.PROGRESS, progress)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            curriculum.run(self.args())
        self.assertEqual(json.loads(curriculum.PROGRESS.read_text()), progress)
        self.assertEqual(curriculum.read_updates(), 796)

    def test_v3_never_discards_a_checkpoint_without_matching_progress(self) -> None:
        self.v3_config()
        self.fake_checkpoint(17)
        with self.assertRaisesRegex(ValueError, "no compatible progress"):
            curriculum.run(self.args())
        self.assertEqual(curriculum.read_updates(), 17)
        self.assertFalse(curriculum.PROGRESS.exists())

    def test_v3_warm_import_rehearses_4v4_and_resumes(self) -> None:
        config = self.v3_config()
        config["stages"][0]["updates"] = 7
        config["totalUpdates"] = 9
        curriculum.write_json(self.curriculum_path, config)
        self.fake_checkpoint(538)  # Full V10 training state from a prior run.
        original_production = curriculum.PROD.read_bytes()
        original_score = curriculum.PROD_SCORE.read_bytes()
        clock = [0.0]
        first_blocks = []

        def interrupt(binary: Path, stage: dict, seconds: int,
                      log_path: Path) -> None:
            first_blocks.append(stage["id"])
            self.fake_checkpoint(
                json.loads(curriculum.TRAINING_FILE.read_text())["training"]["maxUpdates"])
            clock[0] = 240.0

        args = self.args()
        args.warm_start = True
        with patch.object(curriculum.time, "monotonic",
                          side_effect=lambda: clock[0]):
            with patch.object(curriculum, "run_stage", side_effect=interrupt):
                self.assertEqual(curriculum.run(args), 0)
        self.assertEqual(first_blocks, ["hippodrome"])
        self.assertEqual(curriculum.read_updates(), 540)
        progress = json.loads(curriculum.PROGRESS.read_text())
        self.assertEqual(progress["formatVersion"], 3)
        self.assertEqual(progress["initializationSource"], "warm-v10")
        self.assertEqual(progress["baselineUpdates"], 538)
        self.assertEqual(progress["nextStage"], 0)
        self.assertEqual(progress["targetUpdates"], 9)
        self.assertEqual(curriculum.PROD.read_bytes(), original_production)
        self.assertEqual(curriculum.PROD_SCORE.read_bytes(), original_score)

        # The next CI run restores progress and checkpoint; warm-start is OFF.
        args.warm_start = False
        clock[0] = 0.0
        subsequent_blocks = []

        def complete(binary: Path, stage: dict, seconds: int,
                     log_path: Path) -> None:
            subsequent_blocks.append(stage["id"])
            self.fake_checkpoint(
                json.loads(curriculum.TRAINING_FILE.read_text())["training"]["maxUpdates"])

        with patch.object(curriculum.time, "monotonic",
                          side_effect=lambda: clock[0]):
            with patch.object(curriculum, "run_stage", side_effect=complete):
                self.assertEqual(curriculum.run(args), 0)
        self.assertEqual(subsequent_blocks,
                         ["reference-4v4", "hippodrome", "reference-4v4",
                          "hippodrome", "reference-4v4"])
        self.assertEqual(curriculum.read_updates(), 547)
        self.assertEqual(json.loads(curriculum.PROGRESS.read_text())["nextStage"], 2)
        self.assertEqual(curriculum.PROD.read_bytes(), original_production)
        self.assertEqual(curriculum.PROD_SCORE.read_bytes(), original_score)
        self.assertEqual(curriculum.TRAINING_FILE.read_bytes(), self.original_training)
        self.assertEqual(curriculum.SCENARIO_FILE.read_bytes(), self.original_scenario)

    def test_v3_warm_import_rejects_bad_or_ambiguous_state(self) -> None:
        self.v3_config()
        args = self.args()
        args.warm_start = True
        with self.assertRaisesRegex(ValueError, "nonzero V10"):
            curriculum.run(args)
        self.assertFalse(curriculum.PROGRESS.exists())
        self.fake_checkpoint(538)
        curriculum.LATEST.write_text(
            "BLUE_TEAM_TRAINING_V7 7 9 200\nUPDATES 538\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            curriculum.run(args)
        self.fake_checkpoint(538)
        curriculum.write_json(curriculum.PROGRESS, {"formatVersion": 3})
        with self.assertRaisesRegex(ValueError, "without existing curriculum progress"):
            curriculum.run(args)
        curriculum.PROGRESS.unlink()
        args.fresh_start = True
        with self.assertRaisesRegex(ValueError, "without --fresh-start"):
            curriculum.run(args)
        args.fresh_start = False
        curriculum.write_json(self.curriculum_path, self.v2_config())
        with self.assertRaisesRegex(ValueError, "supported by V3 only"):
            curriculum.run(args)
        self.assertEqual(curriculum.read_updates(), 538)

    def test_v3_warm_rehearsal_alternates_reference_with_previous(self) -> None:
        stages = [
            {"id": "patrol", "blue": 1, "red": 0, "hostileZones": False,
             "updates": 200},
            {"id": "wedge", "blue": 2, "red": 0, "hostileZones": False,
             "updates": 200},
            {"id": "three", "blue": 3, "red": 0, "hostileZones": False,
             "updates": 200},
            {"id": "combat", "blue": 4, "red": 4, "hostileZones": True,
             "updates": 200},
        ]
        first = [curriculum.choose_rehearsal_block(
            stages, 0, 538, 738, point, 24, 6, stages[-1])[0]["id"]
            for point in (538, 562, 568, 592)]
        self.assertEqual(first, ["patrol", "combat", "patrol", "combat"])
        second = [curriculum.choose_rehearsal_block(
            stages, 1, 738, 938, point, 24, 6, stages[-1])[0]["id"]
            for point in (762, 792, 822)]
        self.assertEqual(second, ["combat", "patrol", "combat"])
        # Original V3 from-scratch first phase stays uninterrupted.
        self.assertEqual(curriculum.choose_rehearsal_block(
            stages, 0, 0, 200, 48, 24, 6), (stages[0], 200))

    def test_v3_repository_budget_and_patrol_rehearsal(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        cfg = json.loads((repository / "resources/settings/Training/"
                          "BluePatrolCurriculumV3.json").read_text())
        stages, primary, prior = curriculum.validate_curriculum_v2(cfg)
        self.assertEqual(cfg["totalUpdates"], 1200)
        self.assertEqual(sum(stage["updates"] for stage in stages), 1200)
        self.assertEqual((stages[0]["blue"], stages[0]["red"]), (1, 0))
        self.assertEqual((stages[1]["blue"], stages[1]["red"]), (2, 0))
        self.assertEqual((stages[-1]["blue"], stages[-1]["red"]), (4, 4))
        self.assertEqual((primary, prior), (27, 3))
        self.assertEqual(sum(stage["updates"] for stage in stages
                             if stage["red"] == 0), 140)
        self.assertEqual(sum(stage["updates"] for stage in stages
                             if stage["red"] > 0), 1060)
        self.assertEqual(stages[-1]["updates"], 460)
        self.assertTrue(all(count in {s["blue"] for s in stages if s["red"] == 0}
                            for count in (1, 2, 3, 4)))
        self.assertNotIn("resumeFromV2Hash", cfg)



if __name__ == "__main__":
    unittest.main()
