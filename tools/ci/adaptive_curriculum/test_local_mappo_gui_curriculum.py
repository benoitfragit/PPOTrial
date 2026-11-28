"""Deterministic tests for the local visual MAPPO curriculum launcher."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

import local_mappo_gui_curriculum as local  # noqa: E402
import blue_checkpoint  # noqa: E402
import blue_checkpoint_fixtures as fixtures  # noqa: E402


class LocalMappoGuiCurriculumTests(unittest.TestCase):
    """Verify that local GUI scheduling remains identical to CI scheduling."""

    def test_balanced_cycle_has_exact_ci_shares(self) -> None:
        stages, target_index, validated = local.load_production_stages()
        schedule = local.balanced_schedule(
            stages, target_index, validated, "balanced_v1")
        identifiers = [stage["id"] for stage in schedule]

        self.assertEqual(len(identifiers), 20)
        self.assertEqual(Counter(identifiers), Counter({
            "1v0": 3, "2v0": 4, "1v1": 2, "2v1": 3, "2v2": 8,
        }))
        self.assertEqual(identifiers, [
            local.profiles.select_weighted_stage(
                stages, target_index, block, validated, "balanced_v1")["id"]
            for block in range(20)
        ])

    def test_resumed_cycle_starts_at_persisted_block(self) -> None:
        stages, target_index, validated = local.load_production_stages()
        resumed = local.balanced_schedule(
            stages, target_index, validated, "balanced_v1", start_block=17)

        self.assertEqual(
            resumed[0]["id"],
            local.profiles.select_weighted_stage(
                stages, target_index, 17, validated, "balanced_v1")["id"])
        self.assertEqual(
            resumed[-1]["id"],
            local.profiles.select_weighted_stage(
                stages, target_index, 36, validated, "balanced_v1")["id"])

    def test_shared_balanced_cycle_length_for_three_and_four_teams(self) -> None:
        for size in (3, 4):
            document = __import__("json").loads(local.adaptive.GATES_FILE.read_bytes())
            stages = local.adaptive.select_curriculum(document, size)
            index = len(stages) - 1
            for validated in ([], [s["id"] for s in stages[:index]]):
                length = local.profiles.weighted_cycle_length(
                    stages, index, validated, "balanced_v1")
                self.assertGreater(length, 0)
                selected = [local.profiles.select_weighted_stage(
                    stages, index, block, validated, "balanced_v1")
                    for block in range(length)]
                self.assertEqual(len(selected), length)
                # Balanced rehearsals are general and independent of
                # branch/run source and the certified gate history.
                self.assertEqual({s["id"] for s in selected},
                                 {s["id"] for s in stages[:index + 1]})
                self.assertGreater(sum(s["id"] == stages[index]["id"]
                                       for s in selected), length // 2)

    def test_ci_gate_prefixes_and_scenario_isolation(self) -> None:
        for size in (3, 4):
            stages, index, history = local.load_production_stages(size)
            self.assertEqual(stages[index]["id"], f"{size}v{size}")
            self.assertEqual(history, [])
            self.assertEqual(len(stages), 9 if size == 3 else 14)
            with self.assertRaises(ValueError):
                local.load_production_stages(size, "1v0", ["1v1"])
        default = local.parse_args([])
        self.assertTrue(local.legacy_shortcut(default))
        self.assertEqual(local.session_state_root(default), default.state_root)
        for size in (2, 3, 4):
            for zones in (False, True):
                for objective in ("fixed", "randomized"):
                    opts = local.parse_args([
                        "--max-team-size", str(size),
                        "--objective-layout", objective,
                        *(["--hostile-zones"] if zones else [])])
                    variant = objective + ("_zones" if zones else "_no_zones")
                    self.assertEqual(local.scenario_variant(opts), variant)
                    if variant != "fixed_no_zones" or size != 2:
                        self.assertEqual(
                            local.session_state_root(opts),
                            (opts.state_root / f"{size}v{size}"
                             / variant / f"{size}v{size}"))

    def test_distinct_gates_and_histories_do_not_share_resume_state(self) -> None:
        first = local.parse_args([
            "--max-team-size", "4", "--start-gate", "4v0",
            "--validated-gates", "1v0,2v0"])
        second = local.parse_args([
            "--max-team-size", "4", "--start-gate", "4v4",
            "--validated-gates", "1v0,2v0"])
        self.assertNotEqual(local.session_state_root(first),
                            local.session_state_root(second))
        self.assertEqual(first.validated_gates, second.validated_gates)
        self.assertEqual(local.session_state_root(first).name, "4v0")
        self.assertEqual(local.session_state_root(second).name, "4v4")

    def test_update_target_stops_at_next_evaluation(self) -> None:
        self.assertEqual(local.next_update_target(1790, 10, 10), 1800)
        self.assertEqual(local.next_update_target(1793, 10, 10), 1800)
        self.assertEqual(local.next_update_target(1800, 10, 4), 1804)
        with self.assertRaises(ValueError):
            local.next_update_target(1790, 0, 10)

    def test_explicit_checkpoint_pair_uses_v7_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor = Path(directory) / "source-v10.txt"
            training = Path(directory) / "source-v7.txt"
            actor.write_text(fixtures.former_policy(), encoding="utf-8")
            training.write_text(fixtures.former_training(2590), encoding="utf-8")
            self.assertEqual(local._validate_checkpoint_pair(
                2590, actor, training, require_prod_match=False), 2590)
            with self.assertRaisesRegex(ValueError, "Expected source update"):
                local._validate_checkpoint_pair(
                    1790, actor, training, require_prod_match=False)
            actor.write_text(fixtures.policy(), encoding="utf-8")
            training.write_text(fixtures.training(2600), encoding="utf-8")
            self.assertEqual(local._validate_checkpoint_pair(
                2600, actor, training, require_prod_match=False), 2600)
            with self.assertRaisesRegex(ValueError, "not a compatible"):
                local._validate_checkpoint_pair(
                    2600, training, actor, require_prod_match=False)

    def test_explicit_source_needs_an_evaluated_matching_best_manifest(self) -> None:
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "best" / "2v2"
            gate.mkdir(parents=True)
            actor = gate / local.champion.ACTOR_NAME
            training = gate / local.champion.TRAINING_NAME
            actor.write_text(fixtures.policy(), encoding="utf-8")
            training.write_text(fixtures.training(2590), encoding="utf-8")
            with self.assertRaises((ValueError, FileNotFoundError)):
                local.validate_evaluated_best_source(actor, training)
            manifest = {
                "version": 1, "gate": "2v2", "originGate": "2v1",
                "evaluatedForGate": True, "updates": 2590,
                "successPercent": 95.0, "score": 60.0,
                "sha256": {
                    actor.name: hashlib.sha256(actor.read_bytes()).hexdigest(),
                    training.name: hashlib.sha256(training.read_bytes()).hexdigest(),
                }}
            (gate / "champion.json").write_text(json.dumps(manifest),
                                                encoding="utf-8")
            self.assertEqual(
                local.validate_evaluated_best_source(actor, training)["updates"],
                2590)
            with self.assertRaisesRegex(ValueError, "one BEST gate directory"):
                local.validate_evaluated_best_source(actor, gate.parent / training.name)
            training.write_text(training.read_text() + "MIXED EPOCH\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                local.validate_evaluated_best_source(actor, training)
            training.write_text(fixtures.training(2590), encoding="utf-8")
            manifest["evaluatedForGate"] = False
            manifest["successPercent"] = None
            manifest["score"] = None
            (gate / "champion.json").write_text(json.dumps(manifest),
                                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluated BEST"):
                local.validate_evaluated_best_source(actor, training)

    def test_explicit_source_may_be_a_former_champion_of_an_earlier_run(self) -> None:
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "best" / "2v2"
            gate.mkdir(parents=True)
            actor = gate / blue_checkpoint.FORMER_POLICY_NAME
            training = gate / blue_checkpoint.FORMER_TRAINING_NAME
            actor.write_text(fixtures.former_policy(2, 2), encoding="utf-8")
            training.write_text(fixtures.former_training(2590, 2, 2), encoding="utf-8")
            (gate / "champion.json").write_text(json.dumps({
                "version": 1, "gate": "2v2", "originGate": "2v1",
                "evaluatedForGate": True, "updates": 2590,
                "successPercent": 95.0, "score": 60.0,
                "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (actor, training)}}), encoding="utf-8")
            self.assertEqual(
                local.validate_evaluated_best_source(actor, training)["updates"], 2590)
            # Once a new pair is there, it is the one champion.read validates.
            (gate / local.champion.ACTOR_NAME).write_text(fixtures.policy(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical"):
                local.validate_evaluated_best_source(actor, training)

    def test_modern_dry_run_requires_a_source_or_saved_session(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            from unittest.mock import patch
            configured = local.parse_args([
                "--max-team-size", "3", "--dry-run",
                "--state-root", folder])
            # No globally existing latest checkpoint can be accidentally used.
            import os
            previous = Path.cwd()
            try:
                with self.assertRaisesRegex(ValueError, "evaluated BEST"):
                    local._run_unlocked(configured)
            finally:
                os.chdir(previous)

    def test_explicit_source_cli_is_opt_in(self) -> None:
        historical = local.parse_args([])
        self.assertEqual(historical.expected_source_updates, 1790)
        self.assertIsNone(historical.source_actor)
        modern = local.parse_args([
            "--max-team-size", "3", "--source-actor", "best-actor.txt",
            "--source-training", "best-training.txt"])
        self.assertIsNone(modern.expected_source_updates)
        self.assertEqual(modern.source_actor.name, "best-actor.txt")
        self.assertEqual(modern.source_training.name, "best-training.txt")

    def test_session_checkpoint_restores_the_correct_campaign(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_actor = root / "latest" / local.LATEST_ACTOR.name
            live_training = root / "latest" / local.curriculum.LATEST.name
            live_actor.parent.mkdir(parents=True)
            def make_pair(label: str, updates: int) -> None:
                live_actor.write_text(fixtures.policy(body=f"{label}\n"),
                                      encoding="utf-8")
                live_training.write_text(fixtures.training(updates, body=f"{label}\n"),
                                         encoding="utf-8")
            with patch.object(local, "LATEST_ACTOR", live_actor), (
                    patch.object(local.curriculum, "LATEST", live_training)):
                first = root / "first"
                second = root / "second"
                first.mkdir()
                second.mkdir()
                make_pair("first", 101)
                local._capture_session_checkpoint(first, 101)
                make_pair("second", 202)
                local._capture_session_checkpoint(second, 202)
                local._activate_session_checkpoint(first, 101)
                self.assertIn("first", live_actor.read_text())
                self.assertEqual(local.curriculum.read_updates(live_training), 101)
                local._activate_session_checkpoint(second, 202)
                self.assertIn("second", live_actor.read_text())
                self.assertEqual(local.curriculum.read_updates(live_training), 202)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    local._activate_session_checkpoint(first, 202)
                # A session saved before the PPOKit artifacts: former names.
                stored = first / local.SESSION_CHECKPOINT
                (stored / live_actor.name).rename(
                    stored / blue_checkpoint.FORMER_POLICY_NAME)
                (stored / live_training.name).rename(
                    stored / blue_checkpoint.FORMER_TRAINING_NAME)
                local._activate_session_checkpoint(first, 101)
                self.assertIn("first", live_actor.read_text())
                self.assertEqual(local.curriculum.read_updates(live_training), 101)

    def test_explicit_source_transaction_rolls_back_late_failure(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            live = root / "latest"
            live.mkdir()
            actor = live / local.LATEST_ACTOR.name
            training = live / local.curriculum.LATEST.name
            actor.write_bytes(b"previous actor")
            training.write_bytes(b"previous training")
            source = root / "source"
            source.mkdir()
            new_actor = source / actor.name
            new_training = source / training.name
            new_actor.write_bytes(b"verified source actor")
            new_training.write_bytes(b"verified source training")
            session = root / "session"
            session.mkdir()
            with patch.object(local, "LATEST_ACTOR", actor), (
                    patch.object(local.curriculum, "LATEST", training)):
                with self.assertRaisesRegex(RuntimeError, "state write"):
                    with local.install_explicit_source_transaction(
                            session, new_actor, new_training):
                        self.assertEqual(actor.read_bytes(),
                                         b"verified source actor")
                        self.assertEqual(training.read_bytes(),
                                         b"verified source training")
                        raise RuntimeError("state write failed")
                self.assertEqual(actor.read_bytes(), b"previous actor")
                self.assertEqual(training.read_bytes(), b"previous training")
                self.assertFalse((session / "latest-before-install").exists())
                with local.install_explicit_source_transaction(
                        session, new_actor, new_training):
                    pass
                self.assertEqual(actor.read_bytes(), b"verified source actor")
                self.assertEqual(training.read_bytes(),
                                 b"verified source training")

    def test_corrupted_session_checkpoint_does_not_touch_latest(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_actor = root / "latest" / local.LATEST_ACTOR.name
            live_training = root / "latest" / local.curriculum.LATEST.name
            live_actor.parent.mkdir(parents=True)
            live_actor.write_text(fixtures.policy(), encoding="utf-8")
            live_training.write_text(fixtures.training(4), encoding="utf-8")
            session = root / "session"
            session.mkdir()
            with patch.object(local, "LATEST_ACTOR", live_actor), (
                    patch.object(local.curriculum, "LATEST", live_training)):
                local._capture_session_checkpoint(session, 4)
                stored = session / local.SESSION_CHECKPOINT / live_actor.name
                stored.write_text(stored.read_text() + "CORRUPTED")
                before = live_actor.read_bytes()
                with self.assertRaisesRegex(ValueError, "checksum"):
                    local._activate_session_checkpoint(session, 4)
                self.assertEqual(live_actor.read_bytes(), before)

    def test_legacy_mode_rejects_explicit_source(self) -> None:
        configured = local.parse_args([
            "--max-team-size", "2", "--source-actor", "best.txt",
            "--source-training", "best-v7.txt"])
        self.assertFalse(local.legacy_shortcut(configured))
        self.assertIn("fixed_no_zones",
                      local.session_state_root(configured).parts)

    def test_exclusive_gui_lock_blocks_parallel_launches(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(local, "ROOT", Path(directory)):
                with local._exclusive_training_lock():
                    with self.assertRaisesRegex(RuntimeError, "already.*active|is active"):
                        with local._exclusive_training_lock():
                            self.fail("Second GUI writer acquired the shared lock")
                with local._exclusive_training_lock():
                    pass

    def test_runtime_files_are_restored_after_a_visual_run(self) -> None:
        self.assertIn(local.curriculum.TRAINING_FILE, local.PRESERVED_RUNTIME_FILES)
        self.assertIn(local.curriculum.PROD, local.PRESERVED_RUNTIME_FILES)
        with tempfile.TemporaryDirectory() as directory:
            training = Path(directory) / "blue_patrol_training.json"
            created = Path(directory) / "prod.score"
            training.write_bytes(b"original training")
            snapshot = local.FileSnapshot((training, created))

            training.write_bytes(b"runtime training")
            created.write_bytes(b"runtime score")
            snapshot.restore()

            self.assertEqual(training.read_bytes(), b"original training")
            self.assertFalse(created.exists())


class LocalPpokitRunTests(unittest.TestCase):
    """A local block is the CI's block, watched in ppokit-viewer."""

    def test_block_is_the_ci_block_and_completes_at_its_target(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "block" / "training.log"
            combined = root / "training.log"
            calls = []

            def ci_block(binary, stage, seconds, log_path):
                calls.append((binary, stage, seconds, log_path))
                log_path.write_text("update 10/10\n", encoding="utf-8")
            stage = {"id": "2v1", "blue": 2, "red": 1}
            with patch.object(local.curriculum, "run_stage", side_effect=ci_block), (
                    patch.object(local.curriculum, "read_updates", return_value=10)):
                self.assertTrue(local._run_block(Path("ppokit-run"), stage, 10, 300, log, combined))
                self.assertFalse(local._run_block(Path("ppokit-run"), stage, 20, 300, log, combined))
            self.assertEqual(calls[0], (Path("ppokit-run"), stage, 300, log))
            self.assertEqual(combined.read_text(encoding="utf-8"), "update 10/10\n" * 2)

            def failure(binary, stage, seconds, log_path):
                log_path.write_text("crash\n", encoding="utf-8")
                raise RuntimeError("Curriculum stage 2v1 incomplete: process_error")
            with patch.object(local.curriculum, "run_stage", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, "process_error"):
                    local._run_block(Path("ppokit-run"), stage, 10, 300, log, combined)
            self.assertTrue(combined.read_text(encoding="utf-8").endswith("crash\n"))

    def test_viewer_and_visualiser_come_from_the_program_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "ppokit_programs_Release.txt"
            self.assertEqual(local.find_programs(registry), (None, None))
            registry.write_text("C:/b/PPOKit/Release/ppokit-run.exe\n"
                                "C:/b/PPOKit/viewer/Release/ppokit-viewer.exe\n"
                                "C:/b/Examples/BlueDefense/Release/blue_visualiser.dll\n",
                                encoding="utf-8")
            viewer, visualiser = local.find_programs(registry)
            self.assertEqual(viewer.name, "ppokit-viewer.exe")
            self.assertEqual(visualiser.name, "blue_visualiser.dll")
        self.assertEqual(local.visualiser_port(), 47001)

    def test_viewer_opens_for_the_session_and_closes_after(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        process = MagicMock()
        process.poll.return_value = None
        options = SimpleNamespace(viewer=True, programs=Path("registry"))
        with patch.object(local, "find_programs",
                          return_value=(Path("ppokit-viewer"), Path("blue_visualiser.so"))), (
                patch.object(local.subprocess, "Popen", return_value=process)) as popen:
            with local.watched(options):
                popen.assert_called_once()
                command = popen.call_args[0][0]
                self.assertEqual(command, ["ppokit-viewer", "--view", "blue_visualiser.so",
                                           "--port", "47001"])
            process.terminate.assert_called_once()
        # Not built, or not asked: training goes on without a window.
        for found, viewer in (((None, None), True), ((Path("v"), Path("b")), False)):
            with patch.object(local, "find_programs", return_value=found), (
                    patch.object(local.subprocess, "Popen")) as popen:
                with local.watched(SimpleNamespace(viewer=viewer, programs=Path("r"))):
                    pass
                popen.assert_not_called()

    def test_cli_defaults_to_ppokit_run_and_the_viewer(self) -> None:
        options = local.parse_args([])
        self.assertTrue(str(options.binary).replace("\\", "/").endswith(
            "build/release/PPOKit/Release/ppokit-run.exe"))
        self.assertTrue(options.viewer)
        self.assertFalse(local.parse_args(["--no-viewer"]).viewer)


if __name__ == "__main__":
    unittest.main()
