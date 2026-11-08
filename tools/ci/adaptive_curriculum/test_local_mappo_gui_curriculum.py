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
                self.assertTrue(all(s["id"] == stages[index]["id"]
                                    or s["id"] in validated for s in selected))
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
            actor.write_text(
                "BLUE_TEAM_MAPPO_V10 10 10 UNIFIED_PHYSICAL_GUIDANCE_V1\n",
                encoding="utf-8")
            training.write_text(
                "BLUE_TEAM_TRAINING_V7 7 10 UNIFIED_PHYSICAL_GUIDANCE_V1\n"
                "UPDATES 2590\n", encoding="utf-8")
            self.assertEqual(local._validate_checkpoint_pair(
                2590, actor, training, require_prod_match=False), 2590)
            with self.assertRaisesRegex(ValueError, "Expected source update"):
                local._validate_checkpoint_pair(
                    1790, actor, training, require_prod_match=False)

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
                live_actor.write_text(
                    "BLUE_TEAM_MAPPO_V10 10 10 "
                    f"UNIFIED_PHYSICAL_GUIDANCE_V1 {label}\\n".replace(
                        "\\\\n", "\\n"), encoding="utf-8")
                live_training.write_text(
                    "BLUE_TEAM_TRAINING_V7 7 10 "
                    f"UNIFIED_PHYSICAL_GUIDANCE_V1 {label}\\n"
                    f"UPDATES {updates}\\n".replace(
                        "\\\\n", "\\n"), encoding="utf-8")
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
                self.assertEqual(local.curriculum.read_updates(), 101)
                local._activate_session_checkpoint(second, 202)
                self.assertIn("second", live_actor.read_text())
                self.assertEqual(local.curriculum.read_updates(), 202)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    local._activate_session_checkpoint(first, 202)

    def test_corrupted_session_checkpoint_does_not_touch_latest(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_actor = root / "latest" / local.LATEST_ACTOR.name
            live_training = root / "latest" / local.curriculum.LATEST.name
            live_actor.parent.mkdir(parents=True)
            live_actor.write_text(
                "BLUE_TEAM_MAPPO_V10 10 10 UNIFIED_PHYSICAL_GUIDANCE_V1\\n",
                encoding="utf-8")
            live_training.write_text(
                "BLUE_TEAM_TRAINING_V7 7 10 UNIFIED_PHYSICAL_GUIDANCE_V1\\n"
                "UPDATES 4\\n", encoding="utf-8")
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

    def test_gui_files_are_restored_after_a_visual_run(self) -> None:
        self.assertIn(local.GUI_LAYOUT, local.PRESERVED_RUNTIME_FILES)
        self.assertIn(local.RENDERER_SETTINGS,
                      local.PRESERVED_RUNTIME_FILES)
        with tempfile.TemporaryDirectory() as directory:
            layout = Path(directory) / "imgui.ini"
            renderer = Path(directory) / "Renderer.json"
            layout.write_bytes(b"original layout")
            renderer.write_bytes(b"original renderer")
            snapshot = local.FileSnapshot((layout, renderer))

            layout.write_bytes(b"runtime layout")
            renderer.write_bytes(b"runtime renderer")
            snapshot.restore()

            self.assertEqual(layout.read_bytes(), b"original layout")
            self.assertEqual(renderer.read_bytes(), b"original renderer")


if __name__ == "__main__":
    unittest.main()
