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

    def test_update_target_stops_at_next_evaluation(self) -> None:
        self.assertEqual(local.next_update_target(1790, 10, 10), 1800)
        self.assertEqual(local.next_update_target(1793, 10, 10), 1800)
        self.assertEqual(local.next_update_target(1800, 10, 4), 1804)
        with self.assertRaises(ValueError):
            local.next_update_target(1790, 0, 10)

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
