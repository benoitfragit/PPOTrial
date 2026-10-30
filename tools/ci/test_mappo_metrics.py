"""Deterministic smoke tests for the dependency-free MAPPO telemetry report."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mappo_metrics import (
    FIELDS, evaluation_summary, read_eval_episodes, read_rows,
    sparkline, summary, write_chart,
)


class MappoMetricsTests(unittest.TestCase):
    def test_empty_run_still_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(read_rows(root / "missing.csv"), [])
            image = root / "figure.svg"
            write_chart([], image)
            self.assertIn("No telemetry samples", image.read_text(encoding="utf-8"))
            self.assertIn("No live status", summary([], ""))

    def test_progress_and_corrupt_final_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "telemetry.csv"
            def row(t, updates, steps, eval_steps, evaluating):
                return (t, updates, steps, 2048, evaluating, 0.5 if evaluating else 0,
                        1 if evaluating else 0, 8, eval_steps, 180, 1,
                        evaluating, 43.0, 48.0, 0.25, 0.1)
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(FIELDS)
                writer.writerow(row(0, 0, 0, 0, 0))
                writer.writerow(row(20, 0, 200, 0, 0))
                writer.writerow(row(40, 1, 0, 0, 0))
                writer.writerow(row(50, 1, 0, 1024, 1))
                stream.write("60,1,0,bad\n")
            rows = read_rows(source)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[-1]["evaluation_steps"], 1024)
            self.assertTrue(sparkline([0, 1, 2]))
            result = summary(rows, "https://github.com/example/actions/runs/1")
            self.assertIn("Completed PPO updates: **1**", result)
            self.assertIn("Evaluation: **active**", result)
            self.assertIn("https://github.com/example", result)
            image = root / "figure.svg"
            write_chart(rows, image)
            self.assertIn("<polyline", image.read_text(encoding="utf-8"))

    def test_evaluation_episode_causes_in_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.log"
            path.write_text(
                "[MAPPO_EVAL_EPISODE] campaign=1 case=1 blue_alive=1 "
                "terminal=timeout blue_1=alive blue_2=unknown\n"
                "[MAPPO_EVAL_EPISODE] campaign=2 case=1 blue_alive=0 "
                "terminal=leader_lost blue_1=hostile_zone "
                "blue_2=blue_collision\n"
                "[MAPPO_EVAL_EPISODE] campaign=2 case=2 blue_alive=1 "
                "terminal=timeout blue_1=alive blue_2=unknown\n"
                "[MAPPO_EVAL_EPISODE] campaign=broken case=0 "
                "blue_alive=0 terminal=unknown\n",
                encoding="utf-8",
            )
            episodes = read_eval_episodes(path)
            self.assertEqual(len(episodes), 3)
            diagnostics = "\n".join(evaluation_summary(episodes))
            self.assertIn("**#2**", diagnostics)
            self.assertIn("**2 episodes**", diagnostics)
            self.assertIn("blue_collision=1", diagnostics)
            self.assertIn("hostile_zone=1", diagnostics)
            self.assertIn("unknown=1", diagnostics)
            self.assertNotIn("**#1**", diagnostics)
            self.assertIn("No per-episode", evaluation_summary([])[0])
            self.assertIn("Blue loss sources", summary([], "", episodes))

    def test_cli_prints_unicode_when_windows_console_uses_cp1252(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "training.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow({field: 1.0 for field in FIELDS})
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "cp1252"
            command = [
                sys.executable, str(Path(__file__).with_name("mappo_metrics.py")),
                "--csv", str(csv_path), "--out", str(root / "out"),
            ]
            completed = subprocess.run(
                command, env=env, capture_output=True, encoding="utf-8",
                errors="replace", check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("▁", completed.stdout)



if __name__ == "__main__":
    unittest.main()
