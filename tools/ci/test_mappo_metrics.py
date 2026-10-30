"""Deterministic smoke tests for the dependency-free MAPPO telemetry report."""
import csv
import tempfile
import unittest
from pathlib import Path

from mappo_metrics import FIELDS, read_rows, sparkline, summary, write_chart


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
                        1 if evaluating else 0, 8, eval_steps, 180, 1)
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


if __name__ == "__main__":
    unittest.main()
