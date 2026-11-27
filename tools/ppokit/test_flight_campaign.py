"""Exercise the public Flight campaign comparator, including a discrepancy hidden by the aggregates."""
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compare_flight_campaign import compare


class CampaignComparison(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        self.native = folder / "master.csv"
        self.native.write_text(
            "JSBSim startup\nseed,steps,reward,stop,touchdown,crash,stop_m,touchdown_m\n"
            "11,8,4.5,1,1,0,30,12\n12,9,-1.5,0,0,0,0,17\n", encoding="utf-8")
        self.report = folder / "evaluation.json"
        self.episodes = [
            {"seed": 11, "length": 8, "return": 4.5, "terminated": True, "metrics": [1, 1, 0, 0, 30, 12]},
            {"seed": 12, "length": 9, "return": -1.5, "terminated": False, "metrics": [0, 0, 0, 1, None, None]},
        ]

    def write_report(self):
        self.report.write_text(json.dumps({"episodes": self.episodes}), encoding="utf-8")

    def test_altered_returns_preserve_mean_but_fail_the_cli(self):
        self.episodes[0]["return"], self.episodes[1]["return"] = self.episodes[1]["return"], self.episodes[0]["return"]
        self.write_report()
        command = [sys.executable, Path(__file__).with_name("compare_flight_campaign.py"), self.native, self.report]
        difference = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(difference.returncode, 0)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(difference.stdout)))), 2)
        checked = subprocess.run([*command, "--check"], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 1)
        self.assertIn("seed 11, return", checked.stderr)

    def test_reference_formats_and_complete_seed_ranges(self):
        self.write_report()
        paired = io.StringIO()
        compare(self.native, self.report, paired, check=True)
        self.native.write_text(paired.getvalue(), encoding="utf-8")
        self.episodes = self.episodes[1:]
        self.write_report()
        compare(self.native, self.report, io.StringIO(), check=True, seed_range=(12, 12))
        with self.assertRaisesRegex(ValueError, "different or empty campaigns"):
            compare(self.native, self.report, io.StringIO(), check=True, seed_range=(11, 12))
        with self.assertRaisesRegex(ValueError, "one ordered reference episode"):
            compare(self.native, self.report, io.StringIO(), check=True, seed_range=(12, 13))

    def test_episode_boundaries_and_event_distances(self):
        self.episodes[1]["terminated"] = True
        self.write_report()
        with self.assertRaisesRegex(ValueError, "seed 12: episode boundary"):
            compare(self.native, self.report, io.StringIO(), check=True)
        self.episodes[1]["terminated"] = False
        self.episodes[0]["metrics"][4] = None
        self.write_report()
        with self.assertRaisesRegex(ValueError, "seed 11, stop_m"):
            compare(self.native, self.report, io.StringIO(), check=True)


if __name__ == "__main__":
    unittest.main()
