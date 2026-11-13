"""Deterministic regression tests: repeated failures, progression and recovery."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).with_name("curriculum_ci.py")
FAKE_HEADLESS = '''#!/usr/bin/env python3
import json, os
from pathlib import Path
import sys
args = sys.argv
output = Path(args[args.index("--output") + 1])
stage = args[args.index("--stage") + 1]
target = int(args[args.index("--updates") + 1])
with (output / "calls.txt").open("a") as f: f.write(f"{stage} {target}\\n")
checkpoint = output / "latest" / "flight_hybrid_v1.txt"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_text(f"PPO_FLIGHT_HYBRID_V1 56 5 128 {target}\\n")
(output / "training.csv").write_text("episode,updates\\n1,1\\n")
threshold = json.loads(os.environ["PASS_AT"])[stage]
passed = target >= threshold
if os.environ.get("MALFORMED_GATE") == "1":
    print("[FLIGHT_GATE] incorrect result")
else:
    print(f"[FLIGHT_GATE] stage={stage} successes={19 if passed else 0} episodes=20 passed={str(passed).lower()}")
'''


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.output = root / "output"
        (self.output / "latest").mkdir(parents=True)
        self.checkpoint = self.output / "latest" / "flight_hybrid_v1.txt"
        self.checkpoint.write_text("PPO_FLIGHT_HYBRID_V1 56 5 128 0\n")
        self.config = root / "config.json"
        self.config.write_text(json.dumps({"evaluation": {"episodes": 20, "minimumSuccessRate": .92}}))
        self.fake = root / ("headless.py" if os.name == "nt" else "headless")
        self.fake.write_text(FAKE_HEADLESS)
        self.fake.chmod(0o755)
        self.launcher = self.fake
        if os.name == "nt":
            # Windows CreateProcess cannot execute a Python shebang directly.
            self.launcher = root / "headless.cmd"
            self.launcher.write_text(f'@echo off\r\n"{sys.executable}" "{self.fake}" %*\r\n')
        self.summary = root / "summary.txt"

    def execute(self, max_updates=100, stage="progressive", interval=10, malformed=False):
        env = {**os.environ, "PASS_AT": json.dumps({"loc": 30, "glide": 50, "joint": 70, "joint-mobile": 90}),
               "GITHUB_STEP_SUMMARY": str(self.summary), "MALFORMED_GATE": str(int(malformed))}
        return subprocess.run([sys.executable, str(RUNNER), "--executable", str(self.launcher),
                               "--config", str(self.config), "--output", str(self.output),
                               "--stage", stage, "--updates-per-evaluation", str(interval),
                               "--max-additional-updates", str(max_updates),
                               "--runtime-seconds", "30"], env=env, capture_output=True, text=True)

    def test_progressive_retries_each_gate_until_independently_validated(self):
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "done")
        self.assertEqual((self.output / "curriculum_status.txt").read_text().strip(), "completed")
        self.assertEqual((self.output / "calls.txt").read_text().splitlines(),
                         ["loc 10", "loc 20", "loc 30", "glide 40", "glide 50", "joint 60", "joint 70",
                          "joint-mobile 80", "joint-mobile 90"])
        self.assertEqual(len(list(self.output.glob("training_*.csv"))), 9)
        self.assertEqual((self.output / "curriculum_version.txt").read_text().strip(), "5")
        second = self.execute()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(len((self.output / "calls.txt").read_text().splitlines()), 9)

    def test_fresh_v2_curriculum_starts_without_a_checkpoint(self):
        self.checkpoint.unlink()
        self.checkpoint.parent.rmdir()
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "calls.txt").read_text().splitlines(),
                         ["loc 10", "loc 20", "loc 30", "glide 40", "glide 50", "joint 60", "joint 70",
                          "joint-mobile 80", "joint-mobile 90"])
        self.assertEqual(self.checkpoint.read_text().split()[4], "90")
        self.assertEqual((self.output / "curriculum_status.txt").read_text().strip(), "completed")

    def test_missing_resume_checkpoint_is_rejected_before_training(self):
        self.checkpoint.unlink()
        (self.output / "curriculum_stage.txt").write_text("glide\n")
        result = self.execute()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing checkpoint for saved FlightControl3D curriculum stage", result.stderr)
        self.assertFalse((self.output / "calls.txt").exists())

    def test_budget_limit_is_not_a_false_success_and_can_resume(self):
        first = self.execute(max_updates=20)
        self.assertEqual(first.returncode, 2, first.stdout + first.stderr)
        self.assertEqual(self.checkpoint.read_text().split()[4], "20")
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "loc")
        self.assertEqual((self.output / "curriculum_status.txt").read_text().strip(), "incomplete")
        second = self.execute(max_updates=70)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "done")
        self.assertEqual(self.checkpoint.read_text().split()[4], "90")

    def test_joint_fixed_pass_does_not_complete_mobile_gate(self):
        result = self.execute(max_updates=70)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "joint-mobile")
        self.assertEqual((self.output / "curriculum_status.txt").read_text().strip(), "incomplete")
        result = self.execute(max_updates=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "calls.txt").read_text().splitlines()[-2:],
                         ["joint-mobile 80", "joint-mobile 90"])

    def test_legacy_done_revalidates_fixed_then_mobile_landing(self):
        self.checkpoint.write_text("PPO_FLIGHT_HYBRID_V1 56 5 128 70\n")
        (self.output / "curriculum_stage.txt").write_text("done\n")
        result = self.execute(max_updates=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "calls.txt").read_text().splitlines(),
                         ["joint 80", "joint-mobile 90"])

    def test_manual_joint_stops_before_mobile_and_mobile_is_selectable(self):
        result = self.execute(stage="joint")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "joint-mobile")
        result = self.execute(stage="joint-mobile", max_updates=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "done")

    def test_unknown_curriculum_version_is_rejected(self):
        (self.output / "curriculum_version.txt").write_text("999\n")
        result = self.execute()
        self.assertEqual(result.returncode, 1)
        self.assertIn("curriculum version", result.stderr)
        self.assertFalse((self.output / "calls.txt").exists())

    def test_manual_stage_stops_after_validation_and_saves_next_stage(self):
        result = self.execute(stage="loc")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "glide")
        self.assertEqual((self.output / "calls.txt").read_text().splitlines(),
                         ["loc 10", "loc 20", "loc 30"])

    def test_malformed_gate_result_is_not_promoted(self):
        result = self.execute(malformed=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing or invalid independent gate evaluation", result.stderr)
        self.assertEqual((self.output / "curriculum_stage.txt").read_text().strip(), "loc")

    def test_invalid_checkpoint_is_rejected_before_training(self):
        self.checkpoint.write_text("NOT_A_MODEL 56 5 128 0\n")
        result = self.execute()
        self.assertEqual(result.returncode, 1)
        self.assertIn("checkpoint header", result.stderr)
        self.assertFalse((self.output / "calls.txt").exists())


if __name__ == "__main__":
    unittest.main()
