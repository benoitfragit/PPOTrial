"""Deterministic regression tests for the bounded FlightControl3D training runner."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).with_name("curriculum_ci.py")
ROOT = RUNNER.parents[2]
FAKE_HEADLESS = '''#!/usr/bin/env python3
import json, os
from pathlib import Path
import sys
args = sys.argv
dimensions = "33 5 5 128"
if "--inspect-contract" in args:
    print(json.dumps(dict(observations=33, actions=5, outputs=5, width=128,
                          build=os.environ.get("NATIVE_BUILD", "Release"), signature={"contractVersion": 1})))
    sys.exit(0)
if "--check-checkpoint" in args:
    lines = Path(args[args.index("--check-checkpoint") + 1]).read_text().splitlines()
    if os.environ.get("REJECT_CHECKPOINT") == "1":
        print("incompatible checkpoint signature", file=sys.stderr)
        sys.exit(1)
    print(f"[FLIGHT_CHECKPOINT] compatible=true updates={lines[0].split()[5]}")
    sys.exit(0)
output = Path(args[args.index("--output") + 1])
target = int(args[args.index("--updates") + 1])
with Path(os.environ["ALL_CALLS"]).open("a") as calls: calls.write(f"{target}\\n")
checkpoint = output / "latest" / "flight_policy.txt"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_text(f"PPO_FLIGHT_POLICY_V3 {dimensions} {target}\\nsignature\\noptimizer and RNG\\n")
(output / "training.csv").write_text("episode,updates\\n1,1\\n")
full_task = target >= int(os.environ["FULL_TASK_AT"])
passed = full_task and target >= int(os.environ["PASS_AT"])
easy_success = not full_task and os.environ.get("EASY_LEVEL_SUCCEEDS") == "1"
successes = 19 if passed or easy_success else 0
if os.environ.get("MALFORMED_GATE") == "1":
    print("[FLIGHT_GATE] incorrect result")
else:
    print(f"[FLIGHT_GATE] successes={successes} episodes=20 difficulty={1 if full_task else 0.4} "
          f"passed={str(passed).lower()}")
'''


class TrainingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.output = root / "output"
        self.checkpoint = self.output / "latest" / "flight_policy.txt"
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
        self.all_calls = root / "all_calls.txt"
        self.summary = root / "summary.txt"

    def execute(self, pass_at=30, max_updates=100, interval=10, malformed=False, require_release=False,
                native_build="Release", reject_checkpoint=False, full_task_at=0, easy_level_succeeds=False):
        env = {**os.environ, "PASS_AT": str(pass_at), "ALL_CALLS": str(self.all_calls),
               "FULL_TASK_AT": str(full_task_at), "EASY_LEVEL_SUCCEEDS": str(int(easy_level_succeeds)),
               "GITHUB_STEP_SUMMARY": str(self.summary), "MALFORMED_GATE": str(int(malformed)),
               "NATIVE_BUILD": native_build, "REJECT_CHECKPOINT": str(int(reject_checkpoint))}
        command = [sys.executable, str(RUNNER), "--executable", str(self.launcher),
                   "--config", str(self.config), "--output", str(self.output),
                   "--updates-per-evaluation", str(interval),
                   "--max-additional-updates", str(max_updates), "--runtime-seconds", "30"]
        if require_release:
            command.append("--require-release")
        return subprocess.run(command, env=env, capture_output=True, text=True)

    def calls(self):
        return self.all_calls.read_text().split() if self.all_calls.exists() else []

    def test_fresh_policy_trains_blocks_until_independently_validated(self):
        result = self.execute(pass_at=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls(), ["10", "20", "30"])
        self.assertEqual((self.output / "training_status.txt").read_text().strip(), "completed")
        self.assertTrue((self.output / "training_30.csv").is_file())

    def test_easy_curriculum_level_success_does_not_pass_the_gate(self):
        result = self.execute(pass_at=10, full_task_at=30, easy_level_succeeds=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls(), ["10", "20", "30"])
        self.assertEqual((self.output / "training_status.txt").read_text().strip(), "completed")

    def test_exhausted_budget_is_not_a_false_success(self):
        result = self.execute(pass_at=1000, max_updates=25)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(self.calls(), ["10", "20", "25"])
        self.assertEqual((self.output / "training_status.txt").read_text().strip(), "incomplete")

    def test_resume_continues_from_the_saved_update(self):
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_text("PPO_FLIGHT_POLICY_V3 33 5 5 128 70\nsignature\n")
        result = self.execute(pass_at=80)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls(), ["80"])

    def test_previous_checkpoint_format_is_rejected_before_training(self):
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_text("PPO_FLIGHT_HYBRID_V2 39 5 6 128 70\n")
        result = self.execute()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.calls(), [])

    def test_natively_incompatible_checkpoint_is_rejected_before_training(self):
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_text("PPO_FLIGHT_POLICY_V3 33 5 5 128 70\nsignature\n")
        result = self.execute(reject_checkpoint=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.calls(), [])

    def test_malformed_validation_is_not_promoted(self):
        result = self.execute(malformed=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("invalid independent validation", result.stderr)

    def test_non_release_binary_fails_before_training(self):
        result = self.execute(require_release=True, native_build="Debug")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.calls(), [])


class RepositoryWiringTests(unittest.TestCase):
    def test_workflow_trains_the_t6_with_ppokit_run_until_validated(self):
        workflow = (ROOT / ".github" / "workflows" / "flight-control-training.yml").read_text()
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", workflow)
        self.assertIn("FLIGHT_PROJECT: T6", workflow)
        self.assertNotIn("PC7", workflow)
        self.assertIn('ppokit-run train --project "build/flight-training/$FLIGHT_PROJECT"', workflow)
        self.assertIn("--until-passed validation", workflow)
        self.assertIn("--max-seconds", workflow)
        self.assertIn("name: flight-control-training-T6", workflow)
        self.assertIn("build/flight-training/T6/checkpoints/**", workflow)
        self.assertNotIn("curriculum_ci.py", workflow)

    def test_pc7_workflow_is_isolated_from_the_t6(self):
        workflow = (ROOT / ".github" / "workflows" / "flight-control-pc7-training.yml").read_text()
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", workflow)
        self.assertIn("flight-pc7-curriculum", workflow)
        self.assertIn("flight-pc7-resume-", workflow)
        self.assertIn("group: flight-control-pc7-training-", workflow)
        self.assertIn("cp -r Examples/FlightControl3D/projects/PC7 build/flight-pc7-training/PC7", workflow)
        self.assertIn("ppokit-run train --project build/flight-pc7-training/PC7", workflow)
        self.assertIn("--until-passed validation", workflow)
        self.assertIn("name: flight-control-pc7-training", workflow)
        self.assertIn("build/flight-pc7-training/PC7/checkpoints/**", workflow)
        self.assertNotIn("projects/T6", workflow)
        self.assertNotIn("FlightControlHeadless", workflow)
        self.assertNotIn("curriculum_ci.py", workflow)

    def test_launchers_open_their_project_in_the_studio(self):
        for launcher_name, project in (("Lancer_FlightControl3D_Curriculum_Release.bat", "projects\\t6"),
                                       ("Lancer_FlightControl3D_PC7_Curriculum_Release.bat", "projects\\pc7")):
            launcher = (ROOT / launcher_name).read_text().lower()
            self.assertIn("ppokit-studio", launcher)
            self.assertIn("flight_visualiser", launcher)
            self.assertIn(project, launcher)
            self.assertNotIn("flightcontrol3d.exe", launcher)

if __name__ == "__main__":
    unittest.main()
