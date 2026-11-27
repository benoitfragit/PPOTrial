"""Header checks of Blue MAPPO checkpoints: PPOKit artifacts and the former V7/V10 files."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import blue_checkpoint
import blue_checkpoint_fixtures as fixtures


class BlueCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_training_artifact(self) -> None:
        path = self.write("latest.ppokit", fixtures.training(5250, 2, 1))
        checkpoint = blue_checkpoint.read(path, "training")
        self.assertEqual(checkpoint, blue_checkpoint.BlueCheckpoint(
            "training", False, 2, 1, 5250))
        self.assertEqual(blue_checkpoint.read_updates(path), 5250)
        with self.assertRaisesRegex(ValueError, "Expected a Blue policy"):
            blue_checkpoint.read(path, "policy")

    def test_policy_artifact(self) -> None:
        path = self.write("policy.ppokit", fixtures.policy(1, 0))
        self.assertEqual(blue_checkpoint.read(path, "policy"),
                         blue_checkpoint.BlueCheckpoint("policy", False, 1, 0, 0))
        with self.assertRaisesRegex(ValueError, "Expected a Blue training"):
            blue_checkpoint.read_updates(path)

    def test_former_v7_training(self) -> None:
        path = self.write(blue_checkpoint.FORMER_TRAINING_NAME,
                          fixtures.former_training(1791, 9, 10))
        self.assertEqual(blue_checkpoint.read(path),
                         blue_checkpoint.BlueCheckpoint("training", True, 9, 10, 1791))
        self.assertEqual(blue_checkpoint.read_updates(path), 1791)
        path.write_text(fixtures.former_training(1).replace("UPDATES 1", "UPDATES x"))
        with self.assertRaisesRegex(ValueError, "Malformed update marker"):
            blue_checkpoint.read(path)
        path.write_text(fixtures.former_training(1).replace("UPDATES 1\n", ""))
        with self.assertRaisesRegex(ValueError, "Missing update marker"):
            blue_checkpoint.read(path)

    def test_former_v10_policy(self) -> None:
        path = self.write(blue_checkpoint.FORMER_POLICY_NAME, fixtures.former_policy(4, 4))
        self.assertEqual(blue_checkpoint.read(path, "policy"),
                         blue_checkpoint.BlueCheckpoint("policy", True, 4, 4, None))
        with self.assertRaisesRegex(ValueError, "Expected a Blue training"):
            blue_checkpoint.read(path, "training")

    def test_wrong_contract_is_rejected_in_both_formats(self) -> None:
        cases = {
            "observation.ppokit": (fixtures.artifact("training", 3, observation_version="9"),
                                   "observation schema mismatch"),
            "guidance.ppokit": (fixtures.policy().replace(
                blue_checkpoint.GUIDANCE, "UNIFIED_PHYSICAL_GUIDANCE_V0"), "contract"),
            "critic.ppokit": (fixtures.training(3).replace(" 470 256 3 ", " 470 128 3 "),
                              "contract"),
            "capacity.ppokit": (fixtures.policy().replace(" 9 10 8 1 61", " 9 10 8 1 31"),
                                "contract"),
            "task.ppokit": (fixtures.policy().replace('"blue_patrol"', '"flight_control"'),
                            "Not a Blue patrol artifact"),
            "schema.ppokit": (fixtures.policy().replace("ppokit.artifact/1", "ppokit.artifact/2"),
                              "Unsupported"),
            "former_v7.txt": (fixtures.former_training(3, observation_version="9"),
                              "observation schema mismatch"),
            "former_v7_size.txt": (fixtures.former_training(3).replace(
                " 470 256 3 ", " 470 256 2 ", 1), "contract"),
            "former_v6.txt": (fixtures.former_training(3).replace(
                "BLUE_TEAM_TRAINING_V7 7 ", "BLUE_TEAM_TRAINING_V7 6 ", 1), "Unsupported"),
            "former_v10.txt": (fixtures.former_policy().replace(
                blue_checkpoint.GUIDANCE, "OLD_GUIDANCE"), "contract"),
            "former_v10_short.txt": ("BLUE_TEAM_MAPPO_V10 10 10 295 UNIFIED_PHYSICAL_GUIDANCE_V1\n",
                                     "contract"),
            "unknown.txt": ("WRONG\nUPDATES 17\n", "Unsupported"),
        }
        for name, (text, message) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    blue_checkpoint.read(self.write(name, text))

    def test_team_outside_capacity_is_rejected(self) -> None:
        for blue, red in ((0, 0), (10, 0), (1, 11)):
            with self.subTest(blue=blue, red=red):
                for text in (fixtures.training(1, blue, red), fixtures.policy(blue, red),
                             fixtures.former_training(1, blue, red),
                             fixtures.former_policy(blue, red)):
                    with self.assertRaisesRegex(ValueError, "P/N outside"):
                        blue_checkpoint.read(self.write("team", text))
        self.assertEqual(blue_checkpoint.read(self.write("edge", fixtures.policy(9, 10))).blue, 9)

    def test_missing_or_empty_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            blue_checkpoint.read(self.root / "absent.ppokit")
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            blue_checkpoint.read(self.write("empty.ppokit", ""))

    def test_old_name_fallback(self) -> None:
        folder = self.root / "best/2v2"
        folder.mkdir(parents=True)
        self.assertEqual(blue_checkpoint.locate(folder, blue_checkpoint.POLICY_NAME),
                         folder / blue_checkpoint.POLICY_NAME)
        former = self.write("best/2v2/" + blue_checkpoint.FORMER_POLICY_NAME,
                            fixtures.former_policy())
        self.assertEqual(blue_checkpoint.locate(folder, blue_checkpoint.POLICY_NAME), former)
        current = self.write("best/2v2/" + blue_checkpoint.POLICY_NAME, fixtures.policy())
        self.assertEqual(blue_checkpoint.locate(folder, blue_checkpoint.POLICY_NAME), current)
        self.assertEqual(blue_checkpoint.locate(folder, blue_checkpoint.TRAINING_NAME),
                         folder / blue_checkpoint.TRAINING_NAME)

    def test_adopt_moves_restored_former_files_to_the_new_names(self) -> None:
        # The checkout's new files, then an earlier run's artifact restored over it.
        self.write("latest/latest.ppokit", fixtures.training(10))
        self.write("latest/policy.ppokit", fixtures.policy())
        self.write("prod/policy.ppokit", fixtures.policy())
        restored = {
            "latest/" + blue_checkpoint.FORMER_TRAINING_NAME: fixtures.former_training(538),
            "latest/" + blue_checkpoint.FORMER_POLICY_NAME: fixtures.former_policy(2, 1),
            "prod/" + blue_checkpoint.FORMER_POLICY_NAME: fixtures.former_policy(4, 4),
            "prod/" + blue_checkpoint.FORMER_POLICY_NAME + ".score":
                "BLUE_PATROL_GENERALIZED_EVALUATION_V1 4 4 61.5\n",
        }
        for name, text in restored.items():
            self.write(name, text)
        self.assertEqual(blue_checkpoint.main(["adopt", str(self.root)]), 0)
        self.assertEqual(sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                                if p.is_file()),
                         ["latest/latest.ppokit", "latest/policy.ppokit",
                          "prod/policy.ppokit", "prod/policy.ppokit.score"])
        self.assertEqual(blue_checkpoint.read_updates(self.root / "latest/latest.ppokit"), 538)
        self.assertEqual(blue_checkpoint.read(self.root / "latest/policy.ppokit").blue, 2)
        self.assertEqual((self.root / "prod/policy.ppokit.score").read_text(),
                         restored["prod/" + blue_checkpoint.FORMER_POLICY_NAME + ".score"])
        # Nothing left to adopt: a second call changes nothing.
        self.assertEqual(blue_checkpoint.adopt_former_names(self.root), [])

    def test_command_line_check(self) -> None:
        training = self.write("latest.ppokit", fixtures.training(7))
        former = self.write("former.txt", fixtures.former_training(8))
        self.assertEqual(blue_checkpoint.main(
            ["check", "--kind", "training", str(training), str(former)]), 0)
        self.assertEqual(blue_checkpoint.main(["check", "--kind", "policy", str(training)]), 1)


if __name__ == "__main__":
    unittest.main()
