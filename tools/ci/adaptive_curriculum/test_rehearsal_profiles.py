"""Regression tests for opt-in MAPPO rehearsal profiles and CI wiring."""
from __future__ import annotations

from collections import Counter
from itertools import groupby
import hashlib
import json
from pathlib import Path
import unittest

import dispatch_next_gate as dispatcher
import progressive_gate as gate
import rehearsal_profiles as profiles


def reduced_gates() -> list[dict]:
    return [
        dict(id=name, blue=blue, red=red, hostileZones=False,
             thresholdPercent=75, maxSeconds=14400, autoAdvance=True)
        for name, blue, red in (
            ("1v0", 1, 0), ("2v0", 2, 0), ("1v1", 1, 1),
            ("2v1", 2, 1), ("2v2", 2, 2))
    ]


class RehearsalProfileTests(unittest.TestCase):
    def test_all_named_profiles_have_exact_twenty_block_shares(self):
        stages = reduced_gates()
        history = [stage["id"] for stage in stages[:-1]]
        for name, weights in profiles.WEIGHTS.items():
            with self.subTest(profile=name):
                selected = [
                    profiles.select_weighted_stage(
                        stages, 4, block, history, name)["id"]
                    for block in range(20)
                ]
                self.assertEqual(Counter(selected), weights)
                self.assertEqual(
                    selected, [
                        profiles.select_weighted_stage(
                            stages, 4, block + 20, history, name)["id"]
                        for block in range(20)
                    ])
                # Rehearsals are interleaved instead of grouped by scenario.
                self.assertLess(max(
                    len(list(group)) for _, group in groupby(selected)), 8)

    def test_balanced_replays_all_prior_gates_but_never_future_gates(self):
        stages = reduced_gates()
        previous = {stage["id"] for stage in stages[:-1]}
        for certified in ([], ["1v0", "1v1"], list(previous)):
            selected = [profiles.select_weighted_stage(
                stages, 4, block, certified, "balanced_v1")["id"]
                for block in range(20)]
            self.assertEqual(set(selected), previous | {"2v2"})
            self.assertEqual(Counter(selected), profiles.WEIGHTS["balanced_v1"])
            self.assertEqual(
                profiles.weighted_cycle_length(stages, 4, certified,
                                               "balanced_v1"), 20)
        # Existing non-balanced profiles keep their certified-only contract.
        for block in range(100):
            selected = profiles.select_weighted_stage(
                stages, 4, block, ["1v0", "1v1"], "patrol_focus_v1")
            self.assertIn(selected["id"], {"1v0", "1v1", "2v2"})
            selected = profiles.select_weighted_stage(
                stages, 2, block, ["1v0"], "patrol_focus_v1")
            self.assertIn(selected["id"], {"1v0", "1v1"})

    def test_reject_invalid_position_or_unknown_stage(self):
        stages = reduced_gates()
        with self.assertRaises(ValueError):
            profiles.select_weighted_stage(
                stages, 4, -1, [], "balanced_v1")
        with self.assertRaises(ValueError):
            profiles.select_weighted_stage(
                stages, 4, 0, [], "not_a_profile")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            profiles.select_weighted_stage(
                [dict(id="4v4")], 0, 0, [], "balanced_v1")

    def test_adaptive_balanced_three_four_and_resume(self):
        document = json.loads(gate.GATES_FILE.read_bytes())
        for max_team_size in (3, 4):
            with self.subTest(size=max_team_size):
                stages = gate.select_curriculum(document, max_team_size)
                index = len(stages) - 1
                history = [s["id"] for s in stages[:index]]
                weights = profiles._adaptive_balanced_weights(
                    stages, index, stages)
                total = sum(weights.values())
                self.assertGreater(weights[stages[index]["id"]], total / 2)
                sequence = [
                    profiles.select_weighted_stage(
                        stages, index, block, history, "balanced_v1")["id"]
                    for block in range(total)
                ]
                self.assertEqual(Counter(sequence), weights)
                self.assertEqual(sequence, [
                    profiles.select_weighted_stage(
                        stages, index, block + total, history, "balanced_v1")["id"]
                    for block in range(total)
                ])
                self.assertIn("2v0", sequence)
                self.assertIn("2v2", sequence)
                self.assertIn(f"{max_team_size}v0", sequence)

    def test_balanced_3v3_and_4v4_replay_all_smaller_sizes(self):
        document = json.loads(gate.GATES_FILE.read_bytes())
        for max_team_size in (3, 4):
            stages = gate.select_curriculum(document, max_team_size)
            names = [stage["id"] for stage in stages]
            targets = (["3v0", "3v3"] if max_team_size == 3
                       else ["3v0", "3v3", "4v0", "4v4"])
            for target in targets:
                index = names.index(target)
                expected = set(names[:index + 1])
                for certified in ([], ["1v0", "2v2"], names[:index]):
                    with self.subTest(max_team_size=max_team_size,
                                      target=target, certified=certified):
                        length = profiles.weighted_cycle_length(
                            stages, index, certified, "balanced_v1")
                        selected = [profiles.select_weighted_stage(
                            stages, index, block, certified, "balanced_v1")["id"]
                            for block in range(length)]
                        self.assertEqual(set(selected), expected)
                        self.assertGreater(selected.count(target), length / 2)
                        self.assertEqual(
                            selected, [profiles.select_weighted_stage(
                                stages, index, block + length, certified,
                                "balanced_v1")["id"] for block in range(length)])
                        self.assertIn("1v0", selected)
                        self.assertIn("2v0", selected)
                        if target.startswith("4v"):
                            self.assertIn("3v0", selected)
                            self.assertIn("3v3", selected)

    def test_adaptive_balanced_without_prior_validation_keeps_replays(self):
        stages = gate.select_curriculum(
            json.loads(gate.GATES_FILE.read_bytes()), 3)
        current = len(stages) - 1
        count = profiles.weighted_cycle_length(stages, current, [],
                                               "balanced_v1")
        selected = [profiles.select_weighted_stage(
            stages, current, block, [], "balanced_v1")["id"]
            for block in range(count)]
        self.assertEqual(set(selected), {stage["id"] for stage in stages})
        self.assertGreater(selected.count("3v3"), count / 2)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            profiles.select_weighted_stage(
                stages, current, 0, [], "combat_focus_v1")

    def test_adaptive_balanced_independent_of_zone_option(self):
        stages = gate.select_curriculum(
            json.loads(gate.GATES_FILE.read_bytes()), 4)
        index = len(stages)-1
        history = [stage["id"] for stage in stages[:-1]]
        with_zones = [dict(s, hostileZones=True) for s in stages]
        for block in range(100):
            self.assertEqual(
                profiles.select_weighted_stage(stages, index, block,
                                               history, "balanced_v1")["id"],
                profiles.select_weighted_stage(with_zones, index, block,
                                               history, "balanced_v1")["id"])

    def test_legacy_schedule_is_unchanged(self):
        stages = reduced_gates()
        for block in range(80):
            selected = gate.select_training_stage(
                stages, 4, block, 8, ["1v0", "2v0", "1v1", "2v1"])
            expected = ("1v0", "2v0", "1v1", "2v1")[
                (block // 8) % 4] if (block + 1) % 8 == 0 else "2v2"
            self.assertEqual(selected["id"], expected)

    def test_dispatch_propagates_profile_and_rejects_mismatch(self):
        stages = reduced_gates()
        config = json.dumps(dict(
            formatVersion=1, rehearsalEveryBlocks=8,
            gates=stages)).encode()
        digest = hashlib.sha256(config).hexdigest()
        state = dict(
            gateConfigSha256=digest, gate="2v1", gateIndex=3,
            passed=True, validatedGates=["1v0", "2v0", "1v1", "2v1"],
            lastCheckpointUpdates=1240, rehearsalProfile="balanced_v1",
            sameThresholdAllGates=False)
        result = dict(
            gateConfigSha256=digest, gate="2v1", passed=True,
            optimizerUpdates=1240, rehearsalProfile="balanced_v1",
            nextGate="2v2", nextThresholdPercent=75,
            nextMaxSeconds=14400, autoAdvance=True,
            sameThresholdAllGates=False)
        self.assertEqual(
            dispatcher.next_dispatch_from_checkpoint(
                result, state, config, "41948019589")["rehearsal_profile"],
            "balanced_v1")
        with self.assertRaisesRegex(ValueError, "profiles differ"):
            dispatcher.next_dispatch_from_checkpoint(
                dict(result, rehearsalProfile="legacy"),
                state, config, "41948019589")

    def test_fixed_threshold_restart_keeps_profile_and_single_dispatch(self):
        stages = reduced_gates()
        config = json.dumps(dict(
            formatVersion=1, rehearsalEveryBlocks=8,
            gates=stages)).encode()
        digest = hashlib.sha256(config).hexdigest()
        cycle = dict(schemaVersion=1, bootstrapRunId=123456,
                     sourceRunId=41948019589, sameThresholdAllGates=True,
                     autoAdvance=True, successPercent=95.0,
                     rehearsalProfile="patrol_focus_v1")
        state = dict(
            gateConfigSha256=digest, gate="2v1", gateIndex=3,
            passed=True, validatedGates=["1v0", "2v0", "1v1", "2v1"],
            lastCheckpointUpdates=1240, rehearsalProfile="patrol_focus_v1",
            restartCycle=cycle)
        result = dict(
            gateConfigSha256=digest, gate="2v1", passed=True,
            optimizerUpdates=1240, rehearsalProfile="patrol_focus_v1",
            nextGate="2v2", nextThresholdPercent=75,
            nextMaxSeconds=14400, autoAdvance=False,
            thresholdPercent=95, sameThresholdAllGates=True,
            uniformThresholdPercent=95.0)
        inputs = dispatcher.next_dispatch_from_checkpoint(
            result, state, config, "41948019589")
        self.assertEqual(inputs["rehearsal_profile"], "patrol_focus_v1")
        self.assertEqual(inputs["auto_advance"], "false")
        self.assertEqual(inputs["success_percent"], "95")

    def test_workflow_choices_are_opt_in_and_legacy_defaults(self):
        root = Path(__file__).resolve().parents[3]
        progressive = (root / ".github/workflows/mappo-training-windows.yml"
                       ).read_text(encoding="utf-8")
        restart = (root / ".github/workflows/mappo-restart-progressive.yml"
                   ).read_text(encoding="utf-8")
        for workflow in (progressive, restart):
            self.assertIn("      rehearsal_profile:", workflow)
            self.assertIn("        default: legacy", workflow)
            for profile in profiles.PROFILES:
                self.assertIn("          - " + profile, workflow)
        self.assertIn(
            '$pythonArgs += @("--rehearsal-profile", $env:REHEARSAL_PROFILE)',
            progressive)
        self.assertIn(
            'inputs[rehearsal_profile]=${REHEARSAL_PROFILE}', restart)


if __name__ == "__main__":
    unittest.main()
