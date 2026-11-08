"""Tests for one canonical MAPPO schedule and CI max_team_size selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import progressive_gate as gate
import champion_checkpoints as champion
import dispatch_next_gate as dispatcher


ROOT = Path(__file__).resolve().parents[3]
GATES = ROOT / "tools/ci/adaptive_curriculum/gates.json"


class CurriculumSelectionTests(unittest.TestCase):
    def test_single_schedule_includes_exact_fourteen_gates(self):
        source = GATES.read_bytes()
        document = json.loads(source)
        self.assertEqual(document["formatVersion"], 2)
        self.assertEqual(document["contractId"], "triangular-v1")
        expected = ["1v0", "2v0", "1v1", "2v1", "2v2"]
        for blue in (3, 4):
            expected.extend(f"{blue}v{red}" for red in range(blue + 1))
        self.assertEqual([s["id"] for s in gate.validate_gates(document)], expected)
        self.assertEqual(len(document["gates"]), 14)
        self.assertTrue(all(s["red"] <= s["blue"] for s in document["gates"]))
        self.assertTrue(all(not s["hostileZones"] for s in document["gates"]))

    def test_only_team_size_selects_two_three_four(self):
        document = json.loads(GATES.read_bytes())
        for size, count in ((2, 5), (3, 9), (4, 14)):
            with self.subTest(max_team_size=size):
                selected = gate.select_curriculum(document, size)
                self.assertEqual(len(selected), count)
                self.assertEqual(selected[-1]["id"], f"{size}v{size}")
                self.assertEqual(selected, document["gates"][:count])
        for invalid in (1, 5, "4", True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                gate.select_curriculum(document, invalid)

    def test_invalid_gate_order_and_red_majority_are_rejected(self):
        document = json.loads(GATES.read_bytes())
        for blue, red in ((1, 2), (2, 3), (3, 4)):
            copy = json.loads(json.dumps(document))
            copy["gates"][-1] = dict(copy["gates"][-1],
                                      id=f"{blue}v{red}", blue=blue, red=red)
            with self.assertRaisesRegex(ValueError, "triangular"):
                gate.select_curriculum(copy, 4)
        copy = json.loads(json.dumps(document))
        copy["contractId"] = "unknown"
        with self.assertRaises(ValueError):
            gate.select_curriculum(copy, 4)

    def test_resume_must_match_selected_size(self):
        # The selected size is recorded in checkpoint + result, independent
        # of the fingerprint of the same canonical schedule file.
        config = GATES.read_bytes()
        fingerprint = hashlib.sha256(config).hexdigest()
        for size in (2, 3, 4):
            stages = gate.select_curriculum(json.loads(config), size)
            current, following = stages[-2:]
            result = dict(gate=current["id"], passed=True,
                          autoAdvance=True, gateConfigSha256=fingerprint,
                          nextGate=following["id"],
                          nextThresholdPercent=following["thresholdPercent"],
                          nextMaxSeconds=following["maxSeconds"],
                          rehearsalProfile="balanced_v1")
            if size != 2:
                result["maxTeamSize"] = size
            inputs = dispatcher.next_dispatch(result, config, "420001")
            self.assertEqual(inputs["progressive_gate"], following["id"])
            self.assertEqual(inputs.get("max_team_size"),
                             None if size == 2 else str(size))
            if size != 2:
                with self.assertRaises(ValueError):
                    dispatcher.next_dispatch(dict(result, maxTeamSize=2),
                                             config, "420001")

    def test_zone_option_all_gates_and_variant_namespace(self):
        data = GATES.read_bytes()
        stages = json.loads(data)
        for size in (2, 3, 4):
            candidates = gate.select_curriculum(stages, size)
            for enabled in (True, False):
                effective = [dict(s, hostileZones=enabled) for s in candidates]
                self.assertTrue(all(s["hostileZones"] is enabled for s in effective))
                current, following = candidates[-2:]
                result = dict(gate=current["id"], passed=True,
                    autoAdvance=True, gateConfigSha256=hashlib.sha256(data).hexdigest(),
                    nextGate=following["id"],
                    nextThresholdPercent=following["thresholdPercent"],
                    nextMaxSeconds=following["maxSeconds"],
                    hostileZones=enabled, maxTeamSize=size)
                inputs = dispatcher.next_dispatch(result, data, "420001")
                self.assertEqual(inputs.get("scenario_variant"),
                                 "fixed_zones" if enabled else None)
        with patch.object(champion, "ROOT", Path("best")):
            with champion.variant_namespace(True):
                self.assertEqual(champion.ROOT, Path("best/zones-on"))
            self.assertEqual(champion.ROOT, Path("best"))
        with self.assertRaises(ValueError):
            dispatcher.next_dispatch(dict(result, hostileZones="true"), data, "420001")

    def test_objective_layout_namespaces_and_dispatch(self):
        data = GATES.read_bytes()
        stages = gate.select_curriculum(json.loads(data), 4)
        current, following = stages[-2:]
        for objective in ("fixed", "randomized"):
            for zones in (False, True):
                expected = Path("best")
                if objective == "randomized":
                    expected /= "objective-randomized"
                if zones:
                    expected /= "zones-on"
                actual = champion.variant_root(zones, objective, Path("best"))
                self.assertEqual(actual, expected)
                record = dict(
                    gate=current["id"], passed=True,
                    autoAdvance=True, gateConfigSha256=hashlib.sha256(data).hexdigest(),
                    nextGate=following["id"],
                    nextThresholdPercent=following["thresholdPercent"],
                    nextMaxSeconds=following["maxSeconds"],
                    objectiveLayout=objective, hostileZones=zones, maxTeamSize=4)
                result = dispatcher.next_dispatch(record, data, "420001")
                self.assertEqual(result.get("scenario_variant"),
                                 objective + ("_zones" if zones else "_no_zones")
                                 if objective != "fixed" or zones else None)
        with self.assertRaises(ValueError):
            champion.variant_root(True, "unknown")
        with self.assertRaises(ValueError):
            dispatcher.next_dispatch(
                dict(record, objectiveLayout="invalid"), data, "420001")

    def test_ci_workflows_offer_one_size_control(self):
        for workflow in ("mappo-training-windows.yml",
                         "mappo-restart-progressive.yml"):
            content = (ROOT / ".github/workflows" / workflow).read_text(
                encoding="utf-8")
            self.assertEqual(content.count("      max_team_size:"), 1)
            self.assertEqual(content.count("      scenario_variant:"), 1)
            for selection in ("fixed_no_zones", "fixed_zones",
                              "randomized_no_zones", "randomized_zones"):
                self.assertIn("          - " + selection, content)
            for label in ('        default: "2"',
                          '          - "2"', '          - "3"', '          - "4"'):
                self.assertIn(label, content)
            self.assertNotIn("balanced_3v3", content)
            self.assertNotIn("balanced_4v4", content)

    def assert_scenario_contract(self, blues: int) -> None:
        original_training = {"training": {"maxUpdates": 1}}
        original_scenario = {"blueDefense": {
            "blueTeam": {"activeCount": 2},
            "redTeam": {"activeCount": 2},
            "restrictedZones": {"enabled": False},
            "restrictedZoneSpawn": {"enabledProbability": 0.75},
            "objectiveSpawn": {"mode": "fixed"}}}
        stages = gate.select_curriculum(
            json.loads(GATES.read_bytes()), blues)
        candidates = [stage for stage in stages if stage["blue"] == blues]
        self.assertEqual(len(candidates), blues + 1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = root / "training.json"
            scenario = root / "scenario.json"
            with patch.object(gate.existing, "TRAINING_FILE", train), (
                    patch.object(gate.existing, "SCENARIO_FILE", scenario)):
                for stage in candidates:
                    for zones in (False, True):
                        for objective in ("fixed", "randomized"):
                            with self.subTest(gate=stage["id"], zones=zones,
                                              objective=objective):
                                variant = dict(
                                    stage, hostileZones=zones,
                                    objectiveLayout=objective)
                                gate.existing.apply_stage(
                                    variant, 123, original_training,
                                    original_scenario)
                                actual = json.loads(scenario.read_text())
                                settings = actual["blueDefense"]
                                self.assertEqual(settings["blueTeam"]["activeCount"], blues)
                                self.assertEqual(settings["redTeam"]["activeCount"],
                                                 stage["red"])
                                self.assertIs(settings["restrictedZones"]["enabled"], zones)
                                self.assertEqual(settings["objectiveSpawn"]["mode"],
                                                 objective)
                                self.assertEqual(
                                    json.loads(train.read_text())["training"]["maxUpdates"],
                                    123)

    def test_selected_fixed_layout_overrides_randomized_source(self):
        # Regression #67/#73: a reused scenario must not leak its prior variant.
        original_training = {"training": {"maxUpdates": 1}}
        original_scenario = {"blueDefense": {
            "blueTeam": {"activeCount": 2},
            "redTeam": {"activeCount": 2},
            "restrictedZones": {"enabled": True},
            "restrictedZoneSpawn": {"enabledProbability": 1.0},
            "objectiveSpawn": {"mode": "randomized"}}}
        stages = gate.select_curriculum(json.loads(GATES.read_bytes()), 2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(gate.existing, "TRAINING_FILE", root / "training.json"), (
                    patch.object(gate.existing, "SCENARIO_FILE", root / "scenario.json")):
                for mode in ("fixed", "randomized", "fixed"):
                    selected = dict(stages[4], objectiveLayout=mode)
                    gate.existing.apply_stage(
                        selected, 123, original_training, original_scenario)
                    actual = json.loads(
                        (root / "scenario.json").read_text(encoding="utf-8"))
                    self.assertEqual(
                        actual["blueDefense"]["objectiveSpawn"]["mode"], mode)
                self.assertEqual(original_scenario["blueDefense"][
                    "objectiveSpawn"]["mode"], "randomized")

    def test_three_blue_gates_and_runtime_scenario_variants(self):
        self.assert_scenario_contract(3)

    def test_four_blue_gates_and_runtime_scenario_variants(self):
        self.assert_scenario_contract(4)

    def test_gui_uses_same_selected_two_team_prefix(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import local_mappo_gui_curriculum as gui
        stages, index, validated = gui.load_production_stages()
        expected = gate.select_curriculum(json.loads(GATES.read_bytes()), 2)
        self.assertEqual(stages, expected)
        self.assertEqual(index, 4)
        self.assertEqual(validated, ["1v0", "2v0", "1v1", "2v1"])


if __name__ == "__main__":
    unittest.main()
