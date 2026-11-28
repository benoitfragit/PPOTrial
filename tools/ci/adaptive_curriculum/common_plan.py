#!/usr/bin/env python3
"""Translate historical Blue gate/profile configuration to PPOKit's common plan.

This module is a configuration client only: it does not train, evaluate, promote,
persist, or select a stage at runtime. Those responsibilities belong to
``ppokit-run`` and ``CurriculumEngine``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import progressive_gate
import rehearsal_profiles


def build_plan(gates_file: Path, max_team_size: int, profile: str,
               threshold_percent: float | None, hostile_zones: bool,
               objective_layout: str) -> dict:
    """Return a canonical plan preserving the selected historical schedule."""
    document = json.loads(gates_file.read_text(encoding="utf-8"))
    gates = progressive_gate.select_curriculum(document, max_team_size)
    effective = rehearsal_profiles.effective_profile(profile, max_team_size)
    stages: list[dict] = []
    validated: list[str] = []
    for index, gate in enumerate(gates):
        threshold = (gate["thresholdPercent"] if threshold_percent is None
                     else threshold_percent) / 100.0
        stage = {
            "name": gate["id"],
            "payload": {
                "blue": gate["blue"],
                "red": gate["red"],
                "hostileZones": hostile_zones,
                "objectiveLayout": objective_layout,
            },
            "promotion": {
                "source": "evaluation",
                "suite": "validation",
                "metric": "mission_success.rate",
                "threshold": threshold,
            },
            "auto_advance": gate["autoAdvance"],
            "max_seconds": gate["maxSeconds"],
        }
        if effective != "legacy":
            length = rehearsal_profiles.weighted_cycle_length(
                gates, index, validated, effective)
            schedule = [rehearsal_profiles.select_weighted_stage(
                gates, index, block, validated, effective)["id"]
                for block in range(length)]
            stage["rehearsal_weights"] = dict(Counter(schedule))
        stages.append(stage)
        validated.append(gate["id"])
    plan = {
        "schema": "ppokit.curriculum/1",
        "id": f"{document['contractId']}-{max_team_size}-{effective}",
        "block_updates": 20,
        "stages": stages,
    }
    if effective == "legacy":
        plan["rehearsal_every_blocks"] = document["rehearsalEveryBlocks"]
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-team-size", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--profile", choices=rehearsal_profiles.PROFILES, default="legacy")
    parser.add_argument("--threshold-percent", type=float)
    parser.add_argument("--hostile-zones", action="store_true")
    parser.add_argument("--objective-layout", choices=("fixed", "randomized"), default="fixed")
    args = parser.parse_args()
    if args.threshold_percent is not None and not 0 <= args.threshold_percent <= 100:
        parser.error("--threshold-percent must be in [0,100]")
    plan = build_plan(args.gates, args.max_team_size, args.profile,
                      args.threshold_percent, args.hostile_zones,
                      args.objective_layout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
