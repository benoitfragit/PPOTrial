#!/usr/bin/env python3
"""Single-gate, bounded MAPPO training; never chooses an agent action.

Each GitHub workflow_dispatch run trains at most one P/N gate. A passed gate
may dispatch the next workflow run; its full optimizer checkpoint is mandatory.
The V3 curriculum and RLMission controller are not modified by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mappo_curriculum as existing  # noqa: E402


GATES_FILE = Path("tools/ci/adaptive_curriculum/gates.json")
STATE_FILE = existing.MODEL_ROOT / "adaptive-curriculum-progress.json"
RESULT_FILE = existing.OUTPUT / "adaptive-gate-result.json"
SUCCESS_LINE = re.compile(
    r"Evaluation #\d+ P=(\d+) N=(\d+) score=[0-9.eE+-]+ "
    r"success=([0-9.eE+-]+)"
)


def validate_gates(document: dict) -> list[dict]:
    if document.get("formatVersion") != 1:
        raise ValueError("Unexpected adaptive gate schema")
    period = document.get("rehearsalEveryBlocks")
    if type(period) is not int or not 2 <= period <= 100:
        raise ValueError("rehearsalEveryBlocks must be in [2,100]")
    stages = document.get("gates")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Missing gate sequence")
    names: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {
            "id", "blue", "red", "hostileZones",
            "thresholdPercent", "maxSeconds", "autoAdvance"
        }:
            raise ValueError("Invalid gate fields")
        name = stage["id"]
        if (not isinstance(name, str)
                or not existing.STAGE_NAMES.fullmatch(name) or name in names):
            raise ValueError("Invalid or repeated gate name")
        names.add(name)
        if (type(stage["blue"]) is not int or not 1 <= stage["blue"] <= 9
                or type(stage["red"]) is not int or not 0 <= stage["red"] <= 10
                or type(stage["hostileZones"]) is not bool
                or type(stage["thresholdPercent"]) not in (int, float)
                or not 0 < stage["thresholdPercent"] <= 100
                or type(stage["maxSeconds"]) is not int
                or not 1 <= stage["maxSeconds"] <= 18000
                or type(stage["autoAdvance"]) is not bool):
            raise ValueError("Invalid gate dimensions")
    if (stages[0]["blue"], stages[0]["red"]) != (1, 0):
        raise ValueError("First gate must be 1v0")
    if (stages[-1]["blue"], stages[-1]["red"]) != (4, 4):
        raise ValueError("Final gate must be 4v4")
    return stages


def select_training_stage(stages: list[dict], index: int, block: int,
                          rehearsal_every: int) -> dict:
    """Most blocks train the active gate; periodically revisit prior gates.

    Selection is independent of success, episode totals or PPO update limits.
    block counts completed training blocks, persisted across GitHub run retries.
    """
    if not 0 <= index < len(stages) or block < 0 or rehearsal_every < 2:
        raise ValueError("Invalid rehearsal selection")
    if index == 0 or (block + 1) % rehearsal_every:
        return stages[index]
    prior = (block // rehearsal_every) % index
    return stages[prior]


def matching_success(log: Path, blue: int, red: int) -> float | None:
    """Read only finished, matching evaluation campaigns, never a PPO score."""
    value = None
    if not log.is_file():
        return None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SUCCESS_LINE.search(line)
        if match and int(match[1]) == blue and int(match[2]) == red:
            rate = float(match[3])
            if not 0.0 <= rate <= 1.0:
                raise ValueError("Evaluation success outside [0,1]")
            value = rate * 100.0
    return value


def initialize_state(stages: list[dict], gate: str, resume: bool,
                     config_hash: str) -> dict:
    names = [stage["id"] for stage in stages]
    if gate not in names:
        raise ValueError("Unknown gate")
    if resume:
        if not STATE_FILE.is_file() or not existing.LATEST.is_file():
            raise ValueError("Resume requires complete adaptive state and V7 checkpoint")
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        last_success = state.get("lastSuccessPercent")
        if (state.get("formatVersion") != 1
                or state.get("gateConfigSha256") != config_hash
                or type(state.get("gateIndex")) is not int
                or not 0 <= state["gateIndex"] < len(stages)
                or type(state.get("elapsedSeconds")) not in (int, float)
                or not 0 <= state["elapsedSeconds"] <= 1e7
                or type(state.get("passed")) is not bool
                or not isinstance(state.get("gate"), str)
                or type(state.get("bestSuccessPercent")) not in (int, float)
                or not 0 <= state["bestSuccessPercent"] <= 100
                or (last_success is not None and (
                    type(last_success) not in (int, float)
                    or not 0 <= last_success <= 100))
                or "lastSuccessPercent" not in state
                or type(state.get("blockIndex")) is not int
                or state["blockIndex"] < 0
                or type(state.get("lastCheckpointUpdates")) is not int
                or state["lastCheckpointUpdates"] < 0):
            raise ValueError("Incompatible adaptive checkpoint provenance")
        checkpoint_updates = existing.read_updates()
        if checkpoint_updates < state["lastCheckpointUpdates"]:
            raise ValueError("Restored optimizer checkpoint is older than gate progress")
        # A crash may have saved a newer update before progress was persisted.
        # Never discard that checkpoint merely because the log metadata lagged.
        state["lastCheckpointUpdates"] = checkpoint_updates
        expected = state["gateIndex"] + (1 if state.get("passed") else 0)
        if expected >= len(stages):
            raise ValueError("All gates have already passed")
        if state["gate"] != names[state["gateIndex"]] or gate != names[expected]:
            raise ValueError("Resume gate is neither unfinished nor next passed gate")
        if state.get("passed"):
            state.update(gateIndex=expected, gate=gate, passed=False,
                         elapsedSeconds=0.0, bestSuccessPercent=0.0,
                         lastSuccessPercent=None, blockIndex=0)
        return state
    if gate != names[0]:
        raise ValueError("Later gates require resume_run_id from the previous gate")
    if STATE_FILE.exists():
        raise ValueError("Fresh run has preexisting adaptive progress")
    existing.require_clean_fresh_start()
    return dict(formatVersion=1, gateConfigSha256=config_hash, gateIndex=0,
                gate=gate, passed=False, elapsedSeconds=0.0,
                bestSuccessPercent=0.0, lastSuccessPercent=None,
                blockIndex=0, lastCheckpointUpdates=0)


def record_result(state: dict, stages: list[dict], threshold: float,
                  passed: bool, updates: int) -> None:
    index = state["gateIndex"]
    next_gate = stages[index + 1]["id"] if passed and index + 1 < len(stages) else ""
    result = dict(gate=state["gate"], passed=passed,
                  lastSuccessPercent=state["lastSuccessPercent"],
                  bestSuccessPercent=state["bestSuccessPercent"],
                  thresholdPercent=threshold,
                  elapsedSeconds=round(state["elapsedSeconds"], 2),
                  optimizerUpdates=updates, nextGate=next_gate,
                  autoAdvance=bool(state.get("autoAdvance", False)
                                   and stages[index]["autoAdvance"]),
                  nextThresholdPercent=(stages[index + 1]["thresholdPercent"]
                     if next_gate else None),
                  nextMaxSeconds=(stages[index + 1]["maxSeconds"]
                     if next_gate else None),
                  gateConfigSha256=state["gateConfigSha256"],
                  finished=passed and not next_gate)
    existing.write_json(STATE_FILE, state)
    existing.write_json(RESULT_FILE, result)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(f"passed={str(passed).lower()}\n")
            stream.write(f"next_gate={next_gate}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n### MAPPO progressive gate {state['gate']}\n"
                f"- Evaluated mission success: {state['lastSuccessPercent']}\n"
                f"- Best success: {state['bestSuccessPercent']:.1f}%"
                f" (required {threshold:.1f}%)\n"
                f"- Gate: {'PASS' if passed else 'NOT PASSED'};"
                f" consumed {state['elapsedSeconds']:.0f} seconds;"
                f" optimizer update {updates}\n"
                f"- Next gate: {next_gate or 'none'}\n"
            )


def run_gate(args: argparse.Namespace) -> dict:
    if not 0 < args.threshold_percent <= 100:
        raise ValueError("threshold-percent must be in (0,100]")
    if not 1 <= args.max_seconds <= 18000:
        raise ValueError("max-seconds must be [1,18000]")
    if not 1 <= args.block_updates <= 1000:
        raise ValueError("block-updates must be [1,1000]")
    binary = Path(args.binary)
    if not binary.is_file():
        raise FileNotFoundError(f"Missing Release PPO executable: {binary}")
    config_bytes = GATES_FILE.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    configuration = json.loads(config_bytes)
    stages = validate_gates(configuration)
    rehearsal_every = configuration["rehearsalEveryBlocks"]
    state = initialize_state(stages, args.gate, args.resume, config_hash)
    stage = stages[state["gateIndex"]]
    state["autoAdvance"] = args.auto_advance
    if state["passed"] or state["gate"] != args.gate:
        raise ValueError("Inconsistent gate state")
    # Every GitHub run receives a fresh wall-clock budget. elapsedSeconds is
    # cumulative telemetry only, never a gate quota.
    original_training_bytes = existing.TRAINING_FILE.read_bytes()
    original_scenario_bytes = existing.SCENARIO_FILE.read_bytes()
    original_training = json.loads(original_training_bytes)
    original_scenario = json.loads(original_scenario_bytes)
    existing.OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_root = existing.OUTPUT / "adaptive-gates" / stage["id"]
    stage_root.mkdir(parents=True, exist_ok=True)
    # Persist progress before launching PPO; the full checkpoint remains the
    # authoritative optimization state even if the run expires mid-gate.
    existing.write_json(STATE_FILE, state)
    # Candidate promotion from auxiliary gates must never replace a 4v4 model.
    previous_production = {
        path: path.read_bytes() if path.is_file() else None
        for path in (existing.PROD, existing.PROD_SCORE)
    }
    started = time.monotonic()
    deadline = started + args.max_seconds
    passed = False
    try:
        while time.monotonic() + 20 < deadline:
            updates = existing.read_updates()
            target = updates + args.block_updates
            selected = select_training_stage(
                stages, state["gateIndex"], state["blockIndex"],
                rehearsal_every)
            block_dir = stage_root / (
                f"block-{state['blockIndex']:06d}-{selected['id']}"
                f"-updates-{updates:06d}-{target:06d}")
            block_dir.mkdir(parents=True, exist_ok=True)
            stage_log = block_dir / "training.log"
            remaining = max(1, int(deadline - time.monotonic()))
            existing.apply_stage(
                selected, target, original_training, original_scenario)
            baseline = {
                path: path.read_bytes() if path.is_file() else None
                for path in (existing.PROD, existing.PROD_SCORE)
            }
            if (selected["blue"], selected["red"]) != (4, 4):
                existing.PROD.unlink(missing_ok=True)
                existing.PROD_SCORE.unlink(missing_ok=True)
            print(f"[ADAPTIVE_GATE] gate={stage['id']} replay={selected['id']} "
                  f"updates={updates}->{target}", flush=True)
            try:
                existing.run_stage(binary, selected, remaining, stage_log)
            finally:
                # Non-reference stages cannot overwrite the production 4v4
                # candidate, including when a rehearsal interrupts 4v4.
                existing.save_candidate_and_restore(baseline, selected, block_dir)
            observed = existing.read_updates()
            if observed < updates or observed > target:
                raise RuntimeError("Checkpoint update counter is inconsistent")
            if observed == updates and time.monotonic() + 20 < deadline:
                raise RuntimeError("No optimizer progress despite remaining time")
            if observed > updates:
                state["blockIndex"] += 1
                state["lastCheckpointUpdates"] = observed
                # A rehearsal evaluation never certifies the current gate.
                if selected["id"] == stage["id"]:
                    success = matching_success(
                        stage_log, stage["blue"], stage["red"])
                    if success is not None:
                        state["lastSuccessPercent"] = success
                        state["bestSuccessPercent"] = max(
                            state["bestSuccessPercent"], success)
                        passed = success >= args.threshold_percent
            state["elapsedSeconds"] += max(0.0, time.monotonic() - started)
            started = time.monotonic()
            state["passed"] = passed
            existing.write_json(STATE_FILE, state)
            if passed or observed == updates:
                break
    finally:
        existing.TRAINING_FILE.write_bytes(original_training_bytes)
        existing.SCENARIO_FILE.write_bytes(original_scenario_bytes)
        if (stage["blue"], stage["red"]) != (4, 4):
            for path, previous in previous_production.items():
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(previous)
    state["elapsedSeconds"] += max(0.0, time.monotonic() - started)
    state["passed"] = passed
    record_result(state, stages, args.threshold_percent, passed,
                  existing.read_updates())
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/release/Main/Release/PPO.exe")
    parser.add_argument("--gate", default="1v0")
    parser.add_argument("--threshold-percent", type=float, default=75.0)
    parser.add_argument("--max-seconds", type=int, default=14400)
    parser.add_argument("--block-updates", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-advance", action="store_true")
    args = parser.parse_args()
    result = run_gate(args)
    print(f"[ADAPTIVE_GATE] gate={result['gate']} passed={result['passed']} "
          f"best={result['bestSuccessPercent']:.1f}% "
          f"updates={existing.read_updates()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
