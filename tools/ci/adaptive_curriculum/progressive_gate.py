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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mappo_curriculum as existing  # noqa: E402
import champion_checkpoints as champion  # noqa: E402


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
    final = (stages[-1]["blue"], stages[-1]["red"])
    if final == (2, 2):
        # Enforce the reduced live curriculum while keeping historical 4v4
        # contracts readable for provenance-verified checkpoint bootstraps.
        if any(stage["blue"] > 2 or stage["red"] > 2
               or (stage["blue"], stage["red"]) == (1, 2)
               for stage in stages):
            raise ValueError("Reduced curriculum allows at most 2v2, excluding 1v2")
    elif final != (4, 4):
        raise ValueError("Final gate must be 2v2 or historical 4v4")
    return stages


def select_training_stage(stages: list[dict], index: int, block: int,
                          rehearsal_every: int,
                          validated_gates: list[str] | None = None) -> dict:
    """Rehearse only truly passed earlier gates, never gates skipped by choice.

    Omitted validated_gates retains the original sequential caller behavior.
    Selection remains independent of PPO update quotas.
    """
    if not 0 <= index < len(stages) or block < 0 or rehearsal_every < 2:
        raise ValueError("Invalid rehearsal selection")
    prior = stages[:index] if validated_gates is None else [
        stage for stage in stages[:index] if stage["id"] in validated_gates]
    if not prior or (block + 1) % rehearsal_every:
        return stages[index]
    return prior[(block // rehearsal_every) % len(prior)]


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


def equivalent_gate_config_hashes(config_bytes: bytes) -> set[str]:
    """Only accept byte-identical gate contracts modulo LF/CRLF checkout EOLs."""
    normalized = config_bytes.replace(b"\r\n", b"\n")
    return {
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest(),
    }


def validate_source_gate_fingerprint(source_bytes: bytes,
                                     saved_digest: str | None,
                                     source_gate: str | None) -> list[dict]:
    """Validate a historical gate schedule at the source run's exact commit.

    A changed current schedule is allowed only via a fresh bootstrap cycle,
    never during a regular same-gate resume. The old fingerprint and gate must
    match the downloaded source commit; no fake gate success is imported.
    """
    source_gates = validate_gates(json.loads(source_bytes))
    allowed = equivalent_gate_config_hashes(source_bytes)
    if not isinstance(saved_digest, str) or saved_digest not in allowed:
        raise ValueError("Source gate configuration fingerprint mismatch")
    if source_gate not in [stage["id"] for stage in source_gates]:
        raise ValueError("Source gate is absent from its original configuration")
    return source_gates



def validated_rehearsal_history(stages: list[dict], start_gate: str,
                                historical: object) -> list[str]:
    """Keep only prior, source-validated gates with intact evaluated BEST pairs.

    Retired gates can remain in the historical archive, but must never be
    replayed by the new curriculum. Reject missing or unevaluated BEST pairs
    instead of silently losing the requested rehearsal capability.
    """
    names = [stage["id"] for stage in stages]
    if start_gate not in names:
        raise ValueError("Unknown resumed gate")
    if (not isinstance(historical, list)
            or any(type(value) is not str for value in historical)
            or len(historical) != len(set(historical))):
        raise ValueError("Invalid historical gate validation list")
    retained = [gate_id for gate_id in names[:names.index(start_gate)]
                if gate_id in historical]
    for gate_id in retained:
        best = champion.read(gate_id)
        if best is None or not best["evaluatedForGate"]:
            raise ValueError(
                f"Missing evaluated BEST for historical rehearsal {gate_id}")
    return retained



def initialize_state(stages: list[dict], gate: str, resume: bool,
                     config_hash: str) -> dict:
    names = [stage["id"] for stage in stages]
    if gate not in names:
        raise ValueError("Unknown gate")
    if resume:
        # A Linux bootstrap records LF, while Git checkout on Windows can
        # convert gates.json to CRLF. Authenticate the same contract, not EOLs.
        valid_config_hashes = equivalent_gate_config_hashes(GATES_FILE.read_bytes())
        if config_hash not in valid_config_hashes:
            raise ValueError("Invalid current adaptive gate configuration hash")
        if not STATE_FILE.is_file() or not existing.LATEST.is_file():
            raise ValueError("Resume requires complete adaptive state and V7 checkpoint")
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        last_success = state.get("lastSuccessPercent")
        if (state.get("formatVersion") != 1
                or state.get("gateConfigSha256") not in valid_config_hashes
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
        recorded_updates = state["lastCheckpointUpdates"]
        checkpoint_updates = existing.read_updates()
        if checkpoint_updates < recorded_updates:
            raise ValueError("Restored optimizer checkpoint is older than gate progress")
        # A crash may have saved a newer update before progress was persisted.
        # Never discard that checkpoint merely because the log metadata lagged.
        state["lastCheckpointUpdates"] = checkpoint_updates
        current_champion = champion.read(state["gate"])
        if state.get("championRequired", False) and current_champion is None:
            raise ValueError("Mandatory evaluated champion checkpoint is missing")
        if state["gate"] != names[state["gateIndex"]]:
            raise ValueError("Checkpoint gate index/name mismatch")
        validated = state.get("validatedGates")
        if validated is None:
            # One-time conversion of sequential checkpoints (runs <= 34).
            # The former code enforced every preceding gate; none was skipped.
            validated = names[:state["gateIndex"]]
            if state["passed"]:
                validated.append(state["gate"])
        elif (not isinstance(validated, list)
              or any(type(item) is not str or item not in names
                     or names.index(item) > state["gateIndex"]
                     for item in validated)
              or len(set(validated)) != len(validated)
              or (state["gate"] in validated) != state["passed"]):
            raise ValueError("Invalid verified-gate checkpoint history")
        target = names.index(gate)
        if target < state["gateIndex"]:
            raise ValueError("Backward gate selection requires a separate checkpoint")
        if target == state["gateIndex"] and state["passed"]:
            raise ValueError("Gate already passed: select a later gate")
        if target > state["gateIndex"]:
            # Previous stages may have been skipped explicitly. Never mistake
            # their unverified latest model for an evaluated champion.
            previous_gate = state["gate"]
            if champion.read(previous_gate) is None:
                last = state["lastSuccessPercent"]
                best = state["bestSuccessPercent"]
                if (state["passed"] and last is not None
                        and checkpoint_updates == recorded_updates):
                    # Old artifacts stop after a passing final campaign, but
                    # may have lost an EARLIER, higher-scoring model. Import
                    # only the final evaluated V7, and never call it the
                    # historical best if its last score was lower.
                    champion.import_legacy_pass(previous_gate, last)
                    print("[ADAPTIVE_GATE] Imported legacy FINAL passing "
                          f"evaluation for {previous_gate}: success={last}%; "
                          f"historical best={best}% (may be unavailable)",
                          flush=True)
                elif state["passed"]:
                    raise ValueError(
                        "Legacy passing checkpoint does not match its recorded "
                        "evaluation update. Restore that run's evaluated V7 "
                        "artifact; cannot certify a newer LATEST")
                else:
                    # Explicitly skipping an unfinished legacy gate is an
                    # unverified warm start, never a fabricated gate success.
                    champion.seed(previous_gate, None)
                    print("[ADAPTIVE_GATE] Explicit legacy skip: using "
                          "unverified V7 seed, not a certified champion",
                          flush=True)
            predecessor = champion.read(previous_gate)
            if state["passed"] and not predecessor["evaluatedForGate"]:
                raise ValueError(
                    "Passed previous gate has no evaluated champion pair")
            champion.seed(gate, previous_gate)
            restored = champion.restore_latest(gate)
            state.update(gateIndex=target, gate=gate, passed=False,
                         elapsedSeconds=0.0, bestSuccessPercent=0.0,
                         lastSuccessPercent=None, blockIndex=0,
                         lastCheckpointUpdates=restored, championRequired=True)
        # Keep the recorded fingerprint native to the current checkout for
        # subsequent same-gate resumes; no policy or checkpoint bytes change.
        state["gateConfigSha256"] = config_hash
        state["validatedGates"] = validated
        return state
    if gate != names[0]:
        raise ValueError("Later gates require resume_run_id from the previous gate")
    if STATE_FILE.exists():
        raise ValueError("Fresh run has preexisting adaptive progress")
    existing.require_clean_fresh_start()
    return dict(formatVersion=1, gateConfigSha256=config_hash, gateIndex=0,
                gate=gate, passed=False, elapsedSeconds=0.0,
                bestSuccessPercent=0.0, lastSuccessPercent=None,
                blockIndex=0, lastCheckpointUpdates=0,
                validatedGates=[], championRequired=False)


def record_result(state: dict, stages: list[dict], threshold: float,
                  passed: bool, updates: int) -> None:
    if passed and state["gate"] not in state["validatedGates"]:
        state["validatedGates"].append(state["gate"])
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
                  sameThresholdAllGates=state.get("sameThresholdAllGates", False),
                  uniformThresholdPercent=state.get("uniformThresholdPercent"),
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
    cycle = state.get("restartCycle")
    if cycle is not None:
        # Preserve the separate restart action's previously agreed policy.
        uniform = cycle["sameThresholdAllGates"]
        if uniform and float(args.threshold_percent) != float(cycle["successPercent"]):
            raise ValueError("Restart-cycle uniform threshold mismatch")
    else:
        uniform = getattr(args, "same_threshold_all_gates", True)
    if type(uniform) is not bool:
        raise ValueError("same_threshold_all_gates must be boolean")
    state["sameThresholdAllGates"] = uniform
    if uniform:
        # The selected threshold belongs to this curriculum cycle and travels
        # with the checkpoint; per-gate thresholds never replace it on dispatch.
        state["uniformThresholdPercent"] = float(args.threshold_percent)
    else:
        state.pop("uniformThresholdPercent", None)
    state["autoAdvance"] = args.auto_advance
    if state["passed"] or state["gate"] != args.gate:
        raise ValueError("Inconsistent gate state")
    # A completed deterministic campaign is durable evidence. Re-evaluating
    # or further PPO updates are not necessary merely to certify a resumed
    # gate whose saved BEST pair already satisfies the requested threshold.
    existing_best = champion.read(stage["id"])
    if (args.resume and existing_best is not None
            and existing_best["evaluatedForGate"]
            and existing_best["successPercent"] >= args.threshold_percent):
        state["bestSuccessPercent"] = existing_best["successPercent"]
        state["lastSuccessPercent"] = (
            state["lastSuccessPercent"]
            if state["lastSuccessPercent"] is not None
            else existing_best["successPercent"])
        state["championRequired"] = True
        state["passed"] = True
        print(f"[ADAPTIVE_GATE] gate={stage['id']} already satisfied by "
              f"evaluated champion update={existing_best['updates']} "
              f"success={existing_best['successPercent']}%", flush=True)
        record_result(state, stages, args.threshold_percent, True,
                      existing.read_updates())
        return state
    # Every GitHub run receives a fresh wall-clock budget. elapsedSeconds is
    # cumulative telemetry only, never a gate quota.
    original_training_bytes = existing.TRAINING_FILE.read_bytes()
    original_scenario_bytes = existing.SCENARIO_FILE.read_bytes()
    original_training = json.loads(original_training_bytes)
    original_scenario = json.loads(original_scenario_bytes)
    cadence = original_training["training"].get("checkpointEvery")
    if type(cadence) is not int or cadence < 1:
        raise ValueError("checkpointEvery must be a positive evaluation cadence")
    existing.OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_root = existing.OUTPUT / "adaptive-gates" / stage["id"]
    stage_root.mkdir(parents=True, exist_ok=True)
    # LATEST is allowed to regress. BEST is the authoritative gate handoff.
    # Persist progress before launching PPO; optimizer checkpoints remain
    # available even if the wall-clock budget expires.
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
            # One completed evaluation per process block: otherwise a later
            # worse PPO update can overwrite the checkpoint of an earlier win.
            until_evaluation = cadence - (updates % cadence)
            target = updates + min(args.block_updates, until_evaluation)
            selected = select_training_stage(
                stages, state["gateIndex"], state["blockIndex"],
                rehearsal_every, state["validatedGates"])
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
            # C++ evaluates against this gate's own incumbent. The global
            # deployed 4v4 PROD is only a temporary scratch for this process,
            # restored after the block, never published from another gate.
            champion.install_production_for_stage(selected)
            print(f"[ADAPTIVE_GATE] gate={stage['id']} replay={selected['id']} "
                  f"updates={updates}->{target}", flush=True)
            try:
                existing.run_stage(binary, selected, remaining, stage_log)
            finally:
                # The runtime PROD is temporary during any stage/rehearsal.
                # The external 4v4 production actor must never be overwritten
                # just because a different gate evaluated successfully.
                existing.save_candidate_and_restore(baseline, selected, block_dir)
                for path, previous in baseline.items():
                    if previous is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(previous)
            observed = existing.read_updates()
            if observed < updates or observed > target:
                raise RuntimeError("Checkpoint update counter is inconsistent")
            if observed == updates and time.monotonic() + 20 < deadline:
                raise RuntimeError("No optimizer progress despite remaining time")
            if observed > updates:
                state["blockIndex"] += 1
                state["lastCheckpointUpdates"] = observed
                evaluated = champion.evaluation_at_update(
                    stage_log, selected["blue"], selected["red"], observed)
                if evaluated is not None:
                    improved, best = champion.promote(
                        selected["id"], evaluated, observed)
                    print(f"[ADAPTIVE_GATE] champion gate={selected['id']} "
                          f"update={best['updates']} "
                          f"success={best['successPercent']} "
                          f"promoted={improved}", flush=True)
                    if selected["id"] == stage["id"]:
                        state["championRequired"] = True
                        # Rehearsals cannot certify the current gate.
                        success = evaluated["successPercent"]
                        state["lastSuccessPercent"] = success
                        state["bestSuccessPercent"] = best["successPercent"]
                        passed = best["successPercent"] >= args.threshold_percent
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
    if passed:
        certified = champion.read(stage["id"])
        if certified is None or not certified["evaluatedForGate"]:
            raise ValueError("Passing gate has no corresponding evaluated BEST pair")
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
    parser.add_argument("--per-gate-thresholds", action="store_false",
                        dest="same_threshold_all_gates",
                        help="Opt out of the default uniform success threshold")
    parser.set_defaults(same_threshold_all_gates=True)
    args = parser.parse_args()
    result = run_gate(args)
    print(f"[ADAPTIVE_GATE] gate={result['gate']} passed={result['passed']} "
          f"best={result['bestSuccessPercent']:.1f}% "
          f"updates={existing.read_updates()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
