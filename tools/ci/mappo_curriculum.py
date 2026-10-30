#!/usr/bin/env python3
"""Headless, checkpoint-preserving MAPPO curriculum; one Release binary per CI run.

The generalized V6 checkpoint supports changing P/N between process launches.
No C++ agent, simulator, optimizer or observation schema is modified here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


TRAINING_FILE = Path("resources/settings/Training/BluePatrol.json")
SCENARIO_FILE = Path("resources/settings/Scenarios/BlueDefense.json")
MODEL_ROOT = Path("resources/model/BluePatrol")
LATEST = MODEL_ROOT / "latest/blue_patrol_generalized_training_v5.txt"
PROD = MODEL_ROOT / "prod/blue_patrol_generalized_v8.txt"
PROD_SCORE = Path(str(PROD) + ".score")
PROGRESS = MODEL_ROOT / "curriculum-progress.json"
TELEMETRY = MODEL_ROOT / "telemetry/mappo-training.csv"
FRESH_SEEDS = (
    MODEL_ROOT / "latest/blue_patrol_training_v4.txt",
    MODEL_ROOT / "latest/blue_patrol_mappo_v7.txt",
    MODEL_ROOT / "prod/blue_patrol_mappo_v7.txt",
    MODEL_ROOT / "latest/blue_patrol_generalized_training_v5.txt",
    MODEL_ROOT / "latest/blue_patrol_generalized_v8.txt",
)
OUTPUT = Path("build/mappo-telemetry")
EVAL_PATTERN = re.compile(
    r"Evaluation #\d+ P=(\d+) N=(\d+) score=([0-9.eE+-]+)"
)
STAGE_NAMES = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def read_updates(path: Path = LATEST) -> int:
    """Read the authoritative PPO update count, not an actor/CI timestamp."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline()
        if not first.startswith("BLUE_TEAM_TRAINING_V6 "):
            raise ValueError(f"Unsupported training checkpoint header: {path}")
        for line in stream:
            if line.startswith("UPDATES "):
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    raise ValueError(f"Malformed update marker: {path}")
                return int(parts[1])
    raise ValueError(f"Missing update marker: {path}")


def validate_curriculum(document: dict) -> list[dict]:
    if document.get("formatVersion") != 1:
        raise ValueError("Curriculum formatVersion must be 1")
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Curriculum needs a nonempty stages array")
    seen: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {
            "id", "blue", "red", "hostileZones", "updates"
        }:
            raise ValueError("Each stage needs id, blue, red, hostileZones, updates")
        name, blue, red, zones, updates = (
            stage["id"], stage["blue"], stage["red"],
            stage["hostileZones"], stage["updates"]
        )
        if not isinstance(name, str) or not STAGE_NAMES.fullmatch(name) or name in seen:
            raise ValueError(f"Invalid or duplicate curriculum stage id: {name!r}")
        seen.add(name)
        if (type(blue) is not int or not 1 <= blue <= 9
                or type(red) is not int or not 0 <= red <= 10
                or type(zones) is not bool or type(updates) is not int
                or not 1 <= updates <= 10000):
            raise ValueError(f"Invalid counts, zones or update count for {name}")
    if (stages[-1]["blue"], stages[-1]["red"]) != (4, 4):
        raise ValueError("Final stage must be the reference 4v4 case")
    return stages


def fingerprint(document: dict) -> str:
    serialized = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_progress(config_hash: str, checkpoint_updates: int) -> dict:
    if not PROGRESS.is_file():
        progress = {
            "formatVersion": 1,
            "curriculumHash": config_hash,
            "baselineUpdates": checkpoint_updates,
            "nextStage": 0,
        }
        write_json(PROGRESS, progress)
        return progress
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    if (progress.get("formatVersion") != 1
            or progress.get("curriculumHash") != config_hash
            or type(progress.get("baselineUpdates")) is not int
            or type(progress.get("nextStage")) is not int):
        raise ValueError("Existing curriculum progress is incompatible; start a fresh run")
    if checkpoint_updates < progress["baselineUpdates"]:
        raise ValueError("Checkpoint predates saved curriculum progress")
    return progress


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def apply_stage(stage: dict, target_updates: int,
                original_training: dict, original_scenario: dict) -> None:
    training = json.loads(json.dumps(original_training))
    scenario = json.loads(json.dumps(original_scenario))
    training["training"]["maxUpdates"] = target_updates
    settings = scenario["blueDefense"]
    settings["blueTeam"]["activeCount"] = stage["blue"]
    settings["redTeam"]["activeCount"] = stage["red"]
    settings["restrictedZones"]["enabled"] = stage["hostileZones"]
    if not stage["hostileZones"]:
        settings["restrictedZoneSpawn"]["enabledProbability"] = 0.0
    write_json(TRAINING_FILE, training)
    write_json(SCENARIO_FILE, scenario)


def run_stage(binary: Path, stage: dict, seconds: int, log_path: Path) -> str | None:
    """Stream exact child output to CI and disk; reject swallowed MAPPO errors."""
    command = [str(binary.resolve()), "--no-gui", "--example",
               "blue-defense", "--max-seconds", str(seconds)]
    last_eval: str | None = None
    failed = False
    with log_path.open("w", encoding="utf-8") as log, (
        (OUTPUT / "mappo-training.log").open("a", encoding="utf-8")
    ) as combined:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                combined.write(line)
                match = EVAL_PATTERN.search(line)
                if match:
                    last_eval = match.group(0)
                if "[MAPPO_ERROR]" in line or "[MAPPO_EVAL_ERROR]" in line:
                    failed = True
            return_code = process.wait()
    if return_code != 0 or failed:
        raise RuntimeError(
            f"Curriculum stage {stage['id']} failed: "
            f"exit={return_code}, MAPPO_ERROR={failed}; see {log_path}"
        )
    return last_eval


def save_candidate_and_restore(baseline: dict[Path, bytes | None],
                               stage: dict, stage_dir: Path) -> None:
    """Keep non-4v4 promotion entirely separate from production's 4v4 score."""
    if (stage["blue"], stage["red"]) == (4, 4):
        return
    for path in (PROD, PROD_SCORE):
        if path.is_file():
            shutil.copy2(path, stage_dir / path.name)
        if baseline[path] is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(baseline[path])


def require_clean_fresh_start() -> None:
    """Avoid the silent legacy V4/V7 migration when no run ID was supplied.

    These are Blue checkpoints in this runner's temporary checkout ONLY.
    The Red production RLMission model is never touched.
    """
    if PROGRESS.exists():
        raise ValueError("Fresh-start requested but curriculum progress already exists")
    for path in FRESH_SEEDS:
        if path.is_file():
            print(f"[CURRICULUM] Removing old Blue seed from checkout: {path}")
            path.unlink()


def run(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.curriculum).read_text(encoding="utf-8"))
    stages = validate_curriculum(config)
    config_hash = fingerprint(config)
    binary = Path(args.binary)
    if not binary.is_file():
        raise FileNotFoundError(f"Missing built PPO executable: {binary}")
    original_training = json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
    original_scenario = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
    original_training_bytes = TRAINING_FILE.read_bytes()
    original_scenario_bytes = SCENARIO_FILE.read_bytes()
    if args.fresh_start:
        require_clean_fresh_start()
    updates = read_updates()
    progress = load_progress(config_hash, updates)
    if not 0 <= progress["nextStage"] <= len(stages):
        raise ValueError("Invalid curriculum progress stage")
    completed_target = progress["baselineUpdates"] + sum(
        item["updates"] for item in stages[:progress["nextStage"]]
    )
    if updates < completed_target:
        raise ValueError(
            "Checkpoint is behind completed curriculum stages: "
            f"updates={updates}, required={completed_target}"
        )
    if progress["nextStage"] == len(stages):
        print("[CURRICULUM] All stages already completed.")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_root = OUTPUT / "curriculum"
    stage_root.mkdir(parents=True, exist_ok=True)
    baseline = {path: path.read_bytes() if path.is_file() else None
                for path in (PROD, PROD_SCORE)}
    deadline = time.monotonic() + args.max_seconds
    summary: list[str] = [
        "# Blue MAPPO curriculum CI",
        "",
        f"Start PPO update: {updates}; target stages: {len(stages)}.",
        "",
        "| Stage | P/N | Hostile zones | Target update | Current update | Last evaluation |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    try:
        for index in range(progress["nextStage"], len(stages)):
            stage = stages[index]
            target = progress["baselineUpdates"] + sum(
                item["updates"] for item in stages[:index + 1]
            )
            updates = read_updates()
            if updates >= target:
                print(f"[CURRICULUM] Skip completed {stage['id']} at update {updates}.")
                progress["nextStage"] = index + 1
                write_json(PROGRESS, progress)
                continue
            remaining = deadline - time.monotonic()
            if remaining < 20:
                print("[CURRICULUM] Time budget reached; checkpoint can resume.")
                break
            # Equal fair-share time; an incomplete stage never gets skipped.
            seconds = max(1, min(int(remaining), max(
                30, int(remaining / (len(stages) - index))
            )))
            stage_dir = stage_root / f"{index + 1:02d}-{stage['id']}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CURRICULUM] Stage={stage['id']} P={stage['blue']} "
                  f"N={stage['red']} zones={stage['hostileZones']} "
                  f"updates={updates}->{target} budget_s={seconds}",
                  flush=True)
            apply_stage(stage, target, original_training, original_scenario)
            if (stage["blue"], stage["red"]) != (4, 4):
                # Remove the 4v4 production score for isolated level evaluation.
                PROD.unlink(missing_ok=True)
                PROD_SCORE.unlink(missing_ok=True)
            try:
                last_eval = run_stage(
                    binary, stage, seconds, stage_dir / "training.log"
                )
                fresh_updates = read_updates()
                if fresh_updates <= updates:
                    raise RuntimeError(
                        f"Stage {stage['id']} made no optimizer progress: {updates}"
                    )
                if TELEMETRY.is_file():
                    shutil.copy2(TELEMETRY, stage_dir / "telemetry.csv")
                if fresh_updates >= target:
                    progress["nextStage"] = index + 1
                    write_json(PROGRESS, progress)
                row = (f"| {stage['id']} | {stage['blue']}/{stage['red']} | "
                       f"{stage['hostileZones']} | {target} | "
                       f"{fresh_updates} | {last_eval or 'n/a'} |")
                summary.append(row)
                (OUTPUT / "curriculum-summary.md").write_text(
                    "\n".join(summary) + "\n", encoding="utf-8"
                )
                if fresh_updates < target:
                    print(f"[CURRICULUM] Stage {stage['id']} incomplete at "
                          f"{fresh_updates}/{target}; resume via run ID.")
                    break
            finally:
                save_candidate_and_restore(baseline, stage, stage_dir)
    finally:
        # Restore pristine checkout resources even on failures.
        TRAINING_FILE.write_bytes(original_training_bytes)
        SCENARIO_FILE.write_bytes(original_scenario_bytes)

    print(f"[CURRICULUM] Finished at update={read_updates()} "
          f"next_stage={progress['nextStage']}/{len(stages)}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/release/Main/Release/PPO.exe")
    parser.add_argument("--curriculum",
                        default="resources/settings/Training/BluePatrolCurriculum.json")
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--fresh-start", action="store_true",
                        help="ignore checked-in legacy Blue seeds (no resume run ID)")
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 18000:
        parser.error("--max-seconds must be in [1, 18000]")
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"[CURRICULUM_ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
