#!/usr/bin/env python3
"""Bounded and resumable FlightControl3D curriculum runner for GitHub Actions."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

STAGES = ("loc", "glide", "joint")
MAX_UPDATE = 100_000_000
GATE_PATTERN = re.compile(
    r"^\[FLIGHT_GATE\] stage=(loc|glide|joint) successes=(\d+) "
    r"episodes=(\d+) passed=(true|false)$"
)


def checkpoint_update(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().split()
    if len(header) != 5 or header[:4] != ["PPO_FLIGHT_HYBRID_V1", "56", "5", "128"]:
        raise ValueError("Incompatible FlightControl3D checkpoint header")
    if not header[4].isascii() or not header[4].isdecimal():
        raise ValueError("Invalid FlightControl3D update counter")
    updates = int(header[4])
    if updates > MAX_UPDATE:
        raise ValueError("FlightControl3D update counter exceeds maximum")
    return updates


def emit_summary(lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as output:
            output.write(text)


def run_block(executable: Path, config: Path, output: Path, stage: str, target: int,
              expected_episodes: int, min_success_rate: float) -> bool:
    log_path = output / f"gate_{stage}_{target}.log"
    command = [str(executable), "--config", str(config), "--output", str(output),
               "--updates", str(target), "--stage", stage]
    gate_line = None
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if line.startswith("[FLIGHT_GATE]"):
                    gate_line = line.strip()
            result = process.wait()
    if result != 0:
        raise RuntimeError(f"FlightControlHeadless failed ({result}); inspect {log_path}")
    match = GATE_PATTERN.fullmatch(gate_line or "")
    if not match or match[1] != stage or int(match[3]) != expected_episodes:
        raise RuntimeError(f"Missing or invalid independent gate evaluation: {gate_line}")
    successes = int(match[2])
    passed = successes >= math.ceil(min_success_rate * expected_episodes)
    if successes > expected_episodes or (match[4] == "true") != passed:
        raise RuntimeError(f"Inconsistent gate result: {gate_line}")
    if checkpoint_update(output / "latest" / "flight_hybrid_v1.txt") != target:
        raise RuntimeError("FlightControlHeadless did not save the expected PPO update")
    telemetry = output / "training.csv"
    if not telemetry.is_file() or telemetry.stat().st_size == 0:
        raise RuntimeError("Missing FlightControl3D telemetry CSV")
    shutil.copyfile(telemetry, output / f"training_{stage}_{target}.csv")
    emit_summary([gate_line])
    return passed


def run(args: argparse.Namespace) -> int:
    if args.updates_per_evaluation < 1 or args.max_additional_updates < 1:
        raise ValueError("Update budgets must be positive")
    if args.runtime_seconds < 1:
        raise ValueError("Runtime must be positive")
    if args.stage not in (*STAGES, "progressive"):
        raise ValueError("Invalid curriculum stage")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest" / "flight_hybrid_v1.txt"
    progress = output / "curriculum_stage.txt"
    if checkpoint.is_file():
        start_update = checkpoint_update(checkpoint)
    elif progress.exists():
        raise ValueError("Missing checkpoint for saved FlightControl3D curriculum stage")
    else:
        # A new v2 training campaign has no pretrained checkpoint.
        start_update = 0
    if args.max_additional_updates > MAX_UPDATE - start_update:
        raise ValueError("Curriculum update budget exceeds FlightControlHeadless maximum")
    ceiling = start_update + args.max_additional_updates
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_episodes = config["evaluation"]["episodes"]
    min_success_rate = config["evaluation"]["minimumSuccessRate"]
    if not isinstance(expected_episodes, int) or expected_episodes < 1 or not (0 < min_success_rate <= 1):
        raise ValueError("Invalid independent evaluation settings")
    stage = args.stage
    if stage == "progressive":
        stage = progress.read_text(encoding="utf-8").strip() if progress.is_file() else "loc"
    if stage not in (*STAGES, "done"):
        raise ValueError(f"Invalid saved curriculum stage: {stage}")
    deadline = time.monotonic() + args.runtime_seconds
    completed_selected_stage = stage == "done"
    blocks = 0
    current = start_update
    while stage != "done":
        if current >= ceiling or time.monotonic() >= deadline:
            break
        target = min(current + args.updates_per_evaluation, ceiling)
        # Persist stage before training: failures and cancellations must not advance it.
        progress.write_text(stage + "\n", encoding="utf-8")
        emit_summary([f"Training {stage}: PPO update {current} -> {target}"])
        passed = run_block(args.executable, args.config, output, stage, target,
                           expected_episodes, min_success_rate)
        current = target
        blocks += 1
        if passed:
            stage = "glide" if stage == "loc" else "joint" if stage == "glide" else "done"
            progress.write_text(stage + "\n", encoding="utf-8")
            if args.stage != "progressive":
                completed_selected_stage = True
                break
    complete = stage == "done" or completed_selected_stage
    progress.write_text(stage + "\n", encoding="utf-8")
    status = "completed" if complete else "incomplete"
    (output / "curriculum_status.txt").write_text(status + "\n", encoding="utf-8")
    emit_summary([f"Curriculum status: {status}", f"Next stage: {stage}",
                  f"Updates: {start_update} -> {current} ({blocks} evaluated blocks)",
                  "Resume by supplying this workflow run ID as resume_run_id." if not complete else
                  "Curriculum target validated by independent gate evaluations."])
    # A green workflow must not represent an unfinished curriculum.
    return 0 if complete else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("progressive", *STAGES), default="progressive")
    parser.add_argument("--updates-per-evaluation", type=int, default=10)
    parser.add_argument("--max-additional-updates", type=int, default=1000)
    parser.add_argument("--runtime-seconds", type=int, default=18000)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"[FLIGHT_CI_ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
