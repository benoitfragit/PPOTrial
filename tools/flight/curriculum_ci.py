#!/usr/bin/env python3
"""Bounded and resumable FlightControl3D landing training runner for GitHub Actions.

Each block trains the T-6 landing policy for a fixed number of PPO updates,
then reads the independent validation reported by FlightControlHeadless.
The runner stops at the first validated block, or when its update or time
budget is exhausted; an exhausted budget is never reported as a success.
"""

from __future__ import annotations

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

CHECKPOINT_MAGIC = "PPO_FLIGHT_POLICY_V3"
CHECKPOINT_NAME = Path("latest") / "flight_policy.txt"
MAX_UPDATE = 100_000_000
SUPPORTED_WIDTHS = (32, 64, 128, 256)
GATE_PATTERN = re.compile(
    r"^\[FLIGHT_GATE\] successes=(\d+) episodes=(\d+) difficulty=([0-9.]+) passed=(true|false)$")
# Validation campaigns follow the curriculum: only a full-task campaign can pass.
FULL_TASK_DIFFICULTY = 1.0


def checkpoint_header(path: Path) -> list[int]:
    """Return observations, actions, outputs, width and updates from a checkpoint header."""
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().split()
    if len(header) != 6 or header[0] != CHECKPOINT_MAGIC or not all(
            value.isascii() and value.isdecimal() for value in header[1:]):
        raise ValueError(f"Incompatible FlightControl3D checkpoint header: {path}")
    values = [int(value) for value in header[1:]]
    if values[3] not in SUPPORTED_WIDTHS or values[4] > MAX_UPDATE:
        raise ValueError(f"Invalid FlightControl3D checkpoint dimensions: {path}")
    return values


def inspect_runtime(executable: Path, config: Path, require_release: bool) -> dict:
    """Read the actual C++ contract without starting a session or touching output."""
    result = subprocess.run([str(executable), "--config", str(config), "--inspect-contract"],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise ValueError(f"FlightControlHeadless contract check failed: {result.stderr.strip()}")
    contract = json.loads(result.stdout)
    dimensions = [contract.get(key) for key in ("observations", "actions", "outputs")]
    if (not isinstance(contract, dict) or not all(isinstance(value, int) and value > 0 for value in dimensions)
            or contract.get("width") not in SUPPORTED_WIDTHS or not isinstance(contract.get("signature"), dict)):
        raise ValueError("Incompatible Python/C++ FlightControl3D contract")
    if require_release and contract.get("build") != "Release":
        raise ValueError("Training workflow requires a Release FlightControlHeadless binary")
    return contract


def verify_checkpoint(executable: Path, config: Path, checkpoint: Path, contract: dict) -> int:
    """Let the native loader validate signature, weights, optimizer and RNG before training."""
    observations, actions, outputs, width, updates = checkpoint_header(checkpoint)
    if [observations, actions, outputs, width] != [contract[key] for key in
                                                   ("observations", "actions", "outputs", "width")]:
        raise ValueError("Checkpoint dimensions differ from the native contract")
    result = subprocess.run([str(executable), "--config", str(config), "--check-checkpoint", str(checkpoint)],
                            capture_output=True, text=True, timeout=60)
    expected = f"[FLIGHT_CHECKPOINT] compatible=true updates={updates}"
    if result.returncode != 0 or result.stdout.strip() != expected:
        raise ValueError(f"Incompatible/corrupt FlightControl3D checkpoint: {checkpoint}: {result.stderr.strip()}")
    return updates


def emit_summary(lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as output:
            output.write(text)


def run_block(executable: Path, config: Path, output: Path, target: int,
              expected_episodes: int, min_success_rate: float) -> bool:
    """Train up to the target update, then return the independent validation verdict."""
    log_path = output / f"block_{target}.log"
    command = [str(executable), "--config", str(config), "--output", str(output), "--updates", str(target)]
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
    if not match or int(match[2]) != expected_episodes:
        raise RuntimeError(f"Missing or invalid independent validation: {gate_line}")
    successes = int(match[1])
    difficulty = float(match[3])
    passed = (difficulty >= FULL_TASK_DIFFICULTY
              and successes >= math.ceil(min_success_rate * expected_episodes))
    if successes > expected_episodes or not 0.0 <= difficulty <= 1.0 or (match[4] == "true") != passed:
        raise RuntimeError(f"Inconsistent validation result: {gate_line}")
    if checkpoint_header(output / CHECKPOINT_NAME)[4] != target:
        raise RuntimeError("FlightControlHeadless did not save the expected PPO update")
    telemetry = output / "training.csv"
    if not telemetry.is_file() or telemetry.stat().st_size == 0:
        raise RuntimeError("Missing FlightControl3D telemetry CSV")
    shutil.copyfile(telemetry, output / f"training_{target}.csv")
    emit_summary([gate_line])
    return passed


def run(args: argparse.Namespace) -> int:
    if args.updates_per_evaluation < 1 or args.max_additional_updates < 1:
        raise ValueError("Update budgets must be positive")
    if args.runtime_seconds < 1:
        raise ValueError("Runtime must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_episodes = config["evaluation"]["episodes"]
    min_success_rate = config["evaluation"]["minimumSuccessRate"]
    if not isinstance(expected_episodes, int) or expected_episodes < 1 or not (0 < min_success_rate <= 1):
        raise ValueError("Invalid independent evaluation settings")
    contract = inspect_runtime(args.executable, args.config, args.require_release)
    checkpoint = output / CHECKPOINT_NAME
    start_update = verify_checkpoint(args.executable, args.config, checkpoint, contract) if checkpoint.is_file() else 0
    if args.max_additional_updates > MAX_UPDATE - start_update:
        raise ValueError("Update budget exceeds the FlightControlHeadless maximum")
    ceiling = start_update + args.max_additional_updates
    deadline = time.monotonic() + args.runtime_seconds
    current = start_update
    passed = False
    blocks = 0
    while not passed and current < ceiling and time.monotonic() < deadline:
        target = min(current + args.updates_per_evaluation, ceiling)
        emit_summary([f"Training landing policy: PPO update {current} -> {target}"])
        passed = run_block(args.executable, args.config, output, target, expected_episodes, min_success_rate)
        current = target
        blocks += 1
    status = "completed" if passed else "incomplete"
    (output / "training_status.txt").write_text(status + "\n", encoding="utf-8")
    emit_summary([f"Training status: {status}",
                  f"Updates: {start_update} -> {current} ({blocks} validated training blocks)",
                  "Landing policy validated by an independent evaluation." if passed else
                  "Resume by supplying this workflow run ID as resume_run_id."])
    # A green workflow must not represent an unvalidated policy.
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--updates-per-evaluation", type=int, default=10)
    parser.add_argument("--max-additional-updates", type=int, default=1000)
    parser.add_argument("--runtime-seconds", type=int, default=18000)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"[FLIGHT_CI_ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
