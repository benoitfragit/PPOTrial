#!/usr/bin/env python3
"""Run the CI-equivalent 2v2 rehearsal curriculum through the desktop GUI.

The launcher consumes the complete V7 checkpoint already stored in ``latest``.
It changes only the normal Training/Scenario JSON files while a block runs,
restores those files and the GUI layout byte-for-byte afterwards, and keeps an
immutable source backup under ``build/local-mappo-gui``.  Agent actions,
rewards, physics and the production Red controller remain owned by the C++
runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CI_TOOLS = ROOT / "tools" / "ci"
ADAPTIVE_TOOLS = CI_TOOLS / "adaptive_curriculum"
sys.path.insert(0, str(CI_TOOLS))
sys.path.insert(0, str(ADAPTIVE_TOOLS))

import mappo_curriculum as curriculum  # noqa: E402
import progressive_gate as adaptive  # noqa: E402
import rehearsal_profiles as profiles  # noqa: E402


LATEST_ACTOR = Path(
    "resources/model/BluePatrol/latest/blue_patrol_generalized_v10.txt")
DEFAULT_STATE_ROOT = Path("build/local-mappo-gui")
STATE_NAME = "state.json"
SESSION_FORMAT_VERSION = 1
TARGET_GATE = "2v2"
GUI_LAYOUT = Path("imgui.ini")
RENDERER_SETTINGS = Path("resources/settings/Renderer.json")
PRESERVED_RUNTIME_FILES = (
    curriculum.TRAINING_FILE,
    curriculum.SCENARIO_FILE,
    curriculum.PROD,
    curriculum.PROD_SCORE,
    RENDERER_SETTINGS,
    GUI_LAYOUT,
)


@dataclass(frozen=True)
class LauncherOptions:
    """Validated local launcher inputs."""

    binary: Path
    profile: str
    max_seconds: int
    block_updates: int
    expected_source_updates: int
    state_root: Path
    max_blocks: int
    dry_run: bool


class FileSnapshot:
    """Restore a small explicit file set byte-for-byte after orchestration."""

    def __init__(self, paths: Iterable[Path]) -> None:
        self._contents = {
            path: path.read_bytes() if path.is_file() else None
            for path in paths
        }

    def restore(self) -> None:
        """Restore every captured file without touching any other path."""
        for path, contents in self._contents.items():
            if contents is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)


def _sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_production_stages() -> tuple[list[dict], int, list[str]]:
    """Load the repository gate contract and locate the final 2v2 gate."""
    document = json.loads(adaptive.GATES_FILE.read_bytes())
    stages = adaptive.select_curriculum(document, 2)
    names = [stage["id"] for stage in stages]
    if TARGET_GATE not in names:
        raise ValueError("The adaptive curriculum has no 2v2 target gate")
    target_index = names.index(TARGET_GATE)
    if target_index != len(stages) - 1:
        raise ValueError("The local visual curriculum requires final gate 2v2")
    return stages, target_index, names[:target_index]


def balanced_schedule(stages: list[dict], target_index: int,
                      validated_gates: list[str], profile: str,
                      start_block: int = 0) -> list[dict]:
    """Return one exact weighted cycle beginning at ``start_block``."""
    if profile not in profiles.WEIGHTS:
        raise ValueError("The GUI curriculum requires a weighted profile")
    cycle_length = profiles.weighted_cycle_length(
        stages, target_index, validated_gates, profile)
    return [profiles.select_weighted_stage(
        stages, target_index, start_block + offset,
        validated_gates, profile)
        for offset in range(cycle_length)]


def next_update_target(updates: int, cadence: int,
                       block_updates: int) -> int:
    """Match the CI boundary: stop after at most one evaluation campaign."""
    if updates < 0 or cadence < 1 or block_updates < 1:
        raise ValueError("Invalid checkpoint cadence or update count")
    until_evaluation = cadence - (updates % cadence)
    return updates + min(block_updates, until_evaluation)


def _validate_checkpoint_pair(expected_updates: int) -> int:
    """Validate the local run-84 actor, optimizer and production actor."""
    paths = (LATEST_ACTOR, curriculum.LATEST, curriculum.PROD)
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing local run-84 checkpoint: {path}")
    with LATEST_ACTOR.open(encoding="utf-8") as stream:
        actor_header = stream.readline().split()
    with curriculum.LATEST.open(encoding="utf-8") as stream:
        training_header = stream.readline().split()
    semantic = "UNIFIED_PHYSICAL_GUIDANCE_V1"
    if (actor_header[:3] != ["BLUE_TEAM_MAPPO_V10", "10", "10"]
            or training_header[:3]
                != ["BLUE_TEAM_TRAINING_V7", "7", "10"]
            or semantic not in actor_header or semantic not in training_header):
        raise ValueError("Local checkpoint is not a compatible V10/V7 pair")
    updates = curriculum.read_updates()
    if updates != expected_updates:
        raise ValueError(
            f"Expected run-84 update {expected_updates}, found {updates}. "
            "Use the original latest checkpoint or resume the existing session.")
    if _sha256(LATEST_ACTOR) != _sha256(curriculum.PROD):
        raise ValueError(
            "The initial latest/prod actors differ; run-84 source is ambiguous")
    return updates


def _write_state(path: Path, state: dict) -> None:
    """Atomically persist local progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _create_source_backup(session_root: Path, source_updates: int) -> None:
    """Copy the immutable local source checkpoint and record its digests."""
    source_root = session_root / "source-run84"
    source_root.mkdir(parents=True, exist_ok=False)
    source_files = {
        "latestActor": LATEST_ACTOR,
        "latestTraining": curriculum.LATEST,
        "productionActor": curriculum.PROD,
    }
    manifest = {
        "formatVersion": SESSION_FORMAT_VERSION,
        "sourceUpdates": source_updates,
        "files": {},
    }
    for label, source in source_files.items():
        destination = source_root / source.name
        if destination.exists():
            destination = source_root / f"{label}-{source.name}"
        shutil.copy2(source, destination)
        manifest["files"][label] = {
            "path": str(destination.relative_to(session_root)),
            "sha256": _sha256(destination),
        }
    _write_state(source_root / "manifest.json", manifest)


def _load_or_create_state(options: LauncherOptions) -> tuple[dict, Path]:
    """Open a resumable local session or create one from the run-84 pair."""
    state_path = options.state_root / STATE_NAME
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        required = {
            "formatVersion", "profile", "sourceUpdates", "blockIndex",
            "lastCheckpointUpdates", "sessionDirectory",
        }
        if (not isinstance(state, dict) or not required.issubset(state)
                or state["formatVersion"] != SESSION_FORMAT_VERSION
                or state["profile"] != options.profile
                or state["sourceUpdates"] != options.expected_source_updates
                or type(state["blockIndex"]) is not int
                or state["blockIndex"] < 0
                or type(state["lastCheckpointUpdates"]) is not int):
            raise ValueError("Incompatible local GUI curriculum state")
        current_updates = curriculum.read_updates()
        if current_updates < state["lastCheckpointUpdates"]:
            raise ValueError("Latest checkpoint predates local GUI progress")
        if current_updates > state["lastCheckpointUpdates"]:
            print("[LOCAL_GUI] Recovered newer checkpoint at update "
                  f"{current_updates}", flush=True)
            state["lastCheckpointUpdates"] = current_updates
            _write_state(state_path, state)
        session_root = Path(state["sessionDirectory"])
        resolved_session = session_root.resolve()
        if (not resolved_session.is_relative_to(options.state_root.resolve())
                or not resolved_session.is_dir()):
            raise ValueError("Local GUI session directory is missing")
        return state, resolved_session

    source_updates = _validate_checkpoint_pair(
        options.expected_source_updates)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_root = options.state_root / f"session-{timestamp}"
    session_root.mkdir(parents=True, exist_ok=False)
    _create_source_backup(session_root, source_updates)
    state = {
        "formatVersion": SESSION_FORMAT_VERSION,
        "profile": options.profile,
        "sourceUpdates": source_updates,
        "blockIndex": 0,
        "lastCheckpointUpdates": source_updates,
        "sessionDirectory": str(session_root),
    }
    _write_state(state_path, state)
    return state, session_root


def _run_gui_block(binary: Path, seconds: int, log_path: Path,
                   combined_path: Path) -> bool:
    """Run one visible C++ process and stream its diagnostics to both logs."""
    command = [str(binary), "--auto-run", "--example", "blue-defense",
               "--max-seconds", str(seconds)]
    completed = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as block_log, (
            combined_path.open("a", encoding="utf-8")) as combined_log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                block_log.write(line)
                combined_log.write(line)
                if "[GUI_AUTORUN] Automatic block completed" in line:
                    completed = True
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code != 0:
        raise RuntimeError(
            f"GUI training process failed with exit code {return_code}; "
            f"see {log_path}")
    return completed


def _print_schedule(stages: list[dict], target_index: int,
                    validated_gates: list[str], profile: str,
                    start_block: int) -> None:
    """Print the next exact weighted cycle and its block counts."""
    schedule = balanced_schedule(
        stages, target_index, validated_gates, profile, start_block)
    identifiers = [stage["id"] for stage in schedule]
    counts = Counter(identifiers)
    print(f"[LOCAL_GUI] {profile} blocks from index {start_block}: "
          + " -> ".join(identifiers))
    print("[LOCAL_GUI] cycle shares: "
          + ", ".join(f"{name}={counts[name]}"
                      for name in [stage["id"] for stage in stages]))


def run(options: LauncherOptions) -> int:
    """Execute the resumable local balanced curriculum."""
    os.chdir(ROOT)
    if not options.binary.is_file():
        raise FileNotFoundError(
            f"Missing Release GUI executable: {options.binary}")
    if not 1 <= options.max_seconds <= 86400:
        raise ValueError("max-seconds must be in [1, 86400]")
    if not 1 <= options.block_updates <= 1000:
        raise ValueError("block-updates must be in [1, 1000]")
    if options.max_blocks < 0:
        raise ValueError("max-blocks must be nonnegative")

    stages, target_index, validated_gates = load_production_stages()
    if options.dry_run:
        updates = curriculum.read_updates()
        if not (options.state_root / STATE_NAME).is_file():
            _validate_checkpoint_pair(options.expected_source_updates)
        start_block = 0
        state_path = options.state_root / STATE_NAME
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            start_block = int(state["blockIndex"])
        _print_schedule(stages, target_index, validated_gates,
                        options.profile, start_block)
        cadence = json.loads(curriculum.TRAINING_FILE.read_text(
            encoding="utf-8"))["training"]["checkpointEvery"]
        print(f"[LOCAL_GUI] checkpoint update={updates}; next target="
              f"{next_update_target(updates, cadence, options.block_updates)}")
        return 0

    state, session_root = _load_or_create_state(options)
    _print_schedule(stages, target_index, validated_gates,
                    options.profile, state["blockIndex"])
    preserved = FileSnapshot(PRESERVED_RUNTIME_FILES)
    original_training = json.loads(
        curriculum.TRAINING_FILE.read_text(encoding="utf-8"))
    original_scenario = json.loads(
        curriculum.SCENARIO_FILE.read_text(encoding="utf-8"))
    cadence = original_training["training"].get("checkpointEvery")
    if type(cadence) is not int or cadence < 1:
        raise ValueError("checkpointEvery must be a positive integer")

    deadline = time.monotonic() + options.max_seconds
    completed_blocks = 0
    combined_path = session_root / "training.log"
    try:
        while time.monotonic() + 20 < deadline:
            if options.max_blocks and completed_blocks >= options.max_blocks:
                break
            updates = curriculum.read_updates()
            selected = profiles.select_weighted_stage(
                stages, target_index, state["blockIndex"],
                validated_gates, options.profile)
            target = next_update_target(
                updates, cadence, options.block_updates)
            block_root = session_root / "blocks" / (
                f"block-{state['blockIndex']:06d}-{selected['id']}"
                f"-updates-{updates:06d}-{target:06d}")
            log_path = block_root / "training.log"
            remaining = max(1, int(deadline - time.monotonic()))
            curriculum.apply_stage(
                selected, target, original_training, original_scenario)
            print(f"[LOCAL_GUI] block={state['blockIndex']} "
                  f"replay={selected['id']} updates={updates}->{target} "
                  f"remaining_s={remaining}", flush=True)
            completed = _run_gui_block(
                options.binary, remaining, log_path, combined_path)
            observed = curriculum.read_updates()
            if not updates <= observed <= target:
                raise RuntimeError(
                    "Checkpoint update counter escaped the requested block")
            if observed > updates:
                state["lastCheckpointUpdates"] = observed
            if completed:
                if observed != target:
                    raise RuntimeError(
                        "GUI reported completion before the target update")
                state["blockIndex"] += 1
                completed_blocks += 1
            _write_state(options.state_root / STATE_NAME, state)
            if not completed:
                print("[LOCAL_GUI] Visual run stopped before block completion; "
                      "the same rehearsal will resume next time.", flush=True)
                break
    finally:
        preserved.restore()

    print(f"[LOCAL_GUI] stopped at update={curriculum.read_updates()} "
          f"block={state['blockIndex']}; source backup={session_root / 'source-run84'}")
    return 0


def parse_args(argv: list[str] | None = None) -> LauncherOptions:
    """Parse the local launcher command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path,
        default=Path("build/release/Main/Release/PPO.exe"))
    parser.add_argument(
        "--profile", choices=tuple(profiles.WEIGHTS), default="balanced_v1")
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--block-updates", type=int, default=10)
    parser.add_argument("--expected-source-updates", type=int, default=1790)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--max-blocks", type=int, default=0,
        help="Stop after N completed GUI blocks; zero uses the time budget.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate run 84 and print the exact upcoming schedule.")
    args = parser.parse_args(argv)
    binary = args.binary if args.binary.is_absolute() else ROOT / args.binary
    state_root = (args.state_root if args.state_root.is_absolute()
                  else ROOT / args.state_root)
    return LauncherOptions(
        binary=binary, profile=args.profile, max_seconds=args.max_seconds,
        block_updates=args.block_updates,
        expected_source_updates=args.expected_source_updates,
        state_root=state_root, max_blocks=args.max_blocks,
        dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with concise diagnostics for a double-click console."""
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("\n[LOCAL_GUI] Interrupted; latest checkpoint remains resumable.")
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[LOCAL_GUI] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
