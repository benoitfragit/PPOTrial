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
import champion_checkpoints as champion  # noqa: E402


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
    expected_source_updates: int | None
    state_root: Path
    max_blocks: int
    dry_run: bool
    max_team_size: int = 2
    target_gate: str | None = None
    objective_layout: str = "fixed"
    hostile_zones: bool = False
    validated_gates: tuple[str, ...] | None = None
    source_actor: Path | None = None
    source_training: Path | None = None


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


def load_production_stages(
        max_team_size: int = 2, target_gate: str | None = None,
        validated_gates: list[str] | None = None
        ) -> tuple[list[dict], int, list[str]]:
    """Load a canonical gate prefix without assuming new gate certifications."""
    stages = adaptive.select_curriculum(
        json.loads(adaptive.GATES_FILE.read_bytes()), max_team_size)
    names = [stage["id"] for stage in stages]
    requested = target_gate or f"{max_team_size}v{max_team_size}"
    if requested not in names:
        raise ValueError("Gate is outside selected max_team_size")
    index = names.index(requested)
    # Historical run 84 only: 1v0, 2v0, 1v1, 2v1 were validated.
    prior = (names[:index] if max_team_size == 2 and requested == TARGET_GATE
             and validated_gates is None else list(validated_gates or []))
    if len(prior) != len(set(prior)) or any(x not in names[:index] for x in prior):
        raise ValueError("Only unique previously validated gates are allowed")
    return stages, index, prior


def scenario_variant(options: LauncherOptions) -> str:
    """Stable identity for the two orthogonal CI scenario options."""
    if options.objective_layout not in ("fixed", "randomized"):
        raise ValueError("Invalid objective layout")
    return (options.objective_layout
            + ("_zones" if options.hostile_zones else "_no_zones"))


def legacy_shortcut(options: LauncherOptions) -> bool:
    """Only the original run-84 invocation uses the original local state."""
    return (options.max_team_size == 2
            and options.target_gate in (None, "2v2")
            and options.validated_gates is None
            and scenario_variant(options) == "fixed_no_zones")


def session_state_root(options: LauncherOptions) -> Path:
    if legacy_shortcut(options):
        return options.state_root
    gate = options.target_gate or f"{options.max_team_size}v{options.max_team_size}"
    return (options.state_root
            / f"{options.max_team_size}v{options.max_team_size}"
            / scenario_variant(options) / gate)


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


def _validate_checkpoint_pair(
        expected_updates: int | None, actor: Path = LATEST_ACTOR,
        training: Path = curriculum.LATEST,
        require_prod_match: bool = True) -> int:
    """Validate an explicit V10/V7 source before updating latest."""
    paths = (actor, training) + ((curriculum.PROD,) if require_prod_match else ())
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing local V10/V7 checkpoint: {path}")
    with actor.open(encoding="utf-8") as stream:
        actor_header = stream.readline().split()
    with training.open(encoding="utf-8") as stream:
        training_header = stream.readline().split()
    semantic = "UNIFIED_PHYSICAL_GUIDANCE_V1"
    if (actor_header[:3] != ["BLUE_TEAM_MAPPO_V10", "10", "10"]
            or training_header[:3]
                != ["BLUE_TEAM_TRAINING_V7", "7", "10"]
            or semantic not in actor_header or semantic not in training_header):
        raise ValueError("Local checkpoint is not a compatible V10/V7 pair")
    updates = curriculum.read_updates(training)
    if expected_updates is not None and updates != expected_updates:
        raise ValueError(
            f"Expected source update {expected_updates}, found {updates}.")
    if require_prod_match and _sha256(actor) != _sha256(curriculum.PROD):
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


def _create_source_backup(session_root: Path, source_updates: int,
                          actor: Path = LATEST_ACTOR,
                          training: Path = curriculum.LATEST) -> None:
    """Copy the selected source pair and record immutable digests."""
    source_root = session_root / "source-run84"
    source_root.mkdir(parents=True, exist_ok=False)
    source_files = {
        "latestActor": actor,
        "latestTraining": training,
        **({"productionActor": curriculum.PROD}
           if curriculum.PROD.is_file() else {}),
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
    state_path = session_state_root(options) / STATE_NAME
    expected_history = (list(options.validated_gates)
                        if options.validated_gates is not None else
                        ["1v0", "2v0", "1v1", "2v1"]
                        if legacy_shortcut(options) else [])
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        required = {
            "formatVersion", "profile", "sourceUpdates", "blockIndex",
            "lastCheckpointUpdates", "sessionDirectory",
        }
        if (not isinstance(state, dict) or not required.issubset(state)
                or state["formatVersion"] != SESSION_FORMAT_VERSION
                or state["profile"] != options.profile
                or (options.expected_source_updates is not None
                    and state["sourceUpdates"] != options.expected_source_updates)
                or state.get("maxTeamSize", 2) != options.max_team_size
                or state.get("validatedGates", expected_history
                             if legacy_shortcut(options) else None)
                    != expected_history
                or state.get("targetGate", "2v2") !=
                    (options.target_gate or f"{options.max_team_size}v{options.max_team_size}")
                or state.get("objectiveLayout", "fixed") != options.objective_layout
                or state.get("hostileZones", False) is not options.hostile_zones
                or state.get("gateConfigSha256",
                    hashlib.sha256(adaptive.GATES_FILE.read_bytes()).hexdigest())
                    != hashlib.sha256(adaptive.GATES_FILE.read_bytes()).hexdigest()
                or type(state["blockIndex"]) is not int
                or state["blockIndex"] < 0
                or type(state["lastCheckpointUpdates"]) is not int):
            raise ValueError("Incompatible local GUI curriculum state")
        if options.source_actor is not None:
            if (_sha256(options.source_actor) != state.get("sourceActorSha256")
                    or _sha256(options.source_training) !=
                       state.get("sourceTrainingSha256")):
                raise ValueError("Resumed V10/V7 source hashes differ")
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
        if (not resolved_session.is_relative_to(session_state_root(options).resolve())
                or not resolved_session.is_dir()):
            raise ValueError("Local GUI session directory is missing")
        return state, resolved_session

    actor = options.source_actor or LATEST_ACTOR
    training = options.source_training or curriculum.LATEST
    external_source = options.source_actor is not None
    source_updates = _validate_checkpoint_pair(
        options.expected_source_updates, actor, training,
        require_prod_match=not external_source)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_root = session_state_root(options) / f"session-{timestamp}"
    session_root.mkdir(parents=True, exist_ok=False)
    _create_source_backup(session_root, source_updates, actor, training)
    if external_source:
        # Keep recovery copies before installing an explicitly selected BEST.
        previous = session_root / "latest-before-install"
        previous.mkdir()
        for target in (LATEST_ACTOR, curriculum.LATEST):
            if target.is_file():
                shutil.copy2(target, previous / target.name)
        staged = []
        try:
            for source_path, target in ((actor, LATEST_ACTOR),
                                        (training, curriculum.LATEST)):
                temporary = target.with_suffix(target.suffix + ".gui-staged")
                temporary.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, temporary)
                staged.append((temporary, target))
            for temporary, target in staged:
                temporary.replace(target)
        except OSError:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
            for target in (LATEST_ACTOR, curriculum.LATEST):
                original = previous / target.name
                if original.is_file():
                    shutil.copy2(original, target)
                else:
                    target.unlink(missing_ok=True)
            raise
    state = {
        "formatVersion": SESSION_FORMAT_VERSION,
        "profile": options.profile,
        "maxTeamSize": options.max_team_size,
        "targetGate": options.target_gate or f"{options.max_team_size}v{options.max_team_size}",
        "objectiveLayout": options.objective_layout,
        "hostileZones": options.hostile_zones,
        "validatedGates": expected_history,
        "gateConfigSha256": hashlib.sha256(adaptive.GATES_FILE.read_bytes()).hexdigest(),
        "sourceUpdates": source_updates,
        "sourceActorSha256": _sha256(actor),
        "sourceTrainingSha256": _sha256(training),
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
    if not options.dry_run and not options.binary.is_file():
        raise FileNotFoundError(
            f"Missing Release GUI executable: {options.binary}")
    if not 1 <= options.max_seconds <= 86400:
        raise ValueError("max-seconds must be in [1, 86400]")
    if not 1 <= options.block_updates <= 1000:
        raise ValueError("block-updates must be in [1, 1000]")
    if options.max_blocks < 0:
        raise ValueError("max-blocks must be nonnegative")

    if (options.source_actor is None) != (options.source_training is None):
        raise ValueError("The source V10 actor and V7 training must be paired")
    history = (list(options.validated_gates)
               if options.validated_gates is not None else
               None if legacy_shortcut(options) else [])
    stages, target_index, validated_gates = load_production_stages(
        options.max_team_size, options.target_gate, history)
    if options.max_team_size > 2 and options.profile != "balanced_v1":
        raise ValueError("3v3/4v4 GUI requires unified balanced_v1")
    print(f"[LOCAL_GUI] max_team_size={options.max_team_size}"
          f" gate={stages[target_index]['id']}"
          f" variant={scenario_variant(options)}"
          f" sha256={hashlib.sha256(adaptive.GATES_FILE.read_bytes()).hexdigest()}"
          f" validated={','.join(validated_gates) or '-'}", flush=True)
    if options.dry_run:
        updates = curriculum.read_updates()
        if legacy_shortcut(options) and not (
                session_state_root(options) / STATE_NAME).is_file():
            _validate_checkpoint_pair(options.expected_source_updates)
        start_block = 0
        state_path = session_state_root(options) / STATE_NAME
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

    if not legacy_shortcut(options):
        if (not (session_state_root(options) / STATE_NAME).is_file()
                and options.source_actor is None):
            raise ValueError(
                "Nonlegacy training requires --source-actor and --source-training")
        with champion.variant_namespace(
                options.hostile_zones, options.objective_layout):
            for prior_gate in validated_gates:
                best = champion.read(prior_gate)
                if best is None or not best["evaluatedForGate"]:
                    raise ValueError(f"Uncertified rehearsal gate: {prior_gate}")
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
            selected = dict(
                selected, hostileZones=options.hostile_zones,
                objectiveLayout=options.objective_layout)
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
            _write_state(session_state_root(options) / STATE_NAME, state)
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
    parser.add_argument("--expected-source-updates", type=int,
                        help="Expected V7 source update; historical default 1790")
    parser.add_argument("--source-actor", type=Path,
                        help="Explicit V10 BEST actor path for new campaigns")
    parser.add_argument("--source-training", type=Path,
                        help="Matching V7 source training checkpoint path")
    parser.add_argument("--max-team-size", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--start-gate", default=None)
    parser.add_argument("--objective-layout", choices=("fixed", "randomized"),
                        default="fixed")
    parser.add_argument("--hostile-zones", action="store_true")
    parser.add_argument("--validated-gates", default=None,
                        help="Comma-separated previously evaluated BEST gates")
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
        expected_source_updates=(args.expected_source_updates
                                 if args.expected_source_updates is not None
                                 else 1790 if args.max_team_size == 2
                                      and args.start_gate in (None, "2v2")
                                      and args.objective_layout == "fixed"
                                      and not args.hostile_zones
                                      and args.source_actor is None
                                 else None),
        state_root=state_root, max_blocks=args.max_blocks,
        dry_run=args.dry_run, max_team_size=args.max_team_size,
        target_gate=args.start_gate,
        objective_layout=args.objective_layout, hostile_zones=args.hostile_zones,
        validated_gates=(tuple(filter(None, args.validated_gates.split(",")))
                         if args.validated_gates is not None else None),
        source_actor=(None if args.source_actor is None else
                      args.source_actor if args.source_actor.is_absolute()
                      else ROOT / args.source_actor),
        source_training=(None if args.source_training is None else
                         args.source_training if args.source_training.is_absolute()
                         else ROOT / args.source_training))


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
