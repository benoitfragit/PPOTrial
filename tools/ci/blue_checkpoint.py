#!/usr/bin/env python3
"""Blue MAPPO checkpoints of BluePatrol: their file names and their header checks.

The checkpoints are PPOKit artifacts (``.ppokit``), as written by ppokit-run and
the common ``ppokit_graph_network`` DLL::

    {"schema":"ppokit.artifact/1","kind":"training"|"policy","identity":{...},"updates":N,...}
    contract 10 <observation version> 295 156 2 128 3 470 256 3 UNIFIED_PHYSICAL_GUIDANCE_V1
    team <P> <N> 9 10 8 1 61
    actor ...

Artifacts of earlier CI runs (champions, restart seeds, resume artifacts) are in
the former formats, which the C++ loaders still read by content:

    BLUE_TEAM_TRAINING_V7 7 10 295 156 2 128 3 470 256 3 <P> <N> 9 10 8 1 61 UNIFIED_PHYSICAL_GUIDANCE_V1
    BLUE_TEAM_MAPPO_V10 10 10 295 156 2 128 3 <P> <N> 9 10 8 1 61 UNIFIED_PHYSICAL_GUIDANCE_V1

A V7 training file holds its completed PPO update count on an ``UPDATES n``
line; a V10 actor has none. Both formats are accepted with the same contract
checks (observation version, network sizes, capacities, guidance semantics, P/N
range). Former files restored under their former names are moved to the new
names (``adopt``): the loaders find the checkpoints by name, then read them by
content.

Command line (CI workflows)::

    python tools/ci/blue_checkpoint.py adopt <checkpoints root>
    python tools/ci/blue_checkpoint.py check --kind training|policy <file>...
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys


# The project's checkpoint names (latest/ and prod/), and their former names.
POLICY_NAME = "policy.ppokit"
TRAINING_NAME = "latest.ppokit"
FORMER_POLICY_NAME = "blue_patrol_generalized_v10.txt"
FORMER_TRAINING_NAME = "blue_patrol_generalized_training_v7.txt"
FORMER_NAMES = {POLICY_NAME: FORMER_POLICY_NAME, TRAINING_NAME: FORMER_TRAINING_NAME}
# PPO.exe's promotion score sits next to the promoted actor.
SCORE_SUFFIX = ".score"

SCHEMA = "ppokit.artifact/1"
TASK = "blue_patrol"
KINDS = ("training", "policy")
OBSERVATION_VERSION = "10"
POLICY_VERSION = "10"
# Raw observation, compact observation, actions, actor hidden width, actor layers.
ACTOR_SIZES = ("295", "156", "2", "128", "3")
# Critic input, hidden width and layers.
CRITIC_SIZES = ("470", "256", "3")
# Blue capacity, red capacity, blue tracks, objective tracks, lidar rays.
CAPACITIES = ("9", "10", "8", "1", "61")
GUIDANCE = "UNIFIED_PHYSICAL_GUIDANCE_V1"
OBSERVATION_SIZE = 295
ACTION_SIZE = 2
MAX_BLUE = 9
MAX_RED = 10

FORMER_POLICY_MAGIC = "BLUE_TEAM_MAPPO_V10"
FORMER_TRAINING_MAGIC = "BLUE_TEAM_TRAINING_V7"
FORMER_TRAINING_VERSION = "7"
FORMER_UPDATES = "UPDATES"


@dataclass(frozen=True)
class BlueCheckpoint:
    """What the CI needs of one Blue checkpoint.

    ``updates`` is the completed PPO update count: the header's for an
    artifact, the ``UPDATES`` line of a former V7 file, None for a former V10
    actor (which never recorded it).
    """

    kind: str
    former: bool
    blue: int
    red: int
    updates: int | None


def read(path: Path, kind: str | None = None) -> BlueCheckpoint:
    """Read and validate the header of a Blue checkpoint of either format.

    Raises ValueError for a missing or empty file, an unknown format, another
    contract, a P/N outside 1..9 / 0..10, or a kind other than ``kind``.
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"Unknown Blue checkpoint kind: {kind}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty Blue checkpoint: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first = stream.readline()
        tokens = first.split()
        if first.lstrip().startswith("{"):
            checkpoint = _read_artifact(path, first, stream)
        elif tokens[:1] == [FORMER_TRAINING_MAGIC]:
            checkpoint = _read_former_training(path, tokens, stream)
        elif tokens[:1] == [FORMER_POLICY_MAGIC]:
            checkpoint = _read_former_policy(path, tokens)
        else:
            raise ValueError(f"Unsupported Blue checkpoint header: {path}")
    if kind is not None and checkpoint.kind != kind:
        raise ValueError(f"Expected a Blue {kind} checkpoint, found {checkpoint.kind}: {path}")
    return checkpoint


def read_graph(path: Path, kind: str) -> int:
    """Validate the common backend header; the native loader verifies the tensors."""
    if kind not in KINDS:
        raise ValueError(f"Unknown Blue checkpoint kind: {kind}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty Blue checkpoint: {path}")
    with path.open("r", encoding="utf-8") as stream:
        try:
            header = json.loads(stream.readline())
        except (ValueError, UnicodeError):
            raise ValueError(f"Expected a common graph artifact (convert legacy checkpoint first): {path}") from None
        identity = header.get("identity") if isinstance(header, dict) else None
        if (not isinstance(identity, dict) or header.get("schema") != SCHEMA
                or header.get("kind") != kind or identity.get("task") != TASK
                or identity.get("backend") != "graph" or identity.get("scalar") != "f32"
                or identity.get("observation_size") != OBSERVATION_SIZE
                or identity.get("action_size") != ACTION_SIZE
                or not identity.get("semantic_v1")
                or type(header.get("updates")) is not int or header["updates"] < 0
                or not stream.readline().startswith("actor ")):
            raise ValueError(f"Expected a common graph {kind} artifact (convert legacy checkpoint first): {path}")
        return header["updates"]


def read_updates(path: Path) -> int:
    """The completed PPO update count of a training checkpoint."""
    updates = read(path, "training").updates
    assert updates is not None
    return updates


def locate(directory: Path, name: str) -> Path:
    """The checkpoint `name` of `directory`, else its former name when only that exists."""
    current = directory / name
    former = directory / FORMER_NAMES[name]
    if not current.exists() and former.exists():
        return former
    return current


def adopt_former_names(root: Path) -> list[tuple[Path, Path]]:
    """Move restored former-named files of latest/ and prod/ to the new names.

    Call it right after restoring checkpoints of an earlier run: a restored file
    replaces the checkout's file of the new name. Returns the moves made.
    """
    moves: list[tuple[Path, Path]] = []
    for folder in (root / "latest", root / "prod"):
        for name, former_name in FORMER_NAMES.items():
            for suffix in ("", SCORE_SUFFIX):
                former = folder / (former_name + suffix)
                if former.is_file():
                    current = folder / (name + suffix)
                    os.replace(former, current)
                    moves.append((former, current))
    return moves


def _contract_mismatch(path: Path, observation_version: str) -> ValueError:
    if observation_version != OBSERVATION_VERSION:
        return ValueError(f"Blue checkpoint observation schema mismatch (version "
                          f"{observation_version}, expected {OBSERVATION_VERSION}): {path}")
    return ValueError(f"Incompatible Blue checkpoint contract (network sizes, "
                      f"capacities or guidance semantics): {path}")


def _team(path: Path, blue: str, red: str) -> tuple[int, int]:
    if not (blue.isdigit() and red.isdigit()
            and 1 <= int(blue) <= MAX_BLUE and 0 <= int(red) <= MAX_RED):
        raise ValueError(f"Blue checkpoint P/N outside 1..{MAX_BLUE} / 0..{MAX_RED}: {path}")
    return int(blue), int(red)


def _read_artifact(path: Path, first: str, stream) -> BlueCheckpoint:
    """A PPOKit artifact: JSON header, contract line, team line."""
    try:
        header = json.loads(first)
    except json.JSONDecodeError:
        raise ValueError(f"Unsupported Blue checkpoint header: {path}") from None
    if not isinstance(header, dict) or header.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported Blue checkpoint header: {path}")
    kind = header.get("kind")
    identity = header.get("identity")
    updates = header.get("updates")
    if (kind not in KINDS or not isinstance(identity, dict)
            or identity.get("task") != TASK
            or identity.get("observation_size") != OBSERVATION_SIZE
            or identity.get("action_size") != ACTION_SIZE
            or type(updates) is not int or updates < 0):
        raise ValueError(f"Not a Blue patrol artifact of these networks: {path}")
    contract = stream.readline().split()
    if len(contract) < 3 or contract[0] != "contract":
        raise ValueError(f"Blue artifact without a contract line: {path}")
    expected = ["contract", POLICY_VERSION, OBSERVATION_VERSION,
                *ACTOR_SIZES, *CRITIC_SIZES, GUIDANCE]
    if contract != expected:
        raise _contract_mismatch(path, contract[2])
    team = stream.readline().split()
    if len(team) != 3 + len(CAPACITIES) or team[0] != "team":
        raise ValueError(f"Blue artifact without a team line: {path}")
    if tuple(team[3:]) != CAPACITIES:
        raise _contract_mismatch(path, OBSERVATION_VERSION)
    blue, red = _team(path, team[1], team[2])
    return BlueCheckpoint(kind, False, blue, red, updates)


def _read_former_training(path: Path, tokens: list[str], stream) -> BlueCheckpoint:
    """A former V7 training file: header, then its UPDATES line further on."""
    if len(tokens) < 3 or tokens[1] != FORMER_TRAINING_VERSION:
        raise ValueError(f"Unsupported Blue checkpoint header: {path}")
    expected = [FORMER_TRAINING_MAGIC, FORMER_TRAINING_VERSION, OBSERVATION_VERSION,
                *ACTOR_SIZES, *CRITIC_SIZES, "P", "N", *CAPACITIES, GUIDANCE]
    team_index = 3 + len(ACTOR_SIZES) + len(CRITIC_SIZES)
    if (len(tokens) != len(expected)
            or tokens[:team_index] + ["P", "N"] + tokens[team_index + 2:] != expected):
        raise _contract_mismatch(path, tokens[2])
    blue, red = _team(path, tokens[team_index], tokens[team_index + 1])
    for line in stream:
        if line.startswith(FORMER_UPDATES + " "):
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                raise ValueError(f"Malformed update marker: {path}")
            return BlueCheckpoint("training", True, blue, red, int(parts[1]))
    raise ValueError(f"Missing update marker: {path}")


def _read_former_policy(path: Path, tokens: list[str]) -> BlueCheckpoint:
    """A former V10 actor file: one header line."""
    if len(tokens) < 3 or tokens[1] != POLICY_VERSION:
        raise ValueError(f"Unsupported Blue checkpoint header: {path}")
    expected = [FORMER_POLICY_MAGIC, POLICY_VERSION, OBSERVATION_VERSION,
                *ACTOR_SIZES, "P", "N", *CAPACITIES, GUIDANCE]
    team_index = 3 + len(ACTOR_SIZES)
    if (len(tokens) != len(expected)
            or tokens[:team_index] + ["P", "N"] + tokens[team_index + 2:] != expected):
        raise _contract_mismatch(path, tokens[2])
    blue, red = _team(path, tokens[team_index], tokens[team_index + 1])
    return BlueCheckpoint("policy", True, blue, red, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    adopt = commands.add_parser("adopt", help="move restored former-named checkpoints to the new names")
    adopt.add_argument("root", type=Path, help="the project's checkpoints folder")
    check = commands.add_parser("check", help="validate Blue checkpoints of one kind")
    check.add_argument("--kind", choices=KINDS, required=True)
    check.add_argument("--graph-only", action="store_true", help="require a common graph artifact")
    check.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "adopt":
            for former, current in adopt_former_names(args.root):
                print(f"[BLUE_CHECKPOINT] {former} -> {current}", flush=True)
            return 0
        for path in args.paths:
            if args.graph_only:
                updates = read_graph(path, args.kind)
                print(f"[BLUE_CHECKPOINT] {path}: {args.kind} (graph) updates={updates}", flush=True)
                continue
            checkpoint = read(path, args.kind)
            updates = "n/a" if checkpoint.updates is None else checkpoint.updates
            print(f"[BLUE_CHECKPOINT] {path}: {checkpoint.kind} "
                  f"({'former' if checkpoint.former else 'ppokit'}) "
                  f"P={checkpoint.blue} N={checkpoint.red} updates={updates}", flush=True)
        return 0
    except (OSError, ValueError) as error:
        print(f"[BLUE_CHECKPOINT_ERROR] {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
