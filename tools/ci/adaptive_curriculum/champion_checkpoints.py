"""Versioned, immutable-at-evaluation champion pairs for progressive MAPPO.

Only completed deterministic evaluations may promote a pair. The V7 checkpoint
is kept byte-for-byte (actor, critic, Adam, RNG, update counter). The global
4v4 production checkpoint is not the curriculum's cross-gate handoff format.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import mappo_curriculum as existing


ROOT = existing.MODEL_ROOT / "best"
LATEST_ACTOR = existing.MODEL_ROOT / "latest/blue_patrol_generalized_v10.txt"
ACTOR_NAME = "blue_patrol_generalized_v10.txt"
TRAINING_NAME = "blue_patrol_generalized_training_v7.txt"
MANIFEST_NAME = "champion.json"
SAVED = re.compile(
    r"Saved actor and training checkpoint at update (\d+)"
)
START = re.compile(
    r"Starting deterministic P=(\d+) N=(\d+) evaluation after update (\d+)"
)
START_PREFIX = re.compile(
    r"Starting deterministic P=(\d+) N=(\d+) evaluation after"
)
COMPLETE = re.compile(
    r"Evaluation #\d+ P=(\d+) N=(\d+) score=([0-9.eE+-]+) "
    r"success=([0-9.eE+-]+)"
)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _updates(path: Path) -> int:
    return existing.read_updates(path)


def _validate(directory: Path, gate: str) -> dict:
    metadata = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if (metadata.get("version") != 1 or metadata.get("gate") != gate
            or type(metadata.get("updates")) is not int
            or metadata["updates"] < 0
            or type(metadata.get("evaluatedForGate")) is not bool
            or not isinstance(metadata.get("originGate"), str)
            or not isinstance(metadata.get("sha256"), dict)):
        raise ValueError(f"Invalid champion metadata for {gate}")
    actor, training = directory / ACTOR_NAME, directory / TRAINING_NAME
    if not actor.is_file() or not training.is_file():
        raise ValueError(f"Missing mandatory champion pair for {gate}")
    with actor.open("rb") as actor_stream, training.open("rb") as training_stream:
        actor_header = actor_stream.readline().split()
        training_header = training_stream.readline().split()
    semantic = b"UNIFIED_PHYSICAL_GUIDANCE_V1"
    if (actor.stat().st_size == 0 or training.stat().st_size == 0
            or actor_header[:3] != [b"BLUE_TEAM_MAPPO_V10", b"10", b"10"]
            or training_header[:3] != [b"BLUE_TEAM_TRAINING_V7", b"7", b"10"]
            or semantic not in actor_header or semantic not in training_header
            or _updates(training) != metadata["updates"]):
        raise ValueError(f"Incomplete/incompatible champion checkpoint for {gate}")
    for path in (actor, training):
        if metadata["sha256"].get(path.name) != _digest(path):
            raise ValueError(f"Champion checksum mismatch: {path}")
    if metadata["evaluatedForGate"]:
        success, score = metadata.get("successPercent"), metadata.get("score")
        if (type(success) not in (float, int) or not math.isfinite(success)
                or not 0 <= success <= 100
                or type(score) not in (float, int) or not math.isfinite(score)
                or not 0 <= score <= 100):
            raise ValueError(f"Invalid champion evaluation for {gate}")
    elif metadata.get("successPercent") is not None or metadata.get("score") is not None:
        raise ValueError(f"Unevaluated champion seed has evaluation metrics: {gate}")
    return metadata


def read(gate: str, root: Path = ROOT) -> dict | None:
    directory = root / gate
    if not directory.exists():
        return None
    return _validate(directory, gate)


def _store(gate: str, source_actor: Path, source_training: Path,
           origin: str, evaluation: dict | None) -> dict:
    directory = ROOT / gate
    incoming = ROOT / (gate + ".incoming")
    backup = ROOT / (gate + ".backup")
    ROOT.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        # An interrupted rename must never silently replace the last champion.
        raise ValueError(f"Unresolved champion backup for {gate}")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir()
    try:
        shutil.copy2(source_actor, incoming / ACTOR_NAME)
        shutil.copy2(source_training, incoming / TRAINING_NAME)
        record = {
            "version": 1, "gate": gate, "originGate": origin,
            "evaluatedForGate": evaluation is not None,
            "updates": _updates(incoming / TRAINING_NAME),
            "successPercent": evaluation["successPercent"] if evaluation else None,
            "score": evaluation["score"] if evaluation else None,
            "sha256": {
                ACTOR_NAME: _digest(incoming / ACTOR_NAME),
                TRAINING_NAME: _digest(incoming / TRAINING_NAME),
            },
        }
        (incoming / MANIFEST_NAME).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _validate(incoming, gate)
        if directory.exists():
            directory.rename(backup)
        try:
            incoming.rename(directory)
        except Exception:
            if backup.exists():
                backup.rename(directory)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if incoming.exists():
            shutil.rmtree(incoming)
    return read(gate)


def import_legacy_pass(gate: str, success: float) -> dict:
    """Import only an old run whose FINAL passing evaluation is its best.

    Its old evaluation score was not archived with V7. The synthetic zero
    score is solely an intra-gate tie-break fallback; no cross-gate comparison.
    """
    if not math.isfinite(success) or not 0 <= success <= 100:
        raise ValueError("Invalid legacy passing success")
    return _store(gate, LATEST_ACTOR, existing.LATEST,
                  "legacy-v1-gate-pass",
                  {"successPercent": success, "score": 0.0})


def seed(gate: str, source: str | None,
         source_actor: Path | None = None,
         source_training: Path | None = None) -> dict:
    """Initialize next gate from its predecessor's complete champion pair."""
    if source is not None:
        record = read(source)
        if record is None:
            raise ValueError(f"Missing previous gate champion: {source}")
        source_actor = ROOT / source / ACTOR_NAME
        source_training = ROOT / source / TRAINING_NAME
    else:
        source_actor = source_actor or LATEST_ACTOR
        source_training = source_training or existing.LATEST
    assert source_actor is not None and source_training is not None
    return _store(gate, source_actor, source_training,
                  source or "legacy-latest", evaluation=None)


def restore_latest(gate: str) -> int:
    """Restore actor and full V7 training together; never mix their epochs."""
    record = read(gate)
    if record is None:
        raise ValueError(f"No champion to restore: {gate}")
    previous = [(LATEST_ACTOR, LATEST_ACTOR.read_bytes()
                 if LATEST_ACTOR.is_file() else None),
                (existing.LATEST, existing.LATEST.read_bytes()
                 if existing.LATEST.is_file() else None)]
    try:
        for dest, source in (
            (LATEST_ACTOR, ROOT / gate / ACTOR_NAME),
            (existing.LATEST, ROOT / gate / TRAINING_NAME),
        ):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    except Exception:
        for dest, value in previous:
            if value is None:
                dest.unlink(missing_ok=True)
            else:
                dest.write_bytes(value)
        raise
    if existing.read_updates() != record["updates"]:
        raise ValueError("Restored champion update mismatch")
    return record["updates"]


def evaluation_at_update(log: Path, blue: int, red: int,
                         checkpoint_updates: int) -> dict | None:
    """Require a completed evaluation of exactly the saved optimizer update."""
    if not log.is_file():
        return None
    started = None
    saved_update = None
    result = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        saved = SAVED.search(line)
        if saved:
            saved_update = int(saved[1])
        start = START.search(line)
        if start:
            started = (int(start[1]), int(start[2]), int(start[3]))
        elif (prefix := START_PREFIX.search(line)) is not None:
            # C++ stdout from other workers occasionally splits "update"
            # midway. The successful actor+V7 save immediately before the
            # campaign remains an authoritative optimizer epoch.
            started = (int(prefix[1]), int(prefix[2]), saved_update)
        complete = COMPLETE.search(line)
        if complete:
            scenario = (int(complete[1]), int(complete[2]))
            if (started == (*scenario, checkpoint_updates)
                    and saved_update == checkpoint_updates):
                score, success = float(complete[3]), float(complete[4]) * 100.0
                if not (math.isfinite(score) and 0 <= score <= 100
                        and math.isfinite(success) and 0 <= success <= 100):
                    raise ValueError("Nonfinite/out-of-range evaluation result")
                if scenario == (blue, red):
                    result = {"score": score, "successPercent": success}
            started = None
    return result


def promote(gate: str, evaluation: dict, updates: int) -> tuple[bool, dict]:
    current = read(gate)
    if current is not None and current["evaluatedForGate"]:
        if ((evaluation["successPercent"], evaluation["score"])
                <= (current["successPercent"], current["score"])):
            return False, current
    if existing.read_updates() != updates:
        raise ValueError("Evaluation and V7 optimizer checkpoint are not the same update")
    promoted = _store(gate, LATEST_ACTOR, existing.LATEST, gate, evaluation)
    if promoted["updates"] != updates:
        raise ValueError("Promoted champion update mismatch")
    return True, promoted


def install_production_for_stage(stage: dict) -> None:
    """Isolated C++ evaluation incumbent; external 4v4 PROD is restored later."""
    record = read(stage["id"])
    if record is None:
        existing.PROD.unlink(missing_ok=True)
        existing.PROD_SCORE.unlink(missing_ok=True)
        return
    existing.PROD.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / stage["id"] / ACTOR_NAME, existing.PROD)
    if record["evaluatedForGate"]:
        existing.PROD_SCORE.write_text(
            "BLUE_PATROL_GENERALIZED_EVALUATION_V1 "
            f"{stage['blue']} {stage['red']} {record['score']:.9g}\n",
            encoding="utf-8")
    else:
        existing.PROD_SCORE.unlink(missing_ok=True)


