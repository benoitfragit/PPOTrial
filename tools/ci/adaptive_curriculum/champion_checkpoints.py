"""Versioned, immutable-at-evaluation champion pairs for progressive MAPPO.

Only completed deterministic evaluations may promote a pair. The training
checkpoint is kept byte-for-byte (actor, critic, Adam, RNG, update counter). The
global 4v4 production checkpoint is not the curriculum's cross-gate handoff format.
A pair is stored under the names of latest/ (policy.ppokit, latest.ppokit);
champions of earlier runs keep their former V10/V7 names and formats, and are
read, seeded and restored as they are.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import blue_checkpoint
import mappo_curriculum as existing


ROOT = existing.MODEL_ROOT / "best"


def variant_root(hostile_zones: bool, objective_layout: str = "fixed",
                 root: Path | None = None) -> Path:
    """Unique BEST namespace per scenario, preserving historical fixed paths."""
    if type(hostile_zones) is not bool:
        raise ValueError("hostile_zones must be a boolean")
    if objective_layout not in ("fixed", "randomized"):
        raise ValueError("objective_layout must be fixed or randomized")
    result = ROOT if root is None else root
    if objective_layout == "randomized":
        result /= "objective-randomized"
    if hostile_zones:
        result /= "zones-on"
    return result


@contextmanager
def variant_namespace(hostile_zones: bool, objective_layout: str = "fixed"):
    """Use the same namespace for reads, promotions, rehearsals and restores."""
    global ROOT
    previous = ROOT
    ROOT = variant_root(hostile_zones, objective_layout)
    try:
        yield
    finally:
        ROOT = previous

LATEST_ACTOR = existing.LATEST_ACTOR
ACTOR_NAME = blue_checkpoint.POLICY_NAME
TRAINING_NAME = blue_checkpoint.TRAINING_NAME
MANIFEST_NAME = "champion.json"
# One block of ppokit-run: its campaigns ("evaluation 'validation' at update N")
# and the P/N of their episodes; the results are the run journal's events.
EVALUATED = existing.EVAL_PATTERN
EVALUATION_EPISODE = re.compile(
    r"\[MAPPO_EVAL_EPISODE\] .*?\bP=(\d+) N=(\d+)\b"
)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _updates(path: Path) -> int:
    return existing.read_updates(path)


def pair(directory: Path) -> tuple[Path, Path]:
    """The actor and training files of a champion directory (new or former names)."""
    return (blue_checkpoint.locate(directory, ACTOR_NAME),
            blue_checkpoint.locate(directory, TRAINING_NAME))


def _validate(directory: Path, gate: str) -> dict:
    metadata = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if (metadata.get("version") != 1 or metadata.get("gate") != gate
            or type(metadata.get("updates")) is not int
            or metadata["updates"] < 0
            or type(metadata.get("evaluatedForGate")) is not bool
            or not isinstance(metadata.get("originGate"), str)
            or not isinstance(metadata.get("sha256"), dict)):
        raise ValueError(f"Invalid champion metadata for {gate}")
    actor, training = pair(directory)
    if not actor.is_file() or not training.is_file():
        raise ValueError(f"Missing mandatory champion pair for {gate}")
    try:
        blue_checkpoint.read(actor, "policy")
        updates = blue_checkpoint.read_updates(training)
    except ValueError as error:
        raise ValueError(
            f"Incomplete/incompatible champion checkpoint for {gate}: {error}") from None
    if updates != metadata["updates"]:
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


def read(gate: str, root: Path | None = None) -> dict | None:
    directory = (ROOT if root is None else root) / gate
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
        source_actor, source_training = pair(ROOT / source)
    else:
        source_actor = source_actor or LATEST_ACTOR
        source_training = source_training or existing.LATEST
    assert source_actor is not None and source_training is not None
    return _store(gate, source_actor, source_training,
                  source or "legacy-latest", evaluation=None)


def restore_latest(gate: str) -> int:
    """Restore actor and full training state together; never mix their epochs."""
    record = read(gate)
    if record is None:
        raise ValueError(f"No champion to restore: {gate}")
    previous = [(LATEST_ACTOR, LATEST_ACTOR.read_bytes()
                 if LATEST_ACTOR.is_file() else None),
                (existing.LATEST, existing.LATEST.read_bytes()
                 if existing.LATEST.is_file() else None)]
    try:
        actor, training = pair(ROOT / gate)
        for dest, source in ((LATEST_ACTOR, actor), (existing.LATEST, training)):
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


def log_evaluations(log: Path) -> tuple[list[int], set[tuple[int, int]]]:
    """The updates a block's process evaluated, and the P/N of its evaluation episodes."""
    updates: list[int] = []
    scenarios: set[tuple[int, int]] = set()
    if not log.is_file():
        return updates, scenarios
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        evaluated = EVALUATED.search(line)
        if evaluated and evaluated[1] == existing.SUITE:
            updates.append(int(evaluated[2]))
        episode = EVALUATION_EPISODE.search(line)
        if episode:
            scenarios.add((int(episode[1]), int(episode[2])))
    return updates, scenarios


def journal_evaluation(update: int, journal: Path | None = None) -> dict | None:
    """The journal's last campaign of the suite at `update`: score and success."""
    path = existing.JOURNAL if journal is None else journal
    if not path.is_file():
        return None
    result = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A process stopped mid-write leaves at most a partial last line.
        if (event.get("event") != "evaluation" or event.get("suite") != existing.SUITE
                or event.get("update") != update):
            continue
        metrics = event.get("metrics", {})
        score = metrics.get("score.mean")
        success = metrics.get("mission_success.rate")
        if type(score) not in (int, float) or type(success) not in (int, float):
            raise ValueError(f"Journal campaign at update {update} lacks score or success")
        score, success = float(score), float(success) * 100.0
        if not (math.isfinite(score) and 0 <= score <= 100
                and math.isfinite(success) and 0 <= success <= 100):
            raise ValueError("Nonfinite/out-of-range evaluation result")
        result = {"score": score, "successPercent": success}
    return result


def evaluation_at_update(log: Path, blue: int, red: int,
                         checkpoint_updates: int) -> dict | None:
    """Require a completed evaluation of exactly the saved optimizer update.

    The block's process must have evaluated that update (its log), on this P/N
    only (its evaluation episodes); the result is the journal's campaign.
    """
    updates, scenarios = log_evaluations(log)
    if checkpoint_updates not in updates or scenarios != {(blue, red)}:
        return None
    return journal_evaluation(checkpoint_updates)


def promote(gate: str, evaluation: dict, updates: int) -> tuple[bool, dict]:
    current = read(gate)
    if current is not None and current["evaluatedForGate"]:
        if ((evaluation["successPercent"], evaluation["score"])
                <= (current["successPercent"], current["score"])):
            return False, current
    if existing.read_updates() != updates:
        raise ValueError("Evaluation and optimizer checkpoint are not the same update")
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
    shutil.copy2(pair(ROOT / stage["id"])[0], existing.PROD)
    if record["evaluatedForGate"]:
        existing.PROD_SCORE.write_text(
            "BLUE_PATROL_GENERALIZED_EVALUATION_V1 "
            f"{stage['blue']} {stage['red']} {record['score']:.9g}\n",
            encoding="utf-8")
    else:
        existing.PROD_SCORE.unlink(missing_ok=True)


