"""Validate versioned branch-local MAPPO checkpoints for restart CI bootstraps.

The branch's LATEST training checkpoint is the only optimizer warm start. PROD
is preserved as its separate inference incumbent; neither checkpoint certifies
a curriculum gate without a deterministic evaluation.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

# Match the import contract used by progressive_gate.py when tests run via
# unittest discover -s tools/ci/adaptive_curriculum.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import blue_checkpoint


ACTOR_NAME = blue_checkpoint.POLICY_NAME
TRAINING_NAME = blue_checkpoint.TRAINING_NAME
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _checkpoint(path: Path, kind: str) -> blue_checkpoint.BlueCheckpoint:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty branch checkpoint: {path}")
    try:
        return blue_checkpoint.read(path, kind)
    except ValueError as error:
        raise ValueError(f"Incompatible branch checkpoint header: {error}") from None


def _digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def validate_branch_checkpoints(root: Path) -> dict:
    """Require a complete optimizer plus compatible LATEST and PROD actors."""
    latest_actor = root / "latest" / ACTOR_NAME
    latest_training = root / "latest" / TRAINING_NAME
    prod_actor = root / "prod" / ACTOR_NAME
    actor = _checkpoint(latest_actor, "policy")
    training = _checkpoint(latest_training, "training")
    prod = _checkpoint(prod_actor, "policy")
    # Both actors meet the same contract; as the former headers did, they must
    # also record the same P/N.
    if (actor.blue, actor.red) != (prod.blue, prod.red):
        raise ValueError("Branch LATEST/PROD actor architectures differ")
    updates = training.updates
    if updates <= 0:
        raise ValueError("Branch optimizer has no completed PPO updates")
    hashes = {
        "latestActor": _digest(latest_actor),
        "latestTraining": _digest(latest_training),
        "prodActor": _digest(prod_actor),
    }
    return {"updates": updates, "sha256": hashes}


def branch_source_provenance(bootstrap_run_id: int, branch_sha: str,
                             hashes: dict) -> dict:
    """Keep Git commit identity and all checkpoint hashes, not a fake run ID."""
    if (type(bootstrap_run_id) is not int or bootstrap_run_id <= 0
            or not isinstance(branch_sha, str) or not SHA.fullmatch(branch_sha)
            or not isinstance(hashes, dict)
            or set(hashes) != {"latestActor", "latestTraining", "prodActor"}
            or any(not isinstance(value, str) or not DIGEST.fullmatch(value)
                   for value in hashes.values())):
        raise ValueError("Invalid branch-checkpoint source provenance")
    return {
        "sourceMode": "branch",
        "sourceRunId": None,
        "sourceBranchSha": branch_sha,
        "sourceCheckpointSha256": dict(hashes),
    }
