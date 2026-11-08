"""Validate versioned branch-local MAPPO checkpoints for restart CI bootstraps.

The branch's LATEST V7 is the only optimizer warm start. PROD is preserved as
its separate inference incumbent; neither checkpoint certifies a curriculum
gate without a deterministic evaluation.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

# Match the import contract used by progressive_gate.py when tests run via
# unittest discover -s tools/ci/adaptive_curriculum.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mappo_curriculum


ACTOR_NAME = "blue_patrol_generalized_v10.txt"
TRAINING_NAME = "blue_patrol_generalized_training_v7.txt"
SEMANTIC = b"UNIFIED_PHYSICAL_GUIDANCE_V1"
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _header(path: Path, prefix: tuple[bytes, ...]) -> list[bytes]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty branch checkpoint: {path}")
    with path.open("rb") as stream:
        tokens = stream.readline().split()
    if tokens[:len(prefix)] != list(prefix) or SEMANTIC not in tokens:
        raise ValueError(f"Incompatible branch checkpoint header: {path}")
    return tokens


def _digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def validate_branch_checkpoints(root: Path) -> dict:
    """Require a complete V7 optimizer plus compatible LATEST and PROD actors."""
    latest_actor = root / "latest" / ACTOR_NAME
    latest_training = root / "latest" / TRAINING_NAME
    prod_actor = root / "prod" / ACTOR_NAME
    expected_actor = (b"BLUE_TEAM_MAPPO_V10", b"10", b"10")
    actor_header = _header(latest_actor, expected_actor)
    _header(latest_training, (b"BLUE_TEAM_TRAINING_V7", b"7", b"10"))
    prod_header = _header(prod_actor, expected_actor)
    if actor_header[3:] != prod_header[3:]:
        raise ValueError("Branch LATEST/PROD actor architectures differ")
    updates = mappo_curriculum.read_updates(latest_training)
    if updates <= 0:
        raise ValueError("Branch V7 optimizer has no completed PPO updates")
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
