"""Blue checkpoint texts for the Python tests: valid headers of both formats.

Only the header lines that blue_checkpoint reads are real; the tensors are
replaced by a short marker so the tests can tell files apart.
"""
from __future__ import annotations

import json

import blue_checkpoint


def artifact(kind: str, updates: int = 0, blue: int = 4, red: int = 4,
             observation_version: str = blue_checkpoint.OBSERVATION_VERSION,
             body: str = "actor 0\n") -> str:
    """A PPOKit artifact (`latest.ppokit` or `policy.ppokit`) of the Blue patrol."""
    header = {
        "schema": blue_checkpoint.SCHEMA, "kind": kind,
        "identity": {"task": blue_checkpoint.TASK, "task_hash": "c622113e7b80cdce",
                     "graph_hash": "b59d4878de182ff3", "backend": "rltools",
                     "scalar": "f32",
                     "observation_size": blue_checkpoint.OBSERVATION_SIZE,
                     "action_size": blue_checkpoint.ACTION_SIZE,
                     "actor_blocks": [], "critic_blocks": [], "components": []},
        "updates": updates, "seed": 42, "created": "2026-11-24T16:30:00Z",
    }
    contract = " ".join(["contract", blue_checkpoint.POLICY_VERSION, observation_version,
                         *blue_checkpoint.ACTOR_SIZES, *blue_checkpoint.CRITIC_SIZES,
                         blue_checkpoint.GUIDANCE])
    team = " ".join(["team", str(blue), str(red), *blue_checkpoint.CAPACITIES])
    return f"{json.dumps(header, separators=(',', ':'))}\n{contract}\n{team}\n{body}"


def training(updates: int, blue: int = 4, red: int = 4, body: str = "actor 0\n") -> str:
    """A training artifact at `updates` completed PPO updates."""
    return artifact("training", updates, blue, red, body=body)


def policy(blue: int = 4, red: int = 4, body: str = "actor 0\n") -> str:
    """A policy artifact (converted from V10: no update count)."""
    return artifact("policy", 0, blue, red, body=body)


def former_policy_header(blue: int = 4, red: int = 4,
                         observation_version: str = blue_checkpoint.OBSERVATION_VERSION) -> str:
    """The first line of a former V10 actor file."""
    return " ".join([blue_checkpoint.FORMER_POLICY_MAGIC, blue_checkpoint.POLICY_VERSION,
                     observation_version, *blue_checkpoint.ACTOR_SIZES, str(blue), str(red),
                     *blue_checkpoint.CAPACITIES, blue_checkpoint.GUIDANCE]) + "\n"


def former_policy(blue: int = 4, red: int = 4, body: str = "0.5\n") -> str:
    """A former V10 actor file."""
    return former_policy_header(blue, red) + body


def former_training(updates: int, blue: int = 4, red: int = 4,
                    observation_version: str = blue_checkpoint.OBSERVATION_VERSION,
                    body: str = "") -> str:
    """A former V7 training file with its UPDATES line."""
    header = " ".join([blue_checkpoint.FORMER_TRAINING_MAGIC,
                       blue_checkpoint.FORMER_TRAINING_VERSION, observation_version,
                       *blue_checkpoint.ACTOR_SIZES, *blue_checkpoint.CRITIC_SIZES,
                       str(blue), str(red), *blue_checkpoint.CAPACITIES,
                       blue_checkpoint.GUIDANCE])
    return (f"{header}\nACTOR\n{former_policy_header(blue, red, observation_version)}"
            f"{body}UPDATES {updates}\nRNG\nEND_BLUE_TEAM_TRAINING\n")


def graph_artifact(kind: str, updates: int = 0) -> str:
    """A common graph header with a minimal stand-in payload."""
    header = json.loads(artifact(kind, updates).splitlines()[0])
    header["identity"].update(backend="graph", semantic_v1="9750bd0a5e6ef43b")
    return json.dumps(header) + "\nactor 15\n"
