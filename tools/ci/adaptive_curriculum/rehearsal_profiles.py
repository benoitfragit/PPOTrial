"""Versioned, opt-in training rehearsal schedules for progressive Blue MAPPO.

The legacy schedule lives in progressive_gate.py and remains the default.
The balanced_v1 profile replays every earlier curriculum scenario regardless of
checkpoint source or certification. Rehearsal never certifies a gate as passed.
Each named v1 profile is immutable so checkpoint resumes preserve semantics.
"""
from __future__ import annotations

PROFILES = ("legacy", "balanced_v1", "patrol_focus_v1", "combat_focus_v1")

# Integer block shares; each full 2v2 cycle contains exactly 20 blocks.
WEIGHTS = {
    "balanced_v1": {
        "1v0": 3, "2v0": 4, "1v1": 2, "2v1": 3, "2v2": 8,
    },
    "patrol_focus_v1": {
        "1v0": 5, "2v0": 6, "1v1": 2, "2v1": 2, "2v2": 5,
    },
    "combat_focus_v1": {
        "1v0": 2, "2v0": 3, "1v1": 3, "2v1": 4, "2v2": 8,
    },
}



def _adaptive_balanced_weights(
        stages: list[dict], index: int, eligible: list[dict]) -> dict[str, int]:
    """Derive non-uniform 3vX/4vX shares from the current validated gate set.

    A strict majority of every complete deterministic cycle belongs to the
    current gate. Earlier validated patrol and combat gates remain eligible;
    no skipped or future gate can receive weight.
    """
    legacy = ("1v0", "2v0", "1v1", "2v1", "2v2")
    if (len(stages) not in (9, 14)
            or tuple(stage["id"] for stage in stages[:5]) != legacy
            or index < 5 or index >= len(stages)
            or any(stage["id"] != f"{stage['blue']}v{stage['red']}"
                   or not 0 <= stage["red"] <= stage["blue"] <= 4
                   for stage in stages)):
        raise ValueError("Rehearsal profile is incompatible with these gates")
    current = stages[index]
    weights: dict[str, int] = {}
    for previous in eligible:
        if previous["id"] == current["id"]:
            continue
        blue, red = previous["blue"], previous["red"]
        # Keep a patrol and a combat anchor for each validated team size;
        # prioritize the current and immediately preceding team sizes.
        weight = 1 + int(red == 0 or red == blue)
        weight += int(blue in (current["blue"], current["blue"] - 1))
        weights[previous["id"]] = weight
    weights[current["id"]] = max(8, sum(weights.values()) + 1)
    return weights


def _eligible_stages(stages: list[dict], index: int,
                     validated_gates: list[str], profile: str) -> list[dict]:
    """Separate rehearsal eligibility from the certified gate history.

    Balanced rehearses every preceding scenario even for a branch warm start
    or a run with skipped gates. This does not change validatedGates, the
    promotion rules or the selected gate's independent success evaluation.
    Other profiles retain their historical certified-only replay behavior.
    """
    return [stage for position, stage in enumerate(stages)
            if position == index or
            (position < index and
             (profile == "balanced_v1" or stage["id"] in validated_gates))]


def weighted_cycle_length(stages: list[dict], index: int,
                          validated_gates: list[str], profile: str) -> int:
    """Return full cycle length using the CI profile and rehearsal eligibility."""
    if profile not in WEIGHTS or not 0 <= index < len(stages):
        raise ValueError("Invalid weighted rehearsal cycle")
    eligible = _eligible_stages(stages, index, validated_gates, profile)
    weights = (_adaptive_balanced_weights(stages, index, eligible)
               if profile == "balanced_v1" and index >= 5
               else WEIGHTS[profile])
    if any(stage["id"] not in weights for stage in eligible):
        raise ValueError("Rehearsal profile is incompatible with these gates")
    return sum(weights[stage["id"]] for stage in eligible)


def select_weighted_stage(stages: list[dict], index: int, block: int,
                          validated_gates: list[str],
                          profile: str) -> dict:
    """Return a deterministic, evenly interleaved rehearsal stage.

    Balanced_v1 replays all earlier gates independently of checkpoint origin
    and gate certification. Other profiles replay only certified earlier gates.
    Rehearsals never mark an earlier gate passed.
    """
    if profile not in WEIGHTS:
        raise ValueError(f"Unknown weighted rehearsal profile: {profile}")
    if not 0 <= index < len(stages) or block < 0:
        raise ValueError("Invalid weighted rehearsal position")
    if not isinstance(validated_gates, list):
        raise ValueError("Validated rehearsal history must be a list")
    eligible = _eligible_stages(stages, index, validated_gates, profile)
    if profile == "balanced_v1" and index >= 5:
        weights = _adaptive_balanced_weights(stages, index, eligible)
    else:
        weights = WEIGHTS[profile]
    if any(stage["id"] not in weights for stage in eligible):
        raise ValueError("Rehearsal profile is incompatible with these gates")
    total = sum(weights[stage["id"]] for stage in eligible)
    if total <= 0:
        raise ValueError("Empty weighted rehearsal schedule")

    # Smooth weighted round robin avoids long uninterrupted patrol/combat runs.
    credit = {stage["id"]: 0 for stage in eligible}
    schedule = []
    for _ in range(total):
        for stage in eligible:
            credit[stage["id"]] += weights[stage["id"]]
        selected = max(eligible, key=lambda stage: credit[stage["id"]])
        credit[selected["id"]] -= total
        schedule.append(selected)
    return schedule[block % len(schedule)]
