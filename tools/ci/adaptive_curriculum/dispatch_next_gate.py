#!/usr/bin/env python3
"""Validate a completed gate artifact and launch the next independent GitHub run.

This script runs only after the train job has uploaded the full checkpoint.
Never dispatch on missing/malformed artifacts, unsuccessful evaluations,
mismatched configuration, unexpected gates or user-disabled auto-advance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

from progressive_gate import validate_gates, select_curriculum, GATES_FILE, RESULT_FILE
import champion_checkpoints as champion
import rehearsal_profiles as profiles


def matching_gate_fingerprint(recorded: object, config_bytes: bytes) -> bool:
    """Accept only identical gate bytes apart from Git checkout LF/CRLF.

    GitHub Actions trains on Windows but dispatches on Ubuntu. Windows checkout
    hashes CRLF bytes in the checkpoint; Ubuntu sees LF bytes for the same file.
    Both encodings are equivalent without relaxing the actual gate definition.
    """
    if not isinstance(recorded, str):
        return False
    lf_bytes = config_bytes.replace(b"\r\n", b"\n")
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
    return recorded in {
        hashlib.sha256(lf_bytes).hexdigest(),
        hashlib.sha256(crlf_bytes).hexdigest(),
    }


def next_dispatch(result: dict, config_bytes: bytes, run_id: str) -> dict | None:
    document = json.loads(config_bytes)
    contract_size = result.get("maxTeamSize", 2) if isinstance(result, dict) else None
    stages = select_curriculum(document, contract_size)
    if (not isinstance(result, dict)
            or not matching_gate_fingerprint(
                result.get("gateConfigSha256"), config_bytes)):
        raise ValueError("Gate configuration fingerprint mismatch")
    names = [item["id"] for item in stages]
    current = result.get("gate")
    if current not in names:
        raise ValueError("Unknown completed gate")
    index = names.index(current)
    if not result.get("passed") or not result.get("autoAdvance"):
        return None
    if index + 1 >= len(stages):
        return None
    profile = result.get("rehearsalProfile", "legacy")
    if profile not in profiles.PROFILES:
        raise ValueError("Unknown rehearsal profile in gate result")
    next_gate = stages[index + 1]
    if (result.get("nextGate") != next_gate["id"]
            or result.get("nextThresholdPercent") != next_gate["thresholdPercent"]
            or result.get("nextMaxSeconds") != next_gate["maxSeconds"]
            or not isinstance(run_id, str) or not run_id.isdigit()):
        raise ValueError("Invalid next gate handoff")
    inputs = {
        "max_seconds": str(next_gate["maxSeconds"]),
        "save_every_updates": "1",
        "resume_run_id": run_id,
        "progressive_gate": next_gate["id"],
        "success_percent": str(next_gate["thresholdPercent"]),
        "auto_advance": "true",
    }
    # Existing legacy dispatches must keep their exact historical inputs.
    if profile != "legacy":
        inputs["rehearsal_profile"] = profile
    if contract_size != 2:
        inputs["max_team_size"] = str(contract_size)
    return inputs


CHECKPOINT_STATE_FILE = Path("build/mappo-checkpoint/adaptive-curriculum-progress.json")


def next_dispatch_from_checkpoint(
        result: dict, state: dict, config_bytes: bytes, run_id: str) -> dict | None:
    """Dispatch from the verified training output, preserving restart-cycle choices.

    In same-threshold mode the bootstrap deliberately passes auto_advance=false
    to training. This job owns the only continuation; no workflow_run listener
    is required (GitHub-token-dispatched runs may not wake that listener).
    """
    contract_size = state.get("maxTeamSize", 2) if isinstance(state, dict) else None
    stages = select_curriculum(json.loads(config_bytes), contract_size)
    ids = [stage["id"] for stage in stages]
    if (not isinstance(state, dict)
            or not isinstance(result, dict)
            or state.get("maxTeamSize", 2) != contract_size
            or result.get("maxTeamSize", 2) != contract_size
            or not matching_gate_fingerprint(
                state.get("gateConfigSha256"), config_bytes)
            or state.get("gateConfigSha256") != result.get("gateConfigSha256")
            or state.get("gate") != result.get("gate")
            or state.get("gate") not in ids
            or type(state.get("gateIndex")) is not int
            or state["gateIndex"] != ids.index(state["gate"])
            or type(state.get("passed")) is not bool
            or state["passed"] is not result.get("passed")
            or type(state.get("lastCheckpointUpdates")) is not int
            or state["lastCheckpointUpdates"] != result.get("optimizerUpdates")):
        raise ValueError("Gate result and optimizer checkpoint do not match")
    if (not isinstance(state.get("validatedGates"), list)
            or (state["gate"] in state["validatedGates"]) is not state["passed"]):
        raise ValueError("Inconsistent verified gate history")

    profile = state.get("rehearsalProfile", "legacy")
    if (profile not in profiles.PROFILES
            or result.get("rehearsalProfile", "legacy") != profile):
        raise ValueError("Gate result and checkpoint rehearsal profiles differ")
    cycle = state.get("restartCycle")
    if (isinstance(cycle, dict)
            and cycle.get("rehearsalProfile", profile) != profile):
        raise ValueError("Restart-cycle rehearsal profile mismatch")
    if cycle is None:
        inputs = next_dispatch(result, config_bytes, run_id)
        if inputs is None:
            return None
        uniform = state.get("sameThresholdAllGates", False)
        if type(uniform) is not bool:
            raise ValueError("Invalid uniform gate threshold selection")
        if uniform:
            target = state.get("uniformThresholdPercent")
            if (type(target) not in (int, float)
                    or not math.isfinite(target)
                    or not 0 < target <= 100
                    or result.get("sameThresholdAllGates") is not True
                    or result.get("uniformThresholdPercent") != target
                    or result.get("thresholdPercent") != target):
                raise ValueError("Gate result does not match selected uniform threshold")
            inputs["success_percent"] = f"{target:g}"
        else:
            if result.get("sameThresholdAllGates") not in (None, False):
                raise ValueError("Gate result contradicts per-gate threshold selection")
        # Explicit input is essential: the GitHub action now defaults to true.
        inputs["same_threshold_all_gates"] = ("true" if uniform else "false")
        return inputs
    if (not isinstance(cycle, dict)
            or type(cycle.get("schemaVersion")) is not int
            or cycle["schemaVersion"] != 1
            or type(cycle.get("bootstrapRunId")) is not int
            or cycle["bootstrapRunId"] <= 0
            or type(cycle.get("sourceRunId")) is not int
            or cycle["sourceRunId"] <= 0
            or type(cycle.get("sameThresholdAllGates")) is not bool
            or type(cycle.get("autoAdvance")) is not bool
            or type(cycle.get("successPercent")) not in (float, int)):
        raise ValueError("Invalid restart-cycle provenance")
    threshold = float(cycle["successPercent"])
    if not 0 < threshold <= 100:
        raise ValueError("Invalid restart-cycle success threshold")

    if not cycle["sameThresholdAllGates"]:
        if result.get("autoAdvance") is not cycle["autoAdvance"]:
            raise ValueError("Inconsistent auto-advance choice")
        inputs = next_dispatch(result, config_bytes, run_id)
        if inputs is not None:
            inputs["same_threshold_all_gates"] = "false"
        return inputs

    # Only fixed-threshold cycles must retain the bootstrap success target.
    # In per-gate mode a later result legitimately uses a different threshold.
    current_threshold = result.get("thresholdPercent")
    if (not isinstance(current_threshold, (int, float))
            or isinstance(current_threshold, bool)
            or float(current_threshold) != threshold):
        raise ValueError("Restart-cycle success threshold mismatch")
    # No second orchestrator: same-threshold runs explicitly disable the
    # ordinary dispatcher while retaining the user's opt-in in restartCycle.
    if result.get("autoAdvance") is not False:
        raise ValueError("Fixed-threshold cycle enabled a duplicate dispatcher")
    if not cycle["autoAdvance"] or not result["passed"]:
        return None
    inputs = next_dispatch(dict(result, autoAdvance=True), config_bytes, run_id)
    if inputs is not None:
        inputs["success_percent"] = f"{threshold:g}"
        inputs["auto_advance"] = "false"
        inputs["same_threshold_all_gates"] = "true"
    return inputs


def verify_bootstrap_provenance(cycle: dict, repository: str,
                                ref: str, token: str) -> None:
    """Require the source to be a successful, same-branch restart workflow."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/runs/"
        f"{cycle['bootstrapRunId']}",
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(request, timeout=30) as response:
        bootstrap = json.load(response)
    if (bootstrap.get("status") != "completed"
            or bootstrap.get("conclusion") != "success"
            or bootstrap.get("event") != "workflow_dispatch"
            or bootstrap.get("repository", {}).get("full_name") != repository
            or bootstrap.get("head_branch") != ref.removeprefix("refs/heads/")
            or not bootstrap.get("path", "").startswith(
                ".github/workflows/mappo-restart-progressive.yml")):
        raise ValueError("Invalid restart-cycle bootstrap workflow provenance")


def main() -> int:
    result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    state = json.loads(CHECKPOINT_STATE_FILE.read_text(encoding="utf-8"))
    inputs = next_dispatch_from_checkpoint(
        result, state, GATES_FILE.read_bytes(), os.environ["GITHUB_RUN_ID"])
    if inputs is None:
        print("[ADAPTIVE_GATE] No auto-dispatch; gate failed, stopped, or finished.")
        return 0
    # The artifact must contain a verified actor + FULL optimizer pair.
    # Never dispatch an apparently passing run with only a newer LATEST.
    best = champion.read(result["gate"], CHECKPOINT_STATE_FILE.parent / "best")
    if (best is None or not best["evaluatedForGate"]
            or best["successPercent"] != state["bestSuccessPercent"]):
        raise ValueError("Auto-dispatch requires a matching evaluated champion pair")
    token = os.environ["GITHUB_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    ref = os.environ["GITHUB_REF"]
    if not ref.startswith("refs/heads/"):
        raise ValueError("Progressive training only supported on branch refs")
    cycle = state.get("restartCycle")
    if isinstance(cycle, dict) and cycle.get("sameThresholdAllGates") is True:
        verify_bootstrap_provenance(cycle, repository, ref, token)
    url = ("https://api.github.com/repos/" + repository
           + "/actions/workflows/mappo-training-windows.yml/dispatches")
    payload = json.dumps({"ref": ref, "inputs": inputs}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 204:
            raise RuntimeError(f"GitHub dispatch returned {response.status}")
    print(f"[ADAPTIVE_GATE] Scheduled independent run for {inputs['progressive_gate']}; "
          f"continuing checkpoint from run {inputs['resume_run_id']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
