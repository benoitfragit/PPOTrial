#!/usr/bin/env python3
"""Validate a completed gate artifact and launch the next independent GitHub run.

This script runs only after the train job has uploaded the full checkpoint.
Never dispatch on missing/malformed artifacts, unsuccessful evaluations,
mismatched configuration, unexpected gates or user-disabled auto-advance.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from progressive_gate import validate_gates, GATES_FILE, RESULT_FILE


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
    stages = validate_gates(document)
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
    next_gate = stages[index + 1]
    if (result.get("nextGate") != next_gate["id"]
            or result.get("nextThresholdPercent") != next_gate["thresholdPercent"]
            or result.get("nextMaxSeconds") != next_gate["maxSeconds"]
            or not isinstance(run_id, str) or not run_id.isdigit()):
        raise ValueError("Invalid next gate handoff")
    return {
        "max_seconds": str(next_gate["maxSeconds"]),
        "save_every_updates": "1",
        "resume_run_id": run_id,
        "progressive_gate": next_gate["id"],
        "success_percent": str(next_gate["thresholdPercent"]),
        "auto_advance": "true",
    }


def main() -> int:
    result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    inputs = next_dispatch(
        result, GATES_FILE.read_bytes(), os.environ["GITHUB_RUN_ID"])
    if inputs is None:
        print("[ADAPTIVE_GATE] No auto-dispatch; gate failed, stopped, or finished.")
        return 0
    token = os.environ["GITHUB_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    ref = os.environ["GITHUB_REF"]
    if not ref.startswith("refs/heads/"):
        raise ValueError("Progressive training only supported on branch refs")
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
