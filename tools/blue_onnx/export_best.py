"""Export only verified Blue MAPPO BEST checkpoint pairs to ONNX.

This script reads the champion artifact without modifying any source checkpoint.
It supports scenario-specific namespaces produced by champion_checkpoints.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from export import export, parse_checkpoint

GATES = frozenset(("1v0", "2v0", "1v1", "1v2", "2v2", "3v3",
                   "4v0", "4v1", "4v4"))
VARIANTS = {
    "fixed": Path("."),
    "fixed-zones-on": Path("zones-on"),
    "randomized": Path("objective-randomized"),
    "randomized-zones-on": Path("objective-randomized/zones-on"),
}
ACTOR_NAME = "blue_patrol_generalized_v10.txt"
TRAINING_NAME = "blue_patrol_generalized_training_v7.txt"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidates(root: Path, gate: str | None,
                variant: str) -> list[Path]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown BEST scenario variant: {variant}")
    if gate is not None:
        if gate not in GATES:
            raise ValueError(f"Unknown BEST gate: {gate}")
        folder = root / VARIANTS[variant] / gate
        if not (folder / "champion.json").is_file():
            raise ValueError(f"Missing evaluated BEST manifest: {folder}")
        return [folder]
    if not root.is_dir():
        return []
    # Scenario namespaces may be nested under objective-randomized/zones-on.
    return sorted(p.parent for p in root.rglob("champion.json")
                  if p.parent.name in GATES)


def _verified_training(folder: Path) -> Path | None:
    manifest = json.loads(
        (folder / "champion.json").read_text(encoding="utf-8")
    )
    if manifest.get("gate") != folder.name:
        raise ValueError(f"Champion gate mismatch: {folder}")
    if manifest.get("evaluatedForGate") is not True:
        return None

    actor_path = folder / ACTOR_NAME
    training_path = folder / TRAINING_NAME
    for checkpoint in (actor_path, training_path):
        if not checkpoint.is_file():
            raise ValueError(f"Incomplete BEST pair: {folder}")
        if manifest.get("sha256", {}).get(checkpoint.name) != _digest(checkpoint):
            raise ValueError(f"BEST SHA256 mismatch: {checkpoint}")

    v10 = parse_checkpoint(actor_path.read_text(encoding="utf-8"))
    v7 = parse_checkpoint(training_path.read_text(encoding="utf-8"))
    if v10.format != "V10" or v7.format != "V7":
        raise ValueError(f"Unexpected BEST checkpoint format: {folder}")
    if (v10.metadata["blue"] != v7.metadata["blue"]
            or v10.metadata["red"] != v7.metadata["red"]):
        raise ValueError(f"BEST P/N mismatch: {folder}")
    if (type(manifest.get("updates")) is not int
            or v7.metadata["updates"] != manifest["updates"]):
        raise ValueError(f"BEST update mismatch: {folder}")

    for name in ("red", "friendly"):
        for key, value in v10.actor[name].items():
            np.testing.assert_array_equal(value, v7.actor[name][key])
    if len(v10.actor["layers"]) != len(v7.actor["layers"]):
        raise ValueError(f"BEST actor layer mismatch: {folder}")
    for layer10, layer7 in zip(v10.actor["layers"], v7.actor["layers"]):
        np.testing.assert_array_equal(layer10["w"], layer7["w"])
        np.testing.assert_array_equal(layer10["b"], layer7["b"])
    np.testing.assert_array_equal(v10.actor["log_std"], v7.actor["log_std"])
    return training_path


def export_best(root: Path, output: Path, *,
                gate: str | None = None, variant: str = "fixed") -> list[str]:
    exported = []
    for folder in _candidates(root, gate, variant):
        training = _verified_training(folder)
        if training is None:
            if gate is not None:
                raise ValueError(f"BEST not yet evaluated: {folder}")
            continue
        relative = folder.relative_to(root)
        export(training, output / relative)
        exported.append(relative.as_posix())
    if gate is not None and not exported:
        raise ValueError("No evaluated BEST pair to export")
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", choices=sorted(GATES))
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="fixed")
    args = parser.parse_args()
    try:
        exported = export_best(args.best_root, args.output_dir,
                               gate=args.gate, variant=args.variant)
    except (ValueError, OSError, AssertionError) as exc:
        parser.error(str(exc))
    if exported:
        print("ONNX exported for:", ", ".join(exported))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write("### Read-only ONNX export\n")
                stream.write(
                    f"- Source training run: {os.environ.get('SOURCE_RUN_ID', 'local')}\n"
                )
                stream.write(f"- Evaluated BEST: {', '.join(exported)}\n")
                stream.write("- Blue Actor + centralized Critic\n")
                stream.write("- ONNX graph verified; direct C++ RLTools parity "
                             "remains unverified\n")
    else:
        print("No evaluated BEST checkpoint in this run; nothing to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
