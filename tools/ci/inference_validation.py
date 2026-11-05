#!/usr/bin/env python3
"""Read-only deterministic MAPPO evaluation across the reduced curriculum gates.

Operates only on the current CI checkout. It never calls a trainer and never
uploads, modifies, or promotes model checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


GATES = (("1v0", 1, 0), ("2v0", 2, 0), ("1v1", 1, 1),
         ("2v1", 2, 1), ("2v2", 2, 2))
EVALUATION = re.compile(
    r"\[BLUE MAPPO V10\] Evaluation #\d+ P=(\d+) N=(\d+) "
    r"score=([0-9.eE+-]+) success=([0-9.eE+-]+) "
    r"blue-survival=([0-9.eE+-]+) red-neutralization=([0-9.eE+-]+) "
    r"objective-survival=([0-9.eE+-]+)"
)
EPISODE = re.compile(
    r"\[MAPPO_EVAL_EPISODE\] update=\d+ campaign=\d+ "
    r"case=(\d+) seed=(\d+) steps=(\d+) P=(\d+) N=(\d+) "
    r".*? terminal=([a-z_]+)"
)
MODEL_ROOT = Path("resources/model/BluePatrol")
TRAINING_CONFIG = Path("resources/settings/Training/BluePatrol.json")
SCENARIO_CONFIG = Path("resources/settings/Scenarios/BlueDefense.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_fingerprints(root: Path) -> dict[str, str]:
    """Capture all source model files, not just the actor used in inference."""
    if not root.is_dir():
        raise FileNotFoundError(f"Missing model root: {root}")
    return {str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix in (".txt", ".score", ".json")}


def selected_gates(value: str) -> list[tuple[str, int, int]]:
    if value == "all":
        return list(GATES)
    if value == "patrol":
        return [gate for gate in GATES if gate[2] == 0]
    if value == "combat":
        return [gate for gate in GATES if gate[2] > 0]
    gates = [g for g in GATES if g[0] == value]
    if not gates:
        raise ValueError(f"Unsupported gate: {value}")
    return gates


def evaluation_result(log: str, gate: tuple[str, int, int],
                      seed_base: int, episodes: int) -> dict:
    """Accept only a complete campaign and matching per-case provenance."""
    name, blue, red = gate
    matches = list(EVALUATION.finditer(log))
    if len(matches) != 1:
        raise ValueError(f"{name}: expected one completed campaign, got {len(matches)}")
    m = matches[0]
    if (int(m[1]), int(m[2])) != (blue, red):
        raise ValueError(f"{name}: campaign P/N does not match requested gate")
    outcomes = []
    for case in EPISODE.finditer(log):
        if (int(case[4]), int(case[5])) != (blue, red):
            raise ValueError(f"{name}: mixed P/N episode telemetry")
        outcomes.append(dict(case=int(case[1]), seed=int(case[2]),
                             steps=int(case[3]), termination=case[6]))
    # Interleaved simulator logs can break individual lines. Never certify a
    # partially measured campaign; a retry must re-evaluate it.
    if len(outcomes) != episodes:
        raise ValueError(
            f"{name}: expected {episodes} episode diagnostics, got {len(outcomes)}")
    observed = {(item["case"], item["seed"]) for item in outcomes}
    expected = {(i + 1, seed_base + i) for i in range(episodes)}
    if observed != expected:
        raise ValueError(f"{name}: incorrect or repeated independent seeds")
    score, success, blue_survival, red_neutralization, objective_survival = (
        float(m[i]) for i in range(3, 8))
    import math
    values = (score, success, blue_survival, red_neutralization, objective_survival)
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"{name}: nonfinite evaluation metrics")
    if any(not 0.0 <= x <= 1.0 for x in values[1:]):
        raise ValueError(f"{name}: rate outside [0,1]")
    return dict(gate=name, blue=blue, red=red, episodes=episodes,
                success_percent=round(100.0 * success, 4),
                score=round(score, 5),
                blue_survival_percent=round(100.0 * blue_survival, 4),
                red_neutralization_percent=round(100.0 * red_neutralization, 4),
                objective_survival_percent=round(100.0 * objective_survival, 4),
                cases=outcomes)


def configure(gate: tuple[str, int, int], episodes: int,
              seed_base: int, original_training: bytes,
              original_scenario: bytes) -> None:
    _, blue, red = gate
    training = json.loads(original_training)
    scenario = json.loads(original_scenario)
    training["evaluation"]["episodes"] = episodes
    training["evaluation"]["seed"] = seed_base
    parameters = scenario["blueDefense"]
    parameters["blueTeam"]["activeCount"] = blue
    parameters["redTeam"]["activeCount"] = red
    parameters["restrictedZones"]["enabled"] = False
    parameters["restrictedZoneSpawn"]["enabledProbability"] = 0.0
    TRAINING_CONFIG.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    SCENARIO_CONFIG.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")


def render_summary(results: list[dict], actor_hash: str,
                   source_run_id: int, threshold: float) -> str:
    lines = [
        "# MAPPO read-only inference validation",
        "",
        f"- Source workflow run: **{source_run_id}**",
        f"- Actor SHA-256: `{actor_hash}`",
        f"- Deterministic inference: **yes**, PPO updates: **0**",
        f"- No model checkpoint saved or promoted",
        f"- Acceptance threshold: **{threshold:g}%** per gate",
        "",
        "| Gate | Success | Score | Blue survival | Red neutralization | Objective survival | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for x in results:
        passed = x["success_percent"] >= threshold
        lines.append(
            f'| {x["gate"]} | {x["success_percent"]:.1f}% | {x["score"]:.2f} '
            f'| {x["blue_survival_percent"]:.1f}% '
            f'| {x["red_neutralization_percent"]:.1f}% '
            f'| {x["objective_survival_percent"]:.1f}% '
            f'| {"PASS" if passed else "FAIL"} |'
        )
    all_pass = bool(results) and all(
        x["success_percent"] >= threshold for x in results)
    lines += ["", f"**Overall assessment: {'PASS' if all_pass else 'FAIL'}**",
              "", "FAIL is a model assessment, not a CI infrastructure failure.",
              "Raw episode results and logs are retained as report artifacts.",
              ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.episodes <= 100:
        raise ValueError("--episodes must be in [1,100]")
    if not 0 <= args.threshold <= 100:
        raise ValueError("--threshold must be in [0,100]")
    if not 1 <= args.seed_base <= 4294967295 - args.episodes:
        raise ValueError("--seed-base is outside the uint32 evaluation range")
    if not 30 <= args.timeout_seconds <= 18000:
        raise ValueError("--timeout-seconds must be in [30,18000]")
    actor = MODEL_ROOT / "prod/blue_patrol_generalized_v10.txt"
    binary = Path(args.binary).resolve()
    if not binary.is_file() or not actor.is_file():
        raise FileNotFoundError("Missing PPO.exe Release or promoted blue actor")
    with actor.open("rb") as stream:
        actor_header = stream.readline()
    if not actor_header.startswith(b"BLUE_TEAM_MAPPO_V10 10 10 "):
        raise ValueError("Expected V10 unified blue actor")
    baseline = model_fingerprints(MODEL_ROOT)
    actor_hash = digest(actor)
    original_training = TRAINING_CONFIG.read_bytes()
    original_scenario = SCENARIO_CONFIG.read_bytes()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for gate in selected_gates(args.gate):
            configure(gate, args.episodes, args.seed_base,
                      original_training, original_scenario)
            name = gate[0]
            print(f"[INFERENCE] Evaluating gate {name}, {args.episodes} cases", flush=True)
            try:
                completed = subprocess.run(
                    [str(binary), "--no-gui", "--example",
                     "blue-defense-evaluation", "--max-seconds",
                     str(args.timeout_seconds)],
                    # Keep stdout telemetry isolated from concurrent std::clog
                    # reset messages on stderr. Merging OS pipes can splice a
                    # patrol reset into the middle of MAPPO_EVAL_EPISODE.
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf-8", errors="replace", timeout=args.timeout_seconds + 90,
                    check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{name}: inference timed out") from exc
            telemetry = completed.stdout
            diagnostics = completed.stderr
            (output / f"{name}.log").write_text(telemetry, encoding="utf-8")
            (output / f"{name}.stderr.log").write_text(diagnostics, encoding="utf-8")
            if (completed.returncode != 0
                    or any(marker in text for text in (telemetry, diagnostics)
                           for marker in ("[MAPPO_ERROR]", "[MAPPO_FATAL]"))):
                raise RuntimeError(
                    f"{name}: inference failed (exit {completed.returncode}); see logs")
            metric = evaluation_result(telemetry, gate, args.seed_base, args.episodes)
            results.append(metric)
            print(f"[INFERENCE] {name} success={metric['success_percent']:.1f}% "
                  f"score={metric['score']:.2f}", flush=True)
            if model_fingerprints(MODEL_ROOT) != baseline:
                raise RuntimeError("Model files changed during read-only inference")
    finally:
        TRAINING_CONFIG.write_bytes(original_training)
        SCENARIO_CONFIG.write_bytes(original_scenario)
    summary = render_summary(results, actor_hash, args.source_run_id, args.threshold)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    (output / "results.json").write_text(json.dumps(
        dict(source_run_id=args.source_run_id,
             actor_sha256=actor_hash, deterministic=True, training_updates=0,
             seed_base=args.seed_base, results=results),
        indent=2) + "\n", encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "gate", "success_percent", "score", "blue_survival_percent",
            "red_neutralization_percent", "objective_survival_percent"))
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as stream:
            stream.write("\n" + summary)
    print(summary, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/release/Main/Release/PPO.exe")
    parser.add_argument("--output", default="build/inference-validation")
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--gate", choices=("all", "patrol", "combat") + tuple(g[0] for g in GATES),
                        default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2004242)
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[INFERENCE][ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
