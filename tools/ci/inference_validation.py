#!/usr/bin/env python3
"""Read-only deterministic MAPPO evaluation across the reduced curriculum gates.

Operates only on the current CI checkout. It never calls a trainer and never
uploads, modifies, or promotes model checkpoints. Each gate is a ppokit-run
evaluation of the promoted actor (blue_environment, blue_network): the gate's
P/N in scenario.json and a generated suite eval/inference.json whose seeds are
the holdout seeds, one case each (the cases of PPO.exe's campaign for those
seeds). Results are read from the suite's JSON report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import blue_checkpoint


GATES = (("1v0", 1, 0), ("2v0", 2, 0), ("1v1", 1, 1),
         ("2v1", 2, 1), ("2v2", 2, 2))
PROJECT = Path("Examples/BlueDefense/projects/BluePatrol")
MODEL_ROOT = PROJECT / "checkpoints"
SCENARIO_CONFIG = PROJECT / "scenario.json"
TASK = PROJECT / "task.json"
SUITE_NAME = "inference"
SUITE_FILE = PROJECT / "eval" / f"{SUITE_NAME}.json"
REPORT = MODEL_ROOT / f"evaluation_{SUITE_NAME}.json"
ACTOR = Path("checkpoints/prod") / blue_checkpoint.POLICY_NAME


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_fingerprints(root: Path) -> dict[str, str]:
    """Capture all source model files, not just the actor used in inference.

    Evaluation reports written next to the checkpoints (evaluation_*.json) are
    outputs of the evaluation, not model files.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Missing model root: {root}")
    return {str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix in (".ppokit", ".txt", ".score", ".json")
            and not (p.parent == root and p.name.startswith("evaluation_"))}


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


def metric_names(task: dict) -> list[str]:
    """The per-episode metrics of the report, in task.json order."""
    return list(task.get("metrics", {}).keys())


def case_termination(values: dict[str, float]) -> str:
    """A case's outcome from its metrics: timeout, success, or failure."""
    if values["timeout"] >= 0.5:
        return "timeout"
    return "success" if values["mission_success"] >= 0.5 else "failure"


def evaluation_result(report: dict, names: list[str], gate: tuple[str, int, int],
                      seed_base: int, episodes: int) -> dict:
    """Accept only a complete campaign of the requested holdout seeds."""
    name, blue, red = gate
    if report.get("suite") != SUITE_NAME:
        raise ValueError(f"{name}: the report is not the '{SUITE_NAME}' suite")
    cases = report.get("episodes", [])
    if len(cases) != episodes:
        raise ValueError(f"{name}: expected {episodes} episodes, got {len(cases)}")
    if [case["seed"] for case in cases] != [seed_base + i for i in range(episodes)]:
        raise ValueError(f"{name}: incorrect or repeated independent seeds")
    outcomes = []
    for index, case in enumerate(cases):
        if len(case["metrics"]) != len(names):
            raise ValueError(f"{name}: episode metrics do not match task.json")
        values = dict(zip(names, case["metrics"]))
        outcomes.append(dict(case=index + 1, seed=case["seed"], steps=case["length"],
                             termination=case_termination(values)))
    aggregates = report.get("aggregates", {})
    keys = ("score.mean", "mission_success.rate", "blue_survival.mean",
            "red_neutralization.mean", "objective_survived.rate")
    if any(aggregates.get(key) is None for key in keys):
        raise ValueError(f"{name}: missing evaluation metrics")
    score, success, blue_survival, red_neutralization, objective_survival = (
        float(aggregates[key]) for key in keys)
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
              seed_base: int, original_scenario: bytes) -> None:
    """The gate's teams (no restricted zone) and a suite of its holdout seeds."""
    _, blue, red = gate
    scenario = json.loads(original_scenario)
    parameters = scenario["blueDefense"]
    parameters["blueTeam"]["activeCount"] = blue
    parameters["redTeam"]["activeCount"] = red
    parameters["restrictedZones"]["enabled"] = False
    parameters["restrictedZoneSpawn"]["enabledProbability"] = 0.0
    SCENARIO_CONFIG.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    suite = {"schema": "ppokit.eval/1", "suite": SUITE_NAME,
             "seeds": {"first": seed_base, "count": episodes}, "episodes": 1}
    SUITE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUITE_FILE.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")


def render_summary(results: list[dict], actor_hash: str,
                   source_run_id: int, threshold: float) -> str:
    lines = [
        "# MAPPO read-only inference validation",
        "",
        f"- Source workflow run: **{source_run_id}**",
        f"- Actor SHA-256: `{actor_hash}`",
        f"- Deterministic inference (ppokit-run eval): **yes**, PPO updates: **0**",
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
    actor = PROJECT / ACTOR
    runner = Path(args.ppokit_run).resolve()
    if not runner.is_file() or not actor.is_file():
        raise FileNotFoundError("Missing ppokit-run Release or promoted blue actor")
    # A promoted actor: a policy artifact, or a former V10 file of an earlier run.
    blue_checkpoint.read(actor, "policy")
    names = metric_names(json.loads(TASK.read_text(encoding="utf-8")))
    baseline = model_fingerprints(MODEL_ROOT)
    actor_hash = digest(actor)
    original_scenario = SCENARIO_CONFIG.read_bytes()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for gate in selected_gates(args.gate):
            configure(gate, args.episodes, args.seed_base, original_scenario)
            name = gate[0]
            print(f"[INFERENCE] Evaluating gate {name}, {args.episodes} cases", flush=True)
            try:
                completed = subprocess.run(
                    [str(runner), "eval", "--project", str(PROJECT / "project.legacy.json"),
                     "--suite", SUITE_NAME, "--policy", ACTOR.as_posix()],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf-8", errors="replace", timeout=args.timeout_seconds,
                    check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{name}: inference timed out") from exc
            (output / f"{name}.log").write_text(completed.stdout, encoding="utf-8")
            (output / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if (completed.returncode != 0
                    or any(marker in text for text in (completed.stdout, completed.stderr)
                           for marker in ("[MAPPO_ERROR]", "[MAPPO_FATAL]"))):
                raise RuntimeError(
                    f"{name}: inference failed (exit {completed.returncode}); see logs")
            report = json.loads(REPORT.read_text(encoding="utf-8"))
            (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            metric = evaluation_result(report, names, gate, args.seed_base, args.episodes)
            results.append(metric)
            print(f"[INFERENCE] {name} success={metric['success_percent']:.1f}% "
                  f"score={metric['score']:.2f}", flush=True)
            if model_fingerprints(MODEL_ROOT) != baseline:
                raise RuntimeError("Model files changed during read-only inference")
    finally:
        SCENARIO_CONFIG.write_bytes(original_scenario)
        SUITE_FILE.unlink(missing_ok=True)
        REPORT.unlink(missing_ok=True)
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
    parser.add_argument("--ppokit-run", default="build/release/PPOKit/Release/ppokit-run.exe")
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
