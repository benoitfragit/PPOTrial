#!/usr/bin/env python3
"""Offline GitHub Actions MAPPO telemetry charts (stdlib only)."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

FIELDS = (
    "elapsed_seconds", "updates", "rollout_steps", "rollout_target",
    "evaluating", "evaluation_progress", "evaluation_cases_done",
    "evaluation_cases_total", "evaluation_steps",
    "evaluation_steps_per_second", "training", "eval_completed",
    "latest_score", "best_score", "eval_success_rate", "instant_reward",
)


def read_rows(path: Path) -> List[Dict[str, float]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("Unexpected MAPPO telemetry CSV schema")
        for line in reader:
            try:
                sample = {key: float(line[key]) for key in FIELDS}
            except (ValueError, TypeError, KeyError):
                continue  # Interrupted process may leave a partial final row.
            if all(math.isfinite(value) for value in sample.values()):
                rows.append(sample)
    return rows


def sparkline(values: List[float], width: int = 48) -> str:
    ticks = "▁▂▃▄▅▆▇█"
    if not values:
        return "(no samples)"
    count = min(width, len(values))
    sampled = [values[int(i * (len(values) - 1) / max(1, count - 1))]
               for i in range(count)]
    lo, hi = min(sampled), max(sampled)
    if hi <= lo:
        return ticks[0] * len(sampled)
    return "".join(ticks[min(7, int(7 * (x - lo) / (hi - lo)))]
                   for x in sampled)


def write_chart(rows: List[Dict[str, float]], out: Path) -> None:
    """Two aligned SVG plots: updates and evaluation progress over wall time."""
    width, height, left, right = 1050, 665, 85, 28
    plot_width = width - left - right
    max_time = max(1.0, rows[-1]["elapsed_seconds"]) if rows else 1.0
    panels = (("updates", "Completed PPO updates", 75, 175),
              ("evaluation_progress", "Evaluation progress (0–1)", 260, 360),
              ("latest_score", "Evaluation score (0–100)", 445, 545))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="MAPPO training and evaluation progress">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        '<style>text{font:15px sans-serif;fill:#e5e7eb}'
        '.grid{stroke:#334155;stroke-width:1}'
        '.plot{fill:none;stroke:#38bdf8;stroke-width:2.5}'
        '</style>',
        '<text x="30" y="32" font-size="23">MAPPO — training telemetry</text>',
    ]
    for field, title, top, bottom in panels:
        maximum = max(1.0, max((r[field] for r in rows), default=0.0))
        if field == "evaluation_progress":
            maximum = 1.0
        if field == "latest_score":
            maximum = 100.0
        parts.append(f'<text x="{left}" y="{top - 16}">{html.escape(title)}</text>')
        for tick in range(5):
            y = bottom - tick * (bottom - top) / 4.0
            value = maximum * tick / 4.0
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" '
                         f'x2="{width-right}" y2="{y:.1f}"/>')
            parts.append(f'<text x="12" y="{y+5:.1f}">{value:.2f}</text>')
        if rows:
            visible = [r for r in rows
                       if field != "latest_score" or r["eval_completed"] > 0.0]
            if visible:
                points = " ".join(
                    f'{left+plot_width*max(0.0, min(r["elapsed_seconds"]/max_time, 1.0)):.1f},'
                    f'{bottom-(bottom-top)*max(0.0, min(r[field]/maximum, 1.0)):.1f}'
                    for r in visible
                )
                parts.append(f'<polyline class="plot" points="{points}"/>')
    parts.append(f'<text x="{left}" y="600">Wall time: 0 – {max_time:.0f} seconds</text>')
    if not rows:
        parts.append('<text x="250" y="230">No telemetry samples received</text>')
    parts.append('</svg>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")



def read_eval_episodes(path: Path) -> List[Dict[str, str]]:
    """Read evaluation-only key/value rows from the headless training log."""
    if not path.is_file():
        return []
    episodes: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            if "[MAPPO_EVAL_EPISODE]" not in line:
                continue
            payload = line.split("[MAPPO_EVAL_EPISODE]", 1)[1]
            fields = dict(token.split("=", 1)
                          for token in payload.split() if "=" in token)
            if all(key in fields for key in
                   ("campaign", "case", "blue_alive", "terminal")):
                try:
                    int(fields["campaign"])
                    int(fields["case"])
                    int(fields["blue_alive"])
                except ValueError:
                    continue
                episodes.append(fields)
    return episodes


LOSS_FIELDS = (
    "update", "campaign", "case", "seed", "step", "blue", "source",
    "speed_before", "speed", "accel_cmd", "flight_model", "low_speed",
    "fuel_empty", "min_speed", "low_speed_threshold",
    "fuel_remaining", "fuel_capacity",
)


def read_eval_losses(path: Path) -> List[Dict[str, str]]:
    """Read one diagnostic record for each newly destroyed evaluation Blue."""
    if not path.is_file():
        return []
    losses: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            if "[MAPPO_EVAL_LOSS]" not in line:
                continue
            payload = line.split("[MAPPO_EVAL_LOSS]", 1)[1]
            fields = dict(token.split("=", 1)
                          for token in payload.split() if "=" in token)
            if not all(name in fields for name in
                       ("campaign", "case", "blue", "source",
                        "low_speed", "fuel_empty", "speed", "step")):
                continue
            try:
                int(fields["campaign"])
                int(fields["case"])
                int(fields["step"])
                speed = float(fields["speed"])
                low_speed = int(fields["low_speed"])
                fuel_empty = int(fields["fuel_empty"])
            except ValueError:
                continue
            if (not math.isfinite(speed) or low_speed not in (0, 1)
                    or fuel_empty not in (0, 1)):
                continue
            losses.append(fields)
    return losses


def write_loss_csv(losses: List[Dict[str, str]], out: Path) -> None:
    """Export bounded, evaluation-only loss evidence without changing PPO CSV."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOSS_FIELDS)
        writer.writeheader()
        for loss in losses:
            writer.writerow({key: loss.get(key, "") for key in LOSS_FIELDS})


def evaluation_loss_summary(
    losses: List[Dict[str, str]], episodes: List[Dict[str, str]]
) -> List[str]:
    """Differentiate low-speed/fuel evidence from external loss attribution."""
    if not losses:
        return ["No evaluation loss-transition diagnostics available."]
    campaign = episodes[-1]["campaign"] if episodes else losses[-1]["campaign"]
    latest = [loss for loss in losses if loss["campaign"] == campaign]
    if not latest:
        return [f"Campaign #{campaign}: no new Blue loss transitions recorded."]

    low_speed = sum(loss["low_speed"] == "1" for loss in latest)
    fuel_empty = sum(loss["fuel_empty"] == "1" for loss in latest)
    both = sum(loss["low_speed"] == "1" and loss["fuel_empty"] == "1"
               for loss in latest)
    unknown = [loss for loss in latest if loss["source"] == "unknown"]
    unknown_low_speed = sum(loss["low_speed"] == "1" for loss in unknown)
    unknown_fuel = sum(loss["fuel_empty"] == "1" for loss in unknown)
    unexplained = sum(loss["low_speed"] == "0"
                      and loss["fuel_empty"] == "0" for loss in unknown)
    return [
        f"Campaign #{campaign}: **{len(latest)}** newly destroyed Blues "
        "(one record per transition).",
        f"Flight-model condition met: **low_speed={low_speed}**, "
        f"**fuel_empty={fuel_empty}**, **both={both}**. "
        "These flags are sampled at destruction, not an exclusive causal label.",
        f"Of **{len(unknown)}** losses with unknown ECS source: "
        f"low_speed={unknown_low_speed}, fuel_empty={unknown_fuel}, "
        f"neither={unexplained}.",
        "Per-loss speed, fuel and acceleration command: mappo-losses.csv.",
    ]


def evaluation_summary(episodes: List[Dict[str, str]]) -> List[str]:
    """Describe the last complete/partial evaluated campaign separately."""
    if not episodes:
        return ["No per-episode evaluation diagnostics available."]
    last_campaign_id = episodes[-1]["campaign"]
    latest = [episode for episode in episodes
              if episode["campaign"] == last_campaign_id]
    terminals = Counter(episode["terminal"] for episode in latest)
    losses = Counter(value for episode in latest
                     for key, value in episode.items()
                     if key.startswith("blue_") and key[5:].isdigit()
                     and value != "alive")
    alive = sum(int(episode["blue_alive"]) for episode in latest)
    return [
        f"Last evaluation campaign: **#{last_campaign_id}**, "
        f"**{len(latest)} episodes**; final blue survivors: **{alive}** "
        "(sum across cases).",
        "Terminal causes: " + ", ".join(
            f"{name}={count}" for name, count in sorted(terminals.items())),
        "Blue loss sources: " + (
            ", ".join(f"{name}={count}" for name, count in sorted(losses.items()))
            if losses else "none"),
        "Unknown sources are reported explicitly; not assumed to be Red.",
    ]


def summary(rows: List[Dict[str, float]], run_url: str,
            eval_episodes: List[Dict[str, str]] | None = None,
            eval_losses: List[Dict[str, str]] | None = None) -> str:
    last = rows[-1] if rows else None
    updates = int(max((r["updates"] for r in rows), default=0))
    longest_stagnation = 0.0
    if rows:
        previous = (rows[0]["updates"], rows[0]["rollout_steps"],
                    rows[0]["evaluation_steps"])
        since = rows[0]["elapsed_seconds"]
        for row in rows[1:]:
            progress = (row["updates"], row["rollout_steps"],
                        row["evaluation_steps"])
            if progress != previous:
                longest_stagnation = max(
                    longest_stagnation, row["elapsed_seconds"] - since)
                previous, since = progress, row["elapsed_seconds"]
        longest_stagnation = max(
            longest_stagnation, rows[-1]["elapsed_seconds"] - since)
    lines = [
        "## MAPPO training telemetry",
        "",
        f"Samples: **{len(rows)}** | Completed PPO updates: **{updates}**",
    ]
    if last:
        lines.extend([
            f"Latest score: **{last['latest_score']:.2f}** | "
            f"Best: **{last['best_score']:.2f}** "
            f"(available after the first evaluation)",
            f"Elapsed sampled: **{last['elapsed_seconds']:.0f} s** | "
            f"Latest rollout: **{int(last['rollout_steps'])}/{int(last['rollout_target'])}**",
            f"Evaluation: **{'active' if last['evaluating'] else 'inactive'}** | "
            f"cases: **{int(last['evaluation_cases_done'])}/{int(last['evaluation_cases_total'])}**",
            f"Longest unchanged snapshot: **{longest_stagnation:.0f} s** "
            "(may include synchronous optimizer or checkpointing)",
            "",
            "PPO updates over sampled wall time:",
            "",
            "    " + sparkline([r["updates"] for r in rows]),
            "",
            "Evaluation completion over sampled wall time:",
            "",
            "    " + sparkline([r["evaluation_progress"] for r in rows]),
        ])
    else:
        lines.append("No live status was collected; inspect the headless startup log.")
    if eval_episodes is not None:
        lines.extend(["", "### Evaluation episode diagnostics", ""])
        lines.extend(evaluation_summary(eval_episodes))
    if eval_losses is not None:
        lines.extend(["", "### Blue flight-model loss diagnostics", ""])
        lines.extend(evaluation_loss_summary(
            eval_losses, eval_episodes or []))
    if run_url:
        lines.extend([
            "",
            f"[Download CSV and SVG plots from this run's artifacts]({run_url})",
        ])
    lines.extend([
        "",
        "These curves show progress, not training quality or convergence. "
        "Check MAPPO_TIMING logs for optimizer duration. "
        "Losses are not reported unless explicitly measured.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    write_chart(rows, args.out / "mappo-progress.svg")
    evaluation = read_eval_episodes(args.log) if args.log else None
    losses = read_eval_losses(args.log) if args.log else None
    if losses is not None:
        write_loss_csv(losses, args.out / "mappo-losses.csv")
    result = summary(rows, os.environ.get("MAPPO_RUN_URL", ""),
                     evaluation, losses)
    (args.out / "mappo-summary.md").write_text(result, encoding="utf-8")
    # GitHub's Windows runner can expose a CP1252 stdout even though the
    # summary includes Unicode sparkline blocks.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(result, flush=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("a", encoding="utf-8") as stream:
            stream.write(result)


if __name__ == "__main__":
    main()
