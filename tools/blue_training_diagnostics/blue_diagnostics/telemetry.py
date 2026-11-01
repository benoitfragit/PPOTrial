"""Parsers for current and legacy Blue MAPPO CI artifacts."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .models import RunDataset, TelemetrySegment


REQUIRED_TELEMETRY_FIELDS = (
    "elapsed_seconds",
    "updates",
    "rollout_steps",
    "rollout_target",
    "evaluating",
    "evaluation_progress",
    "evaluation_cases_done",
    "evaluation_cases_total",
    "evaluation_steps",
    "evaluation_steps_per_second",
    "training",
    "eval_completed",
    "latest_score",
    "best_score",
    "eval_success_rate",
    "instant_reward",
)

UPDATE_MARKER = "[MAPPO_UPDATE]"
EVALUATION_MARKER = "[MAPPO_EVAL_EPISODE]"
LOSS_MARKER = "[MAPPO_EVAL_LOSS]"
TIMING_MARKER = "[MAPPO_TIMING]"
STATUS_MARKER = "[MAPPO_STATUS]"
PATROL_EPISODE_MARKER = "[MAPPO_PATROL_EPISODE]"
PATROL_AGENT_MARKER = "[MAPPO_PATROL_AGENT]"
ROLE_REWARD_MARKER = "[MAPPO_ROLE_REWARD]"
ERROR_MARKERS = ("[MAPPO_ERROR]", "[MAPPO_EVAL_ERROR]", "[CURRICULUM_ERROR]")
EVALUATION_SCORE = re.compile(
    r"Evaluation\s+#(?P<campaign>\d+)\s+P=(?P<blue>\d+)\s+"
    r"N=(?P<red>\d+)\s+score=(?P<score>[0-9.eE+\-]+)"
)


def load_run(run_root: Path) -> RunDataset:
    """Discover and parse all diagnostic evidence below a local run folder."""
    run_root = run_root.resolve()
    metadata = _read_first_json(run_root, "run.json")
    manifest = _read_first_json(run_root, "run-manifest.json")
    curriculum = _read_first_json(run_root, "curriculum-progress.json")
    segments = _discover_segments(run_root)
    stage_logs = sorted(run_root.rglob("training.log"))
    logs = stage_logs or sorted(run_root.rglob("mappo-training.log"))
    parsed = _parse_logs(logs)
    checkpoints = sorted(
        path
        for path in run_root.rglob("blue_patrol_generalized_*.txt")
        if path.is_file()
    )
    return RunDataset(
        root=run_root,
        metadata=metadata,
        manifest=manifest,
        segments=segments,
        optimizer_updates=parsed["optimizer"],
        timing_events=parsed["timings"],
        status_snapshots=parsed["statuses"],
        evaluations=parsed["evaluations"],
        losses=parsed["losses"],
        patrol_episodes=parsed["patrol_episodes"],
        patrol_agents=parsed["patrol_agents"],
        role_rewards=parsed["role_rewards"],
        log_errors=parsed["errors"],
        checkpoint_files=checkpoints,
        curriculum_progress=curriculum,
    )


def _read_first_json(root: Path, filename: str) -> Dict[str, object]:
    paths = sorted(root.rglob(filename))
    if not paths:
        return {}
    try:
        document = json.loads(paths[0].read_text(encoding="utf-8-sig"))
        return document if isinstance(document, dict) else {}
    except (OSError, ValueError):
        return {}


def _discover_segments(root: Path) -> List[TelemetrySegment]:
    curriculum_csv = sorted(root.rglob("telemetry.csv"))
    candidates = curriculum_csv or sorted(root.rglob("mappo-training.csv"))
    segments = []
    for path in candidates:
        log_path = path.with_name("training.log")
        if not log_path.is_file():
            log_path = _nearest_log(path, root)
        samples, invalid = read_telemetry(path)
        segments.append(
            TelemetrySegment(
                label=_segment_label(path, root),
                csv_path=path,
                log_path=log_path,
                samples=samples,
                invalid_rows=invalid,
            )
        )
    return segments


def _nearest_log(path: Path, root: Path) -> Optional[Path]:
    current = path.parent
    while current != root.parent:
        candidate = current / "mappo-training.log"
        if candidate.is_file():
            return candidate
        if current == root:
            break
        current = current.parent
    return None


def _segment_label(path: Path, root: Path) -> str:
    try:
        relative = path.parent.relative_to(root)
    except ValueError:
        return path.parent.name or "entraînement"
    parts = [part for part in relative.parts if part not in ("resources", "model")]
    if len(parts) >= 2 and parts[-1] == "telemetry":
        return "entraînement principal"
    if "curriculum" in parts:
        position = parts.index("curriculum")
        return " / ".join(parts[position + 1 :]) or "curriculum"
    for index, part in enumerate(parts):
        if part.startswith("curriculum-v"):
            return " / ".join(parts[index + 1 :]) or part
    return " / ".join(parts[-2:]) or "entraînement principal"


def read_telemetry(path: Path) -> Tuple[List[Dict[str, float]], int]:
    """Read finite CSV rows and count malformed or partial records."""
    rows = []
    invalid = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if not set(REQUIRED_TELEMETRY_FIELDS).issubset(fields):
                return [], 1
            for raw in reader:
                try:
                    row = {
                        key: float(raw[key]) for key in REQUIRED_TELEMETRY_FIELDS
                    }
                except (KeyError, TypeError, ValueError):
                    invalid += 1
                    continue
                if not all(math.isfinite(value) for value in row.values()):
                    invalid += 1
                    continue
                rows.append(row)
    except OSError:
        return [], 1
    return rows, invalid


def _parse_logs(paths: Iterable[Path]):
    optimizer_by_update = {}
    timings = []
    statuses = []
    evaluations = []
    losses = []
    patrol_episodes = []
    patrol_agents = []
    role_rewards = []
    errors = []
    score_records = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        label = str(path)
        for line in lines:
            if UPDATE_MARKER in line:
                values = _numeric_fields(line.split(UPDATE_MARKER, 1)[1])
                if "update" in values:
                    optimizer_by_update[int(values["update"])] = values
            if TIMING_MARKER in line:
                event = _mixed_fields(line.split(TIMING_MARKER, 1)[1])
                event["_log"] = label
                timings.append(event)
            if STATUS_MARKER in line:
                event = _mixed_fields(line.split(STATUS_MARKER, 1)[1])
                _split_ratio(event, "rollout", "rollout_steps", "rollout_target")
                _split_ratio(event, "eval_cases", "evaluation_cases_done", "evaluation_cases_total")
                event["_log"] = label
                statuses.append(event)
            if EVALUATION_MARKER in line:
                fields = _text_fields(line.split(EVALUATION_MARKER, 1)[1])
                if all(key in fields for key in ("campaign", "case", "blue_alive", "terminal")):
                    fields["_log"] = label
                    evaluations.append(fields)
            if LOSS_MARKER in line:
                fields = _text_fields(line.split(LOSS_MARKER, 1)[1])
                if all(key in fields for key in ("campaign", "case", "blue", "source")):
                    fields["_log"] = label
                    losses.append(fields)
            if PATROL_EPISODE_MARKER in line:
                fields = _text_fields(line.split(PATROL_EPISODE_MARKER, 1)[1])
                if all(key in fields for key in ("phase", "P", "N", "steps")):
                    fields["_log"] = label
                    patrol_episodes.append(fields)
            if PATROL_AGENT_MARKER in line:
                fields = _text_fields(line.split(PATROL_AGENT_MARKER, 1)[1])
                if all(key in fields for key in ("blue", "role", "valid")):
                    fields["_log"] = label
                    patrol_agents.append(fields)
            if ROLE_REWARD_MARKER in line:
                reward = _role_reward_fields(line.split(ROLE_REWARD_MARKER, 1)[1])
                if reward:
                    reward["_log"] = label
                    role_rewards.append(reward)
            if any(marker in line for marker in ERROR_MARKERS):
                errors.append(line.strip()[:1000])
            match = EVALUATION_SCORE.search(line)
            if match:
                record = dict(match.groupdict())
                record["_log"] = label
                score_records.append(record)
    # Preserve score-only evaluations as explicit records without mixing them
    # with per-episode records used for survival statistics.
    evaluations.extend(
        dict(record, record_type="campaign_score") for record in score_records
    )
    return {
        "optimizer": [optimizer_by_update[key] for key in sorted(optimizer_by_update)],
        "timings": timings,
        "statuses": statuses,
        "evaluations": evaluations,
        "losses": losses,
        "patrol_episodes": patrol_episodes,
        "patrol_agents": patrol_agents,
        "role_rewards": role_rewards,
        "errors": errors,
    }


def _text_fields(payload: str) -> Dict[str, str]:
    return dict(
        token.split("=", 1) for token in payload.split() if "=" in token
    )


def _numeric_fields(payload: str) -> Dict[str, float]:
    result = {}
    for key, value in _text_fields(payload).items():
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            result[key] = number
    return result


def _mixed_fields(payload: str) -> Dict[str, object]:
    result = {}
    for key, value in _text_fields(payload).items():
        try:
            number = float(value)
        except ValueError:
            result[key] = value
        else:
            result[key] = number if math.isfinite(number) else value
    return result


def _split_ratio(
    fields: Dict[str, object], source: str, numerator: str, denominator: str
) -> None:
    value = fields.get(source)
    if not isinstance(value, str) or "/" not in value:
        return
    left, right = value.split("/", 1)
    try:
        fields[numerator] = float(left)
        fields[denominator] = float(right)
    except ValueError:
        return


def _role_reward_fields(payload: str) -> Dict[str, object]:
    prefix = payload.split("[", 1)[0]
    result = _mixed_fields(prefix)
    channels = {}
    for name, value in re.findall(r"\[([^\]]+)\]=([^\s]+)", payload):
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            channels[name] = number
    if "role" not in result or "agent_steps" not in result or not channels:
        return {}
    result["channels"] = channels
    return result

