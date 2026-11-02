"""Deterministic diagnostics and comparable Blue MAPPO quality metrics."""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import DiagnosticAlert, RunAnalysis, RunDataset


def analyze(dataset: RunDataset) -> RunAnalysis:
    """Derive a complete diagnostic without claiming policy convergence."""
    samples = [row for segment in dataset.segments for row in segment.samples]
    durations = [
        segment.samples[-1]["elapsed_seconds"]
        for segment in dataset.segments
        if segment.samples
    ]
    updates = [row["updates"] for row in samples]
    final = samples[-1] if samples else {}
    invalid_rows = sum(segment.invalid_rows for segment in dataset.segments)
    start_update = int(min(updates)) if updates else 0
    end_update = int(max(updates)) if updates else 0
    completed = max(0, end_update - start_update)
    duration = sum(max(0.0, value) for value in durations)

    episode_evaluations = [
        row for row in dataset.evaluations if row.get("record_type") != "campaign_score"
    ]
    evaluation_campaigns = _evaluation_campaigns(episode_evaluations)
    latest_evaluation = evaluation_campaigns[-1] if evaluation_campaigns else {}
    latest_campaign = _latest_campaign(episode_evaluations)
    terminals = Counter(row.get("terminal", "unknown") for row in latest_campaign)
    loss_sources = Counter(row.get("source", "unknown") for row in dataset.losses)

    optimizer = dataset.optimizer_updates
    optimizer_timings = _optimizer_timings(dataset)
    patrol = _patrol_summary(dataset)
    tolerance_by_role = _tolerance_summary(dataset.patrol_agents)
    reward_by_role = _reward_summary(dataset.role_rewards)
    loss_diagnostics = _loss_summary(dataset.losses)
    status_diagnostics = _status_summary(dataset.status_snapshots, samples)
    reward_values = [row["instant_reward"] for row in samples]

    success_rate = _first_available(
        latest_evaluation.get("success_rate"), final.get("eval_success_rate"), 0.0
    )
    blue_survival = _first_available(
        latest_evaluation.get("blue_survival_rate"),
        _latest_blue_survival(latest_campaign),
        0.0,
    )
    metrics = {
        "samples": len(samples),
        "segments": len(dataset.segments),
        "invalid_rows": invalid_rows,
        "start_update": start_update,
        "end_update": end_update,
        "completed_updates": completed,
        "training_seconds": duration,
        "updates_per_hour": completed * 3600.0 / duration if duration > 0 else 0.0,
        "latest_score": float(final.get("latest_score", 0.0)),
        "best_score": max((row["best_score"] for row in samples), default=0.0),
        "success_rate": success_rate,
        "reward_mean": _mean(reward_values),
        "reward_volatility": _population_std(reward_values),
        "reward_recent_mean": _recent_mean(reward_values),
        "reward_distribution": _distribution(reward_values),
        "longest_stagnation_seconds": _longest_stagnation(dataset),
        "evaluation_episodes": len(latest_campaign),
        "evaluation_campaigns": evaluation_campaigns,
        "latest_evaluation": latest_evaluation,
        "blue_survival_rate": blue_survival,
        "red_neutralization_rate": latest_evaluation.get("red_neutralization_rate"),
        "objective_survival_rate": latest_evaluation.get("objective_survival_rate"),
        "leader_survival_rate": latest_evaluation.get("leader_survival_rate"),
        "evaluation_steps_mean": latest_evaluation.get("steps_mean"),
        "terminals": terminals,
        "loss_sources": loss_sources,
        "loss_diagnostics": loss_diagnostics,
        "optimizer_updates": len(optimizer),
        "optimizer_ms_mean": _distribution_field(optimizer_timings, "total_ms", "mean"),
        "optimizer_ms_p95": _distribution_field(optimizer_timings, "total_ms", "p95"),
        "optimizer_timing": _optimizer_summary(optimizer_timings),
        "policy_loss_latest": _last_field(optimizer, "policy_loss"),
        "value_loss_latest": _last_field(optimizer, "value_loss"),
        "entropy_latest": _last_field(optimizer, "entropy"),
        "approx_kl_latest": _last_field(optimizer, "approx_kl"),
        "clip_fraction_latest": _last_field(optimizer, "clip_fraction"),
        "explained_variance_latest": _last_field(optimizer, "explained_variance"),
        "action_std_latest": _last_field(optimizer, "action_std_mean"),
        "status_diagnostics": status_diagnostics,
        "patrol": patrol,
        "tolerance_by_role": tolerance_by_role,
        "reward_by_role": reward_by_role,
        "checkpoint_count": len(dataset.checkpoint_files),
        "checkpoint_bytes": sum(
            path.stat().st_size for path in dataset.checkpoint_files if path.is_file()
        ),
    }
    alerts = _alerts(dataset, metrics, samples)
    return RunAnalysis(
        dataset,
        metrics,
        alerts,
        _score_points(dataset, samples),
        _series_points(dataset, "instant_reward", smooth=True),
        _series_points(dataset, "updates", smooth=False),
    )


def compatibility(left: RunAnalysis, right: RunAnalysis) -> Tuple[bool, List[str]]:
    """Check whether quality indicators have comparable evaluation contexts."""
    reasons = []
    for key, label in (
        ("scenario", "scénario"),
        ("curriculum_version", "version du curriculum"),
        ("curriculum_enabled", "mode curriculum"),
    ):
        left_value = _manifest_value(left.dataset.manifest, key)
        right_value = _manifest_value(right.dataset.manifest, key)
        if left_value is not None and right_value is not None and left_value != right_value:
            reasons.append("{} différent ({} / {})".format(label, left_value, right_value))
    left_context = _evaluation_context(left.dataset)
    right_context = _evaluation_context(right.dataset)
    if left_context and right_context and left_context != right_context:
        reasons.append(
            "dernière évaluation P/N différente ({} / {})".format(
                left_context, right_context
            )
        )
    if not left.score_points or not right.score_points:
        reasons.append("évaluation chiffrée absente dans au moins un run")
    return not reasons, reasons


def paired_evaluation_comparison(
    left: RunAnalysis, right: RunAnalysis
) -> Dict[str, Any]:
    """Compare the newest observations sharing the same P/N, seed and case."""
    left_rows = _latest_evaluation_cases(left.dataset.evaluations)
    right_rows = _latest_evaluation_cases(right.dataset.evaluations)
    common = sorted(set(left_rows).intersection(right_rows))
    pairs = []
    for key in common:
        left_row = left_rows[key]
        right_row = right_rows[key]
        pairs.append(
            {
                "P": key[0],
                "N": key[1],
                "seed": key[2],
                "case": key[3],
                "left_terminal": left_row.get("terminal", "unknown"),
                "right_terminal": right_row.get("terminal", "unknown"),
                "left_blue_alive": _safe_int(left_row.get("blue_alive")),
                "right_blue_alive": _safe_int(right_row.get("blue_alive")),
                "left_red_alive": _safe_int(left_row.get("red_alive")),
                "right_red_alive": _safe_int(right_row.get("red_alive")),
                "left_steps": _safe_int(left_row.get("steps")),
                "right_steps": _safe_int(right_row.get("steps")),
                "left_objective": _safe_int(left_row.get("objective_alive")),
                "right_objective": _safe_int(right_row.get("objective_alive")),
            }
        )
    return {
        "count": len(pairs),
        "pairs": pairs,
        "blue_alive_delta_mean": _mean(
            [row["right_blue_alive"] - row["left_blue_alive"] for row in pairs]
        ),
        "red_alive_delta_mean": _mean(
            [row["right_red_alive"] - row["left_red_alive"] for row in pairs]
        ),
        "steps_delta_mean": _mean(
            [row["right_steps"] - row["left_steps"] for row in pairs]
        ),
        "terminal_changes": sum(
            row["left_terminal"] != row["right_terminal"] for row in pairs
        ),
    }


def observed_advantage(left: RunAnalysis, right: RunAnalysis) -> str:
    """Describe the observed quality leader only for compatible evidence."""
    comparable, reasons = compatibility(left, right)
    if not comparable:
        return "Comparaison descriptive uniquement : {}.".format("; ".join(reasons))
    left_quality = _quality_tuple(left)
    right_quality = _quality_tuple(right)
    if left_quality == right_quality:
        return "Aucun avantage observé sur les indicateurs d’évaluation disponibles."
    winner = left if left_quality > right_quality else right
    return (
        "Avantage observé pour le run {} sur succès, survie de l’objectif, "
        "neutralisation Red puis survie Blue. Ce constat ne prouve pas la "
        "convergence ni la robustesse multi-seeds."
    ).format(_run_label(winner))


def _quality_tuple(analysis: RunAnalysis) -> tuple:
    metrics = analysis.metrics
    return tuple(
        -1.0 if value is None else float(value)
        for value in (
            metrics.get("success_rate"),
            metrics.get("objective_survival_rate"),
            metrics.get("red_neutralization_rate"),
            metrics.get("blue_survival_rate"),
            metrics.get("latest_score"),
        )
    )


def _evaluation_campaigns(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        blue_value = row.get("P", row.get("blue"))
        blue_count = _safe_int(blue_value) if blue_value is not None else _episode_blue_count(row)
        key = (
            _safe_int(row.get("update")),
            _safe_int(row.get("campaign")),
            blue_count,
            _safe_int(row.get("N", row.get("red"))),
        )
        grouped[key].append(row)
    summaries = []
    for key, episodes in sorted(grouped.items()):
        update, campaign, blue_count, red_count = key
        total = len(episodes)
        blue_possible = blue_count * total
        red_possible = red_count * total
        steps = [_safe_float(row.get("steps")) for row in episodes]
        summaries.append(
            {
                "update": update,
                "campaign": campaign,
                "P": blue_count,
                "N": red_count,
                "episodes": total,
                "success_rate": sum(
                    row.get("terminal") == "reds_cleared" for row in episodes
                ) / float(total),
                "timeout_rate": sum(
                    row.get("terminal") == "timeout" for row in episodes
                ) / float(total),
                "blue_survival_rate": (
                    sum(_safe_int(row.get("blue_alive")) for row in episodes)
                    / float(blue_possible)
                    if blue_possible else 0.0
                ),
                "red_neutralization_rate": (
                    1.0
                    - sum(_safe_int(row.get("red_alive")) for row in episodes)
                    / float(red_possible)
                    if red_possible else None
                ),
                "objective_survival_rate": _mean(
                    [_safe_float(row.get("objective_alive")) for row in episodes]
                ),
                "leader_survival_rate": _mean(
                    [_safe_float(row.get("leader_alive")) for row in episodes]
                ),
                "steps_mean": _mean(steps),
                "steps_p95": _percentile(steps, 0.95),
                "terminals": Counter(
                    row.get("terminal", "unknown") for row in episodes
                ),
            }
        )
    return summaries


def _latest_evaluation_cases(rows: Sequence[Dict[str, str]]) -> Dict[tuple, Dict[str, str]]:
    result = {}
    rank = {}
    for row in rows:
        if row.get("record_type") == "campaign_score" or "case" not in row:
            continue
        key = (
            _safe_int(row.get("P", row.get("blue"))),
            _safe_int(row.get("N", row.get("red"))),
            _safe_int(row.get("seed")),
            _safe_int(row.get("case")),
        )
        current_rank = (_safe_int(row.get("update")), _safe_int(row.get("campaign")))
        if key not in rank or current_rank >= rank[key]:
            rank[key] = current_rank
            result[key] = row
    return result


def _patrol_summary(dataset: RunDataset) -> Dict[str, Any]:
    episodes = dataset.patrol_episodes
    circuit = _numeric_values(episodes, "circuit_m")
    qualified = _numeric_values(episodes, "qualified_m")
    spacing = [
        _safe_float(row.get("spacing_margin_m"))
        for row in episodes
        if _safe_int(row.get("spacing_applicable")) != 0
    ]
    resets = [_safe_float(row.get("resets")) for row in episodes]
    violations = [_safe_float(row.get("tracking_violations")) for row in episodes]
    trend = []
    for index, row in enumerate(episodes, 1):
        circuit_value = _safe_float(row.get("circuit_m"))
        qualified_value = _safe_float(row.get("qualified_m"))
        trend.append(
            {
                "episode": index,
                "phase": row.get("phase", "unknown"),
                "P": _safe_int(row.get("P")),
                "N": _safe_int(row.get("N")),
                "qualified_ratio": (
                    qualified_value / circuit_value if circuit_value > 0 else 0.0
                ),
                "spacing_margin_m": _safe_float(row.get("spacing_margin_m")),
                "resets": _safe_int(row.get("resets")),
                "violations": _safe_int(row.get("tracking_violations")),
            }
        )
    return {
        "episodes": len(episodes),
        "phases": Counter(row.get("phase", "unknown") for row in episodes),
        "modes": Counter(row.get("mode", "unknown") for row in episodes),
        "terminations": Counter(
            row.get("termination", "unknown") for row in episodes
        ),
        "latest_mode": episodes[-1].get("mode", "unknown") if episodes else "unknown",
        "validated_waypoints_max": max(
            (_safe_int(row.get("validated_waypoints")) for row in episodes),
            default=0,
        ),
        "active_waypoint_max": max(
            (_safe_int(row.get("active_waypoint")) for row in episodes),
            default=0,
        ),
        "waypoint_distance": _distribution(
            _numeric_values(episodes, "waypoint_distance_m")
        ),
        "cross_track": _distribution(_numeric_values(episodes, "cross_track_m")),
        "heading_error": _distribution(
            _numeric_values(episodes, "heading_error_rad")
        ),
        "speed": _distribution(_numeric_values(episodes, "speed_mps")),
        "arclength": _distribution(_numeric_values(episodes, "arclength_m")),
        "distance_m": sum(_numeric_values(episodes, "distance_m")),
        "qualified_m": sum(qualified),
        "circuit_m": sum(circuit),
        "qualified_ratio": sum(qualified) / sum(circuit) if sum(circuit) > 0 else 0.0,
        "resets": int(sum(resets)),
        "episodes_with_resets": sum(value > 0 for value in resets),
        "tracking_violations": int(sum(violations)),
        "episodes_with_violations": sum(value > 0 for value in violations),
        "spacing_margin": _distribution(spacing),
        "steps": _distribution(_numeric_values(episodes, "steps")),
        "trend": trend,
    }


def _tolerance_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    fields = (
        ("position_tolerances", "position"),
        ("speed_tolerances", "speed"),
        ("heading_tolerances", "heading"),
    )
    roles = sorted(set(row.get("role", "unknown") for row in rows))
    by_role = {}
    for role in roles:
        role_rows = [row for row in rows if row.get("role", "unknown") == role]
        by_role[role] = {
            label: _distribution(_numeric_values(role_rows, field))
            for field, label in fields
        }
        by_role[role]["samples"] = len(role_rows)
        by_role[role]["valid_rate"] = (
            sum(_safe_int(row.get("valid")) != 0 for row in role_rows)
            / float(len(role_rows))
            if role_rows else 0.0
        )
    return {
        "samples": len(rows),
        "valid_rate": (
            sum(_safe_int(row.get("valid")) != 0 for row in rows) / float(len(rows))
            if rows else 0.0
        ),
        "roles": by_role,
    }


def _reward_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        role = str(row.get("role", "unknown"))
        channels = row.get("channels")
        if not isinstance(channels, dict):
            continue
        for channel, value in channels.items():
            number = _optional_float(value)
            if number is not None:
                grouped[(role, str(channel))].append(number)
    by_role = defaultdict(dict)
    for (role, channel), values in sorted(grouped.items()):
        by_role[role][channel] = _distribution(values)
    return {"samples": len(rows), "roles": dict(by_role)}


def _loss_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    speed_delta = []
    fuel_fraction = []
    accelerations = []
    by_blue = Counter()
    by_update = Counter()
    by_seed = Counter()
    low_speed = 0
    fuel_empty = 0
    for row in rows:
        source = row.get("source", "unknown")
        by_blue[(row.get("blue", "unknown"), source)] += 1
        by_update[(_safe_int(row.get("update")), source)] += 1
        by_seed[(_safe_int(row.get("seed")), source)] += 1
        low_speed += _safe_int(row.get("low_speed")) != 0
        fuel_empty += _safe_int(row.get("fuel_empty")) != 0
        before = _optional_float(row.get("speed_before"))
        after = _optional_float(row.get("speed"))
        if before is not None and after is not None:
            speed_delta.append(after - before)
        acceleration = _optional_float(row.get("accel_cmd"))
        if acceleration is not None:
            accelerations.append(abs(acceleration))
        remaining = _optional_float(row.get("fuel_remaining"))
        capacity = _optional_float(row.get("fuel_capacity"))
        if remaining is not None and capacity is not None and capacity > 0:
            fuel_fraction.append(remaining / capacity)
    return {
        "count": len(rows),
        "low_speed": low_speed,
        "fuel_empty": fuel_empty,
        "by_blue": by_blue,
        "by_update": by_update,
        "by_seed": by_seed,
        "speed_delta": _distribution(speed_delta),
        "fuel_fraction": _distribution(fuel_fraction),
        "abs_accel": _distribution(accelerations),
    }


def _optimizer_timings(dataset: RunDataset) -> List[Dict[str, float]]:
    timings = []
    for row in dataset.optimizer_updates:
        converted = {
            field: float(row[field])
            for field in (
                "update", "total_ms", "critic_baselines_ms", "advantages_ms", "ppo_ms"
            )
            if field in row
        }
        if "total_ms" in converted:
            timings.append(converted)
    if timings:
        return timings
    for event in dataset.timing_events:
        if str(event.get("optimizer", "")) != "done":
            continue
        total = _optional_float(event.get("total_ms", event.get("elapsed_ms")))
        if total is None:
            continue
        row = {"total_ms": total, "update": _safe_float(event.get("update"))}
        for field in ("critic_baselines_ms", "advantages_ms", "ppo_ms"):
            value = _optional_float(event.get(field))
            if value is not None:
                row[field] = value
        timings.append(row)
    return timings


def _optimizer_summary(rows: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    fields = (
        ("total_ms", "total"),
        ("critic_baselines_ms", "critic"),
        ("advantages_ms", "advantages"),
        ("ppo_ms", "ppo"),
    )
    return {
        label: _distribution(_numeric_values(rows, field))
        for field, label in fields
    }


def _status_summary(
    statuses: Sequence[Dict[str, Any]], samples: Sequence[Dict[str, float]]
) -> Dict[str, Any]:
    speed = _numeric_values(statuses, "eval_steps_per_second")
    if not speed:
        speed = [row["evaluation_steps_per_second"] for row in samples]
    evaluating = [row for row in samples if row.get("evaluating", 0.0) > 0]
    return {
        "snapshots": len(statuses),
        "evaluation_speed": _distribution(speed),
        "evaluating_sample_rate": len(evaluating) / float(len(samples)) if samples else 0.0,
        "maximum_rollout_progress": max(
            (
                _safe_float(row.get("rollout_steps"))
                / _safe_float(row.get("rollout_target"))
                for row in statuses
                if _safe_float(row.get("rollout_target")) > 0
            ),
            default=0.0,
        ),
    }


def _alerts(
    dataset: RunDataset, metrics: Dict[str, Any], samples: Sequence[Dict[str, float]]
) -> List[DiagnosticAlert]:
    alerts = []
    if dataset.log_errors:
        alerts.append(DiagnosticAlert(
            "critical", "Erreurs MAPPO détectées",
            "{} ligne(s) d’erreur explicite dans les journaux.".format(len(dataset.log_errors)),
        ))
    if not samples:
        alerts.append(DiagnosticAlert(
            "critical", "Télémétrie CSV absente",
            "Aucun échantillon CSV exploitable n’a été trouvé.",
        ))
    if metrics["invalid_rows"]:
        alerts.append(DiagnosticAlert(
            "warning", "Échantillons rejetés",
            "{} ligne(s) incomplète(s), non finie(s) ou incompatibles.".format(metrics["invalid_rows"]),
        ))
    if samples and metrics["completed_updates"] <= 0:
        alerts.append(DiagnosticAlert(
            "critical", "Aucune progression PPO",
            "Le compteur d’updates n’a pas progressé dans la télémétrie disponible.",
        ))
    if not metrics["evaluation_campaigns"]:
        alerts.append(DiagnosticAlert(
            "warning", "Évaluations détaillées absentes",
            "La qualité par seed, cas et composition P/N ne peut pas être contrôlée.",
        ))
    if metrics["longest_stagnation_seconds"] >= 300.0:
        alerts.append(DiagnosticAlert(
            "warning", "Longue période sans progression visible",
            "Le plus long instantané inchangé dure {:.0f} s ; vérifier les temps d’optimisation et de sauvegarde.".format(metrics["longest_stagnation_seconds"]),
        ))
    kl = metrics.get("approx_kl_latest")
    if kl is not None and kl > 0.05:
        alerts.append(DiagnosticAlert(
            "warning", "Mise à jour de politique importante",
            "KL approximative {:.4f} (> 0,05, seuil heuristique).".format(kl),
        ))
    clip = metrics.get("clip_fraction_latest")
    if clip is not None and clip > 0.5:
        alerts.append(DiagnosticAlert(
            "warning", "Clipping PPO très fréquent",
            "Fraction clippée {:.1%} (> 50 %, seuil heuristique).".format(clip),
        ))
    explained = metrics.get("explained_variance_latest")
    if explained is not None and explained < 0.0:
        alerts.append(DiagnosticAlert(
            "warning", "Critic moins informatif qu’une constante",
            "Variance expliquée {:.3f} ; examiner les cibles de valeur et leur échelle.".format(explained),
        ))
    tolerance = metrics["tolerance_by_role"]
    if tolerance["samples"] and tolerance["valid_rate"] < 1.0:
        alerts.append(DiagnosticAlert(
            "warning", "Mesures de patrouille invalides",
            "{:.1%} des diagnostics par agent sont valides.".format(tolerance["valid_rate"]),
        ))
    patrol = metrics["patrol"]
    if patrol["tracking_violations"]:
        alerts.append(DiagnosticAlert(
            "warning", "Violations de suivi de patrouille",
            "{} violation(s) répartie(s) sur {} épisode(s).".format(
                patrol["tracking_violations"], patrol["episodes_with_violations"]
            ),
        ))
    if not dataset.checkpoint_files:
        alerts.append(DiagnosticAlert(
            "warning", "Checkpoint absent",
            "Aucun acteur ou état d’entraînement Blue n’est présent localement.",
        ))
    if not alerts:
        alerts.append(DiagnosticAlert(
            "ok", "Aucune anomalie déterministe détectée",
            "Les contrôles de structure, progression et valeurs finies sont satisfaits.",
        ))
    return alerts


def _score_points(dataset: RunDataset, samples: Sequence[Dict[str, float]]) -> List[tuple]:
    campaign_scores = []
    for row in dataset.evaluations:
        if row.get("record_type") != "campaign_score":
            continue
        try:
            campaign_scores.append((float(row["campaign"]), float(row["score"])))
        except (KeyError, ValueError):
            continue
    if campaign_scores:
        return campaign_scores
    points = []
    previous = None
    for row in samples:
        if row["eval_completed"] <= 0:
            continue
        key = (row["updates"], row["latest_score"])
        if previous is None or row["latest_score"] != previous:
            points.append(key)
            previous = row["latest_score"]
    return points


def _series_points(dataset: RunDataset, field: str, smooth: bool) -> List[tuple]:
    points = []
    offset = 0.0
    values = []
    for segment in dataset.segments:
        for row in segment.samples:
            value = row[field]
            values.append(value)
            plotted = _ewma(values, 0.08) if smooth else value
            points.append((offset + row["elapsed_seconds"], plotted))
        if segment.samples:
            offset += max(0.0, segment.samples[-1]["elapsed_seconds"])
    return points


def _latest_campaign(evaluations: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    if not evaluations:
        return []
    latest_rank = max(
        (_safe_int(row.get("update")), _safe_int(row.get("campaign")))
        for row in evaluations
    )
    return [
        row for row in evaluations
        if (_safe_int(row.get("update")), _safe_int(row.get("campaign"))) == latest_rank
    ]


def _latest_blue_survival(rows: Sequence[Dict[str, str]]) -> Optional[float]:
    possible = sum(_episode_blue_count(row) for row in rows)
    if not possible:
        return None
    return sum(_safe_int(row.get("blue_alive")) for row in rows) / float(possible)


def _episode_blue_count(row: Dict[str, str]) -> int:
    explicit = [key for key in row if key.startswith("blue_") and key[5:].isdigit()]
    return len(explicit) if explicit else max(0, _safe_int(row.get("P", row.get("blue_alive"))))


def _longest_stagnation(dataset: RunDataset) -> float:
    longest = 0.0
    for segment in dataset.segments:
        if not segment.samples:
            continue
        previous = _progress_tuple(segment.samples[0])
        since = segment.samples[0]["elapsed_seconds"]
        for row in segment.samples[1:]:
            progress = _progress_tuple(row)
            if progress != previous:
                longest = max(longest, row["elapsed_seconds"] - since)
                previous = progress
                since = row["elapsed_seconds"]
        longest = max(longest, segment.samples[-1]["elapsed_seconds"] - since)
    return max(0.0, longest)


def _progress_tuple(row: Dict[str, float]) -> tuple:
    return row["updates"], row["rollout_steps"], row["evaluation_steps"]


def _evaluation_context(dataset: RunDataset) -> str:
    contexts = []
    for row in dataset.evaluations:
        blue = row.get("P", row.get("blue"))
        red = row.get("N", row.get("red"))
        if blue is not None and red is not None:
            contexts.append("{}/{}".format(blue, red))
    return contexts[-1] if contexts else ""


def _manifest_value(manifest: Dict[str, Any], key: str):
    if key in manifest:
        return manifest.get(key)
    inputs = manifest.get("inputs")
    return inputs.get(key) if isinstance(inputs, dict) else None


def _run_label(analysis: RunAnalysis) -> str:
    metadata = analysis.dataset.metadata
    return str(metadata.get("run_number") or metadata.get("id") or analysis.dataset.root.name)


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p05": None,
                "p95": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": _mean(finite),
        "median": statistics.median(finite),
        "p05": _percentile(finite, 0.05),
        "p95": _percentile(finite, 0.95),
        "min": min(finite),
        "max": max(finite),
    }


def _numeric_values(rows: Iterable[Dict[str, Any]], field: str) -> List[float]:
    values = []
    for row in rows:
        value = _optional_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _distribution_field(
    rows: Sequence[Dict[str, float]], field: str, statistic: str
) -> float:
    value = _distribution(_numeric_values(rows, field)).get(statistic)
    return 0.0 if value is None else float(value)


def _first_available(*values):
    for value in values:
        if value is not None:
            return float(value)
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _population_std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _recent_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    count = max(1, len(values) // 4)
    return _mean(values[-count:])


def _last_field(rows: Sequence[Dict[str, float]], field: str) -> Optional[float]:
    for row in reversed(rows):
        if field in row:
            return row[field]
    return None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def _ewma(values: Sequence[float], alpha: float) -> float:
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _safe_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    result = _optional_float(value)
    return result if result is not None else 0.0


def _optional_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
