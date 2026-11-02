"""Self-contained HTML/SVG reports for Blue MAPPO diagnostics."""
from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .analysis import compatibility, observed_advantage, paired_evaluation_comparison
from .models import DiagnosticAlert, RunAnalysis


COLORS = ("#38bdf8", "#a78bfa", "#34d399", "#fb7185", "#fbbf24", "#22d3ee")


def write_single_report(analysis: RunAnalysis, destination: Path) -> Path:
    """Write a polished standalone report using every telemetry stream."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = _run_label(analysis)
    metrics = analysis.metrics
    optimizer = analysis.dataset.optimizer_updates
    training_charts = [
        _chart(
            [("Score d’évaluation", analysis.score_points)],
            "Qualité observée en évaluation",
            "Campagne (ou update selon les anciens artefacts)",
            "Score",
            y_bounds=(0.0, 100.0),
        ),
        _chart(
            [("Récompense lissée", analysis.reward_points)],
            "Récompense instantanée — moyenne exponentielle",
            "Temps d’entraînement cumulé (s)",
            "Récompense",
        ),
        _chart(
            [("Updates", analysis.update_points)],
            "Progression de l’optimisation",
            "Temps d’entraînement cumulé (s)",
            "Update PPO",
        ),
    ]
    optimizer_loss = []
    for field, title in (("policy_loss", "Policy loss"), ("value_loss", "Value loss")):
        points = _optimizer_points(optimizer, field)
        if points:
            optimizer_loss.append((title, points))
    if optimizer_loss:
        training_charts.append(
            _chart(
                optimizer_loss,
                "Pertes PPO par update",
                "Update PPO",
                "Loss",
            )
        )
    optimizer_health = []
    for field, title in (
        ("approx_kl", "KL approx."),
        ("clip_fraction", "Fraction clippée"),
        ("explained_variance", "Variance expliquée"),
    ):
        points = _optimizer_points(optimizer, field)
        if points:
            optimizer_health.append((title, points))
    if optimizer_health:
        training_charts.append(
            _chart(
                optimizer_health,
                "Santé numérique de l’optimisation",
                "Update PPO",
                "Valeur",
            )
        )
    body = """
    <header class="hero">
      <div><span class="eyebrow">PPOTrial · Blue MAPPO</span>
      <h1>Diagnostic du run {label}</h1>
      <p>{subtitle}</p></div>
      <div class="status {status_class}">{status}</div>
    </header>
    {cards}
    <section><h2>Constats prioritaires</h2>{alerts}</section>
    <section><div class="section-head"><div><span class="section-index">01</span><h2>Qualité en évaluation</h2></div>
      <p>Succès combat, survivants, objectif et durée, regroupés par campagne et composition P/N.</p></div>
      <div class="charts">{evaluation_charts}</div>{evaluation}</section>
    <section><div class="section-head"><div><span class="section-index">02</span><h2>Patrouille et formation</h2></div>
      <p>Distance qualifiée, marges, resets et tolérances observées par rôle.</p></div>
      {patrol}<div class="charts">{patrol_charts}</div></section>
    <section><div class="section-head"><div><span class="section-index">03</span><h2>Récompenses par rôle</h2></div>
      <p>Décomposition de chaque canal, avec médiane, p95 et extrêmes.</p></div>{rewards}</section>
    <section><div class="section-head"><div><span class="section-index">04</span><h2>Pertes Blue</h2></div>
      <p>Origine, agent, vitesse, carburant et commande au moment de la perte.</p></div>{losses}</section>
    <section><div class="section-head"><div><span class="section-index">05</span><h2>Optimiseur et débit</h2></div>
      <p>Temps par phase, stabilité PPO et vitesse de l’évaluation.</p></div>{optimizer}</section>
    <section><h2>Courbes brutes d’entraînement</h2><div class="charts">{training_charts}</div></section>
    <section><h2>Artefacts et traçabilité</h2>{evidence}</section>
    <section class="method"><h2>Interprétation responsable</h2>
      <p>Les alertes KL, clipping et variance expliquée utilisent des seuils
      heuristiques : elles orientent le diagnostic, elles ne prouvent pas seules un défaut.</p>
      <p>Une CI verte ou un score ponctuel ne démontre pas la convergence. La promotion
      d’une politique demande plusieurs graines indépendantes et plusieurs compositions P/N.</p>
    </section>
    """.format(
        label=html.escape(label),
        subtitle=html.escape(_subtitle(analysis)),
        status_class=_status_class(analysis.alerts),
        status=html.escape(_status_text(analysis.alerts)),
        cards=_cards(
            [
                ("Score récent", _number(metrics["latest_score"], 2), "évaluation"),
                ("Meilleur score", _number(metrics["best_score"], 2), "run local"),
                ("Succès", _percent(metrics["success_rate"]), "dernière campagne"),
                ("Survie objectif", _optional_percent(metrics["objective_survival_rate"]), "dernière campagne"),
                ("Neutralisation Red", _optional_percent(metrics["red_neutralization_rate"]), "dernière campagne"),
                ("Survie Blue", _percent(metrics["blue_survival_rate"]), "épisodes journalisés"),
                ("Updates", str(metrics["end_update"]), "+{} observées".format(metrics["completed_updates"])),
                ("Débit", _number(metrics["updates_per_hour"], 1), "updates / heure"),
            ]
        ),
        alerts=_alerts(analysis.alerts),
        evaluation_charts=_evaluation_charts(analysis),
        patrol_charts=_patrol_charts(analysis),
        training_charts="".join(training_charts),
        evaluation=_evaluation_section(analysis),
        patrol=_patrol_section(analysis),
        rewards=_reward_section(analysis),
        losses=_loss_section(analysis),
        optimizer=_optimizer_section(analysis),
        evidence=_evidence_section(analysis),
    )
    destination.write_text(_page("Diagnostic Blue MAPPO " + label, body), encoding="utf-8")
    return destination


def write_comparison_report(
    left: RunAnalysis, right: RunAnalysis, destination: Path
) -> Path:
    """Write an evidence-aware comparison of two downloaded runs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    left_label = _run_label(left)
    right_label = _run_label(right)
    comparable, reasons = compatibility(left, right)
    paired = paired_evaluation_comparison(left, right)
    quality_chart = _chart(
        [
            ("Run {}".format(left_label), left.score_points),
            ("Run {}".format(right_label), right.score_points),
        ],
        "Scores d’évaluation superposés",
        "Campagne (ou update pour anciens artefacts)",
        "Score",
        y_bounds=(0.0, 100.0),
    )
    reward_chart = _chart(
        [
            ("Run {}".format(left_label), left.reward_points),
            ("Run {}".format(right_label), right.reward_points),
        ],
        "Récompenses lissées",
        "Temps d’entraînement cumulé (s)",
        "Récompense",
    )
    optimizer_chart = _chart(
        [
            ("Run {}".format(left_label), _optimizer_points(left.dataset.optimizer_updates, "total_ms")),
            ("Run {}".format(right_label), _optimizer_points(right.dataset.optimizer_updates, "total_ms")),
        ],
        "Durée de l’optimiseur",
        "Update PPO",
        "Millisecondes",
    )
    success_chart = _chart(
        [
            ("Succès run {}".format(left_label), _campaign_points(left, "success_rate", 100.0)),
            ("Succès run {}".format(right_label), _campaign_points(right, "success_rate", 100.0)),
            ("Objectif run {}".format(left_label), _campaign_points(left, "objective_survival_rate", 100.0)),
            ("Objectif run {}".format(right_label), _campaign_points(right, "objective_survival_rate", 100.0)),
        ],
        "Qualité par campagne",
        "Update (ou campagne)",
        "Pourcentage",
        y_bounds=(0.0, 100.0),
    )
    survival_chart = _chart(
        [
            ("Blue run {}".format(left_label), _campaign_points(left, "blue_survival_rate", 100.0)),
            ("Blue run {}".format(right_label), _campaign_points(right, "blue_survival_rate", 100.0)),
            ("Red neutralisés {}".format(left_label), _campaign_points(left, "red_neutralization_rate", 100.0)),
            ("Red neutralisés {}".format(right_label), _campaign_points(right, "red_neutralization_rate", 100.0)),
        ],
        "Survie Blue et neutralisation Red",
        "Update (ou campagne)",
        "Pourcentage",
        y_bounds=(0.0, 100.0),
    )
    compatibility_detail = (
        "Protocoles compatibles d’après les métadonnées disponibles."
        if comparable
        else "Comparaison qualité non concluante : " + "; ".join(reasons) + "."
    )
    body = """
    <header class="hero compare">
      <div><span class="eyebrow">PPOTrial · Comparateur Blue MAPPO</span>
      <h1>Run {left} <span>vs</span> Run {right}</h1>
      <p>{advantage}</p></div>
      <div class="status {status_class}">{status}</div>
    </header>
    <section><h2>Validité de la comparaison</h2><div class="callout {callout}">{detail}</div></section>
    <section><h2>Tableau comparatif complet</h2>{table}</section>
    <section><h2>Cas strictement communs</h2>{paired}</section>
    <section><h2>Patrouille et causes de perte</h2>{operational}</section>
    <section><h2>Courbes superposées</h2><div class="charts">{charts}</div></section>
    <section><h2>Anomalies par run</h2><div class="two-columns">
      <div><h3>Run {left}</h3>{left_alerts}</div>
      <div><h3>Run {right}</h3>{right_alerts}</div>
    </div></section>
    <section class="method"><h2>Décision recommandée</h2>
      <p>{advantage}</p>
      <p>Comparer prioritairement les campagnes d’évaluation exécutées avec le
      même scénario, la même composition P/N, les mêmes cas et le même contrat
      d’observation. Les récompenses d’entraînement de niveaux de curriculum
      différents ne sont pas directement comparables.</p>
    </section>
    """.format(
        left=html.escape(left_label),
        right=html.escape(right_label),
        advantage=html.escape(observed_advantage(left, right)),
        status_class="ok" if comparable else "warning",
        status="Comparable" if comparable else "Contexte différent",
        callout="ok" if comparable else "warning",
        detail=html.escape(compatibility_detail),
        table=_comparison_table(left, right),
        paired=_paired_section(paired, left_label, right_label),
        operational=_operational_comparison(left, right),
        charts=success_chart + survival_chart + quality_chart + reward_chart + optimizer_chart,
        left_alerts=_alerts(left.alerts),
        right_alerts=_alerts(right.alerts),
    )
    destination.write_text(
        _page("Comparaison Blue MAPPO {} vs {}".format(left_label, right_label), body),
        encoding="utf-8",
    )
    return destination


def _page(title: str, body: str) -> str:
    return """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body><main>{body}<footer>Généré localement par Blue Training Diagnostics · aucune donnée envoyée ailleurs.</footer></main></body></html>
""".format(title=html.escape(title), css=_css(), body=body)


def _css() -> str:
    return """
:root{color-scheme:dark;--bg:#07111f;--panel:#0d1b2d;--panel2:#10233a;--text:#e5eef9;--muted:#91a4bb;--line:#203954;--blue:#38bdf8;--purple:#a78bfa;--green:#34d399;--amber:#fbbf24;--red:#fb7185}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#133357 0,transparent 32rem),linear-gradient(180deg,#07111f,#091524);color:var(--text);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}main{max-width:1240px;margin:auto;padding:44px 28px}.hero{display:flex;justify-content:space-between;gap:32px;align-items:center;padding:28px 32px;background:linear-gradient(135deg,rgba(16,35,58,.96),rgba(13,27,45,.92));border:1px solid #284766;border-radius:22px;box-shadow:0 22px 60px #0006}.hero h1{font-size:clamp(30px,5vw,52px);line-height:1.05;margin:7px 0 12px}.hero h1 span{color:var(--muted);font-weight:400}.hero p{max-width:760px;color:#bbcadb;margin:0}.eyebrow{text-transform:uppercase;letter-spacing:.14em;color:var(--blue);font-weight:700;font-size:12px}.status{white-space:nowrap;padding:10px 16px;border-radius:999px;font-weight:750;border:1px solid}.status.ok{color:#9af0ce;border-color:#1f9d74;background:#0c4a363d}.status.warning{color:#ffe29a;border-color:#b7791f;background:#6b43073d}.status.critical{color:#ffb3c0;border-color:#be3455;background:#6e17303d}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:22px 0}.card{padding:18px;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px}.card .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.card .value{font-size:28px;font-weight:760;margin:4px 0}.card .hint{color:#8296ad;font-size:12px}section{margin-top:22px;padding:24px 26px;background:rgba(13,27,45,.9);border:1px solid var(--line);border-radius:18px;box-shadow:0 10px 32px #0003}h2{margin:0 0 16px;font-size:20px}h3{margin:22px 0 12px;color:#c9d8e8}.section-head{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:18px}.section-head>div{display:flex;gap:12px;align-items:center}.section-head h2{margin:0}.section-head p{max-width:570px;color:var(--muted);margin:0}.section-index{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;color:#bce9ff;background:#103656;border:1px solid #24618b;font-size:12px;font-weight:800}.alerts{display:grid;gap:10px}.alert,.callout{padding:13px 15px;border-radius:12px;border-left:4px solid;background:#0a1727}.alert strong{display:block}.alert span{color:#aebfd1}.alert.ok,.callout.ok{border-color:var(--green)}.alert.warning,.callout.warning{border-color:var(--amber)}.alert.critical{border-color:var(--red)}.charts{display:grid;gap:18px}.chart{padding:12px;background:#081523;border:1px solid #1d354e;border-radius:14px;overflow:auto}.chart svg{display:block;width:100%;min-width:680px;height:auto}.chart-empty,.empty{padding:30px;text-align:center;color:var(--muted);border:1px dashed #35506b;border-radius:12px}.table-wrap{overflow:auto;border:1px solid #1d354e;border-radius:12px}table{width:100%;border-collapse:collapse;overflow:hidden}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #1d354e}tbody tr:hover{background:#112740}th{color:#9fb3c9;font-size:12px;text-transform:uppercase;letter-spacing:.06em}td.numeric{text-align:right;font-variant-numeric:tabular-nums}.two-columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.pill{display:inline-block;padding:3px 9px;border-radius:99px;background:#17324d;color:#bbdcfa;margin:2px 4px 2px 0}.bars{display:grid;gap:10px}.bar-row{display:grid;grid-template-columns:minmax(130px,240px) 1fr 64px;gap:12px;align-items:center}.bar-track{height:10px;background:#07111f;border-radius:99px;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));border-radius:99px}.bar-value{text-align:right;font-variant-numeric:tabular-nums;color:#c9d8e8}.muted{color:var(--muted)}.method{border-color:#355173;background:linear-gradient(135deg,#10243b,#0d1b2d)}code{color:#b9e4ff;background:#07111f;padding:2px 5px;border-radius:5px}footer{text-align:center;color:#6f8399;padding:30px 0 8px;font-size:12px}@media(max-width:760px){main{padding:20px 12px}.hero{align-items:flex-start;flex-direction:column;padding:22px}.two-columns{grid-template-columns:1fr}.section-head{display:block}.section-head p{margin-top:10px}.bar-row{grid-template-columns:110px 1fr 48px}section{padding:18px 15px}}
"""


def _chart(
    series: Sequence[Tuple[str, Sequence[tuple]]],
    title: str,
    x_label: str,
    y_label: str,
    y_bounds: Optional[Tuple[float, float]] = None,
) -> str:
    cleaned = []
    for label, points in series:
        finite = [
            (float(x), float(y))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if finite:
            cleaned.append((label, _downsample(finite, 700)))
    if not cleaned:
        return '<div class="chart-empty">{} : données indisponibles.</div>'.format(
            html.escape(title)
        )
    width, height = 1040, 350
    left, right, top, bottom = 76, 24, 52, 58
    all_points = [point for _, points in cleaned for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    if y_bounds:
        y_min, y_max = y_bounds
    else:
        y_min = min(point[1] for point in all_points)
        y_max = max(point[1] for point in all_points)
        padding = max(1e-6, (y_max - y_min) * 0.08)
        y_min -= padding
        y_max += padding
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [
        '<div class="chart"><svg viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(
            width, height, html.escape(title)
        ),
        '<text x="{}" y="26" fill="#e5eef9" font-size="18" font-weight="700">{}</text>'.format(
            left, html.escape(title)
        ),
    ]
    for tick in range(6):
        fraction = tick / 5.0
        y = top + plot_h * fraction
        value = y_max - (y_max - y_min) * fraction
        parts.append(
            '<line x1="{}" x2="{}" y1="{:.1f}" y2="{:.1f}" stroke="#1d354e"/>'.format(
                left, width - right, y, y
            )
        )
        parts.append(
            '<text x="{}" y="{:.1f}" text-anchor="end" fill="#8296ad" font-size="11">{}</text>'.format(
                left - 8, y + 4, html.escape(_axis(value))
            )
        )
    for tick in range(6):
        fraction = tick / 5.0
        x = left + plot_w * fraction
        value = x_min + (x_max - x_min) * fraction
        parts.append(
            '<text x="{:.1f}" y="{}" text-anchor="middle" fill="#8296ad" font-size="11">{}</text>'.format(
                x, height - 29, html.escape(_axis(value))
            )
        )
    for index, (label, points) in enumerate(cleaned):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(
            "{:.1f},{:.1f}".format(
                left + plot_w * (x - x_min) / (x_max - x_min),
                top + plot_h * (y_max - y) / (y_max - y_min),
            )
            for x, y in points
        )
        parts.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="2.4" vector-effect="non-scaling-stroke"/>'.format(
                coordinates, color
            )
        )
        legend_x = left + index * 190
        parts.append(
            '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{}" stroke-width="3"/>'.format(
                legend_x, legend_x + 20, height - 8, height - 8, color
            )
        )
        parts.append(
            '<text x="{}" y="{}" fill="#aebfd1" font-size="11">{}</text>'.format(
                legend_x + 27, height - 4, html.escape(label)
            )
        )
    parts.extend(
        [
            '<text x="{}" y="{}" text-anchor="middle" fill="#8296ad" font-size="11">{}</text>'.format(
                left + plot_w / 2, height - 41, html.escape(x_label)
            ),
            '<text transform="translate(15,{}) rotate(-90)" text-anchor="middle" fill="#8296ad" font-size="11">{}</text>'.format(
                top + plot_h / 2, html.escape(y_label)
            ),
            "</svg></div>",
        ]
    )
    return "".join(parts)


def _cards(items: Sequence[Tuple[str, str, str]]) -> str:
    return '<div class="cards">{}</div>'.format(
        "".join(
            '<div class="card"><div class="label">{}</div><div class="value">{}</div><div class="hint">{}</div></div>'.format(
                html.escape(label), html.escape(value), html.escape(hint)
            )
            for label, value, hint in items
        )
    )


def _alerts(alerts: Sequence[DiagnosticAlert]) -> str:
    return '<div class="alerts">{}</div>'.format(
        "".join(
            '<div class="alert {}"><strong>{}</strong><span>{}</span></div>'.format(
                html.escape(alert.severity),
                html.escape(alert.title),
                html.escape(alert.detail),
            )
            for alert in alerts
        )
    )


def _evaluation_section(analysis: RunAnalysis) -> str:
    metrics = analysis.metrics
    terminals = metrics["terminals"]
    campaigns = metrics["evaluation_campaigns"]
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}/{}</td><td class=\"numeric\">{}</td>"
        "<td class=\"numeric\">{}</td><td class=\"numeric\">{}</td>"
        "<td class=\"numeric\">{}</td><td class=\"numeric\">{}</td>"
        "<td class=\"numeric\">{}</td><td>{}</td></tr>".format(
            item["update"], item["campaign"], item["P"], item["N"],
            item["episodes"], _percent(item["success_rate"]),
            _percent(item["blue_survival_rate"]),
            _optional_percent(item["red_neutralization_rate"]),
            _percent(item["objective_survival_rate"]),
            _number(item["steps_mean"], 0), _pills(item["terminals"]),
        )
        for item in campaigns[-20:]
    )
    return """
    {cards}
    <h3>Terminaisons de la dernière campagne</h3><p>{terminals}</p>
    <h3>Détail des campagnes récentes</h3>
    <div class="table-wrap"><table><thead><tr><th>Update</th><th>Campagne</th><th>P/N</th>
      <th>Cas</th><th>Succès</th><th>Survie Blue</th><th>Red neutralisés</th>
      <th>Objectif</th><th>Étapes moy.</th><th>Terminaisons</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
    """.format(
        cards=_cards(
            (
                ("Épisodes", str(metrics["evaluation_episodes"]), "dernière campagne"),
                ("Succès", _percent(metrics["success_rate"]), "combat gagné"),
                ("Survie Blue", _percent(metrics["blue_survival_rate"]), "agents survivants"),
                ("Objectif", _optional_percent(metrics["objective_survival_rate"]), "encore en vie"),
                ("Leader", _optional_percent(metrics["leader_survival_rate"]), "encore en vie"),
                ("Durée", _optional_number(metrics["evaluation_steps_mean"], 0), "étapes moyennes"),
            )
        ),
        terminals=_pills(terminals) if terminals else "Données indisponibles.",
        rows=rows or '<tr><td colspan="10" class="muted">Aucune campagne détaillée.</td></tr>',
    )


def _evaluation_charts(analysis: RunAnalysis) -> str:
    return _chart(
        [
            ("Succès", _campaign_points(analysis, "success_rate", 100.0)),
            ("Survie Blue", _campaign_points(analysis, "blue_survival_rate", 100.0)),
            ("Red neutralisés", _campaign_points(analysis, "red_neutralization_rate", 100.0)),
            ("Objectif vivant", _campaign_points(analysis, "objective_survival_rate", 100.0)),
            ("Leader vivant", _campaign_points(analysis, "leader_survival_rate", 100.0)),
        ],
        "Qualité par campagne d’évaluation",
        "Update (ou campagne)",
        "Pourcentage",
        y_bounds=(0.0, 100.0),
    ) + _chart(
        [("Durée moyenne", _campaign_points(analysis, "steps_mean", 1.0))],
        "Durée moyenne des épisodes",
        "Update (ou campagne)",
        "Étapes",
    )


def _campaign_points(analysis: RunAnalysis, field: str, scale: float) -> List[tuple]:
    points = []
    for item in analysis.metrics["evaluation_campaigns"]:
        value = item.get(field)
        if value is None:
            continue
        x_value = item["update"] if item["update"] else item["campaign"]
        points.append((x_value, float(value) * scale))
    return points


def _patrol_charts(analysis: RunAnalysis) -> str:
    trend = analysis.metrics["patrol"]["trend"]
    return _chart(
        [("Distance qualifiée", [(row["episode"], row["qualified_ratio"] * 100.0) for row in trend])],
        "Part du circuit qualifiée par épisode",
        "Épisode journalisé",
        "Pourcentage",
        y_bounds=(0.0, 100.0),
    ) + _chart(
        [("Marge de formation", [(row["episode"], row["spacing_margin_m"]) for row in trend])],
        "Marge d’espacement de la formation",
        "Épisode journalisé",
        "Mètres",
    )


def _patrol_section(analysis: RunAnalysis) -> str:
    patrol = analysis.metrics["patrol"]
    tolerances = analysis.metrics["tolerance_by_role"]
    rows = []
    labels = (("position", "Position"), ("speed", "Vitesse"), ("heading", "Cap"))
    for role, values in tolerances["roles"].items():
        for key, label in labels:
            distribution = values[key]
            rows.append(
                "<tr><td>{}</td><td>{}</td><td class=\"numeric\">{}</td>"
                "<td class=\"numeric\">{}</td><td class=\"numeric\">{}</td>"
                "<td class=\"numeric\">{}</td><td class=\"numeric\">{}</td></tr>".format(
                    html.escape(role), label, distribution["count"],
                    _optional_number(distribution["mean"], 3),
                    _optional_number(distribution["median"], 3),
                    _optional_number(distribution["p95"], 3),
                    _optional_number(distribution["max"], 3),
                )
            )
    spacing = patrol["spacing_margin"]
    sequence_rows = []
    sequence_metrics = (
        ("Distance au waypoint", "waypoint_distance", "m", 1),
        ("Écart latéral piste", "cross_track", "m", 1),
        ("Erreur de cap", "heading_error", "rad", 3),
        ("Vitesse", "speed", "m/s", 1),
        ("Abscisse parcourue", "arclength", "m", 1),
    )
    for label, key, unit, digits in sequence_metrics:
        distribution = patrol[key]
        sequence_rows.append(
            "<tr><td>{}</td><td>{}</td><td class=\"numeric\">{}</td>"
            "<td class=\"numeric\">{}</td><td class=\"numeric\">{}</td>"
            "<td class=\"numeric\">{}</td><td>{}</td></tr>".format(
                label, distribution["count"],
                _optional_number(distribution["mean"], digits),
                _optional_number(distribution["median"], digits),
                _optional_number(distribution["p95"], digits),
                _optional_number(distribution["max"], digits), unit,
            )
        )
    modes = ", ".join(
        "{}: {}".format(html.escape(str(key)), value)
        for key, value in sorted(patrol["modes"].items())
    ) or "indisponible"
    terminations = ", ".join(
        "{}: {}".format(html.escape(str(key)), value)
        for key, value in sorted(patrol["terminations"].items())
    ) or "indisponible"
    return _cards(
        (
            ("Épisodes", str(patrol["episodes"]), "patrouilles journalisées"),
            ("Distance qualifiée", _percent(patrol["qualified_ratio"]), "du circuit déclaré"),
            ("Resets", str(patrol["resets"]), "{} épisode(s)".format(patrol["episodes_with_resets"])),
            ("Violations", str(patrol["tracking_violations"]), "suivi de patrouille"),
            ("Marge médiane", _optional_number(spacing["median"], 1), "mètres"),
            ("Mesures valides", _percent(tolerances["valid_rate"]), "par agent et rôle"),
            ("Mode récent", html.escape(str(patrol["latest_mode"])), "RALLY ou FOLLOW"),
            ("Waypoints validés", str(patrol["validated_waypoints_max"]), "maximum WPT1 à WPT10"),
        )
    ) + """
    <h3>Séquence de waypoints</h3>
    <p class="muted">Modes : {}. Terminaisons : {}.</p>
    <div class="table-wrap"><table><thead><tr>
      <th>Mesure</th><th>N</th><th>Moyenne</th><th>Médiane</th><th>P95</th><th>Pire</th><th>Unité</th>
      </tr></thead><tbody>{}</tbody></table></div>
    <h3>Tolérances par rôle</h3><div class="table-wrap"><table><thead><tr>
      <th>Rôle</th><th>Mesure</th><th>N</th><th>Moyenne</th><th>Médiane</th><th>P95</th><th>Pire</th>
      </tr></thead><tbody>{}</tbody></table></div>
    """.format(
        modes, terminations,
        "".join(sequence_rows),
        "".join(rows) or '<tr><td colspan="7" class="muted">Télémétrie de patrouille indisponible.</td></tr>',
    )


def _reward_section(analysis: RunAnalysis) -> str:
    summary = analysis.metrics["reward_by_role"]
    rows = []
    for role, channels in summary["roles"].items():
        for channel, distribution in channels.items():
            rows.append(
                "<tr><td>{}</td><td>{}</td><td class=\"numeric\">{}</td>"
                "<td class=\"numeric\">{}</td><td class=\"numeric\">{}</td>"
                "<td class=\"numeric\">{}</td><td class=\"numeric\">{} → {}</td></tr>".format(
                    html.escape(role), html.escape(channel), distribution["count"],
                    _optional_number(distribution["mean"], 6),
                    _optional_number(distribution["median"], 6),
                    _optional_number(distribution["p95"], 6),
                    _optional_number(distribution["min"], 6),
                    _optional_number(distribution["max"], 6),
                )
            )
    return """
    <div class="table-wrap"><table><thead><tr><th>Rôle</th><th>Canal</th><th>N</th>
      <th>Moyenne</th><th>Médiane</th><th>P95</th><th>Étendue</th></tr></thead>
      <tbody>{}</tbody></table></div>
    """.format("".join(rows) or '<tr><td colspan="7" class="muted">Aucun canal de récompense détaillé.</td></tr>')


def _loss_section(analysis: RunAnalysis) -> str:
    metrics = analysis.metrics
    detail = metrics["loss_diagnostics"]
    sources = metrics["loss_sources"]
    by_blue = sorted(detail["by_blue"].items(), key=lambda item: (-item[1], item[0]))
    rows = "".join(
        '<tr><td>{}</td><td>{}</td><td class="numeric">{}</td></tr>'.format(
            html.escape(str(key[0])), html.escape(str(key[1])), count
        )
        for key, count in by_blue
    )
    return _cards(
        (
            ("Pertes", str(detail["count"]), "transitions explicites"),
            ("Basse vitesse", str(detail["low_speed"]), "flag low_speed"),
            ("Carburant vide", str(detail["fuel_empty"]), "flag fuel_empty"),
            ("Δ vitesse médian", _optional_number(detail["speed_delta"]["median"], 2), "par transition"),
            ("Carburant médian", _optional_percent(detail["fuel_fraction"]["median"]), "fraction restante"),
            ("|accel| p95", _optional_number(detail["abs_accel"]["p95"], 4), "commande absolue"),
        )
    ) + "<div class=\"two-columns\"><div><h3>Causes</h3>{}</div>".format(
        _bar_list(sources)
    ) + """
      <div><h3>Par agent et cause</h3><div class="table-wrap"><table><thead>
      <tr><th>Blue</th><th>Cause</th><th>Nombre</th></tr></thead><tbody>{}</tbody></table></div></div></div>
    """.format(rows or '<tr><td colspan="3" class="muted">Aucune perte journalisée.</td></tr>')


def _optimizer_section(analysis: RunAnalysis) -> str:
    metrics = analysis.metrics
    status = metrics["status_diagnostics"]
    timing = metrics["optimizer_timing"]
    rows = [
        ("Updates diagnostiquées", str(metrics["optimizer_updates"])),
        ("Durée moyenne", _milliseconds(metrics["optimizer_ms_mean"])),
        ("Durée p95", _milliseconds(metrics["optimizer_ms_p95"])),
        ("Policy loss récente", _optional(metrics["policy_loss_latest"])),
        ("Value loss récente", _optional(metrics["value_loss_latest"])),
        ("Entropie récente", _optional(metrics["entropy_latest"])),
        ("KL approximative", _optional(metrics["approx_kl_latest"])),
        ("Fraction clippée", _optional_percent(metrics["clip_fraction_latest"])),
        ("Variance expliquée", _optional(metrics["explained_variance_latest"])),
        ("Écart-type moyen actions", _optional(metrics["action_std_latest"])),
    ]
    timing_rows = []
    for key, label in (("total", "Total"), ("critic", "Critic"),
                       ("advantages", "Advantages"), ("ppo", "PPO")):
        distribution = timing[key]
        timing_rows.append((
            label,
            "{} / {} / {}".format(
                _optional_milliseconds(distribution["median"]),
                _optional_milliseconds(distribution["p95"]),
                _optional_milliseconds(distribution["max"]),
            ),
        ))
    speed = status["evaluation_speed"]
    return _cards(
        (
            ("Optimiseur médian", _optional_milliseconds(timing["total"]["median"]), "par update"),
            ("Optimiseur p95", _optional_milliseconds(timing["total"]["p95"]), "queue lente"),
            ("Évaluation médiane", _optional_number(speed["median"], 1), "étapes / seconde"),
            ("Temps en évaluation", _percent(status["evaluating_sample_rate"]), "échantillons CSV"),
        )
    ) + '<div class="two-columns"><div><h3>Stabilité PPO</h3>{}</div><div><h3>Temps médian / p95 / max</h3>{}</div></div>'.format(
        _key_value_table(rows), _key_value_table(timing_rows)
    )


def _evidence_section(analysis: RunAnalysis) -> str:
    dataset = analysis.dataset
    metadata = dataset.metadata
    rows = [
        ("Run GitHub", str(metadata.get("id") or "inconnu")),
        ("Branche", str(metadata.get("head_branch") or "inconnue")),
        ("Commit", str(metadata.get("head_sha") or "inconnu")[:12]),
        ("Créé", str(metadata.get("created_at") or "inconnu")),
        ("Segments de télémétrie", str(len(dataset.segments))),
        ("Épisodes d’évaluation", str(len(dataset.evaluations))),
        ("Épisodes de patrouille", str(len(dataset.patrol_episodes))),
        ("Mesures par agent", str(len(dataset.patrol_agents))),
        ("Récompenses par rôle", str(len(dataset.role_rewards))),
        ("Pertes Blue détaillées", str(len(dataset.losses))),
        ("Événements timing", str(len(dataset.timing_events))),
        ("Checkpoints", str(analysis.metrics["checkpoint_count"])),
        ("Taille checkpoints", _bytes(analysis.metrics["checkpoint_bytes"])),
        ("Manifest CI", "présent" if dataset.manifest else "absent (ancien run)"),
    ]
    segment_rows = "".join(
        "<tr><td>{}</td><td class=\"numeric\">{}</td><td class=\"numeric\">{}</td></tr>".format(
            html.escape(segment.label), len(segment.samples), segment.invalid_rows
        )
        for segment in dataset.segments
    )
    return _key_value_table(rows) + """
    <h3>Flux détectés</h3><table><thead><tr><th>Segment</th><th>Échantillons</th><th>Rejets</th></tr></thead>
    <tbody>{}</tbody></table>
    """.format(segment_rows or '<tr><td colspan="3">Aucun flux.</td></tr>')


def _comparison_table(left: RunAnalysis, right: RunAnalysis) -> str:
    rows = (
        ("Score récent", _number(left.metrics["latest_score"], 2), _number(right.metrics["latest_score"], 2)),
        ("Meilleur score", _number(left.metrics["best_score"], 2), _number(right.metrics["best_score"], 2)),
        ("Succès", _percent(left.metrics["success_rate"]), _percent(right.metrics["success_rate"])),
        ("Survie Blue", _percent(left.metrics["blue_survival_rate"]), _percent(right.metrics["blue_survival_rate"])),
        ("Neutralisation Red", _optional_percent(left.metrics["red_neutralization_rate"]), _optional_percent(right.metrics["red_neutralization_rate"])),
        ("Survie objectif", _optional_percent(left.metrics["objective_survival_rate"]), _optional_percent(right.metrics["objective_survival_rate"])),
        ("Survie leader", _optional_percent(left.metrics["leader_survival_rate"]), _optional_percent(right.metrics["leader_survival_rate"])),
        ("Durée évaluation", _optional_number(left.metrics["evaluation_steps_mean"], 0), _optional_number(right.metrics["evaluation_steps_mean"], 0)),
        ("Distance qualifiée", _percent(left.metrics["patrol"]["qualified_ratio"]), _percent(right.metrics["patrol"]["qualified_ratio"])),
        ("Resets patrouille", str(left.metrics["patrol"]["resets"]), str(right.metrics["patrol"]["resets"])),
        ("Violations de suivi", str(left.metrics["patrol"]["tracking_violations"]), str(right.metrics["patrol"]["tracking_violations"])),
        ("Pertes Blue", str(left.metrics["loss_diagnostics"]["count"]), str(right.metrics["loss_diagnostics"]["count"])),
        ("Updates observées", str(left.metrics["completed_updates"]), str(right.metrics["completed_updates"])),
        ("Updates / heure", _number(left.metrics["updates_per_hour"], 1), _number(right.metrics["updates_per_hour"], 1)),
        ("Optimiseur moyen", _milliseconds(left.metrics["optimizer_ms_mean"]), _milliseconds(right.metrics["optimizer_ms_mean"])),
        ("Reward récente", _number(left.metrics["reward_recent_mean"], 4), _number(right.metrics["reward_recent_mean"], 4)),
        ("Alertes critiques", str(_severity_count(left, "critical")), str(_severity_count(right, "critical"))),
        ("Alertes", str(_severity_count(left, "warning")), str(_severity_count(right, "warning"))),
    )
    return """
    <table><thead><tr><th>Indicateur</th><th>Run {}</th><th>Run {}</th></tr></thead><tbody>{}</tbody></table>
    """.format(
        html.escape(_run_label(left)),
        html.escape(_run_label(right)),
        "".join(
            '<tr><td>{}</td><td class="numeric">{}</td><td class="numeric">{}</td></tr>'.format(
                html.escape(label), html.escape(left_value), html.escape(right_value)
            )
            for label, left_value, right_value in rows
        ),
    )


def _paired_section(paired: Dict[str, object], left_label: str, right_label: str) -> str:
    pairs = paired["pairs"]
    if not pairs:
        return '<div class="empty">Aucun quartet P/N + seed + cas commun : comparaison appariée impossible.</div>'
    rows = "".join(
        "<tr><td>{}/{}</td><td>{}</td><td>{}</td><td>{} → {}</td>"
        "<td class=\"numeric\">{} → {}</td><td class=\"numeric\">{} → {}</td>"
        "<td class=\"numeric\">{} → {}</td></tr>".format(
            row["P"], row["N"], row["seed"], row["case"],
            html.escape(row["left_terminal"]), html.escape(row["right_terminal"]),
            row["left_blue_alive"], row["right_blue_alive"],
            row["left_red_alive"], row["right_red_alive"],
            row["left_steps"], row["right_steps"],
        )
        for row in pairs[:80]
    )
    return _cards(
        (
            ("Cas communs", str(paired["count"]), "mêmes P/N, seed et cas"),
            ("Δ Blue vivants", _number(paired["blue_alive_delta_mean"], 2), "{} − {}".format(right_label, left_label)),
            ("Δ Red vivants", _number(paired["red_alive_delta_mean"], 2), "négatif = mieux"),
            ("Δ durée", _number(paired["steps_delta_mean"], 1), "étapes par cas"),
            ("Issues modifiées", str(paired["terminal_changes"]), "terminaison différente"),
        )
    ) + """
    <div class="table-wrap"><table><thead><tr><th>P/N</th><th>Seed</th><th>Cas</th>
      <th>Terminaison {left} → {right}</th><th>Blue vivants</th><th>Red vivants</th><th>Étapes</th>
      </tr></thead><tbody>{rows}</tbody></table></div>
    """.format(left=html.escape(left_label), right=html.escape(right_label), rows=rows)


def _operational_comparison(left: RunAnalysis, right: RunAnalysis) -> str:
    return '<div class="two-columns"><div><h3>Run {}</h3>{}</div><div><h3>Run {}</h3>{}</div></div>'.format(
        html.escape(_run_label(left)),
        _bar_list(left.metrics["loss_sources"], "Cause de perte"),
        html.escape(_run_label(right)),
        _bar_list(right.metrics["loss_sources"], "Cause de perte"),
    ) + _comparison_table_subset(
        left,
        right,
        (
            ("Épisodes patrouille", lambda item: str(item.metrics["patrol"]["episodes"])),
            ("Distance qualifiée", lambda item: _percent(item.metrics["patrol"]["qualified_ratio"])),
            ("Marge médiane", lambda item: _optional_number(item.metrics["patrol"]["spacing_margin"]["median"], 1)),
            ("Tolérances valides", lambda item: _percent(item.metrics["tolerance_by_role"]["valid_rate"])),
            ("Pertes basse vitesse", lambda item: str(item.metrics["loss_diagnostics"]["low_speed"])),
            ("Pertes carburant", lambda item: str(item.metrics["loss_diagnostics"]["fuel_empty"])),
        ),
    )


def _comparison_table_subset(left, right, definitions) -> str:
    rows = "".join(
        '<tr><td>{}</td><td class="numeric">{}</td><td class="numeric">{}</td></tr>'.format(
            html.escape(label), html.escape(formatter(left)), html.escape(formatter(right))
        )
        for label, formatter in definitions
    )
    return '<div class="table-wrap"><table><thead><tr><th>Indicateur</th><th>Run {}</th><th>Run {}</th></tr></thead><tbody>{}</tbody></table></div>'.format(
        html.escape(_run_label(left)), html.escape(_run_label(right)), rows
    )


def _bar_list(counter, empty_label: str = "Aucune donnée") -> str:
    items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    if not items:
        return '<div class="empty">{}</div>'.format(html.escape(empty_label))
    maximum = max(float(value) for _, value in items) or 1.0
    return '<div class="bars">{}</div>'.format("".join(
        '<div class="bar-row"><span>{}</span><div class="bar-track"><div class="bar-fill" style="width:{:.1f}%"></div></div><span class="bar-value">{}</span></div>'.format(
            html.escape(str(key)), float(value) * 100.0 / maximum, value
        )
        for key, value in items[:18]
    ))


def _key_value_table(rows: Sequence[Tuple[str, str]]) -> str:
    return '<table><tbody>{}</tbody></table>'.format(
        "".join(
            '<tr><th>{}</th><td class="numeric">{}</td></tr>'.format(
                html.escape(label), html.escape(value)
            )
            for label, value in rows
        )
    )


def _optimizer_points(rows: Sequence[Dict[str, float]], field: str) -> List[tuple]:
    return [
        (row["update"], row[field])
        for row in rows
        if "update" in row and field in row
    ]


def _pills(counter) -> str:
    return "".join(
        '<span class="pill">{} · {}</span>'.format(html.escape(str(key)), value)
        for key, value in sorted(counter.items())
    )


def _subtitle(analysis: RunAnalysis) -> str:
    metadata = analysis.dataset.metadata
    return "{} · {} · commit {}".format(
        metadata.get("conclusion") or metadata.get("status") or "état inconnu",
        metadata.get("head_branch") or "branche inconnue",
        str(metadata.get("head_sha") or "inconnu")[:10],
    )


def _status_class(alerts: Sequence[DiagnosticAlert]) -> str:
    if any(alert.severity == "critical" for alert in alerts):
        return "critical"
    if any(alert.severity == "warning" for alert in alerts):
        return "warning"
    return "ok"


def _status_text(alerts: Sequence[DiagnosticAlert]) -> str:
    status = _status_class(alerts)
    return {"critical": "Action requise", "warning": "À surveiller", "ok": "Sain"}[status]


def _severity_count(analysis: RunAnalysis, severity: str) -> int:
    return sum(alert.severity == severity for alert in analysis.alerts)


def _run_label(analysis: RunAnalysis) -> str:
    metadata = analysis.dataset.metadata
    return str(metadata.get("run_number") or metadata.get("id") or analysis.dataset.root.name)


def _number(value, digits: int) -> str:
    try:
        return ("{:,.%df}" % digits).format(float(value)).replace(",", " ")
    except (TypeError, ValueError):
        return "n/a"


def _optional(value) -> str:
    return "n/a" if value is None else _number(value, 5)


def _optional_number(value, digits: int) -> str:
    return "n/a" if value is None else _number(value, digits)


def _percent(value) -> str:
    try:
        return "{:.1%}".format(float(value))
    except (TypeError, ValueError):
        return "n/a"


def _optional_percent(value) -> str:
    return "n/a" if value is None else _percent(value)


def _milliseconds(value) -> str:
    return "{:.0f} ms".format(float(value)) if value else "n/a"


def _optional_milliseconds(value) -> str:
    return "n/a" if value is None else "{:.0f} ms".format(float(value))


def _bytes(value: int) -> str:
    size = float(value)
    for unit in ("o", "Kio", "Mio", "Gio"):
        if size < 1024.0 or unit == "Gio":
            return "{:.1f} {}".format(size, unit)
        size /= 1024.0
    return "{} o".format(value)


def _axis(value: float) -> str:
    absolute = abs(value)
    if absolute >= 10000 or (0 < absolute < 0.001):
        return "{:.1e}".format(value)
    if absolute >= 100:
        return "{:.0f}".format(value)
    if absolute >= 1:
        return "{:.2f}".format(value)
    return "{:.3f}".format(value)


def _downsample(points: Sequence[tuple], maximum: int) -> List[tuple]:
    if len(points) <= maximum:
        return list(points)
    return [
        points[int(index * (len(points) - 1) / float(maximum - 1))]
        for index in range(maximum)
    ]
