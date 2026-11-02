"""Generate the existing standalone Blue MAPPO HTML report on the CI runner.

Only standard-library dependencies; inputs are already present at end of training.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

from .analysis import analyze
from .report import write_single_report
from .telemetry import load_run


def generate_ci_report(
    run_root: Path,
    telemetry_csv: Path,
    report_path: Path,
    summary_path: Optional[Path],
    metadata: Dict[str, object],
    checkpoints: Iterable[Path] = (),
) -> Path:
    """Render from local runner evidence, without downloading the current run."""
    run_root.mkdir(parents=True, exist_ok=True)
    if telemetry_csv.is_file():
        destination = run_root / "mappo-training.csv"
        if telemetry_csv.resolve() != destination.resolve():
            shutil.copyfile(str(telemetry_csv), str(destination))
    dataset = load_run(run_root)
    dataset.metadata.update(metadata)
    # The checkpoints live outside build/mappo-telemetry on the runner.
    # Refer to their real paths rather than duplicating large model files.
    available = [path for path in checkpoints if path.is_file()]
    if available:
        dataset.checkpoint_files = available
    analysis = analyze(dataset)
    report = write_single_report(analysis, report_path)
    if summary_path is not None:
        _append_summary(summary_path, analysis.metrics, len(dataset.log_errors))
    return report


def _append_summary(path: Path, metrics: Dict[str, object], errors: int) -> None:
    """GitHub Actions supports Markdown, not embedded standalone HTML."""
    score = float(metrics["latest_score"])
    success = float(metrics["success_rate"])
    patrol = metrics["patrol"]
    lines = [
        "## Diagnostic automatique Blue MAPPO",
        "",
        "**Rapport HTML complet** : téléchargez l’artefact "
        "`blue-mappo-diagnostic` (fichier `diagnostic.html`) "
        "depuis la section *Artifacts* de ce run.",
        "",
        "| Indicateur | Valeur |",
        "| --- | ---: |",
        "| Dernier update PPO | {} |".format(int(metrics["end_update"])),
        "| Score d’évaluation récent | {:.2f} |".format(score),
        "| Succès dernière campagne | {:.1%} |".format(success),
        "| Épisodes de patrouille analysés | {} |".format(patrol["episodes"]),
        "| Distance de patrouille qualifiée | {:.0f} m |".format(patrol["qualified_m"]),
        "| Violations de suivi de patrouille | {} |".format(patrol["tracking_violations"]),
        "| Erreurs MAPPO détectées | {} |".format(errors),
        "",
        "Le rapport HTML contient les courbes, distributions, pertes, "
        "rôles et détails par campagne. Une CI verte ne démontre pas "
        "la convergence de la politique.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> None:
    root = Path("build/mappo-telemetry")
    env = os.environ
    metadata = {
        "id": int(env.get("GITHUB_RUN_ID", "0")),
        "run_number": int(env.get("GITHUB_RUN_NUMBER", "0")),
        "run_attempt": int(env.get("GITHUB_RUN_ATTEMPT", "1")),
        "head_branch": env.get("GITHUB_REF_NAME", ""),
        "head_sha": env.get("GITHUB_SHA", ""),
        "html_url": "{}/actions/runs/{}".format(
            env.get("GITHUB_SERVER_URL", "https://github.com")
            + "/" + env.get("GITHUB_REPOSITORY", ""),
            env.get("GITHUB_RUN_ID", "0"),
        ),
    }
    checkpoint_root = Path("resources/model/BluePatrol/latest")
    checkpoints = (
        checkpoint_root / "blue_patrol_generalized_training_v6.txt",
        checkpoint_root / "blue_patrol_generalized_v9.txt",
    )
    result = generate_ci_report(
        root,
        Path("resources/model/BluePatrol/telemetry/mappo-training.csv"),
        root / "diagnostic.html",
        Path(env["GITHUB_STEP_SUMMARY"]) if env.get("GITHUB_STEP_SUMMARY") else None,
        metadata,
        checkpoints,
    )
    print("Blue MAPPO diagnostic HTML: {}".format(result))


if __name__ == "__main__":
    main()
