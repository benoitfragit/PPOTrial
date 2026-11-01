"""Application services joining GitHub downloads, analysis and reports."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .analysis import analyze
from .github_client import (
    GitHubActionsClient,
    GitHubApiError,
    artifact_names,
    prepare_run_directory,
)
from .models import Artifact, RunAnalysis, WorkflowRun
from .report import write_comparison_report, write_single_report
from .telemetry import load_run


Progress = Callable[[str, int, int], None]


def download_and_analyze(
    client: GitHubActionsClient,
    run: WorkflowRun,
    root: Path,
    progress: Optional[Progress] = None,
) -> Tuple[RunAnalysis, Path]:
    """Download the two Blue artifacts, analyze them and render the report."""
    run_root = prepare_run_directory(root, run)
    artifacts = [
        artifact
        for artifact in client.list_artifacts(run.run_id)
        if artifact.name in artifact_names()
    ]
    names = {artifact.name for artifact in artifacts if not artifact.expired}
    if "blue-mappo-telemetry" not in names:
        raise GitHubApiError(
            "Le run {} ne contient plus de télémétrie Blue téléchargeable.".format(
                run.number
            )
        )
    artifacts_root = run_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        if artifact.expired:
            continue
        destination = artifacts_root / "{}-{}".format(
            artifact.name, artifact.artifact_id
        )
        marker = destination / "artifact.json"
        if marker.is_file():
            if progress:
                progress("{} déjà présent".format(artifact.name), artifact.size, artifact.size)
            continue
        if destination.exists():
            destination = artifacts_root / "{}-{}-nouveau".format(
                artifact.name, artifact.artifact_id
            )
        if progress:
            progress("Téléchargement de {}".format(artifact.name), 0, artifact.size)
        client.download_artifact(
            artifact,
            destination,
            lambda current, total, name=artifact.name: progress(
                "Téléchargement de {}".format(name), current, total
            )
            if progress
            else None,
        )
        _write_artifact_marker(marker if marker.parent == destination else destination / "artifact.json", artifact)
    return analyze_local_run(run_root)


def analyze_local_run(run_root: Path) -> Tuple[RunAnalysis, Path]:
    """Analyze an already downloaded run and regenerate its HTML report."""
    analysis = analyze(load_run(run_root))
    report = write_single_report(analysis, run_root / "reports" / "diagnostic.html")
    return analysis, report


def compare_local_runs(
    left_root: Path, right_root: Path, output_root: Path
) -> Tuple[RunAnalysis, RunAnalysis, Path]:
    """Analyze and compare two local run folders."""
    left, _ = analyze_local_run(left_root)
    right, _ = analyze_local_run(right_root)
    left_id = left.dataset.metadata.get("id") or left_root.name
    right_id = right.dataset.metadata.get("id") or right_root.name
    destination = (
        output_root
        / "comparisons"
        / "{}-vs-{}".format(left_id, right_id)
        / "comparaison.html"
    )
    report = write_comparison_report(left, right, destination)
    return left, right, report


def local_run_directory(root: Path, run_id: int) -> Path:
    """Return the deterministic local path for one GitHub run ID."""
    return root / "runs" / "run-{}".format(int(run_id))


def local_run_available(run_root: Path) -> bool:
    """Return true only when usable Blue telemetry exists locally."""
    artifacts = run_root / "artifacts"
    if not artifacts.is_dir():
        return False
    return any(artifacts.rglob("mappo-training.csv")) or any(
        artifacts.rglob("telemetry.csv")
    )


def _write_artifact_marker(path: Path, artifact: Artifact) -> None:
    document = {
        "id": artifact.artifact_id,
        "name": artifact.name,
        "size_in_bytes": artifact.size,
        "digest": artifact.digest,
        "created_at": artifact.created_at,
        "expires_at": artifact.expires_at,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))

