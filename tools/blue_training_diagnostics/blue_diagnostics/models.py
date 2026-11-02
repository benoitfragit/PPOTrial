"""Typed value objects shared by the Blue training diagnostic tool."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Repository:
    """GitHub repository coordinates used by REST API requests."""

    owner: str
    name: str

    @property
    def full_name(self) -> str:
        """Return the canonical ``owner/name`` form."""
        return "{}/{}".format(self.owner, self.name)

    @classmethod
    def parse(cls, value: str) -> "Repository":
        """Parse a repository name or HTTPS/SSH Git remote URL."""
        cleaned = value.strip().rstrip("/")
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]
        if "github.com/" in cleaned:
            cleaned = cleaned.split("github.com/", 1)[1]
        elif "github.com:" in cleaned:
            cleaned = cleaned.split("github.com:", 1)[1]
        parts = [part for part in cleaned.split("/") if part]
        owner_pattern = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
        repository_pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
        if (len(parts) != 2 or not owner_pattern.fullmatch(parts[0])
                or not repository_pattern.fullmatch(parts[1])):
            raise ValueError("Dépôt GitHub attendu au format propriétaire/dépôt")
        return cls(parts[0], parts[1])



@dataclass(frozen=True)
class WorkflowRun:
    """Relevant metadata for one GitHub Actions workflow run."""

    run_id: int
    number: int
    attempt: int
    status: str
    conclusion: str
    branch: str
    commit: str
    created_at: str
    updated_at: str
    html_url: str
    title: str

    @classmethod
    def from_api(cls, document: Dict[str, Any]) -> "WorkflowRun":
        """Build a validated run value from a GitHub REST response."""
        return cls(
            run_id=int(document["id"]),
            number=int(document.get("run_number", 0)),
            attempt=int(document.get("run_attempt", 1)),
            status=str(document.get("status") or "unknown"),
            conclusion=str(document.get("conclusion") or ""),
            branch=str(document.get("head_branch") or ""),
            commit=str(document.get("head_sha") or ""),
            created_at=str(document.get("created_at") or ""),
            updated_at=str(document.get("updated_at") or ""),
            html_url=str(document.get("html_url") or ""),
            title=str(document.get("display_title") or ""),
        )

    def to_json(self) -> Dict[str, Any]:
        """Return JSON-serializable run metadata for the local cache."""
        return {
            "id": self.run_id,
            "run_number": self.number,
            "run_attempt": self.attempt,
            "status": self.status,
            "conclusion": self.conclusion,
            "head_branch": self.branch,
            "head_sha": self.commit,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "html_url": self.html_url,
            "display_title": self.title,
        }


@dataclass(frozen=True)
class Artifact:
    """Download metadata for one GitHub Actions artifact."""

    artifact_id: int
    name: str
    size: int
    expired: bool
    created_at: str
    expires_at: str
    digest: str
    download_url: str

    @classmethod
    def from_api(cls, document: Dict[str, Any]) -> "Artifact":
        """Build artifact metadata from a GitHub REST response."""
        return cls(
            artifact_id=int(document["id"]),
            name=str(document.get("name") or "artifact"),
            size=int(document.get("size_in_bytes", 0)),
            expired=bool(document.get("expired", False)),
            created_at=str(document.get("created_at") or ""),
            expires_at=str(document.get("expires_at") or ""),
            digest=str(document.get("digest") or ""),
            download_url=str(document["archive_download_url"]),
        )


@dataclass
class TelemetrySegment:
    """One continuous telemetry stream, possibly a curriculum block."""

    label: str
    csv_path: Path
    log_path: Optional[Path]
    samples: List[Dict[str, float]] = field(default_factory=list)
    invalid_rows: int = 0


@dataclass
class DiagnosticAlert:
    """One deterministic diagnostic finding displayed in a report."""

    severity: str
    title: str
    detail: str


@dataclass
class RunDataset:
    """All local evidence extracted for a downloaded training run."""

    root: Path
    metadata: Dict[str, Any]
    manifest: Dict[str, Any]
    segments: List[TelemetrySegment]
    optimizer_updates: List[Dict[str, float]]
    timing_events: List[Dict[str, Any]]
    status_snapshots: List[Dict[str, Any]]
    evaluations: List[Dict[str, str]]
    losses: List[Dict[str, str]]
    patrol_episodes: List[Dict[str, str]]
    patrol_agents: List[Dict[str, str]]
    role_rewards: List[Dict[str, Any]]
    log_errors: List[str]
    checkpoint_files: List[Path]
    curriculum_progress: Dict[str, Any]
    patrol_progress: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class RunAnalysis:
    """Derived metrics and findings for a single run."""

    dataset: RunDataset
    metrics: Dict[str, Any]
    alerts: List[DiagnosticAlert]
    score_points: List[tuple]
    reward_points: List[tuple]
    update_points: List[tuple]

