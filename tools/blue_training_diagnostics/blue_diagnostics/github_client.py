"""Small dependency-free GitHub Actions REST client."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import Artifact, Repository, WorkflowRun


class GitHubApiError(RuntimeError):
    """A user-facing GitHub REST or artifact validation failure."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Expose GitHub's signed download redirect without forwarding a token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


class GitHubActionsClient:
    """Read workflow runs and download artifacts with no third-party package."""

    api_root = "https://api.github.com"

    def __init__(self, repository: Repository, token: str = "") -> None:
        self.repository = repository
        self.token = token.strip()
        self._redirect_blocking_opener = urllib.request.build_opener(
            _RejectRedirects()
        )

    def _headers(self, include_token: bool = True) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "PPOTrial-Blue-Diagnostics/1.0",
            "X-GitHub-Api-Version": "2026-03-10",
        }
        if include_token and self.token:
            headers["Authorization"] = "Bearer {}".format(self.token)
        return headers

    def _request_json(self, path: str) -> Dict[str, Any]:
        url = self.api_root + path
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = _http_error_detail(error)
            raise GitHubApiError(
                "GitHub a répondu HTTP {} : {}".format(error.code, detail)
            ) from error
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise GitHubApiError("Connexion GitHub impossible : {}".format(error)) from error

    def list_runs(
        self, workflow: str = "mappo-training-windows.yml", limit: int = 100
    ) -> List[WorkflowRun]:
        """List the most recent runs for the dedicated MAPPO workflow."""
        safe_workflow = urllib.parse.quote(workflow, safe="")
        per_page = max(1, min(int(limit), 100))
        path = "/repos/{}/actions/workflows/{}/runs?per_page={}".format(
            self.repository.full_name, safe_workflow, per_page
        )
        document = self._request_json(path)
        return [WorkflowRun.from_api(item) for item in document.get("workflow_runs", [])]

    def list_artifacts(self, run_id: int) -> List[Artifact]:
        """List non-pagination-limited artifacts for one training run."""
        path = "/repos/{}/actions/runs/{}/artifacts?per_page=100".format(
            self.repository.full_name, int(run_id)
        )
        document = self._request_json(path)
        return [Artifact.from_api(item) for item in document.get("artifacts", [])]

    def download_artifact(
        self,
        artifact: Artifact,
        destination: Path,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Download, verify and safely extract an artifact into destination."""
        if artifact.expired:
            raise GitHubApiError(
                "L’artefact {} a expiré le {}".format(
                    artifact.name, artifact.expires_at or "jour indiqué par GitHub"
                )
            )
        if not self.token:
            raise GitHubApiError(
                "Le téléchargement nécessite un jeton GitHub avec Actions: read."
            )
        destination.mkdir(parents=True, exist_ok=True)
        redirect_url = self._signed_download_url(artifact.download_url)
        archive = destination.parent / (artifact.name + ".zip.part")
        digest = hashlib.sha256()
        downloaded = 0
        request = urllib.request.Request(
            redirect_url, headers=self._headers(include_token=False)
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response, archive.open(
                "wb"
            ) as output:
                total = int(response.headers.get("Content-Length") or artifact.size or 0)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total)
            _verify_digest(artifact, digest.hexdigest())
            _safe_extract(archive, destination)
        except (urllib.error.URLError, OSError, zipfile.BadZipFile) as error:
            raise GitHubApiError(
                "Téléchargement de {} impossible : {}".format(artifact.name, error)
            ) from error
        finally:
            try:
                archive.unlink()
            except FileNotFoundError:
                pass
        return destination

    def _signed_download_url(self, url: str) -> str:
        request = urllib.request.Request(url, headers=self._headers())
        try:
            self._redirect_blocking_opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            if error.code == 302 and error.headers.get("Location"):
                return str(error.headers["Location"])
            detail = _http_error_detail(error)
            raise GitHubApiError(
                "GitHub a refusé l’artefact (HTTP {}) : {}".format(
                    error.code, detail
                )
            ) from error
        raise GitHubApiError("GitHub n’a pas fourni l’URL signée de l’artefact.")


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        document = json.loads(error.read().decode("utf-8", errors="replace"))
        return str(document.get("message") or error.reason)
    except (ValueError, OSError):
        return str(error.reason)


def _verify_digest(artifact: Artifact, actual: str) -> None:
    if not artifact.digest:
        return
    algorithm, separator, expected = artifact.digest.partition(":")
    if separator and algorithm.lower() == "sha256" and expected.lower() != actual.lower():
        raise GitHubApiError(
            "Empreinte SHA-256 invalide pour {} : archive rejetée.".format(
                artifact.name
            )
        )


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files only and reject ZIP path traversal/symlinks."""
    destination_root = destination.resolve()
    with zipfile.ZipFile(str(archive)) as compressed:
        for member in compressed.infolist():
            member_path = Path(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise GitHubApiError("Chemin ZIP non sûr : {}".format(member.filename))
            mode = (member.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise GitHubApiError("Lien symbolique ZIP refusé : {}".format(member.filename))
            target = (destination / member_path).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as error:
                raise GitHubApiError(
                    "Extraction ZIP hors du dossier cible refusée."
                ) from error
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with compressed.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def prepare_run_directory(root: Path, run: WorkflowRun) -> Path:
    """Create a deterministic local directory and persist run metadata."""
    run_root = root / "runs" / "run-{}".format(run.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    metadata_path = run_root / "run.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(run.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(metadata_path))
    return run_root


def artifact_names() -> tuple:
    """Return the only artifacts in scope for Blue diagnostics."""
    return ("blue-mappo-checkpoints", "blue-mappo-telemetry")
