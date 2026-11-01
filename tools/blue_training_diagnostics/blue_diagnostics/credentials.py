"""Read an existing Git/Git Extensions credential without persisting it."""
from __future__ import annotations

import os
import subprocess
from typing import Dict

from .github_client import GitHubApiError
from .models import Repository


def git_credential_token(repository: Repository) -> str:
    """Return the in-memory GitHub token supplied by Git Credential Manager.

    The standard ``git credential fill`` protocol delegates to the helper
    configured by Git Extensions/Git. The returned password is never printed
    or written by this application.

    Args:
        repository: Validated GitHub repository coordinates.

    Raises:
        GitHubApiError: If Git or its credential helper cannot authenticate.
    """
    request = (
        "protocol=https\n"
        "host=github.com\n"
        "path={}/{}.git\n\n"
    ).format(repository.owner, repository.name)
    startup = None
    creation_flags = 0
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creation_flags = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            ["git", "credential", "fill"],
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=120,
            check=False,
            startupinfo=startup,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GitHubApiError(
            "Git Credential Manager est indisponible : {}".format(error)
        ) from error
    if completed.returncode != 0:
        raise GitHubApiError(
            "Git Credential Manager n’a pas pu authentifier GitHub. "
            "Connectez Git Extensions au dépôt puis réessayez."
        )
    fields: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    token = fields.get("password", "")
    if not token:
        raise GitHubApiError(
            "Aucun identifiant GitHub n’a été fourni par Git Credential Manager. "
            "Effectuez une opération Git authentifiée dans Git Extensions puis réessayez."
        )
    return token
