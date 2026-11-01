"""Polished Tkinter desktop interface for Blue MAPPO CI diagnostics."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .credentials import git_credential_token
from .github_client import GitHubActionsClient
from .models import Repository, WorkflowRun
from .service import (
    analyze_local_run,
    compare_local_runs,
    download_and_analyze,
    local_run_available,
    local_run_directory,
)
from .windows_paths import diagnostic_root


BACKGROUND = "#07111f"
PANEL = "#0d1b2d"
PANEL_ALT = "#10233a"
TEXT = "#e5eef9"
MUTED = "#91a4bb"
BLUE = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#fb7185"


class DiagnosticApplication:
    """Own the Tk widgets and keep all network operations off the UI thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.download_root = diagnostic_root()
        self.runs: Dict[str, WorkflowRun] = {}
        self.events = queue.Queue()
        self.busy = False
        self.repository = tk.StringVar(value=detect_repository().full_name)
        self.status = tk.StringVar(value="Prêt — chargez les runs GitHub Actions.")
        self.progress_text = tk.StringVar(value="")
        self._configure_window()
        self._configure_style()
        self._build()
        self.root.after(80, self._poll_events)
        self.root.after(250, self.refresh_runs)

    def _configure_window(self) -> None:
        self.root.title("PPOTrial · Diagnostic Blue MAPPO")
        self.root.geometry("1220x760")
        self.root.minsize(980, 620)
        self.root.configure(bg=BACKGROUND)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Hero.TFrame", background=PANEL_ALT)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
        style.configure("Hero.TLabel", background=PANEL_ALT, foreground=TEXT)
        style.configure("Muted.TLabel", background=BACKGROUND, foreground=MUTED)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Accent.TLabel", background=PANEL_ALT, foreground=BLUE, font=("Segoe UI Semibold", 9))
        style.configure("Title.TLabel", background=PANEL_ALT, foreground=TEXT, font=("Segoe UI Semibold", 24))
        style.configure("TButton", background="#17324d", foreground=TEXT, borderwidth=0, padding=(14, 9), font=("Segoe UI Semibold", 9))
        style.map("TButton", background=[("active", "#214766"), ("disabled", "#142234")], foreground=[("disabled", "#60758b")])
        style.configure("Primary.TButton", background="#0877a8", foreground="white", padding=(16, 10))
        style.map("Primary.TButton", background=[("active", "#0b91c8"), ("disabled", "#16405a")])
        style.configure("TEntry", fieldbackground="#081523", foreground=TEXT, insertcolor=TEXT, bordercolor="#284766", lightcolor="#284766", darkcolor="#284766", padding=7)
        style.configure("Treeview", background="#091827", fieldbackground="#091827", foreground=TEXT, rowheight=31, borderwidth=0)
        style.configure("Treeview.Heading", background="#142b44", foreground="#b9cadc", relief="flat", font=("Segoe UI Semibold", 9))
        style.map("Treeview", background=[("selected", "#16557a")], foreground=[("selected", "white")])
        style.map("Treeview.Heading", background=[("active", "#1d3b59")])
        style.configure("Horizontal.TProgressbar", background=BLUE, troughcolor="#0a1727", bordercolor="#0a1727", lightcolor=BLUE, darkcolor=BLUE)

    def _build(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill=tk.BOTH, expand=True)
        hero = ttk.Frame(container, style="Hero.TFrame", padding=(24, 19))
        hero.pack(fill=tk.X)
        title_box = ttk.Frame(hero, style="Hero.TFrame")
        title_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_box, text="PPOTrial · BLUE MAPPO", style="Accent.TLabel").pack(anchor=tk.W)
        ttk.Label(title_box, text="Centre de diagnostic CI", style="Title.TLabel").pack(anchor=tk.W, pady=(2, 3))
        ttk.Label(
            title_box,
            text="Téléchargez, inspectez et comparez les entraînements sans dépendance Python externe.",
            style="Hero.TLabel",
        ).pack(anchor=tk.W)
        ttk.Button(hero, text="Ouvrir les téléchargements", command=self.open_downloads).pack(side=tk.RIGHT, padx=(20, 0))

        connection = ttk.Frame(container, style="Panel.TFrame", padding=(18, 14))
        connection.pack(fill=tk.X, pady=(14, 10))
        ttk.Label(connection, text="Dépôt", style="Panel.TLabel").grid(row=0, column=0, sticky=tk.W)
        repo_entry = ttk.Entry(connection, textvariable=self.repository, width=31)
        repo_entry.grid(row=1, column=0, sticky=tk.EW, padx=(0, 12), pady=(4, 0))
        ttk.Label(connection, text="Authentification", style="Panel.TLabel").grid(row=0, column=1, sticky=tk.W)
        auth_panel = ttk.Frame(connection, style="Panel.TFrame")
        auth_panel.grid(row=1, column=1, sticky=tk.EW, padx=(0, 12), pady=(4, 0))
        ttk.Label(
            auth_panel,
            text="✓ Git / Git Extensions automatique",
            style="Panel.TLabel",
            foreground=GREEN,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor=tk.W)
        ttk.Label(
            connection,
            text="Aucun secret n’est affiché ou enregistré par l’application.",
            style="Panel.TLabel",
        ).grid(row=2, column=1, sticky=tk.W, pady=(3, 0))
        self.refresh_button = ttk.Button(connection, text="Actualiser les runs", style="Primary.TButton", command=self.refresh_runs)
        self.refresh_button.grid(row=1, column=2, sticky=tk.EW, pady=(4, 0))
        connection.columnconfigure(0, weight=1)
        connection.columnconfigure(1, weight=1)

        table_panel = ttk.Frame(container, style="Panel.TFrame", padding=(12, 12))
        table_panel.pack(fill=tk.BOTH, expand=True)
        table_header = ttk.Frame(table_panel, style="Panel.TFrame")
        table_header.pack(fill=tk.X, padx=5, pady=(1, 9))
        ttk.Label(table_header, text="Runs MAPPO training", style="Panel.TLabel", font=("Segoe UI Semibold", 12)).pack(side=tk.LEFT)
        ttk.Label(table_header, text="Ctrl+clic pour en choisir deux", style="Panel.TLabel").pack(side=tk.RIGHT)
        columns = ("number", "created", "state", "branch", "commit", "title", "local")
        self.tree = ttk.Treeview(table_panel, columns=columns, show="headings", selectmode="extended")
        headings = {
            "number": "RUN",
            "created": "DATE",
            "state": "ÉTAT",
            "branch": "BRANCHE",
            "commit": "COMMIT",
            "title": "TITRE",
            "local": "LOCAL",
        }
        widths = {"number": 70, "created": 145, "state": 110, "branch": 120, "commit": 95, "title": 360, "local": 90}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], minwidth=55, stretch=column == "title")
        self.tree.tag_configure("success", foreground="#9af0ce")
        self.tree.tag_configure("failure", foreground="#ff9fb0")
        self.tree.tag_configure("running", foreground="#8bdcff")
        scrollbar = ttk.Scrollbar(table_panel, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<Double-1>", lambda _event: self.open_selected_github())

        actions = ttk.Frame(container, padding=(0, 12, 0, 0))
        actions.pack(fill=tk.X)
        self.download_button = ttk.Button(actions, text="Télécharger + diagnostiquer", style="Primary.TButton", command=self.download_selected)
        self.download_button.pack(side=tk.LEFT)
        self.compare_button = ttk.Button(actions, text="Comparer 2 runs", command=self.compare_selected)
        self.compare_button.pack(side=tk.LEFT, padx=(9, 0))
        ttk.Button(actions, text="Analyser un dossier local", command=self.analyze_folder).pack(side=tk.LEFT, padx=(9, 0))
        ttk.Button(actions, text="Voir le run sur GitHub", command=self.open_selected_github).pack(side=tk.LEFT, padx=(9, 0))

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X)
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(footer, textvariable=self.progress_text, style="Muted.TLabel").pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(container, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(5, 0))

    def refresh_runs(self) -> None:
        if self.busy:
            return
        try:
            client = self._client()
        except ValueError as error:
            messagebox.showerror("Dépôt invalide", str(error), parent=self.root)
            return
        self._start(
            "Chargement des runs GitHub Actions…",
            lambda: client.list_runs(),
            self._show_runs,
        )

    def download_selected(self) -> None:
        selected = self._selected_runs()
        if not selected:
            messagebox.showinfo("Sélection requise", "Sélectionnez au moins un run.", parent=self.root)
            return
        repository = Repository.parse(self.repository.get())

        def task():
            self._queue_progress("Authentification Git / Git Extensions", 0, 0)
            client = GitHubActionsClient(
                repository, git_credential_token(repository)
            )
            reports = []
            for run in selected:
                _analysis, report = download_and_analyze(
                    client, run, self.download_root, self._queue_progress
                )
                reports.append(report)
            return reports

        self._start(
            "Téléchargement et diagnostic de {} run(s)…".format(len(selected)),
            task,
            self._downloads_finished,
        )

    def compare_selected(self) -> None:
        selected = self._selected_runs()
        if len(selected) != 2:
            messagebox.showinfo("Deux runs requis", "Sélectionnez exactement deux runs avec Ctrl+clic.", parent=self.root)
            return
        repository = Repository.parse(self.repository.get())

        def task():
            roots = []
            client = None
            for run in selected:
                run_root = local_run_directory(self.download_root, run.run_id)
                if not local_run_available(run_root):
                    if client is None:
                        self._queue_progress(
                            "Authentification Git / Git Extensions", 0, 0
                        )
                        client = GitHubActionsClient(
                            repository, git_credential_token(repository)
                        )
                    download_and_analyze(client, run, self.download_root, self._queue_progress)
                roots.append(run_root)
            return compare_local_runs(roots[0], roots[1], self.download_root)[2]

        self._start("Comparaison des deux runs…", task, self._comparison_finished)

    def analyze_folder(self) -> None:
        path = filedialog.askdirectory(
            parent=self.root,
            title="Choisir un dossier de run extrait",
            initialdir=str(self.download_root),
        )
        if not path:
            return
        self._start(
            "Analyse du dossier local…",
            lambda: analyze_local_run(Path(path))[1],
            self._single_report_finished,
        )

    def open_downloads(self) -> None:
        self.download_root.mkdir(parents=True, exist_ok=True)
        _open_path(self.download_root)

    def open_selected_github(self) -> None:
        selected = self._selected_runs()
        if selected and selected[0].html_url:
            webbrowser.open(selected[0].html_url)

    def _client(self) -> GitHubActionsClient:
        return GitHubActionsClient(Repository.parse(self.repository.get()))

    def _selected_runs(self) -> List[WorkflowRun]:
        return [self.runs[item] for item in self.tree.selection() if item in self.runs]

    def _show_runs(self, runs: Sequence[WorkflowRun]) -> None:
        self.tree.delete(*self.tree.get_children())
        self.runs.clear()
        for run in runs:
            item = str(run.run_id)
            local = local_run_available(
                local_run_directory(self.download_root, run.run_id)
            )
            conclusion = run.conclusion or run.status
            tag = "success" if conclusion == "success" else "running" if run.status != "completed" else "failure"
            self.tree.insert(
                "",
                tk.END,
                iid=item,
                values=(
                    "#{}".format(run.number),
                    _display_date(run.created_at),
                    conclusion,
                    run.branch,
                    run.commit[:8],
                    run.title,
                    "disponible" if local else "—",
                ),
                tags=(tag,),
            )
            self.runs[item] = run
        self.status.set("{} run(s) MAPPO chargés.".format(len(runs)))

    def _downloads_finished(self, reports: Sequence[Path]) -> None:
        self.status.set("Diagnostic terminé pour {} run(s).".format(len(reports)))
        self.refresh_runs()
        if reports:
            webbrowser.open(reports[-1].resolve().as_uri())

    def _comparison_finished(self, report: Path) -> None:
        self.status.set("Comparaison générée : {}".format(report))
        webbrowser.open(report.resolve().as_uri())

    def _single_report_finished(self, report: Path) -> None:
        self.status.set("Rapport généré : {}".format(report))
        webbrowser.open(report.resolve().as_uri())

    def _start(self, label: str, task: Callable, callback: Callable) -> None:
        if self.busy:
            return
        self.busy = True
        self.status.set(label)
        self.progress.configure(value=0, mode="indeterminate")
        self.progress.start(12)
        self._set_buttons(False)

        def worker():
            try:
                result = task()
                self.events.put(("success", callback, result))
            except Exception as error:  # Converted into a main-thread dialog.
                self.events.put(("error", error))

        threading.Thread(target=worker, name="blue-diagnostic-worker", daemon=True).start()

    def _queue_progress(self, label: str, current: int, total: int) -> None:
        self.events.put(("progress", label, current, total))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, label, current, total = event
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=(100.0 * current / total if total else 0.0))
                    self.progress_text.set("{} · {} / {}".format(label, _bytes(current), _bytes(total)) if total else label)
                elif event[0] == "success":
                    _, callback, result = event
                    self._finish_busy()
                    callback(result)
                elif event[0] == "error":
                    self._finish_busy()
                    error = event[1]
                    self.status.set("Échec : {}".format(error))
                    messagebox.showerror("Diagnostic impossible", str(error), parent=self.root)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)

    def _finish_busy(self) -> None:
        self.busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.progress_text.set("")
        self._set_buttons(True)

    def _set_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.refresh_button.configure(state=state)
        self.download_button.configure(state=state)
        self.compare_button.configure(state=state)


def detect_repository() -> Repository:
    """Detect origin from the enclosing checkout, with a project fallback."""
    current = Path(__file__).resolve()
    repository_root = None
    for parent in current.parents:
        if (parent / ".git").exists():
            repository_root = parent
            break
    if repository_root:
        try:
            completed = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(repository_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=5,
                check=False,
            )
            if completed.returncode == 0:
                return Repository.parse(completed.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return Repository("benoitfragit", "PPOTrial")


def run() -> None:
    """Create and enter the desktop event loop."""
    root = tk.Tk()
    DiagnosticApplication(root)
    root.mainloop()


def _display_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
        return parsed.astimezone().strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return value


def _bytes(value: int) -> str:
    size = float(value)
    for unit in ("o", "Kio", "Mio", "Gio"):
        if size < 1024.0 or unit == "Gio":
            return "{:.1f} {}".format(size, unit)
        size /= 1024.0
    return "{} o".format(value)


def _open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.resolve().as_uri())

