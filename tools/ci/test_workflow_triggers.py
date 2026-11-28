"""The pull-request triggers of the workflows that build the C++ follow what they build.

A change of the shared engine (PPOKit) or of the Framework must start the validations of
every example that consumes them; a job that is not triggered is not a job that passed.
The triggers are read as text (no YAML package on every runner): the `on:` section, up to
`jobs:`.
"""
from __future__ import annotations

import unittest
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def trigger_section(name: str) -> str:
    """The `on:` section of a workflow, as text."""
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    start = text.index("\non:") + 1
    end = text.index("\njobs:", start)
    return text[start:end]


def path_filters(name: str) -> list[str]:
    """The `paths:` entries of the pull_request trigger (empty: every change triggers it)."""
    section = trigger_section(name)
    if "pull_request:" not in section:
        raise AssertionError(f"{name} does not run on pull requests")
    filters = []
    collecting = False
    for line in section.splitlines():
        stripped = line.strip()
        if stripped == "paths:":
            collecting = True
            continue
        if collecting and stripped.startswith("- "):
            filters.append(stripped[2:].strip().strip("'\""))
        elif collecting and stripped and not stripped.startswith("#"):
            collecting = False
    return filters


class WorkflowTriggerTests(unittest.TestCase):
    def assert_triggered_by(self, workflow: str, *paths: str) -> None:
        filters = path_filters(workflow)
        for path in paths:
            with self.subTest(workflow=workflow, path=path):
                self.assertTrue(not filters or path in filters,
                                f"{workflow} does not run when {path} changes: {filters}")

    def test_core_runs_on_every_pull_request(self) -> None:
        self.assertEqual(path_filters("core-tests.yml"), [])

    def test_core_builds_and_tests_the_release_directory(self) -> None:
        workflow = (WORKFLOWS / "core-tests.yml").read_text(encoding="utf-8")
        for command in ("cmake -S . -B", "cmake --build", "ctest --test-dir"):
            with self.subTest(command=command):
                self.assertIn(f"run: {command} build/release ", workflow)

    def test_flight_runs_when_the_shared_engine_changes(self) -> None:
        self.assert_triggered_by("flight-control.yml", "PPOKit/**", "Framework/**",
                                 "Examples/FlightControl3D/**", "CMakeLists.txt",
                                 "dependencies.cmake", "tools/flight/**")

    def test_linux_flight_jobs_use_pinned_compiler(self) -> None:
        workflow = (WORKFLOWS / "flight-control.yml").read_text(encoding="utf-8")
        matrix, steps = workflow.split("    steps:", 1)
        self.assertIn("os: [ubuntu-24.04, windows-2022]", matrix)
        self.assertIn("- os: ubuntu-24.04\n            compiler_flags: -DCMAKE_C_COMPILER=gcc-13 "
                      "-DCMAKE_CXX_COMPILER=g++-13", matrix)
        self.assertIn("${{ matrix.compiler_flags }}", steps)

    def test_desktop_runs_on_every_source_it_builds(self) -> None:
        self.assert_triggered_by("desktop-windows.yml", "PPOKit/**", "Framework/**", "Examples/**",
                                 "**/CMakeLists.txt", "**/*.cmake")

    def test_restart_lint_runs_when_its_scripts_change(self) -> None:
        self.assert_triggered_by("mappo-restart-progressive.yml", "tools/ci/**",
                                 ".github/workflows/mappo-training-windows.yml")


if __name__ == "__main__":
    unittest.main()
