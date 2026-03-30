\mainpage PPOTrial Documentation

[TOC]

## Overview

This site gathers the Doxygen comments already present in the C++ codebase and
adds a project-level reference for the JSON configuration files used by the
runtime, the renderer, the scenarios, and the RL agents.

The generated documentation is meant to stay outside the CMake build chain.
You can regenerate it on demand with a standalone script whenever the code or
the JSON formats evolve.

## What Is Included

- API reference generated from the documented headers and source files.
- Source browsing and cross-references between types, functions, files, and
  namespaces.
- A JSON reference page that explains the actual file formats consumed by the
  code.
- Optional Graphviz diagrams when `dot.exe` is installed alongside Doxygen.

## Generate The Docs

1. Double click `generatedoc.bat` from the repository root.
2. The script looks for `doxygen.exe` on the `PATH` and in common Windows
   installation folders.
3. If `dot.exe` from Graphviz is available, call graphs and dependency graphs
   are enabled automatically.
4. The HTML site is generated in `docs/generated/html`.
5. The script opens `docs/generated/html/index.html` automatically on success.

## Recommended Output Format

The default and recommended output for this repository is **HTML**.

- It is the easiest format to browse locally.
- It supports search, diagrams, source navigation, and cross-links well.
- It benefits the most from the custom stylesheet shipped in this repository.

If you want more later, Doxygen can also generate other outputs:

- **HTML + Graphviz**: best day-to-day format and the prettiest choice here.
- **LaTeX/PDF**: useful for frozen deliverables or printable snapshots.
- **XML / QHP**: useful only if you later want IDE or external tooling
  integration.

## JSON Reference

The JSON configuration formats are documented here:

- \subpage json_formats "JSON File Formats"
