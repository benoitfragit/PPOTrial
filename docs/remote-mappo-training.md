# Windows Release remote MAPPO training

## Architecture

The existing `PPO.exe` now accepts a headless training mode, without adding
a second executable or changing the red production `RLMission` agent.

- Blue team: RLTools generalized shared MAPPO actor and centralized critic.
- Red team: unchanged LibTorch `RLMission` production model and weights.
- Simulator: the same ECS physics, world rules, observations, actions and
  episode resets as the desktop.
- UI: Raylib/ImGui is never initialized by `--no-gui`.
- Toolchain: VS2022 x64 **Release**, `PPO_LIBTORCH_VARIANT=Release`.
  `dependencies.cmake` pins LibTorch **2.9.1 +cpu**.

Headless mode skips the desktop's 2 ms artificial delay and publishes ECS
snapshots once per 256 iterations. This does not remove any simulator steps.

## Launch locally

```powershell
cmake -S . -B build/release -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release -DPPO_LIBTORCH_VARIANT=Release -DPPO_BUILD_DESKTOP=ON
cmake --build build/release --config Release --target PPO --parallel 2
& build/release/Main/Release/PPO.exe --no-gui --example blue-defense --max-seconds 14400
```

Only `--example blue-defense` is accepted in headless training mode.
`blue-defense-inference` is a desktop production run and never enters the
headless learner monitor. Training stops at its time limit or when
`maxUpdates` is reached, and returns an error when no resumable training-state
checkpoint exists.

## Visual local run-84 2v2 rehearsal

For the run-84 investigation on Windows, double-click the repository-root
launcher:

```text
Lancer_MAPPO_2v2_Balanced_GUI.bat
```

The launcher incrementally builds only the fixed x64 Release tree, validates
the V10 actor and complete V7 optimizer checkpoint at update 1790, and opens
the normal Raylib/ImGui Blue Defense view. It presses Play automatically and
closes/reopens the window only at process-block boundaries, so every block uses
the same scenario restart semantics as the progressive CI runner.

The schedule directly calls the production `balanced_v1` selector. Each
20-block cycle contains `1v0=3`, `2v0=4`, `1v1=2`, `2v1=3`, and `2v2=8`.
Blocks stop at the next `checkpointEvery` evaluation boundary, exactly like
the CI gate runner. The default wall-clock budget is 7200 seconds.

Before the first update, the launcher copies the source actor, full optimizer
and production actor into `build/local-mappo-gui/session-*/source-run84` and
records SHA-256 digests. Stable progress is stored in
`build/local-mappo-gui/state.json`. Closing the GUI stops the automatic loop;
running the BAT again resumes the same incomplete rehearsal block. During any
exit, `BluePatrol.json`, `BlueDefense.json`, the production actor, its score,
`Renderer.json` and `imgui.ini` are restored byte-for-byte. `latest`
intentionally remains at the newest completed PPO update.

Python 3.12 is required. A non-mutating preflight prints the next exact cycle
and update boundary:

```powershell
py -3.12 tools/local_mappo_gui_curriculum.py --dry-run
```

For a short visual inspection, bypass the BAT after building and stop after a
single completed block:

```powershell
py -3.12 tools/local_mappo_gui_curriculum.py --max-blocks 1 --max-seconds 1800
```

The C++ `--auto-run` option is restricted to an explicitly selected Blue
Defense automatic mode. Unlike `--no-gui`, it keeps rendering and the normal
desktop 2 ms simulation throttle; it is an investigation tool, not a throughput
benchmark.

## Checkpoints

`resources/settings/Training/BluePatrol.json` supports:

- `training.rolloutSteps`: number of environment steps per PPO update
  (currently 2048).
- `training.maxUpdates`: maximum PPO updates (currently 2000 for standard non-curriculum training).
- `training.checkpointEvery`: deterministic evaluation interval (currently 10).
- `training.saveEveryUpdates`: independent checkpoint interval (1 for CI).
  When absent, it defaults to `checkpointEvery` for compatibility.

The learner writes these files at completed PPO-update boundaries:

- `resources/model/BluePatrol/latest/blue_patrol_generalized_v10.txt`:
  inference actor.
- `resources/model/BluePatrol/latest/blue_patrol_generalized_training_v7.txt`:
  **full training state**, using internal format `BLUE_TEAM_TRAINING_V7`.
- `resources/model/BluePatrol/prod/blue_patrol_generalized_v10.txt`:
  promoted actor only after successful evaluation, when present.

Training-state serialization includes actor, critic, Adam moments,
optimizer scalars, RNG and completed PPO updates. Partial in-progress rollout
steps are **not** included; interruptions can lose the current rollout. Saving
is performed atomically at optimizer boundaries.

## GitHub Actions

Use **Actions > MAPPO training (Windows Release) > Run workflow** once the
workflow file has been merged into the repository's default branch.

Choose the scenario, `max_seconds`, and `save_every_updates`; optionally
supply a previous workflow `resume_run_id` to restore the
`blue-mappo-checkpoints` artifact before training.

The workflow checks out the requested commit, downloads the same LibTorch
release through CMake, compiles `PPO` in Release, verifies scenario startup,
launches training and publishes downloadable artifacts. A fresh output
checkpoint is required for a successful training job. Artifact retention is
30 days; download useful states before expiry.

Each new telemetry artifact also contains `run-manifest.json`, with the run,
commit, workflow inputs and configuration SHA-256 hashes needed to audit
comparisons. `mappo-optimizer.csv` records per-update policy/value loss,
entropy, approximate KL, clipping fraction, explained variance, action standard
deviation, derivative activity and optimizer duration. Older artifacts without
these files remain supported by the local diagnostic tool.

## Local download and diagnostic UI

On Windows, double-click:

```text
tools\blue_training_diagnostics\Lancer_Diagnostic_Blue.bat
```

The standard-library-only Tkinter application lists this workflow's runs,
downloads the two Blue artifacts into the user's known Downloads folder,
generates a standalone HTML/SVG diagnostic, and compares two selected runs.
No Python package installation is required. Artifact downloads reuse the
existing Git/Git Extensions authentication through Git Credential Manager.
The credential is held in memory only and is never displayed, logged or stored
by the application.

The report consumes the detailed evaluation, patrol, per-role reward, Blue
loss, status and optimizer timing markers. Comparison pairs only identical
`P/N + seed + case` observations when showing case-level deltas.

See [Blue Training Diagnostics](../tools/blue_training_diagnostics/README.md)
for the metrics, security controls, tests and comparison rules. In particular,
scores or rewards from different curriculum levels and P/N contexts are never
presented as directly comparable quality evidence.

GitHub-hosted runners have limited job duration, so the default time budget is
4 hours, allowing headroom for building and packaging. Workflow execution is
**manual**, not automatic; it must be started by the user. A new workflow with
`workflow_dispatch` normally becomes available in the Actions UI after it
exists on the default branch.

## Complete V3 curriculum from scratch

Select curriculum_enabled=true, curriculum_version=v3, resume_run_id empty,
max_seconds=14400 and save_every_updates=1. V3 starts random actor/critic/Adam
state at update zero, with observation v10 and 2048-step rollouts. It does not
migrate run 22 or any older actor. Twelve phases total 1200 PPO updates, from
1vs0 / 2vs0 / 3vs0 / 4vs0 to progressive combat and a final 4vs4. Rehearsal
uses 24 primary updates followed by 6 updates from earlier phases, round-robin.
The first phase has no earlier task; every phase ends with its primary task.

To continue this new run across jobs, select its own run ID with the same V3
JSON fingerprint and observation v10. Old V2/V3 progress is rejected. The runner
only resets current Blue files in its temporary checkout on fresh start;
archived candidate files and RLMission are untouched. The old training filename
ending in v6 is no longer loaded by the runtime. Format V7 is the full
optimizer envelope, with an independently checked observation version of 10.

The 4-hour budget is a wall-clock limit, not a guarantee that all 1200 updates
finish in one job. Incomplete progressing blocks continue while time remains;
a complete checkpoint can resume later. Scenario changes occur between PPO
blocks/processes; earlier tasks are replayed throughout later phases.

See [blue-mappo-v3.md](blue-mappo-v3.md) for targets, rewards, physical lap
criteria, feasibility checks and diagnostics. No name V4 is used for this work.

## Restart from committed branch checkpoints (PR #75)

The existing **Restart Progressive Blue Defense (checkpoint)** action has a
checkpoint-source selector in the existing `source_run_id` field: enter
`branch` to use the selected branch's committed checkpoints instead of a run
ID. This preserves all ten original GitHub Actions inputs (the platform permits
no more than ten). A numeric ID continues the existing run-source behavior.
Branch mode starts a fresh curriculum from the exact selected commit:

- `resources/model/BluePatrol/latest/blue_patrol_generalized_v10.txt`:
  the LATEST actor corresponding to the full V7 optimizer.
- `resources/model/BluePatrol/latest/blue_patrol_generalized_training_v7.txt`:
  critic, Adam, RNG and PPO update counter; this is the actual training seed.
- `resources/model/BluePatrol/prod/blue_patrol_generalized_v10.txt`:
  separate production inference incumbent, preserved without treating its
  score as a curriculum-gate certification.

All three files must be committed to the selected branch. The bootstrap checks
semantic V10/V7 headers, actor architecture compatibility, positive optimizer
updates and SHA-256 provenance. Its checkpoint artifact records the commit and
three file hashes, and carries an *unverified warm start* into the existing
`Progressive Blue Defense (Windows Release)` action. Existing BEST checkpoints
from the branch are ignored; no prior gates are automatically declared passed.
With `balanced_v1`, rehearsal eligibility is independent of source and gate
certification: every preceding gate in the selected curriculum is replayed
(including zero-Red patrol scenarios), whether the restart originates from
`branch` or from a numeric run ID. For 3v0 through 3v3, this includes the
1v0/2v0/1v1/2v1/2v2 scenarios; for 4v0 through 4v4, earlier 3vX scenarios
are also rehearsed. At 3vX/4vX the current gate receives a strict majority
of blocks; the original 2v2 weighted shares remain unchanged.
For team sizes 3 and 4, selecting legacy automatically uses balanced_v1
(while 2v2 keeps its original legacy behavior). Other 2v2-only profiles are
rejected on these team sizes.
A 3vX or 4vX gate cannot PASS until at least one complete deterministic
balanced rehearsal cycle has executed, including every preceding scenario.
An early 20-case evaluation (even 100%) or previously evaluated BEST cannot
skip this minimum. Rehearsal blockIndex persists across resumed GitHub runs;
a subsequent evaluation of the active gate must meet the success threshold.
Rehearsal does not set `validatedGates`, award an artificial PASS or
replace BEST without evaluation. Only independent gate evaluations
(20 cases per campaign) determine promotion. There is no 2000-update global
limit: the executable maxUpdates is a temporary per-block cumulative target;
each gate has its own wall-clock budget.

To begin at 2v2, 3v3 or 4v4, select `max_team_size` accordingly,
`start_gate` for the desired stage, `rehearsal_profile=balanced_v1`,
your objective/zone variant and `source_run_id=branch`.
This mode **does not upload files directly from a local PC**: push the matching
V10/V7 and PROD files to the selected branch first. When a numeric run ID is
supplied, the pre-existing run-based behavior is unchanged. No new workflow/action
is created and RLMission, physics, actions and rewards are not changed.

## Limitations and acceptance checks

1. This is a Windows x64 headless **runtime**, not a graphics-free compile:
   the existing desktop dependencies are still linked, though no window opens.
2. The GitHub runner builds the project for each workflow run; the dependency
   download may dominate a cold first run.
3. Checkpoint restore preserves optimizer state, but not an interrupted
   in-progress rollout or the full ECS episode state.
4. C++ unit/smoke tests and Windows Release CI are not evidence of RL
   convergence: evaluate across independent seeds and varying P/N scenarios.
5. Before merging, require successful VS2022 Release compilation, CTest,
   a short headless training smoke run that creates new files, and a follow-up
   run restoring those files. No change to RLMission inference or weights.
