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

## Checkpoints

`resources/settings/Training/BluePatrol.json` supports:

- `training.rolloutSteps`: number of environment steps per PPO update
  (currently 2048).
- `training.maxUpdates`: maximum PPO updates (currently 2000 for standard non-curriculum training).
- `training.checkpointEvery`: deterministic evaluation interval (currently 10).
- `training.saveEveryUpdates`: independent checkpoint interval (1 for CI).
  When absent, it defaults to `checkpointEvery` for compatibility.

The learner writes these files at completed PPO-update boundaries:

- `resources/model/BluePatrol/latest/blue_patrol_generalized_v9.txt`:
  inference actor.
- `resources/model/BluePatrol/latest/blue_patrol_generalized_training_v6.txt`:
  **full training state**, using internal format `BLUE_TEAM_TRAINING_V6`.
- `resources/model/BluePatrol/prod/blue_patrol_generalized_v9.txt`:
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

GitHub-hosted runners have limited job duration, so the default time budget is
4 hours, allowing headroom for building and packaging. Workflow execution is
**manual**, not automatic; it must be started by the user. A new workflow with
`workflow_dispatch` normally becomes available in the Actions UI after it
exists on the default branch.

## Complete V3 curriculum from scratch

Select curriculum_enabled=true, curriculum_version=v3, resume_run_id empty,
max_seconds=14400 and save_every_updates=1. V3 starts random actor/critic/Adam
state at update zero, with observation v9 and 2048-step rollouts. It does not
migrate run 16 or any older actor. Twelve phases total 1200 PPO updates, from
1vs0 / 2vs0 / 3vs0 / 4vs0 to progressive combat and a final 4vs4. Rehearsal
uses 24 primary updates followed by 6 updates from earlier phases, round-robin.
The first phase has no earlier task; every phase ends with its primary task.

To continue this new run across jobs, select its own run ID with the same V3
JSON fingerprint and observation v9. Old V2/V3 progress is rejected. The runner
only resets current Blue files in its temporary checkout on fresh start;
archived candidate files and RLMission are untouched. The old training filename
ending in v5 is no longer loaded by the runtime. Format V6 remains the full
optimizer envelope, with an independently checked observation version of 9.

The 4-hour budget is a wall-clock limit, not a guarantee that all 1200 updates
finish in one job. Incomplete progressing blocks continue while time remains;
a complete checkpoint can resume later. Scenario changes occur between PPO
blocks/processes; earlier tasks are replayed throughout later phases.

See [blue-mappo-v3.md](blue-mappo-v3.md) for targets, rewards, physical lap
criteria, feasibility checks and diagnostics. No name V4 is used for this work.

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
