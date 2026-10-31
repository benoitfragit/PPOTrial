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

Use `--example blue-patrol` for N=0. Only these two generalized MAPPO
scenarios are accepted in headless mode. The ordinary desktop behavior remains
unchanged. The mode stops at its time limit or when `maxUpdates` is reached,
and returns an error when no resumable training-state checkpoint exists.

## Checkpoints

`resources/settings/Training/BluePatrol.json` supports:

- `training.rolloutSteps`: number of environment steps per PPO update
  (currently 2048).
- `training.maxUpdates`: maximum PPO updates (currently 1000 for standard non-curriculum training).
- `training.checkpointEvery`: deterministic evaluation interval (currently 10).
- `training.saveEveryUpdates`: independent checkpoint interval (1 for CI).
  When absent, it defaults to `checkpointEvery` for compatibility.

The learner writes these files at completed PPO-update boundaries:

- `resources/model/BluePatrol/latest/blue_patrol_generalized_v8.txt`:
  inference actor.
- `resources/model/BluePatrol/latest/blue_patrol_generalized_training_v5.txt`:
  **full training state**, using internal format `BLUE_TEAM_TRAINING_V6`.
- `resources/model/BluePatrol/prod/blue_patrol_generalized_v8.txt`:
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

## Versioned curricula (V1 / V2)

The manual training workflow has a `curriculum_enabled` switch and a
`curriculum_version` choice. Select **V2** to run the new independent
curriculum; leaving `resume_run_id` empty starts a **fresh** V2 training
session. V1 remains selectable to resume historical sequential sessions.

- **V1:** `BluePatrolCurriculum.json`, 236 update quotas, nine sequential
  phases. Its configuration and progress fingerprint are unchanged.
- **V2:** `BluePatrolCurriculumV2.json`, **1000 update quotas**, the same
  nine scenario families, with rehearsal of earlier phases starting at the
  second phase. The budget allocation is 32, 48, 72, 72, 64, 88, 100, 124,
  and **400 updates in 4v4**.
- **Mixing:** the V2 orchestrator uses 24 PPO updates on the current stage,
  then 6 on a *previously completed* stage, cycling through all earlier
  stages. The first phase has nothing to replay. The final 4v4 remains
  dominant while revisiting all eight older stage families.
- **Resume:** a V2 checkpoint can only resume with its matching V2
  `curriculum-progress.json` fingerprint and full V6 training state.
  A V1 run artifact (including run 11) must **not** be passed directly
  as a V2 resume checkpoint; start V2 fresh instead. V1 progress is
  still accepted in V1 mode.
- **No recompilation between stages:** the orchestration switches JSON
  scenario parameters across headless process launches. The existing
  GitHub workflow still builds `PPO.exe` **once per GitHub job**.
- **Time:** `max_seconds=14400` is a 4-hour budget, **not a promise**
  that all 1000 updates finish in one CI run. Download the V2 checkpoint
  and resume explicitly across jobs if necessary.

Rehearsal occurs **between blocks of PPO updates**, not by sampling a
different scenario on every environment episode. Per-episode mixing would
require modifying the simulator's reset/curriculum integration.

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
