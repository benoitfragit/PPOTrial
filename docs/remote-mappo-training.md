# Windows Release remote MAPPO training

## Architecture

Blue MAPPO trains as the PPOKit project `Examples/BlueDefense/projects/BluePatrol`
under `ppokit-run` (environment DLL `blue_environment`, team network DLL
`ppokit_graph_network`), without changing the red production `RLMission` agent. The
former `PPO.exe` headless training mode has been removed.

- Blue team: RLTools generalized shared MAPPO actor and centralized critic.
- Red team: unchanged `RLMission` production weights (`policy.ppokit`, converted from the original model).
- Simulator: the BlueDefense ECS physics, world rules, observations, actions and
  episode resets of the former desktop mode, now stepped only by the
  environment DLL (the in-world execution modes were removed with `PPO.exe`;
  `Examples/BlueDefense/tests/world_settings_tests.cpp` checks that the DLL's
  world uses the time step and limits of the shared `Environment.json`).
- UI: `ppokit-run` never opens a window. The world is watched, when wanted,
  in a separate `ppokit-viewer` process fed over UDP by the project's
  Visualiseur `blue_visualiser` (port 47001 in the project's `network.json`).
- Toolchain: VS2022 x64 **Release**.

## Launch locally

```powershell
cmake -S . -B build/release -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release -DPPO_BUILD_DESKTOP=ON
cmake --build build/release --config Release --target ppokit-run blue_environment ppokit_graph_network blue_visualiser --parallel 2
& build/release/PPOKit/Release/ppokit-run.exe train --project Examples/BlueDefense/projects/BluePatrol --until-update 4000 --max-seconds 14400
```

`--max-seconds S` ends the training at the first update boundary after S
seconds, as if it were the last update (campaigns included). Production
inference is `ppokit-run play --project Examples/BlueDefense/projects/BluePatrol
--policy checkpoints/shared/prod/policy.ppokit --realtime`, watched with
`ppokit-viewer --view <build>/Examples/BlueDefense/Release/blue_visualiser.dll --port 47001`;
`ppokit-studio` launches the same commands from the project's `project.json`.
The CI does not call `ppokit-run train` directly: `tools/ci/mappo_curriculum.py`
runs it block by block (`--until-update`, `--save-every`, `--max-seconds`) and
promotes PROD after each block as the former `PPO.exe` Training mode did.

## Local visual training

Open `Examples/BlueDefense/projects/BluePatrol` in PPOKit Studio. The former
standalone local curriculum script has been retired.

## Checkpoints

`Examples/BlueDefense/projects/BluePatrol/blue_patrol_training.json` supports:

- `training.rolloutSteps`: number of environment steps per PPO update
  (currently 2048).
- `training.maxUpdates`: maximum PPO updates (currently 2000 for standard non-curriculum training).
- `training.checkpointEvery`: deterministic evaluation interval (currently 10).
- `training.saveEveryUpdates`: independent checkpoint interval (1 for CI).
  When absent, it defaults to `checkpointEvery` for compatibility.

The learner writes these files at completed PPO-update boundaries:

- `Examples/BlueDefense/projects/BluePatrol/checkpoints/shared/latest/policy.ppokit`:
  inference actor (PPOKit artifact, kind `policy`).
- `Examples/BlueDefense/projects/BluePatrol/checkpoints/shared/latest/latest.ppokit`:
  **full training state** (PPOKit artifact, kind `training`).
- `Examples/BlueDefense/projects/BluePatrol/checkpoints/shared/prod/policy.ppokit`:
  promoted actor only after successful evaluation, when present (score in
  `policy.ppokit.score`).

Archived V7/V10 or former backend models require explicit offline conversion
before a current workflow can use them. New runs restore common graph artifacts
and their associated generation state only. The selected run is never silently
replaced by the committed checkpoint.

Training-state serialization includes actor, critic, Adam moments,
optimizer scalars, RNG and completed PPO updates. Partial in-progress rollout
steps are **not** included; interruptions can lose the current rollout. Saving
is performed atomically at optimizer boundaries.

## GitHub Actions

Use **Actions > Blue Defense (Windows Release) > Run workflow** once the
workflow file has been merged into the repository's default branch.

Choose the scenario, `max_seconds`, and `save_every_updates`; optionally
supply a previous workflow `resume_run_id` to restore the
`blue-mappo-checkpoints` artifact before training.

The workflow checks out the requested commit, compiles `ppokit-run` and the
BlueDefense DLLs in Release (`-DPPO_BUILD_DESKTOP=OFF`), verifies the runtime
with a one-episode `ppokit-run play` of the committed production actor,
launches training and publishes downloadable artifacts. A fresh output
checkpoint is required for a successful training job. Artifact retention is
30 days; download useful states before expiry.

Each new telemetry artifact also contains `run-manifest.json`, with the run,
commit, workflow inputs and configuration SHA-256 hashes needed to audit
comparisons. `mappo-optimizer.csv` records per-update policy/value loss,
entropy, approximate KL, clipping fraction, explained variance, action standard
deviation, derivative activity and optimizer duration. Older artifacts without
these files remain available in the published telemetry artifacts.

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

- `Examples/BlueDefense/projects/BluePatrol/checkpoints/shared/latest/policy.ppokit`:
  the LATEST actor corresponding to the full training state.
- `Examples/BlueDefense/projects/BluePatrol/checkpoints/shared/latest/latest.ppokit`:
  critic, Adam, RNG and PPO update counter; this is the actual training seed.
- `Examples/BlueDefense/projects/BluePatrol/checkpoints/shared/prod/policy.ppokit`:
  separate production inference incumbent, preserved without treating its
  score as a curriculum-gate certification.

All three files must be committed to the selected branch. The bootstrap checks
the checkpoint contracts (observation 10, network sizes, guidance semantics),
actor architecture compatibility, positive optimizer
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

1. The CI configures without the desktop programs (`PPO_BUILD_DESKTOP=OFF`):
   it builds `ppokit-run` and the BlueDefense DLLs only, without graphics.
2. The GitHub runner builds the project for each workflow run; the dependency
   download may dominate a cold first run.
3. Checkpoint restore preserves optimizer state, but not an interrupted
   in-progress rollout or the full ECS episode state.
4. C++ unit/smoke tests and Windows Release CI are not evidence of RL
   convergence: evaluate across independent seeds and varying P/N scenarios.
5. Before merging, require successful VS2022 Release compilation, CTest,
   a short headless training smoke run that creates new files, and a follow-up
   run restoring those files. No change to RLMission inference or weights.
