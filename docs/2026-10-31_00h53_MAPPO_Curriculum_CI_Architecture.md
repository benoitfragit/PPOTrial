# 2026-10-31 — Single-workflow MAPPO curriculum without per-level rebuilds

## Scope and architecture

This PR changes **CI orchestration only**. It preserves the generalized RLTools
MAPPO actor/critic, ECS physics, generalized rewards, masks, training checkpoint
format, and the unchanged red `RLMission` production controller.

The V6 full training checkpoint stores policy, critic, both Adam states, RNG and
cumulative PPO update counter. The generalized model already supports P=1..9
and N=0..10 across **process-level** checkpoint resume. Since BlueDefense
creates concrete agents when a scenario is loaded, changing P/N in-place while
an ECS episode is running would require a much more invasive runtime change.

We instead launch the **same already-built** `PPO.exe` repeatedly:

1. Load `resources/settings/Training/BluePatrolCurriculum.json`.
2. Change `Scenarios/BlueDefense.json` P/N and hostile-zone activation for the
   next level; change the **cumulative** `training.maxUpdates`.
3. Run `PPO.exe --no-gui --example blue-defense --max-seconds ...`.
4. Verify a fresh V6 checkpoint with an increased completed-update count.
5. Advance only after reaching the planned update target; otherwise stop
   cleanly for a subsequent workflow run with `resume_run_id`.
6. Restore the original checkout JSON files even on exceptions.

No C++ rebuild or reinitialization of network parameters occurs between
stages. **The GitHub Actions job itself still builds Release once at its start**,
as before; no binary cache is claimed.

## Training levels

| Level | P | N | Hostile zones | Additional PPO updates |
| --- | ---: | ---: | --- | ---: |
| Hippodrome | 1 | 0 | Off | 16 |
| Hippodrome and wedge | 2 | 0 | Off | 20 |
| One-vs-one | 1 | 1 | Off | 24 |
| One-vs-two | 1 | 2 | Off | 24 |
| Single Blue with zones | 1 | 0 | On | 20 |
| Two-vs-one and zones | 2 | 1 | On | 24 |
| Two-vs-two and zones | 2 | 2 | On | 28 |
| Three-vs-three and zones | 3 | 3 | On | 32 |
| Full reference | 4 | 4 | On | 48 |

Total: **236 additional PPO updates** beyond the restored checkpoint, if any.
These counts are configuration values rather than compiled constants.

This first implementation uses **fixed update budgets**, not automatic
learning-quality thresholds. Per-level evaluation is performed by the
existing C++ evaluator, and the final 4v4 stage restores the original
4v4 reference difficulty. **An independent, evaluation-only 4v4 benchmark
between every stage is not implemented**; this distinction matters when
interpreting learning curves.

## Safety and evaluation

- The 4v4 production actor and evaluation score are preserved while any
  non-4v4 stage runs; its best policy is instead copied into a stage-specific
  telemetry directory. A high score from a 1v0 level cannot overwrite a
  previously achieved 4v4 production score.
- The final 4v4 stage uses the original 4v4 production actor/score as the
  baseline. Promotion remains the existing C++ evaluation policy.
- Different P/N evaluation scores must not be put on the same performance
  curve as if they were comparable.
- The CI `MAPPO_ERROR` log marker and nonzero child process exit are fatal.
- Actor and complete training checkpoints remain at their existing paths;
  `curriculum-progress.json` records the curriculum hash, starting PPO
  update and next level.
- Each stage saves `training.log`, `telemetry.csv` and optional promoted
  actor/score under `build/mappo-telemetry/curriculum/<level>/`.
- Failed or interrupted jobs retain existing artifact publication.

## Launch via GitHub Actions

After this PR is merged, open **Actions → MAPPO training (Windows Release) →
Run workflow**, on the updated `master` branch.

- `curriculum_enabled`: **true**.
- `max_seconds`: **14400** recommended; permits continuing with another
  run if the time budget ends before 4v4.
- `save_every_updates`: **1**.
- `resume_run_id`: leave empty for a new curriculum; use the run ID of an
  earlier **curriculum** run to continue its current level. A compatible
  prior ordinary training V6 checkpoint can also seed a new curriculum but
  changes the experiment's initial conditions.
- The `scenario` field is used only by the original non-curriculum mode.
  Curriculum always drives the generalized `blue-defense` example.

When a level cannot finish within the remaining budget, it is **not**
marked complete. Restore the `blue-mappo-checkpoints` artifact from that
run to resume it; the same curriculum JSON must be used.

## Tests and limitations

`python -m unittest discover -s tools/ci -p "test_mappo_curriculum.py" -v`

Tests cover schema and capacity validation, reading the authoritative V6 PPO
counter, exact JSON stage overrides, 4v4 production isolation, continuation
of a partially completed level, and cleanup on a failed stage. The existing
MAPPO telemetry tests also run.

The GitHub workflow still compiles and runs a scenario startup check once
per job. No additional native-code build or change to RLMission is introduced.
A full Windows runner smoke-test and independent-seed policy assessment are
required before interpreting curriculum quality as improved convergence.
