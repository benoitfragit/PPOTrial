# MAPPO — dedicated evaluation World (architecture contract)

Status: **design / not implemented**. This document does not claim a background evaluator is already active.

## Verified current behavior

`GeneralizedPatrolMappoAgent::SetNewState()` executes the first `OptimizeRollout()` synchronously on the ECS thread when the rollout of 2048 team steps fills. In the original implementation, it then enters the deterministic evaluation mode on that same ECS, which suspends training. The former `Examples/Airspace/Evaluation/evaluator.cpp` already has its own `entt::registry`, `std::thread` and snapshot publication. Its existing scenario format cannot simply be applied to the BlueDefense P/N setup.

The telemetry branch restores the configured deterministic evaluation **cadence** (10, 20, etc., plus final update), adds optimizer profiling and reports to GitHub Actions. **It does not implement an isolated evaluation World** and therefore does not fix a true optimizer stall.

## Target ownership

```text
Main / BlueDefense
  TrainingWorld : Environment + BlueDefenseWorldRules
    Red[i] : production TargetSelectorProdAgent / RLMission (unchanged)
    Blue team : GeneralizedPatrolMappoAgent (shared actor + critic + Adam)
      Collect 2048 team steps -> update actor and critic
      Snapshot frozen blue actor parameters (update id, P/N, seed)
                        |
                 immutable message
                        v
  EvaluationWorker : thread
    EvaluationWorld : independent Environment / ECS registry
      Red[i] : the existing production RLMission controller (unchanged)
      Blue team : inference-only frozen generalized V8 actor, no optimizer/critic
      Reproduce 8 scenario cases, up to 4096 steps each
      Publish progress and final evaluation result via mutex/atomic snapshot
                        |
                        v
    TrainingWorld reads final result without blocking; promotes the exact
    evaluated policy snapshot if it improves the metric.
```

## Implementation requirements

1. **World isolation.** Avoid a second `Environment` instance overwriting the process-wide `Serializer` singleton registrations. Evaluation construction must use an isolated, read-only configuration path and must not mutate the existing training instance.
2. **Actor-only runtime.** Construct a standalone generalized actor and reusable inference buffers. Load V8 actor weights once using `ReadGeneralizedPolicy`. No `GeneralizedPatrolLearner`, critic, GAE, Adam optimizer, rollout storage or sampling RNG in the evaluation agent. Apply deterministic actor means, preserve action clamp and LiDAR masks.
3. **Real scenario.** Use exactly the production ECS physics, BlueDefenseWorldRules, randomized geometry and scenario parameters. Build separate entity and agent objects; never share `entt::registry`, pointers, `RLMission` agent instances or live neural buffers. **Do not edit RLMission.**
4. **Immutable snapshot.** On evaluation boundary, freeze the trained actor before resuming collection. Tie the result to the update id and checkpoint source, and promote only this frozen actor when the full campaign is complete and better than the previous best.
5. **Thread synchronization.** Evaluation worker manages its own lifetime with RAII; cancellation and thread join are safe during shutdown. Publish progress and results through immutable message copies protected by a mutex. No ECS or Torch/actor accesses from the GUI thread. Limit to one active evaluation and use a documented bounded queue policy for later checkpoints.
6. **Termination and determinism.** Respect P=1..9 and N=0..10, destroyed teammates and leader outcome, delayed red spawns, restrictions, masks, episode terminal/truncation and independent seeds. Do not change rewards or the PPO algorithm.
7. **Visibility.** UI and headless snapshots show update id, evaluation case / progress, elapsed time and last/best scores. Training continues during the evaluation (subject only to CPU contention). Logger and GitHub Actions distinguish `Collecting`, `Optimizing`, `Evaluating`, `Saving` and `Failed`.
8. **Tests.** Deterministic inference parity vs training actor, no optimizer present in the evaluation runtime, concurrent World independence (including RLMission), cancellation/join, snapshot progress, score provenance and actor promotion, checkpoint restoration and P/N corner cases. Build Debug + Release and run CTest on Windows before approving a PR.

## Current branch scope and limitations

Implemented in `feature/mappo-telemetry-evaluation-20261030`:
- Correct GUI update counter `m_version`; preserve the actor checkpoint observation version.
- Evaluate at `checkpointEvery` completed updates or at the last update, not unconditionally at update 1.
- Log PPO optimizer phase / epoch durations without modifying its mathematics.
- Headless `[MAPPO_STATUS]` heartbeat and detailed CSV (including scores and rewards).
- Github Actions CSV, SVG curves, summary and Python telemetry regression tests.
- Additional C++ tests for evaluation cadence and read-only deterministic actions.

Not implemented: independently threaded BlueDefense evaluation World; score generation remains in the training ECS until that change is complete. No C++ build or CI test has yet been executed in this environment. **Do not merge as fulfillment of the threaded-evaluator requirement.**

Do not modify legacy RLMission files, policy weights or behavior.
