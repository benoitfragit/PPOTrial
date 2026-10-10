# Navigation2D functional parity against the old PPOTrial application

This is a **feature-preservation contract**, not a backwards-compatibility
promise for old tensor weights or the original aircraft/SAM vocabulary.
A migration to RLtools must not start by discarding still-required application
capabilities.

The old `master` tree is the source of truth for functional comparisons.
Replacements must use generic mobile / circular-zone / work-area terms.

| Capability | Former implementation | Current Navigation2D status | Acceptance |
| --- | --- | --- | --- |
| N EnTT entities, multiple controlled mobiles | `Env`, `Entities` | **Restored** (original refactor PR #13) | Stable IDs, separate feedback, tested |
| Chase moving mobile, Prod N-1 as independently controlled target | `Interception` | **Restored** | Same-world transition, frozen independent policies |
| Circular forbidden zones | `Entities`, mission | **Restored** | Continuous collision detection |
| 61-ray 360° LiDAR, work-area edges, zone obstacles | `Agents/mission/src/targetselector.cpp` | **Restored** in PR #14 | Raw and normalized distances, ray display, headless tests |
| Graded proximity avoidance for both zone and work-area | `RewardFunction` | **Adapted** in PR #14 | Generic penalties; different coefficients are documented |
| Random initial mobile states | `RandomInitializerSystem` | **Adapted** in PR #15 | Seeded reproducible resets, tests |
| Full-world deterministic evaluation initial-state buffers | `CheckpointInitializerSystem`, `Evaluator` | **Adapted** in PR #15 | Replayed same-topology .nav cases, evaluation provenance |
| Motion endurance: minimum speed and speed/turn fuel use | `FlightModelSystem` | **Adapted** in PR #15 | Optional per-mobile energy config, terminal reason, tests |
| Editable, individually weighted reward channels | `TrainingAgent`, old renderer | **Adapted** in PR #15 | Per-mobile editor and .nav settings, per-channel info |
| PPO hyperparameter configuration | `TrainingAgent`, old JSON | **Adapted** in PR #15 | GUI parameters passed to new backend learner; no live optimizer edit |
| Simultaneous pre-step controller decisions | `Env::Run` | **Restored** | Common fractional world event and feedback |
| Worker simulation thread with concurrent GUI | `Env/src/environment.cpp` | **Not restored** | Worker owns mutable ECS, UI reads immutable snapshots |
| Double-buffer EnTT snapshots & state/telemetry event publication | `Env::CreateSnapshot` | **Not restored** | No unprotected cross-thread registry access |
| Independent background evaluator with live progress | `Evaluation/src/evaluator.cpp` | **Not restored** | Cancellation/PAUSE safe; training unaffected |
| Per-track attention/pooled variable-N neural network | `Agents/mission/actorcriticmission.cpp` | **Not restored** | Configurable model selection and padded/masked track observations |
| Perception of unrelated moving mobiles | mission `SetNewState` | **Not restored** | Sensor/track slots beyond designated destination |
| Dynamic world limits and time step through desktop controls | `Env::SetMaxRange`, `SetTimeStep` | **Adapted** in PR #15 | Atomic UI-thread updates, safe validation, preserved controller instances |
| Optional 3D mobile assets | old `Renderer` | **Adapted** in PR #15 | GLB artwork packaged with a procedural glyph fallback |
| Destroyed markers, explosions and specialized visualization | old `Renderer` | **Not restored** | Generic visual events and no regression in tactical display |
| Full reflection / runtime component editor | old `Meta`, `Renderer` | **Not restored** | Schema/typed components inspection and safe editing |
| Declarative JSON scene platforms / agent factory configuration | `Meta`, `Serializer`, `Env/scenarioconfig.cpp` | **Not restored** | New Navigation2D schema, no old names required |
| Original model checkpoint compatibility | old `.pt` | **Explicitly out of scope** | User accepts retraining and loss of historical weights |

**Do not mark functional migration complete, merge parity changes as
fully equivalent, or remove this checklist while the rows above are
unresolved.** The tests validate individual replacements, not end-to-end
behavioral equivalence. The renderer and asynchronous lifecycle require a
Windows desktop smoke test, in addition to GitHub Actions compilation.
