# Generic RL core (incremental extraction from PPOTrial master)

## Status and dependency boundaries

This is a structural extraction. No neural network, PPO mathematics, simulation
equation, reward, scenario schema, saved model, or existing desktop control has
been changed. Existing training/evaluation keeps using the legacy runtime.
The new contracts offer a separate entry point for new environments; this
does **not** claim that the existing desktop has been migrated to EpisodeRunner.

| Target | Responsibility | Dependencies |
| --- | --- | --- |
| `GenericRLCore` | Environment/policy contracts, episode driver, world-step order, evaluator state POD | C++17 only |
| `GenericPPO` | Existing PPO implementation and RLAgent; optional TorchLearner bridge | LibTorch, GenericRLCore |
| `GenericRLBindings` | Legacy TrainingAgent and ProdAgent domain-model adapter templates | GenericPPO, GenericSerializer |
| `GenericEntities` | Existing simulation entities and components | EnTT, JSON, GenericRLCore |
| `GenericEnv` | Existing EnTT simulation worker and snapshots | entities, PPO runtime |
| `Evaluation` | Existing evaluator thread, fixed evaluation scenarios/scorer and promotion | entities, env, PPO |
| `Renderer`, `Main` | Existing desktop and orchestration | the above and graphics |

The old `TrainingAgent<Policy, Model>` and `ProdAgent<Policy, Model>`
templates retain their implementations in `Bindings/data` (only explicit standard-library includes were added). They are not part
of the PPO engine library. Concrete networks remain in `Agents/` and are
supplied through the existing `ITrainingNet` contract. The new `TorchLearner`
accepts **any PPO-compatible `ITrainingNet`**, not an arbitrary neural
module without action sampling / log-probability / value/entropy support.

## New environment integration (C++17)

Implement `ppo::core::IEnvironment` with owned flattened float observations,
actions, `Reset`, and `Step`. The environment supplies the next observation,
reward and existing-compatible `done` flag. The contract intentionally does
not reinterpret done into other termination semantics. `EpisodeRunner` calls
the injected policy, steps the environment and optionally notifies the learner.

For the existing LibTorch PPO backend, create a network implementing the
unchanged `ppo::model::ITrainingNet`; pass it to
`ppo::model::TorchLearner(policy, modelpath, obs_size, action_size, rollout_size,
parameters, load_existing_model)`. Use this learner as the acting policy
and as the optional learner argument of `ppo::core::EpisodeRunner`.
Call `ResetRollout()` after a completed rollout before collecting another;
this resets only the collection buffers, not the environment or optimizer.
The client decides when to reset the episode; there is no auto-reset.

`TorchLearner` owns the memory behind the existing RLAgent
`torch::from_blob` view. It cannot be moved or copied. It forwards the raw
network action and the next observation/reward/done unchanged to RLAgent,
including legacy GAE bootstrap semantics and Torch random state consumption.
Existing production/frozen policy wrappers remain unchanged.

To add a different PPO-compatible network, implement `ITrainingNet` and
pass its instance to `TorchLearner`. Do not edit the optimizer or environment.
For a new environment, implement `IEnvironment`; no EnTT or renderer headers
are required. The supplied EpisodeRunner is single-agent; the existing desktop
continues to support N independently controlled entities. The templated
`StepWorld` order primitive is used by both existing simulation and evaluator
loops, but physics, destruction logic and scoring remain domain-owned.

## Build and tests

Run dependency-free core contract tests:

```sh
cmake -S . -B build/core -DPPO_BUILD_DESKTOP=OFF -DPPO_BUILD_CORE_TESTS=ON
cmake --build build/core --config Release
ctest --test-dir build/core -C Release --output-on-failure
```

Desktop defaults to ON and retains existing LibTorch 2.9.1 CPU and graphic
dependencies. It requires the previously supported Windows toolchain. This
change does not add a second application.

## Limitations and future independent work

This PR is an **incremental** architectural extraction, not a completed
replacement of the legacy application. Existing evaluation orchestration,
serialization callbacks and UI remain domain-specific. The prior evaluator
double-buffer lifetime concern and global Serializer callback lifetime still
require separate verified fixes. The legacy PPO tensor mathematics and
checkpoint archive serialization are intentionally untouched.

The core-only test exercises reset/advance/done, feedback and four-phase
ordering; it does not demonstrate compiled Windows desktop compatibility or
numerical policy parity. Such validation requires matching LibTorch and full
simulation test fixtures, and must be recorded separately.
