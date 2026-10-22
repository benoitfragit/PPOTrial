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
not reinterpret done into other termination semantics. `EpisodeRunner` calls the injected policy, steps the environment and
notifies the learner **automatically when constructed with an ILearner**.
The overload taking a plain IPolicy is inference-only.

For the existing LibTorch PPO backend, create a network implementing the
unchanged `ppo::model::ITrainingNet`; pass it to
`ppo::model::TorchLearner(policy, modelpath, obs_size, action_size, rollout_size,
parameters, load_existing_model)`. Pass this learner directly to `ppo::core::EpisodeRunner(environment, learner)`;
the learner receives every transition, without a separate feedback argument.
Call `ResetRollout()` after a completed rollout before collecting another
(or explicitly to discard an invalid/incomplete rollout);
this resets only the collection buffers, not the environment or optimizer.
The client decides when to reset the episode; there is no auto-reset.
A failed world step or malformed next observation requires Reset before
advancing again; if training, discard the inconsistent rollout explicitly.

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
serialization callbacks and UI remain domain-specific. The evaluator's public state snapshots are now published and read under a
mutex, preventing concurrent readers from accessing a buffer being rewritten.
The global Serializer callback lifetime still requires separate lifecycle
validation. The legacy PPO tensor mathematics and
checkpoint archive serialization are intentionally untouched.

The core-only test exercises reset/advance/done, feedback and four-phase
ordering; it does not demonstrate compiled Windows desktop compatibility or
numerical policy parity. Such validation requires matching LibTorch and full
simulation test fixtures, and must be recorded separately.


## Runtime dependency direction

```text
New environments                  Existing EnTT application
  IEnvironment                     Env / Evaluation / Renderer
       |                                  |
  EpisodeRunner                       ECS adapter
       |                                  |
   ILearner                  Generic StepAgents / RunEvaluation
       |                                  |
  TorchLearner                   Legacy agents, evaluator score
       |
  RLAgent -> PPO -> injected ITrainingNet
```

The generic contracts schedule operations and own no model-specific state,
observation layouts, entity list, score or checkpoint files. The application
owns initialization order, fixed `dt`, entity and agent lifetimes, settings,
renderer and worker threads.

## Multi-agent runtime and evaluation orchestration

`ppo/core/multi_agent.hpp` runs any nullable pointer-like agent collection
without referring to EnTT, torch or network dimensions. `ResetAgents`
maintains the caller's order; `StepAgents` executes all decisions, advances the
world once, resolves outcomes once, and delivers feedback to every agent. A
rollout-reset request from one agent does not skip feedback for later agents.
It is used by the original environment and evaluator.

`ppo/core/evaluation.hpp` provides the synchronous `RunEvaluation` campaign
scheduler. It manages test/step iteration, cancellation, early termination and
per-test completion, leaving scenario initialization, the number of agents,
domain dynamics, mission score and checkpoint promotion with the evaluator.
Callbacks run on the *existing* evaluator thread, with unchanged steps and
zero-based indices. No new thread, RNG, checkpoint or training run is created.

The ECS adapter `Entities/data/entityruntime.hpp` advances only live entities
and retains their existing iteration order. The live environment's vector
and evaluator's ordered map remain unchanged. Their intentionally distinct
destruction rules are not merged.

## Read-only compatibility smoke test

`Tests/model_checkpoint_tests.cpp` loads both existing production archives
(`RLSNA` 6 inputs/2 outputs and `RLMission` 152 inputs/3 outputs) with the
unchanged production wrappers. It checks inference shape, finite values and
repeatability without optimizer updates, serializing anything or writing to
the model directory. `PolicyCheckpointTests` is registered with desktop CTest
on Windows alongside the existing headless core tests.

The original checkpoint files remain untouched. The smoke test does not
establish training-time numeric parity or replace functional scenario tests.
