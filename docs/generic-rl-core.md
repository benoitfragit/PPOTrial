# Generic RL core (incremental extraction from PPOTrial master)

## Status and dependency boundaries

This is a structural extraction. No neural network, PPO mathematics, simulation
equation, reward, scenario schema, saved model, or existing desktop control has
been changed. **Existing desktop training and production agents now use the
generic ILearner/IPolicy interfaces**, while the multi-agent simulation and
evaluator call generic step/campaign schedulers. EpisodeRunner remains the
single-agent entry point for new IEnvironment implementations; using it
directly for the existing shared multi-agent world would be incorrect.

| Target | Responsibility | Dependencies |
| --- | --- | --- |
| `GenericRLCore` | Environment/policy contracts, episode driver, world-step order, evaluator state POD | C++17 only |
| `GenericPPO` | Existing PPO implementation and RLAgent; optional TorchLearner bridge | LibTorch, GenericRLCore |
| `GenericRLBindings` | Legacy TrainingAgent and ProdAgent domain-model adapter templates | GenericPPO, GenericSerializer |
| `GenericEntities` | Existing simulation entities and components | EnTT, JSON, GenericRLCore |
| `GenericEnv` | Existing EnTT simulation worker and snapshots | entities, PPO runtime |
| `Evaluation` | Existing evaluator thread, fixed evaluation scenarios/scorer and promotion | entities, env, PPO |
| `Renderer`, `Main` | Existing desktop and orchestration | the above and graphics |

The `TrainingAgent<Policy, Model>` and `ProdAgent<Policy, Model>` templates
remain in `Bindings/data` as domain-specific adapters, **but their backend
invocations now use `ILearner` and `IPolicy` respectively**. Concrete
networks remain in `Agents/`; training networks retain `ITrainingNet`.
`TorchLearner` accepts any PPO-compatible `ITrainingNet`, not an arbitrary
neural module without action sampling/log-probability/value/entropy support.

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
Existing production/frozen policy network implementations remain unchanged;
`TorchProductionPolicy` adapts their existing load/forward interfaces.

To add a different PPO-compatible network, implement `ITrainingNet` and
pass its instance to `TorchLearner`. Do not edit the optimizer or environment.
For a new environment, implement `IEnvironment`; no EnTT or renderer headers
are required. The supplied EpisodeRunner is single-agent. For the existing N-agent desktop,
`ResetAgents`/`StepAgents` handle the shared-world action/advance/resolve/
feedback order, and production/training agents use the same generic policy
interfaces. Physics, destruction logic and scoring remain domain-owned.

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

This PR integrates generic policy adapters and orchestration into the
existing application rather than replacing that application. Scenario
definitions, entity dynamics, score computation, serializer callbacks and UI
remain domain-specific. The evaluator's public state snapshots are now published and read under a
mutex, preventing concurrent readers from accessing a buffer being rewritten.
The global Serializer callback lifetime still requires separate lifecycle
validation. The legacy PPO tensor mathematics and
checkpoint archive serialization are intentionally untouched.

The core-only test exercises reset/advance/done, feedback and four-phase
ordering. Windows CI compiles the desktop with the matching Debug/Release
LibTorch binary and checks saved-policy inference parity for the generic
production adapter. End-to-end scenario parity and training-rollout numerical
parity still require dedicated fixtures; unit tests alone cannot prove them.


## Runtime dependency direction

```text
New single-agent environments         Existing shared multi-agent world
       IEnvironment                         Env / Evaluation
            |                                      |
       EpisodeRunner                        StepAgents / RunEvaluation
            |                                      |
         ILearner                           Bindings: domain adapters
            |                                      |
       TorchLearner                     +----------+------------+
            |                           |                       |
       RLAgent -> PPO                 ILearner                 IPolicy
            |                           |                       |
       ITrainingNet                TorchLearner         TorchProductionPolicy
                                        |                       |
                                   RLAgent -> PPO       Existing Policy::forward
                                        |                       |
                                   ITrainingNet         Existing model.pt
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

## Windows Debug / Release ABI compatibility

The prebuilt Windows LibTorch 2.9.1 **Debug** and **Release** archives use
different MSVC runtime-library variants. Do not compile or execute Release
checkpoint inference against the Debug archive (or the reverse). Each VS2022
configuration requires a separate CMake build directory:

```sh
cmake -S . -B build/debug -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Debug -DPPO_LIBTORCH_VARIANT=Debug
cmake --build build/debug --config Debug
ctest --test-dir build/debug -C Debug --output-on-failure

cmake -S . -B build/release -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release -DPPO_LIBTORCH_VARIANT=Release
cmake --build build/release --config Release
ctest --test-dir build/release -C Release --output-on-failure
```

The Debug binary remains the default, matching the existing project. The optional checkpoint tests are disabled by default to preserve the legacy
Visual Studio multi-configuration setup. Enable them explicitly with
`-DPPO_BUILD_CHECKPOINT_TESTS=ON` in the matching single-configuration build
folders shown above. When checkpoint tests are enabled, CMake rejects mixed
multi-configuration builds
because they can crash even when compilation succeeds. The CI now validates
both configurations against the corresponding LibTorch archives. No model
architecture, weights or checkpoint format is changed.

## Existing training agents use the generic learner

The existing `TrainingAgent<Policy, Model>` now delegates action selection,
rollout transitions, metrics and checkpoint saving to `TorchLearner`
(`ppo::core::ILearner`). The domain-specific model still computes the
observation, action interpretation, reward and episode termination exactly as
before. Fixed-width state and action arrays are copied through the standard
`ppo::core::Observation`/`Action` contract, keeping their element order and
ownership explicit. RLAgent and PPO mathematics remain unchanged, and
TorchLearner still uses the existing training policy and checkpoint format.

## Existing production agents use IPolicy

`ProdAgent<Policy, Model>` now invokes
`TorchProductionPolicy<Policy> : ppo::core::IPolicy` rather than calling the
LibTorch wrapper from inside the domain agent. The adapter owns the tensor
view used for inference and calls the **existing** production network
`load()`/`forward()` methods. The original fixed-size observation and
action layouts and missing-checkpoint behavior are preserved.

The full Windows checkpoint tests compare element-by-element the generic
`IPolicy` result against the original production network for each of the
existing `prod` and `eval` model archives, both architectures, in Debug and
Release. No training or modification of model files is performed.

## Reject incomplete evaluation campaigns

`RunEvaluation` now returns an `EvaluationResult` indicating whether every
configured test was processed without cancellation or reset failure. The
legacy evaluator retains its unchanged score equation and evaluator-thread
callback order, but **never copies `eval/model.pt` to `prod/model.pt`** if
the campaign was interrupted or incomplete. A canceled test still receives
the historical completion callback and progress accounting; no partial score
may promote a candidate checkpoint.

## Runtime contract extracted from the PPO engine

The existing C++17 agent interface, dashboard snapshot types, evaluator
interface and PPO parameter POD layout live in `Runtime/include/ppo/runtime`.
Legacy `PPO/data/iagent.hpp` and `ppoparameters.hpp` remain forwarding
headers, so existing source includes and saved settings are unchanged.

`GenericEnv` and `Evaluation` now depend on the header-only
`LegacyAgentAPI` target rather than on the LibTorch-based `GenericPPO`
target. The dependency direction is:

```text
GenericRLCore <- LegacyAgentAPI <- GenericEnv / Evaluation
       ^                ^
       |                |
       +----------- GenericPPO (LibTorch)
                         ^
                         |
                GenericRLBindings -> concrete agents
```

The agent-state snapshot fields are retained for backwards compatibility with
the desktop UI. They do not make the generic simulation scheduler depend on PPO.

The runtime compatibility contract itself is compiled and exercised in the
headless C++17 core CI (Linux and Windows). This prevents reintroducing an
accidental dependency on Torch, EnTT or the desktop graphics libraries.

## Pre-update PPO compatibility smoke test

`PolicyCheckpointTests` also constructs a legacy `RLAgent` and a new
`TorchLearner` with separate policy instances restored from each existing
production archive. It compares the raw sampled actions after resetting
PyTorch RNG to the same seed, feeds each the same next observation, reward and
terminal flag, then compares rollout step progression. A deliberately short,
incomplete rollout **never calls `UpdatePolicy()` or `Save()`** and cannot
modify the existing checkpoint files. This detects regressions in the
real training-path adapter (not just deterministic inference). It does
not yet establish long-horizon numerical parity of a full PPO update.


## Agent descriptors (non-breaking C++17 extension)

`ppo/core/agent_contract.hpp` describes the semantic identity and widths of
neural observation/action contracts. The existing RLSNA and RLMission models
demonstrate the API while continuing to use their original PPO training,
checkpoint and evaluation paths. See [agent-contracts.md](agent-contracts.md).
