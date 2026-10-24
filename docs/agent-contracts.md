# PPOTrial — Generic agent contracts (C++17)

## Scope and non-regression guarantee

This incremental step describes **existing neural interfaces** without changing
the agent runtime, learned weights, PPO, rollout buffering, rewards, save/load,
evaluation, EnTT world, renderer, or the Play/Pause/Stop flow.

- Existing `RLSNA` and `RLMission` agents are **examples** of this contract,
  not special cases encoded in `GenericRLCore`.
- Both still train when Play is selected, through their unchanged
  `TrainingAgent -> TorchLearner -> RLAgent -> PPO` path.
- Production and evaluation continue to load their existing `model.pt` files
  through the same `ProdAgent -> TorchProductionPolicy` path.
- No existing JSON field or scenario key is introduced, renamed or removed.
- No neural network module name, feature order, normalization, action meaning,
  reward value, random sequence or checkpoint format is modified.

## Contract API

`Core/include/ppo/core/agent_contract.hpp` defines a small value type:

```cpp
struct AgentContract {
    std::string_view profile;
    std::string_view observationLayout;
    std::size_t observationSize;
    std::string_view actionLayout;
    std::size_t actionSize;
};
```

`MakeAgentContract<ObservationVector, ActionVector>()` derives dimensions from
C++ fixed-size observation and action vector types. `IsValid()` validates the
descriptor and `SameContract()` requires profile, semantic layout IDs and
dimensions to match exactly. Matching dimensions **alone** never establish
checkpoint compatibility.

The types only depend on C++17. There is no EnTT, LibTorch or graphics
dependency. Semantic layout IDs are application-owned; the core never
enumerates specific agent types. Contracts are **descriptive**, not a runtime
file format or an implicit migration tool.

## Two existing examples

| Profile | Observations | Actions | Semantic IDs |
| --- | ---: | ---: | --- |
| `RLSNA` | 6 | 2 | `rlsna.observation.v1`, `rlsna.action.v1` |
| `RLMission` | 152 | 3 | `rlmission.observation.v1`, `rlmission.action.v1` |

The historical RLMission input consists of 10 slots of 9 track fields,
followed by 1 scalar and 61 pseudo-LiDAR samples. The action vector comprises
two continuous control values and a third binary policy output; currently,
the third output is not applied by `StatesAndActionsForMission::SetNextAction`.
These semantics are not revised here.

The two domain adapters declare their own `kAgentContract`. Both training and
production wrappers expose the same `static constexpr Contract()` accessor.
This is a static method, not an addition to `IAgent`: no virtual table change
and no behavior change to existing agent instances.

## How to add a third agent

1. Define your own model adapter with `STATES` and `ACTIONS` vector aliases.
2. Declare `kAgentContract = MakeAgentContract<STATES, ACTIONS>(...)` with
   stable semantic layout IDs. The domain module owns those IDs.
3. Implement observation packing and action interpretation in that domain
   adapter. Add a suitable network and register its agent factory through
   existing application registration mechanisms.
4. Build core tests and full Windows desktop/checkpoint tests. Add tests for
   the new model's feature semantics and checkpoint compatibility.

A changed observation meaning, order, normalization, mask policy or action
meaning **requires a new layout ID**, even when dimensions are unchanged.
A new layout ID does not transform old weights into a new model; the old
policy stays bound to its original layout.

## Tests and validation limits

- `GenericRLCoreTests`: new independent example, valid/invalid descriptors,
  same-width but incompatible semantic layout IDs, mismatched dimensions.
- `PolicyCheckpointTests`: compile-time checks of both historical agent
  contracts and train/prod consistency; existing read-only checkpoint loading,
  deterministic inference equivalence and pre-update transition parity
  remain unchanged.
- `Desktop VS2022 build`: Debug and Release CI runs CMake and CTest against
  matching LibTorch variants.

These tests do **not** prove Play-to-save end-to-end training equivalence.
That remains a separate required manual/integration acceptance test before
promoting future changes that touch learner behavior. This PR intentionally
does not touch that behavior.

## Next passes (not included)

An agent-agnostic factory/catalog can consume these descriptors for discovery
and compatibility diagnostics. A later multi-agent experience contract can
add team identity, termination/truncation and rollout metadata for MAPPO.
Those additions must keep the existing legacy schemas and learners intact.
