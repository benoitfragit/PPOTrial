# Generic environment contract: first independent increment

## Working brief

Work on `benoitfragit/PPOTrial`, on a dedicated branch followed by an unmerged
PR. Read the existing implementation before changing it. Build a reusable C++17
RL foundation incrementally, retaining existing numerical algorithms. Describe
the actual behavior of components rather than inferring capabilities from names.
Keep domain types, rendering, ECS and transport details out of public RL contracts.
Use independent, neutral examples to validate the generic foundation. Do not run
model training or change reward functions. Adapt the development rules from
MFDStudio. Report delivered scope, validation and remaining work precisely.

This increment implements the environment contract only. It neither migrates nor
connects the existing simulation/training chain. It is not the completed framework
refactoring originally described.

## Findings from the existing code

Inspected `PPO/data/ppo.hpp`, `rlagent.hpp`, `iagent.hpp`, `PPO/CMakeLists.txt`,
the root CMake file and `Env/data/environment.hpp` at
`a1d1ed0f8e3e1802545c076aa8c3bea284940221`.

| Existing boundary | Observed coupling |
| --- | --- |
| `PPO/CMakeLists.txt` | `GenericPPO` publicly links `GenericEntities` |
| `PPO/data/iagent.hpp` | Includes `components.hpp`; mixes PPO parameters, UI snapshots and `IEvaluator` returning simulator state |
| `RLAgent` public API | Accepts caller-owned float buffers and a single `done` flag |
| `Environment` public API | Owns entities and agents, exposes asynchronous lifecycle and EnTT snapshots |
| Root CMake | Loads Windows LibTorch and graphical dependencies unconditionally |

These are source observations, not evidence of runtime regression. The bodies of
PPO optimization and rollout handling were not audited in this increment.

## New independent boundary

`RL/` builds `ppo::environment` with only the C++ standard library. Consumers
implement `ppo::rl::IEnvironment`; the interface includes no policy, tensor,
simulation, rendering or transport type. It can later be consumed by different
algorithms. The legacy root build is unchanged. Build this module independently:

```sh
cmake -S RL -B build/rl
cmake --build build/rl
ctest --test-dir build/rl --output-on-failure
```

Without CMake, the same test suite can be compiled directly:

```sh
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -I RL/include \
    RL/src/environment.cpp RL/tests/environment_tests.cpp -o /tmp/ppo-contract-tests
/tmp/ppo-contract-tests
```

The lifecycle follows the conceptual [Gymnasium Env API](https://gymnasium.farama.org/api/env/):
reset returns an observation and info; step returns observation, reward, separate
termination/truncation flags and info. Reset is explicit after either end flag;
there is no autoreset. Seeded resets are reproducible, and an omitted seed does
not restart the random stream. This is a C++ contract, not a Python bridge.

| Type | Initial supported representation |
| --- | --- |
| `Box` | Float32 values, row-major flat storage and explicit shape; empty shape means scalar |
| `Discrete` | Int64 count and start; integer values remain exact |
| `Info` and reset options | String-keyed bool, int64, double or string values |
| Results | Owned storage, independent of future steps or resets |

Nonfinite observations/actions are rejected by containment checks, even for
infinite Box bounds. Zero-sized dimensions are unsupported. Space sampling,
composite spaces, images and arbitrary Python objects are outside this subset.
The contract requires implementations to validate actions and lifecycle; an
abstract interface cannot enforce a third-party implementation's behavior.
The tests cover one neutral counter fixture, not all future implementations.

Both end flags may be true. No PPO numerical change is part of this increment;
adding these flags to a result type does not migrate historical `done` handling.

## Future remote boundary

A future remote implementation should implement the same `IEnvironment`, with
GNS and Protobuf in a separate target. The core should not know packet types.
An eventual protocol needs a versioned environment description, spaces, reset
seed/options, action and step results, error status, plus environment, episode,
step and request identifiers. Use a distinct environment identifier per worker,
reject stale responses, bound payloads, and define timeouts and shutdown behavior.
Transport failures should be explicit errors, not silently successful transitions.
No transport implementation or wire schema is delivered here.

## Remaining work and compatibility

- `GenericPPO` still depends on `GenericEntities`; no extraction of `RLAgent` yet.
- The legacy environment and evaluator do not consume the new interface.
- No generic evaluation aggregation or checkpoint promotion implementation added.
- No PPO `done`/GAE migration, policy bridge, network or Python bridge added.
- No existing file/API/checkpoint format changed. Consumers opt into `RL/`.
- Existing simulator builds and behavior are not validated by these contract tests.

The source tree at the inspected revision contains no identified test directory
or test-named file. The new tests exercise bounds, types, integer overflow,
nonfinite input, reset, terminal/truncated transitions and owned observations.
