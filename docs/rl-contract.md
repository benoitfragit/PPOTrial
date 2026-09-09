# Architecture and migration

## Scope and implementation

The active example is now Navigation2D: one mobile, a fixed destination, circular
forbidden zones and square world bounds. The original PPO optimizer, rollout
collector and continuous Gaussian network are extracted rather than rewritten.
Old integration remains available only as an optional compatibility application.

| Target / directory | Responsibility | Dependencies |
| --- | --- | --- |
| `ppo_environment`, `RL/include`, `RL/src` | Spaces, environment API, shared episode driver, metrics, checkpoint promotion | C++17 standard library; localized Windows file publication |
| `ppo_torch`, `RL/torch` | Existing PPO, RLAgent, Gaussian network; generic learner and frozen inference adapters | LibTorch, generic runtime |
| `navigation_environment`, `Examples/Navigation2D` | Scenario, motion, observations, reward, collisions and arrival | generic runtime |
| `navigation2d` | Scripted example | Navigation2D |
| `navigation2d_policy`, `navigation2d_train` | Optional frozen evaluation and training/promotion CLI | Navigation2D, ppo_torch |
| `navigation2d_viewer` | Optional manual visualization | Navigation2D, installed raylib |
| `PPO/`, legacy directories | Historical compatibility wrappers/application | legacy simulator dependencies |

`ppo_torch` has no dependency on EnTT, entities, scenarios or rendering. Historical
`GenericPPO` is explicitly a compatibility wrapper, not the reusable backend.
The implementations moved from `PPO/src` and the former example network each have
one definition under `RL/torch`; their old headers forward to the extracted ones.
PPO parameter/loss types no longer pull in `components.hpp` or `IEvaluator`.
The legacy spelling `PPOParamters` and network parameter names remain for source
and archive compatibility. The optimizer equations/defaults are retained.

## Shared lifecycle and truncation

Training and evaluation use the same `execute` episode loop, via `train_episode`
and `run_episode`. The latter never invokes learner feedback. Evaluation accepts
a policy interface, and `FrozenTorchPolicy` owns a separately loaded network,
uses mean actions and a no-grad guard, and has no optimizer.

The API follows the conceptual [Gymnasium Env contract](https://gymnasium.farama.org/api/env/).
Owned results carry the final observation with `terminated` and `truncated`.
The runner validates actions, observations and finite rewards and applies its
own external step limit. It never substitutes reset observations for final ones.

The historical PPO retains its single masked-GAE `done` representation internally.
The new adapter maps an episode boundary to `done = terminated || truncated` and,
for truncation only, adds `gamma * V(final_observation)` to the learning reward.
This gives the same bootstrap delta while cutting the recurrence across resets.
The reward reported by the environment/evaluator stays unchanged. Ordinary and
terminal transitions retain their existing PPO handling. Raw Gaussian samples
are stored for their original log-probability; applied actions are clipped to
[-1,1], matching the existing continuous-command convention. This is not a new
tanh-squashed distribution. Discrete spaces are supported by the runtime, but the
supplied Torch adapter supports float32 Box observations and continuous actions only.

## Navigation semantics and configuration

`navigation2d-v1` files use strict rows (see the shipped `.nav` example):

| Row | Fields |
| --- | --- |
| `start` | x y heading-radians speed |
| `goal` | x y arrival-radius |
| `world` | half-extent timestep-seconds max-steps |
| `motion` | maximum-speed maximum-acceleration maximum-turn-rate |
| `obstacle` | center-x center-y radius; repeat for multiple obstacles |

Defaults come from `Scenario`. Unknown/trailing fields, nonfinite values,
invalid geometry and excessive sizes are rejected. No custom reset options are
supported in v1. Reset is deterministic for this fixed scenario; supplying any
seed gives the same state. There is no randomized curriculum.

The motion preserves the simple Euler heading/speed integration pattern from the
existing simulator, with nonnegative speed and configurable limits. Collision
and arrival are checked along the whole step segment, preventing tunnelling.
The first event stops the mobile; collision takes precedence on a tie.

Observations contain x/y and goal x/y normalized by world extent, cos/sin heading,
normalized speed and elapsed-step fraction, then each obstacle x/y/radius
normalized by extent. Observation width is `8 + 3 * obstacle_count`.
Actions are normalized turn and acceleration commands. The destination stays fixed.
Reward is normalized distance progress minus 0.001 per step, +1 on arrival and
-1 on collision/boundary exit. These navigation rewards intentionally replace
old domain rewards. Arrival/collision terminate; the time limit truncates.

The new scenario format and observation/reward contract are incompatible with old
scenario/model semantics. Do not rename old archives and load them as Navigation2D.
Use separate training output directories. A training run refuses an existing output
path and stores its scenario and scoring configuration. The old ECS renderer and
JSON loading remain in the legacy application; the navigation viewer is small and
independent. The legacy UI has not been migrated wholesale.

## Evaluation and checkpoint lifecycle

Metrics config rows are `key reduction weight`; reductions: mean, sum, min, max,
std (population) and rate (fraction nonzero). Keys are `episode_reward`,
`episode_length` or numeric/bool final `info` fields. Missing metrics are errors.
`score` is the weighted sum; negative weights express minimization. Initial metric
aggregation uses final episode info, not every step's info.

The opt-in training CLI performs: collect episode -> save immutable candidate ->
load independent frozen policy -> evaluate in a separate environment -> score ->
promote if strictly better. The default example is deterministic and evaluates
one episode of the same fixed scenario; it is a wiring example, not evidence of
out-of-distribution performance. Partial rollouts are retained across episodes;
only full buffers trigger the reused optimizer. A final partial buffer is not
optimized automatically and is not saved for training resume.

Candidates use the existing `eval/model.pt` archive layout. The historical saver
also creates a candidate-local `prod/model.pt` on first save; only `best/best.txt`
is authoritative for promotion in the new workflow. It contains score and an
immutable version directory. Publication uses rename on POSIX or MoveFileExW on
Windows, guarded by an exclusive directory lock. Copy/write failure cannot point
best at a partial candidate. A failed copy can leave an unreferenced version;
remove it manually before retrying that version. A process crash can leave the
lock directory; remove it only after confirming no publisher remains. This is a
single-host store, not a distributed checkpoint service. Candidates must be
trusted local directories outside the store and stable during publication.

## External environments

`RL/transport/environment.proto` is a versioned wire draft for descriptions,
spaces, reset/options/seed, action, results, errors and close. Every envelope
carries environment, episode, step and request IDs for future parallel workers.
The GNS adapter will implement `IEnvironment` and live outside the core.
The protocol needs implementation-side bounds, stale-response checks, explicit
timeouts and idempotency rules before production use. No GNS transport or Python
bridge is implemented, and protoc generation is not in the default build.

## Validation limits

The standard-library test suites use scripted/fake policies and synthetic
checkpoint files. They exercise spaces, lifecycle, shared training/evaluation,
arrival ordering, swept collisions, time limits, bootstrap conversion, metrics,
configuration parsing and checkpoint promotion. They never execute a neural model.

Direct GCC builds are supported as a fallback if CMake is unavailable. LibTorch,
raylib, the Windows compatibility application and protoc must be built in an
environment with those dependencies before considering their paths validated.
No training, learned-model evaluation or checkpoint loading was performed during
this refactoring. See the PR for exact commands/results and outstanding gates.
