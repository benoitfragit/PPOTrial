# Architecture and migration

## Scope and implementation

The active example is now Navigation2D: a controlled mobile and a moving goal mobile, circular
forbidden zones and square world bounds. The original PPO optimizer, rollout
collector and continuous Gaussian network are extracted rather than rewritten.
The desktop is ported in place; the old application and compatibility layer are removed.

| Target / directory | Responsibility | Dependencies |
| --- | --- | --- |
| `ppo_environment`, `RL/include`, `RL/src` | Spaces, environment API, shared episode driver, metrics, checkpoint promotion | C++17 standard library; localized Windows file publication |
| `ppo_torch`, `RL/torch` | Existing PPO, RLAgent, Gaussian network; generic learner and frozen inference adapters | LibTorch, generic runtime |
| `navigation_environment`, `Examples/Navigation2D` | Scenario, motion, observations, reward, collisions and arrival | generic runtime |
| `navigation_application`, `Main` | Desktop clock, explicit mode selection, checkpoint/evaluation orchestration | Navigation2D, injected policy backend interface |
| `GenericRenderer`, `Renderer` | Existing desktop theme/docking, views, controls, inspection and plots | application presentation API, raylib, ImGui, ImPlot |
| `PPO`, `Main/src/main.cpp` | The sole application entry point | controller and renderer |

`ppo_torch` has no dependency on EnTT, entities, scenarios or rendering. The
optimizer, rollout collector and network each have one definition under `RL/torch`.
The internal spelling `PPOParamters` and archive parameter names remain unchanged;
there are no old-header forwarding shims or old application build switches.

## Shared lifecycle and truncation

Training and evaluation use `EpisodeSession`, incrementally in the desktop and
via `train_episode` / `run_episode` for batch callers. `run_episode` never invokes
learner feedback. A session completes act/step/observe before returning control to
the UI. Pause does not reset anything. An external Stop marks the next complete
transition truncated, then pauses; it executes one final action. Reset rejects an
active episode, preventing accidental loss of the rollout boundary. Evaluation accepts
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

`navigation2d-v2` files use strict rows (see the shipped `.nav` example):

| Row | Fields |
| --- | --- |
| `start` | x y heading-radians speed |
| `goal` | initial-x initial-y meeting-radius velocity-x velocity-y |
| `world` | half-extent timestep-seconds max-steps |
| `motion` | maximum-speed maximum-acceleration maximum-turn-rate |
| `obstacle` | center-x center-y radius; repeat for multiple obstacles |

Defaults come from `Scenario`. Unknown/trailing fields, nonfinite values,
invalid geometry and excessive sizes are rejected. No custom reset options are
supported in v2. Reset is deterministic for this fixed scenario; supplying any
seed gives the same state. There is no randomized curriculum.

The motion preserves the simple Euler heading/speed integration pattern from the
existing simulator, with nonnegative speed and configurable limits. Collision
and arrival are checked along the whole step segment, preventing tunnelling.
Meeting uses relative swept motion, rather than a check against a frozen endpoint.
The first event stops both mobiles at the same fractional step time; collision
takes precedence on a tie. A reset restores both initial positions.

Observations contain x/y and current goal x/y normalized by world extent, cos/sin heading,
normalized speed and elapsed-step fraction, goal velocity x/y normalized by maximum
controlled-mobile speed, then each obstacle x/y/radius
normalized by extent. Observation width is `10 + 3 * obstacle_count`.
Actions are normalized turn and acceleration commands. The goal moves linearly at its configured velocity. Its full trajectory must
remain inside the world (including the meeting radius); invalid paths are rejected.
The scripted goal does not steer around obstacles. Forbidden zones constrain the
controlled mobile; a goal inside a zone may be temporarily unreachable.
Reward is normalized distance progress minus 0.001 per step, +1 on arrival and
-1 on collision/boundary exit. These navigation rewards intentionally replace
old domain rewards. Arrival/collision terminate; the time limit truncates.

The new scenario format and observation/reward contract are incompatible with old
scenario/model semantics. Do not rename old archives and load them as Navigation2D.
Checkpoints use a dedicated `checkpoints/navigation2d` directory. Each evaluation
candidate stores its scenario and score configuration alongside the existing
archive format. Historical JSON, assets and model archives remain stored under `resources/` but
are not read, linked or copied by the new application. Preserving those data does
not provide an old application or a compatibility mode.

The GUI keeps the existing theme and dock builder plus the viewport, minimap,
entity selection, inspection, controls, snapshot and plot panels. Their old ECS
bindings are replaced by `navigation::Application`. `Main/src/torch_backend.cpp` alone binds the controller to Torch. The controller
can therefore be exercised headlessly with synthetic policies. The schema catalog now
reports Navigation2D fields instead of EnTT reflection. Domain-specific controls
and assets have been removed; reward rules are displayed as environment data,
not duplicated as PPO settings. Renderer display flags use a new JSON settings
file; there is no old-settings migration. The renderer uses planar coordinates
inside its existing perspective viewport. It does not simulate motion.

## Evaluation and checkpoint lifecycle

Metrics config rows are `key reduction weight`; reductions: mean, sum, min, max,
std (population) and rate (fraction nonzero). Keys are `episode_reward`,
`episode_length` or numeric/bool final `info` fields. Missing metrics are errors.
`score` is the weighted sum; negative weights express minimization. Initial metric
aggregation uses final episode info, not every step's info.

The desktop's explicit candidate command performs: pause training -> save candidate
-> load independent frozen policy -> evaluate in a separate instance of the same
environment -> score -> promote if strictly better. Evaluation advances at most
64 transitions per frame, keeping the UI command path available. Pause/Play suspends/resumes evaluation. Stop cancels
without promotion. The selected fixed scenario is deterministic; repeated episodes
are a wiring check, not evidence of generalization. No background training starts
at application launch. Full rollouts trigger the original optimizer synchronously;
a PPO update may therefore stall a UI frame. Partial rollouts remain in memory
across episodes, but neither optimizer state nor partial buffers are restored from
an inference checkpoint. Rollout settings apply only when constructing a learner.

Candidates use the existing `eval/model.pt` archive layout. The historical saver
also creates a candidate-local `prod/model.pt` on first save; only `prod/best.txt`
is authoritative for promotion in the new workflow. It contains score, an immutable version directory, and a quoted evaluation-context
string. The generic publisher treats that string as opaque and rejects mismatches
before comparing scores. Navigation2D supplies canonical scenario/metrics text,
episode count and seed sequence. This reference format is incompatible with the
earlier two-line reference; use a new checkpoint root rather than silently
comparing against a score without evaluation provenance. Publication uses rename on POSIX or MoveFileExW on
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
raylib and protoc must be built in an
environment with those dependencies before considering their paths validated.
The ported renderer is syntax-checked against real dependency headers; this does
not validate graphics linking or interactive behavior. CMake is unavailable in
the execution environment. No training, learned-model evaluation or checkpoint loading was performed during
this refactoring. See the PR for exact commands/results and outstanding gates.

## Moving-goal correction

The earlier fixed-only goal was an excessive simplification of the requested
navigation example. Version 2 restores the ability to reach another moving point
in the toy 2D environment, without a new guidance law or a PPO change. Existing
progress/reward coefficients are retained; distance now measures the two current
positions, so movement of either mobile affects distance progress.

The default scenario contains a moving goal. A deterministic test reaches that
goal with a forbidden zone present; additional tests check encounters between
step endpoints, collision-before-meeting, reset, moving-state observations and
invalid velocities/paths. Training and evaluation exercise the same moving-goal
environment through the existing runtime. No learned policy is run by these tests.
