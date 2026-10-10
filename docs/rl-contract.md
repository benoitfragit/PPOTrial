# Architecture and migration

## Scope and implementation

The active example is Navigation2D: an EnTT world with N mobile entities, per-mobile
destinations and controllers, forbidden-zone entities and square world bounds. The original PPO optimizer, rollout
collector and continuous Gaussian network are extracted rather than rewritten.
The desktop is ported in place; the old application and compatibility layer are removed.

| Target / directory | Responsibility | Dependencies |
| --- | --- | --- |
| `ppo_environment`, `RL/include`, `RL/src` | Spaces, environment API, shared episode driver, metrics, checkpoint promotion | C++17 standard library; localized Windows file publication |
| `ppo_torch`, `RL/torch` | Existing PPO, RLAgent, Gaussian network; generic learner and frozen inference adapters | LibTorch, generic runtime |
| `navigation_environment`, `Examples/Navigation2D` | ECS world, scenario, shared motion, observations, reward and events | generic runtime, EnTT v3.16.0 |
| `navigation_application`, `Main` | Desktop clock, explicit mode selection, checkpoint/evaluation orchestration | Navigation2D, injected policy backend interface |
| `GenericRenderer`, `Renderer` | Existing desktop theme/docking, views, controls, inspection and plots | read-only example registry, application, raylib, ImGui, ImPlot |
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

## ECS ownership and world lifecycle

The original `Env` owned an EnTT registry and collections of entities and agents;
it chose every agent action before advancing every entity. The first port wrongly
reduced that to two states. The restored example keeps those ECS capabilities:

- `Identity`: stable numeric scenario ID and unique name.
- `State`: position, heading and speed for mobiles; position for zones.
- `Mobile`, `InitialState`, `Destination`, `Commands`: mobile components.
- `ForbiddenZone`: circular radius component on a zone entity.

The registry is the sole runtime state store, not a mirror of a special pair.
`Environment::registry()` exposes a const view for inspection/rendering on the
same thread. EnTT handles are internal; stable `EntityId` values identify controls,
destinations, telemetry and selections. Reset restores every mobile without
changing handles. Validated topology edits may rebuild handles; clients must
resolve them again from stable IDs. Unknown IDs and dangling destinations are errors.

`create_mobile`, `create_zone`, `remove_entity` operate before stepping or at an
episode boundary, with internal policies detached. Validation precedes replacement;
a rejected edit leaves the world intact. The desktop can load/save complete
scenarios while stopped without policies. No dynamic topology mutation is allowed
inside a rollout. Up to 4096 mobiles and 1024 zones are accepted as input bounds,
not an architectural pair limit. Adding new component/system types remains C++ work.

The scenario's `controlled` mobile is the single-agent Gymnasium view driven by
`EpisodeSession`. Other mobiles have coasting/manual commands or independently
owned `IPolicy`/`ILearner` controllers. All actions are collected and validated
against the same pre-step world, then all entities advance exactly once. All
transitions are built before any secondary learner receives feedback. A policy is
only an inference controller unless learning is explicitly requested. The primary
learner receives its feedback through `EpisodeSession`; it cannot be attached twice.

The first meeting, forbidden-zone entry or world-boundary crossing ends the world
step at one common fractional time. Mobile/zone collision takes precedence over a
meeting at the same instant. Each mobile receives its own reward, observation and
flags: its own meeting/collision or its destination's collision terminates it;
an event elsewhere truncates its still-open episode. The time limit and explicit
world stop truncate unfinished episodes. Reset restores all entities together,
matching the original global reset model. No unrelated reset observation enters GAE.

For batch use with additional learners, stop the world with `request_stop()` before
the final advance, or use the world's `max_steps`. An external single-agent runner
limit alone cannot notify embedded learners; the application propagates its Stop
request to both the world and primary session. Updates still run synchronously on
the UI thread. This port does not restore the original worker-thread/snapshot bus;
concurrent consumers require a separate snapshot/transport layer.

## Navigation semantics and configuration

`navigation2d-v3` files use strict rows; `multiple.nav` demonstrates five mobiles
and two forbidden zones without writing C++:

| Row | Fields |
| --- | --- |
| `primary` | externally controlled mobile ID |
| `world` | half-extent timestep-seconds max-steps |
| `motion` | maximum-speed maximum-acceleration maximum-turn-rate |
| `mobile` | ID "name" x y heading-radians speed destination-ID meeting-radius |
| `zone` | ID "name" center-x center-y radius |

Repeat mobile/zone rows. IDs and names are unique across all entity types;
destination IDs must identify another mobile, or `0` for no destination. A learner
or policy requires a destination. The primary always requires one. Entity order
is normalized by ID, including zone observation order. Duplicate singleton rows,
nonfinite values, invalid geometry and trailing fields are rejected. Mobiles may
not start in forbidden zones. Runtime collision checks handle future world exits;
the old full-horizon linear-goal validation has been removed.

Motion retains heading/speed Euler integration, with normalized turn/acceleration
actions, nonnegative speed and shared configurable limits. The same function
advances every mobile. All mobiles obey zones and world bounds, including coasting
ones. Relative swept motion detects a meeting between step endpoints. Reset is
deterministic; seeds currently produce the same configured initial state.

Observations contain own x/y and destination x/y normalized by world extent,
cos/sin own heading, own speed/max-speed, elapsed-step fraction, destination
velocity x/y normalized by max-speed, then zone x/y/radius normalized by extent.
Width is `10 + 3 * zone_count + 61`; the last 61 values are a heading-relative, 360-degree ray-cast scan at six-degree increments, including the repeated endpoint. Rays intersect the nearest forbidden-circle boundary or square work-area edge. Normalization uses `distance/(distance+reference)`, with circle radius or the work-area half-diagonal as reference; raw distances are exposed separately for visualization. Unrelated mobiles do not enter the observation.
A mobile without destination uses itself for observation only, with no meeting event.
Reward includes distance progress/world extent minus 0.001, optional dense proximity penalties near forbidden zones and the work-area perimeter, +1 on its own meeting,
-1 on its own collision. `goal_collision` adds no new penalty. These are the
navigation rewards of the accepted replacement example, not the former domain rewards.

v3 replaces v2's start/goal/obstacle rows; no compatibility shim is provided.
Matching tensor dimensions alone do not establish old-policy compatibility:
world termination and zone semantics changed. Old historical assets/checkpoints
remain under `resources/`, unused by this application. The selected mobile's rays may be displayed in the desktop viewport; the display toggle is saved separately from simulation state. The viewport remains a
perspective presentation of 2D motion, with lists, inspection, selection, minimap
and trails driven by all ECS entities. Controller/observation/metric panels follow
the selected mobile. The catalog describes current component types, not old EnTT
meta reflection. UI preferences retain their explicit separate settings file.

## Evaluation and checkpoint lifecycle

Metrics config rows are `key reduction weight`; reductions: mean, sum, min, max,
std (population) and rate (fraction nonzero). Keys are `episode_reward`,
`episode_length` or numeric/bool final `info` fields. Missing metrics are errors.
`score` is the weighted sum; negative weights express minimization. Initial metric
aggregation uses final episode info, not every step's info.

The desktop's explicit candidate command performs: pause training -> save candidate
-> load independent frozen controllers -> evaluate in a separate instance of the same
environment -> score -> promote if strictly better. Evaluation advances at most
64 transitions per frame, keeping the UI command path available. Pause/Play suspends/resumes evaluation. Stop cancels
without promotion. The selected fixed scenario is deterministic; repeated episodes
are a wiring check, not evidence of generalization. No background training starts
at application launch. Full rollouts trigger the original optimizer synchronously;
a PPO update may therefore stall a UI frame. Partial rollouts remain in memory
across episodes, but neither optimizer state nor partial buffers are restored from
an inference checkpoint. Rollout settings apply only when constructing a learner.

Candidates use the existing `eval/model.pt` archive layout. The historical saver
also creates a candidate-local `prod/model.pt` on first save; only `prod/mobile-<id>/best.txt`
is authoritative for promotion in the new workflow. It contains score, an immutable version directory, and a quoted evaluation-context
string. The generic publisher treats that string as opaque and rejects mismatches
before comparing scores. Navigation2D supplies canonical scenario/metrics text,
episode count, seed sequence and controller snapshot identities. This reference format is incompatible with the
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

## Frozen Prod/N-1 and multiple learners

Choose a versioned `model.pt` for any selected mobile through **Agents / Selected
mobile mode / Frozen Prod policy**. The primary has its own mode/path controls.
A private snapshot is copied on explicit selection and stays fixed after promotion.
Separate learners use separate `work/mobile-<id>` directories; no mutable model or
rollout is shared by default. Manual arrows affect only the selected manual mobile.

**Evaluate selected learner / promote** evaluates any learner, including a
non-primary one, by making it the primary view of a separate world. Every other
controller is independently loaded and frozen. Background learners are saved at
that instant; background frozen policies use their pinned archives. Candidate
folders contain the evaluated scenario, criteria, full evaluation context and
`controllers/<id>/...` archives. Training-world positions and weights remain untouched
by evaluation. References are isolated under `prod/mobile-<id>/best.txt`.

Snapshot identities enter the score context. Changing/reselecting a frozen
controller creates a new identity. Evaluating with background learners also takes
new snapshots each time. Previously stored scores from different backgrounds are
rejected; use a separate checkpoint root or explicitly freeze a stable population
for repeated comparable evaluations. Automatic re-evaluation of the previous best
against a changing population and a content-addressed checkpoint registry are not
implemented. This limitation is explicit; stale scores are never silently compared.

## Validation limits

The tests use only scripted/fake policies and synthetic checkpoint files. Coverage
includes all previous navigation/runtime regressions, N ECS entities, multiple
learners with independent feedback, selected manual control, frozen Prod policies,
non-primary evaluation with a fully frozen background, common event timing, global
reset/truncation, stable IDs, topology validation and scenario round trips.

Local CMake headless builds now use the real EnTT v3.16.0 headers. Renderer/Main
syntax checks use the actual graphics headers. CI validates Linux Debug/Release
and a complete Windows LibTorch/graphics build. No model or desktop window is
executed by these checks; interactive graphical smoke testing remains outstanding.

## Correspondence with the original example

| Original concept | Restored Navigation2D implementation |
| --- | --- |
| Registry and N entities | EnTT registry, stable identities, component views and N entity scenario rows |
| Aircraft | Mobile entity with State, InitialState, Destination and Commands |
| Other Aircraft / destination | Reference to any mobile ID; independent manual, learner or frozen Prod controller |
| SAM regions | ForbiddenZone entities with State and circular radius |
| Distance tolerance | Per-mobile Destination radius and relative swept meeting |
| End/destruction flags | Per-mobile success/collision/goal_collision and terminated/truncated |
| SNA / domain controllers | Normalized turn/acceleration commands; no old domain systems restored |
| Agent collections | Controllers keyed by mobile ID, simultaneous decisions and separate learner feedback |
| Shared reset | Whole-world reset; final transitions close all affected rollouts |
| Renderer entity bindings | Read-only ECS views, N selections, trails, inspection and controller panels |

The neutral example remains a port, not numerical equivalence with the original
domain simulation. The ECS/N-entity capability is restored; the old domain-specific
systems, JSON schema, worker thread and meta-reflection editor are not restored.
PPO's optimisation equations are unchanged by this ECS correction.
