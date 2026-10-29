\page json_formats JSON Configuration v2

[TOC]

## Single configuration format

All active JSON resources under resources/settings use **formatVersion = 2**.
No alternate runtime loaders or old-file path fallbacks exist.

| Purpose | Relative path under resources/settings |
| --- | --- |
| Logical agent profiles | AgentProfiles.json |
| Main scenario | Scenarios/Playground.json |
| Blue MAPPO scenarios | Scenarios/BluePatrol.json, Scenarios/TwoVsTwo.json |
| Evaluation scenarios | Scenarios/Evaluation/RLMission.json, Scenarios/Evaluation/RLSNA.json |
| PPO/MAPPO settings | Training/RLMission.json, Training/RLSNA.json, Training/BluePatrol.json |
| Platform parameters | Platforms/Mobile/*.json, Platforms/RestrictedZone/*.json |
| Evaluation cases | Evaluation/Cases/RLMission.json, Evaluation/Cases/RLSNA.json |
| Named initial-state presets | Evaluation/InitialStates.json |
| Simulation state | Environment.json |
| Display state | Renderer.json |

The `twoVsTwo` object in `Scenarios/TwoVsTwo.json` owns the episode spawn
contract. `redSpawn` defines the front/lateral bearing interval, the scale
beyond the rectangular world boundary and the guaranteed attacker separation.
The shipped `redTeam` instantiates four production RLMission attackers, with
seeded optional activation delays. The legacy `threatCurriculum` remains
available only when `redTeam` is absent.
`restrictedZoneSpawn` defines a side-relative position box and an activation
probability. World limits remain sensor and reward references; they do not stop
the production RLMission attackers.

`blueTeam` and `redTeam` are optional additive contracts. `blueTeam` selects
the generalized blue runtime and `redTeam` sizes its opponents. If both are
absent, the archived 2v2 entities, RNG order, curriculum and v7/v4 checkpoint
path are unchanged. `Scenarios/TwoVsTwo.json` enables `blueTeam` and `redTeam`
directly so the existing GUI entry owns the generalized MAPPO runtime:

~~~json
"blueTeam": {"activeCount": 5},
"redTeam": {
  "activeCount": 7,
  "maximumPlacementAttempts": 512,
  "delayedActivationProbability": 0.25,
  "minimumActivationDelaySteps": 256,
  "maximumActivationDelaySteps": 768
}
~~~

`blueTeam.activeCount` is bounded to `[1,9]`; `redTeam.activeCount` is bounded
to `[0,10]`. Generalized red placement attempts are bounded to `[1,4096]` per
mobile and fail with a diagnostic when `redSpawn.minimumSeparation` cannot be
satisfied. Activation delays are positive, finite, ordered ECS team-step
indices. An enabled legacy `threatCurriculum` and generalized `redTeam` are
mutually exclusive.

The generalized loader clones one agent-free friendly Mobile template into
`blue_1..blue_P`, and one production-RLMission hostile Mobile template into
`red_1..red_N`. Exactly one blue leader is sampled per reset. Every teammate
owns a stable unique V-formation slot and its normalized `[-1,1]` encoding.
One synchronized team step produces P actions before physics and exactly P
same-step transitions after ECS resolution.

See docs/configuration-guide.md for explanations and examples.

## Scenario and agent profiles

A scenario contains **formatVersion**, **agentProfiles** (path to AgentProfiles.json),
and **entities** keyed by logical identifier. Each entity contains **type**, **platform**,
**side**, **target** and a list of agent descriptors.

~~~json
{
  "formatVersion": 2,
  "agentProfiles": "../AgentProfiles.json",
  "entities": {
    "blue": {
      "type": "Aircraft",
      "platform": "trainingaircraft",
      "side": "Friend",
      "target": "obj2",
      "agents": [{"profile": "RLMission", "mode": "training"}]
    }
  }
}
~~~

AgentProfiles.json maps each logical profile plus mode onto the application's
existing registered C++ agent name. The PPO core does not depend on agent names.

## Training

Training/PROFIL.json has root **formatVersion**, plus **training**, **rewards** and
**ppo** objects. Rewards use a map keyed by the model's reward-channel names.
The codec resolves these names in the model's declared order and rejects
missing or unknown channels.

- training.rolloutSteps: transitions in one PPO rollout.
- training.checkpointEvery: PPO updates between evaluation requests.
- training.modelVersion: saved/current policy version.
- rewards.clamp: total reward clamp.
- rewards.weights: per-channel coefficients by name.
- ppo.gamma, gaeLambda, clipEpsilon, valueCoefficient, entropyCoefficient,
  learningRate, epochs, batchSize, targetKL: PPO hyperparameters.

The settings files are updated by the application on exit.

Training/BluePatrol.json configures the separate native RLTools MAPPO backend.
Its patrol object contains the bounded racetrack, wedge, and credible-threat
defence settings. All geometry ranges are sampled once per episode from the
configured seed; no waypoint list is generated. It also contains
training.rolloutSteps, training.maxUpdates,
training.checkpointEvery, training.seed, rewards.clamp and seven named reward
weights. Its rolloutSteps value counts ECS team steps: two defenders generate
two actor transitions per team step. The application presets are 512, 1024,
2048 and 4096; 128 is the PPO mini-batch size, not the application rollout.
Its evaluation object contains episodes, maxSteps, seed and
minimumTrackingFraction. checkpointEvery starts this deterministic live-ECS
campaign; the current Actor is promoted to production only after improving the
recorded score.

Blue MAPPO writes an inference Actor and a separate complete training state in
`model/BluePatrol/latest`. The training state contains Actor, centralized
Critic, both Adam states, update count and RLTools RNG. Rollout samples are not
serialized; resume starts a fresh rollout. `model/BluePatrol/prod` remains an
inference-only promoted Actor plus its evaluation score.

The archived two-blue checkpoint contract is Actor v7 and training-state v4.
The P/N path writes `blue_patrol_generalized_v8.txt` and
`blue_patrol_generalized_training_v5.txt`. Its raw v8 observation has 295
values: own state, exactly 61 ECS pseudo-LiDAR rays, explicit role/slot, ten
masked red tracks, eight masked other-blue tracks, a separate masked objective,
patrol/global fields and normalized P/N. Masked order-invariant attention and
mean/max pooling build the compact 156-value MLP trunk.

The same `evaluation` settings drive a generalized update-boundary campaign.
It uses deterministic v8 Actor means in the live ECS without adding rollout
rows or consuming the learner RNG. Episode survival and neutralization are
normalized by P and N before campaign averaging. A strictly better candidate
is promoted only to `model/BluePatrol/prod/blue_patrol_generalized_v8.txt`;
its P/N-tagged score metadata and all v8 files remain separate from v7/v4.

A strict v7 Actor can initialize the compatible v8 encoders, queries, trunk,
output and standard deviation. The explicit v4 training migration copies only
the compatible Actor and its matching Adam moments. The fixed-2v2 Critic,
optimizer scalar states, RNG, update count and rollout are reset. This is not a
bit-identical training resume. Never delete the v7/v4 files: the legacy 2v2
runtime still owns them and the generalized runtime may use them as migration
sources. V8/v5 writes are atomic and reject wrong versions or truncation. Red
RLMission checkpoints are unaffected.

Restricted zones never consume any of RLMission's ten ranked tracks. Only
living opposing `Mobile` and `Objective` entities are track candidates; zones
remain visible through the dedicated SAM/pseudo-LiDAR path. The RLMission
152-value layout is unchanged.

Each hostile zone may set `oppositeObjective=true`. Its configured east value
is then treated as a magnitude and mirrored onto the side opposite the current
objective. The shipped scenario uses two inside-world and two outside-world
centres, so the objective cannot be covered by a hostile zone.

The blue `Platforms/Mobile/defender_car.json` profile mirrors the
`trainingaircraft` speed, turn-rate and SNA engagement-angle limits and exposes
`FlightModel.CruiseSpeed`. The TwoVsTwo loader uses it consistently for the
initial state and patrol reference. The remaining `FlightModel` and `SNA`
fields define the blue physical and engagement limits without changing the
red `attacker_car` profile.

## Evaluation

An evaluation scenario has the same **entities** and **agentProfiles**
structure, plus an **evaluation** block with:

- **evaluatedEntity**: entity identifier.
- **casesFile**: relative path to the input cases JSON.
- **nbTests**: requested cases, bounded by available data.
- **nbSteps**: maximum steps per case.
- **timeStep**: simulated seconds per step.
- **maxNorth**, **maxEast**: world boundaries in metres.

Evaluation/Cases/PROFIL.json contains:

~~~json
{
  "formatVersion": 2,
  "tests": [
    {"name": "example", "entities": {
      "blue": {"north": 0, "east": 0, "heading": 0, "speed": 300}
    }}
  ]
}
~~~

The actual case files each contain 100 full tests. Case files are **not**
neural-network checkpoints (model.pt).

Evaluation/InitialStates.json is a separate source of named presets consumed
by CheckpointInitializerSystem. Its **presets** object holds per-entity
lists of objects with north, east, heading, speed, turnAcceleration and
acceleration. It does not replace the evaluator's test suites.

## Platforms and state settings

Each platform JSON contains **formatVersion** and a **parameters** object.
Platform paths use Platforms/Aircraft/ or Platforms/Ground/ and file stems
matching the platform names from scenarios.

Environment.json uses a lowerCamelCase **environment** object (maxNorth,
maxEast, timeStep). Renderer.json uses a lowerCamelCase **renderer** object
(scaleLevel, autoAttachCamera, showGrid, camera and the other visual preferences).
All settings are written in the same format that is read, with formatVersion = 2. The Serializer refuses invalid input and does not
overwrite it with default values at shutdown.

Units: North/East positions and limits are metres; speeds are metres/second;
orientation is radians; simulation time is seconds; camera FOV is degrees.

## Extension and validation

Register C++ application agent factories, add a logical profile/mode to
AgentProfiles.json, and reference it from a scenario. Update tests for the
agent's reward contract and evaluation cases. No changes are needed in
GenericRLCore.

Configuration tests assert numeric settings, resolved agent names, initial
case counts and handling of malformed JSON. Run the Windows VS2022 Debug and
Release CTest suites before release.

For fast development iterations, build and run only `TwoVsTwoSpawnTests`,
`TwoVsTwoTeamDomainTests`, `TwoVsTwoTeamDynamicsTests`, the
`TwoVsTwoGeneralized*` tests, and `MissionTrackSelectionTests`. The runtime and
embedded-learner smoke tests plus `PPO.exe --example two-vs-two --check-example`
are the heavier final pre-merge validation, not a requirement after each core
edit.
