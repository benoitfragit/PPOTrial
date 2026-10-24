\page json_formats JSON Configuration v2

[TOC]

## Single configuration format

All active JSON resources under resources/settings use **formatVersion = 2**.
No alternate runtime loaders or old-file path fallbacks exist.

| Purpose | Relative path under resources/settings |
| --- | --- |
| Logical agent profiles | AgentProfiles.json |
| Main scenario | Scenarios/Playground.json |
| Evaluation scenarios | Scenarios/Evaluation/RLMission.json, Scenarios/Evaluation/RLSNA.json |
| PPO settings | Training/RLMission.json, Training/RLSNA.json |
| Platform parameters | Platforms/Aircraft/*.json, Platforms/Ground/*.json |
| Evaluation cases | Evaluation/Cases/RLMission.json, Evaluation/Cases/RLSNA.json |
| Named initial-state presets | Evaluation/InitialStates.json |
| Simulation state | Environment.json |
| Display state | Renderer.json |

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

Environment.json and Renderer.json keep their normal Env and Renderer objects,
with formatVersion = 2. The Serializer refuses invalid input and does not
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
