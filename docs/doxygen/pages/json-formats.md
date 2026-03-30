\page json_formats JSON File Formats

[TOC]

## Common Rules

### Units

- Positions use meters on the north/east plane.
- Speeds use meters per second.
- Angles use radians.
- Time-related values use seconds unless noted otherwise.

### Path Resolution

- Files loaded through `ppo::serialization::Serializer` are resolved relative
  to the serializer root configured in `Main/src/main.cpp`, which is currently
  `resources/settings`.
- Scenario evaluation checkpoint paths are resolved first relative to the
  scenario file, then relative to `RESOURCE_PATH`.
- Aircraft platform names resolve to `resources/settings/Aircrafts/<name>.json`.
- Ground platform names resolve to `resources/settings/Grounds/<name>.json`.

### schemaVersion

- `schemaVersion = 1` is validated for files loaded through
  `ppo::serialization::Serializer`.
- That currently applies to files such as `Environment.json`,
  `Renderer.json`, `RLMission.json`, and `RLSNA.json`.
- Scenario files, evaluation buffers, and platform files are currently parsed
  directly and do not enforce `schemaVersion`.
- When a serializer-managed file is missing `schemaVersion`, the runtime
  currently normalizes it in memory to version `1` and writes it back on save.

## Runtime Settings

### `resources/settings/Environment.json`

```json
{
  "schemaVersion": 1,
  "Env": {
    "MaxNorth": 44658.75,
    "MaxEast": 71454.0,
    "TimeStep": 0.2
  }
}
```

- `Env.MaxNorth`: north-axis world limit.
- `Env.MaxEast`: east-axis world limit.
- `Env.TimeStep`: simulation step used by the environment.

### `resources/settings/Renderer.json`

```json
{
  "schemaVersion": 1,
  "Renderer": {
    "ScaleLevel": 1,
    "AutoAttachCamera": true,
    "ShowGrid": false,
    "ShowDomains": true,
    "ShowDestroyed": true,
    "ShowMetaCatalog": false,
    "ShowPseudoLidar": true,
    "FocusSelectionInMinimap": false,
    "CameraDistance": 35.2,
    "CameraHeight": 19.8,
    "TrailDuration": 3.1,
    "DomainHeight": 1.53,
    "MinimapZoom": 1.0,
    "Camera": {
      "Position": [34.0, 19.8, -17.0],
      "Target": [17.0, 0.35, 14.1],
      "Up": [0.0, 1.0, 0.0],
      "Fovy": 45.0
    }
  }
}
```

- `Renderer.ScaleLevel`: renderer world-to-UI scaling preset.
- `Renderer.AutoAttachCamera`: auto-follow selection behavior.
- `Renderer.ShowGrid`: grid visibility.
- `Renderer.ShowDomains`: threat domain visibility.
- `Renderer.ShowDestroyed`: destroyed entities visibility.
- `Renderer.ShowMetaCatalog`: metadata panel visibility.
- `Renderer.ShowPseudoLidar`: pseudo-lidar overlay visibility.
- `Renderer.FocusSelectionInMinimap`: minimap recentering behavior.
- `Renderer.CameraDistance`: follow-camera distance, clamped to `[5, 60]`.
- `Renderer.CameraHeight`: follow-camera height, clamped to `[2, 25]`.
- `Renderer.TrailDuration`: trail duration in seconds, clamped to `[2, 30]`.
- `Renderer.DomainHeight`: rendered domain height, clamped to `[0.3, 5]`.
- `Renderer.MinimapZoom`: minimap zoom factor, clamped to `[0.3, 4]`.
- `Renderer.Camera.Position`: 3D camera position vector.
- `Renderer.Camera.Target`: 3D camera look-at vector.
- `Renderer.Camera.Up`: 3D camera up vector.
- `Renderer.Camera.Fovy`: field of view in degrees, clamped to `[20, 90]`.

## Scenario Files

### `resources/settings/Scenarios/*.json`

```json
{
  "Scenario": {
    "entities": {
      "blue": {
        "type": "Aircraft",
        "platform": "trainingaircraft",
        "side": "Friend",
        "target": "obj2",
        "agents": ["TargetSelectorTrainingAgent"]
      }
    }
  },
  "Evaluation": {
    "evaluatedEntity": "blue",
    "checkpointFile": "../EvaluationBuffers/RLMissionEvaluationBuffer.json",
    "nbTests": 100,
    "nbSteps": 1024,
    "timeStep": 0.2,
    "maxNorth": 45000.0,
    "maxEast": 60000.0
  }
}
```

`Scenario.entities.<name>` fields:

- `type`: logical entity type. The runtime currently expects `Aircraft`,
  `SAM`, or `Objective`.
- `platform`: platform file stem. Required in practice for `Aircraft` and
  `SAM`, ignored for `Objective`.
- `side`: team side. Parsed values are `Friend` and `Hostile`.
- `target`: name of the primary target entity inside the same scenario.
- `agents`: list of agent factory names attached to the entity.

Optional `Evaluation` section:

- `evaluatedEntity`: scenario entity name whose policy is evaluated.
- `checkpointFile`: relative or absolute path to an evaluation buffer JSON.
- `nbTests`: maximum number of evaluation tests to run.
- `nbSteps`: number of simulation steps per test.
- `timeStep`: evaluation-only environment time step.
- `maxNorth`: north-axis evaluation boundary.
- `maxEast`: east-axis evaluation boundary.

This format maps directly to \ref ppo::config::ScenarioDefinition.

### `resources/settings/EvaluationBuffers/*.json`

```json
{
  "schemaVersion": 1,
  "EvaluationBuffer": {
    "tests": [
      {
        "name": "mission_eval_001",
        "entities": {
          "blue": {
            "north": -23800,
            "east": -14400,
            "heading": 0.255,
            "speed": 310,
            "accpsi": -0.01,
            "acc": -3
          }
        }
      }
    ]
  }
}
```

- `EvaluationBuffer.tests`: ordered array of evaluation cases.
- `tests[].name`: human-readable test identifier.
- `tests[].entities`: per-entity initial state map keyed by scenario entity
  name.
- `north`: optional north position.
- `east`: optional east position.
- `heading`: optional heading angle.
- `speed`: optional scalar speed.
- `accpsi`: optional turn acceleration.
- `acc`: optional longitudinal acceleration.

Validation rules enforced by the evaluator:

- Every scenario entity must exist in every test case.
- Aircraft entities require at least `north`, `east`, `heading`, and `speed`.
- Non-aircraft entities require at least `north` and `east`.

This format maps directly to \ref ppo::config::EvaluationBufferDefinition.

## Platform Files

### `resources/settings/Aircrafts/*.json`

The root object name must match the file stem.

```json
{
  "trainingaircraft": {
    "FlightModel": {
      "MaxAcceleration": 600.0,
      "MaxFuel": 2000000,
      "MaxRotAcceleration": 0.7,
      "MaxSpeed": 700.0,
      "MinSpeed": 10.0
    },
    "SNA": {
      "MaxMissile": 8,
      "MaxNoEscapeZoneRange": 5000.0,
      "MaxShootAngle": 3.1471980571746826
    },
    "SNAControl": {
      "Available": true
    }
  }
}
```

`FlightModel` fields:

- `MaxAcceleration`: maximum longitudinal acceleration.
- `MaxFuel`: maximum fuel quantity.
- `MaxRotAcceleration`: maximum turn acceleration.
- `MaxSpeed`: maximum speed.
- `MinSpeed`: minimum sustainable speed.

`SNA` fields:

- `MaxMissile`: maximum missile count.
- `MaxNoEscapeZoneRange`: no-escape-zone reference range.
- `MaxShootAngle`: maximum shot angle.

`SNAControl` fields:

- `Available`: currently parsed but not yet used to gate controller creation.

### `resources/settings/Grounds/*.json`

The root object name must match the file stem.

```json
{
  "sam": {
    "SNA": {
      "MaxMissile": 8,
      "MaxNoEscapeZoneRange": 7000.0,
      "MaxShootAngle": 3.14159178
    }
  }
}
```

- Ground platform files currently expose only the `SNA` block.
- The field meanings are the same as in aircraft platform files.

## RL Agent Settings

### `resources/settings/RLMission.json`

```json
{
  "schemaVersion": 1,
  "RLMission": {
    "Rollout": 2021,
    "Checkpoint": 5,
    "Version": 112,
    "Reward": {
      "Clamping": 60.0,
      "Scaling": [1.0, 0.15, 1.1, 0.8, 0.9, 1.0, 1.0, 0.7, 0.25, 0.9, 0.02, 0.4]
    },
    "PPO": {
      "Gamma": 0.99,
      "Lambda_GAE": 0.95,
      "Clip_Epsilon": 0.2,
      "Value_Coef": 0.5,
      "Entropy_Coef": 0.02,
      "Learning_Rate": 0.0003,
      "K_epochs": 4,
      "Batch_size": 64,
      "Target_KL": 0.026
    }
  }
}
```

Top-level fields:

- `Rollout`: rollout length collected before PPO update.
- `Checkpoint`: evaluation/checkpoint cadence used by the training flow.
- `Version`: saved model version marker.

`Reward` fields:

- `Clamping`: absolute reward clamp applied before storing the transition.
- `Scaling`: per-channel weight vector. The expected order follows
  `rl::StatesAndActionsForMission`.

Reward scaling order for `RLMission`:

- `0`: `Main target destruction`
- `1`: `Secondary target destruction`
- `2`: `Ownship destruction`
- `3`: `Wingmen destruction`
- `4`: `Civilian destruction`
- `5`: `MissionFailure`
- `6`: `Intercept`
- `7`: `Danger`
- `8`: `Out Of Area`
- `9`: `Range`
- `10`: `Fuel`
- `11`: `NearDanger`

`PPO` fields:

- `Gamma`: discount factor.
- `Lambda_GAE`: generalized advantage estimation lambda.
- `Clip_Epsilon`: PPO clipping parameter.
- `Value_Coef`: value-loss coefficient.
- `Entropy_Coef`: entropy regularization coefficient.
- `Learning_Rate`: optimizer learning rate.
- `K_epochs`: PPO epochs per rollout.
- `Batch_size`: PPO mini-batch size.
- `Target_KL`: early-stop KL target.

### `resources/settings/RLSNA.json`

```json
{
  "schemaVersion": 1,
  "RLSNA": {
    "Rollout": 800,
    "Checkpoint": 5,
    "Version": 888,
    "Reward": {
      "Clamping": 10.0,
      "Scaling": [1.0, 0.216, 1.0, 0.08, 0.791, 0.553, 0.007, 1.0]
    },
    "PPO": {
      "Gamma": 0.99,
      "Lambda_GAE": 0.95,
      "Clip_Epsilon": 0.2,
      "Value_Coef": 0.5,
      "Entropy_Coef": 0.02,
      "Learning_Rate": 0.0003,
      "K_epochs": 4,
      "Batch_size": 64,
      "Target_KL": 0.03
    }
  }
}
```

- The structure is the same as `RLMission.json`.
- The reward vector order follows `rl::StatesAndActions`.
- The sample currently stored in the repository does not include
  `schemaVersion`, but serializer-managed loads normalize it to `1` and save it
  back on write.

Reward scaling order for `RLSNA`:

- `0`: `Target destruction`
- `1`: `Target in EZ`
- `2`: `Ownship destruction`
- `3`: `Closing`
- `4`: `NavPN`
- `5`: `NavAPN`
- `6`: `Fuel`
- `7`: `MissionFailure`

## Legacy Checkpoint File

### `resources/settings/Evaluation.json`

This is a legacy checkpoint store used by `CheckpointInitializerSystem` when an
`sEvaluatorName` context is available. Its layout is different from the newer
scenario-driven `EvaluationBuffer` format.

```json
{
  "Data": {
    "blueeval": [
      [14905.04, -9068.54, 3.9094, 355.47, -0.0633, 47.8438]
    ]
  }
}
```

- `Data`: map keyed by logical checkpoint buffer name.
- `Data.<name>[]`: ordered list of six-scalar entries.
- Entry layout: `[north, east, heading, speed, accpsi, acc]`.

Prefer the newer `EvaluationBuffer` files for evaluator test suites. Keep this
legacy format only where checkpoint-driven initialization is still required.
