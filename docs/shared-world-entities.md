# SharedWorld: model naming and pseudo-LiDAR compatibility

## Purpose
The reusable `Examples/Shared/World` C++17 library owns the concrete 2D world entities
`Mobile`, `RestrictedZone`, `Objective`, the unchanged sensor/control/motion
systems and their state layouts. `Examples/Airspace` owns the historic
scenario, scoring, evaluation campaign (`EvaluationCampaign`) and 2D
Visualiseur (the former evaluator and renderer were removed with `PPO.exe`),
but no longer duplicates these basic entity classes and algorithms.

This extraction deliberately preserves the legacy SNA component names,
`FlightModel` and `SNAControl` serialized sections and the network contract.
Those identifiers are part of the existing checkpoint/JSON interface and are
not renamed as part of the public class and folder cleanup.

## Types, factories and EnTT compatibility
- **Canonical C++ types**: `ppo::model::Mobile`,
  `ppo::model::RestrictedZone`, `ppo::model::Objective`.
- **Canonical scenario types**: `"Mobile"`, `"RestrictedZone"`,
  `"Objective"`.
- **Legacy scenario aliases**: `"Aircraft"` instantiates a **Mobile**;
  `"SAM"` instantiates a **RestrictedZone**: the factory table registers both
  names with the same factories (`RegisterEntityFactory("Aircraft", ...)`
  creates a `Mobile`, `RegisterEntityFactory("SAM", ...)` a `RestrictedZone`).
  There is no reflection lookup any more (`ResolveType` and `entt::meta` were
  removed).
- **Legacy C++ headers**: `aircraft.hpp` and `sam.hpp` are thin aliases;
  there are no duplicate implementations or duplicate ECS components.
- **Serialized category values** remain unchanged: `Mobile`/`Aircraft` = 3,
  `RestrictedZone`/`SAM` = 4, `Objective` = 1. Old and new names resolve
  to the **same underlying enum values**.
- The existing Airspace `AirspaceEntities` CMake target becomes an interface
  bridge to `SharedWorld`; code using it still builds and gets the shared headers.

## Platform files
Canonical directories:
`Examples/Shared/settings/Platforms/Mobile` and
`Examples/Shared/settings/Platforms/RestrictedZone`.
Historical archetype names `aircraft`, `trainingaircraft`,
`evalaircraft`, `sam` are deliberately retained so old scenario files
continue to load. Canonical example files `mobile.json` and
`restrictedzone.json` are also provided. The constructors permit fallback
to the historical platform directories for user-maintained platform files.

## Pseudo-LiDAR
The historical 61-beam ray intersection against stationary zone areas and
world boundaries has been moved from the RLMission adapter to the shared
`restrictedzonelidar.hpp/.cpp`; no ray math, normalization, constants,
beam angles, masking, or iteration order was changed. Existing RLMission
observations call this shared implementation and still write exactly the same
`sPseudoLidarState` fields. The historical behavior of the restricted-zone
warning region is 1.10 times the no-escape radius. Airspace keeps its original
`SNASystem::EvaluateDestruction` behavior by default. TwoVsTwo instead owns a
deterministic swept lethal-circle rule for friendly blue mobiles and refreshes
only their ECS LiDAR components; RLMission red observations remain untouched.
The historical friendly `blue_restricted_zone`, its RNG draw and its SNA effect
against hostile red mobiles are retained unchanged.

## Validation
- `SharedWorldLidarTests`: 61 beams, legacy enum values, forward ray,
  friend/destroyed masking, inside-zone zero, and map-boundary normalization.
- `AirspaceMetaTests`: both old and new names create entities of the same
  canonical class, category and components through the factory table.
- Existing historic CTest/checkpoint tests are required before merge.
- Manual GUI and RLMission training/eval/prod smoke tests are still required
  after successful CI. No model conversion or retraining.

The TwoVsTwo adapter consumes this shared sensor without changing the 61-bin
layout or either legacy RLMission model format.
