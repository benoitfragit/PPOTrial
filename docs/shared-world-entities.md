# SharedWorld: model naming and pseudo-LiDAR compatibility

## Purpose
The reusable `Shared/World` C++17 library owns the concrete 2D world entities
`Mobile`, `RestrictedZone`, `Objective`, the unchanged sensor/control/motion
systems and their state layouts. `Examples/Airspace` owns the historic
scenario, scoring, evaluator and graphical renderer, but no longer duplicates
these basic entity classes and algorithms.

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
  `"SAM"` instantiates a **RestrictedZone**. The factory registry also maps
  `ResolveType("Aircraft")` to the reflected `Mobile` type and
  `ResolveType("SAM")` to the reflected `RestrictedZone` type.
- **Legacy C++ headers**: `aircraft.hpp` and `sam.hpp` are thin aliases;
  there are no duplicate implementations or duplicate ECS components.
- **Serialized category values** remain unchanged: `Mobile`/`Aircraft` = 3,
  `RestrictedZone`/`SAM` = 4, `Objective` = 1. Old and new names resolve
  to the **same underlying enum values**.
- The existing Airspace `AirspaceEntities` CMake target becomes an interface
  bridge to `SharedWorld`; code using it still builds and gets the shared headers.

## Platform files
Canonical directories:
`resources/settings/Platforms/Mobile` and
`resources/settings/Platforms/RestrictedZone`.
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
danger region is 1.10 times the no-escape radius; elimination stays governed
by the original `SNASystem::EvaluateDestruction` rule, and an area without
orientation is still treated differently from a Mobile.

## Validation
- `SharedWorldLidarTests`: 61 beams, legacy enum values, forward ray,
  friend/destroyed masking, inside-zone zero, and map-boundary normalization.
- `AirspaceMetaTests`: both old and new names resolve to the same reflected
  class and create entities of the expected canonical category and components.
- Existing historic CTest/checkpoint tests are required before merge.
- Manual GUI and RLMission training/eval/prod smoke tests are still required
  after successful CI. No model conversion or retraining.

No TwoVsTwo gameplay, RLtools, MAPPO or new renderer has been implemented.
