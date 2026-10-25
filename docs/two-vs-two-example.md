# PPOTrial — TwoVsTwo baseline architecture

## Scope and launch

The binary remains PPO.exe. Launch without arguments to choose an example in the graphical selector, or launch the original example directly with PPO.exe --example airspace. Launch the new scenario directly with PPO.exe --example two-vs-two. Its headless
bootstrap check is PPO.exe --example two-vs-two --check-example.

Examples/Shared/World owns reusable 2D Mobile, Objective, RestrictedZone,
kinematics, SNA/control and the exact 61-bin pseudo-LiDAR implementations.
Examples/Airspace retains the original scenario, production agents and world
rules; Examples/TwoVsTwo composes these with its own scenario, positioning
logic and renderer metadata. Core remains domain-neutral.

## Scenario

The unchanged global workspace is approximately 44.66 km north × 71.45 km
east. A highlighted defended inner rectangle (36 km × 40 km) is centered
at (0, 0); it is visualization metadata, not an additional collision boundary.

Two blue Mobile defenders are initially stationary without agents on a
randomly chosen east/west side of this area. The static blue Objective is
behind the defenders. A friendly RestrictedZone is between the teams.
The blue units can use their historical SNA range/angle destruction system,
but no defender RLtools policy, patrol logic, or training is active yet.

One mandatory red Mobile and an optional second red Mobile spawn on the
opposite side, outside the defended area but within the global workspace.
The red mobile positions/headings/speeds vary each initialization.
red_2 can start destroyed with configurable probability (default 0.15).
To test 2-vs-1, set red_2.enabled=false in TwoVsTwo.json.
Every red Mobile has target blue_objective resolved from its JSON declaration
and uses the *existing* RLMission production agent / prod/model.pt.

The same 61-ray pseudo-LiDAR, world boundary intersection, zone threat and
destruction rules are reused; no RLMission checkpoint, observation, action
or network format has changed. Every car also has a sensor component, including
agent-free defenders. The initial red positions remain in the global bounds,
so the pretrained agent is not immediately penalized for leaving the world.

The renderer uses the existing 3D tactical view and optional 2D minimap.
The car platform archetype triggers a procedural car drawing with heading
and side color; the inner defended rectangle is shown in both views. Enable
the minimap using Panels > Minimap. No third-party model asset is required.

## Build

RLtools is fetched via FetchContent from the immutable v1.0.0 commit and
exposed as RLtools::Core through TwoVsTwoDefenderBackend (C++17, CPU-only).
The upstream RLtools executables/tests and optional backends are disabled.
Its compile smoke test verifies header access. LibTorch remains the
production inference backend for red, unchanged.

Validation: dependency-free core tests, desktop VS2022 x64 Debug/Release,
CTest including TwoVsTwoSpawnTests / TwoVsTwoExampleBootstrap, plus GUI
review. No training, MAPPO, 2D vehicle physics or blue patrol is in this PR.
