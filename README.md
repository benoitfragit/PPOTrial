# PPOTrial

Reusable C++17 reinforcement-learning components and one desktop application,
`PPO`, demonstrating Navigation2D: an EnTT world with N mobiles, per-mobile
destinations/controllers, and circular forbidden-zone entities. Each mobile also receives a 61-ray 360-degree pseudo-LiDAR against forbidden zones and work-area boundaries; proximity penalties provide dense avoidance feedback. The existing ImGui/ImPlot/raylib desktop shell
is connected to this environment. There is no legacy application or separate viewer.

## Build

Headless libraries and tests (no graphics, LibTorch or model execution):

```sh
cmake -S . -B build/headless -DPPO_BUILD_DESKTOP=OFF -DPPO_BUILD_TORCH=OFF
cmake --build build/headless --config Release
ctest --test-dir build/headless -C Release --output-on-failure
```

Desktop: provide a LibTorch distribution matching your compiler, platform and
Debug/Release ABI. The previous backend used LibTorch 2.9.1 CPU. Desktop dependencies
retain ImGui v1.92.1-docking, ImPlot v0.17 and raylib 5.5; rlImGui is pinned in
`dependencies.cmake`. CMake fetches graphics dependencies only for desktop builds. The example restores
EnTT v3.16.0; `cmake -S RL -B build/rl` still builds the generic core without it.

```sh
cmake -S . -B build/desktop -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
cmake --build build/desktop --config Release
```

On Windows use a matching MSVC generator and LibTorch package. Torch DLLs are copied
beside `PPO.exe`. The app loads the shipped scenario by default; its first optional
argument selects a `.nav` scenario; a second optional argument selects the checkpoint
root. The default scenario files are copied beside the executable under `scenarios/`,
so the source checkout is not needed at runtime. It starts stopped in manual mode and does not
create or run a neural policy until explicitly requested in the UI.

## Desktop

- **Simulation:** Play/Pause/Stop, clock scale, display flags and camera controls; optional per-selected-mobile LiDAR rays.
- **Navigation View:** the existing docked viewport with orbit/pan/zoom, selection
  follow and trails. The underlying simulation is planar.
- **Minimap / Entities / Inspection:** select any mobile or zone and
  inspect its current values; double-click an entity to focus the camera.
- **Agents:** manual, training or frozen-policy mode; rollout for a new learner;
  observation snapshot, episode rewards and PPO loss curves; candidate evaluation
  and promotion. Stop before switching policy. Arrow keys steer in manual mode
  while the navigation viewport has focus. They affect the selected manual mobile.
  Additional mobiles can each use their own learner or frozen Prod/N-1 policy.
- **Meta Catalog:** the Navigation2D observation/action and entity schema.

Pause preserves the current episode. Stop truncates on one final complete
transition, so the learner receives a final observation for bootstrap. Play starts
a new episode after Stop. Evaluation pauses training and loads an independent,
deterministic policy; Pause/Play suspends/resumes that evaluation, and Stop cancels
it without promotion.

Display preferences are saved explicitly to `NavigationRenderer.json`; docking
uses ImGui's settings. Scenario and score configuration live in
`Examples/Navigation2D/scenarios`. Checkpoints are isolated under
`checkpoints/navigation2d`; promotion references are `prod/mobile-<id>/best.txt`. Scores are
compared only when scenario, criteria, episode count and seed sequence agree.
Use another checkpoint root for a different evaluation context.

## Migration and validation

The former application sources, compatibility headers and extra Navigation2D entry
points have been removed. ECS support is restored in the example and renderer;
only the generic RL core is independent of EnTT. Historical resources, including
checkpoints and settings, remain stored but are never loaded by this application. Old scenarios and policies are
not compatible with this example. The extracted PPO optimizer and Gaussian network
remain under `RL/torch`; generic code never depends on the desktop or Navigation2D.

See [architecture and lifecycle](docs/rl-contract.md) and [agent rules](AGENTS.md).
GNS/Protobuf transport is an abstraction and wire draft, not an implemented bridge.

GitHub Actions configures headless Debug/Release builds and a Windows Release
desktop build with LibTorch 2.9.1 CPU. It runs only contract tests, never the app
or a model. Check the PR checks for their actual outcome.

Verified in the refactoring environment: CMake headless and direct GCC Release builds and deterministic
contract/runtime tests; syntax compilation of the ported desktop sources against
real dependency headers. No neural model was executed. The preceding implementation passed a complete Windows LibTorch/graphics CI build;
the latest PR checks validate this revision. An interactive desktop smoke test
remains required. See [code review](docs/code-review.md) for fixes and limitations.

## Episode starts and generic motion endurance

A Navigation2D episode can start from its original fixed scene, use a randomized
set of mobile states (seeded resets are reproducible), or replay complete-world
presets from a manifest. The desktop **Simulation > Episode initialization**
selects these modes; use `scenarios/evaluation.cases` for the bundled default
scenario. Each line names a `navigation2d-v3` file relative to the manifest.
Presets retain every mobile/zone ID and all attached controller relationships.
Frozen-policy evaluations use the same configured distribution and record the
configuration in the candidate score context.

Editable rewards can be configured independently for each mobile in the **Agents** panel.
Optional `reward` rows persist eight nonnegative channel weights:
`reward mobile_id progress step_cost arrival collision zone_caution edge_caution fuel_cost failure`.
These change the scoring signals, not the shared motion or RL backend.

For the former minimum-speed/fuel feature, an optional `energy` row may be added:
`energy mobile_id minimum_speed fuel_capacity idle_rate speed_cubic_rate turn_rate_cost`.
The physical 2D model uses `idle + speed_cubic * speed^3 + turn_cost * turn_rate^2`
units per second; capacity zero disables fuel. `energy-demo.nav` shows the feature.
Energy exhaustion or too-low speed ends the mobile's episode; the inspector shows
energy remaining and last-step consumption. This remains an example-level behavior,
not a dependency of the generic RL engine.

## Multiple entities and controllers

`scenarios/multiple.nav` contains five mobiles and two zones. Start `PPO` with that
path, or use **Entities / Scenario / Load scenario** while stopped without policies.
Each `mobile` row defines a stable ID, name, initial position/heading/speed,
destination ID (`0` for none), and meeting radius. Each `zone` row creates an ECS
entity. See [the v3 schema](docs/rl-contract.md#navigation-semantics-and-configuration).

Select a mobile in **Entities**. In **Agents**, choose its Training or Frozen Prod
mode and apply its controller. The primary mode controls the externally driven
Gymnasium agent; every additional mobile has independent controls and telemetry.
**Evaluate selected learner / promote** freezes all controllers in a separate
world and promotes only the selected learner. Prod selection never follows a
new promotion automatically. All decisions precede the single shared physics step.

The v3 scenario format replaces the pair-specific v2 format. Observations retain
`10 + 3 * zone_count + 61` values from each mobile's viewpoint; additional unrelated
mobiles are not automatically added to the policy input. All mobiles now obey zone
and world bounds, including coasting mobiles. Do not assume old trained models
remain behaviorally equivalent merely because tensor widths match.
