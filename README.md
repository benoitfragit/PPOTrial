# PPOTrial

Reusable C++17 reinforcement-learning components and one desktop application,
`PPO`, demonstrating Navigation2D: a controlled mobile reaches another moving mobile while
avoiding circular forbidden zones. The existing ImGui/ImPlot/raylib desktop shell
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
`dependencies.cmake`. CMake fetches graphics dependencies only for desktop builds.

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

- **Simulation:** Play/Pause/Stop, clock scale, display flags and camera controls.
- **Navigation View:** the existing docked viewport with orbit/pan/zoom, selection
  follow and trails. The underlying simulation is planar.
- **Minimap / Entities / Inspection:** select either mobile or a zone and
  inspect its current values; double-click an entity to focus the camera.
- **Agents:** manual, training or frozen-policy mode; rollout for a new learner;
  observation snapshot, episode rewards and PPO loss curves; candidate evaluation
  and promotion. Stop before switching policy. Arrow keys steer in manual mode
  while the navigation viewport has focus.
- **Meta Catalog:** the Navigation2D observation/action and entity schema.

Pause preserves the current episode. Stop truncates on one final complete
transition, so the learner receives a final observation for bootstrap. Play starts
a new episode after Stop. Evaluation pauses training and loads an independent,
deterministic policy; Pause/Play suspends/resumes that evaluation, and Stop cancels
it without promotion.

Display preferences are saved explicitly to `NavigationRenderer.json`; docking
uses ImGui's settings. Scenario and score configuration live in
`Examples/Navigation2D/scenarios`. Checkpoints are isolated under
`checkpoints/navigation2d`; the promotion reference is `prod/best.txt`. Scores are
compared only when scenario, criteria, episode count and seed sequence agree.
Use another checkpoint root for a different evaluation context.

## Migration and validation

The former application sources, ECS integration, compatibility headers and extra
Navigation2D entry points have been removed. Historical resources, including
checkpoints and settings, remain stored but are never loaded by this application. Old scenarios and policies are
not compatible with this example. The extracted PPO optimizer and Gaussian network
remain under `RL/torch`; generic code never depends on the desktop or Navigation2D.

See [architecture and lifecycle](docs/rl-contract.md) and [agent rules](AGENTS.md).
GNS/Protobuf transport is an abstraction and wire draft, not an implemented bridge.

GitHub Actions configures headless Debug/Release builds and a Windows Release
desktop build with LibTorch 2.9.1 CPU. It runs only contract tests, never the app
or a model. Check the PR checks for their actual outcome.

Verified in the refactoring environment: direct GCC Release builds and deterministic
contract/runtime tests; syntax compilation of the ported desktop sources against
real dependency headers. No neural model was executed. The preceding implementation passed a complete Windows LibTorch/graphics CI build;
the latest PR checks validate this revision. An interactive desktop smoke test
remains required. See [code review](docs/code-review.md) for fixes and limitations.

The goal mobile has a configurable constant velocity (`goal x y radius vx vy`).
Its full linear path must remain within the world for the episode duration; zero
velocity provides a stationary goal. Observations contain its current position
and velocity. Meeting detection uses the relative motion of both mobiles during
each step, while forbidden zones still terminate the controlled mobile's episode.
The new `navigation2d-v2` schema has `10 + 3 * obstacle_count` observation values.
Policies trained with the previous observation width cannot be reused. No policy
training or avoidance performance has been measured by this change.
