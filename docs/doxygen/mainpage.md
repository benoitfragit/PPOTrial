\mainpage PPOTrial Documentation

PPOTrial provides a C++17 environment contract and PPO backend with a Navigation2D
example, used by a single ImGui/ImPlot/raylib desktop application.

- `RL`: generic spaces, transitions, episode sessions, evaluation and promotion.
- `RL/torch`: extracted PPO optimizer, rollout collector, network and policy adapters.
- `Examples/Navigation2D`: planar motion, destination, obstacles and scenario files.
- `Main`: desktop orchestration and the sole application entry point.
- `Renderer`: the ported docked desktop views and controls.

See `docs/rl-contract.md` for lifecycle, schema, migration and validation limits,
and the root README for build commands. The old ECS application and JSON formats
are removed; they have no compatibility layer.

Run `generatedoc.bat` on Windows to generate HTML under `docs/generated/html`.
Doxygen and optional Graphviz remain independent from the application build.
