# PPOTrial — Compiled example declaration (C++17)

## Scope

The desktop's example list now belongs to `Examples`, not `Main`. This is a
small composition boundary, **not** a plugin loader and **not** a renderer
abstraction. Airspace remains the default CLI example; interactive startup now shows an example selector. The TwoVsTwo example also uses the
existing renderer, the existing production RLMission agent and the canonical
shared 2D world under `Examples/Shared/World`.

## Declaration and boot order

An example owns `ExampleDefinition` (`Examples/data/example.hpp`):

- `id` — stable command-line name (`airspace` today);
- `windowTitle` — existing desktop window caption;
- `defaultScenario` — path relative to `RESOURCE_PATH`;
- `registerTypes` — registers concrete ECS entities, reflection and **agent
  factories** (Airspace uses `RegisterAirspaceMeta`);
- `loadScenario` — sets the world initializer/rules and instantiates entities
  and agents declared in the scenario.

`ExampleCatalog` (`Examples/data/examplecatalog.hpp`) validates all
entries at construction time, rejects empty or duplicate IDs, and retains
insertion order. It deliberately uses explicit C++17 composition rather than
static auto-registration, global constructors or dynamic plugin loading.

`CreateExampleCatalog` (`Examples/src/examplecatalog.cpp`) defines the
compiled list. To register another example in the future:

1. Provide its own `Definition()` returning an `ExampleDefinition` with
   its registration and scenario-loading callbacks.
2. Add that definition to `CreateExampleCatalog()` and link its library in
   `Examples/CMakeLists.txt`.
3. Add tests for that example's initialization and specific agent contracts.

Do not add concrete examples to `Main/src/main.cpp` or `GenericRLCore`. The selector (`Main/src/exampleselector.cpp`) reads the same catalog and does not hardcode example names. Optional descriptions live in `ExampleDefinition`.

The application still calls, in the original order:

1. Choose the requested example (`--example <id>` directly, or an interactive selection screen when launched without arguments). Headless `--check-example` without an ID retains Airspace as the default.
2. Register the example's concrete entity/agent types.
3. Initialize the serializer path and (unless headless) the existing
   Airspace renderer.
4. Build the environment and load the example scenario.
5. Load serializer settings; run or validate startup with `--check-example`.

`--list-examples`, `--example airspace` and `--check-example` retain their previous behavior. Launching `PPO.exe` without arguments now opens a resizable, high-contrast example selector before loading any scenario. The GUI lists all entries from `ExampleCatalog`, supports mouse and arrow keys / Enter, and exits cleanly if cancelled. Closing the selector does not launch or initialize an example. A missing example name or an
unknown name is still rejected.

## Validation and compatibility

`ExampleCatalogTests` is a dependency-free C++17 executable. It tests ordered
lookup, default selection, missing IDs, duplicate IDs, empty catalogs and
invalid startup callbacks/configuration. Existing desktop tests still exercise
the real `PPO.exe` in headless mode and checkpoint parity. The desktop CI must
compile and run CTest on Windows VS2022 x64 Debug and Release.

The TwoVsTwo example adds no RLtools training, and does not change RLSNA/RLMission networks, weights, observations,
actions, PPO math, checkpoint filenames/formats, JSON settings, scenario
semantics, editor UI or render logic. It neither implements nor prepares
runtime logic for the planned 2v2 or MAPPO examples.

## Deferred, explicitly out of scope

The host still instantiates Airspace's concrete `EnvironmentRenderer` and
links `AirspaceRenderer`. Selecting a renderer supplied by each example is a
separate architectural step, to consider only when a second concrete example
needs a different renderer. No new renderer interface or generic GUI has been
introduced here.
