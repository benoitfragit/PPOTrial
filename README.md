# PPOTrial: RL core and Navigation2D

C++17 reinforcement-learning interfaces, an extracted LibTorch PPO backend and a
local Navigation2D example. The default build needs only CMake and a C++ compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

`navigation2d` runs a **scripted** straight-line smoke scenario, not a neural model:

```sh
./build/Examples/Navigation2D/navigation2d Examples/Navigation2D/scenarios/default.nav
```

For the optional CPU LibTorch backend, supply a matching LibTorch installation:

```sh
cmake -S . -B build-torch -DPPO_BUILD_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build-torch
```

This also builds `navigation2d_policy` and `navigation2d_train`. Neither is run by
CTest. Training is explicitly opt-in; no model execution is required for the
standard test suite. See their usage messages and [architecture/migration](docs/rl-contract.md).

An optional manual viewer uses installed raylib (`PPO_NAVIGATION_VIEWER=ON`):
arrow keys control the mobile, `R` resets, gray circles are forbidden and green is
the fixed destination. It consumes exactly the same `Environment::step`.

The historical Windows desktop application remains opt-in via
`PPO_BUILD_LEGACY_SIMULATOR=ON`. Its domain-specific wrappers and renderer are not
part of the reusable core. Old checkpoints/scenarios are not Navigation2D models.
