# Third-Party Assets

This repository uses third-party 3D assets that are not covered by the MIT
license of the project source code.

## Aircraft models

- `resources/3D/aircraft/blue_fighter_poly_pizza.glb`
- `resources/3D/aircraft/red_jet_poly_pizza.glb`

These assets keep their original license and attribution requirements.
Refer to:

- `resources/3D/aircraft/README.md`

When redistributing the project, keep this notice together with the asset
attribution file.

## FlightControl3D simulation data

FetchContent pins JSBSim 1.3.3 at `66f7fafc07fe13e7ede9db20873030fed2881713`.
The JSBSim library declares LGPL; its `aircraft/c310/c310.xml` declares GPL
separately. The original C310 folder, engine/system XML notices and JSBSim
`COPYING` are preserved when staged under the build's FlightControl3D data.
These upstream simulation assets are not relicensed under PPOTrial's MIT license.
