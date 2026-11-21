# PPOKit

Framework commun des exemples PPOTrial : PPO/MAPPO, observations,
récompenses, artefacts, évaluation, inférence et studio.

> En cours de construction. Conception :
> [`docs/architecture/ppokit-framework-design.md`](../docs/architecture/ppokit-framework-design.md).
> Avancement et reprise : [`PROGRESS.md`](PROGRESS.md).

## Disponible aujourd'hui

### `PPOKit::Artifacts` — fichiers d'artefacts transactionnels

Un lecteur ne voit jamais un checkpoint à moitié écrit :

```cpp
#include "ppokit/artifacts/atomic_file.hpp"

struct PolicyWriter
{
    Policy const& policy;
    void operator()(std::ostream& stream) const { policy.Write(stream); }
};

// `model.txt.tmp` est écrit, puis remplace `model.txt` ; en cas d'échec,
// l'ancien fichier reste intact et une ArtifactWriteError décrit la cause.
ppokit::artifacts::WriteFileAtomically("models/model.txt", PolicyWriter{policy},
    ppokit::artifacts::AtomicWriteOptions{}, "MyExample checkpoint");

// Pour garder la génération précédente à côté (`model.txt.previous`) :
auto const options = ppokit::artifacts::AtomicWriteOptions::KeepingPrevious(".previous");
```

CMake : `target_link_libraries(<cible> PRIVATE PPOKit::Artifacts)`.

### `PPOKit::Core` — briques PPO (header-only)

```cpp
#include "ppokit/core/gae.hpp"
#include "ppokit/core/ppo_surrogate.hpp"
#include "ppokit/heads/gaussian.hpp"

// GAE, en remontant la trajectoire : `terminated` coupe le bootstrap,
// `truncated` (limite de temps) le garde mais arrête la trace.
advantage = ppokit::core::GaeStep(reward, value, nextValue, advantage, gamma, lambda, terminated, truncated);

// Politique gaussienne, axe par axe (un axe inactif est simplement sauté).
logPi += ppokit::heads::GaussianAxisLogDensity(sample, mean, logStd);

// Surrogate PPO clippé, puis règle de chaîne vers les sorties de l'acteur.
auto const surrogate = ppokit::core::EvaluateClippedSurrogate(logPi, oldLogPi, advantage, clip);
auto const gradient = ppokit::heads::GaussianAxisLogDensityGradient(sample, mean, logStd);
double const dLossDMean = surrogate.logProbabilityGradient * gradient.mean;
```

CMake : `target_link_libraries(<cible> PUBLIC PPOKit::Core)`.

FlightControl3D et BlueDefense utilisent exactement ces fonctions.

### `PPOKit::Schema` et `ppokit-codegen` — décrire une tâche dans `task.json`

Un fichier `task.json` décrit la forme de ce que l'agent observe et fait :
groupes d'observation (champs nommés, vecteurs, ensembles d'entités masqués),
entrée du critique, groupes d'actions, canaux de récompense. Exemples complets :
[`Examples/FlightControl3D/task.json`](../Examples/FlightControl3D/task.json),
[`Examples/BlueDefense/Agents/BluePatrol/task.json`](../Examples/BlueDefense/Agents/BluePatrol/task.json).

```json
{
  "schema": "ppokit.task/1",
  "name": "my_task",
  "observation": {
    "own":    { "fields": ["speed", "heading_error"] },
    "lidar":  { "size": 61 },
    "tracks": { "set": { "capacity": 10, "features": 13, "mask": "last" } }
  },
  "actions": { "motion": { "head": "gaussian", "names": ["turn", "acceleration"], "bounds": [-1, 1], "map": "clip" } },
  "rewards": { "Progress": {}, "Crash": { "label": "Crash penalty", "weight": 10 } }
}
```

CMake génère un en-tête de layout à chaque modification du fichier :

```cmake
ppokit_generate_task_layout(TARGET my_target TASK task.json HEADER my_task_layout.hpp)
```

```cpp
#include "my_task_layout.hpp"
using Layout = ppokit::tasks::my_task::TaskLayout;
float const lidarFirst = observation[Layout::Observation::Lidar::Offset];
static_assert(Layout::Action::Size == 2);
```

Une faute dans le fichier est signalée avec son chemin exact :

```
ppokit-codegen: task.json: observation.tracks.set.capacty: unknown key (accepted: capacity, features, fields, mask)
```

## Organisation

```
PPOKit/
  include/ppokit/<module>/   en-têtes publics, sans rl_tools, libtorch ni GUI
  src/<module>/              implémentations
  tests/                     tests sans dépendance, lancés par CTest
```
