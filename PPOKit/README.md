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

### Réseaux décrits dans `graph.json`

`graph.json` décrit l'acteur et le critique. Leurs tailles d'entrée et de sortie
viennent de `task.json`, donc les deux fichiers ne peuvent pas se contredire.
Un réseau écrit à la main reste possible (`"model": "external"`, cas de
l'acteur à attention de BlueDefense).

```json
{
  "schema": "ppokit.graph/1",
  "actor":  { "model": "mlp", "hidden": { "layers": 2, "width": 128 }, "activation": "tanh",
              "log_std": "state_independent" },
  "critic": { "model": "mlp", "input": "observation", "hidden": { "layers": 2, "width": 128 },
              "activation": "tanh" }
}
```

```cmake
ppokit_generate_networks(TARGET my_target TASK task.json GRAPH graph.json HEADER my_networks.hpp)
```

```cpp
#include <rl_tools/operations/cpu_mux.h>   // d'abord les opérations rl_tools
#include "my_networks.hpp"
using Actor = ppokit::tasks::my_task::networks::Actor::Network<Index, BatchSize>;
```

Les types générés sont exactement ceux que les exemples écrivaient à la main :
le MLP de FlightControl3D chargé depuis un checkpoint de `FlightLearner` donne
les mêmes sorties au bit près, et le critique de BlueDefense est le même type
rl_tools.

### Entraîner une tâche : collecte, rollout et learner PPO

En mode *Owned*, l'exemple écrit seulement sa tâche (`Reset`, `Step`) et
PPOKit mène la boucle. Le learner rl_tools s'appuie sur les réseaux générés
depuis `graph.json` :

```cpp
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include "my_networks.hpp"
#include "ppokit/backend/rltools/gaussian_mlp_learner.hpp"

using Learner = ppokit::backend::rltools::GaussianMlpLearner<networks::Actor, networks::Critic>;
Learner learner(settings);                      // PpoSettings : gamma, lambda, clip, learning rates…
Learner::Buffer buffer(2048);                   // alloué une fois
ppokit::runtime::RolloutCollector<MyTask, Learner, Learner::Buffer> collector(task, learner);
for (int update = 0; update < 100; ++update)
{
    collector.Collect(buffer);                  // épisodes enchaînés, troncatures bootstrapées
    auto const statistics = learner.Update(buffer);   // pertes, KL, clip fraction, variance expliquée
}
auto const action = learner.Mean(observation);  // politique déterministe
```

Exemple complet : `PPOKit/tests/backend_learner_tests.cpp` (couloir jouet
résolu en 30 updates, de façon reproductible à partir du seed).

### Évaluer une politique sur des seeds indépendants

Une suite d'évaluation est un fichier :

```json
{
  "schema": "ppokit.eval/1",
  "suite": "corridor_smoke",
  "seeds": [1001, 1002, 1003],
  "episodes": 2,
  "policy": "deterministic",
  "gates": { "success_rate": ">= 1.0", "mean_length": "<= 8" }
}
```

```cpp
#include "ppokit/eval/evaluator.hpp"
auto const suite = ppokit::eval::LoadEvaluationSuite("eval/smoke.json");
ppokit::eval::RequireDisjointSeeds(suite, trainingSeeds);   // refuse un seed d'entraînement
auto const report = ppokit::eval::Evaluate(suite, MakeTask, learner, maximumSteps);
std::cout << report.ToJson();                               // métriques, gates, épisodes
```

Métriques intégrées : `mean_return`, `success_rate` (épisodes terminés),
`truncation_rate`, `mean_length`. La politique n'est jamais échantillonnée
pendant l'évaluation, ce qui laisse intact le générateur aléatoire de
l'entraînement. CMake : `PPOKit::Evaluation`.

### Scénarios et curriculum

`task.json` déclare ce qui varie d'un épisode à l'autre :

```json
"scenario": {
  "wind_kt":   { "type": "float", "range": [0, 25], "default": 0 },
  "start":     { "type": "enum", "values": ["rolling", "airborne"], "default": "airborne" },
  "gusts":     { "type": "bool", "default": false },
  "opponents": { "type": "int", "range": [0, 10] }
}
```

`train.json` décrit le curriculum, du plus facile au plus difficile. Un
paramètre non mentionné par une étape prend sa valeur par défaut :

```json
{
  "schema": "ppokit.train/1",
  "curriculum": {
    "window": 200, "promote_at": 0.8, "rehearse": 0.1,
    "stages": [
      { "name": "calm",  "sample": { "opponents": 0, "start": "rolling" } },
      { "name": "windy", "sample": { "opponents": { "uniform": [1, 3] }, "wind_kt": { "uniform": [5, 15] },
                                     "start": { "choice": { "rolling": 1, "airborne": 3 } } } },
      { "name": "full",  "window": 400, "sample": { "opponents": { "normal": { "mean": 8, "std": 4 } } } }
    ]
  }
}
```

Une étape est promue quand le taux de succès sur une fenêtre d'épisodes de
cette étape atteint `promote_at`. La fenêtre repart à zéro après chaque
décision. Une part `rehearse` des épisodes rejoue une étape précédente sans
compter pour la promotion.

```cpp
ppokit::runtime::Curriculum curriculum(*ppokit::schema::LoadTrainingPlan("train.json", task).curriculum);
ppokit::runtime::LoadCurriculumStage("latest/stage.txt", curriculum);   // reprise
auto const draw = curriculum.Sample(rng, values);
auto const scenario = Layout::Scenario::From(values);                  // champs typés générés
// ... épisode ...
if (curriculum.Record(draw, success)) { ppokit::runtime::SaveCurriculumStage("latest/stage.txt", curriculum); }
```

## Organisation

```
PPOKit/
  include/ppokit/<module>/   en-têtes publics, sans rl_tools, libtorch ni GUI
  src/<module>/              implémentations
  tests/                     tests sans dépendance, lancés par CTest
```
