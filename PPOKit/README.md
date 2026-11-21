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

## Organisation

```
PPOKit/
  include/ppokit/<module>/   en-têtes publics, sans rl_tools, libtorch ni GUI
  src/<module>/              implémentations
  tests/                     tests sans dépendance, lancés par CTest
```
