# PPOKit — journal d'avancement (pour reprendre le travail)

Ce fichier dit **où en est la migration** et **comment reprendre**. Il est mis
à jour à chaque commit de la branche `design/ppokit-framework`.

- Conception : [`docs/architecture/ppokit-framework-design.md`](../docs/architecture/ppokit-framework-design.md)
- Règles du dépôt : [`AGENTS.md`](../AGENTS.md), à relire avant toute modification.

## Reprendre en 5 minutes

1. `git checkout design/ppokit-framework && git pull`
2. Lire la section **État** ci-dessous, puis **Prochaine action**.
3. Tout vérifier d'une commande (builds Debug et Release, CTest, golden
   trace Flight, tests Blue hors desktop, trace MAPPO Blue au bit près) :

   ```sh
   tools/ppokit/verify.sh            # ou: tools/ppokit/verify.sh release
   ```

   Le détail des commandes suit.
4. Configurer et construire (Linux, sans desktop ni libtorch, comme la CI
   `flight-control.yml`). Seuls `build/debug` et `build/release` sont utilisés :

   ```sh
   cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DPPO_BUILD_DESKTOP=OFF -DPPO_BUILD_CORE_TESTS=ON -DPPO_BUILD_FLIGHT_CONTROL=ON
   cmake --build build/release -j2
   (cd build/release && ctest)
   ```

   Même chose avec `build/debug` et `-DCMAKE_BUILD_TYPE=Debug`. Le premier build
   compile JSBSim, ce qui prend environ 15 minutes sur 2 cœurs.
5. Garde-fou numérique : `build/<config>/Examples/FlightControl3D/FlightControl_golden_trace_tests`
   doit afficher la référence ci-dessous en Debug **et** en Release.

## Garde-fous numériques

| Garde-fou | Valeur de référence | Plateforme |
|---|---|---|
| `FlightControl_golden_trace_tests` (3 updates PPO, checkpoint complet hashé) | `0x0f0414a118b47b70` | Linux x86-64, GCC 13, Debug = Release |
| Trace du learner MAPPO Blue (`generalized_learner_tests`, 10 updates, diagnostics à 9 chiffres) | `PPOKit/tests/reference/blue_learner_trace_linux_gcc13.txt` | Linux x86-64, GCC 13, `-O2` |

Sur une autre plateforme, le test affiche son empreinte et se déclare *skipped*
(code 77). Pour y enregistrer une référence, lancer le test avec `--print`
**sur le code de master**, puis l'ajouter dans `golden_trace_tests.cpp`.

## Vérifier le code Blue sans le desktop

Le code d'entraînement Blue n'est compilé que par le build desktop, qui a
besoin de libtorch (indisponible hors Windows CI). Les tests Blue qui
exercent les checkpoints ne dépendent pourtant que de rl_tools ; on les
compile à la main **dans `build/release`** :

```sh
B=Examples/BlueDefense/Agents/BluePatrol; R=build/release/_deps; O=build/release/manual_checks
mkdir -p $O
INC="-IPPOKit/include -I$B/Core -I$B/Runtime -I$B/Training -IExamples/BlueDefense/Evaluation \
     -IFramework/RL/Core/include -I$R/rltools-src/include -I$R/json-src/include"
for t in generalized_actor_tests generalized_learner_tests attention_tests; do
  g++ -std=c++17 -O2 $INC Examples/BlueDefense/tests/$t.cpp PPOKit/src/artifacts/atomic_file.cpp -o $O/$t && $O/$t > /dev/null; echo "$t rc=$?"
done
```

Le fichier `generalized_patrol_team_agent.cpp` se vérifie en `-fsyntax-only`
avec EnTT v3.16.0 cloné dans `build/release/manual_checks/entt`. Avec GCC,
`Framework/Simulation/Entities/data/baseentity.hpp:113` produit une erreur
**déjà présente sur master** (code accepté par MSVC) : ce n'est pas une
régression.

La validation complète du desktop (BlueDefense, Airspace) passe par la CI
Windows `desktop-windows.yml`, déclenchée par une PR vers `master`.

## État

| Étape (plan §16) | État | Notes |
|---|---|---|
| 0. Golden trace Flight | fait | Absente de master : ajoutée avant toute refonte |
| 1a. `ppokit::artifacts` écriture atomique, 4 implémentations remplacées | fait | Voir le détail ci-dessous |
| 1b. En-tête d'identité et RNG portable des artefacts | reporté | Change les formats sur disque ; regroupé avec l'étape 5 (schémas et hashes) pour ne migrer les formats qu'une fois |
| 2. GAE commun (`ppokit::core::GaeStep`) | fait | Flight et Blue identiques au bit près ; voir ci-dessous |
| 2b. Surrogate PPO commun | reporté à l'étape 3 | Les deux implémentations **diffèrent** : décision à prendre, voir ci-dessous |
| 3. Têtes de sortie | à faire | **Prochaine action** |
| 4 à 11 | à faire | |

### Détail de l'étape 1a

- Bibliothèque `PPOKit/` (cible `PPOKit::Artifacts`), sans dépendance tierce,
  ajoutée avant la séparation desktop/headless du `CMakeLists.txt` racine.
- `WriteFileAtomically(path, writer, options, context)` : écriture dans
  `path.tmp`, puis commit avec une copie de rollback.
- Comportements conservés à l'identique :
  - Flight (learner, manifeste BEST) : copie `.previous` **conservée**, car la
    CI `flight-control-training.yml` l'archive. Le manifeste ne crée pas son
    dossier.
  - Blue (`SavePatrolAtomically`, score promu) : copie `.bak` supprimée après
    succès.
- Différences assumées :
  - Messages d'erreur unifiés : propriétaire, action et chemin. Aucun test ni
    outil ne dépendait des anciens textes.
  - Le temporaire est supprimé après un échec côté Flight, comme Blue le
    faisait déjà.
  - L'erreur levée est `ppokit::artifacts::ArtifactWriteError` (dérivée de
    `std::runtime_error`) au lieu de `std::filesystem::filesystem_error`.
- Vérifié : Debug et Release, `ctest` 17/17, golden trace identique, tests
  Blue `generalized_actor`, `generalized_learner` et `attention` compilés et
  passés, syntaxe de `generalized_patrol_team_agent.cpp` identique à master.
- Non vérifié ici : build desktop Windows (libtorch), à confirmer par la CI.

### Détail de l'étape 2

- `PPOKit::Core` (header-only) : `TemporalDifference<T>` et `GaeStep<T>`,
  documentés (différence entre `terminated` et `truncated`).
- Flight : `ComputeAdvantages` appelle `GaeStep<double>` ; `AdvantageDelta`
  est supprimé.
- Blue : `GeneralizedMissionAdvantage` devient un appel à `GaeStep<float>` (le
  nom est gardé pour les tests et le learner).
- Les deux formules avaient déjà la même structure, `((r + γ·b) − v) + (γ·λ)·trace`.
  `PPOKitGaeTests` vérifie l'égalité **bit à bit** avec les deux anciennes
  formules sur 200 000 tirages et sur les zéros signés.
- Vérifié : golden trace Flight inchangée, trace MAPPO Blue identique sur ses
  80 lignes, `ctest` 18/18 en Debug et en Release.

### Surrogate PPO : différences constatées (à trancher à l'étape 3)

| | FlightControl3D (`ClippedLoss`) | BlueDefense (learner) |
|---|---|---|
| Borne du log-ratio avant `exp` | ±80 | ±20 |
| Gradient actif | `A ≥ 0 ? r ≤ 1+ε : r ≥ 1−ε`, et log-ratio non saturé | `r·A ≤ clip(r)·A` |
| Différence observable | à `A = 0` exactement (signe de zéro du gradient) et quand le log-ratio sature | |

Unifier changerait les calculs dans des cas limites. La fonction commune doit
donc soit prendre ces conventions en paramètre, soit en imposer une avec
l'accord de l'utilisateur et une nouvelle référence de trace.

## Prochaine action

Étape 3 : `ppokit/heads`.
1. Têtes gaussienne, Bernoulli, catégorielle, composite : log-prob,
   entropie, action déterministe, gradient.
2. Surrogate commun (voir ci-dessus) : proposer à l'utilisateur de garder les
   deux conventions en paramètre, ce qui garantit des traces identiques.
3. Brancher Flight (`hybrid_policy.hpp`) puis Blue (learner), en gardant les
   deux garde-fous identiques.

## Pièges connus

- Le `.gitignore` du dépôt exclut tout dossier nommé `artifacts/` (règle .NET).
  Les dossiers de PPOKit portant ce nom sont réintégrés explicitement. Après
  avoir ajouté un dossier, vérifier `git status --ignored PPOKit`.
- Le build desktop (BlueDefense, Airspace) n'est pas compilable ici, faute de
  libtorch : passer par les vérifications manuelles ci-dessus et par la CI
  Windows.

## Décisions prises avec l'utilisateur

- Génération de code depuis JSON pour rl_tools ; libtorch derrière le même
  frontal pour Airspace.
- Composants DLL boîte noire, câblés par leurs ports.
- Les trois exemples migrent et deviennent des exemples de référence, faciles
  à prendre en main.
- RLMission migre en dernier, sur approbation explicite.
- Interface : curriculum, récompenses, paramètres PPO avec infobulles, sondes,
  GPU via libtorch.
