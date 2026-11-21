# PPOKit — journal d'avancement (pour reprendre le travail)

Ce fichier dit **où en est la migration** et **comment reprendre**. Il est mis
à jour à chaque commit de la branche `design/ppokit-framework`.

- Conception : [`docs/architecture/ppokit-framework-design.md`](../docs/architecture/ppokit-framework-design.md)
- Règles du dépôt : [`AGENTS.md`](../AGENTS.md), à relire avant toute modification.
- PR de suivi (brouillon) : [#105](https://github.com/benoitfragit/PPOTrial/pull/105).
  Elle déclenche la CI Windows `desktop-windows.yml`, seule à compiler
  BlueDefense et Airspace avec libtorch. Consulter ses jobs après chaque push :
  `gh api "repos/benoitfragit/PPOTrial/actions/runs?branch=design/ppokit-framework&per_page=10" --jq '.workflow_runs[] | "\(.name) \(.status) \(.conclusion)"'`

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
| `FlightControl_golden_trace_tests` de master : épisodes JSBSim, récompenses, actions, 2 updates PPO, checkpoint hashé | `0x67f683c02a76946c` (depuis l'étape 3 ; avant : `0x471aaf859adb0883`) | Linux x86-64, **GCC 13**, Debug = Release |
| Trace du learner MAPPO Blue (`generalized_learner_tests`, 10 updates, diagnostics à 9 chiffres) | `PPOKit/tests/reference/blue_learner_trace_linux_gcc13.txt` (réenregistrée à l'étape 3) | Linux x86-64, GCC 13, `-O2` |

Sur une autre plateforme, le test affiche son empreinte et se déclare *skipped*
(code 77). **La CI Ubuntu utilise GCC 15 : le test y est toujours sauté.** Il
n'est vérifié que localement, avec GCC 13. La référence de master datait
d'avant #87 et échouait sur master : elle a été actualisée sur le code de
master, inchangé (commit `test(flight): refresh the golden reference…`).

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
| 0. Golden trace Flight | fait | Master (#87) en apporte une plus complète : elle remplace celle de la branche, avec une référence actualisée |
| 1a. `ppokit::artifacts` écriture atomique, 4 implémentations remplacées | fait | Voir le détail ci-dessous |
| 1b. En-tête d'identité et RNG portable des artefacts | reporté | Change les formats sur disque ; regroupé avec l'étape 5 (schémas et hashes) pour ne migrer les formats qu'une fois |
| 2. GAE commun (`ppokit::core::GaeStep`) | fait | Flight et Blue identiques au bit près ; voir ci-dessous |
| 2b. Surrogate PPO commun | reporté à l'étape 3 | Les deux implémentations **diffèrent** : décision à prendre, voir ci-dessous |
| 3a. Tête gaussienne et surrogate PPO communs | fait | Décision B de l'utilisateur ; voir ci-dessous |
| 3b. Têtes Bernoulli, catégorielle, composite | reporté à l'étape 5/6 | Aucun exemple ne les utilise aujourd'hui ; l'AGENTS.md interdit les interfaces sans appelant. Elles arriveront avec le schéma et le codegen |
| 4. Backend rl_tools unique | fusionné dans l'étape 6 | Voir ci-dessous |
| 5. Schéma `task.json`, `ppokit-codegen`, layouts générés | fait (partie forme) | Voir ci-dessous ; scénarios et métriques viendront avec le runtime (étape 7) |
| 6. Codegen des réseaux (inclut l'ancienne étape 4) | à faire | **Prochaine action** |
| 4 à 11 | à faire | |

### Détail de l'étape 1a

- Master (#87) avait entre-temps factorisé Flight dans `flight_atomic_file.hpp`.
  Après la fusion de master, ce fichier est supprimé au profit de PPOKit.
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
- Vérifié (avant la fusion de #87, puis à nouveau après) : Debug et Release, `ctest`, golden trace identique, tests
  Blue `generalized_actor`, `generalized_learner` et `attention` compilés et
  passés, syntaxe de `generalized_patrol_team_agent.cpp` identique à master.
- Non vérifié ici : build desktop Windows (libtorch), à confirmer par la CI.

### Détail de l'étape 2

- `PPOKit::Core` (header-only) : `TemporalDifference<T>` et `GaeStep<T>`,
  documentés (différence entre `terminated` et `truncated`).
- Flight : `ComputeAdvantages` appelle `GaeStep<double>` ; `AdvantageDelta`
  (dans `flight_policy.hpp` depuis #87) est supprimé.
- Blue : `GeneralizedMissionAdvantage` devient un appel à `GaeStep<float>` (le
  nom est gardé pour les tests et le learner).
- Les deux formules avaient déjà la même structure, `((r + γ·b) − v) + (γ·λ)·trace`.
  `PPOKitGaeTests` vérifie l'égalité **bit à bit** avec les deux anciennes
  formules sur 200 000 tirages et sur les zéros signés.
- Vérifié : golden trace Flight inchangée, trace MAPPO Blue identique sur ses
  80 lignes, `ctest` 18/18 en Debug et en Release.

### Détail de l'étape 3a (décision B : « on veut du commun »)

- `ppokit/heads/gaussian.hpp` : `GaussianAxisLogDensity`, `GaussianAxisEntropy`,
  `GaussianAxisLogDensityGradient`, par axe, ce qui permet le masquage d'axes de
  Flight (frein).
- `ppokit/core/ppo_surrogate.hpp` : `EvaluateClippedSurrogate` (log-ratio borné
  à ±20, objectif `min(rA, clip(r)A)`, gradient `−A·r` là où le terme non
  clippé est le minimum, indicateur pour la clip fraction).
- Flight (`flight_policy.hpp`) et Blue (`generalized_patrol_learner.hpp`)
  utilisent ces fonctions ; leurs anciennes formules sont supprimées.
- Écarts mesurés :
  - log-densité : au plus 1 ulp (2,2e-16 en double, 7,2e-7 en relatif en float),
    vérifié par `PPOKitHeadsTests` ;
  - golden trace Flight : les 19 épisodes rejoués ont les mêmes actions et les
    mêmes issues ; seuls les derniers bits des log-probabilités et des poids
    changent ;
  - trace MAPPO Blue : value loss, entropie, KL et clip fraction identiques ;
    écart relatif maximal de 8,6e-6 sur la policy loss (moyenne de termes qui
    se compensent), 3,7e-7 sur les normes de gradient.
- **Reste à faire avant de fusionner dans master (AGENTS.md)** : une évaluation
  multi-seed d'une politique entraînée avec les formules communes, par exemple
  avec les workflows `flight-control-training.yml` et
  `mappo-blue-defense-windows.yml` lancés sur la branche. Les tests unitaires
  ne prouvent pas la convergence.

### Pourquoi l'étape 4 rejoint l'étape 6

rl_tools définit dans un en-tête une fonction **non inline**
(`rl_tools::devices::cpu::sanitize_file_name`, `devices/cpu.h`). Deux unités de
traduction qui incluent rl_tools dans le même namespace provoquent donc une
définition multiple au lien. C'est pour cela que Flight l'inclut sous
`RL_TOOLS_NAMESPACE_WRAPPER ppo_flight_backend` (test de lien :
`tests/other_rltools_backend.cpp`). Tant que chaque exemple inclut rl_tools
lui-même, ce wrapper est la bonne solution. Le backend unique n'apporte
quelque chose que lorsqu'une seule bibliothèque possède tout le code rl_tools,
ce qui est précisément le rôle du code généré (étape 6).

### Détail de l'étape 5

Contrainte de l'utilisateur : d'autres développements sont en cours sur les
modèles, et il rebasera sa branche plus tard. L'étape est donc **additive** :
aucun code d'exécution des exemples n'est modifié.

- `PPOKit::Schema` (`ppokit/schema/task_schema.hpp`) : lecture stricte de
  `task.json` (`ppokit.task/1`), avec l'ordre d'écriture conservé
  (`nlohmann::ordered_json`), et des erreurs de la forme
  `fichier: chemin.de.la.clé: attendu …`. Contenu : observation et critique
  (groupes `fields`, `size` ou `set` avec `capacity`, `features`/`fields` et
  `mask`), actions (tête `gaussian`, `names` ou `size`, `bounds`, `map`),
  récompenses (`label`, `weight`), `agents` (min, max), `namespace`.
- `ShapeHash` : FNV-1a des formes et des noms. Les labels et les poids par
  défaut n'en font pas partie.
- `ppokit-codegen` et la fonction CMake `ppokit_generate_task_layout`
  (`PPOKit/cmake/PPOKitTask.cmake`) produisent un en-tête de constantes : une
  struct par groupe (`Offset`, `Size`, `Capacity`, `Features`, `MaskFeature`,
  enum `Feature`), un enum `Slot` et `SlotNames` quand tout est nommé,
  `Action::Component` et `ComponentNames`, `Reward::Channel`, `Names`,
  `Labels` et `DefaultWeights`.
- `task.json` ajoutés : `Examples/FlightControl3D/task.json` (33 observations
  en 6 groupes, 5 actions, 7 canaux) et
  `Examples/BlueDefense/Agents/BluePatrol/task.json` (observation v10 de 295
  valeurs, critique de 470, 2 actions, 15 canaux).
- Tests de correspondance : `FlightControl_task_layout_tests` et
  `BlueDefenseTaskLayoutTests` vérifient par `static_assert` et à
  l'exécution que le `task.json` et le code décrivent exactement les mêmes
  layouts et noms. Ils échouent dès que l'un change sans l'autre.
- Les poids de récompense effectifs restent dans les configurations des
  exemples (pas de seconde source de vérité). `DefaultWeights` vaut 1 tant
  que la migration du runtime (étape 7) n'a pas eu lieu.

## Prochaine action

Étape 6 : génération des réseaux (`graph.json`) vers rl_tools, avec un backend
unique (l'ancienne étape 4).
1. Définir `ppokit.graph/1` : `dense_stack`, activations, entrée depuis
   `task.json`, tête gaussienne, critique `value`.
2. Générer le MLP de Flight (`mlp_unconditional_stddev`, voir
   `flight_learner.cpp`) et vérifier une sortie forward **identique** à
   l'implémentation actuelle sur des poids et des entrées fixés.
3. Observation : `flight_learner.cpp` instancie déjà le MLP pour trois
   largeurs précompilées (32, 64, 128) et choisit à l'exécution selon
   `training.hiddenWidth`. Le code généré peut reprendre ce motif pour
   éviter une recompilation à chaque essai de largeur.
4. L'attention Blue passera d'abord comme composant interne différentiable.
5. Comme pour l'étape 5, rester additif tant que l'utilisateur n'a pas rebasé
   ses travaux sur les modèles.

## Dernier état de la CI (PR #105, commit `e149d36`, étape 5)

Tout est au vert : Core C++17 tests (configuration sans Flight, où PPOKit
récupère lui-même nlohmann_json) ; FlightControl3D JSBSim (Ubuntu GCC 15 et
Windows, Debug et Release) ; Desktop VS2022 Debug et Release, y compris
`ppokit-codegen` et `FlightControl_task_layout_tests` sous MSVC (vérifié dans
les logs des jobs).

## Pièges connus

- Master avance pendant la migration : avant chaque session, faire
  `git fetch --depth=50 origin master` puis fusionner master dans la branche
  (sans réécrire l'historique). En cas de conflit sur un exemple, prendre la
  version de master, refaire l'intégration PPOKit par-dessus, puis vérifier
  avec `tools/ppokit/verify.sh`.

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
