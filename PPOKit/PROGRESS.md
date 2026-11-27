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

La PR 105 intègre maintenant le master `b5ea319` (PR 112 et 113). L’audit du
25 novembre donne les références T-6 340/870 et PC-7 610, les résultats par
épisode et les écarts encore ouverts. [Audit d’intégration](../docs/audits/ppokit-integration/README.md)

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
| `FlightControl_golden_trace_tests` sur le scénario T-6 340 archivé, avec épisodes JSBSim, récompenses, actions, deux updates PPO et checkpoint | `0x20c6e13722644e9d` | Linux x86-64, GCC 13, libm Ubuntu 24.04, Debug et Release |
| Trace du learner MAPPO Blue (`generalized_learner_tests`, 10 updates, diagnostics à 9 chiffres) | `PPOKit/tests/reference/blue_learner_trace_linux_gcc13.txt` (réenregistrée à l'étape 3) | Linux x86-64, GCC 13, `-O2` |

Les deux jobs Flight Ubuntu 24.04 sélectionnent GCC 13 et activent
`PPO_FLIGHT_REQUIRE_GOLDEN_REFERENCE`. Un saut y fait échouer CTest.
Avec GCC 14 ou MSVC, le test affiche l’empreinte locale et retourne le code 77.

## Vérifier le code Blue sans le desktop

Le code d'entraînement Blue n'est compilé que par le build desktop, qui a
besoin de libtorch (indisponible hors Windows CI). Les tests Blue qui
exercent les checkpoints ne dépendent pourtant que de rl_tools ; on les
compile à la main **dans `build/release`** :

```sh
B=Examples/BlueDefense/Agents/BluePatrol; R=build/release/_deps; O=build/release/manual_checks
mkdir -p $O
INC="-IPPOKit/include -I$B/Core -I$B/Runtime -I$B/Training -IExamples/BlueDefense/Evaluation \
     -IFramework/RL/Core/include -I$R/rltools-src/include -I$R/json-src/include \
     -Ibuild/release/BlueDefenseCoreTests/ppokit_generated/BlueDefenseNetworkTypeTests"
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
| 6a. `graph.json` et types rl_tools générés (MLP, modèles externes) | fait | Voir ci-dessous |
| 6b. Brancher les learners sur les types générés | fait : Flight (golden inchangée) et critique Blue (trace MAPPO inchangée) | Voir ci-dessous |
| 6c. Attention Blue comme composant interne | à faire | |
| 7a. Rollout, collecte Owned, learner PPO rl_tools générique | fait | Voir ci-dessous |
| 7b. Mode Hosted (MAPPO, pilotage par l'hôte) | à faire | |
| 8a. Scénarios (`task.json`) et curriculum (`train.json`), moteur de curriculum | fait | Voir ci-dessous |
| 9. Graphes de blocs (encodeurs, attention, pooling) exécutés par un moteur PPOKit | **en cours** | Demande de l'utilisateur ; conception : `docs/architecture/ppokit-graph-blocks.md` |
| 7c. Suites d'évaluation `ppokit.eval/1` | fait | Voir ci-dessous ; grilles de scénarios et métriques propres à la tâche viendront avec le schéma des scénarios |
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

### Détail de l'étape 6a

- `ppokit.graph/1` (`ppokit/schema/network_schema.hpp`) : acteur et critique,
  `"model": "mlp"` (`hidden.layers`, `hidden.width`, `activation` parmi
  tanh, relu ou identity, `log_std: state_independent` pour l'acteur) ou
  `"model": "external"`. Le critique lit l'observation, ou l'entrée `critic`
  de la tâche (par défaut quand elle existe). Les tailles viennent du
  `task.json`.
- Les helpers privés communs aux deux schémas ont été extraits :
  `src/schema/json_validation.*` (validation, messages) et
  `src/schema/code_writer.hpp` (mise en forme du code généré).
- `ppokit-codegen` a deux sous-commandes : `layout` et `networks`. CMake
  fournit `ppokit_generate_task_layout` et `ppokit_generate_networks`.
- L'en-tête généré n'inclut pas rl_tools : il exige que l'appelant l'ait
  inclus avant (`#error` sinon) et suit son `RL_TOOLS_NAMESPACE_WRAPPER`. La
  largeur est un paramètre template, avec par défaut celle de `graph.json`,
  ce qui garde possible le choix 32/64/128 de Flight.
- `graph.json` ajoutés : Flight (acteur et critique MLP 2×128 tanh) et Blue
  (acteur `external`, critique MLP 2×256 ReLU sur l'entrée centralisée de
  470).
- Preuves :
  - `FlightControl_network_parity_tests` : un checkpoint écrit par
    `FlightLearner` est relu dans les réseaux générés (le fichier est consommé
    entièrement), et l'acteur généré donne les **mêmes moyennes au bit près**
    que `FlightLearner::Evaluate` sur 8 observations, ainsi que les mêmes
    log-std ;
  - `BlueDefenseNetworkTypeTests` : `std::is_same` entre le critique généré
    et `GeneralizedCriticConfig::Network`.
- Toujours additif : aucun learner n'utilise encore les types générés
  (étape 6b, après le rebase).

### Détail de l'étape 7a

- `ppokit/runtime/rollout.hpp` : `Transition` (fin d'épisode `Running`,
  `Terminated` ou `Truncated`, indicateur `learnable` pour MAPPO),
  `RolloutBuffer` (alloué une fois, refuse les valeurs non finies),
  `ComputeAdvantages` (GAE commun), `NormalizeAdvantages` (forme canonique :
  somme puis division, variance de population, epsilon 1e-8 sous la racine ;
  Flight et Blue normalisaient de façon équivalente mais pas identique au bit
  près, à aligner lors de leur migration, décision B), `ExplainedVariance`.
- `ppokit/runtime/collector.hpp` : `RolloutCollector` en mode Owned. Il
  enchaîne les épisodes d'un rollout à l'autre, ne bootstrape jamais un état
  terminé, bootstrape les troncatures, et tient des statistiques d'épisodes.
  La tâche et la politique sont des classes simples (aucun appel virtuel dans
  la boucle chaude).
- `ppokit/backend/rltools/gaussian_mlp_learner.hpp` : learner PPO complet
  sur les réseaux générés (acteur gaussien, critique), avec Adam, minibatches
  mélangés, clamp du log-std et statistiques. Il réutilise
  `EvaluateClippedSurrogate` et les fonctions de la tête gaussienne.
- PPOKit possède maintenant la dépendance rl_tools (même version figée) ;
  Flight et `dependencies.cmake` réutilisent la cible. Cible
  `PPOKit::BackendRltools`.
- Tests : `PPOKitRuntimeTests` (buffer, GAE sur plusieurs épisodes,
  troncature, normalisation, collecte sur un couloir) et
  `PPOKitBackendLearnerTests`, qui entraîne de bout en bout un couloir jouet
  décrit par `tests/fixtures/corridor/task.json` et `graph.json`. Le retour
  moyen passe de −3,0 à 3,9 en 30 updates, la politique déterministe atteint
  le but en 6 pas (optimum), Debug et Release identiques. C'est un test de
  fumée du moteur, pas une preuve de convergence.
- Un piège de conception de tâche, documenté dans le test : une récompense
  qui ne dépend pas de l'action (coût par pas seul) ne donne aucun signal
  avant le premier succès, rare en exploration aléatoire.

### Détail de l'étape 7c

- `ppokit/eval/evaluation.hpp` et `src/eval/evaluation.cpp` : lecture
  stricte de `ppokit.eval/1` (`suite`, `seeds` uniques, `episodes`,
  `policy: deterministic`, `gates` de la forme `">= x"` ou `"<= x"` sur
  `mean_return`, `success_rate`, `truncation_rate`, `mean_length`),
  `RequireDisjointSeeds` (refuse un seed d'entraînement), `EvaluationReport`
  (métriques, verdict par gate avec sa valeur mesurée, rapport JSON).
- `ppokit/eval/evaluator.hpp` : `Evaluate(suite, fabrique de tâche par
  seed, politique, borne de pas)`. La politique doit seulement fournir
  `Mean` ; un épisode qui ne se termine jamais est signalé au lieu de boucler.
- `RequireObject` ne renvoie plus de référence : cela supprime une famille de
  faux positifs `-Wdangling-reference` de GCC 13.
- Tests : `PPOKitEvaluationTests`. Le test du learner enchaîne désormais
  entraînement puis évaluation de la suite
  `tests/fixtures/corridor/eval/smoke.json`, qui passe ses gates.

### Détail de l'étape 8a (scénarios et curriculum)

- `task.json` accepte `"scenario"` : des paramètres `float` et `int`
  (avec `range`), `bool` et `enum` (avec `values`), et une valeur par défaut
  optionnelle, vérifiée. Les scénarios ne font pas partie du `ShapeHash`.
- `ppokit.train/1` (`ppokit/schema/train_schema.hpp`) : `curriculum` avec
  `window`, `promote_at`, `rehearse` et des `stages` nommées. Par paramètre :
  une valeur fixe, `uniform`, `normal` (bornée au domaine et arrondie pour un
  entier) ou `choice` pondéré. Un paramètre absent prend sa valeur par
  défaut, sinon c'est une erreur. Les erreurs pointent la clé fautive.
- `PPOKit::Training` (`ppokit/runtime/curriculum.hpp`) : `Curriculum`
  (`Sample`, `Record`, `Restore`), avec la même règle de promotion que
  `FlightCurriculum` (fenêtre complète, puis nouvelle fenêtre vide), la
  répétition jamais comptée, et `Save`/`LoadCurriculumStage` par **nom**
  d'étape, en écriture atomique.
- L'en-tête généré contient une struct `Scenario` typée (énumérations,
  champs, `From(values)`).
- Tests : `PPOKitCurriculumTests`.
- Correspondance avec Flight : `FlightCurriculum` (niveaux de difficulté,
  `curriculum_level.txt`) n'est pas encore migré. C'est la prochaine étape
  logique de 6b pour Flight, et elle touche au code d'exécution.

## Prochaine action

Étape 9, graphes de blocs (`docs/architecture/ppokit-graph-blocks.md`).
L'utilisateur l'exige : sans attention ni encodeurs, le framework ne peut
pas porter l'acteur Blue ni servir l'éditeur à boîtes et connecteurs.
Décision : un moteur de graphe PPOKit construit au chargement (sans
recompilation, buffers alloués une fois), à côté de la voie rl_tools des MLP.
Ordre :
1. ~~Blocs~~ **fait** : `ppokit/graph/blocks.hpp` (`Dense`, `EntityEncoder`
   à poids partagés, `MaskedAttention` à produit scalaire normalisé,
   `MaskedPool` moyenne et max), buffers alloués au constructeur, aucune
   allocation par pas. `PPOKitGraphBlocksTests` compare tous les gradients
   (entrées et paramètres) aux différences finies et vérifie les entités
   absentes et l'ensemble vide.
2. ~~Moteur~~ **fait** : `ppokit/graph/graph.hpp` (`Graph` : nœuds ajoutés
   après leurs sources, donc acyclique par construction ; formes vérifiées à
   chaque connexion avec le nœud et le port nommés ; gradients sommés quand
   une sortie a plusieurs consommateurs ; `OutputOf` et `OutputGradient` pour
   les sondes) et ses nœuds `input`, `dense`, `mlp`, `concat`,
   `entity_encoder`, `masked_attention`, `masked_pool`, `HeadNode`
   (gaussienne avec log-std en paramètre, Bernoulli, valeur).
   `ppokit/graph/adam.hpp` : Adam avec correction de biais sur
   `Graph::Parameters()`. `PPOKitGraphEngineTests` : copie réduite de
   l'acteur Blue (encodeur partagé par l'attention et deux poolings),
   gradients de bout en bout aux différences finies (paramètres et entrées,
   ensemble partiel et vide), accumulation, erreurs de construction, Adam.
3. ~~Format~~ **fait** : `ppokit/schema/block_graph_schema.hpp`
   (`ParseBlockGraph`, `LoadBlockGraph`, `BuildGraph`). Passes : blocs et
   paramètres, arêtes vers ports, ordre topologique (ordre du fichier à
   égalité, donc entrées dans l'ordre écrit), inférence des formes en
   construisant les mêmes nœuds que `BuildGraph`. Erreurs nommant réseau,
   nœud, port ou arête : type inconnu, set/vector, largeurs, port non câblé
   ou câblé deux fois, cycle, sortie inutilisée, tête qui alimente un nœud,
   têtes manquantes ou en double (acteur : gaussienne puis Bernoulli selon
   `task.json` ; critique : une `value`). Le graphe accepte plusieurs sorties
   (`AddOutput`, `Backward(gradients, count)`). `PPOKitBlockGraphSchemaTests`
   lit les vrais `task.json` de Blue et Flight ; fixture
   `tests/fixtures/blue_attention/graph.json` (acteur à attention, critique
   poolé), non encore comparée à l'acteur écrit à la main.
4. ~~Learner~~ **fait** : `ppokit/graph/graph_learner.hpp`
   (`GraphLearner<ObservationSize, ActionSize>`, même interface et même mise
   à jour que `GaussianMlpLearner` : `Decide`, `Mean`, `Value`, `Update`,
   `LogStd`, plus `Actor()`/`Critic()` pour les sondes). Têtes gaussienne et
   Bernoulli (action Bernoulli stockée 1 ou 0 ; `Mean` rend 1 si logit > 0).
   `ppokit/graph/policy_layout.hpp` : `InputBinder` (tranches de
   l'observation, masque des ensembles lu dans leur colonne `mask`, aucune
   allocation par pas). `schema::MakePolicyLayout` relie `task.json` et
   `graph.json` ; il refuse un critique qui lit des groupes `critic` (il faut
   le mode Hosted). `PpoSettings`/`UpdateStatistics` déplacés dans
   `ppokit/core/ppo_settings.hpp` (alias conservés côté rl_tools) ;
   `initial_log_std` retiré de `graph.json` : seule source,
   `PpoSettings::initialLogStd`. Tests `PPOKitGraphLearnerTests` : couloir
   (même tâche que le learner rl_tools, `tests/support/corridor_task.hpp`,
   retour 2,2 → 4,0, but en 7 pas, porte `corridor_smoke` sur graines
   indépendantes) ; tâche « switch » gaussienne + Bernoulli (interrupteur
   7,5 % → 100 %, distance 0,19 → 0,02) ; déterminisme par graine. Note :
   lr acteur 1e-2 instable sur « switch » (KL jusqu'à 4), 3e-3 retenu.
5. ~~Parité Blue~~ **fait** : `Examples/BlueDefense/Agents/BluePatrol/graph_blocks.json`
   décrit `GeneralizedAttentionActor` à l'identique : encodeurs, requêtes
   tanh depuis `own`, attention, moyenne et max pour les Reds (D=8) et pour
   les amis (D=6, `concat_sets` des 8 pistes bleues et de l'objectif), puis
   l'entrée compacte v7 : `select` garde patrol[0..7,9] et global[0] (role
   écrase patrol[8] et global[1] dans le code écrit à la main), tronc ReLU
   2×128. Nouveaux blocs `select` et `concat_sets` ; le max pooling partage
   le gradient entre ex æquo, comme le code écrit à la main.
   `schema::MakeInputSlices` lie un seul réseau (le critique Blue est
   centralisé). `Examples/BlueDefense/tests/graph_actor_parity_tests.cpp`
   (cible `BlueDefenseGraphActorParityTests`, desktop ; compilé à la main
   par `verify.sh`) : poids rl_tools copiés (permutation des colonnes de la
   première couche), 1024 observations (0 à 10 Reds, 0 à 9 amis), moyennes
   à 2,9e-7 et gradients à ≤ 4,3e-7 en relatif. Une permutation erronée
   (role[0] et role[1]) fait passer l'écart à 0,39 : le test discrimine.
6. Éditeur, en deux temps.
   a. ~~Modèle~~ **fait** : `PPOKit::Editor`
      (`ppokit/editor/graph_document.hpp`, sans dépendance graphique).
      `GraphDocument` garde ce que l'utilisateur dessine, valide ou non
      (type inconnu, port non câblé) : `AddNode` (nom libre `dense_2`,
      paramètres par défaut), `RemoveNode`, `RenameNode`, `SetParameter`,
      `Connect` (remplace le fil du port), `Disconnect`,
      `NextNumberedPort`, `MoveNode` + `CommitMove`, `Undo`/`Redo`.
      `Validate` passe par `ParseBlockGraph` (le même code que
      l'entraînement) et localise la première erreur (réseau, nœud,
      port/paramètre ou indice de fil) grâce à `SchemaError::Path()` ; si
      le graphe est valide, il donne les formes inférées de chaque nœud.
      Positions dans un fichier à part (`ppokit.graph-layout/1`) ; sans
      positions, placement par profondeur le long des fils.
      `BlockCatalog()` : titre et infobulle de chaque bloc ;
      `InputChoices` : groupes proposés à un nœud `input`. Tests
      `PPOKitGraphDocumentTests` : dessin du couloir de zéro jusqu'à un
      graphe valide en corrigeant chaque erreur signalée, undo/redo,
      renommage, fichiers cassés, aller-retour du graphe Blue.
   b. ~~Vue~~ **faite** : `PPOKit/studio/` (`GraphEditorView`, ImGui +
      imnodes, sans raylib ; exécutable `ppokit-studio <task.json>
      <graph.json> [layout.json]`, fenêtre raylib/rlImGui comme le
      lanceur ; `--check` valide sans fenêtre). Palette avec infobulles,
      onglets acteur/critique (un contexte imnodes chacun), ports carrés
      orange pour les ensembles, port libre « + » des concaténations,
      formes inférées sous chaque nœud, nœud/port/fil fautif en rouge avec
      le message, bouton « Show » qui centre le nœud fautif, inspecteur
      (nom, groupe, largeur, couches, activation, mode, indices, avec
      aides), Ctrl+clic détache un fil, Suppr, Ctrl+Z/Y/S, déplacement =
      une étape d'annulation (`BeginMove`/`EndMove`), Échap ne ferme pas.
      imnodes épinglé à 2d0c280 (compile avec ImGui 1.92.1), sources seules
      via `SOURCE_SUBDIR`. Tests : `PPOKitStudioViewTests` (images ImGui
      réelles sans rendu : graphe Blue, graphe cassé, Ctrl+Z/Y, Ctrl+S puis
      réouverture identique) et `PPOKitStudioCheckBlueGraph` — tous deux
      dans le build desktop ; ici avec
      `PPOKIT_STUDIO_DEPS=<dossier imgui+imnodes> tools/ppokit/verify.sh`.
      Non testé automatiquement : gestes souris (tirer un fil, glisser un
      nœud) ; la fenêtre raylib n'a pas été ouverte ici (pas d'affichage).
7. ~~Remplacement de l'acteur Blue~~ **fait** (accord de l'utilisateur le
   22/11) : `GeneralizedAttentionActor` garde son stockage rl_tools
   (paramètres, moments Adam, `log_std`), donc checkpoints, migrations V5/V7
   et modèle de production `blue_patrol_generalized_v10.txt` restent
   compatibles sans conversion. Son calcul est le graphe de
   `graph_blocks.json` (`Training/generalized_actor_graph.hpp`) : poids
   copiés dans le graphe avant chaque passe, gradients du graphe ajoutés aux
   gradients rl_tools après `backward` (rl_tools Adam les applique). Le
   graphe vérifie au chargement qu'il garde la structure du stockage (ordre
   de `features`, indices des `select`, largeurs 8/6, profondeur du tronc).
   `task.json` et `graph_blocks.json` sont embarqués à la compilation
   (`ppokit-codegen embed`, `ppokit_embed_files`). Le learner fait une seule
   passe par équipe : moyennes et lignes compactes pour le critique
   (`evaluate_with_compact`) ; le tronc rl_tools n'est plus appelé.
   L'ancien code est l'oracle `Examples/BlueDefense/tests/reference/` ;
   `graph_actor_parity_tests` compare l'acteur de production et l'oracle
   par l'API rl_tools : moyennes ≤ 3,9e-7, gradients ≤ 5,2e-7, lignes
   compactes identiques (0), et sensibilité (poids du slot de formation).
   Trace MAPPO de référence inchangée à la précision imprimée (9 chiffres).
   Plus rapide : `ppo_ms` 1247 → 961, test learner 13,9 s → 12,0 s.
8. Mode Hosted (MAPPO piloté par la boucle du client), en trois temps.
   a. ~~`TeamRollout`~~ **fait** (`ppokit/runtime/team_rollout.hpp`) : une
      ligne par agent et pas d'équipe (indice = pas × agents + agent), GAE
      par agent le long des pas, agents détruits gardés (critique et
      retour de mission) mais non apprenables. Normalisation et variance
      expliquée factorisées dans `ppokit/runtime/advantages.hpp`, avec
      l'arithmétique de Blue (variance comparée à epsilon, et non plus la
      somme des carrés : correction de `RolloutBuffer` aussi, sans effet
      sur l'entraînement). `PPOKitTeamRolloutTests` : **identité au bit
      près** avec les formules MAPPO de Blue (oracle copié) pour 1, 3 et
      9 agents, avec morts, fins et troncatures aléatoires.
   b. ~~`HostedTrainer`~~ **fait** (`ppokit/runtime/hosted_trainer.hpp`) :
      `ChooseActions` / `ChooseActionsFromMeans` puis `Observe`, mise à
      jour automatique quand le rollout est plein ; algorithme de Blue
      (lignes de base du critique en fin de rollout, GAE, normalisation sur
      les disponibles, époques de minibatches mélangés, acteur sur les
      disponibles, critique sur tout, un pas d'optimiseur par réseau,
      mêmes diagnostics). Réseaux derrière une interface `Models`
      (rl_tools, graphes PPOKit, libtorch...) et générateur aléatoire du
      client (`Shuffle`, `Normal`) pour garder son état sauvegardé.
      `ppokit/graph/team_graph_models.hpp` : implémentation sur les
      graphes PPOKit (Adam, log-std borné), critique sur la disposition
      `critic` de `task.json` (`schema::MakeCriticInputSlices`).
      `PPOKitHostedTrainerTests` : équipe de 3 agents pilotée par la boucle
      du test, un agent détruit à mi-épisode ; distance finale 0,72 → 0,04 ;
      mécanique (ordre des appels, lignes de remplissage, aucun gradient
      des agents détruits, erreurs d'ordre).
   c. ~~Blue client~~ **fait** (CI desktop verte, voir plus bas) : `GeneralizedPatrolLearner` garde son
      interface publique, ses checkpoints et ses diagnostics, mais le
      rollout, le GAE, la normalisation, les minibatches et la boucle PPO
      sont ceux de PPOKit (`HostedTrainer`). Blue ne fournit plus que son
      adaptateur rl_tools (`TeamModels` : acteur, critique, optimiseurs,
      lot rl_tools fixe), son générateur (`TeamRandom`, état sauvegardé dans
      l'état d'entraînement), l'entrée du critique centralisé et ses logs
      `[MAPPO_TIMING]` / `[MAPPO_UPDATE]` (via `HostedUpdateHooks`).
      `generalized_patrol_learner.hpp` passe de 1428 à ~960 lignes.
      **Trace MAPPO de référence identique.** Temps : profil CPU identique
      (gprof) ; à l'horloge, ancien 14,3–15,9 s, nouveau 15,9–16,2 s sur
      cette machine bruitée (au plus ~5 %).
9. Studio côté entraînement (ordre annoncé à l'utilisateur le 22/11) :
   journal de run, tableau de bord, paramètres PPO commentés, poids des
   récompenses en cours de run, sondes, éditeur de curriculum. Ensuite :
   composants DLL, Flight sur le runtime PPOKit, artefacts versionnés,
   libtorch/Airspace/GPU, outillage des exemples ; stockage natif de
   l'acteur Blue sur demande ; RLMission sur approbation.
   a. ~~Journal de run~~ **fait** : `ppokit/runtime/reward_ledger.hpp`
      (`RewardLedger` : par canal, moyenne par pas brute et pondérée,
      moyenne par épisode terminé, cumul depuis le début ; plusieurs flux
      (agents) ; poids changeables entre deux updates) et
      `ppokit/runtime/run_log.hpp` (`metrics.jsonl`, `ppokit.run/1` :
      événements `start`, `update`, `weights` ; une ligne écrite et vidée
      par événement ; lecture tolérante à une dernière ligne incomplète,
      ligne fautive nommée). `PpoMetrics` / `PpoSettingsValues` donnent
      des noms stables. Tests `PPOKitRunLogTests`.
   b. ~~Tableau de bord~~ **fait** : `ppokit-studio --run metrics.jsonl`
      (`PPOKit/studio/training_dashboard_view.*`, ImPlot). Onglet
      Récompenses (par canal : par pas pondéré ou brut, par épisode, cumul ;
      canaux à cocher avec libellé et poids ; retour par épisode ; lignes
      verticales aux changements de poids), onglet PPO (un petit graphe par
      statistique), onglet Paramètres. Infobulle sur chaque courbe,
      statistique et paramètre : `ppokit/editor/ppo_help.hpp` (quoi, valeurs
      typiques, symptômes) ; un test impose qu'un texte existe pour chaque
      nom écrit par PPOKit. « Follow » relit le journal pendant le run ; un
      journal corrompu garde la dernière version valide à l'écran. Modèle
      sans GUI : `ppokit/editor/run_series.hpp` (trous pour les valeurs
      manquantes, pas de faux zéros). Tests `PPOKitRunSeriesTests`,
      `PPOKitStudioDashboardTests` (sans affichage ; ici avec
      `PPOKIT_STUDIO_DEPS`, qui contient maintenant aussi implot v0.17).
   c. ~~Journal écrit par les learners~~ **fait** :
      - Flight : `FlightSession` écrit `metrics.jsonl` dans le dossier du
        modèle (libellés du `task.json` via le layout généré, paramètres
        Flight, statistiques `mean_reward`, `max_approx_kl`,
        `max_sampled_kl`, `actor_kl_skips`, résumé par canal) ; chaque
        changement de poids (onglet de réglage) est journalisé avec la
        première update collectée sous ces poids. Un `latest` repris
        continue son journal (`RunLogWriter::Resume`). Trace dorée
        inchangée : le journal ne fait qu'observer. Test `RunJournal` dans
        `FlightControl_session_tests`.
      - Blue : `GeneralizedPatrolMappoAgent` écrit `metrics.jsonl` à côté
        du checkpoint d'entraînement ; un flux par Blue, canaux bruts
        avant clamp ; les éditions de poids en direct (déjà existantes,
        non modifiées) sont détectées à chaque pas et journalisées depuis
        l'update en cours de collecte. Une erreur d'écriture désactive le
        journal avec un message, sans arrêter l'entraînement. Accès
        `LastUpdate()` et `PpoSettings()` ajoutés au learner ; trace MAPPO
        identique. Compilé ici en `-fsyntax-only` ; la compilation et les
        tests complets sont ceux du job Desktop.
      - Owned PPOKit : `RolloutCollector::Track(RewardLedger*)` ; une
        tâche peut exposer `LastRewardChannels()`, sinon le total compte
        comme un canal. Test `TracksRewardChannels`.
   d. ~~Paramètres PPO éditables~~ **fait** :
      - Catalogue unique `ppokit/core/ppo_parameters.hpp` (nom, groupe,
        bornes, défaut, entier, échelle log) ; `PpoParameterValues::Settings()`
        donne les `PpoSettings` d'un learner, `RolloutSteps()` la taille du
        buffer.
      - `train.json` gagne la section `ppo` (`TrainingPlan::ppo`,
        `ParsePpoParameters` sans tâche) : messages avec la clé, la plage,
        « did you mean ».
      - `ppokit/editor/train_settings_document.hpp` : édition sans GUI,
        autres sections conservées, « Use default » retire la clé,
        annuler/rétablir (un glisser = un changement), sauvegarde atomique ;
        `CheckPpoParameters` : remarques en clair (horizon de gamma, travail
        d'une update, minibatch > rollout, découpe inégale, learning rates et
        clip trop grands, gamma = 1). Test `PPOKitTrainSettingsTests`.
      - Studio : `ppokit-studio --train train.json`, combinable avec `--run`
        (deux onglets, point « non enregistré » sur l'onglet). Vue
        `PPOKit/studio/train_settings_view.*` ; infobulles partagées
        (`help_tooltip.*`). Test `PPOKitStudioSettingsTests` (headless).
      - Flight et Blue gardent leurs propres fichiers de réglages : Flight
        passera à `train.json` avec l'étape « Flight sur le runtime PPOKit »,
        Blue sur demande.
   e. ~~Poids modifiables pendant un run~~ **fait** :
      - `ppokit/runtime/weight_request.hpp` : le studio écrit
        `weights.request.json` (seuls les canaux changés, un id), le run le
        prend à une frontière d'update (`WeightRequestInbox::Take`, une
        existence de fichier quand il n'y a rien), répond dans
        `weights.answer.json`. Requête illisible, canal inconnu ou poids non
        fini : refusée avec la raison, consommée. `RunHeader::
        acceptsWeightRequests` (`"weight_requests": true`) dit au studio
        qu'il peut proposer l'édition.
      - `ppokit/editor/live_weights.hpp` (sans GUI) : poids en vigueur
        depuis le journal, brouillon par canal, envoi, attente, réponse ;
        les ids reprennent après la dernière réponse (un studio rouvert ne
        prend pas une vieille réponse pour la sienne). Test
        `PPOKitWeightRequestTests`.
      - Tableau de bord : onglet « Reward weights » (poids en vigueur,
        nouveau poids, « edited », Apply at next update / Discard, réponse
        en couleur, historique des changements ; texte explicatif quand le
        run ne lit pas les requêtes ; point sur l'onglet tant qu'un poids
        édité n'est pas envoyé).
      - Flight : lit les requêtes quand il a un fichier de scénario ;
        validation de toute la requête avant d'appliquer (tout ou rien), puis
        le chemin existant `SetRewardTuning` (sauvegarde dans le scénario,
        BEST à revalider, journal). Une erreur de fichier n'arrête jamais
        l'entraînement. Test `StudioWeightRequests` dans
        `FlightControl_session_tests`.
      - Blue : non branché (son réglage en direct existant reste tel quel) ;
        à faire seulement sur demande explicite.
   f. ~~Sondes~~ **fait** :
      - `ppokit/graph/probes.hpp` (`GraphProbes`) : par bloc (entrées
        exclues), moyenne, écart-type, min, max, part de zéros exacts (ReLU
        morts), part |x| ≥ 0,99 (saturation), norme L2 moyenne du gradient
        des paramètres par pas d'optimiseur ; ensembles : entités présentes
        seulement. Accumulateurs alloués une fois.
      - `GraphLearner::SetProbes` (dernière époque de l'update) et
        `TeamGraphModels::SetProbes` (toutes les lignes d'entraînement ;
        reset par l'appelant). Coupées : un test de pointeur, rien d'autre.
        Identité vérifiée : mêmes statistiques et même politique avec et
        sans sondes (`PPOKitProbesTests`), même résultat team reach que
        sans sondes (comparé au binaire d'avant le changement).
      - Journal : événement `probes` (`ProbeRecord`), conservé par
        `Resume` jusqu'à l'update reprise. `train.json` : `"probes": {
        "every": N }`, `ProbesDue` ; réglages du studio : section
        Diagnostics.
      - Studio : onglet « Probes » (réseau, statistique avec infobulle,
        courbe par bloc, échelle log pour le gradient, remarques en clair :
        unités mortes ou saturées, sortie constante, gradient absent ou
        énorme). `ppokit/editor/probe_series.hpp`.
      - Pas encore branché : Blue (acteur graphe, sur demande) ; Flight
        n'a pas de graphe PPOKit (réseaux rl_tools), il en aura avec
        l'étape « Flight sur le runtime PPOKit ».
   g. ~~Éditeur de curriculum~~ **fait** :
      - `ppokit/editor/curriculum_editing.hpp` (sans GUI) : section JSON ↔
        étapes éditables (une règle par paramètre du task.json, champs
        gardés quand on change de règle), valeurs écrites comme un humain
        (noms d'énumération, true/false, entiers) ; `CheckCurriculum` =
        le parseur de l'entraînement ; `PreviewStage` tire les scénarios
        avec `runtime::Curriculum` (sans répétition). Aller-retour sans
        perte de sens vérifié sur le plan parsé. Test
        `PPOKitCurriculumEditingTests`.
      - `TrainSettingsDocument::AttachTask` / `SetCurriculum` /
        `CurriculumProblem` : annuler/rétablir communs, document propre
        tant qu'on ne modifie rien.
      - Studio : `PPOKit/studio/curriculum_view.*`, onglet « Curriculum »
        de `--train train.json --task task.json`. Test
        `PPOKitStudioCurriculumTests` (headless).
   h. Studio terminé pour cette série (journal, tableau de bord, réglages
      PPO, poids en direct, sondes, curriculum).
10. Composants DLL (étape 8 du plan) **fait** :
    - `ppokit/component/abi.h` (C pur, ABI 1), `sdk.hpp` (header-only,
      manifeste écrit depuis `Describe()`, exceptions arrêtées avant la
      frontière, macro `PPOKIT_COMPONENTS` jusqu'à 8 classes),
      `library.hpp`/`src/component/library.cpp` (chargement dlopen /
      LoadLibraryW, nom nu → `x.dll`/`libx.so`, une seule instance par
      fichier, négociation d'ABI, cohérence fonctions/capacités, validation
      des réglages, hash FNV du binaire + manifeste + réglages),
      `component_node.hpp` (nœud de graphe : symboles résolus au câblage,
      masques reliés aux ensembles, paramètres en miroir dans un
      `graph::Parameter`, aucune allocation dans Forward/Backward).
    - `graph.json` : `"components"` et `"component:<nom>@<version>"` ;
      règle 12.4 vérifiée (`RequireGradientPaths`) ; état et sorties
      multiples refusés pour l'instant avec un message.
    - Studio : palette « Components », ports et réglages du manifeste,
      liste des bibliothèques gardée et annulable.
    - Tests : `PPOKitComponentTests` (chargement et messages, réglages,
      valeurs, gradient comparé à une différence finie, règles de câblage,
      exception de la DLL, couloir qui s'entraîne avec un composant
      entraînable), `PPOKitGraphDocumentTests` (composants), vue studio.
    - Vérifié aussi : le checkpoint T-6 de master (update 158) se charge
      (`CommittedCheckpoints` dans `FlightControl_learner_tests`).
11. Flight sur le runtime PPOKit **fait** (sans toucher au format des
    scénarios ni des checkpoints, golden trace **identique au bit près** en
    Debug et Release) :
    - `ppokit/runtime/promotion_window.hpp` (`PromotionWindow`) : la règle
      de promotion unique, utilisée par `runtime::Curriculum` et par
      `FlightCurriculum` (qui garde ses niveaux de difficulté continue).
    - Niveau de curriculum Flight écrit par `artifacts::WriteFileAtomically`.
    - Avantages Flight normalisés par `runtime::NormalizeLearnableAdvantages`
      (`Transition` : `ScalarType`, `learnable`, `valueTarget`), variance
      expliquée calculée par `ExplainedVarianceOf` et publiée
      (`[FLIGHT_UPDATE] explained_variance=`, journal).
    - `ppokit/core/kl.hpp` : `SampledKl`, `ApproximateKl`, `KlSkipFactor`,
      `ActorStepAllowed` ; Flight les utilise, et les learners PPOKit
      (`GraphLearner`, `GaussianMlpLearner`) gagnent la garde KL
      (`PpoSettings::targetKl`, `target_kl` dans `train.json`, 0 = coupée,
      `actor_kl_skips` dans les statistiques et le journal). Test `KlGuard`.
    - Restent propres à Flight, par nature : simulateur JSBSim, tête de
      train, format de checkpoint (`PPO_FLIGHT_POLICY_V3`), boucle de
      minibatch sur rl_tools. Les checkpoints committés se chargent toujours.
12. Artefacts versionnés **fait** :
    - `ppokit/artifacts/artifact.hpp` : en-tête `ppokit.artifact/1`
      (kind, identité : tâche, `taskHash` = ShapeHash, `graphHash` des blocs
      et composants, backend, scalaire, tailles, blocs lisibles, hashes des
      composants ; updates, seed, date UTC), `RequireCompatible` qui refuse
      en listant les différences.
    - `schema::ArtifactIdentityOf(task, graphs)`.
    - `GraphLearner::SaveTraining/LoadTraining/SavePolicy/LoadPolicy` :
      texte exact au bit (9 chiffres), Adam (`Adam::Save/Restore`), RNG,
      compteur ; tout est lu avant d'être appliqué.
    - Outil `ppokit-artifact inspect|check`.
    - Fixture committée `tests/fixtures/corridor/training.ppokit` : un
      changement involontaire du format casse la CI (comme la golden trace).
    - Tests `PPOKitArtifactTests` (reprise exacte au bit près avec un autre
      seed de départ, politique, refus expliqués, fichier tronqué),
      `PPOKitArtifactInspect`, `PPOKitArtifactCheck`.
    - Non fait volontairement : convertir les formats Flight et Blue. Ils
      se chargent directement (§8.1) et des modèles sont en cours
      d'entraînement ; `TeamGraphModels` (hosted) n'a pas encore de
      checkpoint PPOKit.
13. Backend libtorch, **vert en CI Desktop** (seul endroit où Torch existe ;
    aucun GPU nulle part, le chemin CUDA n'est donc pas exécuté) :
    - `PPOKit/torch` : `TorchGraph` (bloc par bloc en tenseurs par lot,
      autograd, mêmes masques que le moteur), `TorchGraphLearner` (même
      interface que `GraphLearner`, PPO par minibatch en tenseurs, Adam de
      torch, garde KL, `ChooseDevice("auto"|"cpu"|"cuda")`), échange de
      politiques avec le moteur de graphe.
    - Test `PPOKitTorchTests` : parité sorties (1e-5) et gradients (1e-4)
      couloir et acteur Blue, couloir entraîné, politique torch → moteur,
      choix de l'appareil.
    - Airspace **non touché** : son agent Mission est RLMission (règle de
      l'utilisateur, et le plan le met en dernier, sur approbation).
      Interception n'a pas été migré non plus (même dossier, même risque).
14. Outillage des exemples :
    - `ppokit-new <nom> [--into DIR]` (`tools/ppokit_new.cpp`) écrit un
      exemple complet qui s'entraîne déjà (un point rejoint une cible) :
      `task.json`, `graph.json` (format blocs), `train.json` (PPO, sondes,
      curriculum 2 étapes), `eval/smoke.json`, `task.hpp`, `main.cpp`,
      `CMakeLists.txt`, `README.md`, `.gitignore`. Nom refusé s'il n'est
      pas `[a-z][a-z0-9_]*` ; dossier non vide jamais écrasé.
    - `ppokit/tools/command_line.hpp` : `RunCommandLine<Obs, Act>(argc,
      argv, MakeTask, dossier)` donne `check`, `train`, `eval`, `help`.
      `train` reprend `model/latest.ppokit` (refus expliqué + « train
      --fresh » si les réseaux ont changé), écrit journal, checkpoints,
      politique, sondes, étape du curriculum (`model/curriculum.txt`,
      restaurée à la reprise), applique les poids demandés par le studio.
      Membres optionnels de la tâche détectés à la compilation :
      `Configure`, `Succeeded`, `LastRewardChannels`,
      `RewardWeights`/`SetRewardWeights`. Erreurs d'usage : code 2 avec
      l'aide ; échec : code 1 avec la raison.
    - `examples/reach` est la sortie exacte de `ppokit-new reach` :
      `PPOKitScaffold` régénère et compare (fins de ligne mises à part).
      `PPOKitReachCheck` lance `ppokit_reach check` ;
      `PPOKitCommandLineTests` : usage, check, 2+1 updates avec reprise,
      requête de poids, étape restaurée, reprise refusée, eval 0/1.
      Quelques updates seulement : plomberie, pas apprentissage.

15. Studio utilisable et visuel, pour Flight, Blue et les exemples (revue
    sur captures : rendu logiciel des vues ImGui en PNG, sur des journaux qui
    portent les vrais canaux Flight et Blue) :
    - Accueil (`studio/home_view.*`, `editor/workspace.*`) : runs, réseaux
      et réglages trouvés dans le projet ; `Lancer_PPOKit_Studio.bat`.
    - Tableau de bord : état du run (en cours / arrêté), temps total, cartes
      de santé (`editor/run_health.*`), cible KL tracée, libellés des
      canaux, thème (`studio/studio_theme.*`), aides `probe_every`,
      `blue_mobiles`, `red_mobiles`.
    - Sondes rl_tools en lecture seule (`backend/rltools/mlp_probes.hpp`,
      accumulateur commun `runtime/probe_moments.hpp`) : dans
      `GaussianMlpLearner`, Flight (couches des MLP) et Blue (graphe acteur
      + critic rl_tools), toutes les 10 updates. Tests : checkpoint Flight
      identique au bit près avec/sans sondes ; Blue : mêmes actions et même
      état d'entraînement ; `PPOKitBackendLearnerTests` idem.
    - Trace Blue réenregistrée : les 80 lignes de référence sont identiques,
      le nouveau test de sondes (placé en dernier) ajoute 32 lignes.
    - Blue prend les requêtes de poids du studio (règles des curseurs du
      jeu : 0 à 10, toute la requête ou rien, pour la session) ;
      `acceptsWeightRequests` vrai dans son journal.
    - Bugs corrigés : temps écoulé remis à zéro à chaque reprise (le writer
      ajoute maintenant le temps de l'historique gardé, pour Flight, Blue et
      les exemples) ; remarques de sondes fausses sur les sorties linéaires
      (saturation) et les blocs sans paramètres (unités « mortes ») ; poids
      changés depuis le studio perdus à la reprise d'un exemple ; message de
      l'onglet Probes qui renvoyait à un train.json que Flight et Blue
      n'ont pas.

Ne pas toucher au code d'exécution des exemples : l'utilisateur modifie les
modèles et demandera une remise à jour depuis master.

## Refonte du studio : projets autonomes (en cours)

Demande de l'utilisateur (« pas d'appli qui liste tous les modèles », zoom,
look MFDStudio, projet autonome, lancer train/inférence, infobulles, DLL
visibles, vue haut niveau acteur-critique avec GAE). Fait :

- Format projet `ppokit.project/1` (`schema/project.hpp`) : `project.json`,
  `task.json`, `network.json` (+ `.layout.json`), `training.json`, `eval/`,
  `components/`, `checkpoints/`. `{project}` remplacé dans les arguments.
- Programmes : `ppokit_register_program(target)` + `ppokit_write_program_registry()`
  écrivent `ppokit_programs_<config>.txt` ; `FindProgram` cherche dossier du
  projet, registre, PATH.
- CLI des exemples : `--project DIR`, commande `play`, arrêt propre par
  `checkpoints/stop.request`.
- Studio : écran de départ (dossier choisi par l'utilisateur, nouveau
  projet, récents), fenêtre projet (Overview cliquable, Network, Training,
  Task, Run, Console), boutons Train/Evaluate/Inference/Stop/Kill
  (`editor/process.hpp`), thème MFDStudio (Segoe UI ; Inter ou DejaVu hors
  Windows), zoom molette/boutons/Fit, DLL de `components/` dans la palette.
- Supprimé : `editor/workspace.*`, `studio/home_view.*` et leurs tests.
- Corrigé en route : le placement automatique des blocs se chevauchait ; un
  zoom recopiait les anciennes positions imnodes dans le document.

Fait ensuite : Flight et Blue en projets complets (`Projects/`). `--project`
dans FlightControlHeadless, FlightControl3D et PPO.exe ; un scénario placé
dans un projet garde ses modèles dans `checkpoints/` (`ProjectFolderOf`) ;
`stop.request` pris à la fin d'une update (Flight : boucle headless ; Blue :
l'agent, qui sauve puis termine) ; `ppokit_register_program` pour les trois
programmes ; `project.json` gagne `settings`, `networkEditable`, `journal` et
une commande peut avoir son propre exécutable. Vérifié en local sur une
copie du T-6 : une update (243 → 244), puis arrêt sur demande à l'update 245
avec checkpoint compatible. Blue et les sources desktop (main.cpp,
flight_desktop.cpp, bluedefenseexample.cpp) ne compilent pas ici : vérifiés en
`-fsyntax-only` seulement, la CI Desktop fait foi.

Reste :
1. Migrer les learners Flight et Blue sur PPOKit : conversion des
   checkpoints, traces dorées réenregistrées, code mort retiré. Montrer à
   l'utilisateur l'écart numérique ancien/nouveau avant de remplacer le run
   T-6.

### Fusion de master #109

Master #109 change le modèle exprès (départs d'approche tournés vers l'axe,
seuils du curriculum T-6 et PC-7, checkpoint T-6 à l'update 243). Trace dorée
Flight : avec les départs et scénarios de #109 annulés, la branche redonne
`0x419024d0dcfe2a0d` ; avec eux, `0x65f430ba639ad945` (Debug et Release),
nouvelle référence. `--check-checkpoint` charge le checkpoint T-6 de master
(`compatible=true updates=243`).

## LibTorch retiré du dépôt (accord de l'utilisateur, 23/11/2026)

L'utilisateur a autorisé à porter le contrôleur rouge RLMission et à abandonner
le backend GPU libtorch.

- Production rouge : `Examples/Airspace/models/{RLMission,RLSNA}/{prod,eval}/policy.txt`,
  convertis de `model.pt` sans torch (`tools/legacy_torch/convert_legacy_model.py`),
  calculés par `AirspacePolicies` (MissionPolicy, InterceptionPolicy) avec le même
  forward, y compris le cas sans piste (le -1e9 des pistes absentes atteint les
  têtes : actions ~5e8, coupées à ±1 par le modèle). Parité LibTorch prouvée en CI
  au commit 97279dd (`LegacyPolicyTorchParityTests`, Debug et Release, cas de
  référence + 200 observations aléatoires par modèle, tolérance relative 1e-4),
  puis le test a été retiré avec LibTorch. `LegacyPolicyTests` garde la parité
  avec les actions de référence (numpy, `tools/legacy_torch/reference_outputs.py`).
- Entraînement rouge : `TrainingAgent<Model>` (Examples/Airspace/Agents/Bindings) apprend avec
  un `GraphLearner` PPOKit sur `Agents/Mission/{task,graph}.json` et
  `Agents/Interception/{task,graph}.json` ; écrit `training.ppokit` et
  `eval/policy.ppokit` ; l'évaluateur promeut `policy.ppokit` dans `prod/`, que
  les agents de production préfèrent à `policy.txt`. Les poids LibTorch ne se
  réentraînent pas : un entraînement repart d'une nouvelle politique PPOKit.
  Différences assumées avec l'ancien réseau : log-std indépendant de l'état,
  requête d'attention en une couche (même famille de fonctions), ensemble vide
  à zéro. La dernière transition d'un épisode réinitialisé est marquée tronquée.
  `TrainingAgentTests` (desktop) ; vérifié aussi en local avec ASan/UBSan.
- Supprimés : `Framework/RL/PPO` (TorchLearner, RLAgent, PPO), `PPOKit/torch`,
  `TorchProductionPolicy`, les `model.pt`, la dépendance et les copies de DLL,
  `PPO_LIBTORCH_VARIANT` dans les workflows (clés de cache `nolibtorch`).

## Rangement du dépôt par exemple (accord de l'utilisateur, 23/11/2026)

Chaque exemple garde dans `Examples/<Nom>` son code, ses réglages, ses modèles
et ses projets ; `resources/`, `Projects/` et `Tests/` n'existent plus.

- `Examples/Shared` : `World2D` (ex-`Framework/Simulation/World2D`), `Host/example.hpp`
  (contrat d'un exemple, cible `PPOExampleContract`), `settings/` (Environment,
  Renderer, AgentProfiles, Platforms) et `3D/`.
- `Examples/Airspace` : `Agents/Bindings` (ex-`Framework/RL/Bindings`),
  `settings/` (Scenarios, Training, Evaluation), `models/{RLMission,RLSNA}`,
  `tests/policy_checkpoint_tests.cpp` (ex-`Tests/`).
- `Examples/BlueDefense` : `projects/BluePatrol` (scénario, réglage MAPPO,
  checkpoints) et `curricula/BluePatrolCurriculumV3.json`.
- `Examples/FlightControl3D/projects/{T6,PC7}`.
- `Main/Catalog` : catalogue des exemples (ex-`Examples/{data,src,tests}`).
- Une seule copie : les projets. Les scénarios par défaut des exemples, les
  `.bat` et les workflows pointent sur eux. Supprimés avec l'accord de
  l'utilisateur : modèle T-6 v3, archive BluePatrol v5, curricula V1 et V2
  (le workflow Blue n'accepte plus que `v3`).
- Chemins : cible `PPOLayout` (racine CMake) → `PPO_SHARED_SETTINGS`,
  `PPO_SHARED_3D`, `PPO_AIRSPACE_SETTINGS`, `PPO_BLUEDEFENSE_PROJECT`,
  `PPO_FLIGHT_T6_PROJECT`, `PPO_FLIGHT_PC7_PROJECT` ; `RESOURCE_PATH` disparaît.
  `Serializer::SetSearchPaths` : PPO.exe cherche un fichier dans les réglages
  d'Airspace puis dans les réglages partagés (test dans `SerializerSaveSafetyTests`).
- Un scénario hors projet garde ses modèles à côté de lui
  (`ModelDirectoryFor` pour Flight, `LocationsFor` pour Blue).
- Corrigé au passage : `agentProfiles` du scénario BluePatrol pointait vers un
  fichier absent (chemin relatif de l'ancien emplacement).

## Dernier état de la CI (PR #105)

Commit `406e245` (studio utilisable, sondes Flight/Blue, poids Blue), logs
lus job par job : Core Ubuntu et Windows 43/43 ; Flight 51/51 sur les 4
jobs ; Desktop VS2022 Debug et Release 103/103 (agent Blue, studio,
`PPOKitStudioHomeTests` (supprimé depuis), `PPOKitTorchTests` compris). La trace dorée Flight
est « Skipped » hors Linux GCC 13 comme avant ; vérifiée localement
(`0x419024d0dcfe2a0d`, identique). `7391f7a` échouait sous MSVC
(`LogStd` instancié pour la première fois depuis un lambda) : corrigé dans
`406e245`.

Commit `17898e6` (outillage des exemples), logs lus job par job : Core
Ubuntu et Windows 41/41 ; Flight 49/49 sur les 4 jobs (Ubuntu GCC 15 et
Windows MSVC, Debug et Release) ; Desktop VS2022 Debug et Release
100/100. `PPOKitScaffold`, `PPOKitReachCheck`, `PPOKitCommandLineTests`
passent partout, `PPOKitTorchTests` passe en Desktop Debug et Release
(déjà vert sur `24ac940`, 97/97). Les avertissements C4267 du job Desktop
viennent des en-têtes de libtorch (ATen, dynamo), pas de PPOKit.

Commit `937eb90` (studio étapes 3 et 4 : réglages PPO, poids en cours de
run) : Core vert ; Flight vert sur les 4 jobs (40/40, dont
`PPOKitWeightRequestTests` et `FlightControl_session_tests` avec
`StudioWeightRequests`) ; Desktop VS2022 Debug et Release 89/89 (studio
compilé sous MSVC, `PPOKitStudioSettingsTests` et
`PPOKitStudioDashboardTests` passés). `b57e55c` (étape 3) : 86+3 tests
verts aussi.

Commit `9f1017b` (journal écrit par Flight, Blue et le collecteur Owned) :
Core vert ; Flight vert sur les 4 jobs (38/38 Windows) ; Desktop VS2022
Debug et Release 86/86 (agent Blue avec journal compilé sous MSVC,
`PPOKitStudioDashboardTests` passé). Deux échecs Windows corrigés en
route : `d639091`, Flight `session_tests` — Windows refuse de renommer un
fichier ouvert, or un test reprend un dossier pendant que la première
session vit encore ; le writer ouvre maintenant le fichier à chaque
événement et `Resume` ne réécrit que s'il retire des événements.
`25dad3a`, Desktop Release `PPOKitStudioDashboardTests` 0xc0000409 :
`remove` du journal encore ouvert levait une exception hors de `main` ;
corrigé par la même cause, et le test signale désormais les exceptions
(`73c270a`).

Commit `de5ccc1` (Blue client du `HostedTrainer`) : Desktop VS2022 Debug
et Release 83/83 (`generalized_patrol_team_agent.cpp` compilé,
`BlueDefenseGeneralizedLearnerTests`, `PPOKitHostedTrainerTests` passés
sous MSVC). Le job Flight Ubuntu GCC 15 Release échouait sur le test
d'identité de `TeamRollout` : variance expliquée à quelques ulps avec un
agent. Cause confirmée : GCC fusionne multiplication et addition
(`-ffp-contract=fast`, mode GNU) différemment dans deux copies de la même
formule ; avec `-ffp-contract=off` pour ce seul test, identité au bit près
partout (commit `f412668`, Flight vert sur les 4 jobs). Précédemment
`1382815` (acteur Blue calculé par le graphe) : 81/81. Historique
ci-dessous.

### Commit `b3ed698`, étape 7c

Tout est au vert : Core C++17 tests ; FlightControl3D JSBSim (Ubuntu GCC 15 et
Windows MSVC, Debug et Release, 28/28, dont `PPOKitBackendLearnerTests` et
`PPOKitEvaluationTests`) ; Desktop VS2022 Debug et Release (72/72). Le
couloir s'entraîne et passe ses gates aussi avec la `std::normal_distribution`
de MSVC (vérifié dans les logs des jobs).

## Pièges connus

- Quand master change Flight, mesurer l'empreinte golden du master pur
  **dans `build/release`** : `git checkout origin/master -- Examples/FlightControl3D`,
  construire et lancer `FlightControl_golden_trace_tests`, puis restaurer avec
  `git reset -q HEAD -- Examples/FlightControl3D && git checkout -- Examples/FlightControl3D && git clean -fdq Examples/FlightControl3D`.
  Comparer ensuite les lignes `[FLIGHT_GOLDEN_EPISODE]` des deux versions.

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

### Détail de l'étape 6b (Flight)

- `FlightControlLearner` génère `flight_networks.hpp` depuis `task.json` et
  `graph.json`, et `flight_learner.cpp` n'écrit plus ses
  `StructureSpecification` à la main : `Actor<Width>` et `Critic<Width>` sont
  les types générés, toujours instanciés pour 32, 64 et 128 et choisis à
  l'exécution. Des `static_assert` relient le graphe au contrat de Flight.
- Golden trace inchangée au bit près : ce sont exactement les mêmes types.
- Blue : avec l'accord de l'utilisateur, le minimum CMake passe à **3.19**.
  `ppokit_generate_networks` accepte alors une cible `INTERFACE` (cible de
  génération dédiée et dépendance propagée aux consommateurs). Le critique
  centralisé `GeneralizedCriticConfig::Network` est le type généré depuis
  `BluePatrol/graph.json` ; `HiddenDim` (256) et `Layers` (3), écrits dans
  l'en-tête des checkpoints d'entraînement, en sont dérivés.
  `BlueDefenseNetworkTypeTests` vérifie que cette forme ne change pas sans
  qu'on le voie. La trace MAPPO Blue reste identique au bit près. L'acteur à
  attention reste écrit à la main.

### Fusion de master #106

- Master a ajouté une tête **Bernoulli** (demande de sortie du train), deux
  observations (35 au total), un trim JSBSim et un scénario PC-7.
- `ppokit/heads/bernoulli.hpp` remplace les fonctions de Flight (Softplus,
  Sigmoid, log-probabilité, entropie et gradients du train). `task.json`
  accepte `"head": "bernoulli"`. `GaussianMlpLearner` refuse pour l'instant
  les tâches avec des actions Bernoulli (`static_assert` sur `GaussianOnly`).
- Le test de correspondance de `task.json` a détecté le changement de
  contrat dès la fusion : c'est son rôle.

## Décisions prises avec l'utilisateur

- Génération de code depuis JSON pour rl_tools ; libtorch derrière le même
  frontal pour Airspace.
- Composants DLL boîte noire, câblés par leurs ports.
- Les trois exemples migrent et deviennent des exemples de référence, faciles
  à prendre en main.
- RLMission migre en dernier, sur approbation explicite.
- Interface : curriculum, récompenses, paramètres PPO avec infobulles, sondes,
  GPU via libtorch.
- Studio : pas de liste de tous les modèles ; l'utilisateur choisit le
  dossier. Un projet est un dossier autonome (découpage ci-dessus accepté).
  Flight et Blue deviennent des projets complets, checkpoints copiés. Train,
  évaluation et inférence se lancent depuis des boutons du studio, en
  arrière-plan, avec console et suivi en direct. Les learners Flight et Blue
  migrent sur PPOKit.
- Exécution (23/11/2026, option B) : le studio est l'exe principal et commande
  en direct ; l'entraînement tourne dans `ppokit-run` (DLL d'environnement et
  DLL réseau à API C stable qui cache rl_tools) ; `ppokit-viewer` voit le monde
  via les blocs Visualiseur (observateurs, UDP caché dans la DLL). Plan :
  design §18.
- Étape 1 (§18) faite : blocs Visualiseur. Capacité `observer` des
  composants, section `world` de task.json (hors ShapeHash : les checkpoints
  restent valides), section `observers` de graph.json (nœuds input + blocs,
  hors identité des artefacts), `runtime::ObserverSet` (appel après chaque pas,
  sans allocation, un observateur en échec est coupé sans arrêter le run),
  canevas « Observers » du studio. Tests : PPOKitComponentTests,
  PPOKitGraphDocumentTests.
- Étape 2 (§18), première moitié faite : canal en direct (`PPOKit::Live`).
  Socket local (127.0.0.1, port annoncé dans `checkpoints/live.json`), journal
  envoyé entier au raccrochage puis événement par événement, `status` 4 fois
  par seconde (état, pas, épisodes, récompenses de la fenêtre), commandes
  pause / play / stop / poids. `LiveControl` (StepWatcher du collecteur) :
  pause entre deux pas ou à une frontière d'update, stop = sauvegarde puis
  fin. Branché dans la commande `train` générique (reach, ppokit-new) et dans
  FlightControlHeadless (essayé sur une copie du projet T-6 : pause, play,
  stop à l'update 244). Le tableau de bord du studio s'y raccroche seul et
  offre Pause / Play / Stop ; les poids passent par le canal. Reste :
  `ppokit-run` générique, qui a besoin de la DLL réseau (étape 3), et Blue
  (étape 6).
- Étape 3 (§18) faite pour le moteur de graphes : DLL réseau à API C stable
  (`ppokit/network/network_abi.h`, ABI 1) : `ppokit_policy_*` pour
  l'inférence, `ppokit_learner_*` pour l'entraînement (décider, ajouter,
  mettre à jour, sauver, reprendre). `ppokit_add_network_library(TARGET TASK)`
  la construit pour un projet (taille fixée par la tâche, structure lue dans
  network.json à l'ouverture). Client C++ `PPOKit::Network` (RAII). Chargement
  des DLL commun (`component/native_module`). Test : la DLL de reach donne au
  bit près les décisions, la mise à jour et la politique du learner en
  processus. Le moteur rl_tools derrière la même API viendra avec le T-6.
- Étape 2, seconde moitié faite : `ppokit-run` générique (train / eval /
  play) sur deux DLL nommées dans project.json (`"libraries"`) : la DLL
  d'environnement (`ppokit/environment/environment_abi.h`, SDK
  `PPOKIT_ENVIRONMENT`) et la DLL réseau. Il tient la boucle, les checkpoints,
  le journal, le canal direct, les poids (fichier ou direct) et appelle les
  observateurs de network.json après chaque pas avec l'état du monde. Test :
  la tâche reach en DLL d'environnement, entraînement, reprise, jeu, et un
  observateur qui reçoit le monde (PPOKitRunTests).
- Étape 4 (§18) faite : `ppokit-viewer` (raylib, programme à part) charge la
  vue d'une DLL Visualiseur (`ppokit/view/view_abi.h`), reçoit les trames UDP
  (`ppokit/view/frames.hpp`, `live::DatagramReceiver`) et lui fournit une
  table de dessin (pas de raylib dans les DLL de vue). Raccrochage à chaud,
  ligne d'état (port, trames, silence). SDK : `view::FrameSender` côté
  observateur, `PPOKIT_VIEWS` côté vue. Bouton « View » du studio. Test sans
  fenêtre : observateur → UDP → vue → table de dessin enregistreuse
  (PPOKitViewTests). Le programme lui-même n'est compilé qu'en CI desktop
  (raylib absent en local ; vérifié en syntaxe contre une maquette).
- Étape 5 (§18), première partie faite : le T-6 sous `ppokit-run`, à côté du
  run T-6 existant (non remplacé). DLL d'environnement `flight_environment`
  (JSBSim, mission, récompenses, curriculum de départs ; niveau gardé dans
  `checkpoints/ppokit_curriculum_level.txt`), DLL réseau `flight_network`
  (moteur de graphes), DLL Visualiseur `flight_visualiser` : observateur
  `flight_watch@1` (avion, piste, commandes, récompenses → UDP) et vue 3D
  `flight` (piste, axe d'approche, modèle de l'avion, caméra de poursuite,
  données de vol). Projet `Examples/FlightControl3D/projects/T6_ppokit` :
  tâche T-6 + `world` (avion 8, terrain 6), réseaux et scénario de T6,
  training.json aux valeurs PPO du scénario, `play --realtime`. Les DLL de
  composants nommées dans network.json sont aussi cherchées parmi les
  bibliothèques du build (`component::SetKnownModules`, registre des
  programmes) : ppokit-run, le studio, la DLL réseau et le test des projets du
  dépôt. Test : PPOKitRepositoryProjectsTests vérifie T6_ppokit contre T6.
- Écart numérique ancien / nouveau learner (8 updates `--fresh`, 32768 pas
  chacune, même scénario). Ancien (FlightControlHeadless, rl_tools) : épisodes
  de 280 à 455 pas, récompense par pas −0,04 à −0,09. Nouveau avec
  l'initialisation Glorot du moteur de graphes : épisodes de 60 à 80 pas,
  récompense par pas qui glisse de −0,06 à −0,11 (moyennes initiales
  |μ| ≈ 0,41 profondeur, 0,17 ailerons : l'avion s'écrase en ~70 pas).
- Le moteur de graphes initialise désormais comme rl_tools (poids et biais
  uniformes dans ±1/√entrées, toutes les couches) : |μ| initiales ≈ 0,10.
  Nouveau learner remesuré : épisodes de 104 pas qui montent à 173–182 pas,
  récompense par pas −0,053 à −0,065, dans la plage de l'ancien. Les
  épisodes restent plus courts qu'avec l'ancien (une graine de chaque côté).
  Les checkpoints existants se reprennent (l'initialisation ne sert qu'à un
  départ à neuf). Run T-6 existant toujours non remplacé.
- Learner graphes aligné sur le learner FlightControl3D. Écarts trouvés en
  comparant le code : (1) composantes actives selon l'état — l'ancien ne tire
  ni n'apprend le frein en vol ni le train une fois verrouillé, le nouveau
  les comptait toujours dans la vraisemblance, le KL, l'entropie et les
  gradients ; (2) bornes du log-std [−3,5 ; −1] contre [−5 ; 2] ;
  (3) initialisation (corrigée juste avant). Ajouts : section `"active"` des
  groupes d'actions de task.json (`{"when": feature, "above" | "at_most":
  valeur}`, hors hash de forme ; GraphLearner ne tire pas, ne note pas et
  n'apprend pas une composante inactive) ; paramètres `minimum_log_std` et
  `maximum_log_std` de training.json. Tâche FlightControl3D (et T6, PC7,
  T6_ppokit) : frein actif si `rollout` > 0,5, train si `gear_commanded`
  ≤ 0,5, la règle que l'ancien learner codait en dur ; le checkpoint T-6
  (update 243) reste compatible. Tests : PPOKitGraphLearnerTests
  (ActivitiesMaskComponents, ActivityErrors).
- Comparaison sur 3 graines (42, 7, 123), 20 updates `--fresh` chacune.
  Longueur moyenne des épisodes aux updates 16–20 : ancien 692 / 117 / 131
  pas, nouveau 179 / 198 / 130. Récompense par pas aux updates 16–20 : ancien
  −0,007 / −0,036 / −0,052, nouveau −0,033 / −0,035 / −0,054. Les deux
  learners sont dans la même plage ; la graine 42 de l'ancien est meilleure
  (ses graines d'environnement sont tirées au hasard à chaque run : une
  première série avec cette graine donnait ~400 pas et −0,065). Pas de
  conclusion de convergence sur 20 updates.
- Portage unique du checkpoint T-6 (update 243) : le checkpoint rl_tools
  (`PPO_FLIGHT_POLICY_V3`) a été converti une fois vers le learner graphes
  (poids, biais, log-std, moments d'Adam, niveau de curriculum ; l'état
  aléatoire repart de la graine de training.json), sans outil gardé dans le
  dépôt. Vérifié : sur 2000 pas de vol, écart des moyennes 0, du critique
  3,6e-6, du log-std 0, décision du train identique ; validation 17/20 à la
  difficulté 0,727273 comme FlightControlHeadless ; reprise 244–248 dans la
  même plage (récompense par pas 0,0026–0,0044 contre 0,0027–0,0042).
- Évaluation générique (sans perte par rapport à FlightControlHeadless) :
  métriques d'épisode de task.json, ABI environnement 2 (départs d'évaluation,
  métriques de fin d'épisode), suites eval/ avec seuils, objectif de fin
  d'entraînement ("goal"), rythme pendant l'entraînement et classement du
  BEST ; ppokit-run : campagnes pendant train, BEST dans checkpoints/best/,
  journal et direct, `--until-passed`, `--evaluate-every`, `--max-seconds`,
  eval par suites, play --best ; studio : section Evaluation (édition des
  suites, campagnes en direct, BEST).
- Étape 5 terminée : le T-6 est un projet ppokit-run
  (`Examples/FlightControl3D/projects/T6` : task avec world et métriques,
  network avec le Visualiseur, training.json aux valeurs PPO du scénario,
  eval/validation.json = validation de FlightControlHeadless, checkpoint porté
  à l'update 243). Le workflow d'entraînement T-6 lance ppokit-run
  (`--until-passed validation`, budget d'updates et de temps, reprise d'un run
  précédent) ; le lanceur T-6 ouvre le projet dans le studio. PC7 reste sous
  FlightControlHeadless.
- Étape 6, monde pas à pas : `Environment::BeginStep` / `EndStep` (et
  `RequestReinitialization`), la boucle du fil exécute les mêmes moitiés.
  Le monde ECS compile avec GCC (`PPO_BUILD_WORLD` : monde et exemples 2D sans
  graphisme ; le CI Core le construit sous Ubuntu et Windows).
- Étape 6, multi-agents : ABI environnement 3 (lignes d'agents, `available`,
  récompense par agent en plus des canaux par agent, `begin_step` quand le
  monde change entre deux pas), ABI réseau 2 (learner et politique d'équipe
  `ppokit_team_*`, réglages, sondes ; `team_sdk.hpp`). ppokit-run entraîne,
  évalue et joue une équipe (journal par agent comme Blue dans PPO.exe) ;
  `--policy FILE` pour eval/play, `--until-update N`, noms de checkpoints dans
  project.json (`"checkpoints"`). Tests : PPOKitRunTests (équipe team_line),
  PPOKitNetworkLibraryTests, PPOKitProjectTests.
- Étape 6, Blue sous ppokit-run : `blue_environment` (monde BlueDefense,
  rouges RLMission, agent bleu en mode `External`) et `blue_network`
  (GeneralizedPatrolLearner et acteur de production inchangés, fichiers V7/V10
  tels quels). Parité (BlueDefenseExternalParityTests) : après 3 updates de
  2048 pas (fin d'épisode comprise), état V7 et acteur V10 identiques au bit
  près à ceux du mode Training de PPO.exe ; campagne de l'acteur de prod :
  cas 2 à 8 identiques (durée, issue, survivants, qualités), cas 1 même
  graine (PPO.exe crée son premier cas au chargement : son premier lidar voit
  déjà le cas ; toute autre réinitialisation, dans PPO.exe comme dans
  ppokit-run, vient après les règles d'avant pas et garde le lidar du dernier
  état — mesuré : seules les 61 valeurs de lidar diffèrent au premier pas). Suite
  `eval/validation.json` = campagne du fichier d'entraînement (graines 4242+,
  8 cas, toutes les 10 updates, BEST au meilleur score). Le projet BluePatrol
  s'entraîne (jusqu'à l'update 4000) et s'évalue avec ppokit-run ; l'inférence
  reste dans PPO.exe (vue 2D) jusqu'à la vue Blue. Limite constatée, inchangée :
  l'état V7 écrit sous Windows ne se relit pas sous Linux (texte de
  `std::mt19937` : 624 nombres MSVC, 625 libstdc++).
- Étape 6, vue Blue : monde de BluePatrol (une ligne par bleu + scène),
  `blue_visualiser` (observateur UDP et vue 2D pour ppokit-viewer),
  BlueDefenseVisualiserTests (bout en bout). Workflow « Inference Validation »
  sur ppokit-run (suite générée par gate, rapport JSON), tests Python mis à
  jour.
- Étape 6, correction (dégradation mesurée en Inference Validation : 1v0 et
  2v0 de 50 % à 0 %, 2v2 de 80 % à 40 %, même acteur BEST, mêmes réglages) :
  `blue_environment` n'appliquait pas Environment.json, que PPO.exe applique
  par `Serializer::Load` après son scénario. Le monde tournait avec les
  valeurs par défaut de sEnvironmentState (pas de 0,02 s au lieu de 0,2 s,
  limites 112 500 × 150 000 m au lieu de 44 659 × 71 454 m) pendant que
  l'agent comptait en pas de 0,2 s : les mobiles parcouraient un dixième de
  leur chemin par pas, aucun tour de patrouille ne finissait. Le modèle
  n'était pas en cause : acteur V10 identique à 1e-6 entre master et la
  branche, réglages identiques octet par octet. `Environment::LoadSettings`
  applique le fichier comme le rappel du Serializer (même fonction) ;
  `LoadExternalScenario` l'appelle. Le test de parité ne l'avait pas vu, sa
  référence PPO.exe n'appliquant pas non plus le fichier : elle l'applique
  désormais, et le test vérifie le pas et les limites du monde de la DLL.
  1v0 local (acteur de prod, 20 graines) : 0/20 → 15/20 succès.
- Étape 6, entraînement par gates sous ppokit-run (« Restart Progressive Blue
  Defense », mappo-training-windows, mappo-blue-defense-windows) :
  `run_stage` lance `ppokit-run train --until-update <maxUpdates>
  --save-every <saveEveryUpdates|checkpointEvery> --max-seconds` ; pendant le
  bloc, la suite prend la campagne du fichier d'entraînement (graines, cas,
  cadence, que la CI change : 20 graines à partir de 1004242), puis retrouve
  ses octets ; la campagne d'un bloc se
  lit dans son journal (ligne `evaluation 'validation' at update N`, P/N des
  `[MAPPO_EVAL_EPISODE]`) et son résultat dans le journal d'exécution. La
  promotion prod/.score de PPO.exe passe par le BEST de la suite (vidé avant
  chaque bloc, les champions `best/<gate>/` ne sont pas touchés) : même règle
  (score strictement supérieur au .score du même P/N, ou aucun), même format.
  Mode sans curriculum : `mappo_curriculum.py --single-stage`. Télémétrie :
  `mappo_metrics.py --journal --journal-from-line` (le journal remplace le
  CSV et voyage avec les checkpoints). Mode External : les lignes
  `[MAPPO_EVAL_EPISODE]` et `[MAPPO_EVAL_LOSS]` sont écrites aussi (fonctions
  communes, sortie de PPO.exe identique à l'octet ; campagnes numérotées dans
  le processus, une ligne par écriture) ; `[MAPPO_PATROL_PROGRESS]` dit
  `phase=evaluation` pendant une campagne External. Fumée CI : un épisode
  `ppokit-run play` de l'acteur de prod commité.
- Étape 6, vue et inférence Blue : `blue_visualiser` (`blue_watch@2`) dessine
  ce que la mini-carte de PPO.exe dessine (zone défendue, rectangle d'échec,
  zones interdites par camp, trace et sens de patrouille, waypoints
  actif/validés/cap, cônes d'engagement, badges L/W k, références projetées et
  de guidage), le pseudo-lidar du bleu cliqué (leader par défaut) et une
  explosion par destruction ; cadrage initial sur le rectangle d'échec. Le
  monde gagne six groupes (`blue_view`, `red_cones`, `lidar`, `zones`,
  `bounds`, `patrol`), remplis par l'agent depuis l'ECS (capture de scène,
  sans allocation) ; la zone défendue est celle du snapshot de PPO.exe (les
  limites du monde ; elle manquait à la vue). Vérifié sur des rendus SVG de
  vraies trames (4v4 avec zones, 2v0 en patrouille). Inférence BluePatrol :
  `ppokit-run play` de l'acteur de prod (`--episodes 1000000 --realtime
  --step-seconds 0.004`) ; `play` s'arrête après l'épisode en cours sur
  `stop.request` (Stop du studio) ; le studio ouvre ppokit-viewer au
  lancement d'une inférence quand le projet a un Visualiseur. Lanceur local
  MAPPO 2v2 : blocs de la CI (`run_stage`) et ppokit-viewer au lieu de la
  GUI de PPO.exe.
- Étape 6, vue de FlightControl3D au niveau de FlightControl3D.exe : le monde
  de T6 (et PC7) passe de 14 à 331 valeurs (`Core/flight_world.hpp`, neuf
  groupes) : tout ce que le rendu de bureau lisait de la simulation (piste
  et pont, guidage, mission et dernier épisode, atterrissage, dernier crash
  gardé d'un épisode à l'autre, sommes d'épisode et poids des récompenses,
  diagnostics de convergence sur le dernier épisode complet). `flight_watch@2`
  et `flight_visualiser` : scène 3D portée de `flight_view.cpp` (mer, houle,
  sillage, zone d'atterrissage, piste et marquages, guidage, avion, crash
  figé et explosion), trois caméras, étiquettes (seuil, fiche avion, pont,
  CRASH), profil d'approche, panneau en onglets (vol, commandes et
  récompenses, diagnostics, crash). API de vue 2 (triangles 2D/3D, mesure de
  texte) ; API d'environnement 4 (`set_learning`) : `play` et les évaluations
  ne font plus avancer le curriculum de FlightControl3D (`play` l'avançait
  et réécrivait `ppokit_curriculum_level.txt`). Helpers de saturation et
  `DeckState` déplacés dans Training/Core pour être partagés ;
  `flight_display.hpp` passe dans Core. Test bout en bout
  (FlightControl_visualiser_tests : ppokit-run joue T6, la vue dessine scène,
  profil et chaque onglet choisi au clic ; seul l'apprentissage compte dans
  la fenêtre de promotion). Vérifié sur des rendus SVG de vraies trames
  (poursuite, vue d'ensemble, bout de piste, houle au dernier niveau).
- PC7 sous ppokit-run : mêmes réseaux, task (avec monde), training.json et
  suite que T6, sans checkpoint (premier entraînement depuis zéro) ; le
  lanceur ouvre le studio sur PC7 ; le workflow d'entraînement de vol choisit
  T6 ou PC7. Inférence T6/PC7 : vol en temps réel jusqu'au Stop.
- Étape 7, FlightControl3D : FlightControl3D.exe, son rendu (Renderer/), le
  pont ECS de PPO.exe (`flight_example`), ses entrées du catalogue
  (`flight-control-3d`, `-inference`), les raccourcis
  (`CreateFlightShortcuts.ps1`) et le mode FlightControl3D du rendu de
  PPO.exe sont retirés ; FlightControl3D se construit sans le bureau. Les
  vérifications des panneaux de bureau (display_tests) passent sur la vue :
  trames connues (cibles, pente, horloge, pont, taux de chute, profil, barres
  de récompense, raisons du crash) ; le statut compte les épisodes, posés,
  arrêts, crashs et limites de temps comme la fenêtre Learning de PPO.exe.
  FlightControlHeadless reste (trim d'un avion, parité du learner).
- Graphes étendus pour porter les agents torch d'Airspace sans perdre leur
  architecture : `masked_attention` avec `"keys": true` (scores sur un
  ensemble de clés distinct, sortie moyennée sur l'ensemble),
  `gaussian_head` avec `"log_std": "state"` borné (`log_std_min`,
  `log_std_max`), nœuds du critique `"share": "<nœud de l'acteur>"` avec
  apprentissage joint (un Adam, `value_coefficient`, section `adam` du
  checkpoint, reprise bit à bit). Identité inchangée pour les graphes qui
  n'utilisent pas ces attributs. Studio : case des clés, mode du log-std,
  sélecteur de partage ; changements groupés en une étape d'annulation.
  Tests : gradients aux différences finies (attention à clés, log-std d'état
  et ses bornes), apprentissage du jouet `switch` à tronc partagé, reprise
  exacte, refus de deux taux d'apprentissage, erreurs du schéma.
  `max_gradient_norm` (0 : coupé) : borne de la norme globale des gradients
  par pas d'optimiseur, comme `clip_grad_norm_` de torch (0.5 dans l'ancien
  PPO de RLMission et RLSNA) ; refusé par le learner rl_tools et les modèles
  d'équipe plutôt qu'ignoré.
- Format `.ppokit` pour tous les exemples. BlueDefense : le learner MAPPO
  rl_tools écrit `checkpoints/latest/latest.ppokit`, `latest/policy.ppokit`
  et `prod/policy.ppokit` (en-tête d'identité `backend: rltools` depuis
  task.json et graph_blocks.json embarqués, lignes `contract` et `team`
  reprenant les contrôles stricts des V7/V10, sections de PPOKit, lecture
  mise en attente avant application) ; les V7/V10 se lisent encore (artefacts
  des runs CI précédents). `blue_checkpoint_import` a converti les fichiers
  du projet : relus puis réécrits en V7/V10, identiques octet pour octet.
  Scripts CI (`tools/ci/blue_checkpoint.py`, curriculum, champions, reprise,
  validation d'inférence) et workflows suivent, en acceptant les anciens noms.
- État RNG portable (`ppokit/artifacts/random_state.hpp`) : forme standard
  de `std::mt19937` (624 valeurs) écrite partout ; lecture de la forme
  libstdc++ (625) convertie exactement (inversion d'un « twist »). Le
  moteur de Blue devient `std::mt19937` explicite (rl_tools utilisait
  `std::default_random_engine`, un autre moteur sous libstdc++), semé comme
  avant : identique sous MSVC. Un checkpoint passe de Windows à Linux.
- Airspace : checkpoints LibTorch convertis (voir §18.9, item 7, du document
  de conception) ; `InferenceSettings` pour jouer une politique dont le
  critique partage le tronc de l'acteur.
- Fusion de master (#110 guidage des hippodromes, contrat 8 ; #111 workflow PC-7) :
  le seed T-6 de master (`t6texan2_joint_v5`, run CI 52, update 340, niveau 11) est
  porté en `.ppokit` dans `projects/T6` par l'outil de conversion du T-6 (restauré le
  temps de la conversion) : mêmes moyennes et log-std, critique à 3e-6, même décision
  du train sur 2000 pas ; `ppokit-run eval` donne 19/20 comme FlightControlHeadless.
  Le workflow PC-7 de #111 (labels `flight-pc7-curriculum`, `flight-pc7-resume-<id>`,
  artefact `flight-control-pc7-training`) passe sous ppokit-run sur `projects/PC7` ;
  le workflow T-6 n'entraîne plus que le T-6.
- Airspace RLMission sous ppokit-run : mode External de `TrainingAgent`,
  DLL `rlmission_environment` (ABI d'équipe à un agent : récompense bornée
  comme dans PPO.exe ; troncature à la taille du rollout, décision de
  l'utilisateur), DLL `rlmission_network`, projet
  `Examples/Airspace/projects/RLMission` qui reprend le checkpoint converti
  (update 127). Test de parité External / entraînement dans le monde (même
  checkpoint après une update, vérifié discriminant) et test de fumée
  ppokit-run (play, train une update sur une copie).
- Airspace : suites d'évaluation et projet RLSNA. La boucle de cas de
  l'évaluateur est partagée (`EvaluationCampaign`) entre `Evaluator` et les
  DLL d'environnement (`ResetEvaluation` = cas de numéro seed, six métriques
  par épisode, `eval/campaign.json` sur les seeds 0 à 99, BEST sur
  `score.mean`). DLL `rlsna_environment` / `rlsna_network` et projet
  `projects/RLSNA` (jeu et évaluation, sans entraînement : aucun scénario).
  `Serializer::Unregister` : plusieurs environnements dans un même processus
  ne laissent plus de rappels pendants. Parité des notes Evaluator /
  `ppokit-run eval` à 1e-4 dans `Airspace_ppokit_run_smoke`.
- Airspace : vue 2D tactique. Ligne de monde des DLL RLMission et RLSNA
  (`tactical_world.hpp` : entités et scène), DLL `airspace_visualiser`
  (observateur `airspace_watch@1`, vue `airspace` : la minimap tactique de
  PPO.exe) branchée dans les deux projets ; `AirspaceVisualiserTests`.
  Correction : le premier épisode d'entraînement de `rlmission_environment`
  commence par la réinitialisation du monde (il finissait à son premier pas).
- Étape 7 : PPO.exe retiré. `Main/` (hôte, catalogue et sélecteur
  d'exemples, leurs tests), le rendu d'Airspace (`Examples/Airspace/Renderer`,
  `Framework/Support/Renderer`), `Renderer.json`, `imgui.ini`, les raccourcis
  `PPO.exe.lnk` et l'entrée PPO.exe d'Airspace ; le contrat
  `Examples/Shared/Host/example.hpp` réduit aux modes de BlueDefense dans le
  monde (références des tests de parité). Les exemples tournent sous
  ppokit-run, ppokit-viewer et ppokit-studio.
- Modes dans le monde et plomberie de l'interface retirés (décision de
  l'utilisateur) : le contrôleur Blue et `TrainingAgent` sont toujours pilotés
  de l'extérieur ; `Evaluator`, `Definition`/`InferenceDefinition`/
  `EvaluationDefinition`, la campagne Blue dans le monde, leurs tests de
  parité ; dans Framework, le thread, les snapshots, les commandes temps
  réel, `Serializer::Save`, la réflexion de Meta, `runner`, `evaluation`,
  `learning_monitor`, `team_episode` et `sEvaluatorState`. `IsLastStep` d'un
  agent Airspace reste faux, comme sous ppokit-run avant ce retrait : la
  récompense d'entraînement de RLMission ne change pas (même suite de
  récompenses, mêmes mises à jour, vérifié sur une graine fixée) et
  `AirspaceTrainingRewardTests` vérifie qu'aucun pas ne paie l'échec de mission
  du dernier pas.
- Graine des épisodes RLMission : l'initialiseur aléatoire du monde tire de
  `sInitializationRandom` (`InitializationEngine`), que
  `Environment::SeedInitialization` initialise ; `rlmission_environment` lui
  donne la graine de `ppokit-run` (celle de `training.json` + 1 pour `train`).
  Un entraînement RLMission est reproductible (mêmes retours, mêmes
  évaluations) ; sans graine, le monde tire de l'entropie du système comme
  avant. `AirspaceTrainingRewardTests` vérifie que la même graine donne les
  mêmes récompenses et qu'une autre en donne d'autres. Inventaire des sources
  et guide de configuration remis à jour.
- Audit P0-02 (intégrité du BEST) : une génération BEST était nommée
  `<suite>_<update>` ; un autre entraînement arrivé au même update gardait
  l'ancien fichier de poids et réécrivait le manifeste, qui pointait alors sur
  d'autres poids que ceux évalués. Les générations sont nommées par leur
  contenu (`<suite>_<update>_<16 premiers chiffres du SHA-256>`), copiées dans
  un fichier temporaire, vérifiées contre l'empreinte des poids évalués puis
  renommées ; le manifeste `ppokit.best/2` garde le SHA-256 complet et n'est
  écrit qu'après : un échec laisse le BEST précédent. `ReadBest` refuse une
  politique dont l'empreinte diffère ; les manifestes `ppokit.best/1` restent
  lus. `tools/ci/mappo_curriculum.py` vérifie l'empreinte avant de promouvoir
  PROD. Tests : `PPOKitBestStoreTests` (vecteurs FIPS 180-4, deux entraînements
  au même update, corruption, absence, copie interrompue, collision de nom,
  ancien manifeste) et `test_mappo_curriculum.py`.
- Audit P0-03 / P1-02 (fresh et état d'entraînement de l'environnement) :
  l'environnement FlightControl3D relisait `ppokit_curriculum_level.txt`
  dans son constructeur, avant que ppokit-run ne choisisse entre reprise et
  `--fresh`, et le réécrivait à chaque promotion, indépendamment des poids.
  API d'environnement 5 : `training_state` / `restore_training_state` (SDK :
  `TrainingState()` / `RestoreTrainingState()`, facultatifs ; l'état vide
  est celui d'un run qui commence). ppokit-run écrit l'état de l'update dans
  `checkpoints/environment_state/<update>.json`, puis le checkpoint, puis
  retire les états des autres updates : interrompu n'importe où, le
  checkpoint retrouve l'état de son update. La reprise restaure cet état et
  refuse un état d'un autre run ; un run sans reprise (`--fresh`, ou sans
  checkpoint) archive le run précédent (checkpoint, politique, journal,
  états, BEST des suites) dans `checkpoints/archive/run-<date>/` et donne
  l'état vide à l'environnement. Les campagnes pendant l'entraînement sont
  jouées à l'état de l'environnement d'entraînement, `eval` et `play` à
  l'état de la dernière sauvegarde. FlightControl3D : l'état contient le
  niveau, la fenêtre de promotion et le flux d'entraînement (moteur
  aléatoire, prochain côté d'approche) ; le fichier de niveau n'est plus
  écrit, seulement lu pour un checkpoint sans état. Tests : `PPOKitRunTests`
  (état sauvegardé et repris, état en avance après interruption, fresh qui
  archive, BEST du nouveau run, états d'un autre run refusés, checkpoint
  sans état) et `FlightControl_visualiser_tests` (fresh au niveau 0 malgré
  un niveau avancé, reprise du même départ suivant, états invalides
  refusés).
- Audit P1-06 (déclencheurs de la CI) : `flight-control.yml` ne surveillait
  ni `PPOKit/**` ni `Framework/**`, que son build compile ; `desktop-windows.yml`
  ignorait les en-têtes C (`environment_abi.h`), les `*.cmake` et les JSON des
  projets (qui génèrent des layouts et figent des scores) ; le lint de
  `mappo-restart-progressive.yml` ne suivait pas `tools/ci/**` ni
  `mappo-training-windows.yml`, qu'il vérifie. Les filtres suivent ce que
  chaque job construit ; `tools/ci/test_workflow_triggers.py` (dans Core) le
  vérifie. Restent sans déclencheur de PR, par conception, les entraînements
  (`flight-control-*-training`, `mappo-*-windows` : étiquette ou dispatch) et
  l'inférence Blue (dispatch ; sa PR ne teste que ses scripts).
- Audit P0-01 (RLMission sans piste) : le réseau LibTorch encodait toutes les
  pistes et masquait par `masked_fill` −1e9 (max et scores d'attention) ;
  sans piste présente, ses moyennes valent ~1e8 (le tour et l'accélération
  saturent, pas de tir). Le graphe PPOKit renvoyait des zéros : les rouges
  RLMission (y compris ceux de BlueDefense) changeaient de comportement sans
  piste. Blocs : `encode_absent` (entity_encoder), `mask_fill` (masked_pool
  max, masked_attention : score + fill, softmax sur toutes les entités, en
  float comme LibTorch), dans l'identité du réseau. `graph.json` et
  `network.json` de RLMission les activent (acteur et critique) ; les trois
  artefacts RLMission sont re-tamponnés (`airspace_checkpoint_import
  --restamp`, tenseurs identiques). `AirspaceConvertedPolicyTests` compare
  désormais tous les cas de référence, y compris sans piste ; seul le cas à
  présence fractionnaire reste hors domaine (l'environnement écrit 0 ou 1).
  Tests : `PPOKitGraphBlocksTests` (gradients, ensemble vide = softmax
  LibTorch en float), `PPOKitBlockGraphSchemaTests` (lecture, identité,
  refus). Les scores d'évaluation figés ne bougent pas.
- Audit P1-03 (entraînement RLSNA) : sur master, RLSNA s'entraînait dans le
  monde (SNATrainingAgent, hyperparamètres de `settings/Training/RLSNA.json`)
  sur un scénario écrit par l'utilisateur ; aucun scénario d'entraînement
  n'était versionné, et `rlsna_environment` refusait l'entraînement.
  Restauré (décision de l'utilisateur) : `Scenarios/RLSNATraining.json`
  (l'intercepteur en mode `training` contre sa cible, plates-formes de la
  campagne d'évaluation), `projects/RLSNA` entraîne (`settings`, `train`,
  politique `eval/policy.ppokit` comme RLMission, journal), `training.json`
  reprend les hyperparamètres PPO de master (rollout 800, gamma 0,99,
  lambda 0,95, clip 0,2, epochs 4, minibatch 64, target KL 0,03, entropie
  0,02) et la suite joue toutes les 5 mises à jour (le `checkpointEvery`
  historique). Les deux environnements Airspace partagent
  `Environment/training_scenario.hpp` (même entraînement RLMission, sorties
  de `ppokit-run train` identiques avant et après). Tests : le smoke test
  reprend l'entraînement RLSNA (update 888 -> 889),
  `AirspaceTrainingRewardTests_RLSNA`.
- Audit P0-04 (non-régression FlightControl3D) : le T-6 entraîné de master
  (`t6texan2_joint_v5`, update 340, son BEST : 19 arrêts sur 20) est gardé en
  fixture. `FlightControl_master_checkpoint_parity_tests` : FlightLearner
  charge le checkpoint de master, le graphe PPOKit de `projects/T6` son
  `latest.ppokit` ; moyennes, log-std, logit du train et valeur du critique
  identiques bit à bit sur 64 observations et sur chacun des 43 730 pas des 20
  épisodes de la campagne de master (départs `EvaluationStart`, niveau 11),
  volés avec les actions de master : 19 arrêts, 19 toucher, sommes des
  distances égales au BEST de master à 0,25 mm près (enregistré sous Windows).
  `FlightControl_t6_evaluation_smoke` : `ppokit-run eval` sur les DLL donne
  la même campagne (0,95, 1349,59 m). Le PC-7 n'a pas de checkpoint sur
  master : rien à comparer. Les moments Adam et l'état RNG du T-6 ne sont
  pas encore comparés tenseur par tenseur.
- Audit P1-04 (T-6 / PC-7) : leurs `task.json` et réseaux sont identiques,
  donc leurs checkpoints aussi pour l'identité : un échange passait sans
  erreur. `task.json` peut nommer son environnement (`"contract"`,
  `t6texan2@8`, `pc7@8` : avion et version du contrat d'observation),
  hors `ShapeHash` ; il entre dans l'identité des artefacts. Compatibilité
  structurelle (tâche, formes, réseaux) et sémantique (contrat) : un
  checkpoint d'un autre contrat est refusé par train, eval et play ; un
  checkpoint sans contrat (antérieur) reste accepté. `ppokit-run transfer
  --project DIR --from FILE` rend un transfert explicite (copie sous le
  contrat du projet, tenseurs inchangés, rechargée, jamais par-dessus un
  fichier). L'environnement FlightControl3D refuse un `task.json` dont le
  contrat n'est pas celui de son scénario. Les checkpoints T-6 versionnés
  portent leur contrat (tenseurs identiques). Tests : `PPOKitRunTests`,
  `PPOKitRepositoryProjectsTests`, `FlightControl_t6_evaluation_smoke`
  (le PC-7 refuse la politique T-6).
