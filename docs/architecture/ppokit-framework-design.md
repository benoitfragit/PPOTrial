# PPOKit — framework commun PPO/MAPPO, codegen de réseaux et studio

> Statut : **document de conception**, aucune modification de code.
> Nom de travail : `PPOKit` (bibliothèque `ppokit`, namespace `ppokit`).
> Le nom est provisoire.

## 0. Portée et garde-fous

**Dans la portée :**
- BlueDefense, c'est-à-dire le MAPPO Blue généralisé (`Examples/BlueDefense/Agents/BluePatrol`).
- FlightControl3D (`Examples/FlightControl3D`).

**Hors de la portée :**
- Airspace et la chaîne libtorch existante (`Framework/RL/PPO`, `Framework/RL/Bindings`).
  Ils restent inchangés. Leur raccordement éventuel, via un backend libtorch
  derrière les mêmes interfaces, est une question ouverte (§17).
- Le contrôleur rouge `RLMission` n'est pas modifié (AGENTS.md).

**Invariants à respecter pendant toute la migration :**
- Aucun canal de récompense Blue n'est ajouté, renommé, repondéré ou renormalisé.
  Les 15 canaux de `GeneralizedRewardChannel` et les 10 canaux de
  `flight::Reward` sont transcrits à l'identique.
- Les capacités 1..9 Bleus / 0..10 Rouges, le masquage et les permutations sont
  préservés.
- Physique, politiques, calcul des récompenses et adaptation ECS restent
  séparés. PPOKit ne connaît ni EnTT ni JSBSim.
- Aucune allocation dans les boucles par pas : acteur, critique, observation,
  récompense.
- Chaque étape de migration est validée par une trace bit-identique, ou par
  une différence justifiée et approuvée explicitement. Les tests unitaires ne
  prouvent pas la convergence : l'évaluation multi-seed indépendante reste
  obligatoire.

## 1. Objectifs

1. Une seule implémentation, testée une fois, de ce qui est aujourd'hui
   dupliqué : rollout, GAE (terminé ≠ tronqué), surrogate PPO clippé, entropie,
   optimisation, MAPPO (critique centralisé, masques), checkpoints atomiques,
   état RNG.
2. Un moteur générique d'observations (schéma, normalisation, masques) et de
   récompenses (canaux nommés, poids, agrégation). Les **termes** restent chez
   le client.
3. Des têtes de sortie composables : catégorielle, Bernoulli, gaussienne
   (bornée par clip ou tanh), et composite.
4. Un graphe de réseau décrit en JSON, **compilé par génération de code** vers
   rl_tools. rl_tools n'est jamais exposé au client.
5. Un cycle outillé **design → codegen → entraînement → évaluation →
   inférence**, avec identité vérifiable des artefacts.
6. Des composants DLL boîte noire, déclarés par leurs ports et capacités et
   câblés librement dans le graphe, via une ABI C et un SDK C++.
7. Un studio ImGui + imnodes + ImPlot, construit **en dernier**, sur des
   formats déjà stabilisés.

**Non-objectifs :**
- Un moteur de simulation commun.
- Une autodiff générale.
- L'exécution dynamique de graphes à dimensions variables à l'exécution.

## 2. Vue d'ensemble

```
graph.json ─┐
task.json  ─┼─► ppokit-codegen ─► <model>_generated.hpp/.cpp ─┐
components/*.manifest.json ─┘                                     │
                                                               ▼
client C++ (Task / HostedAgent, hooks) ──► ppokit runtime ◄── backend rl_tools (privé)
                                             │
              ┌──────────────────────────────┼──────────────────────────┐
              ▼                              ▼                          ▼
        Trainer (owned/hosted)        Evaluator (suites)          Inference (forward seul)
              │                              │                          │
              ▼                              ▼                          ▼
     artefacts training/policy        rapports JSON, rejeu      artefact promoted
              └──────────── télémétrie / commandes ──► Studio (GUI) ou CLI headless
```

Sens des dépendances : `client → ppokit (API publique) → ppokit/backend
(privé)`. L'API publique n'inclut aucun en-tête rl_tools, JSBSim, EnTT, raylib
ni ImGui.

## 3. Modules

| Module | Contenu | Dépend de |
|---|---|---|
| `ppokit/core` | Tampons de rollout à capacité fixe, GAE templaté sur le scalaire, surrogate PPO, entropie, statistiques d'update, ordonnancement des minibatches, RNG sérialisable | STL |
| `ppokit/schema` | Lecture et validation de `task.json`, layouts d'observation, d'action et de scénario, hash canonique | core, nlohmann_json |
| `ppokit/obs` | Normaliseurs (fixe, running mean-std, clip, custom), application des masques, vérification des valeurs finies | schema |
| `ppokit/reward` | Canaux nommés, poids, agrégation par agent et par équipe, snapshots par canal | schema |
| `ppokit/heads` | Têtes catégorielle, Bernoulli, gaussienne, composite : échantillonnage, log-prob, entropie, gradient, action déterministe | core |
| `ppokit/blocks` | Dense, activations, Concat/Split, encodeur par entité, attention masquée, pooling masqué | backend |
| `ppokit/backend` | **Privé.** Instanciation rl_tools, Adam, buffers, une seule cible CMake (supprime `RL_TOOLS_NAMESPACE_WRAPPER`) | rl_tools |
| `ppokit/runtime` | Trainer owned/hosted, MAPPO, curriculum, évaluation périodique et promotion, budgets, télémétrie et commandes | tous les précédents |
| `ppokit/eval` | Suites, grilles, gates, rapports, rejeu | runtime |
| `ppokit/artifacts` | Formats de checkpoint, en-têtes d'identité, écriture atomique, migrations | core |
| `ppokit/component` | Chargement des DLL, ABI C des composants boîte noire, SDK C++ | core |
| `ppokit-codegen` | Exécutable JSON → C++ | schema |
| `ppokit-studio` | Éditeur imnodes, suivi des runs, rapports | runtime, eval, imgui, imnodes, implot |

## 4. Contrat client

Le client fournit deux choses :
- des **données** (JSON) : formes, bornes, normalisations, scénarios,
  curricula, suites d'évaluation, graphe ;
- du **code** (C++) : sémantique, physique, valeur des features, termes de
  récompense, métriques.

Règle de séparation : **tout ce qui change une dimension ou une borne passe
par le JSON**, donc par le codegen. Tout ce qui porte du sens reste en C++.

### 4.1 Fichiers

| Fichier | Rôle | Versionné dans git |
|---|---|---|
| `task.json` | Observations, actions, rôles, entrée du critique, canaux de récompense, espace de scénarios, métriques | oui |
| `graph.json` | Réseaux acteur et critique, blocs, têtes | oui |
| `train.json` | Hyperparamètres PPO, curriculum, budgets, évaluation périodique, promotion | oui |
| `eval/*.json` | Suites d'évaluation | oui |
| `*_generated.*` | Sortie du codegen | non (dossier de build) |

### 4.2 Types générés

À partir de `task.json`, le codegen produit, pour une tâche `Flight` :
- `FlightObs` et `FlightCriticInput`, avec des champs nommés, plus des vues
  sur les ensembles masqués ;
- `FlightAction`, décodée et bornée ;
- `FlightRewards`, un champ par canal ;
- `FlightScenario` et `FlightMetrics` ;
- `FlightSpec`, qui regroupe les constantes de dimension et le hash du schéma.

Le client ne manipule ni indice ni tenseur.

### 4.3 Mode Owned : le framework mène la boucle

Cas de FlightControl3D.

```cpp
class FlightTask final : public ppokit::Task<FlightSpec>
{
public:
    void Reset(FlightScenario const& scenario, ppokit::Rng& rng, FlightObs& obs) override;
    ppokit::StepEnd Step(FlightAction const& action, FlightObs& obs, FlightRewards& rewards) override;
    void EpisodeMetrics(FlightMetrics& metrics) const override;
    // Optionnels :
    void StepMetrics(FlightMetrics& metrics) const override;
    void AdaptAction(FlightAction& action) const noexcept override;
};
PPOKIT_REGISTER_TASK("flight", FlightTask);
```

`StepEnd` vaut `{ Running, Terminated, Truncated }`. GAE annule le bootstrap
sur `Terminated` et coupe la trace sur `Terminated` et `Truncated`, ce qui
reproduit le comportement actuel de `ComputeAdvantages`.

### 4.4 Mode Hosted : l'hôte mène la boucle

Cas de BlueDefense, où l'ECS de `Framework/Simulation` appelle `ChooseAction`
et `SetNewState`.

```cpp
template<class Spec>
class HostedDriver
{
public:
    void BeginEpisode(typename Spec::Scenario const& scenario, std::uint64_t seed);
    // Remplit une action par slot vivant ; les slots masqués sont ignorés.
    void Act(ppokit::AgentSlots<typename Spec::Obs const> observations,
             typename Spec::CriticInputs const& critic,
             ppokit::AgentSlots<typename Spec::Action> actions);
    // Une fin par agent (perte individuelle) et une fin d'épisode.
    void Observe(ppokit::AgentSlots<typename Spec::Rewards const> rewards,
                 ppokit::AgentEnds const& agentEnds, ppokit::StepEnd episodeEnd);
    ppokit::Telemetry Snapshot() const noexcept;
    void Command(ppokit::Command const& command) noexcept;
};
```

`GeneralizedPatrolMappoAgent` devient un adaptateur ECS mince, qui garde
`ppo::model::IAgent` vers l'hôte et délègue au `HostedDriver`. Ce que
l'adaptateur conserve :
- l'ordre des slots physiques ;
- la succession du leader ;
- la correspondance rôle → index physique ;
- les snapshots avant et après le pas.

Le framework exige seulement que les slots soient stables pendant un épisode.

### 4.5 Hooks

| Hook | Usage | Exemple existant |
|---|---|---|
| `AdaptAction` | Règle métier après décodage et bornage | Consigne de throttle de référence, `ApplyPhaseControls` (Flight) |
| Fournisseur de références | Calcul non appris injecté dans l'observation | `GeneralizedPatrolWaypointSequencer`, `guidancePose` (Blue) ; `Guide` (Flight) |
| Normaliseur custom | `"norm": "custom:<nom>"` | Enroulement de cap |
| `StepMetrics` | Métrique continue | RMS d'erreur de glide |

Les hooks sont des fonctions nommées enregistrées, sans lambda capturante
(AGENTS.md).

## 5. Schémas JSON

Chaque fichier porte `"schema": "ppokit.<type>/<major>.<minor>"`. Un champ
inconnu est une erreur, pas un avertissement.

### 5.1 `task.json`

```json
{
  "schema": "ppokit.task/1.0",
  "name": "flight",
  "agents": { "capacity": [1, 1], "roles": [] },
  "observation": {
    "groups": {
      "own": { "fields": {
        "airspeed": { "norm": { "scale": 0.01 }, "clip": [-5, 5] },
        "glideErr": { "norm": "running" } } },
      "tracks": { "set": true, "capacity": 10, "masked": true,
                  "fields": { "range": { "norm": { "scale": 1e-4 } } } }
    }
  },
  "critic": { "source": "actor" },
  "action": {
    "controls": { "head": "gaussian", "size": 4, "bounds": [-1, 1], "map": "clip",
                  "logStd": { "init": -0.5, "learned": true } },
    "gear":     { "head": "bernoulli" }
  },
  "rewards": { "path": {}, "localizer": {}, "crash": {} },
  "scenario": {
    "windKt":    { "type": "float", "range": [0, 25] },
    "placement": { "type": "enum", "values": ["aligned", "loc", "glide", "random"] }
  },
  "metrics": { "landed": { "type": "bool" }, "glideRms": { "type": "float" } }
}
```

Points clés :
- **Groupes** : l'ordre de déclaration fixe l'ordre des offsets. Un groupe
  `set` produit `capacity × fields` valeurs et un masque.
- **Rôles** : `"roles": ["leader", "wingman"]` génère une entrée one-hot ou
  scalaire, au choix (`"roleEncoding"`). La valeur est fournie par le client.
- **`critic`** : `"source": "actor"` réutilise l'observation de l'acteur.
  `"source": "custom"` déclare des groupes propres, avec `"perAgent": true`
  pour une entrée conditionnée par participant (critique Blue de 470 entrées).
- **`rewards`** : les poids et normalisations par défaut sont neutres (poids 1,
  sans normalisation). Les valeurs effectives, y compris la normalisation P/N
  de Blue, sont transcrites telles qu'elles existent aujourd'hui, sans
  modification.
- **Versionnement** : le hash canonique de `observation` remplace les
  constantes manuelles comme `GeneralizedObservationLayout::Version = 10`. On
  peut garder un alias lisible (`"version": 10`) pour la compatibilité.

### 5.2 `graph.json`

```json
{
  "schema": "ppokit.graph/1.0",
  "task": "flight",
  "scalar": "float",
  "actor": {
    "nodes": [
      { "id": "in",  "type": "input", "from": "observation" },
      { "id": "mlp", "type": "dense_stack", "widths": [128, 128, 128], "activation": "tanh" },
      { "id": "out", "type": "heads", "from": "action" }
    ],
    "edges": [["in", "mlp"], ["mlp", "out"]]
  },
  "critic": {
    "nodes": [
      { "id": "in", "type": "input", "from": "critic" },
      { "id": "v",  "type": "dense_stack", "widths": [256, 256], "activation": "tanh" },
      { "id": "value", "type": "value" }
    ],
    "edges": [["in", "v"], ["v", "value"]]
  },
  "init": { "seed": "train", "scheme": "rl_tools_default" }
}
```

- Types de nœuds de la v1 : `input`, `dense`, `dense_stack`, `activation`,
  `concat`, `split`, `entity_encoder`, `masked_attention`, `masked_pool`,
  `heads`, `value`, `component:<nom>@<version>` (§12).
- `input.from` peut viser un groupe (`"observation.tracks"`) pour router les
  ensembles vers un encodeur par entité.
- Les arêtes relient des nœuds ou des ports nommés (`"filter.out:features"`).
- Le codegen vérifie : formes, acyclicité, présence et unicité de
  `heads`/`value`, cohérence avec `task.json` et les manifestes des
  composants, masques propagés jusqu'aux blocs qui les consomment, et règles
  de câblage de §12.4.

### 5.3 `train.json`

```json
{
  "schema": "ppokit.train/1.0",
  "ppo": { "gamma": 0.99, "lambda": 0.95, "clip": 0.2, "epochs": 10, "minibatch": 256,
           "rolloutSteps": 4096, "learningRate": 3e-4, "entropy": 0.0, "valueCoef": 0.5,
           "maxGradNorm": 0.5 },
  "seed": 1,
  "budget": { "updates": 2000, "mode": "total" },
  "curriculum": [
    { "name": "glide",
      "sample": { "placement": "glide", "windKt": { "uniform": [0, 5] } },
      "promote": { "metric": "landed", "gate": ">= 0.8", "window": 200 } },
    { "name": "joint",
      "sample": { "placement": { "choice": { "aligned": 0.6, "random": 0.4 } },
                  "windKt": { "uniform": [0, 25] } },
      "rehearse": { "loc": 0.1 } }
  ],
  "periodicEvaluation": { "every": 50, "suite": "eval/smoke.json",
                          "promoteOn": { "metric": "landed", "rule": "max" } },
  "checkpoint": { "every": 10, "keep": 5 }
}
```

- Échantillonneurs : `fixed`, `uniform`, `normal` (bornée par le schéma),
  `choice` pondéré.
- `rehearse` généralise `locReminderProbability` (Flight).
- `budget.mode` vaut `"total"` ou `"additional"`, comme `TrainingBudget`.

### 5.4 `eval/*.json`

```json
{
  "schema": "ppokit.eval/1.0",
  "suite": "landing-regression",
  "grid": { "windKt": [0, 10, 20], "placement": ["aligned", "loc"] },
  "cases": [ { "name": "crosswind-max", "windKt": 25 } ],
  "seeds": [101, 202, 303],
  "episodes": 20,
  "policy": "deterministic",
  "gates": { "landed": ">= 0.9", "crash": "<= 0.02" }
}
```

## 6. Têtes de sortie

Chaque tête implémente : `Sample(rng)`, `LogProb(action)`, `Entropy()`,
`Deterministic()`, `Backward(dLogProb, dEntropy)`.

| Tête | Paramètres réseau | Action déterministe | Notes |
|---|---|---|---|
| `categorical` | logits `K` | argmax | log-softmax stable, masque d'actions optionnel (logits masqués à −∞) |
| `bernoulli` | logit | p > 0,5 | |
| `gaussian` | moyenne `n`, log-std (paramètre libre ou sortie réseau) | moyenne | log-prob et entropie calculées sur le **latent** ; `map: clip` ou `map: tanh` appliqué après |
| `composite` | concaténation | composante par composante | log-prob = somme, entropie = somme (têtes indépendantes) |

**Compatibilité Flight :** `hybrid_policy.hpp` calcule déjà la densité du
latent gaussien × Bernoulli et un gradient exact du surrogate. La tête
composite `gaussian(map: clip) + bernoulli` doit reproduire ces valeurs à la
précision près, avec un test de parité.

**`map: tanh` :** il faut choisir explicitement entre deux variantes :
- log-prob sur le latent (cohérent avec le clip actuel) ;
- log-prob corrigée du jacobien.

Le défaut de la v1 est la log-prob sur le latent, pour aligner les deux modes.

## 7. Génération de code

- **Entrées** : `graph.json`, `task.json`, manifestes des composants.
- **Sorties** :
  - `<model>_generated.hpp` : types de §4.2 et `Spec` ;
  - `<model>_generated.cpp` : instanciation rl_tools, forward, backward,
    sérialisation des paramètres.
- **Déterminisme** : à entrée identique, sortie identique octet pour octet.
  Le fichier généré porte en en-tête le hash des entrées et la version du
  générateur.
- **CMake** : `ppokit_add_model(<target> GRAPH graph.json TASK task.json)`
  ajoute une commande personnalisée et une bibliothèque statique. Tout
  changement de JSON déclenche une régénération. Les répertoires
  `build/debug` et `build/release` restent les seuls utilisés (AGENTS.md).
- **Composants** : un nœud `component:` génère un appel à la table d'appels
  de la DLL, avec les formes résolues au câblage. Le composant n'est pas
  inliné.
- **Test d'acceptation n°1** : régénérer les réseaux actuels (MLP Flight,
  acteur à attention et critique Blue) depuis JSON, avec une sortie forward
  identique et des gradients identiques sur des entrées fixées.

## 8. Artefacts et identité

| Type | Contenu | Usage |
|---|---|---|
| `training` | Paramètres acteur et critique, état Adam, RNG (`mt19937` portable), statistiques des normaliseurs, étape de curriculum, compteurs d'updates, budget | Reprise exacte |
| `policy` | Paramètres de l'acteur, statistiques des normaliseurs | Évaluation, inférence |
| `promoted` | `policy` avec le score et le rapport d'évaluation qui l'ont promu | Production |

**En-tête commun :**
- magic et version du format ;
- `taskHash`, `graphHash`, `generatorVersion`, `scalar` ;
- hash du binaire, du manifeste et de la configuration de chaque composant ;
- horodatage et seed d'entraînement.

**Règles :**
- Au chargement, tout hash incompatible est **refusé**, sauf migration
  déclarée explicitement.
- L'écriture est atomique : fichier temporaire, flush, puis renommage. Une
  seule implémentation remplace les quatre actuelles.
- Les sauvegardes se font uniquement à une frontière d'update, comme
  aujourd'hui dans `FlightLearner::Save` et `SaveAtUpdateBoundary`.

**Migration des formats existants** (Blue V7 / obs v10, checkpoints Flight) :
un convertisseur ponctuel lit l'ancien format, vérifie la correspondance des
formes et écrit le nouveau. Les anciens lecteurs restent disponibles jusqu'à
la fin de la migration.

## 9. Entraînement

- **Runner pas à pas** : `Tick(maxSteps)` avance d'au plus N pas, sans jamais
  dépasser une frontière d'update. Il sert aussi bien au desktop
  (`FlightSession::Tick`, `ContinueTraining`) qu'au headless
  (`flight_headless`).
- **MAPPO** :
  - acteur partagé ;
  - critique centralisé conditionné par participant ;
  - fins d'épisode par agent (`OwnLoss`) ;
  - masques de slots vivants ;
  - permutation des pistes, vérifiée par un test d'invariance.
- **Évaluation périodique et promotion** : le runner suspend la collecte,
  exécute la suite configurée sur des seeds dédiés, puis promeut le modèle si
  la règle est satisfaite. Il reprend ensuite en restaurant exactement l'état
  du générateur d'épisodes, comme `RestoreTrainingInitializer` aujourd'hui.
  Ce service remplace `StartEvaluation` et `PromoteEvaluatedActor` (Blue),
  ainsi que `StartValidation` et `FlightModelStore::LoadBest` (Flight).
- **Télémétrie** : un snapshot publié sous verrou ou en double tampon, qui
  contient :
  - les statistiques d'update (KL, clip fraction, entropie, perte de valeur) ;
  - les récompenses par canal ;
  - l'étape de curriculum ;
  - les métriques d'évaluation.
- **Commandes** : pause, reprise, sauvegarde, évaluation immédiate,
  changement de mode (entraînement ou inférence).
- **Garde d'échec** : toute valeur non finie fait échouer le learner, avec
  reprise depuis un checkpoint valide, comme `FlightLearner::Impl::Check`.

## 10. Évaluation

- Seeds **disjoints** de ceux de l'entraînement, vérifiés au chargement.
- Mode `deterministic` ou `stochastic`.
- Agrégations : `rate`, `mean`, `p50`, `p95`, `max`.
- Gates exprimés par des comparaisons simples.
- Rapport JSON par cellule de grille et par cas, avec intervalles de
  confiance binomiaux pour les taux.
- **Rejeu** : chaque épisode est identifié par (`scenario`, `seed`,
  `artifactHash`). Le studio peut le rejouer, et un épisode raté peut être
  exporté comme `case` de régression.
- Suites types :
  - `smoke` : CI, quelques épisodes ;
  - `regression` ;
  - `heldout` : hors de la distribution d'entraînement.
- Remplace `GeneralizedEvaluationCampaign` et `generalized_evaluation_metrics`.
  Les règles de rejet des campagnes incomplètes déjà en place sont conservées
  (`generic-rl-core.md`).

## 11. Inférence

- Bibliothèque `<model>_inference` : forward de l'acteur seul, sans Adam, sans
  rollout et sans critique.
- Action déterministe par tête (§6), puis `AdaptAction`.
- Remplace `GeneralizedPatrolProductionPolicy`.
- Test de parité : la sortie d'inférence est égale à l'action déterministe
  calculée par le learner sur les mêmes entrées.

## 12. Composants externes (DLL boîte noire)

### 12.1 Principe

L'utilisateur fournit une DLL qui déclare un ou plusieurs **composants**. Le
framework ignore ce qu'ils calculent. Il connaît seulement ce qu'il faut
pour les **câbler** :
- des ports d'entrée et de sortie typés ;
- des capacités déclarées (différentiable, entraînable, avec état…).

Le **rôle** d'un composant n'est pas déclaré par le plugin. Il découle de son
câblage dans `graph.json` :

| Câblage | Rôle effectif | Exemples |
|---|---|---|
| `observation.*` → composant → réseau | Prétraitement, extracteur de features | Modèle capteur, filtre de pistes, encodeur pré-entraîné figé |
| réseau → composant → réseau | Bloc interne | Attention masquée, couche custom |
| tête → composant → `action.*` | Post-traitement de l'action, côté environnement | Allocateur de commandes, limiteur, autopilote bas niveau |
| `observation.*` → composant → `action.*` | Politique externe complète | Modèle C/C++ existant, entraînable ou figé |
| composant → `critic.*` | Entrée privilégiée du critique | Estimateur d'état global |

La même DLL peut exposer plusieurs composants. Le studio les affiche dans la
palette d'imnodes avec leurs ports, et le codegen vérifie le câblage.

### 12.2 Descripteur de ports

Chaque port déclare :

| Champ | Valeurs | Rôle |
|---|---|---|
| `name` | identifiant | Référence dans les arêtes (`"comp.out:features"`) |
| `direction` | `in` / `out` | |
| `dtype` | `f32`, `f64`, `i32`, `bool` | `bool` pour les masques, `i32` pour les indices catégoriels |
| `shape` | `[n]`, `[capacité, n]` ou dimension symbolique (`"N"`) | Les dimensions symboliques sont résolues au câblage puis figées par le codegen |
| `mask` | nom d'un port `bool` associé | Propagation des masques d'ensembles |
| `unit` | texte libre, optionnel | Documentation et vérification facultative |
| `optional` | booléen | Le port peut rester non connecté |

### 12.3 Capacités

| Capacité | Conséquence vérifiée par le codegen et le runtime |
|---|---|
| `differentiable` | Fournit `backward` vers ses entrées. Obligatoire si un nœud entraînable se trouve **en amont**. |
| `trainable` | Expose paramètres et gradients ; l'optimiseur reste celui du framework. Implique `differentiable` vers ses paramètres. |
| `stateful` | Fournit `reset`, `stateSize`, `saveState` et `loadState`. Voir §12.5. |
| `deterministic` | Faux : le framework fournit un seed par instance et par épisode, que le composant doit utiliser. |
| `threadSafe` | Faux : une instance par worker. |
| `batch` | Accepte un lot de B échantillons par appel, sinon le framework boucle. |

### 12.4 Règles de câblage

1. **Gradient** : un composant non différentiable ne peut pas se trouver
   entre deux nœuds entraînables. En pratique il doit être :
   - soit avant tout nœud entraînable (prétraitement, gelé ou non) ;
   - soit après l'échantillonnage de l'action.

   Tout autre câblage est une erreur du codegen.
2. **Post-traitement d'action** : un composant placé après une tête transforme
   l'action **échantillonnée**. Pour PPO, il fait partie de l'environnement :
   la log-prob est calculée sur la sortie de la tête, pas sur celle du
   composant. On garde ainsi un ratio PPO correct quel que soit le contenu de
   la boîte noire. Le studio affiche cette frontière.
3. **Formes** : chaque arête relie des ports de même `dtype` et de même forme,
   après résolution des dimensions symboliques. Les masques suivent leur port.
4. **Cycles** : interdits dans le graphe d'un pas. La récurrence passe par
   l'état d'un composant `stateful`.

### 12.5 Composants avec état

- `reset` est appelé à chaque début d'épisode, par agent.
- Pour l'apprentissage, le framework **stocke dans le rollout la sortie** des
  composants avec état placés avant les nœuds entraînables. La mise à jour
  PPO rejoue alors le réseau sur ces sorties, sans réexécuter le composant.
  C'est une condition de validité du ratio PPO.
- Un composant avec état **entraînable** (récurrent) n'est pas supporté en v1,
  faute de rétropropagation dans le temps. Le codegen le refuse.
- L'état fait partie du checkpoint `training`, et l'état initial fait partie
  de l'artefact `policy`.

### 12.6 ABI C

```c
#define PPOKIT_COMPONENT_ABI 1u

typedef struct PpokitBuffer
{
    void* data;
    uint32_t dtype;
    uint32_t rank;
    const uint32_t* shape;
} PpokitBuffer;

typedef struct PpokitComponentApi
{
    uint32_t abi;
    const char* name;
    uint32_t version;
    const char* manifestJson;                 /* ports, capacités, paramètres */
    void* (*create)(const char* configJson, const char* resolvedShapesJson,
                    char* error, size_t errorSize);
    void  (*destroy)(void* instance);
    int   (*forward)(void* instance, uint32_t batch,
                     const PpokitBuffer* inputs, PpokitBuffer* outputs, void* cache);
    /* NULL sauf si differentiable */
    int   (*backward)(void* instance, uint32_t batch, const PpokitBuffer* outputGradients,
                      const void* cache, PpokitBuffer* inputGradients, float* parameterGradients);
    /* NULL sauf si trainable */
    int   (*parameters)(void* instance, float** data, size_t* count);
    /* NULL sauf si stateful */
    int   (*reset)(void* instance, uint32_t agentSlot, uint64_t seed);
    int   (*saveState)(void* instance, void* buffer, size_t size);
    int   (*loadState)(void* instance, const void* buffer, size_t size);
    size_t (*stateSize)(void* instance);
    size_t (*cacheSize)(void* instance, uint32_t batch);
    int   (*lastError)(void* instance, char* error, size_t errorSize);
} PpokitComponentApi;

/* Point d'entrée unique de la DLL : liste de ses composants. */
PPOKIT_EXPORT int ppokit_components(uint32_t requestedAbi,
                                    const PpokitComponentApi* const** components,
                                    size_t* count);
```

Règles de frontière :
- Pas de STL, d'exception ni d'allocation partagée à la frontière.
- Le framework possède tous les buffers (entrées, sorties, cache, gradients),
  ce qui garantit l'absence d'allocation dans la boucle chaude.
- Erreurs par code de retour et `lastError`. Le SDK intercepte toute exception
  C++ avant la frontière.
- Négociation : refus explicite si la version majeure de l'ABI diffère.
- Le hash du binaire, du manifeste et de la configuration est enregistré dans
  les artefacts (§8). Changer de DLL invalide donc un checkpoint, sauf
  déclaration de compatibilité.

### 12.7 Manifeste

```json
{
  "schema": "ppokit.component/1.0",
  "name": "trackfilter", "version": 1,
  "ports": [
    { "name": "tracks",   "direction": "in",  "dtype": "f32", "shape": ["N", 13], "mask": "trackMask" },
    { "name": "trackMask","direction": "in",  "dtype": "bool","shape": ["N"] },
    { "name": "features", "direction": "out", "dtype": "f32", "shape": [64] }
  ],
  "capabilities": { "differentiable": false, "trainable": false, "stateful": true,
                    "deterministic": true, "threadSafe": false, "batch": true },
  "config": { "alpha": { "type": "float", "range": [0, 1], "default": 0.3 } }
}
```

La section `config` est validée par le framework et éditable dans le studio.
La DLL reçoit la configuration déjà validée.

### 12.8 Câblage dans `graph.json`

```json
{ "id": "filter", "type": "component:trackfilter@1", "config": { "alpha": 0.2 } },
{ "id": "mlp", "type": "dense_stack", "widths": [128, 128], "activation": "tanh" }
```
```json
"edges": [
  ["observation.red", "filter.in:tracks"],
  ["observation.red.mask", "filter.in:trackMask"],
  ["filter.out:features", "concat.in:1"],
  ["concat.out", "mlp"]
]
```

La dimension symbolique `N` est résolue à 10 par le câblage sur
`observation.red`, puis figée dans le code généré.

### 12.9 SDK C++

```cpp
class TrackFilter final : public ppokit::sdk::Component
{
public:
    static ppokit::sdk::Manifest Describe();              // ports, capacités, configuration
    explicit TrackFilter(ppokit::sdk::Config const& config, ppokit::sdk::Shapes const& shapes);
    void Forward(ppokit::sdk::Batch batch, ppokit::sdk::Inputs in, ppokit::sdk::Outputs out) override;
    void Reset(std::uint32_t agentSlot, std::uint64_t seed) override;
    // Backward / Parameters / SaveState… seulement si déclarés dans Describe().
};
PPOKIT_COMPONENTS(TrackFilter /*, AutreComposant */);
```

Le SDK est header-only et ne dépend pas de rl_tools. Il génère le manifeste à
partir de `Describe()`, ce qui garantit que le manifeste et le code restent
cohérents. Il vérifie aussi, en Debug, que les capacités déclarées
correspondent aux méthodes surchargées.

Le même mécanisme sert aux tâches externes : une DLL peut fournir un
simulateur conforme à l'interface `Task` (table `reset`, `step`, `metrics`
sur des buffers plats décrits par son `task.json`). Normaliseurs et métriques
custom sont simplement des composants câblés sur `observation.*` ou sur
`metrics.*`.

## 13. Studio

Il est développé **en dernier** (§16, étape 8).

- **Éditeur de graphe** (imnodes) :
  - palette des blocs intégrés et des composants découverts dans les DLL,
    avec leurs ports ;
  - vérification des formes en direct, avec la même bibliothèque que le
    codegen ;
  - export vers `graph.json`.
- **Bouton Build** : lance la configuration et le build CMake du modèle dans
  `build/<config>` et affiche le journal.
- **Runs** :
  - lancement en headless (processus séparé) ou en mode embarqué (desktop
    existant) ;
  - courbes ImPlot alimentées par la télémétrie ;
  - comparaison de runs.
- **Évaluation** : rapports par cellule de grille, rejeu d'épisodes, export
  de cas de régression.

Le studio n'entre jamais dans la boucle d'entraînement : c'est un client de
la télémétrie et des commandes.

## 14. Matrice de couverture

### 14.1 BlueDefense (MAPPO Blue généralisé)

| Service actuel | Fichier(s) | PPOKit | Reste client |
|---|---|---|---|
| Boucle ECS `IAgent` (`ChooseAction`, `SetNewState`) | `generalized_patrol_team_agent.*` | `runtime` mode Hosted (§4.4) | Adaptateur ECS mince |
| Modes Training / ProductionInference | `GeneralizedPatrolExecutionMode` | `runtime` + `inference` | Choix du mode par l'hôte |
| Commandes temps réel, snapshot | `ProcessCommand`, `GetSnapshot` | `runtime` télémétrie et commandes | Champs métier du snapshot |
| Observation v10 (own, lidar 61, rôle 2, Rouges 10×13, Bleus 8×9, objectif, patrouille, global, effectifs 2) | `generalized_observation.hpp`, `generalized_runtime_state.hpp` | `schema` + `obs` (groupes, ensembles masqués, rôles) | Calcul des features depuis l'ECS |
| Entrée critique centralisée par participant (470) | `generalized_critic_observation.hpp`, `generalized_critic_network.hpp` | `critic.source: custom`, `perAgent` | Remplissage |
| Rôles leader/équipiers, succession | `blue_team_roles.hpp`, `generalized_leader_succession.hpp` | Champ de rôle du schéma, slots stables | Affectation et succession |
| Capacités P 1..9, N 0..10, masques, permutations | `team_capacity.hpp` | `agents.capacity`, masques, test d'invariance | — |
| Acteur à attention masquée, backward manuel | `generalized_attention_actor.hpp`, `attention_actor.hpp` | Bloc `masked_attention` (ou composant interne pendant la transition) | — |
| Tête gaussienne (action 2) | `generalized_patrol_learner.hpp` | `heads.gaussian` | — |
| 15 canaux de récompense, normalisation P/N | `generalized_cooperative_reward.hpp` | `reward` (canaux, poids, agrégation) | Termes et normalisation P/N, transcrits à l'identique |
| Façonnage dense (patrouille, formation, proximité, frontière, basse vitesse) | `generalized_reward_shaping.hpp` | — | Entièrement client |
| Séquenceur RALLY/FOLLOW, `guidancePose` | `generalized_patrol_waypoint_sequencer.hpp` | Hook fournisseur de références | Entièrement client |
| Géométrie de l'hippodrome | `patrol_geometry.hpp` | — | Client |
| Fin d'épisode (succès, échec, timeout) et pertes individuelles | `generalized_episode_outcome.hpp` | `StepEnd` + `AgentEnds` | Décision de fin |
| GAE MAPPO (float) | `generalized_advantage.hpp` | `core` GAE `<float>` | — |
| Collecte, PPO, minibatches | `generalized_patrol_learner.hpp` | `core` + `runtime` | — |
| Configuration RLtools, hyperparamètres | `patrol_rltools.hpp`, `patrol_training_settings.hpp`, `patrol_config.hpp` | `backend` + `train.json` | — |
| Checkpoints acteur, training V7, production | `generalized_checkpoint.hpp`, `generalized_training_checkpoint.hpp`, `patrol_*checkpoint.hpp` | `artifacts` (training, policy, promoted) + convertisseur | — |
| Évaluation, promotion, score promu | `StartEvaluation`, `PromoteEvaluatedActor`, `SavePromotedScore` | `runtime` évaluation périodique | Métriques métier |
| Campagnes d'évaluation, métriques | `generalized_evaluation_campaign.hpp`, `generalized_evaluation_metrics.hpp` | `eval` | Déclaration des métriques |
| Restauration de l'initialiseur après évaluation | `RestoreTrainingInitializer` | `runtime` (restauration exacte du générateur) | — |
| Spawn, contraintes de placement | `spawn.hpp`, `spawn_config.hpp`, `initializer.hpp` | `scenario` (§5.1) + curriculum | Placement physique dans `Reset` |
| Règles de monde (collisions, zones, menaces) | `bluedefenserules.hpp` | — | Client (ECS) |
| Politique de production déterministe | `generalized_patrol_production_policy.hpp` | `inference` | — |

### 14.2 FlightControl3D

| Service actuel | Fichier(s) | PPOKit | Reste client |
|---|---|---|---|
| Environnement `Reset` / `Step` | `flight_environment.*` | `runtime` mode Owned (§4.3) | Corps de `Reset` / `Step` |
| JSBSim, mouvement de piste | `jsbsim_aircraft.*`, `runway_motion.*` | — | Client |
| Mission, guidage (LOC, glide, flare, rollout) | `flight_mission.*`, `flight_types.hpp` | Hook fournisseur de références | Client |
| Observation (56) | `flight_types.hpp` | `schema` + `obs` | Calcul des features |
| Tête hybride gaussienne × Bernoulli, clip, entropie analytique | `hybrid_policy.hpp` | `heads.composite(gaussian clip, bernoulli)` + test de parité | — |
| Contrôles forcés par phase, throttle de référence | `ApplyPhaseControls`, `hybrid_policy.hpp` | Hook `AdaptAction` | Règles |
| 10 canaux de récompense, poids | `flight_types.hpp`, `flight_config.*` | `reward` | Termes |
| Curriculum Loc/Glide/Joint, reminder LOC, placements | `flight_config.*`, `flight_environment.cpp` | `train.json` curriculum + `rehearse` | Placement dans `Reset` |
| GAE (double), PPO, MLP rl_tools, Adam | `flight_learner.*` | `core` GAE `<double>` + `backend` | — |
| Checkpoint atomique, RNG portable | `flight_learner.cpp` | `artifacts` | — |
| Validation, meilleur modèle | `FlightSession::StartValidation`, `flight_model_store.*` | `runtime` évaluation périodique + `promoted` | Métriques (`ValidationMetrics`) |
| Session pas à pas, budgets, snapshot | `flight_session.*` | Runner `Tick`, budget, télémétrie | Champs métier du snapshot |
| Headless | `flight_headless.cpp` | CLI `ppokit-train` ou main client minimal | — |
| Rendu et instruments | `Renderer/*` | — | Client (lit la télémétrie) |
| Golden trace | `tests/golden_trace_tests.cpp` | Inchangé : critère d'acceptation de chaque étape | — |

**Conclusion :** aucun service des deux exemples ne reste sans emplacement.
Ce qui reste chez le client est soit de la physique, soit de la sémantique
métier (termes de récompense, guidage, placement, rôles), conformément aux
garde-fous de §0.

## 15. Transcription (esquisses)

### 15.1 Blue (`task.json`, extrait)

```json
{
  "name": "blue-patrol",
  "agents": { "capacity": [1, 9], "roles": ["leader", "wingman"], "roleEncoding": "onehot" },
  "observation": { "groups": {
    "own": { "size": 7 }, "lidar": { "size": 61 }, "role": { "builtin": "role" },
    "red":  { "set": true, "capacity": 10, "fields": 13, "masked": true },
    "blue": { "set": true, "capacity": 8,  "fields": 9,  "masked": true },
    "objective": { "set": true, "capacity": 1, "fields": 9, "masked": true },
    "patrol": { "size": "…" }, "global": { "size": "…" }, "teamCounts": { "size": 2 } } },
  "critic": { "source": "custom", "perAgent": true, "groups": { "…": "470 entrées" } },
  "action": { "steer": { "head": "gaussian", "size": 2, "bounds": [-1, 1], "map": "clip" } },
  "rewards": { "patrolTracking": {}, "formationKeeping": {}, "pursuitProgress": {},
               "controlEffort": {}, "blueSurvival": {}, "redNeutralization": {},
               "threatContainment": {}, "objectiveSurvival": {}, "missionOutcome": {},
               "timeout": {}, "blueSeparation": {}, "restrictedZoneClearance": {},
               "boundaryClearance": {}, "lowSpeedRisk": {}, "ownLoss": {} }
}
```

Les « … » sont repris des constantes existantes au moment de la migration. La
normalisation et les éventuels normaliseurs déjà appliqués dans
`generalized_runtime_state.hpp` doivent être transcrits sans changement.

### 15.2 Flight

Voir §5.1. Les tailles de la tête `controls` et la liste exacte des canaux
sont reprises de `flight_types.hpp` et `hybrid_policy.hpp` lors de l'étape 3.

## 16. Plan de migration

Chaque étape fait l'objet d'une PR séparée.

| # | Étape | Critère d'acceptation |
|---|---|---|
| 1 | `ppokit/artifacts` : écriture atomique, en-tête, RNG portable. Remplacement des 4 implémentations | Golden trace Flight inchangée ; tests Blue de checkpoint au vert ; Debug et Release, CTest |
| 2 | `ppokit/core` : GAE `<T>` et surrogate PPO ; Flight et Blue les appellent | Golden trace Flight identique ; tests `generalized_learner_tests` identiques au bit près |
| 3 | `ppokit/heads` : têtes, puis composite pour Flight et gaussienne pour Blue | Parité de log-prob, d'entropie et de gradient avec `hybrid_policy.hpp` et le learner Blue ; golden trace |
| 4 | `ppokit/backend` unique : suppression de `RL_TOOLS_NAMESPACE_WRAPPER` | Les deux exemples liés dans le même exécutable ; pas de régression |
| 5 | `schema` + `obs` + `reward` + codegen des types (§4.2), sans réseaux | Layouts générés égaux aux offsets actuels (`static_assert`) ; hash stable |
| 6 | Codegen des réseaux, y compris l'attention Blue | Test d'acceptation n°1 (§7) |
| 7 | `runtime` Owned (Flight) puis Hosted (Blue), évaluation périodique, `eval`, `inference` | Golden trace ; parité d'inférence ; campagne Blue identique sur les mêmes seeds |
| 8 | `component` (ABI, SDK) puis `ppokit-studio` | Composant d'exemple (prétraitement avec état, et bloc différentiable) chargé en Debug et Release ; graphe Flight édité, régénéré et entraîné depuis le studio |

L'évaluation multi-seed des politiques entraînées est relancée après les
étapes 3, 6 et 7. Aucune conclusion de convergence n'est tirée des tests
unitaires.

## 17. Risques et questions ouvertes

1. **Bit-exactitude** : float pour Blue, double pour Flight ; l'ordre des
   réductions dans GAE et dans les pertes. Mitigation : templater sur le
   scalaire et reproduire l'ordre exact des opérations ; une différence non
   nulle est un échec.
2. **Attention Blue** : backward manuel de 731 lignes. Il faut extraire le
   bloc tel quel, d'abord comme composant interne différentiable, avant toute
   réécriture.
3. **Sur-abstraction** : seulement deux clients (mono-agent et multi-agent).
   Ne rien ajouter à l'API sans cas d'usage dans l'un des deux.
4. **Temps de compilation** du codegen rl_tools, avec un modèle par cible.
   À mesurer dès l'étape 6.
5. **Airspace / libtorch** : faut-il un backend libtorch derrière les mêmes
   interfaces, ou laisser Airspace sur `Framework/RL` ? Décision à prendre
   par l'utilisateur.
6. **`map: tanh`** : log-prob sur le latent ou corrigée du jacobien (§6).
7. **Relation avec `Framework/RL/Core`** : `ppo::core::IEnvironment`,
   `IPolicy` et `ILearner` existent déjà côté libtorch. Il faut choisir entre
   faire converger ces interfaces vers PPOKit ou les garder séparées
   jusqu'à la décision n°5.
8. **Nom définitif** de la bibliothèque.
