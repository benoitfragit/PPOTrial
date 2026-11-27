# PPOKit — framework commun PPO/MAPPO, backends rl_tools et libtorch, codegen et studio

> **LibTorch removed (November 2026).** The red agents' production weights are now
> `policy.txt` (converted from `model.pt`, same actions, `AirspacePolicies`) and their
> training uses PPOKit graph learners (`Examples/Airspace/Agents/Bindings`). The LibTorch classes
> described below no longer exist; this page is kept for history.

> Statut : **document de conception**, aucune modification de code.
> Nom de travail : `PPOKit` (bibliothèque `ppokit`, namespace `ppokit`).
> Le nom est provisoire.

## 0. Portée et garde-fous

**Dans la portée : les trois exemples.**
- BlueDefense, c'est-à-dire le MAPPO Blue généralisé
  (`Examples/BlueDefense/Agents/BluePatrol`), sur le backend rl_tools.
- FlightControl3D (`Examples/FlightControl3D`), sur le backend rl_tools.
- Airspace (`Examples/Airspace` : agents Interception/SNA et Mission/RLMission),
  sur le backend **libtorch**, caché derrière le même frontal (§3.1). La chaîne
  `Framework/RL/PPO` et `Examples/Airspace/Agents/Bindings` devient un détail
  d'implémentation de ce backend, puis disparaît.

**Contrainte sur RLMission :** BlueDefense utilise la politique de production
RLMission pour ses Rouges, et l'AGENTS.md interdit de la modifier pendant
l'évolution du MAPPO. Sa migration est donc une **étape séparée et la
dernière** (§16, étape 11). Elle exige l'approbation explicite de
l'utilisateur et une parité d'inférence sur le modèle `model.pt` existant.
Jusque-là, BlueDefense continue de charger RLMission par la chaîne actuelle.

**Exemples de référence :** à la fin de la migration, les trois exemples ont
la même structure et servent de modèles pour un nouvel utilisateur (§13.1).

**Invariants à respecter pendant toute la migration :**
- Aucun canal de récompense Blue n'est ajouté, renommé, repondéré ou renormalisé.
  Les 15 canaux de `GeneralizedRewardChannel` et les 7 canaux (depuis #87) de
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
4. Un graphe de réseau décrit en JSON, exécuté par l'un des deux backends :
   **compilé par génération de code** vers rl_tools, ou construit à
   l'exécution avec libtorch. Aucun des deux n'est exposé au client.
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
client C++ (Task / HostedAgent, hooks) ──► ppokit runtime ◄── backend rl_tools | libtorch (privé)
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
| `ppokit/backend/rltools` | **Privé.** Instanciation rl_tools, Adam, buffers, une seule cible CMake (supprime `RL_TOOLS_NAMESPACE_WRAPPER`) | rl_tools |
| `ppokit/backend/torch` | **Privé.** Construction du graphe en `torch::nn`, autograd, Adam libtorch, absorbe `Framework/RL/PPO` et `Examples/Airspace/Agents/Bindings` | libtorch |
| `ppokit/runtime` | Trainer owned/hosted, MAPPO, curriculum, évaluation périodique et promotion, budgets, télémétrie et commandes | tous les précédents |
| `ppokit/eval` | Suites, grilles, gates, rapports, rejeu | runtime |
| `ppokit/artifacts` | Formats de checkpoint, en-têtes d'identité, écriture atomique, migrations | core |
| `ppokit/component` | Chargement des DLL, ABI C des composants boîte noire, SDK C++ | core |
| `ppokit-codegen` | Exécutable JSON → C++ | schema |
| `ppokit-studio` | Éditeur imnodes, suivi des runs, rapports | runtime, eval, imgui, imnodes, implot |

### 3.1 Deux backends, un seul frontal

| | `backend/rltools` | `backend/torch` |
|---|---|---|
| Réseaux | Code généré depuis `graph.json` (§7), dimensions figées | Graphe construit **à l'exécution** depuis le même `graph.json` (modules `torch::nn`), sans recompilation |
| Types client (`Obs`, `Action`, `Rewards`…) | Générés | Générés, identiques |
| Gradient | Forward et backward explicites | Autograd |
| Composants DLL (§12) | Appel à la table C | Enveloppés dans une `torch::autograd::Function` qui appelle `forward` et `backward` de la table C |
| Exemples | BlueDefense, FlightControl3D | Airspace |
| Option CMake | toujours présent | `PPOKIT_WITH_TORCH` (par défaut ON, car le desktop actuel lie libtorch) |

**Règles :**
- Le backend est choisi dans `graph.json` (`"backend": "rltools"` ou
  `"torch"`). Le client ne change pas une ligne de code pour passer de l'un à
  l'autre.
- Aucun en-tête rl_tools ni libtorch n'apparaît dans l'API publique. Le
  frontal manipule des buffers plats et les types générés.
- Les deux backends **coexistent dans le même processus**. C'est nécessaire
  pour BlueDefense : MAPPO Blue sur rl_tools et Rouges RLMission sur libtorch.
- Les têtes, le GAE, le surrogate PPO, les schémas, les artefacts,
  l'évaluation et le runtime sont communs. Seuls l'acteur, le critique et
  l'optimiseur sont propres au backend.
- **Parité :** à l'intérieur d'un backend, l'exigence de traces identiques au
  bit près demeure. Entre backends, il n'y a pas de bit-exactitude. Il y a un
  test de parité du forward à tolérance fixée, sur des poids convertis, pour
  les graphes sans composant, et une comparaison statistique par évaluation
  multi-seed pour l'entraînement.
- Les artefacts portent leur backend (§8). Un convertisseur de poids
  rl_tools ↔ torch est fourni pour les blocs intégrés.
- La compatibilité ABI Windows Debug/Release de libtorch, déjà traitée dans
  `generic-rl-core.md`, reste à la charge du backend torch.

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
- **Composantes actives** : `"active": {"brake": {"when": "rollout", "above":
  0.5}}` dans un groupe d'actions (ou `"at_most"`) rend une composante active
  seulement dans certains états. Inactive, elle n'est pas tirée (action 0) et
  n'entre ni dans la vraisemblance, ni dans le KL, l'entropie ou les
  gradients (frein au sol seulement, train tant qu'il n'est pas verrouillé,
  comme le learner FlightControl3D). Hors du hash de forme.
- **Versionnement** : le hash canonique de `observation` remplace les
  constantes manuelles comme `GeneralizedObservationLayout::Version = 10`. On
  peut garder un alias lisible (`"version": 10`) pour la compatibilité.

### 5.2 `graph.json`

```json
{
  "schema": "ppokit.graph/1.0",
  "task": "flight",
  "backend": "rltools",
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
           "maxGradNorm": 0.5, "targetKL": null },
  "device": "auto",
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
- `device` vaut `"cpu"`, `"cuda"` ou `"auto"` (§13.6). Il n'agit que sur le
  backend torch en v1.
- La signification de chaque paramètre PPO est décrite en §13.4.

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

**Compatibilité Flight :** depuis #87, `flight_policy.hpp` utilise une tête
gaussienne seule (5 actions, log-std indépendant de l'état, latent clippé), avec
un gradient exact du surrogate. La tête `gaussian(map: clip)` doit reproduire
ces valeurs au bit près ; la tête composite reste exercée par ses propres tests.

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
- `backend` (`rltools` ou `torch`) ;
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

### 8.1 Reprise des modèles existants

Les modèles déjà entraînés des trois exemples doivent rester utilisables
**en inférence et en reprise d'entraînement**, sans repartir de zéro.

| Exemple | Ancien format | Reprise d'entraînement | Inférence |
|---|---|---|---|
| BlueDefense | Acteur v10, training V7, production promue | Training V7 → `training` : poids acteur et critique, Adam, RNG, compteurs, étape de curriculum | Production → `promoted` |
| FlightControl3D | Checkpoint learner, meilleur modèle du `FlightModelStore` | → `training` | Meilleur modèle → `promoted` |
| Airspace | `model.pt` (SNA, RLMission) | → `training` (backend torch) si l'état de l'optimiseur est présent, sinon reprise avec Adam réinitialisé et un avertissement explicite | → `policy` / `promoted` |

**Règles :**
- `ppokit-import <exemple> <ancien-fichier>` convertit, vérifie les formes et
  écrit l'artefact. Le chargeur reconnaît aussi l'ancien format directement,
  pour que les raccourcis actuels (`.bat`, `PPO.exe`) continuent de marcher
  pendant la migration.
- **Critères d'acceptation** :
  - même action déterministe, avant et après conversion, sur un jeu d'états
    fixé ;
  - pour la reprise d'entraînement, la première update après conversion
    produit les mêmes poids, au bit près pour rl_tools et à tolérance fixée
    pour torch, que l'ancienne chaîne à partir du même checkpoint.
- Un ancien checkpoint dont la forme ne correspond plus au `task.json` est
  refusé, avec un message qui donne les dimensions attendues et trouvées.

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

**Implémentation (v1).** `ppokit/component/abi.h`, `sdk.hpp`,
`library.hpp`, `component_node.hpp`. Écarts assumés par rapport au
texte ci-dessus : `create` reçoit aussi le seed de l'instance (paramètres
initiaux reproductibles, dérivé du nom du nœud) ; `backward` **ajoute** ses
gradients (accumulation sur le minibatch, comme les blocs intégrés). Dans
un graphe, v1 accepte des entrées float et des masques bool, une seule
sortie float, et refuse les composants à état (§12.5 demande de stocker
leurs sorties dans le rollout, à faire avec les tâches externes).

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

Il est développé **après** la stabilisation des formats (§16, étape 10).
Principe directeur : **tout réglage visible porte une infobulle**, qui dit ce
qu'il fait, sa plage habituelle et les symptômes d'une mauvaise valeur. Le
texte de ces infobulles vient d'un seul fichier de métadonnées
(`ppokit/meta/parameters.json`), partagé par le studio, la CLI
(`ppokit-train --explain gamma`) et la documentation générée. Il n'y a donc
qu'une seule source de vérité.

- **Éditeur de graphe** (imnodes) :
  - palette des blocs intégrés et des composants découverts dans les DLL,
    avec leurs ports ;
  - vérification des formes en direct, avec la même bibliothèque que le
    codegen ;
  - export vers `graph.json`.
- **Bouton Build** : lance la configuration et le build CMake du modèle dans
  `build/<config>` et affiche le journal. Avec le backend torch, le graphe est
  reconstruit sans build.
- **Runs** :
  - lancement en headless (processus séparé) ou en mode embarqué (desktop
    existant) ;
  - courbes ImPlot alimentées par la télémétrie ;
  - comparaison de runs.
- **Évaluation** : rapports par cellule de grille, rejeu d'épisodes, export
  de cas de régression.

Le studio n'entre jamais dans la boucle d'entraînement : c'est un client de
la télémétrie et des commandes.

### 13.1 Exemples de référence, faciles à prendre en main

Chaque exemple migré a la même structure, qui sert de gabarit :

```
Examples/<Nom>/
  README.md          ce que fait l'exemple, comment le lancer, quoi modifier en premier
  task.json          observations, actions, récompenses, scénarios, métriques
  graph.json         réseaux (éditable dans le studio)
  train.json         hyperparamètres, curriculum, évaluation périodique
  eval/              suites smoke, regression, heldout
  src/<nom>_task.cpp Reset / Step / métriques (+ hooks éventuels)
  src/physics/…      simulateur propre à l'exemple
  render/…           rendu optionnel, branché sur la télémétrie
  CMakeLists.txt     ppokit_add_example(<nom> TASK task.json GRAPH graph.json)
```

Ce qui rend les exemples faciles à prendre en main :
- **Une ligne pour enregistrer un exemple** : `PPOKIT_REGISTER_TASK` alimente
  aussi le catalogue du desktop (`ExampleDefinition`), qui n'a plus à être
  rempli à la main.
- **Un seul lanceur** : `ppokit-train <exemple>`, `ppokit-eval <exemple>
  <suite>`, `ppokit-play <exemple> <artefact>`, et les mêmes actions dans le
  studio.
- **Un gabarit** : `ppokit-new <nom>` crée un exemple minimal qui compile et
  s'entraîne : une tâche jouet, un graphe MLP, une suite smoke.
- **Des messages d'erreur orientés utilisateur** : chaque erreur de schéma,
  de câblage ou de checkpoint indique le fichier, le champ et la correction
  attendue.
- **Les README des exemples** suivent le même plan : ce que l'agent apprend,
  lancer, observer, modifier la récompense, modifier le réseau, évaluer.
- **Les fichiers complexes restent dans l'exemple, pas dans le chemin
  d'apprentissage** : la physique, le rendu et la géométrie métier sont
  rangés à part de `task.json` et de `*_task.cpp`, qui sont les deux fichiers
  qu'un nouvel utilisateur lit en premier.

### 13.2 Éditeur de curriculum

Il édite la section `curriculum` de `train.json`. Le fichier reste la source
de vérité, et le studio n'en est qu'une vue.

- **Frise des étapes** : glisser-déposer pour ordonner, dupliquer, renommer.
  Chaque étape affiche son état : en attente, active, validée.
- **Échantillonneurs par paramètre de scénario** : le widget dépend du type
  déclaré dans `task.json` :
  - curseur de plage pour `uniform` ;
  - curseur avec moyenne et écart-type pour `normal` ;
  - poids par valeur pour `choice` sur un `enum` ;
  - case à cocher pour un `bool`.

  Les bornes du schéma sont imposées et ne peuvent pas être dépassées.
- **Aperçu** : bouton « Échantillonner 1 000 scénarios », qui affiche
  l'histogramme de chaque paramètre et quelques scénarios rejouables. On
  voit ce qu'une étape produit avant de lancer l'entraînement.
- **Règle de passage** : métrique, seuil, fenêtre glissante. Pendant
  l'entraînement, une jauge montre la valeur courante par rapport au seuil.
- **Répétition** (`rehearse`) : proportion d'épisodes tirés dans les étapes
  précédentes, pour éviter l'oubli.
- **Contrôle manuel** : avancer ou reculer d'étape en cours de run. L'action
  est journalisée dans le run et dans le checkpoint.
- **Validation** :
  - étape vide ;
  - seuil sur une métrique inconnue ;
  - fenêtre plus grande que le budget ;
  - recouvrement entre les seeds d'entraînement et d'évaluation.

  Chaque erreur affiche un message qui pointe le champ fautif.
- **Modèles prédéfinis** : « progression d'une seule variable », « du plus
  facile au plus dur sur N variables », « une étape fixe ». Ils produisent un
  JSON lisible et modifiable ensuite.

### 13.3 Tableau de bord des récompenses

Pour chaque canal déclaré dans `task.json` :

| Vue | Contenu |
|---|---|
| Moyenne par pas | Valeur brute et valeur pondérée, sur une fenêtre glissante |
| Cumul par épisode | Moyenne, p5 et p95 sur les N derniers épisodes |
| Contribution | Part de chaque canal dans le retour total (barres empilées), en valeur absolue et signée |
| Par agent et par rôle | Pour MAPPO : leader et équipiers séparés, ou un slot précis |
| Par étape de curriculum | Mêmes vues filtrées par étape |
| Corrélation | Cumul d'un canal comparé à une métrique d'évaluation (nuage de points), pour voir si un canal sert réellement l'objectif |

**Réglage des poids en cours de run :**
- Le changement s'applique à la **prochaine frontière d'update**, jamais au
  milieu d'un rollout.
- C'est un **événement versionné** du run : ancien poids, nouveau poids,
  numéro d'update. Il est visible sur toutes les courbes sous la forme d'un
  repère vertical et enregistré dans le checkpoint.
- Une infobulle avertit que changer un poids change l'objectif. La fonction
  de valeur devient temporairement fausse : attendez-vous à un pic de perte
  de valeur et de KL.
- Les poids par défaut viennent de `task.json`. Le studio propose
  d'enregistrer les nouveaux poids dans le fichier ou de les garder pour ce
  seul run.
- Pour BlueDefense, les canaux, les poids et les normalisations existants ne
  changent que sur décision explicite de l'utilisateur (AGENTS.md).
  L'outil le permet, mais rien n'est modifié automatiquement.

### 13.4 Paramètres PPO commentés

Chaque paramètre a une infobulle et un statut **modifiable à chaud**
(appliqué à la prochaine frontière d'update) ou **au redémarrage**.

| Paramètre | Ce que ça fait | Plage habituelle | Symptômes d'une mauvaise valeur | Modifiable |
|---|---|---|---|---|
| `gamma` | Horizon : poids des récompenses futures. L'horizon effectif vaut environ 1/(1−γ) pas | 0,95–0,999 | Trop bas : agent myope. Trop haut : apprentissage lent et bruité | à chaud |
| `lambda` (GAE) | Compromis biais/variance de l'avantage | 0,9–0,98 | Bas : biaisé, sous-estime le long terme. Haut : bruité | à chaud |
| `clip` | Écart maximal du nouveau ratio de probabilité par update | 0,1–0,3 | Trop haut : politique instable, KL élevée. Trop bas : apprentissage lent | à chaud |
| `learningRate` | Taille des pas de l'optimiseur | 1e-5 – 1e-3 | Trop haut : KL et clip fraction élevées, effondrement. Trop bas : courbes plates | à chaud |
| `epochs` | Passes sur chaque rollout | 3–15 | Trop haut : sur-ajustement au rollout, KL élevée | à chaud |
| `minibatch` | Taille des lots d'optimisation | 64–4 096 | Petit : gradients bruités. Grand : moins d'updates par rollout | à chaud |
| `rolloutSteps` | Pas collectés avant chaque update | 1 024–65 536 | Petit : avantages bruités. Grand : updates rares | au redémarrage (tampons à capacité fixe) |
| `entropy` | Bonus d'exploration | 0–0,02 | Trop bas : convergence prématurée. Trop haut : politique qui reste aléatoire | à chaud |
| `valueCoef` | Poids de la perte du critique | 0,25–1 | Trop bas : critique en retard, avantages faux | à chaud |
| `maxGradNorm` | Écrêtage de la norme du gradient | 0,5–1 | Trop bas : apprentissage freiné | à chaud |
| `targetKL` (optionnel) | Arrêt anticipé des époques si la KL dépasse la cible | 0,01–0,05 | — | à chaud |

À côté de chaque réglage, le studio affiche les **indicateurs de santé**
concernés : KL, clip fraction, entropie, variance expliquée du critique,
norme du gradient. Chacun a son infobulle d'interprétation, par exemple
« clip fraction > 0,3 : updates trop agressives, baissez le learning rate
ou le clip ». Tout changement est un événement versionné, comme pour les
poids de récompense.

### 13.5 Sondes

Une **sonde** se pose sur n'importe quel point observable :
- un champ d'observation nommé ;
- une arête ou un port du graphe : activation intermédiaire, sortie d'un
  composant DLL, poids d'attention ;
- les paramètres d'une tête : moyenne, log-std, logits, probabilités ;
- l'action échantillonnée et l'action après `AdaptAction` ;
- la valeur, l'avantage, le retour ;
- un canal de récompense, une métrique.

| Type | Affichage |
|---|---|
| Série temporelle | Valeur scalaire ou composante choisie, au fil des pas |
| Histogramme | Distribution sur une fenêtre, utile pour la saturation (tanh) et les entrées mal normalisées |
| Carte de chaleur | Vecteurs et matrices : poids d'attention par piste, sorties par slot |
| Statistiques | Min, max, moyenne, écart-type, nombre de valeurs non finies |
| Déclencheur | Condition (`> x`, non fini, saturation) qui met le run en pause et enregistre l'épisode pour le rejeu |

**Coût et mise en œuvre :**
- Une sonde désactivée ne coûte **rien** : aucun test dans la boucle chaude.
- Backend rl_tools : le codegen génère des points de prélèvement
  compilables, actifs en Debug et en Release. Activer une sonde arme un
  pointeur vers un tampon circulaire préalloué. Le prélèvement est décimé
  (1 pas sur N, réglable) et sans allocation.
- Backend torch : hooks `forward` sur les modules concernés.
- Composants DLL : leurs ports sont sondables comme les autres, puisque le
  framework possède les buffers.
- Les sondes fonctionnent en entraînement, en évaluation, en inférence et en
  rejeu. Sur un épisode rejoué, on peut poser une sonde après coup et revoir
  exactement ce qui s'est passé.
- Les configurations de sondes s'enregistrent dans un fichier
  (`probes.json`), séparé du modèle : elles n'affectent ni le hash ni les
  résultats.

### 13.6 GPU

- **Backend torch** : `"device": "cuda"` dans `train.json` (ou `"auto"`). Les
  réseaux, l'optimiseur et les lots de mise à jour vont sur le GPU. La
  simulation et la collecte restent sur CPU.
- **Backend rl_tools** : CPU par défaut (`RL_TOOLS_BACKEND_ENABLE_CUDA` est
  aujourd'hui à OFF dans `dependencies.cmake`). Le support CUDA de rl_tools
  est prévu comme option expérimentale du backend, sans engagement.
- **Repli** : si CUDA est absent ou indisponible, le run passe sur CPU avec un
  avertissement explicite dans le journal et dans le studio. Il n'y a pas
  d'échec silencieux.
- **Déterminisme** : les traces de référence (golden traces) et les tests de
  parité restent **CPU uniquement**. Sur GPU, la reproductibilité n'est que
  statistique (opérations cuDNN non déterministes), et l'artefact enregistre
  le device utilisé.
- **Attente réaliste** : avec de petits MLP et une simulation sur CPU, le
  GPU accélère peu, voire ralentit, à cause des transferts. Il devient utile
  pour les gros lots, l'attention sur beaucoup de pistes et les grands
  critiques. Le studio affiche le temps passé en collecte et en mise à jour,
  pour en juger.
- **Validation** : la CI n'a pas de GPU. Le chemin CUDA est compilé et testé
  avec un device simulé (tests de placement, de repli et de sérialisation).
  Les performances réelles sont à mesurer sur une machine équipée.

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
| Observation (33 depuis #87) | `flight_types.hpp` | `schema` + `obs` | Calcul des features |
| Tête gaussienne (5 actions, log-std indépendant de l'état), clip, entropie analytique (depuis #87) | `flight_policy.hpp` | `heads.gaussian(map: clip)` + test de parité | — |
| Contrôles forcés par phase, throttle de référence | `ApplyPhaseControls`, `flight_policy.hpp` | Hook `AdaptAction` | Règles |
| 7 canaux de récompense, poids, réglage des récompenses (`flight_reward_tuning`) | `flight_types.hpp`, `flight_config.*`, `Runtime/flight_reward_tuning.*` | `reward` | Termes |
| Curriculum Loc/Glide/Joint, reminder LOC, placements | `flight_config.*`, `flight_environment.cpp` | `train.json` curriculum + `rehearse` | Placement dans `Reset` |
| GAE (double), PPO, MLP rl_tools, Adam | `flight_learner.*` | `core` GAE `<double>` + `backend` | — |
| Checkpoint atomique, RNG portable | `flight_learner.cpp` | `artifacts` | — |
| Validation, meilleur modèle | `FlightSession::StartValidation`, `flight_model_store.*` | `runtime` évaluation périodique + `promoted` | Métriques (`ValidationMetrics`) |
| Session pas à pas, budgets, snapshot | `flight_session.*` | Runner `Tick`, budget, télémétrie | Champs métier du snapshot |
| Headless | `flight_headless.cpp` | CLI `ppokit-train` ou main client minimal | — |
| Rendu et instruments | `Renderer/*` | — | Client (lit la télémétrie) |
| Golden trace | `tests/golden_trace_tests.cpp` | Inchangé : critère d'acceptation de chaque étape | — |

### 14.3 Airspace (backend libtorch)

| Service actuel | Fichier(s) | PPOKit | Reste client |
|---|---|---|---|
| Agent Interception (SNA) : entraînement et production | `Interception.*` (`SNATrainingAgent`, `SNAProdAgent`) | `runtime` Hosted + `inference` | Adaptateur ECS mince |
| Réseau acteur-critique MLP, log-std en sortie du réseau | `actorcriticnet.*` | `graph.json` backend `torch`, `heads.gaussian` avec `logStd: "network"` | — |
| Agent Mission (RLMission), attention sur les pistes, 3 actions | `actorcriticmission.*`, `missionlayout.hpp` | Bloc `masked_attention` (torch), `heads.gaussian` ; **étape 11** | — |
| Filtre de pistes de mission | `missiontrackfilter.hpp` | Composant de prétraitement non entraînable (§12) ou fournisseur de références | Logique de filtrage |
| Sélection de cible | `targetselector.*` | — | Client (pas de contrôleur d'affectation externe ajouté) |
| PPO libtorch, learner, bindings | `Framework/RL/PPO`, `Examples/Airspace/Agents/Bindings` | `backend/torch` + `core` (GAE et surrogate communs) | — |
| Canaux de récompense (destruction de cible, de l'ownship…) | `interception.hpp` | `reward` | Termes |
| Initialisation aléatoire ou depuis checkpoint | `randominitializersystem.*`, `checkpointinitializersystem.*` | `scenario` + curriculum | Placement dans l'ECS |
| Méta-registre, règles de monde | `airspacemeta.*`, `airspaceworldrules.*` | — | Client (ECS) |
| Évaluateur | `evaluator.*` | `eval` | Métriques métier |
| Rendu tactique | `Renderer/*` | — | Client (lit la télémétrie) |
| Checkpoints `model.pt` | `TorchLearner`, `TorchProductionPolicy` | `artifacts` (backend `torch`) + lecteur de l'ancien format | — |

**Conclusion :** aucun service des trois exemples ne reste sans emplacement.
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
sont reprises de `flight_types.hpp` et `flight_policy.hpp` lors de l'étape 3.

## 16. Plan de migration

Chaque étape fait l'objet d'une PR séparée. L'avancement réel est suivi dans
[`PPOKit/PROGRESS.md`](../../PPOKit/PROGRESS.md).

| # | Étape | Critère d'acceptation |
|---|---|---|
| 0 | Golden trace Flight (absente de master) : 3 updates PPO depuis un seed fixe, checkpoint complet hashé | Empreinte identique en Debug et Release, enregistrée sur master avant toute refonte |
| 1a | `ppokit/artifacts` : écriture atomique. Remplacement des 4 implémentations | Golden trace Flight inchangée ; tests Blue de checkpoint au vert ; Debug et Release, CTest |
| 1b | En-tête d'identité et RNG portable des artefacts (reporté avec l'étape 5, pour ne changer les formats sur disque qu'une fois) | Lecture des anciens formats conservée (§8.1) |
| 2 | `ppokit/core` : GAE `<T>` ; Flight et Blue l'appellent. Le surrogate PPO passe à l'étape 3, car les deux implémentations diffèrent (bornes du log-ratio, règle d'activation du gradient) | Golden trace Flight identique ; trace de `generalized_learner_tests` identique au bit près |
| 3 | `ppokit/heads` (gaussienne) et surrogate PPO communs ; une **formule unique** (décision de l'utilisateur) remplace celles de Flight et de Blue. Bernoulli, catégorielle et composite arrivent avec le codegen | Écart d'au plus 1 ulp sur la log-densité ; épisodes de la golden trace inchangés ; nouvelles références enregistrées ; évaluation multi-seed avant fusion |
| 4 | `ppokit/backend/rltools` unique : suppression de `RL_TOOLS_NAMESPACE_WRAPPER` | BlueDefense et Flight liés dans le même exécutable ; pas de régression |
| 5 | `schema` + codegen des layouts (§4.2), sans réseaux. Fait de façon additive : `task.json` de Flight et de Blue, tests de correspondance. Les normaliseurs (`obs`) et le moteur de récompense (`reward`) arrivent avec le runtime (étape 7) | Layouts générés égaux aux offsets actuels (`static_assert`) ; hash stable |
| 6 | Codegen des réseaux. Fait : `graph.json` v1 (MLP, modèles externes), types rl_tools générés, parité prouvée (Flight au bit près, critique Blue de même type). Restent : brancher les learners (après le rebase de l'utilisateur), backend unique, attention Blue | Test d'acceptation n°1 (§7) |
| 7 | `runtime` Owned (Flight) puis Hosted (Blue), évaluation périodique, `eval`, `inference`. Fait : rollout, collecte Owned et learner PPO générique sur réseaux générés, validés sur une tâche jouet | Golden trace ; parité d'inférence ; campagne Blue identique sur les mêmes seeds |
| 8 | `component` (ABI, SDK) | Composant d'exemple (prétraitement avec état, et bloc différentiable) chargé en Debug et Release |
| 9 | `backend/torch` derrière le même frontal (avec `device` CPU/CUDA et repli, §13.6) ; Airspace Interception (SNA) migré | Parité d'inférence sur le `model.pt` existant ; entraînement comparable en évaluation multi-seed ; coexistence des deux backends dans `PPO.exe` |
| 10 | Exemples de référence (§13.1) : structure commune, lanceur, `ppokit-new`, `ppokit-import`, README ; puis `ppokit-studio` : graphe, curriculum, récompenses, paramètres PPO commentés, sondes (§13.2 à §13.5) | Un exemple créé par `ppokit-new` compile et s'entraîne ; les modèles existants des trois exemples reprennent en inférence et en entraînement (§8.1) ; graphe Flight édité, régénéré et entraîné depuis le studio ; une sonde désactivée ne change ni les traces ni le temps de pas mesuré |
| 11 | Airspace RLMission migré, puis utilisé par les Rouges de BlueDefense — **sur approbation explicite** | Actions de production identiques (tolérance fixée) sur les mêmes états ; campagne BlueDefense inchangée sur les mêmes seeds |

L'évaluation multi-seed des politiques entraînées est relancée après les
étapes 3, 6, 7, 9 et 11. Aucune conclusion de convergence n'est tirée des tests
unitaires.

## 17. Risques et questions ouvertes

1. **Bit-exactitude** : float pour Blue, double pour Flight ; l'ordre des
   réductions dans GAE et dans les pertes. Mitigation : templater sur le
   scalaire et reproduire l'ordre exact des opérations ; une différence non
   nulle est un échec.
2. **Attention Blue** : backward manuel de 731 lignes. Il faut extraire le
   bloc tel quel, d'abord comme composant interne différentiable, avant toute
   réécriture.
3. **Sur-abstraction** : trois clients (mono-agent, MAPPO, et deux agents
   libtorch). Ne rien ajouter à l'API sans cas d'usage dans l'un d'eux.
4. **Temps de compilation** du codegen rl_tools, avec un modèle par cible.
   À mesurer dès l'étape 6.
5. **Airspace / libtorch** : *décidé*. Backend libtorch derrière le même
   frontal (§3.1). Restent ouverts : la tolérance de parité entre backends, et
   le coût d'enveloppement des composants DLL dans l'autograd.
6. **`map: tanh`** : log-prob sur le latent ou corrigée du jacobien (§6).
7. **Relation avec `Framework/RL/Core`** : `ppo::core::IEnvironment`,
   `IPolicy` et `ILearner` deviennent des adaptateurs vers PPOKit pendant les
   étapes 9 à 11, puis sont supprimés une fois les trois exemples migrés.
8. **Facilité de prise en main** : à mesurer concrètement, par exemple en
   comptant le nombre de lignes de C++ client et de fichiers à lire pour
   l'exemple généré par `ppokit-new`. Cible : un seul fichier C++ et quatre
   fichiers JSON.
9. **Nom définitif** de la bibliothèque.

## 18. Exécution : ppokit-run, studio et viewer (décidé le 23/11/2026)

Décision de l'utilisateur : **option B**. Le studio est l'exe principal ; il
lance, commande et regarde tout. L'entraînement tourne dans un programme à
part, `ppokit-run`, auquel le studio se raccroche en direct. La vue du monde
est un troisième programme, `ppokit-viewer`, alimenté par des **blocs
Visualiseur** du graphe.

```
             commandes (pause, play, stop, poids)          ┌──────────────┐
 ┌────────┐ ─────────────────────────────────────────────▶ │  ppokit-run  │
 │ studio │ ◀───────────────────────────────────────────── │ env  (DLL)   │
 └────────┘   mesures en direct (récompenses, pertes,      │ réseau (DLL) │
                entropie, KL, sondes) — socket local       │ blocs Visu.  │
                                                           └──────┬───────┘
                                    UDP, octets opaques à PPOKit  │
                                                           ┌──────▼───────┐
                                                           │ppokit-viewer │
                                                           │ vue (DLL)    │
                                                           └──────────────┘
```

### 18.1 Les programmes

| Programme | Rôle | Fenêtre |
|---|---|---|
| `ppokit-studio` | Projets, graphe, réglages ; lance `ppokit-run` ; courbes et commandes **en direct** ; journal pour l'historique | oui |
| `ppokit-run` | Boucle d'entraînement, d'évaluation ou d'inférence d'un projet ; charge la DLL d'environnement de l'exemple et la DLL réseau ; exécute les blocs Visualiseur ; option `--realtime` | non |
| `ppokit-viewer` | Voit le monde en 2D ou en 3D ; se raccroche et se décroche à chaud ; charge la DLL de vue ; **ni récompenses ni commandes** | oui |

Un entraînement survit à la fermeture du studio ; le studio s'y raccroche à
sa réouverture. La CI lance `ppokit-run` directement. Un plantage de
l'environnement (JSBSim) ne fait pas tomber le studio.

### 18.2 DLL réseau : API C stable

rl_tools fixe les formes à la compilation : le codegen produit **une DLL par
réseau** depuis `graph.json`, mais toutes ont la même API C. Le client ne voit
jamais rl_tools.

```c
ppokit_policy* p = ppokit_policy_open("policy.ppokit", error, sizeof error);
ppokit_policy_info(p, &info);            /* tailles, noms des observations et actions */
ppokit_policy_act(p, obs, actions, 1);   /* 1 = action déterministe */
ppokit_policy_close(p);
```

La même DLL porte l'apprentissage pour `ppokit-run` : création depuis
`task.json` et `graph.json`, collecte, mise à jour, statistiques, sauvegarde
et reprise. Un en-tête C++ fin enveloppe l'API C.

### 18.3 DLL d'environnement

L'exemple fournit son environnement en DLL (API C) : contrat (`task.json`),
reset, step, observations, récompenses par canal, ports `world.*` (§18.4),
réglage des poids. `ppokit-run` ne connaît que cette API. Les programmes
actuels (FlightControlHeadless, PPO.exe pour Blue et Airspace) deviennent ces
DLL.

### 18.4 Blocs Visualiseur

- **Rôle observateur** : un composant (§12) qui n'a **que des entrées**. Le
  manifeste déclare `"observer": true` ; aucune arête ne peut partir de lui.
  Il ne change donc ni le réseau, ni les actions, ni les artefacts : il
  n'entre pas dans le hash du réseau.
- **Où il vit** : dans une section `observers` de `graph.json`, à part des
  réseaux, sous la même forme : des nœuds `input` lisent un signal du pas
  (`observation.<groupe>`, `world.<groupe>`, `action`, `action.<groupe>`,
  `reward`) et sont reliés aux ports du bloc. Le studio l'affiche dans un
  troisième canevas, « Observers », à côté d'Actor et Critic.
- **Manifeste** : capacité `observer` (`Observer()` dans le SDK) : entrées
  f32 seulement, aucune sortie, ni gradient ni état, et `batch` (un appel
  reçoit tous les agents du pas, une ligne par agent).
- **Ports `world.*`** : `task.json` gagne une section `world`, l'état du monde
  dont la vue a besoin (position, attitude, zones…). L'environnement les
  remplit ; le learner les ignore ; seuls les observateurs peuvent s'y relier.
- **Quand il tourne** : `ppokit-run` l'appelle après chaque pas
  d'environnement (collecte, évaluation, inférence), **jamais** pendant les
  passes d'optimisation PPO. Sans allocation, sans attente.
- **Ce qu'il cache** : le transport. Le bloc Visualiseur des exemples envoie
  ses entrées en UDP (`udp://hôte:port` dans sa configuration) : un paquet
  « schéma » répété de temps en temps, puis des paquets de données (numéro,
  épisode, pas, valeurs). Si personne n'écoute, les paquets sont perdus.
  PPOKit ne voit que des tampons.

### 18.5 DLL de vue

La DLL du Visualiseur exporte aussi, en plus de `ppokit_components`, une
table `ppokit_views` (`ppokit/view/view_abi.h`) chargée par `ppokit-viewer` :
`create(config)`, `receive(octets, taille)`, `input(souris)`,
`render(PpokitDraw)`, `destroy()`. Décidé en le faisant : la vue dessine par
une **table de fonctions fournie par le viewer** (lignes, cercles, texte,
caméra 3D, modèles chargés une fois), pas par raylib. Une DLL de vue n'a donc
aucune bibliothèque graphique, et raylib reste statique : rien de plus à
livrer. Trames UDP auto-décrites (`ppokit/view/frames.hpp`, nom et taille de
chaque port) ; `FrameSender` envoie les entrées de l'observateur sans
allocation, à `rate` trames par seconde au plus. Le studio lance le viewer
avec le port du bloc (bouton « View »).

### 18.6 Canal de commande et de mesures

Socket local entre le studio et `ppokit-run`. Messages versionnés :
- du studio : pause, play, stop (sauvegarde puis arrêt), poids des récompenses,
  demande d'évaluation ;
- de `ppokit-run` : état, mesures par pas et par mise à jour (récompenses par
  canal, pertes, entropie, KL, sondes), fin de bloc, erreurs.

Le journal sur disque reste l'historique ; `stop.request` reste valide pour
la CI.

### 18.7 Ce qui disparaît

L'interface de PPO.exe (catalogue, sélecteur, panneaux de récompenses et
d'entraînement) et la fenêtre de FlightControl3D.exe. Le rendu est gardé :
`flight_view.cpp` et les vues tactiques d'Airspace deviennent des DLL de vue.

### 18.8 Étapes (une PR chacune)

1. Blocs observateurs : manifeste, section `observers` de `graph.json`,
   règles (aucune sortie, ports autorisés), éditeur, DLL de test.
2. Canal de commande et de mesures, et `ppokit-run` sur un environnement de
   test ; le studio en direct.
3. DLL réseau (API C stable) produite par le codegen ; parité avec le learner
   actuel.
4. `ppokit-viewer`, raylib partagé, API de vue, raccrochage UDP.
5. T-6 : DLL d'environnement, ports `world.*`, DLL Visualiseur (depuis
   `flight_view.cpp`), `--realtime`. Montrer l'écart numérique avant de
   remplacer le run T-6 existant. *Infrastructure faite (projet
   T6) ; learner graphes aligné sur rl_tools ; validation, BEST et fin
   d'entraînement repris en suites d'évaluation génériques ; T6 passé sous
   ppokit-run avec son checkpoint de l'update 243 porté une fois. Vue : voir
   §18.10. PC7 sous ppokit-run (mêmes réseaux et suite que T6, entraînement
   depuis zéro).*
6. Blue et Airspace : DLL d'environnement et vues 2D.
7. Retrait de l'interface de PPO.exe et de FlightControl3D.exe ; `.bat` et
   workflows. *FlightControl3D fait : FlightControl3D.exe, son rendu
   (Renderer/), son pont ECS et ses entrées du catalogue de PPO.exe sont
   retirés (vue au niveau, §18.10) ; FlightControlHeadless reste (trim d'un
   avion, référence des tests de parité). PPO.exe fait : l'hôte (`Main/` :
   PPO.exe, catalogue et sélecteur d'exemples, leurs tests), le rendu
   d'Airspace (`Examples/Airspace/Renderer`, `Framework/Support/Renderer`),
   `Renderer.json`, `imgui.ini` et les raccourcis de la racine sont retirés,
   ainsi que l'entrée PPO.exe d'Airspace (`Definition`, `LoadScenario`) ; les
   lanceurs et workflows passaient déjà par ppokit-run. Puis, sur décision de
   l'utilisateur, les modes dans le monde et la plomberie de l'interface :
   - BlueDefense : `GeneralizedPatrolMappoAgent` n'a plus de mode (Training,
     ProductionInference, ProductionEvaluation) : il est toujours piloté de
     l'extérieur (blue_environment ; learner et acteur de production dans
     blue_network ; journal, checkpoints, promotion et suites dans
     ppokit-run). `Definition`, `InferenceDefinition`, `EvaluationDefinition`
     et le contrat `Examples/Shared/Host/example.hpp` sont retirés, comme la
     campagne dans le monde (`GeneralizedEvaluationCampaign`, l'accumulateur
     de métriques) et les tests de parité contre ces modes ;
   - Airspace : `TrainingAgent` n'a plus de learner, de checkpoint ni
     d'évaluateur ; il lit de `settings/Training/<agent>.json` la taille du
     rollout, la borne et les poids des récompenses (les hyperparamètres PPO
     sont ceux de `projects/RLMission/training.json`) ; `IsLastStep` reste
     faux, comme sous ppokit-run avant ce retrait (la récompense
     d'entraînement ne change pas ; `AirspaceTrainingRewardTests`). La classe `Evaluator` est
     retirée (`EvaluationCampaign` reste, pour les suites des DLL) ; le test de
     fumée fige les notes qu'elle donnait (RLMission 465,70…, RLSNA 5485,93…) ;
   - Framework : `Environment` est piloté par son appelant seulement
     (BeginStep/EndStep, `SteppedWorld`) : plus de thread, de snapshots
     (`CopySnapshot` des règles du monde), de commandes temps réel ni de mode
     headless ; `IAgent` se réduit à Init, ChooseAction et SetNewState ; le
     `Serializer` ne fait que charger ; Meta n'est plus qu'une table de
     fabriques ; RL/Core garde le contrat d'agent, les tampons, `IPolicy` et
     l'ordonnancement multi-agents (runner, évaluation, suivi d'apprentissage
     et `sEvaluatorState` retirés).*

### 18.9 Étape 6 en détail : Blue (MAPPO) et Airspace

Blue tourne aujourd'hui dans PPO.exe : le monde ECS (`Framework/Simulation/Env`)
avance dans son propre fil, les rouges sont des agents RLMission, et un seul
agent ECS, `GeneralizedPatrolMappoAgent`, mêle la simulation de l'équipe bleue
(observations, entrée du critique, rôles et succession du leader, séquenceur de
patrouille, récompenses, fin d'épisode) et l'apprentissage (HostedTrainer de
PPOKit avec les réseaux rl_tools de Blue, campagnes d'évaluation, promotion
`prod`, checkpoints V7/V10). Rien de ce comportement ne doit changer.

1. **Monde pas à pas** (fait). `Environment` gagne une avance synchrone, sans
   fil : `BeginStep()` (commandes, règles avant le pas, réinitialisation si le
   pas précédent a fini l'épisode ; le monde est alors observable) et
   `EndStep()` (choix des agents, physique, résolution, retour d'état, règles
   après le pas, snapshot), qui dit si un agent a fini l'épisode. La boucle du
   fil exécute les deux mêmes moitiés : un seul code.
2. **Interfaces multi-agents** (fait). ABI environnement 3 : `agentCapacity`
   lignes d'agents dans chaque tampon (observations, actions, récompenses,
   canaux par agent), `available` (un octet par agent présent), une récompense
   par agent en plus de ses canaux (Blue borne la somme), et `begin_step`
   avant chaque pas d'un épisode sauf le premier : un monde dont les règles
   d'avant pas changent l'état réécrit ce sur quoi les agents agissent
   (l'observation de bootstrap et celle de l'action suivante diffèrent chez
   Blue). Le critique centralisé reste dans le learner : il se construit
   depuis les observations et la présence de l'équipe. L'ABI mono-agent en est
   le cas d'un agent (le SDK adapte les classes mono-agent). ABI réseau 2 : un
   learner d'équipe (`act` échantillonné ou déterministe, `observe` qui met à
   jour seul quand son rollout est plein, réglages et sondes pour le journal,
   sauvegarde entre deux rollouts seulement) et une politique d'équipe ;
   `team_sdk.hpp`. ppokit-run suit la capacité (task.json "agents" max) ; un
   arrêt demandé pendant une collecte d'équipe prend effet à l'update suivante.
3. **Blue séparé en deux** (fait). `blue_environment` : le monde ECS de
   BlueDefense chargé depuis scenario.json, les rouges RLMission, l'agent bleu
   en mode `External` (mêmes calculs que Training, sans learner ni campagne).
   `blue_network` : GeneralizedPatrolLearner et l'acteur de production,
   inchangés ; les fichiers V7 et V10 gardent leur format (project.json
   `"checkpoints"` les nomme). Parité : 3 updates au bit près contre le mode
   Training de PPO.exe (BlueDefenseExternalParityTests). Le monde piloté par
   la DLL applique Environment.json (`Environment::LoadSettings`, la fonction
   du rappel de `Serializer::Load` de PPO.exe) : sans lui, il garde le pas et
   les limites par défaut de sEnvironmentState (0,02 s au lieu de 0,2 s), et
   deux mondes au même pas faux s'accordent aussi — le test vérifie donc le
   pas et les limites du monde de la DLL. Différence assumée :
   PPO.exe réinitialise son épisode d'entraînement après une campagne dans le
   monde ; ppokit-run joue ses campagnes dans un environnement à part.
4. **Évaluation de Blue en suites** (fait). Métriques d'épisode de Blue dans
   task.json (issue, survie, neutralisation, qualités, score par épisode,
   dont la moyenne est le score de campagne) ; `eval/validation.json` = la
   campagne du fichier d'entraînement (mêmes graines, cas, rythme), BEST au
   meilleur score comme la promotion `prod`. Cas 2 à 8 identiques à PPO.exe,
   cas 1 même graine : PPO.exe crée son premier cas au chargement, son premier
   lidar voit donc déjà le cas, alors que toute autre réinitialisation (dans
   PPO.exe comme dans ppokit-run) suit les règles d'avant pas et garde le
   lidar du dernier état ; au premier pas, seules les 61 valeurs de lidar
   diffèrent.
5. **Vue 2D** (fait). Le monde de Blue (task.json "world") a une ligne par
   emplacement bleu : ce bleu (présence, vie, leader, position, cap, vitesse,
   référence de patrouille) puis la scène que chaque ligne répète (rouges,
   objectif et zone défendue, waypoints du circuit et actif) ; l'ABI
   environnement range le monde par agent comme les observateurs le lisent.
   `blue_visualiser` : observateur `blue_watch@2` (UDP) et vue 2D vue de
   dessus pour ppokit-viewer : ce que la mini-carte tactique de PPO.exe
   dessine (zone défendue, rectangle d'échec, zones interdites, objectif,
   trace de patrouille et son sens, waypoints actif/validés/à venir et leur
   cap, bleus et rouges avec cap, traînée et cône d'engagement, badges L/W k,
   références projetées et de guidage), plus le pseudo-lidar du bleu choisi
   (du leader par défaut, comme la vue 3D le montre pour le mobile suivi) et
   une explosion à chaque destruction ; la vue cadre d'abord le rectangle
   d'échec. Le monde porte ces données (groupes `blue_view`, `red_cones`,
   `lidar`, `zones`, `bounds`, `patrol` de task.json, lus dans l'ECS par
   l'agent). Test bout en bout : ppokit-run joue BluePatrol, la vue reçoit et
   dessine la scène (zone défendue, badges, zones, cadrage). L'inférence du
   projet est un `ppokit-run play` de l'acteur de production (épisodes
   enchaînés au rythme de PPO.exe, Stop du studio après l'épisode en cours) ;
   le studio ouvre ppokit-viewer quand une inférence démarre.
6. **Projet BluePatrol sous ppokit-run** (projet fait : train jusqu'à
   l'update 4000, eval de l'acteur de prod). « Inference Validation » (fait) :
   chaque gate est un `ppokit-run eval --policy` de l'acteur certifié sur une
   suite des graines de test, résultats lus dans le rapport JSON ; le build
   ne fait plus que ppokit-run et les DLL Blue (monde sans graphisme).
   Entraînement par gates (fait : « Restart Progressive Blue Defense »,
   mappo-training-windows, mappo-blue-defense-windows) : chaque bloc est un
   `ppokit-run train --until-update --save-every --max-seconds` ; pendant le
   bloc, la suite `eval/validation.json` prend la campagne du fichier
   d'entraînement (graines, cas, cadence, comme la campagne de PPO.exe ; la CI
   les change), puis retrouve ses octets ; une campagne du bloc
   compte si le journal du bloc l'annonce (`evaluation 'validation' at update
   N`, épisodes `[MAPPO_EVAL_EPISODE]` du bon P/N) et le journal porte son
   résultat (`score.mean`, `mission_success.rate`). La promotion prod/.score
   de PPO.exe (score strictement au-dessus du .score du même P/N, ou aucun)
   passe par le BEST de la suite, vidé avant chaque bloc : il garde la
   meilleure campagne du bloc, promue après le bloc même s'il est interrompu.
   La télémétrie lit le journal (`--journal`, lignes de ce run seulement) ; il
   voyage avec les checkpoints. Reste : les lanceurs (.bat).
7. **Airspace** (RLMission, RLSNA) : même chemin. État des lieux (relevé
   dans le code, rien de porté encore) :
   - l'apprentissage est déjà PPOKit : `TrainingAgent<Model>`
     (Agents/Bindings) apprend sur les graphes `Agents/Mission` et
     `Agents/Interception` (task.json, graph.json) avec le learner de graphes ;
     ses réglages sont `settings/Training/<agent>.json` (PPO, poids et borne
     des récompenses, rollout, cadence des checkpoints) ;
   - un épisode d'entraînement finit quand la fonction de récompense de
     l'agent le dit (`SetDone`) ou quand son rollout est plein (fin forcée,
     transition tronquée) ; les agents de production ne finissent jamais un
     épisode ; la récompense de l'agent est la somme pondérée de ses canaux,
     bornée par `clamp` ;
   - l'évaluation (`Evaluation/evaluator.cpp`) joue les cas de
     `settings/Evaluation/Cases/<agent>.json` dans son propre monde EnTT (pas
     les règles du monde d'Airspace : sa propre résolution des destructions),
     note la campagne (100 × moyenne de cible détruite + Pk principal −
     joueur détruit + durée + bonus des cibles secondaires) et copie
     `eval/policy.ppokit` dans `models/<agent>/prod/` quand la note bat la
     meilleure ;
   - Playground.json n'entraîne que RLMission (l'avion « blue ») ; aucun
     scénario du dépôt n'a d'agent RLSNA en mode training.

   Plan : un mode External de `TrainingAgent` (action reçue, mêmes
   récompenses, canaux du pas, fin par `SetDone`, pas de learner) ; une DLL
   `airspace_environment` (scénario du projet, Environment.json, l'agent
   d'entraînement trouvé après le chargement ; récompense bornée par l'ABI
   d'équipe à un agent comme Blue) ; deux DLL réseau depuis les task.json
   des agents ; des projets `Examples/Airspace/projects/RLMission` et
   `RLSNA` ; la campagne de l'évaluateur en suite (métriques par épisode,
   BEST sur la note) en partageant sa boucle de cas avec la DLL ; un test de
   parité de quelques updates contre `TrainingAgent` ; la vue 2D tactique.
   Décisions de l'utilisateur : la fin d'épisode au rollout plein devient
   une limite de pas de l'environnement, égale à la taille du rollout de
   l'agent (troncature, valeur du critique en bootstrap) ; RLSNA n'a pas de
   scénario d'entraînement : son projet joue et évalue sa politique, sans
   entraînement tant qu'aucun scénario n'est fourni.

   Fait — RLMission sous ppokit-run :
   - mode External de `TrainingAgent` (`IExternalTrainingAgent`) : pas de
     learner, de checkpoint ni d'évaluateur ; l'action vient de l'appelant,
     l'agent publie l'observation suivante, la récompense bornée (`clamp`),
     les canaux pondérés du pas et la fin d'épisode ; choisi avant le
     chargement des réglages (`LoadExternalScenario`) ;
   - DLL `rlmission_environment` (scénario du projet, ABI d'équipe à un agent
     pour la récompense bornée, troncature à la taille du rollout) et DLL
     réseau `rlmission_network` ; projet `Examples/Airspace/projects/RLMission`
     (Playground.json, checkpoints `models/RLMission/training.ppokit` et
     `eval/policy.ppokit` : la promotion vers `prod/` reste explicite ;
     training.json aux valeurs de Training/RLMission.json, poids des
     récompenses lus dans l'environnement) ;
   - parité : sur le modèle jouet des tests de TrainingAgent, une update sur
     les transitions du mode External écrit le même checkpoint que
     l'entraînement dans le monde ; le monde d'Airspace tire ses épisodes de
     `std::random_device` (comme dans PPO.exe), d'où un test de fumée
     (`Airspace_ppokit_run_smoke` : une partie, une update depuis le
     checkpoint converti, sur une copie des dossiers).
   Fait — suites d'évaluation et projet RLSNA :
   - la boucle de cas de l'évaluateur est extraite dans
     `Evaluation/evaluation_campaign.{hpp,cpp}` (`EvaluationCampaign` : monde
     du cas, reset, pas ; `CaseResult` : note d'un cas) ; `Evaluator` et les
     DLL d'environnement la partagent ; les agents de production prennent
     aussi la politique en argument (`ExternalPolicy` : l'action de l'appelant) ;
   - les DLL exposent la campagne en suite : `ResetEvaluation` joue le cas de
     numéro seed, six métriques par épisode (task.json `metrics` : score,
     cible détruite, joueur détruit, Pk principal, cibles secondaires,
     temps écoulé) ; `eval/campaign.json` : seeds 0 à 99, BEST sur
     `score.mean` ;
   - DLL `rlsna_environment` et `rlsna_network`, projet
     `Examples/Airspace/projects/RLSNA` : jeu et évaluation de
     `models/RLSNA/prod/policy.ppokit`, sans entraînement (SetLearning le
     refuse : aucun scénario ne l'entraîne) ;
   - `Serializer::Register` rend un identifiant et `Unregister` retire ses
     rappels (l'environnement et `TrainingAgent` les retirent à leur
     destruction ; les chemins sont résolus à chaque Load/Save) : ppokit-run
     construit plusieurs environnements dans le même processus ;
   - parité : `Airspace_ppokit_run_smoke` évalue la politique eval de chaque
     famille avec `ppokit-run eval` et avec `Evaluator` (dossier de modèles
     temporaire : `prod/` du dépôt jamais écrit) ; les notes sont égales à
     1e-4 près (RLMission 465,708 ; RLSNA 5485,94).
   Fait — vue 2D tactique :
   - les deux DLL écrivent une ligne de monde (`Environment/tactical_world.hpp`,
     task.json `world` : `entities`, 16 entités × présent, camp, type, nord,
     est, cap, détruite, rayon de non-échappement, demi-angle de tir, rôle —
     contrôlée ou cible principale ; `scene`, demi-étendues de la zone de
     travail, nombre d'entités, Pk et missiles de l'avion contrôlé), lue dans
     le monde où joue l'agent (`Environment::SteppedWorld` pour le scénario
     d'entraînement, le monde de la campagne pour un cas) ;
   - DLL `airspace_visualiser` : l'observateur `airspace_watch@1` (port
     47002) et la vue `airspace` de ppokit-viewer, la minimap tactique de
     PPO.exe (zone de travail et cercles de distance, avions, objectifs,
     zones interdites et leur cercle, cônes d'engagement, traces, croix et
     explosions, avion contrôlé cerclé, halo de sa cible) ; branchée dans les
     network.json de RLMission et RLSNA (`AirspaceVisualiserTests` de bout en
     bout, vérification des projets du dépôt) ;
   - correction : le premier épisode d'entraînement de `rlmission_environment`
     partait de l'état brut du scénario (le monde commence dans l'état INIT,
     qui ne réinitialise pas) et finissait à son premier pas ; il commence
     maintenant, comme les suivants, par la réinitialisation du monde (vérifié
     discriminant dans `Airspace_ppokit_run_smoke`).
   L'étape 7 (retrait de l'interface de PPO.exe) est faite (§18.8).

   Fait — conversion des checkpoints LibTorch (sans perte, vérifiée) :
   - graph.json reprend l'architecture torch exacte (attention à clés
     séparées, log-std calculé de l'état et borné [−20, 2], tronc du critique
     partagé avec l'acteur, un Adam joint, `max_gradient_norm` 0.5 et
     `value_coef` du JSON comme l'ancien `ppo.cpp`) ;
   - `Tools/export_legacy_checkpoint.py` relit les `model.pt` dans l'historique
     git (05bcb4f^), `airspace_checkpoint_import` écrit les artefacts et les
     vérifie contre le forward torch de référence : eval → `training.ppokit`
     (poids, moments Adam, pas 8197 RLMission / 3340 RLSNA ; RNG neuf depuis
     la graine) et `eval/policy.ppokit` ; prod → `prod/policy.ppokit` (choix
     de l'utilisateur : les rouges de Blue jouent aussi le graphe). Écart
     relatif maximal 2,3e−6 (critique compris) ;
   - sans aucune piste présente, l'ancien réseau propageait des sentinelles
     −1e9 (sorties ~1e8) ; le graphe le reproduit (audit P0-01, décision de
     l'utilisateur de restaurer la sémantique historique) : `encode_absent`
     sur les encodeurs, `mask_fill` −1e9 sur le max et l'attention de
     RLMission. Les trois artefacts RLMission ont été re-tamponnés
     (`airspace_checkpoint_import --restamp`, mêmes tenseurs octet pour
     octet). Une présence fractionnaire n'est jamais écrite par
     l'environnement (la moyenne historique pondérait par la présence) ;
   - `AirspaceConvertedPolicyTests` : les politiques converties agissent
     comme `policy.txt` (forward C++ historique) sur les cas de référence.

### 18.10 Vue de FlightControl3D (parité avec FlightControl3D.exe)

Tout ce que le rendu de bureau lisait de la simulation dans le
`FlightSnapshot` passe par le monde de la tâche (`Core/flight_world.hpp`,
331 valeurs en neuf groupes : `aircraft`, `airfield`, `deck`, `guidance`,
`mission`, `landing`, `crash`, `episode_rewards`, `diagnostics`). La DLL
d'environnement les remplit ; `flight_watch@2` les envoie avec l'action et les
récompenses du pas ; `flight_visualiser` les décode dans les types de
l'exemple (Runway, AircraftState, Guidance, DeckState…) et dessine avec les
mêmes fonctions de géométrie (FlightControlCore) :
- la scène 3D : mer, crêtes de houle et sillage de Kelvin d'un pont en
  mouvement, flancs et surface de la zone d'atterrissage, piste et marquages,
  cible de guidage et vecteur commandé, avion (modèle, jambes de train,
  marqueur en vue d'ensemble), pose figée 1,5 s après un crash avec
  l'explosion ; trois caméras (poursuite, piste / plan de vol, bout de
  piste), la distance de poursuite à la molette ;
- les étiquettes : seuil, fiche avion (altitude, hauteur piste, vitesse, cap,
  pente), bandeau du pont, CRASH, par une projection identique à celle du
  viewer ;
- le profil d'approche relatif à la piste ;
- le panneau, en onglets : vol (statut, compteurs d'épisodes, cibles, piste et
  pont, atterrissage),
  commandes et récompenses (barres d'épisode), diagnostics de convergence,
  raisons du dernier crash.

Ce que la vue ne peut pas savoir et qui vient de l'apprenant (moyennes PPO par
canal, KL, validations, BEST) est dans le studio, qui lit le journal ; le
panneau le dit. La fenêtre des diagnostics est le dernier épisode complet (la
DLL ne voit pas les updates) ; la colonne d'action est l'action échantillonnée
en entraînement et la moyenne du réseau en inférence (ce que ppokit-run
envoie).

Pour la parité, l'API de vue passe à 2 (`triangle2d`, `triangle3d` vus des
deux côtés, `measureText`) et l'API d'environnement à 4
(`set_learning`) : l'inférence de bureau ne faisait pas avancer le
curriculum, `ppokit-run play` non plus désormais (ni les évaluations) ; seul
`train` le fait. L'inférence des projets T6 et PC7 vole en temps réel
(`--step-seconds 0.1`, la durée simulée d'une action) jusqu'au Stop du studio.
