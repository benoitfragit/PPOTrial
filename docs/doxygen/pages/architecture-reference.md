\page architecture_reference Architecture et dépendances du dépôt

[TOC]

## Cartographie

Le dépôt comprend un framework C++17 et trois démonstrateurs : **RLMission**
(PPO 2D), **Blue MAPPO** (MA-PPO 2D) et **FlightControl3D** (PPO 3D).
PPOKit fournit le runner, le studio, les réseaux et les outils de visualisation.

~~~text
ppokit-run train|eval|play --project <dossier>   (lancé à la main ou par ppokit-studio)
  |
  +--> project.json : DLL d'environnement + DLL réseau, checkpoints, journal
  |
  +--> DLL d'environnement (blue_environment, rlmission_environment, ...)
  |      |
  |      +--> Scenario loader + Environment (BeginStep / EndStep)
  |      |      +--> Entity::Init / AdvanceLivingEntities
  |      |      +--> ppo::core::ResetAgents / StepAgents
  |      |      +--> IWorldRules::BeforeStep / Resolve / AfterStep
  |      |
  |      +--> BlueDefense runtime
  |      |      |--> GeneralizedPatrolMappoAgent (seul agent bleu, piloté de l'extérieur)
  |      |      |    |--> état / observations / rewards / rôles
  |      |      +--> N agents RLMission rouges en production
  |      |
  |      +--> Airspace / RLSNA + RLMission
  |
  +--> DLL réseau (ppokit_graph_network : PPO ou MA-PPO selon le projet)
  +--> Observateurs de network.json (Visualiseur) --UDP--> ppokit-viewer
  +--> Serializer / settings
  +--> checkpoints / metrics.jsonl / campagnes eval/*.json
~~~

Ce diagramme exprime les responsabilités architecturales, pas les appels de chaque fonction.

## Modules réels

| Répertoire | Responsabilité et frontières |
| --- | --- |
| `PPOKit/` | Framework commun : `ppokit-run` (train / eval / play d'un projet), `ppokit-viewer` (vues des DLL Visualiseur reçues en UDP), `ppokit-studio` (ouvre les projets et lance leurs commandes). |
| `Framework/RL/Core` | Contrats agnostiques du domaine (`Observation`, `Action`, `IPolicy`, descripteurs d'agents, buffers) et parcours multi-agent (`ResetAgents` / `StepAgents`). |
| `Framework/RL/Runtime` | Interface `IAgent` des agents du monde ECS (`Init`, `ChooseAction`, `SetNewState`). |
| `PPOKit/include/ppokit/graph`, `PPOKit/include/ppokit/runtime` | Learners PPO et MA-PPO et collecte des trajectoires. |
| `Examples/Airspace/Agents/Bindings` | Adaptateurs agent apprenant / production des agents rouges (learners PPOKit). |
| `Framework/Simulation/Entities` | Entités, données ECS, composants, utilitaires et progression des éléments vivants. |
| `Framework/Simulation/Env` | Monde ECS avancé par son appelant (`BeginStep` / `EndStep`), entités, agents, règles de monde et réglages `Environment.json`. |
| `Examples/Shared/World2D` | Mobile, Objective, RestrictedZone, physique, SNA, pseudo-LiDAR. |
| `Framework/Simulation/Meta` | Tables de fabriques des scénarios (`RegisterEntityFactory` / `RegisterAgentFactory`, `CreateEntity` / `CreateAgent`). |
| `Framework/Support/Serialization` | Chargement JSON (lecture seule), validation des paramètres d'entraînement. |
| `Examples/Airspace` | Démonstrateur PPO 2D RLMission et exemple d'interception RLSNA, avec le Visualiseur `airspace_visualiser`. |
| `Examples/BlueDefense` | Démonstrateur MA-PPO 2D BluePatrol : `blue_environment`, réseau partagé, `blue_visualiser`, rôles et règles de monde. |
| `Examples/FlightControl3D` | Démonstrateur PPO 3D de vol JSBSim : projets T-6 et PC-7 et Visualiseur `flight_visualiser`. |
| `*/tests/` | Tests des contrats, physiques, réseaux, reproductibilité, checkpoints et scénarios. |
| `Examples/Shared/settings`, `Examples/<Nom>/settings`, `Examples/<Nom>/projects` | Paramètres de scène, véhicules, environnements, récompenses, entraînement. |
| `tools/ci`, `.github/workflows` | Orchestration curriculum, métriques, campagnes Windows et vérifications. |

## Ordre exact d'un pas ECS

Extrait de `Framework/Simulation/Env/src/environment.cpp`, méthodes `Environment::BeginStep` et `Environment::EndStep` :

1. `BeginStep` : `IWorldRules::BeforeStep`, si règles de monde installées.
2. `BeginStep` : si le pas précédent a terminé l'épisode ou si une réinitialisation est demandée (`RequestReinitialization`) : `Init` de toutes les entités, puis `ResetAgents(..., agent.Init())`. Le monde peut alors être observé (`SteppedWorld`).
3. `EndStep` : `ppo::core::StepAgents` : tous les agents choisissent l'action (`ChooseAction`), les entités vivantes avancent **une fois**, les règles résolvent les effets (`Resolve`), puis tous les agents reçoivent le nouvel état (`SetNewState`).
4. `EndStep` : `AfterStep` ; la valeur rendue indique si l'épisode est terminé.

L'`Environment` n'a ni thread propre, ni snapshots, ni pause : les DLL d'environnement le font avancer depuis le thread de `ppokit-run` en ces deux moitiés et lisent le monde entre elles. Les vues (ppokit-viewer) ne pilotent pas le calcul physique : elles lisent ce que les observateurs envoient après chaque pas.

## Séparation des responsabilités

- **Physique/destruction** : `Examples/Shared/World2D` et `BlueDefenseWorldRules`. Une pénalité RL n'est pas une cause de destruction.
- **Rôles et objectifs RL** : `Examples/BlueDefense/Agents/BluePatrol/Core`.
- **Lecture ECS et interface IAgent** : `Runtime/generalized_patrol_team_agent.*`.
- **Apprentissage et paramètres optimisables** : learners PPOKit ; le backend Blue historique reste utilisé par la CI et les comparaisons.
- **Évaluation et promotion** : cas et score par épisode dans `Evaluation/*` (`generalized_evaluation_case.hpp`, `generalized_evaluation_metrics.hpp`) ; suites, agrégation et promotion dans `ppokit-run`.
- **RLMission rouge** : `Examples/Airspace/Agents/Mission` et ses adaptateurs PPOKit.
- **Préservation des artefacts** : fichiers `latest` pour la reprise, `prod` pour la politique promue.

## Points de lecture critiques

`PPOKit/tools/ppokit_run.cpp` -> `Examples/BlueDefense/Environment/blue_environment.cpp` -> `Examples/BlueDefense/Scenario/bluedefenseexample.cpp` -> `Framework/Simulation/Env/src/environment.cpp` -> `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp` -> `Core/generalized_cooperative_reward.hpp` -> `Evaluation/generalized_evaluation_case.hpp`.

Pour un diagnostic de frontière, LiDAR ou vitesse, consulter `bluedefenserules.cpp`, `generalized_reward_shaping.hpp`, `generalized_runtime_state.hpp`, `restrictedzonelidar.cpp`, `flightmodelsystem.cpp`.
