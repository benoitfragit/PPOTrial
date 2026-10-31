\page architecture_reference Architecture et dépendances du dépôt

[TOC]

## Cartographie

Le dépôt est une **application C++17** avec un socle RL neutre, un adaptateur LibTorch PPO historique, un simulateur ECS, des exemples et le programme desktop/headless.

~~~text
Main/src/main.cpp
  |
  +--> ExampleCatalog / Scenario loader ------------------------------+
  |                                                                  |
  +--> Environment (thread ECS, commandes, snapshot)                 |
  |      |                                                           |
  |      +--> Entity::Init / AdvanceLivingEntities                    |
  |      +--> ppo::core::ResetAgents / StepAgents                     |
  |      +--> IWorldRules::BeforeStep / Resolve / AfterStep           |
  |      |                                                           |
  |      +--> BlueDefense runtime ------------------------------------+
  |      |      |--> GeneralizedPatrolMappoAgent (seul agent bleu)
  |      |      |    |--> état / observations / rewards / rôles
  |      |      |    +--> GeneralizedPatrolLearner (RLTools)
  |      |      |          +--> acteur partagé + critic centralisé
  |      |      +--> N agents RLMission rouges en production (LibTorch)
  |      |
  |      +--> Airspace / RLSNA + RLMission historiques
  |
  +--> EnvironmentRenderer (uniquement en mode graphique)
  +--> Serializer / settings
  +--> Monitor CSV / checkpoints (headless)
~~~

Ce diagramme exprime les responsabilités architecturales, pas les appels de chaque fonction. Les graphes d'appels Doxygen sont générés à partir des sources réelles.

## Modules réels

| Répertoire | Responsabilité et frontières |
| --- | --- |
| `Main/` | Entrée CLI, sélecteur d'exemple, interface graphique ou exécution sans GUI, suivi CSV. |
| `Examples/data`, `Examples/src` | Catalogue des exemples; évite de coder le scénario dans l'entrée principale. |
| `Framework/RL/Core` | Contrats agnostiques du domaine, parcours multi-agent, évaluation et orchestration. |
| `Framework/RL/Runtime` | API de compatibilité des agents et structures des états affichés. |
| `Framework/RL/PPO` | Moteur PPO classique dépendant de LibTorch. |
| `Framework/RL/Bindings` | Adaptateurs agent apprenant / production et réseaux torch. |
| `Framework/Simulation/Entities` | Entités, données ECS, composants, utilitaires et progression des éléments vivants. |
| `Framework/Simulation/Env` | Thread d'exécution, contrôle, chargement/scénario, double tampon de snapshots. |
| `Framework/Simulation/World2D` | Mobile, Objective, RestrictedZone, physique, SNA, pseudo-LiDAR. |
| `Framework/Simulation/Meta` | Enregistrement des types et création par réflexion. |
| `Framework/Support/Serialization` | JSON, sérialisation, validation des paramètres d'entraînement. |
| `Framework/Support/Renderer` | Contrat de rendu. |
| `Examples/Airspace` | Exemple historique (RLMission, interception RLSNA, évaluateur, renderer). |
| `Examples/BlueDefense` | Exemples blue-defense/blue-patrol, MAPPO, rôles, règles de monde, évaluations et tests. |
| `Tests/` et `*/tests/` | Tests des contrats, physiques, réseaux, reproductibilité, checkpoints et scénarios. |
| `resources/settings` | Paramètres de scène, véhicules, environnements, récompenses, entraînement. |
| `tools/ci`, `.github/workflows` | Orchestration curriculum, métriques, campagnes Windows et vérifications. |

## Ordre exact d'un pas ECS

Extrait de `Framework/Simulation/Env/src/environment.cpp`, méthode `Environment::Impl::Run` :

1. Application des commandes de la file (`m_dispatcher.update()`).
2. `IWorldRules::BeforeStep`, si règles de monde installées.
3. Si arrêt précédent ou demande de reset : `Init` de toutes les entités, puis `ResetAgents(..., agent.Init())`.
4. `ppo::core::StepAgents` : tous les agents choisissent l'action (`ChooseAction`), les entités vivantes avancent **une fois**, les règles résolvent les effets (`Resolve`), puis tous les agents reçoivent le nouvel état (`SetNewState`).
5. `AfterStep`, publication d'un snapshot et, en desktop uniquement, pause artificielle de 2 ms.

Le renderer ne pilote pas le calcul physique. En headless, le snapshot ECS est publié une fois tous les 256 pas, au lieu de chaque itération, tandis que la physique suit toujours les pas du monde.

## Séparation des responsabilités

- **Physique/destruction** : `Framework/Simulation/World2D` et `BlueDefenseWorldRules`. Une pénalité RL n'est pas une cause de destruction.
- **Rôles et objectifs RL** : `Examples/BlueDefense/Agents/BluePatrol/Core`.
- **Lecture ECS et interface IAgent** : `Runtime/generalized_patrol_team_agent.*`.
- **Apprentissage et paramètres optimisables** : `Training/*` (RLTools côté bleu).
- **Évaluation et promotion** : `Evaluation/*` + méthodes du contrôleur bleu.
- **RLMission rouge** : `Examples/Airspace/Agents/Mission` et adaptateurs LibTorch; **ne pas confondre** avec l'acteur MAPPO bleu.
- **Préservation des artefacts** : fichiers `latest` pour la reprise, `prod` pour la politique promue.

## Points de lecture critiques

`Main/src/main.cpp` -> `Examples/BlueDefense/Scenario/bluedefenseexample.cpp` -> `Framework/Simulation/Env/src/environment.cpp` -> `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp` -> `Training/generalized_patrol_learner.hpp` -> `Core/generalized_cooperative_reward.hpp` -> `Evaluation/generalized_evaluation_campaign.hpp`.

Pour un diagnostic de frontière, LiDAR ou vitesse, consulter `bluedefenserules.cpp`, `generalized_reward_shaping.hpp`, `generalized_runtime_state.hpp`, `restrictedzonelidar.cpp`, `flightmodelsystem.cpp`.
