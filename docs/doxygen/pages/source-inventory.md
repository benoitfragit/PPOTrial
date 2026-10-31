\page source_inventory Inventaire exhaustif des sources C++

[TOC]

## Contrat de couverture

Inventaire établi à partir de l'arbre Git **e003f0c5235057470db824b5d9bf12b23d6e64c8** : **165 fichiers C/C++** (66 .cpp, 99 .hpp). Chaque fichier figure une fois ci-dessous. Les titres de rôle sont un index d'orientation, pas une spécification exhaustive du comportement; les symboles et les corps de fonctions sont extraits des vrais fichiers par Doxygen.

Pour consulter chaque implémentation, ouvrir **Files > File List > le fichier > Source**, ou utiliser la recherche Doxygen. Les paramètres `EXTRACT_ALL`, `EXTRACT_PRIVATE`, `EXTRACT_STATIC`, `SOURCE_BROWSER` et `INLINE_SOURCES` rendent également les éléments sans commentaires et le contenu des `.cpp` consultables.

## Application / Main (4 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Main/data/exampleselector.hpp` | Catalogue / sélection d'exemple |
| `Main/src/exampleselector.cpp` | Catalogue / sélection d'exemple |
| `Main/src/main.cpp` | Entrée CLI, mode graphique et headless |
| `Main/tests/example_selector_tests.cpp` | Test de non-régression / contrat |

## Framework — contrats RL (29 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Framework/RL/Bindings/data/prodagent.hpp` | Contrat ou implémentation d'agent |
| `Framework/RL/Bindings/data/prodagent_def.hpp` | Contrat ou implémentation d'agent |
| `Framework/RL/Bindings/data/torchproductionpolicy.hpp` | Implémentation / interface du module torchproductionpolicy |
| `Framework/RL/Bindings/data/trainingagent.hpp` | Entraînement et/ou paramètres |
| `Framework/RL/Bindings/data/trainingagent_def.hpp` | Entraînement et/ou paramètres |
| `Framework/RL/Core/include/ppo/core/agent_contract.hpp` | Contrat ou implémentation d'agent |
| `Framework/RL/Core/include/ppo/core/buffers.hpp` | Implémentation / interface du module buffers |
| `Framework/RL/Core/include/ppo/core/environment.hpp` | Boucle d'environnement, commandes et instantanés ECS |
| `Framework/RL/Core/include/ppo/core/evaluation.hpp` | Campagnes, métriques et états d'évaluation |
| `Framework/RL/Core/include/ppo/core/evaluator_state.hpp` | Campagnes, métriques et états d'évaluation |
| `Framework/RL/Core/include/ppo/core/learning_monitor.hpp` | Implémentation / interface du module learning_monitor |
| `Framework/RL/Core/include/ppo/core/multi_agent.hpp` | Contrat ou implémentation d'agent |
| `Framework/RL/Core/include/ppo/core/runner.hpp` | Implémentation / interface du module runner |
| `Framework/RL/Core/include/ppo/core/team_episode.hpp` | Implémentation / interface du module team_episode |
| `Framework/RL/Core/tests/core_tests.cpp` | Test de non-régression / contrat |
| `Framework/RL/Core/tests/learning_monitor_tests.cpp` | Test de non-régression / contrat |
| `Framework/RL/PPO/data/iagent.hpp` | Contrat ou implémentation d'agent |
| `Framework/RL/PPO/data/ppo.hpp` | Algorithme PPO LibTorch historique |
| `Framework/RL/PPO/data/ppoparameters.hpp` | Implémentation / interface du module ppoparameters |
| `Framework/RL/PPO/data/rlagent.hpp` | Agent RL PPO historique |
| `Framework/RL/PPO/data/torchlearner.hpp` | Adaptateur d'entraînement LibTorch |
| `Framework/RL/PPO/data/trainingnet.hpp` | Entraînement et/ou paramètres |
| `Framework/RL/PPO/src/ppo.cpp` | Algorithme PPO LibTorch historique |
| `Framework/RL/PPO/src/rlagent.cpp` | Agent RL PPO historique |
| `Framework/RL/PPO/src/torchlearner.cpp` | Adaptateur d'entraînement LibTorch |
| `Framework/RL/Runtime/include/iagent.hpp` | Contrat ou implémentation d'agent |
| `Framework/RL/Runtime/include/ppo/runtime/iagent.hpp` | Contrat ou implémentation d'agent |
| `Framework/RL/Runtime/include/ppo/runtime/ppoparameters.hpp` | Implémentation / interface du module ppoparameters |
| `Framework/RL/Runtime/include/ppoparameters.hpp` | Implémentation / interface du module ppoparameters |

## Framework — simulateur / ECS (35 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Framework/Simulation/Entities/data/baseentity.hpp` | Modèle de données ECS et entités |
| `Framework/Simulation/Entities/data/components.hpp` | Modèle de données ECS et entités |
| `Framework/Simulation/Entities/data/entity.hpp` | Modèle de données ECS et entités |
| `Framework/Simulation/Entities/data/entityruntime.hpp` | Modèle de données ECS et entités |
| `Framework/Simulation/Entities/data/entitytypes.hpp` | Modèle de données ECS et entités |
| `Framework/Simulation/Entities/data/interface.hpp` | Implémentation / interface du module interface |
| `Framework/Simulation/Entities/data/utils.hpp` | Implémentation / interface du module utils |
| `Framework/Simulation/Entities/src/entity.cpp` | Modèle de données ECS et entités |
| `Framework/Simulation/Entities/src/utils.cpp` | Implémentation / interface du module utils |
| `Framework/Simulation/Env/data/environment.hpp` | Boucle d'environnement, commandes et instantanés ECS |
| `Framework/Simulation/Env/data/scenarioconfig.hpp` | Implémentation / interface du module scenarioconfig |
| `Framework/Simulation/Env/data/worldrules.hpp` | Arbitrage des collisions et de la mission |
| `Framework/Simulation/Env/src/environment.cpp` | Boucle d'environnement, commandes et instantanés ECS |
| `Framework/Simulation/Env/src/scenarioconfig.cpp` | Implémentation / interface du module scenarioconfig |
| `Framework/Simulation/Env/tests/scenario_config_tests.cpp` | Test de non-régression / contrat |
| `Framework/Simulation/Meta/data/metaregistry.hpp` | Implémentation / interface du module metaregistry |
| `Framework/Simulation/Meta/src/metaregistry.cpp` | Implémentation / interface du module metaregistry |
| `Framework/Simulation/Meta/tests/meta_contract_tests.cpp` | Test de non-régression / contrat |
| `Framework/Simulation/World2D/data/flightmodelsystem.hpp` | Intégration cinématique et physique mobile |
| `Framework/Simulation/World2D/data/mobile.hpp` | Modèle cinématique / entité mobile |
| `Framework/Simulation/World2D/data/objective.hpp` | Implémentation / interface du module objective |
| `Framework/Simulation/World2D/data/restrictedzone.hpp` | Implémentation / interface du module restrictedzone |
| `Framework/Simulation/World2D/data/restrictedzonegeometry.hpp` | Implémentation / interface du module restrictedzonegeometry |
| `Framework/Simulation/World2D/data/restrictedzonelidar.hpp` | Rayons pseudo-LiDAR et zones interdites |
| `Framework/Simulation/World2D/data/sharedworldcomponents.hpp` | Modèle de données ECS et entités |
| `Framework/Simulation/World2D/data/snacontrolsystem.hpp` | Implémentation / interface du module snacontrolsystem |
| `Framework/Simulation/World2D/data/snasystem.hpp` | Système d'engagement SNA |
| `Framework/Simulation/World2D/src/flightmodelsystem.cpp` | Intégration cinématique et physique mobile |
| `Framework/Simulation/World2D/src/mobile.cpp` | Modèle cinématique / entité mobile |
| `Framework/Simulation/World2D/src/restrictedzone.cpp` | Implémentation / interface du module restrictedzone |
| `Framework/Simulation/World2D/src/restrictedzonegeometry.cpp` | Implémentation / interface du module restrictedzonegeometry |
| `Framework/Simulation/World2D/src/restrictedzonelidar.cpp` | Rayons pseudo-LiDAR et zones interdites |
| `Framework/Simulation/World2D/src/snacontrolsystem.cpp` | Implémentation / interface du module snacontrolsystem |
| `Framework/Simulation/World2D/src/snasystem.cpp` | Système d'engagement SNA |
| `Framework/Simulation/World2D/tests/restrictedzone_lidar_tests.cpp` | Test de non-régression / contrat |

## Framework — support (7 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Framework/Support/Renderer/data/irenderer.hpp` | Rendu graphique |
| `Framework/Support/Serialization/data/serializer.hpp` | Lecture, validation et sauvegarde des paramètres JSON |
| `Framework/Support/Serialization/data/training_settings.hpp` | Entraînement et/ou paramètres |
| `Framework/Support/Serialization/src/serializer.cpp` | Lecture, validation et sauvegarde des paramètres JSON |
| `Framework/Support/Serialization/src/training_settings.cpp` | Entraînement et/ou paramètres |
| `Framework/Support/Serialization/tests/serializer_safety_tests.cpp` | Test de non-régression / contrat |
| `Framework/Support/Serialization/tests/training_settings_tests.cpp` | Test de non-régression / contrat |

## BlueDefense — MAPPO (55 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Examples/BlueDefense/Agents/BluePatrol/Core/blue_team_roles.hpp` | Leader, équipiers et positions de formation |
| `Examples/BlueDefense/Agents/BluePatrol/Core/defenderbackend.hpp` | Implémentation / interface du module defenderbackend |
| `Examples/BlueDefense/Agents/BluePatrol/Core/generalized_cooperative_reward.hpp` | Canaux, responsabilités et agrégation des récompenses |
| `Examples/BlueDefense/Agents/BluePatrol/Core/generalized_observation.hpp` | Contrat et validation des observations v8 |
| `Examples/BlueDefense/Agents/BluePatrol/Core/generalized_team_dynamics.hpp` | Implémentation / interface du module generalized_team_dynamics |
| `Examples/BlueDefense/Agents/BluePatrol/Core/patrol_config.hpp` | Chargement/validation des paramètres de patrouille |
| `Examples/BlueDefense/Agents/BluePatrol/Core/patrol_core.hpp` | Implémentation / interface du module patrol_core |
| `Examples/BlueDefense/Agents/BluePatrol/Core/patrol_defense.hpp` | Implémentation / interface du module patrol_defense |
| `Examples/BlueDefense/Agents/BluePatrol/Core/patrol_geometry.hpp` | Géométrie et projection de l'hippodrome |
| `Examples/BlueDefense/Agents/BluePatrol/Core/team_capacity.hpp` | Implémentation / interface du module team_capacity |
| `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp` | Agent MAPPO partagé; observations, actions, retours et évaluation |
| `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.hpp` | Agent MAPPO partagé; observations, actions, retours et évaluation |
| `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_reward_shaping.hpp` | Construction ou validation des récompenses |
| `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_runtime_state.hpp` | État ECS normalisé et fabrication des observations |
| `Examples/BlueDefense/Agents/BluePatrol/Training/attention_actor.hpp` | Architecture ou calcul de réseau de politique/valeur |
| `Examples/BlueDefense/Agents/BluePatrol/Training/generalized_advantage.hpp` | Implémentation / interface du module generalized_advantage |
| `Examples/BlueDefense/Agents/BluePatrol/Training/generalized_attention_actor.hpp` | Acteur partagé et attention masquée |
| `Examples/BlueDefense/Agents/BluePatrol/Training/generalized_checkpoint.hpp` | Format de checkpoint de l'acteur |
| `Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_network.hpp` | Architecture ou calcul de réseau de politique/valeur |
| `Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_observation.hpp` | Architecture ou calcul de réseau de politique/valeur |
| `Examples/BlueDefense/Agents/BluePatrol/Training/generalized_patrol_learner.hpp` | Collecte P-agent, optimiseur PPO et états d'entraînement |
| `Examples/BlueDefense/Agents/BluePatrol/Training/generalized_training_checkpoint.hpp` | Sérialisation des états de formation |
| `Examples/BlueDefense/Agents/BluePatrol/Training/patrol_checkpoint.hpp` | Chargement/sauvegarde et compatibilité des états |
| `Examples/BlueDefense/Agents/BluePatrol/Training/patrol_rltools.hpp` | Implémentation / interface du module patrol_rltools |
| `Examples/BlueDefense/Agents/BluePatrol/Training/patrol_training_checkpoint.hpp` | Chargement/sauvegarde et compatibilité des états |
| `Examples/BlueDefense/Agents/BluePatrol/Training/patrol_training_settings.hpp` | Entraînement et/ou paramètres |
| `Examples/BlueDefense/Evaluation/generalized_evaluation_campaign.hpp` | Ordonnancement de campagne d'évaluation |
| `Examples/BlueDefense/Evaluation/generalized_evaluation_metrics.hpp` | Agrégation et notation de campagne |
| `Examples/BlueDefense/Evaluation/patrol_evaluation_settings.hpp` | Campagnes, métriques et états d'évaluation |
| `Examples/BlueDefense/Scenario/bluedefenseexample.cpp` | Chargement du scénario BlueDefense et branchement du contrôleur |
| `Examples/BlueDefense/Scenario/bluedefenseexample.hpp` | Chargement du scénario BlueDefense et branchement du contrôleur |
| `Examples/BlueDefense/Scenario/bluedefenserules.cpp` | Règles de destruction, collisions et activation |
| `Examples/BlueDefense/Scenario/bluedefenserules.hpp` | Règles de destruction, collisions et activation |
| `Examples/BlueDefense/Scenario/initializer.cpp` | Initialisation des épisodes et placement des entités |
| `Examples/BlueDefense/Scenario/initializer.hpp` | Initialisation des épisodes et placement des entités |
| `Examples/BlueDefense/Scenario/spawn.cpp` | Initialisation des épisodes et placement des entités |
| `Examples/BlueDefense/Scenario/spawn.hpp` | Initialisation des épisodes et placement des entités |
| `Examples/BlueDefense/Scenario/spawn_config.cpp` | Initialisation des épisodes et placement des entités |
| `Examples/BlueDefense/Scenario/spawn_config.hpp` | Initialisation des épisodes et placement des entités |
| `Examples/BlueDefense/tests/attention_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/collision_rules_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/generalized_actor_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/generalized_advantage_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/generalized_initializer_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/generalized_learner_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/generalized_reward_evaluation_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/generalized_reward_shaping_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/generalized_runtime_state_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/patrol_defense_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/patrol_geometry_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/patrol_runtime_smoke.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/restricted_zone_snapshot_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/spawn_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/team_domain_tests.cpp` | Test de non-régression / contrat |
| `Examples/BlueDefense/tests/team_dynamics_tests.cpp` | Test de non-régression / contrat |

## Airspace — RLMission et RLSNA historiques (30 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Examples/Airspace/Agents/Interception/Interception.cpp` | Implémentation / interface du module Interception |
| `Examples/Airspace/Agents/Interception/Interception.hpp` | Implémentation / interface du module Interception |
| `Examples/Airspace/Agents/Interception/actorcriticnet.cpp` | Architecture ou calcul de réseau de politique/valeur |
| `Examples/Airspace/Agents/Interception/actorcriticnet.hpp` | Architecture ou calcul de réseau de politique/valeur |
| `Examples/Airspace/Agents/Mission/actorcriticmission.cpp` | Architecture ou calcul de réseau de politique/valeur |
| `Examples/Airspace/Agents/Mission/actorcriticmission.hpp` | Architecture ou calcul de réseau de politique/valeur |
| `Examples/Airspace/Agents/Mission/missionlayout.hpp` | Disposition des observations RLMission |
| `Examples/Airspace/Agents/Mission/missiontrackfilter.hpp` | Implémentation / interface du module missiontrackfilter |
| `Examples/Airspace/Agents/Mission/targetselector.cpp` | Contrôleur rouge RLMission / choix de cible |
| `Examples/Airspace/Agents/Mission/targetselector.hpp` | Contrôleur rouge RLMission / choix de cible |
| `Examples/Airspace/Evaluation/evaluator.cpp` | Campagnes, métriques et états d'évaluation |
| `Examples/Airspace/Evaluation/evaluator.hpp` | Campagnes, métriques et états d'évaluation |
| `Examples/Airspace/Renderer/environmentrenderer.cpp` | Rendu graphique |
| `Examples/Airspace/Renderer/environmentrenderer.hpp` | Rendu graphique |
| `Examples/Airspace/Scenario/Entities/data/aircraft.hpp` | Implémentation / interface du module aircraft |
| `Examples/Airspace/Scenario/Entities/data/airspacecomponents.hpp` | Modèle de données ECS et entités |
| `Examples/Airspace/Scenario/Entities/data/sam.hpp` | Implémentation / interface du module sam |
| `Examples/Airspace/Scenario/Initialization/data/checkpointinitializersystem.hpp` | Chargement/sauvegarde et compatibilité des états |
| `Examples/Airspace/Scenario/Initialization/data/randominitializersystem.hpp` | Initialisation des épisodes et placement des entités |
| `Examples/Airspace/Scenario/Initialization/src/checkpointinitializersystem.cpp` | Chargement/sauvegarde et compatibilité des états |
| `Examples/Airspace/Scenario/Initialization/src/randominitializersystem.cpp` | Initialisation des épisodes et placement des entités |
| `Examples/Airspace/Scenario/airspaceexample.cpp` | Implémentation / interface du module airspaceexample |
| `Examples/Airspace/Scenario/airspaceexample.hpp` | Implémentation / interface du module airspaceexample |
| `Examples/Airspace/Scenario/airspacemeta.cpp` | Implémentation / interface du module airspacemeta |
| `Examples/Airspace/Scenario/airspacemeta.hpp` | Implémentation / interface du module airspacemeta |
| `Examples/Airspace/Scenario/airspaceworldrules.cpp` | Arbitrage des collisions et de la mission |
| `Examples/Airspace/Scenario/airspaceworldrules.hpp` | Arbitrage des collisions et de la mission |
| `Examples/Airspace/tests/airspace_meta_tests.cpp` | Test de non-régression / contrat |
| `Examples/Airspace/tests/airspace_world_rules_tests.cpp` | Test de non-régression / contrat |
| `Examples/Airspace/tests/mission_track_selection_tests.cpp` | Test de non-régression / contrat |

## Catalogue et tests des exemples (4 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Examples/data/example.hpp` | Implémentation / interface du module example |
| `Examples/data/examplecatalog.hpp` | Catalogue / sélection d'exemple |
| `Examples/src/examplecatalog.cpp` | Catalogue / sélection d'exemple |
| `Examples/tests/examplecatalog_tests.cpp` | Test de non-régression / contrat |

## Tests racine (1 fichiers)

| Fichier (chemin relatif) | Rôle / point d'entrée |
| --- | --- |
| `Tests/model_checkpoint_tests.cpp` | Test de non-régression / contrat |

## Hors C++ — configurations, génération, automatisation

Les fichiers suivants ne sont **pas des symboles C++**. Ils sont inventoriés pour garantir la traçabilité et commentés dans \ref config_ci_reference et \ref json_formats. Les JSON ne sont pas présentés à tort comme des interfaces compilées.

### Fichiers JSON (25)

- `CMakeSettings.json`
- `resources/settings/AgentProfiles.json`
- `resources/settings/Environment.json`
- `resources/settings/Evaluation/Cases/RLMission.json`
- `resources/settings/Evaluation/Cases/RLSNA.json`
- `resources/settings/Evaluation/InitialStates.json`
- `resources/settings/Platforms/Mobile/aircraft.json`
- `resources/settings/Platforms/Mobile/attacker_car.json`
- `resources/settings/Platforms/Mobile/defender_car.json`
- `resources/settings/Platforms/Mobile/evalaircraft.json`
- `resources/settings/Platforms/Mobile/mobile.json`
- `resources/settings/Platforms/Mobile/trainingaircraft.json`
- `resources/settings/Platforms/RestrictedZone/restrictedzone.json`
- `resources/settings/Platforms/RestrictedZone/sam.json`
- `resources/settings/Renderer.json`
- `resources/settings/Scenarios/BlueDefense.json`
- `resources/settings/Scenarios/Evaluation/RLMission.json`
- `resources/settings/Scenarios/Evaluation/RLSNA.json`
- `resources/settings/Scenarios/Playground.json`
- `resources/settings/Training/BluePatrol.json`
- `resources/settings/Training/BluePatrolCurriculum.json`
- `resources/settings/Training/BluePatrolCurriculumV2.json`
- `resources/settings/Training/RLMission.json`
- `resources/settings/Training/RLSNA.json`

### CMake / scripts / CI (34)

- `.github/workflows/core-tests.yml`
- `.github/workflows/desktop-windows.yml`
- `.github/workflows/mappo-training-windows.yml`
- `CMakeLists.txt`
- `Examples/Airspace/Agents/Interception/CMakeLists.txt`
- `Examples/Airspace/Agents/Mission/CMakeLists.txt`
- `Examples/Airspace/CMakeLists.txt`
- `Examples/Airspace/Evaluation/CMakeLists.txt`
- `Examples/Airspace/Renderer/CMakeLists.txt`
- `Examples/Airspace/Scenario/Entities/CMakeLists.txt`
- `Examples/Airspace/Scenario/Initialization/CMakeLists.txt`
- `Examples/BlueDefense/Agents/BluePatrol/CMakeLists.txt`
- `Examples/BlueDefense/CMakeLists.txt`
- `Examples/BlueDefense/tests/CMakeLists.txt`
- `Examples/CMakeLists.txt`
- `Framework/RL/Bindings/CMakeLists.txt`
- `Framework/RL/Core/CMakeLists.txt`
- `Framework/RL/PPO/CMakeLists.txt`
- `Framework/RL/Runtime/CMakeLists.txt`
- `Framework/Simulation/Entities/CMakeLists.txt`
- `Framework/Simulation/Env/CMakeLists.txt`
- `Framework/Simulation/Meta/CMakeLists.txt`
- `Framework/Simulation/World2D/CMakeLists.txt`
- `Framework/Support/Renderer/CMakeLists.txt`
- `Framework/Support/Serialization/CMakeLists.txt`
- `Main/CMakeLists.txt`
- `Tests/CMakeLists.txt`
- `dependencies.cmake`
- `generatedoc.bat`
- `tools/ci/mappo_curriculum.py`
- `tools/ci/mappo_metrics.py`
- `tools/ci/test_mappo_curriculum.py`
- `tools/ci/test_mappo_metrics.py`
- `tools/generatedoc.ps1`

## Limite explicite

Doxygen extrait la signature et le code de tous les symboles C++ accessibles à son analyseur, y compris ceux sans documentation manuelle. Cela **ne prouve pas** que chacun de leurs contrats fonctionnels, préconditions et effets secondaires possède déjà une explication humaine individualisée. Le présent guide d'architecture décrit les parcours critiques; la liste ci-dessus rend les lacunes auditables sans modifier les sources.
