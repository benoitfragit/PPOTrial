\page legacy_rl_reference Airspace — démonstrateur PPO 2D

[TOC]

## Projets

**RLMission** est un démonstrateur PPO 2D de sélection de cible. **RLSNA**
est un exemple PPO 2D complémentaire d'interception. Chacun possède un projet
PPOKit dans `Examples/Airspace/projects/`, avec sa tâche, son réseau, ses
paramètres d'entraînement et sa campagne d'évaluation.

Leurs environnements `rlmission_environment` et `rlsna_environment` utilisent
le monde ECS partagé `Examples/Shared/World2D`. Le learner et les politiques
PPOKit sont fournis par `ppokit_graph_network`. Les agents de production
utilisés dans le monde sont définis dans `Examples/Airspace/Agents/Policy` et
leurs poids sont conservés dans `Examples/Airspace/models`.

## Responsabilités

- `Examples/Airspace/Agents/Mission` : logique et observations de RLMission.
- `Examples/Airspace/Agents/Interception` : logique et observations de RLSNA.
- `Examples/Airspace/Agents/Bindings` : adaptation des agents au runner PPOKit.
- `Examples/Airspace/Evaluation` : cas et campagnes d'évaluation.
- `airspace_visualiser` : vue 2D transmise à `ppokit-viewer`.

Blue MAPPO réutilise la politique RLMission de production pour les Rouges ;
l'apprentissage multi-agent des Bleus reste séparé.
