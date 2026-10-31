\page legacy_rl_reference RL classique et exemple Airspace

[TOC]

## Deux piles d'apprentissage

**Airspace (historique)** : `RLSNA` et `RLMission` utilisent les wrappers d'agents et de réseaux fondés sur **LibTorch**. Les sources comprennent `Framework/RL/PPO/src/ppo.cpp`, `rlagent.cpp`, `torchlearner.cpp`, `Framework/RL/Bindings/data/trainingagent_def.hpp`, `prodagent_def.hpp`, `Examples/Airspace/Agents/Interception/*` et `Agents/Mission/*`.

**BlueDefense** : acteur partagé, critic centralisé et optimisation RLTools sont codés dans `Examples/BlueDefense/Agents/BluePatrol/Training/*`. Ces deux piles ne doivent pas être confondues : en BlueDefense, **les Rouges** peuvent utiliser les politiques de production RLMission existantes alors que **les Bleus** sont entraînés avec MAPPO RLTools.

## Contrats réutilisés

`Framework/RL/Core/include/ppo/core` définit les abstractions `IEnvironment`, `ILearner`, `IPolicy`, `EpisodeRunner`, `ResetAgents`, `StepAgents`, `RunEvaluation`, les buffers et snapshots. Il ne contient ni règles SNA ni récompenses BlueDefense; il orchestre les callbacks et leurs transactions.

`Framework/RL/Runtime/include/ppo/runtime` maintient les états `IAgent` et paramètres PPO compatibles avec le desktop. `Framework/RL/Bindings` adapte les modèles LibTorch aux contrats de formation et d'inférence sans changer l'ordre des features ni la politique rouge.

## RLSNA

`Examples/Airspace/Agents/Interception/Interception.cpp` porte l'agent d'interception, `actorcriticnet.cpp` le réseau associé. `Framework/Simulation/World2D/src/snasystem.cpp` et `snacontrolsystem.cpp` gèrent le domaine SNA et la commande physique. Les systèmes de monde déterminent l'effet physique réel après `ChooseAction`.

## RLMission

`Examples/Airspace/Agents/Mission/targetselector.cpp` définit l'agent de sélection de cible, `actorcriticmission.cpp` le réseau historique, `missionlayout.hpp` les dimensions/layout et `missiontrackfilter.hpp` le filtrage des pistes. Les zones interdites et leurs rayons n'occupent pas à tort des pistes « hostile Mobile » RLMission. Les règles physiques restent dans le simulateur, non dans le PPO.

## Évaluation / interface héritée

`Examples/Airspace/Evaluation/evaluator.cpp` orchestre les cas d'évaluation historiques; `Examples/Airspace/Renderer/environmentrenderer.cpp` fournit l'affichage interactif et les états de contrôle. `Examples/Airspace/Scenario` construit le monde via les composants communs. Les tests dédiés sont dans `Examples/Airspace/tests`.

## Invariants importants

Aucune modification à RLMission, ses poids, les dimensions de ses observations ni le contrat de sauvegarde n'est nécessaire pour documenter BlueDefense. La recherche Doxygen indexe séparément l'API historique et l'API généralisée.
