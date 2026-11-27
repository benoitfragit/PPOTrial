\mainpage PPOTrial — documentation complète du code

[TOC]

## Objectif

Cette documentation accompagne les sources C++17 et les projets PPOKit :

- **RLMission** : démonstrateur PPO 2D.
- **Blue MAPPO** : démonstrateur MA-PPO 2D.
- **FlightControl3D** : démonstrateur PPO 3D.

Les pages d'architecture complètent les commentaires Doxygen des sources. La
référence générée comprend les corps de fonctions. Les diagrammes d'architecture
sont maintenus dans les pages dédiées ; les graphes automatiques ne sont pas
générés par défaut.

## Parcours de lecture

- \ref architecture_reference "Architecture et dépendances des modules"
- \ref blue_mappo_reference "BlueDefense : fonctionnement complet du MAPPO"
- \ref legacy_rl_reference "RLMission et RLSNA : PPO 2D"
- \ref config_ci_reference "Configuration, exécution, CI et tests"
- \ref json_formats "Contrats JSON actifs"
- **Files / File List** : fonctions et fichiers `.cpp` jusqu'à leurs implémentations.
- **Classes / Namespaces / Search** : signatures, champs, appels et usages.

## Construction autonome (sans compiler l'application)

Sur Windows, installer Doxygen et facultativement Graphviz, puis exécuter `generatedoc.bat` à la racine du dépôt (ou `tools/generatedoc.ps1 -NoOpen`). Le script exécute Doxygen indépendamment de CMake, LibTorch et RLTools.

Résultat : `docs/generated/html/index.html`. Les avertissements (dont API sans commentaires) sont dans `docs/generated/doxygen-warnings.log`. Ils sont intentionnellement **non bloquants** : la référence doit rester consultable même avec des lacunes documentaires.

## Comprendre ce que « complet » signifie

Doxygen extrait les déclarations et définitions qu'il reconnaît et fournit les
sources navigables. Les pages d'architecture décrivent les responsabilités et
les contrats qui ne se déduisent pas des signatures.
