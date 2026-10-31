\mainpage PPOTrial — documentation complète du code

[TOC]

## Objectif

Cette documentation décrit le **code tel qu'il existe**, sans changer la simulation, les agents, les récompenses ou les poids. Le référentiel de lecture est le commit `e003f0c5235057470db824b5d9bf12b23d6e64c8` (31 octobre 2026). Les pages d'architecture sont complémentaires des **commentaires Doxygen présents dans les sources `.cpp` et `.hpp`**, et de la référence HTML générée depuis leurs implémentations.

L'ancienne configuration Doxygen analysait des dossiers qui n'existent plus et masquait les symboles sans commentaires. La nouvelle configuration analyse le dépôt depuis sa racine, inclut **165 fichiers C++**, leurs corps de fonctions, les symboles privés et statiques, les méthodes locales, les références et les graphes d'appels si Graphviz est présent.

## Parcours de lecture

- \ref architecture_reference "Architecture et dépendances des modules"
- \ref blue_mappo_reference "BlueDefense : fonctionnement complet du MAPPO"
- \ref legacy_rl_reference "PPO classique et agents Airspace historiques"
- \ref config_ci_reference "Configuration, exécution, CI et tests"
- \ref json_formats "Contrats JSON actifs"
- \ref source_inventory "Inventaire de chaque fichier C++ et des scripts"
- **Files / File List** : fonctions et fichiers `.cpp` jusqu'à leurs implémentations.
- **Classes / Namespaces / Search** : signatures, champs, appels et usages.

## Construction autonome (sans compiler l'application)

Sur Windows, installer Doxygen et facultativement Graphviz, puis exécuter `generatedoc.bat` à la racine du dépôt (ou `tools/generatedoc.ps1 -NoOpen`). Le script exécute Doxygen indépendamment de CMake, LibTorch et RLTools.

Résultat : `docs/generated/html/index.html`. Les avertissements (dont API sans commentaires) sont dans `docs/generated/doxygen-warnings.log`. Ils sont intentionnellement **non bloquants** : la référence doit rester consultable même avec des lacunes documentaires.

## Comprendre ce que « complet » signifie

Doxygen extrait toutes les déclarations et définitions qu'il reconnaît et fournit les sources navigables. Pour les fonctions sans commentaires historiques, le corps du code est affiché; une documentation contractuelle rédigée de chaque ligne ne peut pas être inférée de façon fiable par l'outil. Les pages suivantes explicitent les cheminements importants et signalent cette limite. Des commentaires Doxygen ont été ajoutés dans les fichiers `.cpp`, `.hpp` et les sources des tests lorsqu'ils en manquaient, sans modifier aucune instruction C++, signature, constante ni contrôle de flux. Les 37 sources déjà suffisamment commentées sont conservées telles quelles. Au total, 128 sources reçoivent des annotations, dont 443 fonctions et 58 types supplémentaires. Aucun CMake applicatif, JSON opérationnel, modèle ou checkpoint n'est modifié.
