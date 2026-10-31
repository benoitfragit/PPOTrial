\page config_ci_reference Configuration, exécution, CI, tests

[TOC]

## Points d'entrée

`Main/src/main.cpp` accepte la sélection interactive ou les options : `--list-examples`, `--example <id>`, `--check-example`, `--no-gui`, `--max-seconds <1..86400>`. Le mode `--no-gui` impose explicitement `blue-patrol` ou `blue-defense` et n'instancie aucun renderer. `--check-example` valide le bootstrap puis sort; cela ne constitue ni un entraînement ni une évaluation de convergence.

Le mode headless :
1. charge scénario et JSON,
2. lance l'environnement ECS,
3. échantillonne toutes les 2 s les snapshots dans `model/BluePatrol/telemetry/mappo-training.csv`,
4. affiche des heartbeats périodiques (environ 15 s),
5. arrête à la limite `max-seconds` ou lorsque l'entraînement est terminé,
6. exige un checkpoint complet de reprise pour déclarer le job réussi.

Aucune fenêtre graphique n'est ouverte; aucune recompilation entre deux phases du curriculum n'est requise **à l'intérieur d'un même job CI** car le script adapte les JSON avant de relancer l'exécutable. La CI Windows compile toujours le binaire au début du job selon sa configuration.

## Configuration et autorité des valeurs

| Configuration | Source | Lecteur / responsabilité |
| --- | --- | --- |
| Monde, limites, `timeStep` | `Environment.json` | `Environment`, `ReadParameters` et initialisation d'épisode. |
| Paramètres physiques et capacités SNA | `Platforms/Mobile/*.json` | Entités et `FlightModelSystem` / `SNASystem`. |
| Zones | `Platforms/RestrictedZone/*.json` | Entité de zone et pseudo-LiDAR. |
| P/N, objectifs, rouges, zones et spawn | `Scenarios/BlueDefense.json`, `BluePatrol.json` | `LoadScenario`, `ParseSpawnSettings`, `InitializerContext`. |
| Hippodrome, wedge, défenses, poids | `Training/BluePatrol.json` | `patrol_config.hpp`, contrôleur MAPPO et learner. |
| Séquences d'entraînement V1/V2 | `Training/BluePatrolCurriculum*.json` | `tools/ci/mappo_curriculum.py` (orchestration entre exécutions). |
| Profils agents | `AgentProfiles.json` | Catalogue/Meta/Serializer. |
| Affichage | `Renderer.json` | `EnvironmentRenderer`, sans incidence intrinsèque sur PPO. |
| Cas de tests Airspace | `Evaluation/Cases/*.json` | Évaluateur historique. |

Les JSON opérationnels sont détaillés dans \ref json_formats. Ne pas confondre les attributs du monde et les paramètres de l'optimiseur. Par exemple `rolloutSteps` contrôle le nombre de pas d'équipe enregistrés; `maxUpdates` borne le nombre de mises à jour PPO; `maxSteps` limite les pas par évaluation.

## CI et curriculum

`.github/workflows/mappo-training-windows.yml` est déclenchable manuellement. Entrées : `scenario`, `curriculum_enabled`, `curriculum_version`, `max_seconds`, `save_every_updates` et `resume_run_id`. Il construit le binaire Release Windows, permet la reprise à partir d'un artefact précédent et publie checkpoints / télémétrie.

`tools/ci/mappo_curriculum.py` orchestre les étapes sans changer le binaire. La V2 présente **neuf étapes** avec un budget de **1000 mises à jour**, par tranches 24 sur la phase courante puis 6 sur les phases précédentes terminées lorsque le réentraînement est applicable. Elle exige un progrès de curriculum V2 et un checkpoint V6 cohérents lors de la reprise; une simple reprise d'un run V1 ne constitue pas un état V2 valide.

`tools/ci/mappo_metrics.py` traite les mesures produites; `test_mappo_curriculum.py` et `test_mappo_metrics.py` assurent les régressions du scripting CI. Les parcours CI Windows sont distincts des tests de cœur C++.

## Navigation des tests

- `Framework/RL/Core/tests` : scheduling neutre, contrats, monitoring.
- `Framework/Simulation/*/tests` : paramètres, ECS, pseudo-LiDAR, sérialisation.
- `Examples/BlueDefense/tests` : apprentissage, masques, géométrie, rewards, collision, spawn, campagne, acteur, smoke ECS.
- `Examples/Airspace/tests` : règles du monde, sélection de pistes, types.
- `Tests/model_checkpoint_tests.cpp` : lecture de modèles historiques sans entraînement.
- `Main/tests`, `Examples/tests` : découverte et sélection des exemples.

## Générer uniquement la documentation

```powershell
# Depuis la racine du dépôt, avec Doxygen installé
.\tools\generatedoc.ps1 -NoOpen
# ou : .\generatedoc.bat
```

Ouvrir `docs/generated/html/index.html`. `docs/generated/doxygen-warnings.log` fournit un inventaire des symboles sans documentation manuelle. Doxygen + Graphviz est recommandé pour les graphes de classes, appels et dépendances; aucune compilation VS2022/LibTorch n'est requise pour ce travail documentaire.

## Validation de couverture

Comparer la page \ref source_inventory avec l'arborescence Git : **165 fichiers C++** au commit indiqué. Inspecter particulièrement les entrées `.cpp` sous **Files**; leurs corps sont visibles avec `SOURCE_BROWSER=YES` et `INLINE_SOURCES=YES`. Une documentation générée n'est **pas** une preuve que le binaire compile, ni que le modèle converge, ni que chaque commentaire existant est exact.
