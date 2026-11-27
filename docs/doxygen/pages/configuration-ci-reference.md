\page config_ci_reference Configuration, exécution, CI, tests

[TOC]

## Points d'entrée

Tout s'exécute sous les programmes PPOKit ; l'ancien hôte `PPO.exe` (et ses options `--example`, `--no-gui`, `--max-seconds`, son sélecteur d'exemples) a été supprimé.

- `ppokit-run train|eval|play --project <dossier>` entraîne, évalue ou rejoue un projet à partir des deux DLL nommées dans son `project.json` (`"libraries"`). Options : `train` `[--updates N | --until-update N] [--save-every N] [--fresh] [--until-passed SUITE] [--evaluate-every N] [--max-seconds S]` ; `eval` `[--suite NAME] [--best | --policy FILE]` ; `play` `[--episodes N] [--best [--suite NAME] | --policy FILE] [--realtime [--step-seconds S]]`. `--max-seconds S` termine l'entraînement à la première frontière d'update après S secondes, comme si c'était la dernière (campagnes comprises). Sans checkpoint à reprendre, ou avec `--fresh`, `train` déplace le run précédent (checkpoint, politique, journal, états d'environnement, BEST des suites) dans `checkpoints/archive/run-<date UTC>/` avant de commencer : rien n'est écrasé et le nouveau run ne part de rien de l'ancien. L'état d'entraînement de l'environnement (par exemple le curriculum de FlightControl3D) est écrit dans `checkpoints/environment_state/<update>.json` avant chaque checkpoint et repris avec lui ; un checkpoint sans état (versions antérieures) garde l'état que l'environnement lit lui-même.
- `ppokit-viewer --view <DLL Visualiseur> --port <port>` affiche les vues 2D/3D d'une DLL Visualiseur à partir des trames UDP envoyées par l'observateur du `network.json` du projet.
- `ppokit-studio [<dossier>]` ouvre un projet ; ses boutons « Train », « Evaluate » et « Inference » lancent les commandes de `project.json`.

Projets : Blue MAPPO 2D dans `Examples/BlueDefense/projects/BluePatrol`,
FlightControl3D PPO 3D dans `Examples/FlightControl3D/projects/T6` et `PC7`,
RLMission PPO 2D dans `Examples/Airspace/projects/RLMission`, et l'exemple
complémentaire RLSNA dans `Examples/Airspace/projects/RLSNA`. Ils utilisent
`ppokit_graph_network` et leurs DLL d'environnement et de visualisation propres.
Le projet Blue historique utilisé par la CI conserve `blue_network`.

Un entraînement `ppokit-run train` n'ouvre aucune fenêtre ; la vue est un processus `ppokit-viewer` séparé. Il écrit ses checkpoints et le journal `metrics.jsonl` du projet ; les campagnes des suites `eval/*.json` marquées `during_training` vont au journal et au store BEST. En CI, `tools/ci/mappo_curriculum.py` lance `ppokit-run train` sur BluePatrol bloc par bloc (`--until-update`, `--save-every`, `--max-seconds`) et exige un checkpoint complet de reprise pour déclarer le job réussi. Aucune recompilation entre deux phases du curriculum n'est requise **à l'intérieur d'un même job CI** car le script adapte les JSON avant de relancer l'exécutable. La CI Windows compile toujours le binaire au début du job selon sa configuration.

## Configuration et autorité des valeurs

| Configuration | Source | Lecteur / responsabilité |
| --- | --- | --- |
| Monde, limites, `timeStep` | `Environment.json` | `Environment::LoadSettings` et initialisation d'épisode. |
| Paramètres physiques et capacités SNA | `Platforms/Mobile/*.json` | Entités et `FlightModelSystem` / `SNASystem`. |
| Zones | `Platforms/RestrictedZone/*.json` | Entité de zone et pseudo-LiDAR. |
| P/N, objectifs, rouges, zones et spawn | `scenario.json` du projet BluePatrol | `LoadScenario`, `ParseSpawnSettings`, `InitializerContext`. |
| Hippodrome, wedge, défenses, poids | `blue_patrol_training.json` du projet BluePatrol | `patrol_config.hpp` et contrôleur MAPPO. |
| PPO et rollout du projet partagé | `training.json` du projet BluePatrol | Learner `ppokit_graph_network`. |
| Séquences d'entraînement V3 et gates CI | `Examples/BlueDefense/curricula/BluePatrolCurriculumV3.json`, `tools/ci/adaptive_curriculum/gates.json` | Lanceurs CI externes, distincts de Train dans Studio. |
| Profils agents | `AgentProfiles.json` | Catalogue/Meta/Serializer. |
| Cas de tests Airspace | `Evaluation/Cases/*.json` | `EvaluationCampaign` (`Examples/Airspace/Evaluation`), jouée par les suites des DLL d'environnement Airspace. |

Les JSON opérationnels sont détaillés dans \ref json_formats. Ne pas confondre les attributs du monde et les paramètres de l'optimiseur. Par exemple `rolloutSteps` contrôle le nombre de pas d'équipe enregistrés; `maxUpdates` borne le nombre de mises à jour PPO; `maxSteps` limite les pas par évaluation.

## CI et curriculum

`.github/workflows/mappo-blue-defense-windows.yml` (« Blue Defense (Windows Release) ») est déclenchable manuellement. Entrées : `curriculum_enabled`, `curriculum_version`, `curriculum_warm_start`, `max_seconds`, `save_every_updates` et `resume_run_id`. Il construit `ppokit-run` et les DLL BlueDefense en Release Windows (sans les programmes desktop), vérifie le runtime par un épisode `ppokit-run play` de l'acteur de production, permet la reprise à partir d'un artefact précédent et publie checkpoints / télémétrie.

`tools/ci/mappo_curriculum.py` orchestre les étapes du curriculum V3 sans
changer le binaire ; le lanceur adaptatif utilise son propre calendrier de
gates. Les deux utilisent le projet Blue historique de la CI et ses checkpoints.

`tools/ci/mappo_metrics.py` traite les mesures produites; `test_mappo_curriculum.py` et `test_mappo_metrics.py` assurent les régressions du scripting CI. Les parcours CI Windows sont distincts des tests de cœur C++.

## Navigation des tests

- `Framework/RL/Core/tests` : scheduling neutre, contrats, monitoring.
- `Framework/Simulation/*/tests` : paramètres, ECS, pseudo-LiDAR, sérialisation.
- `Examples/BlueDefense/tests` : apprentissage, masques, géométrie, rewards, collision, spawn, campagne, acteur, smoke ECS.
- `Examples/Airspace/tests` : règles du monde, sélection de pistes, types.
- `Examples/Airspace/tests/policy_checkpoint_tests.cpp` : lecture de modèles historiques sans entraînement.
- `PPOKit/tests` : projets, `ppokit-run`, artefacts, évaluation, studio.
- `Examples/BlueDefense/tests/visualiser_tests.cpp`, `Examples/Airspace/tests/visualiser_tests.cpp` : Visualiseurs de bout en bout (ppokit-run, trames UDP, vue).

## Générer uniquement la documentation

```powershell
# Depuis la racine du dépôt, avec Doxygen installé
.\tools\generatedoc.ps1 -NoOpen
# ou : .\generatedoc.bat
```

Ouvrir `docs/generated/html/index.html`. `docs/generated/doxygen-warnings.log`
recense les diagnostics de génération. Les graphes automatiques ne sont pas
générés par défaut. La documentation se génère sans compiler le projet.

## Validation de couverture

La rubrique **Files** présente les sources analysées et leurs corps. Une
documentation générée ne démontre ni la compilation du binaire ni la convergence
d'un modèle.
