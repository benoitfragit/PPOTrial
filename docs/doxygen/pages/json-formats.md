\page json_formats Contrats JSON du dépôt au 31 octobre 2026

[TOC]

## Principe de lecture

Les ressources actives sont sous `resources/settings` et utilisent `formatVersion: 2`. Les véritables contrats sont imposés par les validateurs C++/Python, non par des exemples historiques restés dans des guides. Les chemins et noms ci-dessous correspondent aux fichiers effectivement présents dans l'arbre Git **e003f0c5235057470db824b5d9bf12b23d6e64c8**.

## Scénarios

| Fichier | Contenu |
| --- | --- |
| `Scenarios/BlueDefense.json` | Défense opérationnelle : objectif, templates bleu/rouge, zones, P=4/N=4 par défaut, paramètres de placement et retards. |
| `Scenarios/BluePatrol.json` | Patrouille d'apprentissage : objectif, template bleu, zones, P=2/N=0 par défaut, `patrolOnly=true`. |
| `Scenarios/Playground.json` | Scénario d'exemple Airspace. |
| `Scenarios/Evaluation/RLMission.json`, `RLSNA.json` | Cas/campagnes historiques Airspace. |

Une entité déclarative contient notamment `type`, `platform`, `side`, `target` si nécessaire, et `agents`. Les types courants sont `Mobile`, `RestrictedZone`, `Objective`. `AgentProfiles.json` relie les modes `training`/`production` de `RLSNA` et `RLMission` aux types C++ enregistrés.

Le bloc `blueDefense` paramètre `blueController`, `blueTeam.activeCount`, `redTeam.activeCount`, `redSpawn`, `restrictedZoneSpawn` et `restrictedZones`. Le code valide les effectifs P∈[1,9], N∈[0,10] et les contraintes de placement; les rouges sont des RLMission de production (pas de nouveaux agents RL rouges). Les activations différées doivent être prises en compte dans le critère « tous les rouges neutralisés ».

## Apprentissage BluePatrol

`Training/BluePatrol.json` contient :

| Clé | Interprétation |
| --- | --- |
| `patrol.racetrack.lengthMeters/widthMeters/headingDegrees/anchorNorthMeters/anchorEastMeters` | Intervalles de tirage d'hippodrome par épisode. |
| `patrol.racetrack.clockwise/boundaryMarginMeters/maximumSamplingAttempts` | Sens et contraintes d'échantillonnage. |
| `patrol.formation.separationDistanceMeters/separationAngleDegrees` | Intervalle des positions du wedge. |
| `patrol.defense.*` | Activation/désactivation du suivi de menaces, horizon d'intrusion et protection de l'objectif. |
| `training.rolloutSteps` | Pas synchronisés d'équipe par mise à jour PPO (actuellement 2048). |
| `training.maxUpdates` | Plafond de mises à jour (1000 dans le JSON de base). |
| `training.checkpointEvery` | Fréquence des campagnes d'évaluation. |
| `training.seed` | Graine de l'entraînement. |
| `evaluation.episodes/maxSteps/seed/minimumTrackingFraction` | Campagne embarquée déterministe. |
| `rewards.clamp` | Borne scalaire absolue de la récompense. |
| `rewards.weights` | **Sept noms hérités** mis en correspondance avec les canaux généralisés par le C++. |
| `rewards.generalized.ownLossWeight/lowSpeedRiskWeight` | Poids supplémentaires des pertes individuelles et de basse vitesse. |
| `rewards.shaping` | Coefficients du suivi, de formation et de l'effort de contrôle. |

La référence exacte des **15 canaux généralisés** et de leur affectation par rôle est \ref blue_mappo_reference. Modifier un poids n'est pas équivalent à changer une condition de destruction. Le fichier `training.saveEveryUpdates` peut être fixé par le workflow/runner, avec des valeurs par défaut prises en charge par la lecture de settings.

## Curriculum

`Training/BluePatrolCurriculum.json` : V1, séquence historique. `Training/BluePatrolCurriculumV2.json` : V2, **1000 updates**, `rehearsal.primaryUpdates=24`, `rehearsal.priorUpdates=6`, neuf étapes :

| Étape | P | N | Zones hostiles | Budget updates |
| --- | ---: | ---: | --- | ---: |
| 01-hippodrome | 1 | 0 | non | 32 |
| 02-hippodrome-wedge | 2 | 0 | non | 48 |
| 03-one-vs-one | 1 | 1 | non | 72 |
| 04-one-vs-two | 1 | 2 | non | 72 |
| 05-one-vs-zones | 1 | 0 | oui | 64 |
| 06-two-vs-one-zones | 2 | 1 | oui | 88 |
| 07-two-vs-two-zones | 2 | 2 | oui | 100 |
| 08-three-vs-three-zones | 3 | 3 | oui | 124 |
| 09-four-vs-four | 4 | 4 | oui | 400 |

Le curriculum V2 est piloté par `tools/ci/mappo_curriculum.py`, pas par un tirage dynamique de scénarios à l'intérieur de `GeneralizedPatrolLearner`. Les contraintes de reprise sont documentées dans \ref config_ci_reference.

## Physique, rendu, évaluation historique

- `Environment.json` : `environment.maxEast`, `maxNorth`, `timeStep` (positions en mètres, temps en secondes).
- `Platforms/Mobile/*.json` : paramètres de vol et SNA, dont `FlightModel.CruiseSpeed`, limites d'accélération, vitesse minimale, carburant.
- `Platforms/RestrictedZone/restrictedzone.json`, `sam.json` : zones et géométrie.
- `Renderer.json` : `renderer.camera`, `showGrid`, `showPatrolTrack`, `showHardBoundary`, `showPseudoLidar` et préférences de visualisation.
- `Training/RLMission.json`, `Training/RLSNA.json` : paramètres PPO LibTorch historiques.
- `Evaluation/Cases/RLMission.json`, `RLSNA.json` : ensembles de tests historiques; `Evaluation/InitialStates.json` : états initiaux nommés.

La liste de **tous les JSON** et de leurs chemins, y compris les fichiers legacy, figure dans \ref source_inventory.
