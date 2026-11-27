\page json_formats Contrats JSON du dépôt

[TOC]

## Principe de lecture

Les ressources de simulation sont sous `Examples/Shared/settings`,
`Examples/<Nom>/settings` et les projets `Examples/<Nom>/projects`. Les
validateurs C++ et Python définissent les contrats effectifs de chaque format.

## Scénarios

| Fichier | Contenu |
| --- | --- |
| `scenario.json` du projet `Examples/BlueDefense/projects/BluePatrol` | Défense opérationnelle partagée entre entraînement et inférence : objectif, templates bleu/rouge, zones, P=4/N=4 par défaut, paramètres de placement et retards. |
| `Scenarios/Playground.json` | Scénario d'exemple Airspace. |
| `Scenarios/Evaluation/RLMission.json`, `RLSNA.json` | Cas/campagnes historiques Airspace. |
| `Scenarios/RLSNATraining.json` | Scène d'entraînement de RLSNA (intercepteur en mode `training`, cible), projet `Examples/Airspace/projects/RLSNA`. |

La commande `ppokit-run` sélectionne le mode : `train` entraîne le learner
MAPPO, tandis que `play --model PROD` exécute
uniquement l'acteur de production (`project.json` du projet BluePatrol).

Une entité déclarative contient notamment `type`, `platform`, `side`, `target` si nécessaire, et `agents`. Les types courants sont `Mobile`, `RestrictedZone`, `Objective`. `AgentProfiles.json` relie les modes `training`/`production` de `RLSNA` et `RLMission` aux types C++ enregistrés.

Le bloc `blueDefense` paramètre `blueController`, `blueTeam.activeCount`, `redTeam.activeCount`, `redSpawn`, `restrictedZoneSpawn` et `restrictedZones`. Le code valide les effectifs P∈[1,9], N∈[0,10] et les contraintes de placement; les rouges sont des RLMission de production (pas de nouveaux agents RL rouges). Les activations différées doivent être prises en compte dans le critère « tous les rouges neutralisés ».

## Apprentissage BluePatrol

`blue_patrol_training.json` du projet `Examples/BlueDefense/projects/BluePatrol` contient :

| Clé | Interprétation |
| --- | --- |
| `patrol.racetrack.lengthMeters/widthMeters/headingDegrees/anchorNorthMeters/anchorEastMeters` | Intervalles de tirage d'hippodrome par épisode. |
| `patrol.racetrack.clockwise/boundaryMarginMeters/maximumSamplingAttempts` | Sens et contraintes d'échantillonnage. |
| `patrol.formation.separationDistanceMeters/separationAngleDegrees` | Intervalle des positions du wedge. |
| `patrol.defense.*` | Activation/désactivation du suivi de menaces, horizon d'intrusion et protection de l'objectif. |
| `training.*` | Réglages du backend Blue historique ; la graine sert aussi à l'environnement. Le projet partagé lit son rollout et ses réglages PPO dans `training.json`. |
| `evaluation.*` | Réglages d'épisode et de métriques. La suite `eval/validation.json` du projet partagé définit ses graines, le nombre de cas et sa fréquence. |
| `rewards.clamp` | Borne scalaire absolue de la récompense. |
| `rewards.weights` | **Sept noms hérités** mis en correspondance avec les canaux généralisés par le C++. |
| `rewards.generalized.ownLossWeight/lowSpeedRiskWeight` | Poids supplémentaires des pertes individuelles et de basse vitesse. |
| `rewards.shaping` | Coefficients du suivi, de formation et de l'effort de contrôle. |

La référence exacte des **15 canaux généralisés** et de leur affectation par rôle est \ref blue_mappo_reference. Modifier un poids n'est pas équivalent à changer une condition de destruction.

## Curriculum

Le curriculum V3 se trouve dans `Examples/BlueDefense/curricula/` et le
calendrier des gates adaptatifs dans `tools/ci/adaptive_curriculum/gates.json`.
Ils sont pilotés par les lanceurs CI ; le bouton Train du projet partagé dans
Studio ne les exécute pas. Les contraintes de reprise sont décrites dans
\ref config_ci_reference.

## Physique, évaluation historique

- `Environment.json` : `environment.maxEast`, `maxNorth`, `timeStep` (positions en mètres, temps en secondes).
- `Platforms/Mobile/*.json` : paramètres de vol et SNA, dont `FlightModel.CruiseSpeed`, limites d'accélération, vitesse minimale, carburant.
- `Platforms/RestrictedZone/restrictedzone.json`, `sam.json` : zones et géométrie.
- `Training/RLMission.json`, `Training/RLSNA.json` : `training.rolloutSteps` (limite de pas d'un épisode sous `ppokit-run`), `rewards.clamp` et `rewards.weights` du `TrainingAgent`. Les hyperparamètres PPO sont dans `training.json` des projets `Examples/Airspace/projects/RLMission` et `RLSNA` ; la fréquence d'évaluation dans leur `eval/campaign.json` (`during_training.every_updates`).
- `Evaluation/Cases/RLMission.json`, `RLSNA.json` : ensembles de tests historiques; `Evaluation/InitialStates.json` : états initiaux nommés.

Les fichiers JSON de chaque projet se trouvent dans son dossier sous `Examples`.
