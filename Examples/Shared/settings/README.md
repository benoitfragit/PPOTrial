# PPOTrial — Configurations

Toutes les configurations utilisent **formatVersion: 2**, avec un unique lecteur C++17 par type de données.

Ce dossier (`Examples/Shared/settings`) contient les couches communes aux exemples.
Les réglages propres à un exemple sont dans son dossier : `Examples/Airspace/settings`
pour les agents rouges, le projet `Examples/BlueDefense/projects/BluePatrol` pour Blue.

| Je veux… | Je modifie… |
| --- | --- |
| Choisir les entités, leurs cibles et leurs agents | Examples/Airspace/settings/Scenarios/Playground.json |
| Définir les profils d'agents et leur mode | AgentProfiles.json |
| Régler les rewards et le rollout d'Airspace | Examples/Airspace/settings/Training/RLMission.json ou RLSNA.json |
| Régler PPO et la fréquence d'évaluation d'Airspace | training.json et eval/campaign.json de Examples/Airspace/projects/RLMission ou RLSNA |
| Régler le MAPPO bleu (PPO, rewards, fréquence d'évaluation) | training.json, blue_patrol_training.json et eval/validation.json du projet BluePatrol |
| Composer l'équipe P/N commune à l'entraînement et à l'inférence | scenario.json du projet BluePatrol |
| Modifier les caractéristiques physiques | Platforms/Mobile/*.json ou Platforms/RestrictedZone/*.json |
| Régler les limites du monde et son pas | Environment.json |
| Changer la composition ou les limites d'une campagne | Examples/Airspace/settings/Scenarios/Evaluation/RLMission.json ou RLSNA.json |
| Changer les 100 cas d'évaluation | Examples/Airspace/settings/Evaluation/Cases/RLMission.json ou RLSNA.json |
| Utiliser des collections nommées d'états initiaux | Examples/Airspace/settings/Evaluation/InitialStates.json |

Dans **Playground.json**, un agent est simplement :

~~~json
"agents": [
  {"profile": "RLMission", "mode": "training"}
]
~~~

**AgentProfiles.json** relie ce profil aux fabriques d'agents existantes. Les scénarios n'exposent pas les noms de classes C++.

Les poids des récompenses sont identifiés par leur **nom**, puis remis dans l'ordre défini par le modèle C++. Un nom oublié ou ajouté est refusé, pour éviter de changer les récompenses par erreur.

Les **Training/*.json** d'Airspace ne contiennent que `rolloutSteps`, le clamp et les poids des récompenses. Les hyperparamètres PPO sont dans `training.json` du projet (`Examples/Airspace/projects/RLMission` ou `RLSNA`) ; dans `eval/campaign.json` de RLMission et de RLSNA, `"during_training": { "every_updates": 5 }` fait jouer la campagne d'évaluation par `ppokit-run` toutes les cinq mises à jour PPO (le `checkpointEvery` 5 historique des deux agents). Les paramètres des campagnes d'évaluation (nbTests, nbSteps) sont dans Examples/Airspace/settings/Scenarios/Evaluation/.

Pour le projet BluePatrol partagé, **training.json** règle PPO, le rollout et la
graine du learner. **blue_patrol_training.json** configure notamment la géométrie,
les récompenses et les réglages d'épisode de l'environnement. La suite
**eval/validation.json** fixe les graines et la fréquence de campagne utilisées par
`ppokit-run`. P Bleus produisent P transitions par pas d'équipe ; le mini-batch
livré est de 1024 et le rollout de 2048 pas d'équipe.

Après une mise à jour PPO, `checkpoints/shared/latest/` du projet BluePatrol contient l'acteur
`policy.ppokit` et `latest.ppokit` (état d'entraînement complet, artefact PPOKit). Ce dernier permet
une vraie reprise de l'acteur,
du critique, d'Adam, du compteur et du RNG ; le buffer de rollout repart vide.
`checkpoints/shared/prod/` contient la politique de production déclarée dans `models.json`.

`rewards.noThreatPatrol` configure aussi le ralliement puis le suivi des dix
waypoints : `rallyEnterDistanceMeters`, `rallyExitDistanceMeters` et
`maxEpisodeSeconds`. Cette dernière valeur est une durée simulée ; avec
`dt=0.2`, 480 secondes donnent 2400 pas et une troncature sans succès.

Les capacités propres aux défenseurs bleus sont dans
`Platforms/Mobile/defender_car.json`. `FlightModel.CruiseSpeed` fixe leur
vitesse initiale et leur référence de patrouille ; les autres champs
`FlightModel` règlent leurs limites physiques et `SNA` leur domaine
d'engagement. `MaxRotAcceleration` est exprimé en rad/s. Les valeurs bleues
sont propres au scénario Bleu : rotation maximale 0,8 rad/s,
`CruiseSpeed` à 300 m/s, `MaxSpeed` à 750 m/s, et `MaxShootAngle` à
3,147198 rad. Ces paramètres ne modifient pas
`attacker_car` ni RLMission.

Le `scenario.json` du projet BluePatrol contient aussi `redSpawn`, `threatCurriculum` et
`restrictedZoneSpawn`. Le premier règle l'arc extérieur avant/latéral, le
recul hors du monde et la séparation minimale des deux rouges. Le curriculum
règle les probabilités de zéro, une ou deux menaces et les plages d'activation
différée. Le dernier règle la position de la zone dans le côté défendu et sa
probabilité d'activation. Ces paramètres ne changent ni le réseau ni le
checkpoint RLMission.

**Environment.json** et les **Training/*.json** d'Airspace sont lus par les programmes PPOKit (`ppokit-run` et les DLL d'environnement) ; contrairement à l'ancien PPO.exe, ils ne les réécrivent pas à l'arrêt. Modifier ces fichiers seulement lorsqu'aucun run n'est en cours. Une configuration invalide ne doit pas être remplacée par des valeurs implicites.

Plus d'informations : [Guide complet](../../../docs/configuration-guide.md).

Les types canoniques sont `Mobile`, `RestrictedZone` et `Objective`.
Les noms historiques de scénarios `Aircraft` et `SAM` restent acceptés :
ils instancient respectivement `Mobile` et `RestrictedZone` sans
changer les valeurs ECS ni les modèles existants.
