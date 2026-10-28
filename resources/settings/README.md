# PPOTrial — Configurations

Toutes les configurations utilisent **formatVersion: 2**, avec un unique lecteur C++17 par type de données.

| Je veux… | Je modifie… |
| --- | --- |
| Choisir les entités, leurs cibles et leurs agents | Scenarios/Playground.json |
| Définir les profils d'agents et leur mode | AgentProfiles.json |
| Régler PPO, rewards, rollout, fréquence d'évaluation | Training/RLMission.json, Training/RLSNA.json ou Training/BluePatrol.json |
| Composer la patrouille bleue ou le combat 2v2 | Scenarios/BluePatrol.json ou Scenarios/TwoVsTwo.json |
| Modifier les caractéristiques physiques | Platforms/Mobile/*.json ou Platforms/RestrictedZone/*.json |
| Régler les limites du monde et son pas | Environment.json |
| Régler l'IHM et la caméra | Renderer.json |
| Changer la composition ou les limites d'une campagne | Scenarios/Evaluation/RLMission.json ou RLSNA.json |
| Changer les 100 cas d'évaluation | Evaluation/Cases/RLMission.json ou RLSNA.json |
| Utiliser des collections nommées d'états initiaux | Evaluation/InitialStates.json |

Dans **Playground.json**, un agent est simplement :

~~~json
"agents": [
  {"profile": "RLMission", "mode": "training"}
]
~~~

**AgentProfiles.json** relie ce profil aux fabriques d'agents existantes. Les scénarios n'exposent pas les noms de classes C++. Le comportement PPO et les modèles .pt sont inchangés.

Les poids des récompenses sont identifiés par leur **nom**, puis remis dans l'ordre défini par le modèle C++. Un nom oublié ou ajouté est refusé, pour éviter de changer les récompenses par erreur.

**checkpointEvery: 5** signifie une demande d'évaluation toutes les cinq mises à jour PPO. Les paramètres des campagnes d'évaluation (nbTests, nbSteps) sont dans Scenarios/Evaluation/.

Pour les deux défenseurs RLTools, **Training/BluePatrol.json** règle le nombre
de pas d'équipe par rollout (2048 par défaut), la limite de mises à jour, la
graine, la campagne d'évaluation intégrée et les sept récompenses MAPPO. Deux
bleus produisent deux transitions
par pas d'équipe. Le mini-batch interne de 128 n'est pas la longueur du
rollout. `checkpointEvery` déclenche les cas déterministes 2v1/2v2 ; le
checkpoint de production n'est remplacé qu'après un meilleur score.

Après une mise à jour PPO, `model/BluePatrol/latest/` contient l'acteur v7 et
`blue_patrol_training_v4.txt`. Ce dernier permet une vraie reprise de l'acteur,
du critique, d'Adam, du compteur et du RNG ; le buffer de rollout repart vide.
`model/BluePatrol/prod/` ne contient que le meilleur acteur évalué.

Les capacités propres aux défenseurs bleus sont dans
`Platforms/Mobile/defender_car.json`. `FlightModel.CruiseSpeed` fixe leur
vitesse initiale et leur référence de patrouille ; les autres champs
`FlightModel` règlent leurs limites physiques et `SNA` leur domaine
d'engagement. `MaxRotAcceleration` est exprimé en rad/s. Les valeurs bleues
restent légèrement supérieures à celles d'`attacker_car` : 0,42 contre
0,40 rad/s, `CruiseSpeed` et `MaxSpeed` à 420 contre 400 m/s, et
`MaxShootAngle` à 1,10 contre 1,047198 rad. Ces paramètres ne modifient pas
`attacker_car` ni RLMission.

`Scenarios/TwoVsTwo.json` contient aussi `redSpawn`, `threatCurriculum` et
`restrictedZoneSpawn`. Le premier règle l'arc extérieur avant/latéral, le
recul hors du monde et la séparation minimale des deux rouges. Le curriculum
règle les probabilités de zéro, une ou deux menaces et les plages d'activation
différée. Le dernier règle la position de la zone dans le côté défendu et sa
probabilité d'activation. Ces paramètres ne changent ni le réseau ni le
checkpoint RLMission.

L'application sauvegarde à l'arrêt **Environment.json**, **Renderer.json** et **Training/*.json**. Modifier ces fichiers seulement lorsque l'application est arrêtée. Une configuration invalide ne doit pas être remplacée par des valeurs implicites.

Plus d'informations : [Guide complet](../../docs/configuration-guide.md).

Les types canoniques sont `Mobile`, `RestrictedZone` et `Objective`.
Les noms historiques de scénarios `Aircraft` et `SAM` restent acceptés :
ils instancient respectivement `Mobile` et `RestrictedZone` sans
changer les valeurs ECS ni les modèles existants.
