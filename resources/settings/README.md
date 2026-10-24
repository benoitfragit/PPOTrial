# PPOTrial — Configurations

Toutes les configurations utilisent **formatVersion: 2**, avec un unique lecteur C++17 par type de données.

| Je veux… | Je modifie… |
| --- | --- |
| Choisir les entités, leurs cibles et leurs agents | Scenarios/Playground.json |
| Définir les profils d'agents et leur mode | AgentProfiles.json |
| Régler PPO, rewards, rollout, fréquence d'évaluation | Training/RLMission.json ou Training/RLSNA.json |
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

L'application sauvegarde à l'arrêt **Environment.json**, **Renderer.json** et **Training/*.json**. Modifier ces fichiers seulement lorsque l'application est arrêtée. Une configuration invalide ne doit pas être remplacée par des valeurs implicites.

Plus d'informations : [Guide complet](../../docs/configuration-guide.md).

Les types canoniques sont `Mobile`, `RestrictedZone` et `Objective`.
Les noms historiques de scénarios `Aircraft` et `SAM` restent acceptés :
ils instancient respectivement `Mobile` et `RestrictedZone` sans
changer les valeurs ECS ni les modèles existants.
