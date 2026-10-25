# PPO Patrol — défenseurs (25 octobre 2026)

## Utilisation

Dans l'exécutable principal `PPO.exe`, sélectionner `BLUE PATROL PPO`
puis `Play`. Il n'est pas nécessaire de lancer un programme différent.
`Pause` suspend la boucle existante d'`Environment`, donc aussi les mises
à jour d'apprentissage. `Stop` arrête cette boucle. Aucun thread supplémentaire
n'est créé pour RLtools.

Le scénario `resources/settings/Scenarios/BluePatrol.json` contient
`"patrolControllers": ["ppo", "reference"]`. Les bleus n'ont **pas de cible
par défaut**. Le premier mobile est un `PatrolTrainingAgent : IAgent`
et le second un `PatrolReferenceAgent : IAgent`, qui suit la trajectoire
cinématique via les commandes SNA. Les rouges, `RLMission` et leurs JSON
restent inchangés. Un scénario sans aucun rôle `ppo` est refusé.

## Cycle de vie et architecture

`ppo::core::StepAgents` (appelé par `Environment`) exécute :

1. `ChooseAction()` des deux agents.
2. Dynamique ECS/SNA et règles de collision.
3. `SetNewState()` des deux agents.

`PatrolTrainingAgent` dérive de `PatrolProdAgent` et délègue
l'optimisation PPO au composant réutilisable `PatrolLearner` (RLtools).
Il conserve son état d'optimisation lors des resets d'épisode ECS.
Toutes les 128 transitions ECS, `SetNewState()` exécute une mise
à jour PPO ; les poids sont publiés dans `PatrolPolicy` de façon
séquentielle avant la prochaine action. Le snapshot de **l'agent PPO
lui-même** porte le nombre d'updates et `m_training`.

**Limitation à connaître** : les trajectoires utilisées pour calculer
les gradients proviennent pour l'instant de l'environnement de rollouts
RLtools (`patrol_core.hpp`), non des transitions ECS visibles dans
la fenêtre. La fenêtre présente la politique effectivement mise à jour.
Ce curriculum entraîne un rôle PPO par épisode RLtools ; l'équipier
suit la référence, en alternant les rôles avec Actor partagé.
Ce n'est pas encore MAPPO et ce n'est pas un apprentissage directement
fondé sur les trajectoires ECS. Une mise à jour PPO est synchrone et peut
provoquer une pause perceptible dans la fenêtre : ne pas introduire
un thread implicite pour masquer ce coût.

Le modèle est sauvegardé atomiquement tous les 10 updates et en fin
de session sous
`resources/model/BluePatrol/prod/blue_patrol_v2.txt`.
La signature `BLUE_PATROL_PPO_V2` est conservée pour la compatibilité
des checkpoints existants. Si aucun checkpoint compatible n'existe,
l'Actor est initialisé de façon déterministe puis entraîné.

## Observation et contrôle

Contrat 124 entrées, 2 sorties : 7 caractéristiques propres, 61 rayons
LiDAR, deux pistes rouges (13 caractéristiques chacune, masquées dans
ce premier scénario), deux pistes amies/objectif (9 chacune),
10 caractéristiques de patrouille et 2 globales. Deux branches
d'attention masquées précèdent l'Actor MLP. Les actions sont la vitesse
de rotation et l'accélération longitudinales normalisées ; la physique
est appliquée par le système ECS existant.

La patrouille est en râteau : gardes décalés longitudinalement et
latéralement, demi-tours à des moments distincts. Les observations
rouges restent masquées tant qu'aucun attaquant n'est présent.

## Récompense et terminaison

`TeamReward` combine suivi de route, progression, cap, vitesse et
coût des commandes. Une approche à moins de 250 m entraîne une
pénalité progressive jusqu'à −0,5 par pas ; une collision à 50 m
ou moins entraîne −100 et une terminaison. La détection tient
compte de la séparation minimale pendant tout le pas (collision
entre deux échantillons). Sortie de monde et états invalides
terminent également l'épisode. Timeout et troncature restent distincts.

`PatrolEvaluator` fournit un résultat explicite
`Success`, `Failure` ou `Truncated` pour deux politiques simultanées.
Il teste la collision, les zones interdites configurées, les
limites du monde et la qualité de la patrouille.

## Build et validation

Le projet reste en C++17. `PPO_BUILD_CORE_TESTS` est OFF par défaut
en local et explicitement ON en CI. Les programmes autonomes
`PPOBlueTrain` et `PPOPatrolEvaluate` sont facultatifs :
`PPO_BUILD_PATROL_TOOLS=ON` (OFF par défaut).

Tests : `TwoVsTwoPatrolTests`, `TwoVsTwoAttentionTests`,
`TwoVsTwoEmbeddedLearnerSmoke`, `TwoVsTwoPatrolRuntimeSmoke`
et `TwoVsTwoPatrolEvaluationTests`. Les smoke tests prouvent la
progression mécanique et l'exécution d'un update PPO, pas la convergence
du modèle. La compilation CI VS2022 et le comportement graphique
doivent être validés avant fusion de la PR.
