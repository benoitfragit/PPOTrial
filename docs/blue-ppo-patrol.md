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

`ppo::core::StepAgents` (appelé par `Environment`) conserve l'ordre
`ChooseAction -> ECS/SNA + règles -> SetNewState`. Aucune modification
des interfaces génériques, de `ALMission`/`RLMission`, ni du runtime
de leurs modèles. Pas de thread RLtools supplémentaire.

`PatrolTrainingAgent : IAgent` est le seul adaptateur d'apprentissage
de la patrouille. Il utilise `PatrolLearner` (RLtools CPU) :
- `ChooseAction` échantillonne la politique **du learner** et enregistre
  l'observation, l'action brute gaussienne, la moyenne et la log-probabilité
  sous la politique courante. Les commandes SNA reçoivent l'action bornée ;
  l'action brute reste celle du calcul PPO.
- `SetNewState` lit les deux mobiles **après** la physique et les règles ECS,
  calcule `TeamReward` sur leurs états réels, distingue `terminated` et
  `truncated`, et fournit l'observation suivante au collecteur.
- Le buffer on-policy contient 128 pas ECS avec 1 environnement
  (`N_ENVIRONMENTS=1`, `ON_POLICY_RUNNER_STEPS_PER_ENV=128`). Dès qu'il
  est plein, le même moteur RLtools calcule GAE et PPO (4 époques).
  **Aucun rollout headless supplémentaire n'est généré par l'agent ECS.**
- Une fois la mise à jour effectuée, l'Actor entraîné est publié
  séquentiellement dans `PatrolPolicy`, puis éventuellement sauvegardé.

Le retour est on-policy : aucun gradient n'utilise un couple observation /
action différent de celui effectivement envoyé à l'ECS. Pendant un
rollout, les poids restent constants. Les frontières d'épisode à
l'intérieur d'un rollout conservent leurs marqueurs de terminaison.
`Init()` réinitialise le contrôleur physique mais ne supprime pas le
buffer d'apprentissage en cours ni l'optimiseur.

Limite connue : le PPO est toujours **synchrone** avec le thread ECS ;
un update de 128 transitions peut provoquer une pause d'affichage
ponctuelle. Le CPU doit être profilé sous VS2022 Debug et Release.
Le smoke test vérifie le chemin de données et l'update, pas le temps réel.

Le checkpoint v2 demeure sous
`resources/model/BluePatrol/prod/blue_patrol_v2.txt`. Le premier
curriculum entraîne le seul rôle PPO du scénario, accompagné du
contrôleur de référence ; il ne s'agit pas encore de MAPPO. Le
programme CLI facultatif `PPOBlueTrain` conserve une méthode headless
indépendante, qui n'est **pas** appelée par l'agent ECS.

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
