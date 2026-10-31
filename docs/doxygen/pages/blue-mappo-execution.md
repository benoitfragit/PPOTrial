\page blue_mappo_reference BlueDefense — exécution exacte du MAPPO généralisé

[TOC]

## Modèle et équipes

`GeneralizedPatrolMappoAgent` représente **un contrôleur commun pour les Bleus** et implémente `IAgent`. `TeamCapacity` fixe **P=1..9** Bleus et **N=0..10** Rouges, avec tableaux de capacité fixe. Les Rouges instanciés portent chacun le contrôleur de production historique `RLMission`; la politique rouge n'est pas modifiée par l'apprentissage bleu.

Deux exemples utilisent le même moteur : `blue-patrol` (par défaut P=2, N=0) et `blue-defense` (par défaut P=4, N=4). Ces valeurs sont chargées depuis `Scenarios/BluePatrol.json` et `Scenarios/BlueDefense.json`. `blue_template`, `red_template`, `blue_objective` et les zones fournissent les paramètres d'entités. `bluedefenseexample.cpp` choisit et instancie les agents avec `BlueDefenseWorldRules`.

## Reset et rôles de formation

`InitializerContext` et les fonctions de `Scenario/initializer.*`, `spawn.*` et `spawn_config.*` assurent le tirage des positions, de l'hippodrome, des zones et des activations retardées. `Core/blue_team_roles.hpp` désigne **un leader tiré à l'initialisation** et des équipiers aux emplacements de wedge stables pendant l'épisode. `Core/patrol_geometry.hpp` projette la position du leader sur l'hippodrome analytique et calcule les écarts. La disparition du leader n'implique pas à elle seule l'échec de l'équipe; vérifier les conditions exactes dans `GeneralizedBlueTeamFailed`.

## Observation par Bleu

`Runtime/generalized_runtime_state.hpp` copie les états ECS dans `GeneralizedRuntimeState`, valide la finitude et crée **P observations décentralisées**. `Core/generalized_observation.hpp` fixe le contrat v8, ses offsets et masques :

- État propre, **61 rayons** pseudo-LiDAR, rôle explicite et position normalisée dans le wedge.
- Jusqu'à **10 pistes rouges** et **8 pistes de Bleus voisins**, avec masques d'existence.
- Objectif séparé, géométrie de patrouille, variables globales, effectifs P/N normalisés.
- Les pistes sont normalisées et validées; l'attention n'est pas censée prendre en compte les slots masqués.

Le tableau brut v8 et le compactage du réseau sont définis par les constantes du code, **pas** par un réarrangement arbitraire de JSON. `Training/generalized_attention_actor.hpp` construit des résumés invariants à l'ordre (attention masquée, moyenne et maximum) avant le réseau acteur. Le critic est centralisé et utilise des informations d'équipe et l'identité de l'agent, via `generalized_critic_observation.hpp`.

## Actions et boucle d'entraînement

`ChooseAction` produit un lot synchronisé de **P commandes** (guidage). La physique ECS avance une fois, les règles de monde résolvent les pertes; ensuite `SetNewState` capture la transition et alimente `GeneralizedPatrolLearner::Observe`.

`GeneralizedPatrolLearner` emploie l'acteur partagé et un critic centralisé. Les masques distinguent participation historique de l'épisode et disponibilité physique courante; les transitions des agents morts ne deviennent pas automatiquement des actions entraînables. `rolloutSteps` est exprimé en **pas d'équipe synchronisés**; la collecte s'achève sur une frontière d'optimisation. Une frontière PPO ne termine **pas nécessairement** un épisode ECS (important lorsque N=0 et qu'un épisode dépasse la taille du rollout).

`Training/generalized_advantage.hpp` et `Training/generalized_patrol_learner.hpp` portent la GAE, l'optimisation et les contrôles de buffers. `GeneralizedOptimizationGuard` arrête un optimiseur en état invalide et journalise fichier/ligne/phase.

## Conditions de fin réellement codées

Dans `GeneralizedPatrolMappoAgent::SetNewState` :

- `noThreatRemains` : **N > 0**, aucun Rouge vivant, **aucune menace retardée encore en attente**.
- `terminated` : échec de l'équipe bleue, objectif détruit ou `noThreatRemains`.
- `truncated` : pas d'arrêt terminal et compteur de pas atteignant `training_.evaluation.maxSteps` (en campagne) ou `parameters.episodeSteps` (en entraînement).
- Avec **N=0**, la condition « tous les rouges détruits » n'est **pas** une victoire terminale; l'épisode de patrouille se termine à l'horizon ou sur échec physique/objectif.

Le test `GeneralizedBlueTeamFailed` et les règles ECS définissent l'échec des Bleus et les frontières. Il ne faut pas déduire d'une pénalité de frontière qu'elle cause directement la destruction. La limite de pas est une **troncature**, à ne pas confondre avec le succès de mission. Le fuel est un effet de la physique et des paramètres véhicule, pas une condition générique de réussite ajoutée par le moteur PPO.

## Récompenses : 15 canaux effectifs

Énumération exacte définie par `Core/generalized_cooperative_reward.hpp` :

| Canal | Sens |
| --- | --- |
| Patrol tracking | Progrès et suivi de l'hippodrome du leader. |
| Formation keeping | Progrès de l'équipier vers sa formation. |
| Pursuit progress | Progrès de déplacement vers les menaces. |
| Control effort | Coût des commandes de guidage. |
| Blue survival | Pertes bleues collectives normalisées par P. |
| Red neutralization | Neutralisations rouges normalisées par N (zéro pour N=0). |
| Threat containment | Diminution du danger sur l'objectif. |
| Objective survival | Évènement de destruction de l'objectif. |
| Mission outcome | -1 sur échec, +1 si N>0 et tous les rouges éliminés sans attente. |
| Timeout | Pénalité de troncature si des rouges restent / sont attendus. |
| Blue separation | Risque de proximité entre Bleus. |
| Restricted zone clearance | Risque lié aux zones interdites. |
| Boundary clearance | Risque de proximité/dépassement de frontière. |
| Low speed risk | Risque et transition de perte liés à la basse vitesse. |
| Own loss | Perte individuelle non diluée. |

`AggregateGeneralizedCooperativeRewards` calcule canaux communs et responsabilités individuelles, distingue leader / équipiers, normalise les événements collectifs, applique des poids et un clamp lors de la réduction `Total`. Les coefficients de base en JSON sont les **sept poids historiques**; le runtime les projette dans les 15 canaux via `MapGeneralizedRewardWeights`, ajoute des coefficients dédiés (notamment `ownLossWeight` et `lowSpeedRiskWeight`), ainsi que des constantes de protection. Il serait donc inexact de dire que les 15 poids sont tous directement modifiables dans `rewards.weights`.

La responsabilité `lowSpeedRisk` ne doit pas être confondue avec l'élimination physique « low_speed ». Les fonctions de pénalité sont dans `Runtime/generalized_reward_shaping.hpp`.

## Évaluation et promotion

`Evaluation/generalized_evaluation_campaign.hpp` ordonnance des cas déterministes en réutilisant les règles physiques ECS. Les graines, la direction de patrouille et la phase sont contrôlées; les sorties de l'acteur sont déterministes pendant la campagne, le rollout d'entraînement n'est pas alimenté par ces cas. `SetNewState` calcule les métriques d'évaluation; un score **strictement meilleur** permet de sauvegarder l'acteur de production. Les logs `[MAPPO_EVAL_LOSS]` et `[MAPPO_EVAL_EPISODE]` expliquent les pertes, le statut de l'objectif, le leader et les issues. L'évaluation partage le contrôleur ECS; ne pas déduire de cette page qu'un moteur d'évaluation indépendant existe.

## Checkpoints et reprise

`latest/blue_patrol_generalized_v8.txt` : poids acteur d'inférence.
`latest/blue_patrol_generalized_training_v5.txt` : état d'entraînement complet (signature interne V6 : acteur, critic, Adam, RNG, compteur). `prod/blue_patrol_generalized_v8.txt` : acteur promu si meilleur résultat. Le contenu du rollout incomplet n'est pas sérialisé. Les lecteurs de migration v7/v4 sont des voies de **conversion**, pas une reprise binaire complète des anciens 2v2. Ne pas mélanger les checkpoints MAPPO et RLMission.

## Diagnostic guidé par le code

| Symptôme | Chemin à suivre |
| --- | --- |
| Bleus immobiles / N=0 | `blue_team_roles.hpp` -> `patrol_geometry.hpp` -> `generalized_reward_shaping.hpp` -> `SetNewState`. |
| Mauvais nombre d'agents | `team_capacity.hpp` -> `spawn_config.cpp` -> `initializer.cpp` -> `BuildGeneralizedObservations`. |
| Faible vitesse / fuel | `flightmodelsystem.cpp` -> `bluedefenserules.cpp` -> logs `MAPPO_EVAL_LOSS` -> `lowSpeedRisk`. |
| Dépassement frontière | `GeneralizedBoundaryRisk` -> `GeneralizedBlueTeamFailed` -> `BlueDefenseWorldRules`. |
| « Invalid training buffer » | `GeneralizedPatrolLearner::Observe` et `GeneralizedOptimizationGuard`; vérifier P actions / P retours. |
| Évaluation dégradée | `generalized_evaluation_metrics.hpp` -> `generalized_evaluation_campaign.hpp` -> `PromoteEvaluatedActor`. |
| Modèle non repris | `generalized_training_checkpoint.hpp` -> `generalized_checkpoint.hpp` -> constructeur du contrôleur. |
