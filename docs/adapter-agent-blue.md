# Adapter l'agent Blue

Ce guide décrit le contrôleur Blue actuellement compilé :
`GeneralizedPatrolMappoAgent`. Il couvre les entrées, les sorties, les réseaux,
les rewards et les points d'entrée à modifier. Les anciens contrôleurs à deux
Bleus ne constituent pas un second chemin d'exécution.

Le contrat supporte **P = 1..9 Bleus** et **N = 0..10 Rouges**. Un seul acteur
partagé pilote tous les Bleus. Chaque Rouge conserve son contrôleur RLMission
de production.

Pour une vue synthétique des boîtes, des flux et des sorties, consulter le
[schéma d'architecture des réseaux Blue](architecture/blue-network-architecture.md).

## Flux d'un pas

~~~text
Scénario JSON
  -> création des entités et du contrôleur Blue
  -> ChooseAction
       -> lecture de l'état ECS
       -> P observations brutes de 295 valeurs
       -> attention masquée et compactage à 156 valeurs
       -> acteur partagé
       -> P actions de 2 valeurs
       -> commandes rotation / accélération
  -> physique et règles du monde
  -> SetNewState
       -> nouvel état ECS et condition de fin
       -> 15 canaux de reward par trajectoire
       -> P transitions et observations critic de 470 valeurs
       -> mise à jour PPO à la fin du rollout
       -> checkpoint et évaluation selon la cadence configurée
~~~

Le mode `blue-defense-inference` s'arrête après l'application des actions et la
détection de fin d'épisode : il ne calcule ni reward d'apprentissage, ni
rollout, ni optimisation, ni sauvegarde.

## Modes d'exécution

| Entrée CLI | Mode du contrôleur | Modèle Blue |
| --- | --- | --- |
| `blue-defense` | `Training` | acteur `latest`, critic, rollout, Adam et évaluations périodiques |
| `blue-defense-inference` | `ProductionInference` | acteur `prod` déterministe uniquement |
| `blue-defense-evaluation` | `ProductionEvaluation` | acteur `prod` immuable et campagne déterministe, sans PPO |

Les trois entrées utilisent le scénario
[`resources/settings/Scenarios/BlueDefense.json`](../resources/settings/Scenarios/BlueDefense.json).
Le choix du mode et la création de `GeneralizedPatrolMappoAgent` sont dans
[`bluedefenseexample.cpp`](../Examples/BlueDefense/Scenario/bluedefenseexample.cpp).

## Entrée de l'acteur : observation v10

Chaque Bleu reçoit sa propre observation décentralisée de **295 `float`**.
Les dimensions et offsets sont fixés par
[`GeneralizedObservationLayout`](../Examples/BlueDefense/Agents/BluePatrol/Core/generalized_observation.hpp),
pas par le JSON.

| Indices | Taille | Groupe | Source |
| ---: | ---: | --- | --- |
| 0–6 | 7 | état du Bleu observateur | composants ECS du mobile |
| 7–67 | 61 | pseudo-LiDAR à 360 degrés | `sPseudoLidarState` |
| 68–69 | 2 | rôle et slot de formation | `BlueTeamRoles` |
| 70–199 | 130 | 10 pistes rouges × 13 valeurs | mobiles rouges masqués |
| 200–271 | 72 | 8 pistes alliées × 9 valeurs | autres Bleus masqués |
| 272–280 | 9 | objectif défendu | piste typée séparée |
| 281–290 | 10 | guidage de patrouille/formation | cible physique du rôle |
| 291–292 | 2 | variables globales de compatibilité | constantes du runtime |
| 293–294 | 2 | effectifs P/N | métadonnées de l'épisode |

La construction effective est dans
[`BuildGeneralizedObservations`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_runtime_state.hpp).
`GeneralizedPatrolMappoAgent::BuildObservations` impose en plus que les 61
rayons LiDAR proviennent réellement du capteur ECS.

### État propre : indices 0–6

| Index | Valeur |
| ---: | --- |
| 0 | position nord / demi-longueur nord du monde |
| 1 | position est / demi-longueur est du monde |
| 2 | vitesse / vitesse maximale |
| 3 | cosinus du cap |
| 4 | sinus du cap |
| 5 | vitesse de rotation / vitesse de rotation maximale |
| 6 | disponibilité physique : 1 vivant, 0 détruit |

Les positions sont nominalement dans `[-1, 1]`, mais ne sont pas bornées : une
sortie du rectangle peut exister avant la résolution de l'échec. Les valeurs
non finies sont rejetées.

### LiDAR : indices 7–67

Les 61 rayons couvrent 360 degrés et sont bornés dans `[0, 1]`. Ils combinent
les limites du monde, les zones interdites hostiles et les enveloppes de
sécurité des autres Bleus. Les collisions physiques restent résolues
séparément par les règles du monde.

Le calcul géométrique est dans
[`restrictedzonelidar.cpp`](../Framework/Simulation/World2D/src/restrictedzonelidar.cpp)
et l'alimentation du snapshot Blue dans
[`bluedefenserules.cpp`](../Examples/BlueDefense/Scenario/bluedefenserules.cpp).

### Rôle : indices 68–69

| Index | Valeur |
| ---: | --- |
| 68 | `-1` pour le leader, `+1` pour un équipier |
| 69 | slot d'équipier normalisé dans `[-1, 1]`; `0` pour le leader |

Le leader est tiré à l'initialisation de l'épisode. Il ne faut pas supposer que
`blue_1` est toujours leader. Les indices physiques restent stables pendant
l'épisode. Voir
[`blue_team_roles.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/blue_team_roles.hpp).

### Piste rouge : 13 valeurs par slot

Pour le Rouge `k`, la base vaut `70 + 13 × k`, avec `k` dans `[0, 9]`.

| Décalage | Valeur |
| ---: | --- |
| 0–1 | position nord/est relative, normalisée par les dimensions doubles du monde |
| 2–3 | vitesse vectorielle relative nord/est, divisée par la vitesse maximale |
| 4 | distance relative / double diagonale du monde |
| 5–6 | cosinus/sinus du gisement relatif au cap du Bleu |
| 7 | vitesse du Rouge / vitesse maximale |
| 8 | type hostile, constant à `1` |
| 9–10 | position absolue nord/est du Rouge, normalisée au monde |
| 11 | présence dans le rectangle nominal du monde |
| 12 | masque : `1` si la piste est disponible, sinon `0` |

Un slot absent, retardé ou détruit contient treize zéros. Le masque est la
dernière valeur et commande l'attention.

### Piste alliée ou objectif : 9 valeurs par slot

Les huit pistes alliées commencent à l'indice 200. L'objectif utilise le même
format aux indices 272–280.

| Décalage | Valeur |
| ---: | --- |
| 0–1 | position nord/est relative normalisée |
| 2–3 | vitesse vectorielle relative nord/est normalisée |
| 4 | distance relative normalisée |
| 5–6 | cosinus/sinus du gisement relatif |
| 7 | type : `0` pour un Bleu, `1` pour l'objectif |
| 8 | masque de disponibilité |

L'observateur n'est jamais ajouté à ses propres pistes alliées. Les slots non
utilisés restent à zéro.

### Guidage : indices 281–290

| Index | Valeur |
| ---: | --- |
| 281–282 | position nord/est de la cible de guidage relative au Bleu |
| 283–284 | vitesse vectorielle cible relative au Bleu |
| 285 | distance à la cible de guidage |
| 286–287 | cosinus/sinus du gisement relatif |
| 288 | validité de la cible |
| 289 | rôle, `-1` leader ou `+1` équipier |
| 290 | pression de menace continue dans `[0, 1]` |

Les positions et vitesses relatives sont bornées dans `[-2, 2]`; la distance
est bornée dans `[0, 2]`. Une cible invalide met les huit premières valeurs à
zéro.

En N=0, le leader passe de `RALLY` à `FOLLOW`, puis suit dix waypoints figés.
Les équipiers reçoivent leurs poses wedge dérivées du waypoint actif. En
combat, le même acteur continue de voir les Rouges, l'objectif et le guidage :
aucun contrôleur externe ne lui attribue une cible rouge.

### Variables globales et effectifs : indices 291–294

| Index | Valeur |
| ---: | --- |
| 291 | ratio de munitions historique, actuellement constant à `1` |
| 292 | emplacement réservé, actuellement constant à `0` |
| 293 | `(P - 1) / 8` |
| 294 | `N / 10` |

## Compactage et acteur partagé

[`GeneralizedAttentionActor`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_attention_actor.hpp)
transforme les 295 valeurs en **156 valeurs** avant le MLP.

| Bloc compact | Indices | Traitement |
| --- | ---: | --- |
| état propre | 0–6 | copie directe |
| LiDAR | 7–67 | copie directe |
| pistes rouges | 68–91 | attention + moyenne + maximum, embedding de largeur 8 |
| effectifs P/N | 92–93 | copie des deux effectifs normalisés |
| alliés + objectif | 94–111 | attention + moyenne + maximum, embedding de largeur 6 |
| guidage | 112–121 | copie des dix valeurs |
| global | 122 | ratio historique |
| slot de formation | 123 | copie du slot normalisé |
| suffixe central historique | 124–155 | zéro pour l'acteur |

Pour chaque ensemble de pistes, l'encodeur `tanh` produit un embedding. Une
requête calculée depuis les sept valeurs propres pondère les pistes valides.
Le contexte d'attention, la moyenne et le maximum sont concaténés. L'ordre des
pistes ne doit donc pas changer le résultat, et les slots masqués ne doivent
pas contribuer.

L'acteur dense est configuré dans
[`patrol_rltools.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/patrol_rltools.hpp) :

- entrée : 156 valeurs ;
- deux couches cachées de 128 neurones, activation ReLU ;
- sortie : 2 moyennes, activation identité ;
- distribution d'entraînement : gaussienne diagonale avec écart-type appris ;
- nouvel entraînement : `log_std = 0`, donc écart-type initial `1.0` ;
- mêmes poids pour tous les Bleus et toutes les compositions P/N.

## Critic centralisé

Le critic sert uniquement pendant l'apprentissage. Son entrée contient
**470 valeurs** définies dans
[`generalized_critic_observation.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_observation.hpp).

| Bloc | Taille | Contenu |
| --- | ---: | --- |
| observation propre compacte | 156 | zéro si le Bleu évalué est détruit |
| moyenne des Bleus vivants | 156 | agrégation invariante à l'ordre |
| maximum des Bleus vivants | 156 | agrégation invariante à l'ordre |
| fraction vivante | 1 | survivants / 9 |
| fraction initiale | 1 | P / 9 |

Le réseau défini dans
[`generalized_critic_network.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_network.hpp)
possède deux couches cachées de 256 neurones ReLU et une sortie scalaire à
activation identité. Il estime `V(équipe, blue_i)` pour chaque trajectoire.

Cette séparation réalise le contrat CTDE : entraînement centralisé par le
critic, exécution décentralisée par l'acteur.

## Sorties et commandes physiques

L'action contient exactement deux valeurs :

| Index | Commande ECS | Effet physique |
| ---: | --- | --- |
| 0 | `m_rotSpeedCmd` | multipliée par `MaxRotAcceleration` |
| 1 | `m_accCmd` | multipliée par `MaxAcceleration` |

Pendant l'entraînement, l'action est échantillonnée dans la gaussienne et la
valeur non bornée est conservée dans la transition PPO. Au moment de piloter le
mobile, chaque composante est bornée dans `[-1, 1]`. En inférence et en
évaluation, la moyenne déterministe est utilisée, puis bornée de la même façon.

Les affectations sont dans
[`GeneralizedPatrolMappoAgent::ChooseAction`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp).
La conversion en accélérations physiques est dans
[`flightmodelsystem.cpp`](../Framework/Simulation/World2D/src/flightmodelsystem.cpp).

## Rewards : contrat des 15 canaux

L'ordre est un contrat partagé par l'apprentissage, les snapshots, l'IHM et
les tests. L'énumération et l'agrégation sont dans
[`generalized_cooperative_reward.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/generalized_cooperative_reward.hpp).

| Index | Canal | Portée | Valeur non pondérée |
| ---: | --- | --- | --- |
| 0 | Patrol tracking | leader vivant | progression et qualité de suivi quand aucun Rouge n'est visible |
| 1 | Formation keeping | équipier vivant | amélioration du slot wedge ou suivi de sa cible de patrouille |
| 2 | Pursuit progress | partagé | rapprochement causé par les Bleus vers les Rouges vivants |
| 3 | Control effort | individuelle | coût négatif des deux commandes normalisées |
| 4 | Blue survival | partagé | `-nombre de pertes Bleues / P` |
| 5 | Red neutralization | partagé | `nombre de neutralisations / N`, zéro si N=0 |
| 6 | Threat containment | partagé | diminution du risque moyen pesant sur l'objectif |
| 7 | Objective survival | partagé | `-1` à la destruction de l'objectif |
| 8 | Mission outcome | partagé, sauf succès N=0 | `-1` échec, `+1` succès ; le succès de tour N=0 revient seulement au leader |
| 9 | Timeout | partagé | `-1` pour une troncature sans succès |
| 10 | Blue separation | individuelle | pénalité bornée de proximité et d'aggravation du risque Bleu/Bleu |
| 11 | Restricted zone clearance | individuelle | pénalité bornée d'exposition aux zones hostiles |
| 12 | Boundary clearance | individuelle | pénalité bornée d'approche ou de sortie des limites |
| 13 | Low speed risk | individuelle | pénalité bornée de basse vitesse et d'aggravation |
| 14 | Own loss | individuelle | `-1` pour le Bleu perdu pendant la transition |

Le shaping dense est calculé dans
[`generalized_reward_shaping.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_reward_shaping.hpp).
Les pertes, collisions et destructions restent décidées par la physique et les
règles du monde ; un reward ne détruit pas une entité.

Pour chaque Bleu, le scalaire PPO vaut :

~~~text
clamp(somme(canal[i] * poids[i]), -rewardClamp, +rewardClamp)
~~~

Les valeurs exposées dans le snapshot sont non pondérées. Les poids affichés
sont appliqués seulement lors de la réduction vers le scalaire PPO.

### Mapping du JSON vers les 15 poids

Le fichier
[`resources/settings/Training/BluePatrol.json`](../resources/settings/Training/BluePatrol.json)
conserve sept clés historiques. Le runtime les projette explicitement avec
`MapGeneralizedRewardWeights`.

| Canal interne | Source du poids | Valeur actuelle |
| --- | --- | ---: |
| Patrol tracking | `weights["Patrol tracking"]` | 1 |
| Formation keeping | `weights["Formation safety"]` | 4 |
| Pursuit progress | `weights["Interception"]` | 2 |
| Control effort | `weights["Control effort"]` | 1 |
| Blue survival | `weights["Mission outcome"]` | 1.5 |
| Red neutralization | `weights["Interception"]` | 2 |
| Threat containment | `weights["Zone protection"]` | 2 |
| Objective survival | `weights["Zone protection"]` | 2 |
| Mission outcome | `weights["Mission outcome"]` | 1.5 |
| Timeout | `weights["Timeout"]` | 1.5 |
| Blue separation | constante runtime | 1 |
| Restricted zone clearance | constante runtime | 1 |
| Boundary clearance | `kGeneralizedBoundaryClearanceWeight` | 0.01 |
| Low speed risk | `generalized.lowSpeedRiskWeight` | 5 |
| Own loss | `generalized.ownLossWeight` | 24 |

Modifier une clé historique peut donc affecter plusieurs canaux. Changer
`Interception` modifie à la fois la poursuite dense et la neutralisation
clairsemée. Changer `Mission outcome` modifie l'issue de mission et la perte
collective Bleue.

Les réglages de shaping tels que `controlEffortWeight`, les tolérances de
formation et `stepRewardScale` interviennent avant ces poids de canaux.
Pour le shaping de patrouille N=0, le signal est divisé par
`max(poidsDuCanal, 1)` avant l'agrégation : un poids supérieur ou égal à 1 ne
multiplie pas `stepRewardScale`; un poids inférieur à 1 réduit le signal.

## PPO et rollout

La collecte et l'optimisation sont dans
[`generalized_patrol_learner.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_patrol_learner.hpp).

| Paramètre | Valeur compilée ou configurée |
| --- | ---: |
| `rolloutSteps` | JSON, 2048 actuellement ; compte des pas synchronisés d'équipe |
| transitions par pas | P, une par participant initial |
| taille de batch | 1024 |
| époques PPO | 4 |
| gamma | 0.99 |
| lambda GAE | 0.95 |
| clip PPO | 0.2 |
| coefficient d'entropie | 0.01 |
| learning rate acteur / critic | 0.0003 / 0.0003 |
| optimiseur | Adam |
| normalisation des observations | non ; les features sont normalisées à la construction |
| normalisation des avantages | oui |

Un participant détruit conserve une ligne stable et les retours coopératifs,
mais son masque empêche tout gradient acteur sur une action impossible. Une
terminaison ne bootstrappe pas la valeur suivante. Une troncature de temps
conserve le bootstrap du critic.

## Conditions de fin

[`generalized_episode_outcome.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_episode_outcome.hpp)
sépare explicitement terminaison et troncature.

- Terminaison : échec de l'équipe, objectif détruit, tous les Rouges éliminés
  sans activation retardée en attente, ou tour N=0 physiquement validé.
- Troncature : horizon de pas atteint sans condition terminale.
- En N=0, `maxEpisodeSeconds` borne en plus la durée simulée. Avec la valeur
  actuelle de 480 s et `dt=0.2`, la limite vaut 2400 pas.
- Le succès N=0 exige les dix waypoints dans l'ordre et le tour physique
  qualifié ; un timeout n'est pas un succès.
- La perte du leader termine un épisode N=0. La perte ou l'erreur de formation
  d'un équipier conserve ses pénalités propres, mais ne disqualifie pas le tour
  correctement suivi par un leader vivant.

## Points d'entrée par besoin

| Besoin | Fichiers à modifier ou vérifier |
| --- | --- |
| Changer P/N, les spawns ou les zones | [`BlueDefense.json`](../resources/settings/Scenarios/BlueDefense.json), [`spawn_config.cpp`](../Examples/BlueDefense/Scenario/spawn_config.cpp), [`initializer.cpp`](../Examples/BlueDefense/Scenario/initializer.cpp) |
| Changer la dynamique du véhicule | [`defender_car.json`](../resources/settings/Platforms/Mobile/defender_car.json), [`flightmodelsystem.cpp`](../Framework/Simulation/World2D/src/flightmodelsystem.cpp), [`patrol_config.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/patrol_config.hpp) |
| Changer la géométrie ou les poids existants | [`BluePatrol.json`](../resources/settings/Training/BluePatrol.json), [`patrol_config.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/patrol_config.hpp) |
| Changer une feature d'observation | [`generalized_observation.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/generalized_observation.hpp), [`generalized_runtime_state.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_runtime_state.hpp), [`generalized_attention_actor.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_attention_actor.hpp) |
| Changer le rôle ou les slots wedge | [`blue_team_roles.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/blue_team_roles.hpp), [`patrol_geometry.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/patrol_geometry.hpp), [`generalized_runtime_state.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_runtime_state.hpp) |
| Changer les actions | [`patrol_core.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/patrol_core.hpp), [`generalized_patrol_team_agent.cpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp), [`flightmodelsystem.cpp`](../Framework/Simulation/World2D/src/flightmodelsystem.cpp) |
| Changer l'acteur | [`patrol_rltools.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/patrol_rltools.hpp), [`generalized_attention_actor.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_attention_actor.hpp) |
| Changer le critic | [`generalized_critic_observation.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_observation.hpp), [`generalized_critic_network.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_network.hpp), [`generalized_patrol_learner.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_patrol_learner.hpp) |
| Changer un reward existant | [`generalized_reward_shaping.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_reward_shaping.hpp), [`generalized_cooperative_reward.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Core/generalized_cooperative_reward.hpp), [`generalized_patrol_team_agent.cpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp) |
| Changer succès, échec ou timeout | [`generalized_episode_outcome.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_episode_outcome.hpp), [`generalized_patrol_team_agent.cpp`](../Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp) |
| Changer sauvegarde ou compatibilité | [`generalized_checkpoint.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_checkpoint.hpp), [`generalized_training_checkpoint.hpp`](../Examples/BlueDefense/Agents/BluePatrol/Training/generalized_training_checkpoint.hpp) |
| Changer promotion/évaluation | [`generalized_evaluation_campaign.hpp`](../Examples/BlueDefense/Evaluation/generalized_evaluation_campaign.hpp), méthodes `StartEvaluation` et `PromoteEvaluatedActor` du contrôleur |

## Procédures d'adaptation

### Ajuster sans changer le contrat réseau

Pour modifier les effectifs, les positions initiales, la géométrie, les
tolérances ou un poids existant :

1. modifier le JSON concerné ;
2. vérifier la validation dans `patrol_config.hpp` ou `spawn_config.cpp` ;
3. ajouter ou adapter un test déterministe ;
4. réentraîner ou reprendre uniquement avec une décision explicite sur le
   changement d'objectif ;
5. évaluer sur plusieurs graines avant toute promotion.

Un changement de reward ne change pas la taille des poids neuronaux, mais il
change l'objectif optimisé. La compatibilité binaire d'un checkpoint n'est pas
une justification suffisante pour reprendre son optimiseur.

### Ajouter ou modifier une entrée

1. modifier `GeneralizedObservationLayout` et documenter l'index ;
2. écrire la valeur et son masque dans `BuildGeneralizedObservations` ;
3. adapter le compactage de `GeneralizedAttentionActor` ;
4. adapter l'entrée du critic si la feature doit être centralisée ;
5. incrémenter la version sémantique d'observation et les formats concernés ;
6. rejeter explicitement les anciens checkpoints ou écrire une migration
   testée ;
7. repartir de zéro si la signification présentée à l'acteur change.

Réutiliser un index avec une nouvelle signification sans changer la version est
interdit, même si la dimension reste identique.

### Ajouter ou modifier une sortie

1. modifier `ObservationLayout::ActionSize` et le type `Action` ;
2. définir unité, plage et comportement de saturation ;
3. appliquer la commande dans `ChooseAction` et dans la physique ECS ;
4. adapter la pénalité d'effort et les tests ;
5. mettre à jour les en-têtes de checkpoints ;
6. effectuer un nouvel entraînement.

### Modifier l'architecture

Une largeur, un nombre de couches, un embedding d'attention ou la taille du
critic modifie la forme des poids. Il faut mettre à jour les contrôles de
checkpoint, les tests de lecture/écriture et démarrer un entraînement neuf,
sauf migration de poids explicitement conçue et testée.

### Modifier les rewards

Ne pas ajouter un canal pour contourner un comportement mal diagnostiqué.
Vérifier d'abord la physique, l'observation, le masque, le rôle et la condition
de fin. Si un changement de canal est explicitement décidé, mettre à jour dans
le même changement :

- `GeneralizedRewardChannel`, les noms et le nombre de canaux ;
- le calcul de shaping et l'agrégation par rôle ;
- le mapping JSON ;
- le snapshot IHM et la télémétrie ;
- les checkpoints si le contrat sérialisé change ;
- les tests numériques et la documentation.

Pour N=0, le reward de tour existant du leader doit rester indépendant des
erreurs ou pertes de formation des équipiers. Les collisions et la perte
individuelle restent autoritaires.

## Checkpoints

| Artefact | Rôle |
| --- | --- |
| `resources/model/BluePatrol/latest/blue_patrol_generalized_v10.txt` | acteur courant pour reprise et évaluation |
| `resources/model/BluePatrol/latest/blue_patrol_generalized_training_v7.txt` | acteur, critic, Adam, RNG et compteurs |
| `resources/model/BluePatrol/prod/blue_patrol_generalized_v10.txt` | acteur promu utilisé en production |

Le checkpoint acteur enregistre la version de politique et la version
d'observation. L'état complet enregistre aussi l'architecture du critic et les
optimiseurs. Un changement de sens d'une feature nécessite une nouvelle
version même si toutes les dimensions restent identiques.

## Tests à exécuter

Tests ciblés selon la modification :

| Modification | Tests principaux |
| --- | --- |
| observation, masques, P/N | `BlueDefenseGeneralizedRuntimeStateTests`, `BlueDefenseGeneralizedActorTests` |
| rewards | `BlueDefenseGeneralizedRewardEvaluationTests`, `BlueDefenseGeneralizedRewardShapingTests` |
| GAE, terminal/troncature | `BlueDefenseGeneralizedAdvantageTests`, `BlueDefenseGeneralizedEpisodeOutcomeTests` |
| patrouille et waypoints | `BlueDefenseGeneralizedPatrolReferenceTests`, `BlueDefenseGeneralizedPatrolWaypointTests`, `BlueDefenseGeneralizedUnifiedGuidanceTests` |
| acteur, critic, rollout | `BlueDefenseAttentionTests`, `BlueDefenseGeneralizedLearnerTests` |
| configuration et scène | `BlueDefenseGeneralizedInitializerTests`, `BlueDefenseCombatRuntimeSmoke`, `BlueDefenseInferenceRuntimeSmoke` |
| checkpoint | `PolicyCheckpointTests` et tests de fixture d'inférence |

Les builds x64 doivent réutiliser exclusivement les répertoires imposés :

~~~powershell
cmake --build build/debug --config Debug
ctest --test-dir build/debug -C Debug --output-on-failure

cmake --build build/release --config Release
ctest --test-dir build/release -C Release --output-on-failure
~~~

Les régressions doivent couvrir au minimum P=1, N=0, plusieurs Rouges,
plusieurs pistes, collisions, masques, valeurs finies, bornes et distinction
terminaison/troncature. Une CI verte valide les contrats testés, pas la
convergence. La convergence exige des évaluations multi-seed indépendantes.
