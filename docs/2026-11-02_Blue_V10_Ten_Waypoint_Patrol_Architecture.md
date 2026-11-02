# Blue MAPPO V10 — patrouille séquencée sur dix waypoints

Date : 2026-11-02  
Périmètre : `GeneralizedPatrolMappoAgent`, scénarios N=0 uniquement.

## Décisions et contraintes

Le guidage V9 recalculait continuellement une cible anticipée depuis la
projection du leader. Près de la piste, cette cible pouvait rester presque
immobile pendant que l'avion orbitait localement. V10 conserve la projection
comme **mesure de piste**, mais fige une suite de dix cibles pour imposer une
progression observable sur le tour.

Trois conséquences sont explicites :

1. Les formules, poids, normalisations, noms et ordre des rewards restent
   inchangés. Leurs valeurs peuvent néanmoins évoluer parce que la politique
   reçoit une référence de guidage plus stable.
2. La taille de l'observation ne change pas, mais la sémantique temporelle de
   ses colonnes de patrouille change. Les poids V9 du run 22 sont incompatibles
   avec V10 ; aucun chargement ou warm start silencieux n'est autorisé.
3. RALLY → FOLLOW est irréversible pour le tour. L'hystérésis à deux seuils
   stabilise l'acquisition du couloir avant cette transition ; elle ne crée pas
   d'oscillation de mode après la génération.

RLMission, la physique, les comportements Rouges, les capacités P=1..9 et
N=0..10 et les fonctions de reward ne sont pas modifiés.

## Machine d'état

`GeneralizedPatrolWaypointSequencer` est un objet à capacité fixe, sans
allocation dans la boucle temps réel. Il appartient au contrôleur de l'équipe,
donc au seul leader logique.

- **RALLY** : la cible d'observation reste la prévision historique calculée
  depuis la projection. Le couloir est acquis sous
  `rallyEnterDistanceMeters`; il est relâché au-dessus de
  `rallyExitDistanceMeters` tant que FOLLOW n'a pas commencé.
- **FOLLOW** : depuis l'abscisse projetée à l'acquisition, dix poses sont
  calculées par `EvaluatePatrolCurve()` aux distances `L/10 ... 10L/10`.
  Positions, caps tangents, vitesses et distances requises sont stockés dans
  un `std::array<..., 10>` et ne changent plus pendant le tour.

`ChooseAction` construit les références depuis le waypoint actif et les garde
fixes pour tout le pas physique. `SetNewState` observe le segment résolu et
avance au plus une fois. Le leader porte la validation ; les équipiers ne font
que recevoir leurs poses wedge dérivées de la pose active du leader.

La présence de tout Rouge configuré, y compris une activation retardée,
conserve le chemin de guidage et la durée d'épisode historiques.

## Validation physique d'un waypoint

Une validation exige simultanément :

- une progression projetée dans le sens configuré, bornée par la vitesse
  maximale et `dt` ;
- le franchissement dirigé du plan tangent et la proximité du segment de pas ;
- la distance cumulée requise, ce qui interdit de couper entre les deux
  branches proches de l'hippodrome ;
- les tolérances existantes de piste, cap, vitesse, wedge et séparation pour
  tous les Bleus.

Le rayon de passage est borné par un quart de l'espacement entre waypoints et
s'adapte à la tolérance de position ainsi qu'à la distance parcourue en deux
pas. Un saut impossible ou un retour significatif invalide le tour sans
rembobiner l'index. WPT10 exige presque une longueur de circuit complète et ne
produit `patrolCompleted` que lorsque le validateur physique historique
confirme le même tour. Tourner autour d'un ancien waypoint ne peut donc ni
accumuler dix validations ni produire le succès terminal.

## Reward, timeout et terminaison

La projection instantanée reste dans `PatrolTarget::pose`; le waypoint figé est
dans `guidancePose`. `GeneralizedPatrolConvergenceShaping()` et les autres
fonctions de reward ne sont pas modifiés. Le succès WPT10 réutilise le canal
terminal `Mission outcome` existant.

Pour N=0, `rewards.noThreatPatrol.maxEpisodeSeconds` vaut 480 par défaut. La
limite est convertie avec `ceil(seconds / dt)` puis bornée par la limite
globale existante : 480 s donnent 2400 pas à `dt=0.2`. Son expiration pose
`truncated=true`, `terminated=false` en l'absence d'une autre cause et ne
déclenche aucun succès. Les scénarios N>0 ne consultent pas cette limite.

## Observation et checkpoints

- observation : version 10 ;
- acteur : `BLUE_TEAM_MAPPO_V10`, fichier
  `blue_patrol_generalized_v10.txt` ;
- état complet : `BLUE_TEAM_TRAINING_V7`, fichier
  `blue_patrol_generalized_training_v7.txt`.

Le changement de version est volontaire malgré des dimensions identiques. Le
premier entraînement V10 doit repartir de zéro. Une reprise ultérieure exige
un état complet V7 portant explicitement l'observation 10 et la même empreinte
de curriculum.

## Diagnostic et rendu

`MAPPO_PATROL_EPISODE` publie le mode, l'index actif, le nombre validé, la
distance au waypoint, l'écart latéral réel, l'erreur de cap, la vitesse,
l'abscisse parcourue et la cause `success`, `timeout` ou `failure`. Le rapport
HTML agrège ces champs sans ajouter de travail significatif par pas.

Le snapshot ECS transporte séparément dix entrées de rendu. La case
`Show patrol waypoints`, persistée comme `Show patrol references`, contrôle les
marqueurs `WP1` à `WP10` dans la vue 3D et la mini-carte. L'actif est doré, les
futurs sont verts et les validés sont gris. Cet overlay est en lecture seule et
n'intervient ni dans la politique, ni dans la physique, ni dans les rewards.

## Validation attendue

Les tests déterministes couvrent la transition, l'équidistance curviligne, les
deux sens, le séquencement exact, un seul avancement par pas, le raccourci, une
orbite locale, P=1/2/3/4/9, le wedge, le chemin combat historique, le timeout
et les en-têtes de checkpoint. Debug, Release et CTest vérifient les contrats
logiciels ; seule une campagne d'évaluation V10 sur plusieurs graines
indépendantes peut établir la convergence apprise.
