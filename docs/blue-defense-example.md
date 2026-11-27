# BlueDefense — MAPPO coopératif P/N

Le projet utilise **un seul contrôleur MAPPO généralisé** :
`GeneralizedPatrolMappoAgent`. Il pilote P Bleus (1 à 9) dans le
simulateur ECS. Les N Rouges (0 à 10) restent pilotés individuellement
par RLMission en mode production. Il n'existe plus d'agent PPO distinct
pour un scénario figé à deux Bleus.

## Une scène, deux modes d'exécution

La scène est le projet PPOKit `Examples/BlueDefense/projects/BluePatrol`
(DLL d'environnement `blue_environment`, DLL réseau d'équipe `ppokit_graph_network`),
exécuté par `ppokit-run`.

| Commande | Politique bleue | Scénario |
|---|---|---|
| `ppokit-run train` | learner MAPPO `latest`, entraînement et évaluations périodiques | `scenario.json` du projet |
| `ppokit-run play --model PROD` | acteur déterministe promu `prod`, sans entraînement ni évaluation | `scenario.json` du projet |

Le JSON déclare `blueDefense.blueTeam.activeCount` et
`blueDefense.redTeam.activeCount`. Le loader instancie dans les deux modes
`blue_1..blue_P` et `red_1..red_N` à partir de templates
`blue_template` et `red_template`. Le nombre de membres n'est
pas codé en dur dans la vue ni dans les classes du contrôleur.

### Exécution

```sh
ppokit-run train --project Examples/BlueDefense/projects/BluePatrol --updates 100
ppokit-run eval  --project Examples/BlueDefense/projects/BluePatrol --model PROD
ppokit-run play  --project Examples/BlueDefense/projects/BluePatrol --model PROD --realtime --step-seconds 0.004
ppokit-viewer --view <build>/Examples/BlueDefense/Release/blue_visualiser.dll --port 47001
```

`ppokit-studio Examples/BlueDefense/projects/BluePatrol` ouvre le projet ; ses
boutons « Train », « Evaluate » et « Inference » lancent les commandes de
`project.json`, et l'inférence ouvre ppokit-viewer sur le monde. Les anciens
modes `blue-defense` et `blue-defense-inference` de PPO.exe ont été supprimés
avec lui : `Examples/BlueDefense/Scenario/bluedefenseexample.cpp` ne fournit
plus que `LoadExternalScenario`, qui charge le monde de la DLL
`blue_environment`.

## MAPPO et récompenses

- Acteur partagé entre les P Bleus ; pistes rouges/amies masquées avec attention.
- Critic centralisé conditionné par l'identité de l'agent et ses observations.
- Leader : hippodrome analytique, progression, écart latéral et écart de cap.
- Équipiers : positions de formation dans le repère physique du leader.
- Poursuite : récompense liée aux mouvements des Bleus vers les menaces,
  sans cible prédéfinie.
- Défense : protection de l'objectif, neutralisation, survie, collisions et
  zones interdites.

Les **15 canaux** de récompense sont : Patrol tracking, Formation keeping,
Pursuit progress, Control effort, Blue survival, Red neutralization,
Threat containment, Objective survival, Mission outcome, Timeout,
Blue separation, Restricted zone clearance, Boundary clearance,
Low speed risk et Own loss.

Les poids des canaux restent lus de `blue_patrol_training.json` (projet BluePatrol)
et sont convertis explicitement au runtime généralisé.

## Visualisation

Le mode est fixé par la commande : `train` entraîne, `play` lance l'acteur de
production. Le runner de production ne contient ni critic, ni optimizer, ni
buffer de rollout, ni API d'évaluation ou de sauvegarde.

L'observateur `blue_watch@2` du `network.json` du projet (port 47001) envoie le
monde après chaque pas ; le Visualiseur `blue_visualiser` en dessine une vue 2D
de dessus dans ppokit-viewer, reprise de la mini-carte tactique de l'ancien
PPO.exe, plus les pseudo-lidars et les explosions de sa vue 3D : zone défendue,
rectangle d'échec, zones interdites, objectif ; piste de patrouille du leader
et ses waypoints (actif, validés, cap) ; pour chaque Bleu son cap, sa traînée,
son badge de rôle (**L** pour le leader, **W1**... pour les équipiers), son cône
d'engagement, ses faisceaux de pseudo-lidar et sa référence de patrouille
(point projeté, point de guidage et cap) ; pour chaque Rouge son cap, sa
traînée et son cône (une croix une fois détruit). Le nord est en haut ; on
déplace la vue en la faisant glisser et on zoome à la molette ; un clic sur un
Bleu affiche son pseudo-lidar. La vue ne fait que lire les rôles et
références déjà calculés par le runtime : elle n'intervient ni dans la
physique, ni dans la politique.

## Compatibilité de checkpoints

Le runtime courant exige l'acteur V10 et l'état d'entraînement V7. Le guidage
figé change la sémantique des colonnes de patrouille sans changer leur taille :
les poids V9, dont ceux du run 22, sont donc incompatibles et ne sont pas
repris silencieusement. L'entraînement MAPPO V10 repart de zéro et
s'enregistre dans `Examples/BlueDefense/projects/BluePatrol/checkpoints/latest`.

## Vérification

- Tests de contrat P=1,2,4,9 et N=0,2,4,10.
- Initialisation du wedge maximal, LiDAR et collisions.
- Test du Visualiseur de bout en bout (`Examples/BlueDefense/tests/visualiser_tests.cpp` : ppokit-run joue BluePatrol, les trames arrivent en UDP et la vue dessine la scène) et mouvement via le vrai ECS.
- Compilation VS2022 Debug/Release et CTest Core Windows/Linux.

Un résultat de CI ou de CTest vert démontre la validité du build et des
contrats testés, pas la convergence de la politique. Cette dernière
requiert des évaluations multi-seed indépendantes.
