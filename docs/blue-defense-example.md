# BlueDefense — MAPPO coopératif P/N

Le projet utilise **un seul contrôleur MAPPO généralisé** :
`GeneralizedPatrolMappoAgent`. Il pilote P Bleus (1 à 9) dans le
simulateur ECS. Les N Rouges (0 à 10) restent pilotés individuellement
par RLMission en mode production. Il n'existe plus d'agent PPO distinct
pour un scénario figé à deux Bleus.

## Une scène, deux modes d'exécution

| Exemple (sélecteur IHM / CLI) | Politique bleue | Scénario |
|---|---|---|
| `blue-defense` | learner MAPPO `latest`, entraînement et évaluations périodiques | `Scenarios/BlueDefense.json` |
| `blue-defense-inference` | acteur déterministe promu `prod`, sans entraînement ni évaluation | `Scenarios/BlueDefense.json` |

Le JSON déclare `blueDefense.blueTeam.activeCount` et
`blueDefense.redTeam.activeCount`. Le loader instancie dans les deux modes
`blue_1..blue_P` et `red_1..red_N` à partir de templates
`blue_template` et `red_template`. Le nombre de membres n'est
pas codé en dur dans l'IHM ni dans les classes du contrôleur.

### Exécution

```sh
PPO.exe --example blue-defense
PPO.exe --example blue-defense-inference
PPO.exe --example blue-defense --check-example
```

Sans option, le sélecteur graphique d'exemples propose les deux modes
et l'exemple Airspace. Ses cartes proviennent de `ExampleCatalog`,
sans logique spéciale pour une équipe de deux.

## MAPPO et récompenses

- Acteur partagé entre les P Bleus ; pistes rouges/amies masquées avec attention.
- Critic centralisé conditionné par l'identité de l'agent et ses observations.
- Leader : hippodrome analytique, progression, écart latéral et écart de cap.
- Équipiers : positions de formation dans le repère physique du leader.
- Poursuite : récompense liée aux mouvements des Bleus vers les menaces,
  sans cible prédéfinie.
- Défense : protection de l'objectif, neutralisation, survie, collisions et
  zones interdites.

Les **15 canaux** du snapshot sont : Patrol tracking, Formation keeping,
Pursuit progress, Control effort, Blue survival, Red neutralization,
Threat containment, Objective survival, Mission outcome, Timeout,
Blue separation, Restricted zone clearance, Boundary clearance,
Low speed risk et Own loss.

Les poids des canaux restent lus de `Training/BluePatrol.json`
et sont convertis explicitement au runtime généralisé.

## Intégration IHM

Le bouton Play utilise l'exécution ECS habituelle. Le mode est immuable dès le
chargement : Play entraîne dans `blue-defense` et lance l'acteur de production
dans `blue-defense-inference`. Le runner de production ne contient ni critic,
ni optimizer, ni buffer de rollout, ni API d'évaluation ou de sauvegarde.

La vue tactique multi-caméras, active par défaut, affiche une vue ajustée par
Bleu dans une grille bornée à neuf cellules. Les paramètres distance, hauteur
et champ de vision sont communs à toutes les vues. `Multiple blue cameras`
permet de revenir à la caméra unique historique.

## Compatibilité de checkpoints

Le moteur MAPPO conserve le chargement des modèles v8 ainsi que des lecteurs
de migration V4/V7 **uniquement pour convertir d'anciens poids**. Ces lecteurs
ne créent aucun contrôleur à deux agents. L'entraînement MAPPO s'enregistre
dans `resources/model/BluePatrol/latest`.

## Vérification

- Tests de contrat P=1,2,4,9 et N=0,2,4,10.
- Initialisation du wedge maximal, LiDAR et collisions.
- Test de rendu/snapshot et mouvement via le vrai ECS.
- Compilation VS2022 Debug/Release et CTest Core Windows/Linux.

Un résultat de CI ou de CTest vert démontre la validité du build et des
contrats testés, pas la convergence de la politique. Cette dernière
requiert des évaluations multi-seed indépendantes.
