# PPOTrial

PPOTrial est un ensemble de démonstrateurs de reinforcement learning en C++17,
entraînés et exécutés avec PPOKit (`ppokit-run`, `ppokit-viewer`, `ppokit-studio`) :

- **RLMission** : démonstrateur PPO 2D pour un agent individuel.
- **Blue MAPPO** : démonstrateur MA-PPO 2D pour une équipe de 1 à 9 Bleus face à
  0 à 10 Rouges ; les Rouges utilisent RLMission en production.
- **FlightControl3D** : démonstrateur PPO 3D d'approche et d'atterrissage.

## Démarrage

```powershell
cmake -S . -B build/debug -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Debug -DPPO_BUILD_DESKTOP=ON
cmake --build build/debug --config Debug
ppokit-run train --project Examples/BlueDefense/projects/BluePatrol
ppokit-run play  --project Examples/BlueDefense/projects/BluePatrol --model PROD --realtime
ppokit-run eval  --project Examples/Airspace/projects/RLMission
```

Les builds x64 utilisent toujours `build/debug` et `build/release`, conformément
à `AGENTS.md`. Chaque exemple est un projet PPOKit (`project.json` : DLL
d'environnement et de réseau, commandes Train, Evaluate et Inference) ;
`Lancer_PPOKit_Studio.bat` ouvre le studio, qui lance ces commandes, et
`ppokit-viewer` montre le monde d'une partie (Visualiseur du projet).

## Exemples

| Projet | Description | Visualiseur |
|---|---|---|
| `Examples/Airspace/projects/RLMission` | Démonstrateur PPO 2D RLMission : entraînement, jeu et évaluation | `airspace_visualiser` (vue 2D tactique) |
| `Examples/Airspace/projects/RLSNA` | Exemple PPO 2D complémentaire : interception RLSNA | `airspace_visualiser` |
| `Examples/BlueDefense/projects/BluePatrol` | Démonstrateur MA-PPO 2D : équipe bleue contre RLMission | `blue_visualiser` (vue 2D) |
| `Examples/FlightControl3D/projects/T6`, `PC7` | Démonstrateur PPO 3D : approche et atterrissage JSBSim | `flight_visualiser` (vue 3D) |

### Organisation du dépôt

Chaque exemple garde dans `Examples/<Nom>` son code, ses réglages, ses modèles
et ses projets PPOKit ; les couches qu'ils partagent sont dans `Examples/Shared`.

~~~text
PPOKit/                     framework commun (schémas, learners, studio)
Framework/                  ECS, environnement piloté par l'appelant, chargement des réglages, contrat des agents
Examples/
├── Shared/
│   ├── World2D/            entités 2D partagées (mobile, zones, SNA, LiDAR)
│   ├── settings/           Environment.json, AgentProfiles.json, Platforms/
│   └── 3D/                 modèles 3D des vues
├── Airspace/               agents rouges RLMission et RLSNA
│   ├── projects/           RLMission, RLSNA (projets ppokit-run)
│   ├── settings/           Scenarios/, Training/, Evaluation/
│   └── models/             RLMission/ et RLSNA/ (training.ppokit, prod/ et eval/)
├── BlueDefense/
│   ├── projects/BluePatrol scenario.json, blue_patrol_training.json, checkpoints/
│   └── curricula/          BluePatrolCurriculumV3.json
└── FlightControl3D/
    └── projects/T6, PC7    scenario.json, checkpoints/ (T6 : task, network, training, eval/)
~~~

Le T-6 (`Examples/FlightControl3D/projects/T6`) s'entraîne avec `ppokit-run` : la
simulation est la DLL `flight_environment` (scénario `scenario.json`), le réseau la
DLL `ppokit_graph_network` (`network.json`, `training.json`), et le Visualiseur
`flight_visualiser` envoie le vol à `ppokit-viewer`. Le checkpoint
`checkpoints/latest.ppokit` reprend l'update 870 au niveau de curriculum 12 du
run CI 53. Les poids, les deux moments d'Adam et l'état aléatoire du learner
proviennent de l'archive FlightControlHeadless. À difficulté maximale, les deux
moteurs obtiennent 19 arrêts sur 20 avec les graines 4342–4361 et 20 sur 20 avec
les graines 90001–90020. L'update 340 du niveau 11 reste une référence de test.
[Run CI 53](https://github.com/benoitfragit/PPOTrial/actions/runs/45572775396) ·
[Comparaisons par tenseur et par épisode](docs/audits/ppokit-integration/README.md)

La suite `eval/validation.json` reprend la validation de
FlightControlHeadless : 20 épisodes indépendants toutes les 10 updates, 92 % d'arrêts
sur la piste, BEST classé par difficulté, taux d'arrêt puis distance d'arrêt
(`checkpoints/best/`), entraînement fini quand une validation passe sur la tâche
complète. `Lancer_FlightControl3D_Curriculum_Release.bat` ouvre le projet dans le
studio (Train, Evaluate, Inference, View).

```
ppokit-run train --project Examples/FlightControl3D/projects/T6 [--until-passed validation]
ppokit-run eval  --project Examples/FlightControl3D/projects/T6 [--best]
ppokit-run play  --project Examples/FlightControl3D/projects/T6 --best --realtime --step-seconds 0.002
```

Le Pilatus PC-7 JSBSim a son propre scénario dans `Examples/FlightControl3D/projects/PC7`.
Le projet utilise les mêmes DLL et la même architecture de réseau que le T-6.
Le premier entraînement du projet livré part de zéro. Son BEST historique,
l'update 610, provient du [run PC-7](https://github.com/benoitfragit/PPOTrial/actions/runs/45460292649).
Le lanceur `Lancer_FlightControl3D_PC7_Curriculum_Release.bat` ouvre le projet dans le Studio.
Chaque avion a son propre modèle. `FlightControlHeadless --trim --config FICHIER` donne le point
neutre des actions d'un avion (`initial.elevatorTrim` et `initial.throttle`) : trim
JSBSim quand il fonctionne, sinon trim par vol simulé (cas du turbopropulseur du PC-7).

L'inférence d'un projet de vol (bouton Inference du studio) fait voler la
politique en accéléré (`ppokit-run play --realtime --step-seconds 0.002`) : chaque
action avance de 0,1 seconde simulée, avec une attente de 2 ms comme dans l'ancien
FlightControl3D. La vitesse effective dépend du calcul et du système. Pour un vol
proche du temps réel, utiliser `--step-seconds 0.1`. Le vol continue épisode après
épisode jusqu'au Stop, sans toucher au curriculum ; `ppokit-viewer` montre la scène
3D, le profil d'approche et les panneaux de FlightControl3D (vol, commandes et
récompenses, diagnostics, raisons du crash).

La piste mesure 3 600 × 120 m et sa zone plane s'étend à 50 m de chaque
bord. Le profil d’approche de base part à 7–14 km du seuil et à 2,5–12° de chaque côté de
l’axe, avec un cap tiré indépendamment entre −20° et +20° du cap de piste.
L'altitude initiale suit le plan de 3° qui passe à 15 m au-dessus du seuil, avec
une variation de ±60 m. Un curriculum adaptatif commence par des finales courtes
alignées et élargit les départs à mesure que les arrêts réussissent ; les
validations jouent le même mélange (roulages, finales courtes à très déportées)
jusqu'au niveau courant, et seule la tâche complète peut valider la CI. Pour le T-6,
une promotion demande au moins 28 arrêts sur 30 épisodes d'entraînement en vol ;
un BEST demande au moins 19 arrêts sur 20 épisodes de validation. Le niveau déjà
sauvegardé reste acquis. Le train reste verrouillé sorti une fois commandé.
Dans `swell`, la houle fait pilonner le pont (±0,6 m, période 9 s).
Dans `moving-deck`, le pont avance aussi à 6 m/s. Ces paramètres viennent de
`deckMotion` dans chaque étape. Le dernier niveau, `runway-heading-offset`,
place l’avion à 3–7 km sur l’axe et le plan, avec une route initiale décalée
de 0 à 10° de part ou d’autre de la piste. Pour les étapes de pont mobile, la vue dessine les crêtes de
houle et le sillage du pont, et le panneau Runway affiche sa vitesse et son pilonnement.

La caméra `Runway`, placée à 300 m du seuil ou à mi-piste si elle est courte,
suit l'avion avec un zoom adapté. Un crash déclenche une animation visuelle de
1,5 seconde, sans suspendre la simulation. La campagne et ses évaluations sont
décrites dans [FlightControl3D T-6 en CI](docs/PPOTrial_FlightControl3D_T6_Approach_CI.md).
L'architecture du code et le guide de reprise sont dans
[FlightControl3D : architecture](docs/FlightControl3D_Architecture.md).
Les résultats d'une politique doivent être évalués sur des graines indépendantes.

L'entraînement et l'inférence de BluePatrol utilisent **exactement la même
scène**, la même physique et le même contrôleur ECS `GeneralizedPatrolMappoAgent`. La section
JSON `blueDefense` exige
`blueTeam.activeCount` de 1 à 9 et `redTeam.activeCount` de 0 à 10.
Les entités sont créées dynamiquement depuis les templates
`blue_template` et `red_template` ; un seul contrôleur distribue toutes
les actions Bleues.

L'entraînement (`ppokit-run train`) reprend l'état d'entraînement `latest`.
L'inférence (`ppokit-run play --model PROD`) charge
uniquement l'acteur promu dans `prod`, produit des actions déterministes et ne
possède aucun rollout, critic, optimizer, sauvegarde ou campagne d'évaluation. Dans les deux
cas, les Rouges conservent leur contrôleur RLMission production historique.

## Apprentissage MAPPO

- Acteur partagé de largeur 128, observation masquée des pistes rouges et
  amies, pooling et attention à capacité fixe.
- Critic centralisé individualisé de largeur 256.
- Un seul hippodrome analytique par épisode : longueur, largeur,
  orientation, ancrage et wedge échantillonnés dans les plages configurées.
- Sans Rouge, le leader rejoint d'abord la piste, puis suit une séquence figée
  de dix waypoints équidistants en abscisse curviligne ; les équipiers dérivent
  leurs cibles wedge du waypoint actif.
- Un Bleu leader ; les P−1 autres se positionnent dans leur slot de
  formation, selon le cap **physique** du leader.
- Reward du leader fondé sur la progression, l'écart latéral et le cap.
  Reward des équipiers pénalisant la déviation hors tolérance.
- Reward partagé de poursuite : progression géométrique des Bleus vers
  les Rouges, indépendamment du déplacement propre des Rouges ;
  pondération selon le risque pour l'objectif.
- Le compromis patrouille/formation/poursuite varie avec la distance et
  la menace. Aucun ordonnanceur externe n'attribue les cibles.
- Capteurs : 61 rayons LiDAR ; zones interdites rouges et enveloppes de
  sécurité autour des Bleus. Les collisions physiques restent terminales.

Les **15 canaux de récompense** sont : Patrol tracking,
Formation keeping, Pursuit progress, Control effort, Blue survival,
Red neutralization, Threat containment, Objective survival,
Mission outcome, Timeout, Blue separation, Restricted zone clearance
Boundary clearance, Low speed risk et Own loss.

La géométrie et les poids de récompense de l'environnement sont définis dans
`Examples/BlueDefense/projects/BluePatrol/blue_patrol_training.json`. Le learner
partagé lit ses paramètres PPO et son rollout dans le `training.json` du projet.
La comparaison des réglages et des modèles avec master est détaillée dans
[l'audit PPOKit](docs/audits/ppokit-integration/customization-master-2026-11-27.md).

Dans PPOKit Studio, **Environment** permet de modifier les nombres et booléens
du scénario BluePatrol. La sauvegarde initialise une copie isolée avec le
chargeur BlueDefense et refuse les configurations invalides. Les changements
s'appliquent au prochain lancement. **Curriculum** présente le fichier partagé
`tools/ci/adaptive_curriculum/gates.json` : ses seuils, durées, options de zones,
avancement et période de répétition sont modifiables ; la séquence triangulaire
et les effectifs restent fixés par le contrat CI. Ce curriculum appartient au
lanceur CI : le bouton Train de Studio ne l'exécute pas. Modifier le fichier
change son empreinte et peut imposer un nouveau bootstrap des progrès CI.

## Sauvegardes et compatibilité

Les modèles du projet partagé se trouvent sous
`Examples/BlueDefense/projects/BluePatrol/checkpoints/shared` :
`latest/policy.ppokit` pour l'acteur, `latest/latest.ppokit` pour l'état
complet d'apprentissage et `prod/policy.ppokit` pour le modèle de production.
Les anciens fichiers `blue_patrol_generalized_v10.txt`
et `blue_patrol_generalized_training_v7.txt` (formats V10/V7) restent lisibles :
les chargeurs reconnaissent le format au contenu, et la CI place ceux des runs
précédents sous les nouveaux noms.

Le guidage de patrouille de l'observation V10 n'a plus la sémantique V9 : les
checkpoints du run 22 restent archivés, mais ne sont ni chargés ni migrés vers
V10. Un nouvel entraînement est requis. Les lecteurs historiques conservés ne
réactivent pas d'ancien agent ou entraînement 2v2.
Les règles RLMission et les modèles des Rouges restent inchangés.

## Vues

`ppokit-viewer` affiche le monde que le Visualiseur d'un projet envoie après
chaque pas. Dans Studio, Train ouvre cette vue automatiquement si le projet en
possède une ; **View** permet de la rouvrir. Pendant l'entraînement, l'image
peut rester figée durant l'optimisation ou les sauvegardes et reprendre à la
collecte suivante. La vue 2D de BluePatrol montre, vue de dessus, la zone défendue,
le rectangle d'échec, les zones interdites, l'objectif, l'hippodrome du leader
et ses waypoints (`WP1`…, actif doré, validés grisés), chaque Bleu avec son
cap, sa trace, son badge de rôle (`L` doré, `W1`… cyan), son cône
d'engagement, ses rayons LiDAR et sa référence de patrouille, chaque Rouge avec
son cap, sa trace et son cône, les explosions. La vue 2D d'Airspace
(RLMission, RLSNA) reprend la mini-carte tactique de l'ancien PPO.exe : zone
de travail, avions, objectifs, zones interdites, cônes d'engagement, traces,
avion contrôlé et halo de sa cible principale. Ces vues sont purement
visuelles : elles ne modifient ni la politique, ni les récompenses, ni la
physique.

## Validation

Pour exécuter les tests de base sans interface :

```powershell
cmake -S . -B build/core -DPPO_BUILD_DESKTOP=OFF -DPPO_BUILD_CORE_TESTS=ON
cmake --build build/core --config Release
ctest --test-dir build/core -C Release --output-on-failure
```

Pour la régression Desktop, les workflows CI Windows construisent
Debug et Release, puis lancent CTest : démarrage de l'exemple,
instanciation P/N, mouvement après Play, rôle leader, collisions,
LiDAR, checkpoints et format de snapshot.

Une CI verte vérifie la compilation et les contrats testés ; la
**convergence** sur plusieurs seeds et compositions P/N nécessite
une campagne de validation distincte.

Les régressions ciblées vérifient ensemble WPT1–WPT10, la validation physique
du tour, le bonus terminal existant et le bootstrap PPO/GAE au timeout de 480 s.
Elles couvrent aussi l'hystérésis sur deux observations et les valeurs numériques
des rewards avec guidage figé.

## Documentation

- [Index de la documentation](docs/README.md)
- [Schéma d'architecture des réseaux Blue](docs/architecture/blue-network-architecture.md)
- [Guide pratique pour adapter l'agent Blue](docs/adapter-agent-blue.md)
- [Guide des modes BlueDefense](docs/blue-defense-example.md)
- [Architecture MAPPO à effectifs variables](docs/mappo-variable-teams.md)
- [Configuration JSON et paramètres](docs/configuration-guide.md)
- [Instructions pour les contributions](AGENTS.md)
- [Entraînement distant MAPPO Windows Release / GitHub Actions](docs/remote-mappo-training.md)
- [Architecture du séquenceur de dix waypoints](docs/architecture/blue-patrol-waypoint-sequencer.md)

Le curriculum **V3 depuis zéro** et le contrat d’observation v10 sont décrits dans [docs/blue-mappo-v3.md](docs/blue-mappo-v3.md).
