# PPOTrial

Simulateur C++17 avec environnement ECS, agents de reinforcement learning et
interface graphique. Les défenseurs bleus sont pilotés par **un unique
contrôleur MAPPO P/N**. Les attaquants rouges utilisent RLMission en production.

## Démarrage

```powershell
cmake -S . -B build/debug -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Debug -DPPO_LIBTORCH_VARIANT=Debug -DPPO_BUILD_DESKTOP=ON
cmake --build build/debug --config Debug
build/debug/Main/Debug/PPO.exe --list-examples
build/debug/Main/Debug/PPO.exe --example blue-defense
build/debug/Main/Debug/PPO.exe --example blue-defense-inference
```

Les builds x64 utilisent toujours `build/debug` et `build/release`, conformément
à `AGENTS.md` et aux deux raccourcis présents à la racine.
Sans `--example`, l'application affiche le sélecteur graphique. Avec
`--check-example`, l'exemple est initialisé sans démarrer la fenêtre.

## Exemples

| ID CLI / sélection IHM | Description | Configuration |
|---|---|---|
| `airspace` | Environnement historique d'agents LibTorch | `Scenarios/Playground.json` |
| `blue-defense` | Entraînement MAPPO P/N contre RLMission | `Scenarios/BlueDefense.json` |
| `blue-defense-inference` | Inférence déterministe avec l'acteur promu `prod` | `Scenarios/BlueDefense.json` |

Les deux modes BlueDefense utilisent **exactement la même scène**, la même
physique et le même contrôleur ECS `GeneralizedPatrolMappoAgent`. La section
JSON `blueDefense` exige
`blueTeam.activeCount` de 1 à 9 et `redTeam.activeCount` de 0 à 10.
Les entités sont créées dynamiquement depuis les templates
`blue_template` et `red_template` ; un seul contrôleur distribue toutes
les actions Bleues.

Le mode `blue-defense` charge l'état d'entraînement `latest`. Le mode
`blue-defense-inference` charge uniquement l'acteur promu dans `prod`, produit
des actions déterministes et ne possède aucun rollout, critic, optimizer,
reward d'apprentissage, sauvegarde ou campagne d'évaluation. Dans les deux
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

Les **15 canaux de récompense** du snapshot IHM sont : Patrol tracking,
Formation keeping, Pursuit progress, Control effort, Blue survival,
Red neutralization, Threat containment, Objective survival,
Mission outcome, Timeout, Blue separation, Restricted zone clearance
Boundary clearance, Low speed risk et Own loss.

Les paramètres, le rollout et les poids de récompense sont définis dans
`resources/settings/Training/BluePatrol.json`.

## Sauvegardes et compatibilité

Les modèles généralisés se trouvent sous `resources/model/BluePatrol` :
`latest/blue_patrol_generalized_v10.txt` pour l'acteur et
`latest/blue_patrol_generalized_training_v7.txt` pour l'état complet
d'apprentissage. La promotion après évaluation utilise `prod/`.

Le guidage de patrouille de l'observation V10 n'a plus la sémantique V9 : les
checkpoints du run 22 restent archivés, mais ne sont ni chargés ni migrés vers
V10. Un nouvel entraînement est requis. Les lecteurs historiques conservés ne
réactivent pas d'ancien agent ou entraînement 2v2.
Les règles RLMission et les modèles des Rouges restent inchangés.

## IHM

Le sélecteur lit les exemples depuis `Examples/src/examplecatalog.cpp`,
sans cardinalité fixe. Après Play, les voitures sont pilotées par l'ECS
normal. Dans l'entrée `blue-defense-inference`, le bouton Play existant lance
directement l'acteur de production sans apprentissage ni évaluation.

Le mode multi-caméras est activé par défaut : la vue tactique est découpée en
P vues, une par Bleu, dans une grille 1×1 à 3×3. La distance, la hauteur et le
champ de vision sont partagés par toutes les caméras. La case
`Multiple blue cameras` permet de retrouver la vue tactique unique.

Les voitures utilisent un modèle procédural plus lisible avec éclairage,
vitrages, roues et accent de rôle. Un badge doré **L** identifie le leader de
l'épisode et un badge cyan **E** chaque équipier, dans la vue 3D, les vignettes
multi-caméras et la mini-carte. Le suivi privilégie un Bleu vivant, se recale
immédiatement lors d'un changement de cible ou d'un reset et reste indépendant
du nombre d'images par seconde. L'inspection affiche uniquement les données
opérationnelles utiles ; le navigateur de méta-réflexion a été retiré de l'IHM.

La case `Show patrol references` de **World and display settings** affiche les
références exactes du dernier calcul fourni à MAPPO, dans la vue 3D et sur la
mini-carte. Un anneau repère la projection instantanée utilisée pour mesurer le
suivi ; une flèche repère la cible anticipée et son cap, effectivement encodés
dans l'observation. Le leader `L` est doré et les slots `W1` à `W8` sont cyan.
Cet overlay à capacité fixe est exclusivement visuel et ne modifie ni la
politique, ni les rewards, ni la physique.

La case `Show patrol waypoints`, placée au même endroit, affiche `WP1` à
`WP10` en vue 3D et sur la mini-carte. Le waypoint actif est doré, les futurs
waypoints sont verts et ceux déjà validés sont grisés. L'état de la case est
persisté comme celui des références de patrouille.

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

Les runs d'entraînement distants peuvent être téléchargés, diagnostiqués et
comparés avec l'interface Windows sans dépendance externe
[Blue Training Diagnostics](tools/blue_training_diagnostics/README.md). Son
lanceur double-clic produit des rapports HTML autonomes et des courbes SVG dans
le dossier Téléchargements de l'utilisateur.

Une CI verte vérifie la compilation et les contrats testés ; la
**convergence** sur plusieurs seeds et compositions P/N nécessite
une campagne de validation distincte.

## Documentation

- [Guide des modes BlueDefense](docs/blue-defense-example.md)
- [Architecture MAPPO à effectifs variables](docs/mappo-variable-teams.md)
- [Configuration JSON et paramètres](docs/configuration-guide.md)
- [Instructions pour les contributions](AGENTS.md)
- [Entraînement distant MAPPO Windows Release / GitHub Actions](docs/remote-mappo-training.md)
- [Architecture du séquenceur de dix waypoints](docs/2026-11-02_Blue_V10_Ten_Waypoint_Patrol_Architecture.md)
- [Téléchargement, diagnostic et comparaison des runs Blue](tools/blue_training_diagnostics/README.md)

Le curriculum **V3 depuis zéro** et le contrat d’observation v10 sont décrits dans [docs/blue-mappo-v3.md](docs/blue-mappo-v3.md).
