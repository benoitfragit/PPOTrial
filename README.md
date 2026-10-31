# PPOTrial

Simulateur C++17 avec environnement ECS, agents de reinforcement learning et
interface graphique. Les défenseurs bleus sont pilotés par **un unique
contrôleur MAPPO P/N**. Les attaquants rouges utilisent RLMission en production.

## Démarrage

```powershell
cmake -S . -B build/desktop -G "Visual Studio 17 2022" -A x64 -DPPO_BUILD_DESKTOP=ON
cmake --build build/desktop --config Debug
build/desktop/Main/Debug/PPO.exe --list-examples
build/desktop/Main/Debug/PPO.exe --example blue-defense
build/desktop/Main/Debug/PPO.exe --example blue-defense-inference
```

Les chemins de l'exécutable dépendent du générateur et du preset sélectionné.
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
`latest/blue_patrol_generalized_v8.txt` pour l'acteur et
`latest/blue_patrol_generalized_training_v5.txt` pour l'état complet
d'apprentissage. La promotion après évaluation utilise `prod/`.

Les lecteurs des formats anciens V4/V7 sont conservés **pour migration
uniquement**. Ils ne réactivent pas d'ancien agent ou entraînement 2v2.
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

## Documentation

- [Guide des modes BlueDefense](docs/blue-defense-example.md)
- [Architecture MAPPO à effectifs variables](docs/mappo-variable-teams.md)
- [Configuration JSON et paramètres](docs/configuration-guide.md)
- [Instructions pour les contributions](AGENTS.md)
- [Entraînement distant MAPPO Windows Release / GitHub Actions](docs/remote-mappo-training.md)
