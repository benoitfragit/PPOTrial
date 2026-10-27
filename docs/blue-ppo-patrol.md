# Patrouille bleue MAPPO avec RLTools

## Objectif

Les deux défenseurs bleus apprennent une politique coopérative pour tenir une
patrouille et protéger l'objectif contre les attaquants rouges. Un unique
`PatrolMappoAgent` pilote les deux bleus avec un acteur partagé et un critique
centralisé. Les rouges du scénario `two-vs-two` restent pilotés par le
`RLMission` de production existant ; son code, ses observations et ses modèles
ne sont pas modifiés.

Deux scénarios sont disponibles dans `PPO.exe` :

- `blue-patrol` entraîne la tenue de patrouille sans attaquant ;
- `two-vs-two` entraîne la protection face à zéro, un ou deux rouges RLMission,
  éventuellement activés en cours d'épisode.

`Play` démarre la simulation et la collecte on-policy. `Pause` suspend aussi la
collecte et `Stop` arrête la boucle d'environnement. RLTools n'ajoute pas de
thread de simulation caché.

## Patrouille

Le circuit en hippodrome suit l'axe est/ouest, qui est l'axe long du monde
configuré. Les demi-tours se font près des extrémités est et ouest. Les deux
gardes partagent le même circuit en file leader–suiveur. Les lignes droites
parcourent presque toute la longueur de la zone ; les demi-tours conservent un
rayon complet de marge avec la frontière. Le déphasage longitudinal vaut 16 %
d'une ligne droite, avec un minimum de 2,5 fois la distance de prudence :
environ 9,6 km dans le monde distribué. Les demi-tours restent donc décalés.

La référence est recalculée depuis `Environment.json` : elle reste donc dans le
monde si ses dimensions changent.

Les capacités initiales des bleus sont alignées sur la plateforme historique
`trainingaircraft` : 600 m/s² d'accélération, 700 m/s en vitesse maximale et
0,7000006 rad/s en rotation maximale, soit environ 40,1°/s. La patrouille
démarre à une vitesse de croisière configurable de 300 m/s. Leur domaine SNA
est supérieur à celui des
rouges : zone sans échappement de 5 km, zone d'engagement maximale de 10 km et
angle de tir de 3,147198 rad. Les rouges RLMission conservent leurs paramètres
existants (4 km, 8 km et 1,047198 rad).

Ces valeurs se règlent dans
`resources/settings/Platforms/Mobile/defender_car.json`. `CruiseSpeed` est lu
par le spawn, la référence de patrouille, les observations et l'évaluateur afin
d'éviter des vitesses contradictoires entre entraînement et exécution.

## MAPPO et CTDE

Le contrôleur applique le cycle ECS normal :

1. lire les deux observations locales et l'état d'équipe ;
2. échantillonner les deux actions avec l'acteur partagé ;
3. laisser la physique et les règles ECS appliquer les commandes ;
4. calculer une récompense d'équipe et enregistrer les deux transitions ;
5. lancer GAE puis PPO lorsque le rollout est complet.

Chaque pas d'équipe produit deux transitions, une par défenseur. Un rollout de
2048 pas contient donc 4096 transitions acteur ; c'est également le réglage
par défaut.

L'entraînement suit le principe CTDE (centralized training, decentralized
execution) :

- l'acteur ne voit que les 124 entrées locales de son défenseur ;
- le critique voit en plus 32 entrées centralisées décrivant les deux bleus,
  les deux rouges, l'objectif et la phase de patrouille ;
- le suffixe centralisé est explicitement mis à zéro avant chaque passage dans
  l'acteur, y compris pendant l'entraînement.

Les deux acteurs partagent strictement les mêmes poids. La récompense d'équipe
est fournie aux deux trajectoires.

## Observation locale et attention masquée

L'observation complète contient 156 flottants : 124 locaux et 32 réservés au
critique. La partie locale contient notamment :

- l'état propre et 61 rayons pseudo-LiDAR ;
- deux pistes rouges de 13 valeurs ;
- le coéquipier et l'objectif sous forme de pistes typées ;
- les erreurs de suivi de patrouille et les données globales locales.

Une piste rouge encode la position et la vitesse relatives, la distance, le
relèvement, la vitesse, le type hostile, la position monde normalisée,
l'appartenance à la zone défendue et, en dernière valeur, le masque
vivant/disponible. L'ECS ne possède pas de points de vie continus : une entité
est vivante ou détruite. Ce masque binaire est donc l'équivalent exact du champ
`Available` de RLMission. Une piste absente ou détruite est entièrement nulle.

Comme dans RLMission, les pistes passent par un encodeur partagé, puis par :

- une attention par produit scalaire avec softmax masqué ;
- un pooling moyen masqué ;
- un pooling maximum masqué.

Les pistes invalides ne contribuent ni au contexte, ni au pooling, ni aux
gradients. Les pistes amies utilisent une branche séparée construite sur le
même principe.

Les règles du monde testent aussi les trajectoires balayées des deux voitures
bleues entre deux pas ECS. Une distance minimale inférieure ou égale à 1 km
détruit les deux véhicules, publie les deux explosions et termine l'épisode
avec la pénalité de mission, même si les voitures se sont croisées entre deux
positions échantillonnées. Le canal `Formation safety` applique une pénalité
progressive entre 2 km et 1 km et, sans menace, une récompense dense sur la
pose relative leader–suiveur. Le suivi de référence utilise une échelle locale
proche du rayon du circuit plutôt que les dimensions complètes du monde : une
dérive de quelques kilomètres n'est donc plus presque invisible.

La limite normale du monde est une frontière souple : la franchir ne termine
plus l'épisode. Le canal `Zone protection` applique une pénalité quadratique
sur les 25 % de profondeur suivants. Cette pénalité est complète sans menace
ou lorsqu'une menace est proche de l'objectif, puis descend jusqu'à 20 % pour
une menace éloignée afin d'autoriser sa poursuite. Seule la limite de sécurité
à deux fois les dimensions du monde reste terminale contre les dérives
numériques.

## Rollout, mini-batch et le nombre 128

Le rollout applicatif est configuré dans
`resources/settings/Training/BluePatrol.json`. Valeur par défaut :

~~~json
"rolloutSteps": 2048
~~~

L'IHM propose 512, 1024, 2048 et 4096 pas d'équipe. La taille ne peut être
changée à chaud qu'avant la première transition, car changer les dimensions
du buffer en cours d'apprentissage invaliderait son contenu et l'état de
l'optimiseur.

Le nombre **128 n'est pas la durée de l'entraînement**. C'est la taille interne
des mini-batchs utilisés pendant les époques PPO. Un test rapide utilise aussi
un rollout réduit à 128 pour garder un temps d'exécution court, mais
l'application n'utilise pas cette valeur par défaut.

À la fin de chaque rollout complet, PPO fait une mise à jour synchrone. Une
brève pause CPU peut donc être visible tous les 2048 pas par défaut, mais la
collecte reprend ensuite jusqu'à `maxUpdates`.

## Récompenses visibles dans l'IHM

La fonction de récompense change explicitement de mission selon les menaces :

- sans rouge actif, les deux bleus sont récompensés pour tenir le circuit ;
- avec un rouge, le bleu le plus proche l'intercepte et son équipier reste en
  patrouille de réserve ;
- avec deux rouges, une affectation un-à-un de distance totale minimale libère
  les deux bleus de la référence de patrouille ;
- l'intercepteur affecté est récompensé lorsqu'il devient plus proche de sa
  cible que son équipier ; avec deux menaces, l'écartement de l'équipe est
  également récompensé pour éviter que les voitures suivent le même danger ;
- le rapprochement de chaque intercepteur vers sa cible fournit un signal dense,
  puis la destruction et la victoire fournissent les bonus terminaux ;
- atteindre l'horizon sans menace n'est pas pénalisé ; `Timeout` vaut -15
  uniquement lorsqu'une menace reste active.

Le panneau de l'agent `Blue MAPPO (RLTools)` expose la valeur instantanée et le
coefficient de chaque canal :

- `Patrol tracking` ;
- `Formation safety` ;
- `Control effort` ;
- `Zone protection` ;
- `Interception` ;
- `Mission outcome` ;
- `Timeout`.

Le clamp et les coefficients peuvent être ajustés dans l'IHM. Leur valeur
initiale est lue dans `Training/BluePatrol.json`. Les canaux de protection
récompensent notamment l'éloignement des menaces de l'objectif et leur
interception ; la perte de l'objectif ou d'un défenseur est terminale.

Le scénario `two-vs-two` applique pendant l'entraînement un curriculum défini
dans `Scenarios/TwoVsTwo.json`. Par défaut, 10 % des épisodes n'activent aucun
rouge, 30 % en activent un et 60 % en activent deux. La première activation est
retardée dans 20 % des cas ; lorsqu'une seconde menace existe, elle est décalée
dans 45 % des cas. La destruction de la première ne termine pas l'épisode tant
qu'une autre activation est planifiée. Ce mécanisme ne change ni le réseau ni
le contrôleur RLMission.

## Évaluation automatique et promotion

Une première campagne est lancée après le premier rollout pour établir la
référence, puis toutes les `checkpointEvery` mises à jour PPO. La collecte est suspendue et
l'acteur courant est évalué sans bruit d'exploration dans le vrai ECS. Cette
campagne n'est pas l'ancien utilitaire de patrouille isolé : dans
`two-vs-two`, les rouges de production RLMission continuent à piloter les
attaquants. Les huit cas par défaut couvrent les deux côtés, plusieurs phases
du circuit et un mélange reproductible de 2v2 et de 2v1.

Dans `blue-patrol`, un cas réussit si les deux bleus restent sains, suivent la
patrouille et atteignent la durée configurée. Dans `two-vs-two`, il réussit si
l'objectif et les bleus survivent et si toutes les menaces initiales sont
détruites. Le score sur 100 pondère : 60 % succès de mission, 20 % survie de
l'objectif, 10 % survie des bleus et 10 % qualité de suivi.

L'IHM affiche la progression, les cas terminés, les timeouts, le débit, le
score courant et le meilleur score. L'apprentissage reprend automatiquement
après la campagne. Le candidat n'est copié dans `prod` que si son score est
strictement meilleur ; le meilleur score est conservé dans un fichier
compagnon `.score`. La barre n'est affichée que pendant une campagne active.

L'onglet `Diagnostics` publie à chaque mise à jour les pertes totale, acteur,
valeur et entropie du dernier mini-batch PPO RLTools. Avant le premier rollout
complet, il indique explicitement qu'aucune mise à jour n'est encore disponible.

## Configuration et checkpoints

`Training/BluePatrol.json` contient :

- `rolloutSteps`, `maxUpdates`, `checkpointEvery` et `seed` ;
- `evaluation.episodes`, `maxSteps`, `seed` et
  `minimumTrackingFraction` ;
- le clamp de récompense ;
- les sept coefficients nommés.

Le contrat d'acteur courant est v4. Le candidat acteur est écrit dans
`resources/model/BluePatrol/latest/blue_patrol_mappo_v4.txt` ; seul le meilleur
acteur évalué est écrit dans
`resources/model/BluePatrol/prod/blue_patrol_mappo_v4.txt`. Au redémarrage,
`latest` est chargé en priorité, puis `prod` si aucun candidat n'existe.

La reprise d'entraînement complète utilise en plus
`resources/model/BluePatrol/latest/blue_patrol_training_v1.txt`. Ce fichier
atomique contient l'acteur partagé, le critique centralisé, les moments et les
compteurs Adam des deux optimiseurs, le nombre de mises à jour et le RNG
RLTools. Le buffer de rollout et une éventuelle transition en attente ne sont
volontairement pas sérialisés : la reprise ouvre un rollout vide avec le même
niveau d'entraînement. Cela permet aussi de changer `rolloutSteps` entre deux
exécutions sans invalider le modèle ni les optimiseurs.

En l'absence de ce nouveau fichier, un ancien v4 reste accepté comme warm-start
de l'acteur uniquement ; le premier checkpoint complet sera créé après la
prochaine mise à jour PPO. Un checkpoint incompatible est refusé explicitement.
Les fichiers `.pt` de RLMission ne sont jamais écrits par ce contrôleur.

## Build et validation

Le code est en C++17. Exemple de build Visual Studio Debug :

~~~powershell
cmake --build out/build/x64-Debug --config Debug --target PPO
ctest --test-dir out/build/x64-Debug -C Debug -R "TwoVsTwo|BluePatrol" --output-on-failure
~~~

Le projet impose actuellement la plateforme x64 dans sa configuration CMake.
Les smoke tests valident les scénarios avec et sans rouges, les observations
masquées, l'attention, le rollout MAPPO, la physique et l'évaluation. Ils
valident le chemin logiciel, pas la convergence finale de la politique.
