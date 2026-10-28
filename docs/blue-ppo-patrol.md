# Patrouille bleue MAPPO avec RLTools

## Objectif

Les défenseurs bleus apprennent une politique coopérative pour tenir une
patrouille et protéger l'objectif contre les attaquants rouges. Le chemin
historique emploie `PatrolMappoAgent` pour deux bleus ; le chemin P/N optionnel
emploie `GeneralizedPatrolMappoAgent` pour 1 à 9 bleus. Les rouges restent
pilotés individuellement par le
`RLMission` de production existant ; son code, ses observations et ses modèles
ne sont pas modifiés.

Trois scénarios sont disponibles dans `PPO.exe` :

- `blue-patrol` entraîne la tenue de patrouille sans attaquant ;
- `two-vs-two` entraîne la protection face à zéro, un ou deux rouges RLMission,
  éventuellement activés en cours d'épisode, sur le chemin v7/v4 historique ;
- `p-vs-n` active le runtime généralisé v8/v5, configurable de P=1..9 et
  N=0..10 dans `Scenarios/PvsN.json`.

`Play` démarre la simulation et la collecte on-policy. `Pause` suspend aussi la
collecte et `Stop` arrête la boucle d'environnement. RLTools n'ajoute pas de
thread de simulation caché.

## Patrouille

Sur le chemin 2v2 historique, le circuit en hippodrome suit l'axe est/ouest,
qui est l'axe long du monde
configuré. Les demi-tours se font près des extrémités est et ouest. Il n'existe
qu'une piste : seul le leader y est projeté. L'équipier suit directement la
pose du leader avec un décalage arrière/intérieur défini par le wedge ; il ne
possède ni courbe ni reward de piste. À chaque reset, une permutation aléatoire
attribue exactement un rôle à chacune des deux voitures physiques.

La référence est recalculée depuis `Environment.json` : elle reste donc dans le
monde si ses dimensions changent.

Les bleus disposent d'un léger avantage cinématique pendant l'entraînement :
600 m/s² d'accélération, 420 m/s en vitesse maximale et 0,50 rad/s en
rotation maximale, contre 400 m/s et 0,40 rad/s pour les rouges. La patrouille
démarre à une vitesse de croisière configurable de 420 m/s. Leur domaine SNA
reste supérieur à celui des rouges : zone sans échappement de 5 km, zone
d'engagement maximale de 10 km et angle de tir de 1,10 rad. Les rouges
RLMission conservent leurs paramètres existants (4 km, 8 km et 1,047198 rad).

Ces valeurs se règlent dans
`resources/settings/Platforms/Mobile/defender_car.json`. `CruiseSpeed` est lu
par le spawn, la référence de patrouille, les observations et l'évaluateur afin
d'éviter des vitesses contradictoires entre entraînement et exécution.

## MAPPO et CTDE

Le contrôleur 2v2 historique applique le cycle ECS normal :

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

Les deux acteurs partagent strictement les mêmes poids. La défense, les
interceptions et les résultats de mission sont récompensés en commun. Le suivi
d'hippodrome ne récompense que le leader et le wedge que l'équipier ; le
critique partagé reçoit donc aussi le rôle dans l'observation. Il évalue les
retours pendant l'apprentissage et n'intervient pas dans les commandes en
inférence. Aucun coordinateur ne fournit de cible imposée aux acteurs.

## Observation locale et attention masquée

L'observation v7 du chemin historique contient 156 flottants : 124 locaux et
32 réservés au critique. La partie locale contient notamment :

- l'état propre et 61 rayons pseudo-LiDAR ;
- deux pistes rouges de 13 valeurs ;
- le coéquipier et l'objectif sous forme de pistes typées ;
- les erreurs de suivi de patrouille et les données globales locales.

Les 61 valeurs ne sont plus recalculées dans le domaine MAPPO lorsque le
runtime ECS est utilisé : `PatrolMappoAgent` copie le `sPseudoLidarState` du
bleu correspondant. Le composant fusionne la frontière rectangulaire et les
zones hostiles actives en conservant l'impact le plus proche. Les outils
headless sans ECS gardent le calcul de frontière historique comme repli, ce
qui préserve les tests et la dimension de l'acteur v7.

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

## Contrat généralisé P/N

Le runtime bleu généralisé est activé par la présence de `blueTeam` dans
l'objet `twoVsTwo`; `redTeam` dimensionne les attaquants sur ce chemin.
L'absence des deux blocs garde le chemin 2v2 legacy inchangé.
`blueTeam.activeCount` accepte `[1,9]` et `redTeam.activeCount` `[0,10]`.
L'entrée `p-vs-n` renseigne les deux blocs explicitement, alors que les
scénarios historiques continuent à les omettre.

`BlueTeamRoles` tire exactement un leader par épisode. Les autres indices
physiques obtiennent des slots de formation uniques `0..P-2`, stables pendant
l'épisode, alternés de chaque côté puis reculés par rangées pour former le V.
Le rôle et le slot normalisé dans `[-1,1]` font partie de l'observation.

Le cycle d'un pas reste strictement synchronisé :

1. construire les P observations depuis le même état ECS ;
2. évaluer une seule fois l'acteur partagé et produire P actions ;
3. laisser la physique et les règles résoudre le pas ;
4. enregistrer exactement P transitions avec le même index de pas d'équipe.

Une voiture détruite garde son indice stable mais son masque d'apprentissage
vaut zéro. Le learner préalloue son rollout et l'acteur ses matrices de travail
RLTools : ses chemins critiques `evaluate`, `forward` et `backward` ne font pas
d'allocation dynamique par appel.

`GeneralizedObservationLayout` définit une représentation brute v8 de 295
valeurs. Elle contient 7 valeurs propres, exactement 61 rayons LiDAR ECS, le
rôle/slot, 10 pistes rouges de 13 valeurs, 8 autres bleus de 9 valeurs,
l'objectif séparé sur 9 valeurs, puis patrouille, données globales et P/N
normalisés. Le dernier champ de chaque piste est son masque.

Cette représentation brute n'est pas injectée directement dans les couches
denses. `GeneralizedAttentionActor` applique aux ensembles rouges et amis une
attention, une moyenne et un maximum masqués, invariants à leur ordre, puis
construit un tronc compact de 156 valeurs. Les P/N et le slot utilisent des
colonnes historiquement nulles ; le suffixe central privilégié reste masqué.
Le runtime copie les 61 rayons depuis `sPseudoLidarState`, exactement comme le
chemin bleu v7.

Les `RestrictedZone` ne sont jamais des pistes mobiles : elles restent dans le
LiDAR/SAM dédié et ne consomment aucune des dix pistes classées de RLMission.
Seuls les `Mobile` et `Objective` opposés peuvent occuper ces dix slots, ce qui
préserve l'observation rouge de 152 valeurs.

Le spawn P/N est borné. Chaque rouge teste au plus
`redTeam.maximumPlacementAttempts` candidats, configuration limitée à 4096,
en respectant `redSpawn.minimumSeparation`. Une géométrie impossible produit
un diagnostic. Les délais d'activation sont positifs, finis et ordonnés ; un
`redTeam` généralisé ne peut pas être combiné avec le curriculum 2v2 actif.

Sur le chemin historique, les règles du monde testent aussi les trajectoires
balayées des deux voitures bleues entre deux pas ECS. Une distance minimale
inférieure ou égale à 1 km
détruit les deux véhicules, publie les deux explosions et termine l'épisode
avec la pénalité de mission, même si les voitures se sont croisées entre deux
positions échantillonnées. Le canal `Formation safety` applique une pénalité
progressive entre 2 km et 1 km et entraîne uniquement l'équipier sur la pose
relative au leader. Le suivi de référence du leader utilise une échelle locale
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

La récompense sépare les tâches de rôle et la défense coopérative :

- le suivi du seul hippodrome concerne uniquement le leader et diminue avec la
  pression des menaces ;
- le wedge concerne uniquement l'équipier, relativement au cap physique du
  leader. La pénalité `erreur / (1 + erreur)` conserve une variation même loin
  de la pose souhaitée, au lieu d'un plateau au-delà d'une séparation ;
- avec un seul rouge, le mutual support reste actif ; avec deux rouges, la
  pression de menace peut relâcher le wedge, jamais la sécurité de séparation ;
- les deux bleus reçoivent la même récompense de poursuite, d'interception et
  de résultat de mission. Aucun numéro de cible n'est envoyé à un acteur ;
- le coût de couverture partagé évalue les deux appariements possibles sous
  deux menaces, sans en imposer un à l'exécution. Son amélioration d'un pas au
  suivant récompense le rapprochement et l'effort de mise en cap. Une position
  immobile face au rouge ne produit pas de bonus permanent ;
- toutes les pistes actives participent immédiatement au signal de défense,
  indépendamment du seuil de risque. Le pooling et l'attention restent inchangés ;
- atteindre l'horizon sans menace n'est pas pénalisé ; `Timeout` vaut -15
  lorsqu'une menace reste active.

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
dans `Scenarios/TwoVsTwo.json`. Le profil courant active deux rouges dans 77,5 %
des épisodes, un rouge dans 15 % et aucun dans 7,5 %. Toutes les menaces sont
actives dès le début de l'épisode, sans activation retardée. La destruction de
la première ne termine pas l'épisode tant
qu'une autre activation est planifiée. Le canal `Formation safety`, propre à
l'équipier, utilise un poids global de 4, une pénalité d'erreur de 2 et une
tolérance de 3 %. Ce mécanisme ne change ni le réseau ni le contrôleur
RLMission.

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

Le contrat d'acteur du chemin historique est v7. Le candidat acteur est écrit dans
`resources/model/BluePatrol/latest/blue_patrol_mappo_v7.txt` ; seul le meilleur
acteur évalué est écrit dans
`resources/model/BluePatrol/prod/blue_patrol_mappo_v7.txt`. Au redémarrage,
`latest` est chargé en priorité, puis `prod` si aucun candidat n'existe.

La reprise d'entraînement complète utilise en plus
`resources/model/BluePatrol/latest/blue_patrol_training_v4.txt`. Ce fichier
atomique contient l'acteur partagé, le critique centralisé, les moments et les
compteurs Adam des deux optimiseurs, le nombre de mises à jour et le RNG
RLTools. Le buffer de rollout et une éventuelle transition en attente ne sont
volontairement pas sérialisés : la reprise ouvre un rollout vide avec le même
niveau d'entraînement. Cela permet aussi de changer `rolloutSteps` entre deux
exécutions sans invalider le modèle ni les optimiseurs.

Les contrats antérieurs sont incompatibles et refusés explicitement : les
poids conditionnés par les anciennes affectations ne sont pas réutilisés.
Les anciens fichiers sont conservés sur disque et ignorés par les chemins v7/v4.
Les fichiers `.pt` de RLMission ne sont jamais écrits par ce contrôleur.

Le chemin P/N possède des noms séparés :

- `latest/blue_patrol_generalized_v8.txt` pour l'acteur d'inférence ;
- `latest/blue_patrol_generalized_training_v5.txt` pour l'état complet.
- `prod/blue_patrol_generalized_v8.txt` pour le meilleur acteur évalué.

L'évaluation P/N démarre uniquement à une frontière d'update (première update,
cadence `checkpointEvery`, puis fin). Elle emploie les moyennes déterministes
du même acteur partagé : aucun tirage gaussien, aucune transition de rollout et
aucune avancée du RNG RLTools. Le seed de campagne et les phases miroir rendent
les resets ECS reproductibles, tout en conservant les vrais contrôleurs rouges
RLMission et les zones du scénario. Les taux de survie bleue et de
neutralisation rouge sont d'abord divisés par P et N pour chaque épisode
(`N=0` vaut neutralisation complète), puis moyennés avec la survie de
l'objectif, la qualité leader/slots, le succès et les timeouts. La promotion
v8 et son `.score` P/N sont séparés de la promotion historique v7.

Au premier démarrage sans v8/v5, `GeneralizedPatrolMappoAgent` peut charger un
acteur v7 et migrer les encodeurs, requêtes, tronc, sortie et écart-type
compatibles. Les trois nouvelles colonnes ont poids et moments nuls ; la sortie
2v2 est vérifiée à `1e-6`. La méthode explicite
`GeneralizedPatrolLearner::MigrateV4TrainingState` sait aussi lire un état v4,
mais ne reprend que l'acteur et ses moments Adam compatibles. Le critique fixe
2v2, les états scalaires des optimiseurs, le RNG, le compteur et le rollout
sont réinitialisés. Ce n'est pas une reprise bit-à-bit de l'entraînement.

Ne jamais supprimer `blue_patrol_mappo_v7.txt` ni
`blue_patrol_training_v4.txt` : ils restent utilisés par le scénario legacy et
servent de sources de migration. Les écritures v8/v5 sont atomiques et portent
P, N et les capacités compilées ; une version incorrecte ou un fichier tronqué
est refusé.

## Build et validation

Le code est en C++17. Boucle rapide de développement Visual Studio Debug :

~~~powershell
cmake --build out/build/x64-Debug --config Debug --target TwoVsTwoSpawnTests TwoVsTwoTeamDomainTests TwoVsTwoTeamDynamicsTests TwoVsTwoGeneralizedActorTests TwoVsTwoGeneralizedRuntimeStateTests TwoVsTwoGeneralizedLearnerTests TwoVsTwoGeneralizedRewardEvaluationTests MissionTrackSelectionTests --parallel 4
ctest --test-dir out/build/x64-Debug -C Debug -R "^(TwoVsTwoSpawnTests|TwoVsTwoTeamDomainTests|TwoVsTwoTeamDynamicsTests|TwoVsTwoGeneralized.*|MissionTrackSelectionTests)$" --output-on-failure
~~~

La validation finale ajoute le build `PPO`, les smoke tests
`TwoVsTwoPatrolRuntimeSmoke`, `TwoVsTwoCombatRuntimeSmoke` et
`TwoVsTwoEmbeddedLearnerSmoke`, puis `PPO.exe --example p-vs-n
--check-example`. Le projet impose actuellement la plateforme x64 dans sa
configuration CMake. Les smoke tests valident les scénarios avec et sans
rouges, les observations masquées, l'attention, le rollout MAPPO, la physique
et l'évaluation. Ils
valident le chemin logiciel, pas la convergence finale de la politique.
