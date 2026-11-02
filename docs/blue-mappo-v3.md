# MAPPO Bleu — curriculum V3 et observation V10

V3 reste le nom du curriculum. Le numéro **v10** identifie séparément le
contrat d'observation et l'acteur. La taille du réseau reste inchangée, mais le
guidage de patrouille sans Rouge devient un waypoint figé : V9 et V10 ne sont
donc pas interchangeables.

## Références et observation

En N=0, le runtime commence en RALLY avec la référence historique issue de la
projection. Lorsque le couloir à hystérésis est acquis, il génère exactement
dix poses équidistantes sur un tour et entre définitivement en FOLLOW. Le
leader possède l'unique séquenceur ; les P−1 références wedge sont dérivées du
waypoint actif. ChooseAction conserve la cible pendant tout le pas, puis
SetNewState peut avancer d'un waypoint au maximum.

La projection analytique instantanée reste la référence de mesure de l'écart à
la piste, du cap et de la progression. Observation, reward, tour qualifiant,
métriques et diagnostics utilisent donc une mesure géométrique de piste et non
une fausse distance latérale au waypoint.
Les primitives géométriques restent pures dans Core ; aucune référence ne
produit directement une commande ou une attribution de cible de combat.

Chaque référence contient une pose/vitesse instantanée de mesure et une
pose/vitesse de guidage pour l'observation. Cette dernière est la prévision
historique en RALLY ou en combat et le waypoint actif figé en FOLLOW. Les
rewards continuent d'utiliser la référence instantanée.

| Colonne patrouille | Contenu v10 |
| --- | --- |
| 0–1 | Position de la cible avancée relative au Bleu, normalisée aux dimensions du monde |
| 2–3 | Vitesse vectorielle relative, normalisée à la vitesse maximale |
| 4 | Distance normalisée |
| 5–6 | Cosinus/sinus du gisement relatif |
| 7 | Validité de la référence |
| 8 | Rôle, inchangé |
| 9 | Priorité de menace continue |

Chaque Bleu voit une seule référence. Les vrais Bleus, Rouges et l'objectif
conservent leurs canaux d'attention et leurs masques. La présence d'un Rouge
éloigné ne supprime pas la référence. Padding, Bleu détruit et leader perdu
produisent une référence invalide, dont les huit colonnes sont nulles.
Les survivants PvsN conservent leurs observations de combat.

## Rewards et fin d'épisode

Le shaping combine la diminution du coût position/vitesse/cap entre deux
snapshots et la qualité du suivi pendant l'avancement physique du leader.
Le coût est une saturation rationnelle pondérée 0,5/0,3/0,2. Un stationnement
exact sur la piste n'offre aucun bonus. Les progrès inverses ou les sauts
d'abscisse impossibles ne donnent aucun bonus d'avancement.

stepRewardScale=0.004 borne le shaping par Bleu. Le runtime divise par
max(poids_du_canal, 1) avant agrégation : le poids formation de 4 ne
quadruple donc pas cette échelle effective. Les poids inférieurs à 1
diminuent le signal. Sous menace proche, le suivi de patrouille est réduit ;
le progrès de poursuite et la formation relative au leader physique prennent
progressivement la priorité. Les canaux de sécurité, de pertes et de
neutralisation conservent leurs identifiants et leur ordre.

En Pvs0, la victoire exige WPT1 à WPT10 dans l'ordre et un tour physique
complet, avec au moins **95 % de la distance** parcourue dans les tolérances de
tous les Bleus. Un franchissement dirigé, un rayon adapté à `dt` et à la vitesse
et une progression projetée plausible empêchent les raccourcis, le retour en
arrière et l'orbite locale. La séparation
reste obligatoire à chaque pas ; perte d'un Bleu, marche arrière, arrêt ou
saut impossible invalident la progression. Une perte termine immédiatement
la mission Pvs0. Le budget de 5 % couvre les transitoires des raccords de
courbure de l'hippodrome ; il ne constitue pas une exemption de sécurité.
Le timeout N=0 est 480 secondes simulées par défaut, soit 2400 pas à `dt=0.2`.
Il reste une troncature, sans succès, et reçoit sa pénalité en Pvs0.
En PvsN, la neutralisation du dernier Rouge reste terminale immédiatement,
hors activations différées encore attendues.

## Faisabilité du wedge

Avant tout placement, le sampler vérifie l'enveloppe conservatrice de tous
les slots sur le circuit complet, leur vitesse et leur rotation maximales,
ainsi que la séparation max(2.25*collisionDistance, minimumSeparationMeters).
Les distances entre slots sont invariantes par transformation rigide.

Les plages sont adaptées au monde et à P=9 : piste de 40–45 km sur 16–19 km,
séparation nominale de 2,7–3,2 km et angle de 30–35 degrés. Un grand wedge
pouvant tenir à une phase mais sortir du monde plus tard est refusé. Les
tentatives restent bornées et reproductibles. Le spawn utilise la vitesse
et le cap du slot, y compris dans les virages. Cette vérification ne prétend
pas supprimer les transitoires d'accélération aux raccords.

## Curriculum et lancement

| Phase | P/N | Zones | Updates |
| --- | --- | --- | ---: |
| Hippodrome | 1/0 | Non | 160 |
| Wedge | 2/0 | Non | 160 |
| Trois Bleus | 3/0 | Non | 80 |
| Quatre Bleus | 4/0 | Non | 80 |
| Combat | 1/1 | Non | 100 |
| Combat | 1/2 | Non | 60 |
| Combat | 2/1 | Non | 60 |
| Combat | 2/2 | Non | 100 |
| Combat | 3/2 | Oui | 80 |
| Combat | 3/3 | Oui | 80 |
| Patrouille avec zones | 4/0 | Oui | 80 |
| Référence finale | 4/4 | Oui | 160 |

Total : 1200 updates, rollouts de 2048 pas. Après la première phase, les
blocs alternent 24 updates de la tâche principale et 6 d'une tâche antérieure,
en rotation. Chaque phase se termine par sa tâche principale.

Workflow "MAPPO training (Windows Release)" :

- curriculum_enabled=true
- curriculum_version=v3
- resume_run_id vide pour le premier lancement
- max_seconds=14400 ; save_every_updates=1

Le budget horaire peut interrompre le programme avant 1200 updates. Reprendre
alors uniquement avec l'ID de ce nouveau run V3, son empreinte JSON et ses
checkpoints. Le runtime ne migre automatiquement aucun ancien poids.
L'acteur est blue_patrol_generalized_v10.txt ; l'état complet est
blue_patrol_generalized_training_v7.txt, enveloppe V7 avec observation 10.
Les anciens candidats restent sous leurs anciens noms.

Les journaux contiennent MAPPO_PATROL_EPISODE (distance, distance qualifiée,
remises à zéro, violations, marge de séparation, mode, index actif, nombre
validé, distance au waypoint, écart piste, cap, vitesse, abscisse et
terminaison), MAPPO_PATROL_AGENT
(erreurs normalisées par Bleu) et MAPPO_ROLE_REWARD (moyennes pondérées par
pas d'agent et par rôle pour les canaux patrouille, formation, effort et basse
vitesse). L'évaluation réutilise la même géométrie de référence.

## Incompatibilité explicite avec V9 et le run 22

Le séquencement change la signification temporelle de la cible encodée dans les
huit colonnes de patrouille. Charger un acteur V9 lui présenterait donc des
entrées de même dimension mais de sens différent. L'acteur V10, l'enveloppe
d'entraînement V7 et l'observation 10 sont vérifiés ensemble ; les checkpoints
V9 du run 22 sont conservés comme archives et rejetés par le runtime courant.

Le premier run V10 doit utiliser `--fresh-start` avec acteur, critic, Adam et
RNG neufs. Une reprise est autorisée uniquement depuis un checkpoint complet
V7/observation 10 et une progression V3 ayant la même empreinte JSON. Cette
règle ne touche ni le contrôleur RLMission ni les poids des Rouges.

## Validation

Les tests déterministes couvrent P=1/2/3/4/9, N=0 et N>0, les dix waypoints,
les deux directions, l'orbite locale, les raccourcis, les masques, le wedge,
la séparation, le timeout et la fin de tour. Les tests Python vérifient le départ
à zéro, la reprise compatible, le refus des anciens états et la répétition
des tâches. Exécuter Debug/Release et CTest ; vérifier la CI VS2022 avant
fusion. Seule une évaluation de politiques entraînées sur des graines
indépendantes pourra établir leur convergence. RLMission et ses poids ne
sont pas modifiés.
