# MAPPO Bleu — curriculum V3 (depuis zéro ou reprise V9 explicite)

V3 reste le nom du curriculum. Le numéro **v9** identifie séparément le
contrat d'observation et l'acteur : les huit premières colonnes de patrouille
changent de sens, malgré une dimension de réseau inchangée.

## Références et observation

Le runtime calcule une seule projection du leader par snapshot, puis P
références dans un tableau de capacité 9. ChooseAction conserve celles de
l'état avant ; SetNewState calcule celles de l'état après. Observation,
reward, tour qualifiant, métriques et diagnostics réutilisent ces tableaux.
Les primitives géométriques restent pures dans Core ; aucune référence ne
produit directement une commande ou une attribution de cible de combat.

Chaque référence contient une pose/vitesse instantanée de mesure et une
pose/vitesse avancée de 3 secondes pour l'observation. La dérivée analytique
du slot donne son cap et sa vitesse propres en virage. Les rewards utilisent
la référence instantanée, sans pénaliser le décalage du lookahead.

| Colonne patrouille | Contenu v9 |
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

En Pvs0, la victoire exige un tour physique complet, avec au moins **95 % de
la distance** parcourue dans les tolérances de tous les Bleus. La séparation
reste obligatoire à chaque pas ; perte d'un Bleu, marche arrière, arrêt ou
saut impossible invalident la progression. Une perte termine immédiatement
la mission Pvs0. Le budget de 5 % couvre les transitoires des raccords de
courbure de l'hippodrome ; il ne constitue pas une exemption de sécurité.
Le timeout reste une troncature et reçoit maintenant sa pénalité en Pvs0.
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
L'acteur est blue_patrol_generalized_v9.txt ; l'état complet est
blue_patrol_generalized_training_v6.txt, enveloppe V6 avec observation 9.
Les anciens candidats restent sous leurs anciens noms.

Les journaux contiennent MAPPO_PATROL_EPISODE (distance, distance qualifiée,
remises à zéro, violations, marge de séparation), MAPPO_PATROL_AGENT
(erreurs normalisées par Bleu) et MAPPO_ROLE_REWARD (moyennes pondérées par
pas d'agent et par rôle pour les canaux patrouille, formation, effort et basse
vitesse). L'évaluation réutilise la même géométrie de référence.

## Reprendre le candidat V9 du run 19 dans V3

Le mode par défaut V3 reste un entraînement intégral **depuis zéro**. Une
**importation explicite, unique** peut désormais démarrer les 1200 mises à jour
V3 à partir d'un checkpoint d'apprentissage V9 compatible, sans recréer
l'acteur, le critic ou l'optimiseur. L'enveloppe V6 avec observation 9 et
le marqueur UPDATES sont vérifiés. L'import ne convertit pas les anciens
modèles V8 et ne touche pas au modèle Rouge RLMission.

Pour la première exécution depuis le run 19 dans GitHub Actions :

- `curriculum_enabled=true` ; `curriculum_version=v3`
- `resume_run_id=41286282285`
- `curriculum_warm_start=true` **uniquement à l'import initial**
- `max_seconds=14400` ; `save_every_updates=1`

Les checkpoints complets `latest` et le candidat `prod` du run 19 sont
restaurés, mais seul le **checkpoint complet latest** de l'update 538 sert à
reprendre l'optimiseur. Le meilleur acteur `prod` reste isolé et disponible
comme référence 4vs4, sous réserve des règles normales de promotion.

Le progrès V3 est créé avec `baselineUpdates=538`,
`initializationSource=warm-v9`, `nextStage=0` et la même empreinte de
configuration V3. Les phases restent déterministes, et leur cible finale
est l'update 1738 (538 + 1200). Les 160 premières mises à jour de
patrouille 1vs0 comprennent déjà des rappels 4vs4 tous les 24/6
updates ; dans les autres phases, les 6 updates de répétition alternent
entre le 4vs4 et les tâches antérieures. L'objectif est de limiter l'oubli
du combat tout en acquérant la patrouille et le wedge, sans garantir une
convergence à l'avance.

**Pour la reprise après ce premier run V3 :** fournir l'identifiant du
nouveau run V3 comme `resume_run_id`, et repasser
`curriculum_warm_start=false`. La progression restaurée détermine la
phase et la répétition ; demander une nouvelle importation si une
progression existe déjà provoque un rejet explicite, plutôt qu'un reset.

Sans `resume_run_id`, le workflow utilise toujours `--fresh-start`
avec un V3 à update 0. Avec un `resume_run_id` V9 sans progression V3,
il exige explicitement `curriculum_warm_start=true`.
Les checkpoints V8 / historiques et les progressions V2 incompatibles
sont rejetés.

## Validation

Les tests déterministes couvrent P=1/2/3/9, N=0 et N>0, directions de tour,
dérivées des slots, masques, perte du leader, séparation, stationnement,
transitoires de vitesse et fin de tour. Les tests Python vérifient le départ
à zéro, la reprise compatible, le refus des anciens états et la répétition
des tâches. Exécuter Debug/Release et CTest ; vérifier la CI VS2022 avant
fusion. Seule une évaluation de politiques entraînées sur des graines
indépendantes pourra établir leur convergence. RLMission et ses poids ne
sont pas modifiés.
