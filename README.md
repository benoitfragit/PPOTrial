# PPOTrial

PPOTrial est un simulateur C++17 d'apprentissage par renforcement multi-agent.
Le scénario distribué utilise un unique contrôleur bleu MAPPO/RLTools face aux
attaquants rouges RLMission. Son contrat permet d'instancier de 1 à 9 bleus
face à 0 à 10 rouges, sans créer une seconde entrée d'application.

## Démarrage rapide

Depuis un terminal Visual Studio configuré pour C++ :

~~~powershell
cmake --build out/build/x64-Debug --config Debug --target PPO
out/build/x64-Debug/Main/Debug/PPO.exe --example blue-patrol
out/build/x64-Debug/Main/Debug/PPO.exe --example two-vs-two
~~~

Le chemin exact de l'exécutable dépend du générateur CMake. Le projet impose
actuellement x64 ; les builds Debug sont recommandés pour le diagnostic.

## Scénarios principaux

- `airspace` : scénario historique et agents LibTorch ;
- `blue-patrol` : deux bleus MAPPO apprennent la patrouille sans ennemi ;
- `two-vs-two` : l'unique MAPPO bleu généralisé protège l'objectif dans un
  entraînement livré à P=2/N=4 contre des rouges pilotés par RLMission.

Le scénario `two-vs-two` matérialise `blue_1..blue_P` depuis les templates
typés du JSON. `blueTeam.activeCount` accepte 1 à 9 et
`redTeam.activeCount` accepte 0 à 10. La configuration livrée utilise quatre
rouges, dont chacun a 25 % de probabilité d'être activé avec retard. Les sorties sont
exclusivement
`blue_patrol_generalized_v8.txt` et
`blue_patrol_generalized_training_v5.txt` dans `BluePatrol/latest` ; les
checkpoints `blue_patrol_mappo_v7.txt` et `blue_patrol_training_v4.txt` restent
des sources de migration en lecture seule.

Dans le sélecteur, ces entrées sont libellées explicitement `BLUE PATROL ONLY
- NO RED` et `TWO VS TWO - MAPPO vs RLMission`. Le second scénario conserve
une seule entrée GUI mais matérialise quatre entités rouges RLMission. Les
activations retardées et les positions sont reproductibles avec le seed.

À chaque épisode P/N, les rouges sont tirés sur un arc allant de l'avant aux
deux côtés de la zone défendue. Ils commencent entre 1,05 et 1,35 fois la
limite du monde et à au moins 10 km l'un de l'autre. La limite est un repère de
zone, pas un mur : RLMission continue normalement à piloter les rouges vers
l'objectif. Les zones rouges sont décrites par la collection typée
`twoVsTwo.restrictedZones`. Deux centres sont tirés à l'intérieur du monde et
deux à l'extérieur, toujours du côté opposé à l'objectif ; au plus deux zones
actives appartiennent donc à chaque région. Elles sont vues
par les 61 rayons pseudo-LiDAR des bleus. Le chemin RLMission des rouges, son
observation 152 et ses checkpoints restent inchangés. La zone bleue intérieure
`blue_restricted_zone` est désactivée ; seule la bordure de la zone défendue
reste affichée.

La patrouille est un unique hippodrome analytique continu, sans waypoint. Sa
longueur, sa largeur, son orientation, son ancrage et le wedge sont tirés une
fois par épisode. Les deux voitures reçoivent alors aléatoirement des rôles
bijectifs : exactement un leader et un équipier. Seul le leader suit la piste ;
l'équipier tient une pose arrière/intérieure relative au leader. Un seul
contrôleur MAPPO possède l'acteur partagé, le critique centralisé, le buffer
on-policy et l'optimiseur des deux bleus.

La configuration livrée conserve une longueur de 50 à 60 km et échantillonne
une largeur de 16 à 20 km. Ces rayons progressifs facilitent l'exploration
initiale du leader et la tenue du wedge dans les demi-tours.

La plateforme bleue reprend les limites de `trainingaircraft` : vitesse de
croisière et vitesse maximale de 700 m/s, contre 400 m/s pour les rouges,
accélération de 600 m/s² et rotation maximale de 0,700000596 rad/s, contre
0,40.
Sa zone SNA sans
échappement vaut 5 km et sa zone d'engagement atteint 10 km, contre 4 et 8 km
pour les rouges. Son domaine angulaire de destruction vaut 3,147198 rad, contre
1,047198 rad pour les rouges. Les rouges restent inchangés.

## Configuration bleue

Les réglages sont dans
`resources/settings/Training/BluePatrol.json`. Le rollout par défaut est de
2048 pas d'équipe, soit 4096 transitions pour deux défenseurs. L'IHM permet de
choisir 512, 1024, 2048 ou 4096 avant le début de la collecte. La valeur 128 du
backend est la taille d'un mini-batch PPO et non la durée de l'entraînement.

Le panneau `Blue MAPPO (RLTools)` expose les récompenses de patrouille, sécurité
de formation, effort de contrôle, protection de zone, interception, résultat
de mission et timeout.

Une évaluation déterministe intégrée est lancée à chaque checkpoint : elle
utilise les vrais rouges RLMission du scénario P/N, publie ses mesures dans l'IHM et
ne promeut dans `resources/model/BluePatrol/prod` que le meilleur acteur. Les
acteurs non encore validés restent dans `BluePatrol/latest`. Le même dossier
contient `blue_patrol_training_v4.txt`, qui restaure l'acteur, le critique,
les deux états Adam, le compteur de mises à jour et le RNG. Le buffer de rollout
n'est volontairement pas sauvegardé : une reprise commence un rollout vide au
même niveau d'optimisation.

La collision entre les deux bleus est terminale à 1 km, avec une pénalité de
sécurité progressive à partir de 2 km. Le leader reçoit seul le suivi de
l'unique hippodrome ; l'équipier reçoit seul le wedge de 6 à 9 km, mesuré dans
le repère du cap physique du leader. Les deux bleus voient toutes les pistes
rouges actives et apprennent à les couvrir par une récompense défensive commune
et le critique centralisé. Il n'existe aucun coordinateur ni cible imposée.
Le pooling et l'attention sont inchangés. Deux menaces simultanées peuvent
relâcher le wedge ; la séparation de sécurité reste toujours active.

Les 61 entrées LiDAR de l'acteur v7 consomment désormais directement le
`sPseudoLidarState` ECS partagé avec RLMission. Chaque rayon conserve le plus
proche impact entre frontière du monde et zone hostile active. Une zone utilise
le rayon létal du profil (`MaxNoEscapeZoneRange`, 7 km dans l'exemple) et un
rayon d'avertissement LiDAR distinct de 110 % (7,7 km). Seuls les mobiles bleus
sont publiés par l'adaptateur TwoVsTwo; les rouges gardent strictement leur
pipeline RLMission historique.

Le contrat bleu v7 démarre un nouvel apprentissage : les anciens poids bleus
conditionnés par les affectations v6 ne sont pas repris. Les nouveaux fichiers
sont `blue_patrol_mappo_v7.txt` et `blue_patrol_training_v4.txt`.
Les anciens fichiers restent sur disque mais sont ignorés. Aucun modèle rouge
n'est concerné.

L'ajout des zones ne change ni la dimension ni le layout v7 : il ne faut pas
supprimer `blue_patrol_mappo_v7.txt` ni `blue_patrol_training_v4.txt`. La
reprise reste possible avec ces fichiers; un nouvel entraînement sert seulement
à apprendre progressivement l'évitement des zones.

Dans le scénario TwoVsTwo, la barre de la vue tactique propose `Show patrol
track`. Cette option dessine l'unique piste du leader et son sens dans la vue 3D et la
minimap ; elle n'existe pas dans Airspace et n'influence jamais la simulation.
Sur les scénarios historiques, le champ `blueController` accepte `mappo` ou
`reference`, ce qui permet le même affichage avec ou sans PPO. La présence de
`blueTeam` exige le contrôleur MAPPO généralisé.

## Équipes P contre N optionnelles

Les champs `twoVsTwo.blueTeam` et `twoVsTwo.redTeam` sont additifs.
`Scenarios/TwoVsTwo.json` fournit `blueTeam` directement : l'entrée GUI
existante `two-vs-two` sélectionne donc l'unique runtime MAPPO généralisé.
`redTeam` est présent dans la configuration livrée avec N=4. Chaque rouge garde
son propre contrôleur RLMission de production ; seule la cardinalité du monde
est généralisée.

~~~json
"blueTeam": {
  "activeCount": 5
},
"redTeam": {
  "activeCount": 7,
  "maximumPlacementAttempts": 512,
  "delayedActivationProbability": 0.25,
  "minimumActivationDelaySteps": 256,
  "maximumActivationDelaySteps": 768
}
~~~

`blueTeam.activeCount` accepte `[1,9]` et `redTeam.activeCount` accepte
`[0,10]`. Le chargeur génère alors `blue_1..blue_P` à partir d'un unique
template bleu et `red_1..red_N` à partir d'un unique template RLMission de
production. Le placement rouge est borné par `maximumPlacementAttempts`
(`[1,4096]`) et échoue avec un diagnostic si `minimumSeparation` est
impossible ; les délais sont positifs, finis et ordonnés.

À chaque reset, `BlueTeamRoles` choisit exactement un leader. Les autres bleus
reçoivent des slots stables et distincts sur une formation en V, avec un index
normalisé dans `[-1,1]`. `GeneralizedPatrolMappoAgent` échantillonne les P
actions avant la physique puis remet exactement P transitions du même pas
d'équipe à `GeneralizedPatrolLearner`; une voiture détruite conserve son slot
mais son masque d'apprentissage devient nul.

L'observation généralisée v8 sépare le stockage brut de 295 valeurs du tronc
MLP compact de 156 valeurs. Elle réserve dix pistes rouges, huit autres bleus,
un objectif distinct, des masques explicites, le rôle/slot, P/N normalisés et
exactement 61 rayons issus du `sPseudoLidarState` ECS. L'attention et les
poolings masqués sont invariants à l'ordre des pistes. Les zones restreintes
restent dans le LiDAR/SAM et ne consomment jamais l'une des dix pistes classées
de RLMission.

Le chemin généralisé écrit `blue_patrol_generalized_v8.txt` et
`blue_patrol_generalized_training_v5.txt`. Un acteur v7 peut amorcer l'acteur
v8 avec une sortie 2v2 équivalente à `1e-6`. La migration explicite d'un état
v4 ne reprend que l'acteur et ses moments Adam compatibles : critique, états
d'optimiseur non compatibles, RNG, compteur et rollout repartent à neuf. Les
fichiers v7/v4 restent la source de repli du chemin legacy et ne doivent jamais
être supprimés ou écrasés par cette migration.

À chaque frontière d'update configurée, le runtime P/N suspend la collecte et
évalue les moyennes déterministes de l'acteur dans le vrai ECS. Les resets sont
reproductibles par seed et miroir. Survie bleue, neutralisation rouge, survie
de l'objectif et succès sont normalisés épisode par épisode par P et N avant
agrégation. Seul un meilleur candidat est promu séparément dans
`BluePatrol/prod/blue_patrol_generalized_v8.txt` ; cette campagne ne modifie ni
le rollout, ni le RNG d'apprentissage, ni les artefacts legacy v7/v4.

## Documentation

- [Architecture de la patrouille MAPPO](docs/blue-ppo-patrol.md)
- [Architecture et feuille de route de la patrouille](docs/blue-patrol-architecture-roadmap.md)
- [Architecture v7 : rôles leader/équipier et défense apprise](docs/blue-patrol-v6-architecture.md)
- [Scénario TwoVsTwo](docs/two-vs-two-example.md)
- [Guide de configuration](docs/configuration-guide.md)

## Validation

Boucle rapide pendant le développement :

~~~powershell
cmake --build out/build/x64-Debug --config Debug --target TwoVsTwoSpawnTests TwoVsTwoTeamDomainTests TwoVsTwoTeamDynamicsTests TwoVsTwoGeneralizedActorTests TwoVsTwoGeneralizedRuntimeStateTests TwoVsTwoGeneralizedLearnerTests TwoVsTwoGeneralizedRewardEvaluationTests MissionTrackSelectionTests --parallel 4
ctest --test-dir out/build/x64-Debug -C Debug -R "^(TwoVsTwoSpawnTests|TwoVsTwoTeamDomainTests|TwoVsTwoTeamDynamicsTests|TwoVsTwoGeneralized.*|MissionTrackSelectionTests)$" --output-on-failure
~~~

Validation finale, plus lourde, avant fusion :

~~~powershell
ctest --test-dir out/build/x64-Debug -C Debug -R "TwoVsTwoPatrolRuntimeSmoke|TwoVsTwoCombatRuntimeSmoke|TwoVsTwoEmbeddedLearnerSmoke" --output-on-failure
out/build/x64-Debug/Main/Debug/PPO.exe --example blue-patrol --check-example
out/build/x64-Debug/Main/Debug/PPO.exe --example two-vs-two --check-example
~~~

Les modèles rouges `resources/model/RLMission/**/*.pt` et leur contrat réseau
restent indépendants du backend bleu RLTools.
