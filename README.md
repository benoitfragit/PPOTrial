# PPOTrial

PPOTrial est un simulateur C++17 d'apprentissage par renforcement multi-agent.
La branche courante ajoute une équipe de deux défenseurs bleus entraînée avec
MAPPO/RLTools face aux attaquants rouges RLMission existants.

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
- `two-vs-two` : deux bleus MAPPO protègent l'objectif dans un curriculum à
  zéro, un ou deux rouges pilotés par RLMission.

Dans le sélecteur, ces entrées sont libellées explicitement `BLUE PATROL ONLY
- NO RED` et `TWO VS TWO - MAPPO vs RLMission`. Le second scénario conserve
toujours les deux entités rouges. Le curriculum active deux rouges dans 77,5 %
des épisodes, un rouge dans 15 % et aucun dans 7,5 %. Tous sont actifs dès le
début de l'épisode. Les évaluations restent des cas déterministes 2v1/2v2.

À chaque épisode 2v2, les rouges sont tirés sur un arc allant de l'avant aux
deux côtés de la zone défendue. Ils commencent entre 1,05 et 1,35 fois la
limite du monde et à au moins 10 km l'un de l'autre. La limite est un repère de
zone, pas un mur : RLMission continue normalement à piloter les rouges vers
l'objectif. Les zones rouges sont décrites par la collection typée
`twoVsTwo.restrictedZones`. Elles sont tirées une fois par épisode, puis vues
par les 61 rayons pseudo-LiDAR des bleus. Le chemin RLMission des rouges, son
observation 152 et ses checkpoints restent inchangés. La zone bleue historique
`blue_restricted_zone`, son tirage `restrictedZoneSpawn` et sa règle SNA sont
conservés à l'identique pour ne pas modifier le comportement des rouges.

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

La plateforme bleue dispose d'un léger avantage d'entraînement : vitesse de
croisière et vitesse maximale de 420 m/s, contre 400 m/s pour les rouges,
accélération de 600 m/s² et rotation maximale de 0,50 rad/s, contre 0,40.
Sa zone SNA sans
échappement vaut 5 km et sa zone d'engagement atteint 10 km, contre 4 et 8 km
pour les rouges. Son domaine angulaire de destruction vaut 1,10 rad, contre
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
utilise les vrais rouges RLMission dans le 2v2, publie ses mesures dans l'IHM et
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
Le champ de scénario `blueController` accepte `mappo` ou `reference`, ce qui
permet le même affichage avec ou sans PPO.

## Documentation

- [Architecture de la patrouille MAPPO](docs/blue-ppo-patrol.md)
- [Architecture et feuille de route de la patrouille](docs/blue-patrol-architecture-roadmap.md)
- [Architecture v7 : rôles leader/équipier et défense apprise](docs/blue-patrol-v6-architecture.md)
- [Scénario TwoVsTwo](docs/two-vs-two-example.md)
- [Guide de configuration](docs/configuration-guide.md)

## Validation

~~~powershell
ctest --test-dir out/build/x64-Debug -C Debug -R "TwoVsTwo|BluePatrol" --output-on-failure
out/build/x64-Debug/Main/Debug/PPO.exe --example blue-patrol --check-example
out/build/x64-Debug/Main/Debug/PPO.exe --example two-vs-two --check-example
~~~

Les modèles rouges `resources/model/RLMission/**/*.pt` et leur contrat réseau
restent indépendants du backend bleu RLTools.
