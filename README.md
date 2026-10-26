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
toujours les deux entités rouges, mais son curriculum d'entraînement tire 20 %
d'épisodes sans menace, 35 % avec une menace et 45 % avec deux menaces. La
première menace peut être retardée et, dans un épisode à deux menaces, la
seconde peut apparaître plus tard. Les campagnes d'évaluation restent des cas
immédiats et déterministes 2v1/2v2.

À chaque épisode 2v2, les rouges sont tirés sur un arc allant de l'avant aux
deux côtés de la zone défendue. Ils commencent entre 1,05 et 1,35 fois la
limite du monde et à au moins 10 km l'un de l'autre. La limite est un repère de
zone, pas un mur : RLMission continue normalement à piloter les rouges vers
l'objectif. La zone restreinte est déplacée dans la moitié défendue et n'est
active que dans 75 % des épisodes.

La patrouille suit l'axe est/ouest, le plus long du monde configuré. Un seul
contrôleur MAPPO possède l'acteur partagé, le critique centralisé, le buffer
on-policy et l'optimiseur des deux bleus.

La plateforme bleue reprend la dynamique de `trainingaircraft` : vitesse de
croisière initiale 300 m/s, vitesse maximale 700 m/s, accélération 600 m/s² et
rotation maximale d'environ 40,1°/s (0,7000006 rad/s). Sa zone SNA sans
échappement vaut 5 km et sa zone d'engagement atteint 10 km, contre 4 et 8 km
pour les rouges, qui restent inchangés.

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
contient `blue_patrol_training_v1.txt`, qui restaure l'acteur, le critique,
les deux états Adam, le compteur de mises à jour et le RNG. Le buffer de rollout
n'est volontairement pas sauvegardé : une reprise commence un rollout vide au
même niveau d'optimisation.

La collision entre les deux bleus est terminale à 1 km, avec une pénalité de
sécurité progressive à partir de 2 km. Sans menace, le canal de formation
récompense aussi explicitement une file leader–suiveur espacée d'environ
9,6 km dans le monde par défaut. Avec plusieurs menaces, le façonnage récompense
au contraire leur séparation et la couverture d'une cible distincte chacun.

## Documentation

- [Architecture de la patrouille MAPPO](docs/blue-ppo-patrol.md)
- [Architecture et feuille de route de la patrouille](docs/blue-patrol-architecture-roadmap.md)
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
