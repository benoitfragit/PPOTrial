# PPO défenseurs bleus — première étape sans rouges

Date : 2026-10-25. Contrat source : spécification bibliothèque `2026-10-25_11h44_PPO_2v2_Defenseurs_Specification_Initiale.md`.

## Implémenté

- RLtools CPU (FetchContent déjà verrouillé), PPO clip, GAE, politique gaussienne Actor commune, Critic local, C++17 ; exécutable dédié `PPOBlueTrain`.
- Contrat `BLUE_PATROL_PPO_V2` : **124 entrées / 2 sorties** ; 7 propres, 61 LiDAR, 2×13 slots rouges masqués, 2×9 amis/objectif, 10 patrouille, 2 globales. Les sorties sont rotation et accélération SNA normalisées.
- Patrouille de référence en râteau (deux pistes latéralement décalées, phases décalées, demi-tours en demi-cercle) et reward d'équipe de progression, tenue de trajectoire, cap, vitesse, sécurité et timeout.
- La patrouille de référence donne un **objectif cinématique**, jamais une action imposée au défenseur entraîné. Spawns et monde LiDAR n'ont pas été modifiés dans `TwoVsTwo`.
- Terminaison : débordement du monde ; troncature : 4096 pas ; la configuration sans rouge n'est pas une victoire instantanée.
- `Core/include/ppo/core/team_episode.hpp` distingue observation acteur, reward individuel/collectif, `terminated`, `truncated` et état global pour l'évolution MAPPO, sans dépendance de simulation.

## Important : frontière de la première étape

Ce premier curriculum RLtools est **autonome/headless**. À chaque épisode un des deux rôles est entraîné et l'autre suit la référence de patrouille. Les épisodes alternent les rôles et partagent l'Actor. **Ce n'est pas encore un apprentissage couplé de deux PPO agissant simultanément**, ni du MAPPO.

L'Actor RLtools dispose dès cette première étape de deux branches d'attention masquée entraînables, en amont du MLP de décision. Les poids de la branche rouge ne s'entraînent pas tant qu'aucune piste rouge n'est présente. La fonction d'observation intègre la frontière rectangulaire et les 61 directions du LiDAR, mais ne prend pas encore en charge d'autres obstacles/capteurs ECS. La dynamique d'entraînement transpose le mouvement SNA (mêmes conventions d'action et de pas) : elle n'appelle **pas encore** directement le système d'entités d'Airspace. Une validation de parité est nécessaire avant production.

Les acteurs sont exportés dans le checkpoint `BLUE_PATROL_PPO_V2`. L'inférence côté deux défenseurs existe dans le scénario GUI `blue-patrol` ; l'entraînement véritablement collaboratif n'est **pas encore** inclus. Les modèles `RLMission` et `RLSNA` sont intacts.

## Compilation et exécution

Configurer normalement le desktop avec `PPO_BUILD_DESKTOP=ON`, puis construire la cible `PPOBlueTrain`. Exécuter `PPOBlueTrain [seed] [checkpoint.txt] [updates]` (nombre de mises à jour facultatif, 300 par défaut). Tests `TwoVsTwoPatrolTests`, `TwoVsTwoAttentionTests`, `TwoVsTwoPPOTrainingSmoke`, `BluePatrolExampleBootstrap` via CTest. Le binaire `PPO.exe` historique est inchangé.

## Points à compléter

1. Mutualiser définitivement le noyau de dynamique SNA avec ECS (les limites du monde et du véhicule sont déjà lues depuis les JSON).
2. Intégrer une paire d'Actors agissant simultanément dans la boucle on-policy ; vérifier la récompense d'équipe sans double comptage, plus l'évaluation des demi-tours.
3. Évaluer l'inférence RLtools déjà connectée au GUI et la cohérence avec le moteur ECS.
4. Fournir de vraies pistes rouges aux branches d'attention existantes, puis enrichir la simulation des réactions ; préserver le contrat `RLMission`.
5. Évoluer vers MAPPO avec Critic centralisé séparé, sans changer l'Actor ou les classes d'entités.


Le test `TwoVsTwoPPOTrainingSmoke` effectue une mise à jour RLtools PPO puis vérifie la sortie du processus ; aucune métrique de convergence n'est revendiquée par ce smoke test.

## Inference dans le scénario GUI sans rouges

Le scénario additionnel `blue-patrol` charge `Scenarios/BluePatrol.json` sans aucun rouge.
Après apprentissage, la politique est enregistrée par défaut sous
`resources/model/BluePatrol/prod/blue_patrol_v2.txt` (non versionné).
Les deux bleus chargent une **seule instance du même Actor** RLtools et
choisissent chacun leur action avant le pas physique du monde ECS.

Lancer `PPO.exe --example blue-patrol`. Sans modèle, les deux gardes
sont pilotés par des contrôleurs de référence explicitement identifiés.
Un checkpoint PPO incompatible est signalé et provoque le même fallback
de référence, jamais une exécution de poids incompatibles.

## Réseau avec attention (v2)

Attention **apprise** native RLtools, initialisée avant la première phase d'entraînement.
Deux branches paramétriques indépendantes : rouges (13 caractéristiques, 2 pistes,
dimension latente 8), amis/objectif (9 caractéristiques, 2 pistes, latente 6).
Chacune utilise un encodeur dense partagé, une requête issue de l'état propre,
un softmax masqué puis concatène contexte, moyenne et maximum masqués. Les
gradients sont rétropropagés vers les paramètres de l'encodeur et de la requête
avec Adam lors des mises à jour PPO. Une branche entièrement masquée retourne
strictement zéro, sans NaN. En patrouille initiale, seuls les poids de la branche
amie reçoivent des gradients ; les rouges seront activés à la seconde phase.
Le MLP d'Actor et le Critic restent locaux ; les observations existantes,
les deux commandes SNA et le budget de 124 entrées sont inchangés.
Checkpoint v2 distinct et non compatible avec les premiers poids v1.

## Evaluation déterministe et arrêt des épisodes (complément)

Le pilote RLtools `PatrolTrainingAgent` possède le cycle de vie du learner
(headless, sans classe ECS); `PatrolProdAgent` reste l'adaptateur ECS d'inférence.
`PatrolEvaluator` utilise deux prédictions PPO par pas simulé et la même
dynamique `Advance` pour les deux mobiles. Il ne prétend pas être un
rollout multi-agent d'entraînement.

Cas d'arrêt : échec sur limite du monde, collision entre défenseurs, zone
circulaire interdite configurée ou dynamique/action non finie. Après le
budget de pas, succès seulement si la durée représente suffisamment de
circuits, que les deux agents tournent effectivement et que la fraction
de suivi position/cap/vitesse dépasse le seuil ; sinon échec qualité, ou
troncature si l'horizon est trop court. Les demi-tours individuels
n'arrêtent pas l'épisode.

La zone interdite est optionnelle dans l'évaluateur headless (rayon
par défaut 0, donc désactivée). Une évaluation de parité complète avec
la géométrie ECS et les collisions/destructions RLMission reste nécessaire
avant validation finale du comportement. Ne pas confondre le score
d'évaluation avec le signal de retour de PPO.

`PPOPatrolEvaluate [checkpoint.txt] [episodes] [steps]` renvoie 0
uniquement lorsque tous les scénarios ont un résultat succès. Tests
`TwoVsTwoPatrolEvaluationTests`.

## Patrouille graphique sans modèle entraîné (correction de démarrage)

La scène `blue-patrol` pilote toujours ses deux gardes dès le premier Play,
même quand `resources/model/BluePatrol/prod/blue_patrol_v2.txt` est absent.
Dans ce cas, le programme annonce explicitement que la politique PPO est
indisponible et utilise deux `PatrolReferenceAgent` déterministes. Chaque
contrôleur transmet des commandes SNA au moteur ECS, sans positionnement forcé
à chaque pas. Il s'agit d'une démonstration cinématique, pas d'une preuve
de convergence PPO.

`Scenarios/BluePatrol.json` choisit le comportement de chaque garde avec
`twoVsTwo.patrolControllers` (deux chaînes `"ppo"` ou `"reference"`).
Par défaut dans l'exemple : garde 1 PPO, garde 2 référence ; faute de
checkpoint valide, les demandes PPO basculent en référence avec indication
explicite en console. Un checkpoint incompatible déclenche également
une bascule explicite et journalisée vers la référence ; les poids
incompatibles ne sont pas exécutés. Quand le mode patrouille est actif,
les deux gardes sont initialisés sur leur trajectoire à la vitesse de
croisière même sans checkpoint. Les JSON des attaquants et les agents rouges
restent inchangés.

Test `TwoVsTwoPatrolTests` : progression simulée des deux contrôleurs,
demi-tours effectifs, bornes de la zone, et erreur maximale de suivi.
