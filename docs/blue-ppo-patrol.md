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

Le modèle initial RLtools est un MLP : les slots de pistes sont définis et masqués mais les **encodeurs/attentions appris des rouges et amis ne sont pas encore implémentés**. Ils nécessiteront une version de modèle distincte. La fonction d'observation intègre la frontière rectangulaire et les 61 directions du LiDAR, mais ne prend pas encore en charge d'autres obstacles/capteurs ECS. La dynamique d'entraînement transpose le mouvement SNA (mêmes conventions d'action et de pas) : elle n'appelle **pas encore** directement le système d'entités d'Airspace. Une validation de parité est nécessaire avant production.

Les acteurs entraînés sont exportés en format texte versionné `BLUE_PATROL_PPO_V1` ; l'inférence dans `PPO.exe`, le chargement par les voitures bleues et l'entraînement véritablement collaboratif ne sont **pas** inclus dans cette première étape. Les modèles `RLMission` et `RLSNA` sont intacts.

## Compilation et exécution

Configurer normalement le desktop avec `PPO_BUILD_DESKTOP=ON`, puis construire la cible `PPOBlueTrain`. Exécuter `PPOBlueTrain [seed] [checkpoint.txt] [updates]` (nombre de mises à jour facultatif, 300 par défaut). Cible de test `TwoVsTwoPatrolTests` via CTest. Le binaire `PPO.exe` historique est inchangé.

## Points à compléter

1. Faire converger les paramètres d'entraînement avec les fichiers JSON de plateformes et de monde ; partager définitivement le noyau de dynamique SNA au lieu de le dupliquer dans le curriculum.
2. Intégrer une paire d'Actors agissant simultanément dans la boucle on-policy ; vérifier la récompense d'équipe sans double comptage, plus l'évaluation des demi-tours.
3. Ajouter chargement/inférence RLtools et représentation dans le GUI, puis évaluer avec le même moteur ECS.
4. Ajouter attention masquée rouge/ami, les adversaires lointains puis l'interception, tout en préservant le contrat `RLMission`.
5. Évoluer vers MAPPO avec Critic centralisé séparé, sans changer l'Actor ou les classes d'entités.


Le test `TwoVsTwoPPOTrainingSmoke` effectue une mise à jour RLtools PPO puis vérifie la sortie du processus ; aucune métrique de convergence n'est revendiquée par ce smoke test.

## Inference dans le scénario GUI sans rouges

Le scénario additionnel `blue-patrol` charge `Scenarios/BluePatrol.json` sans aucun rouge.
Après apprentissage, la politique est enregistrée par défaut sous
`resources/model/BluePatrol/prod/blue_patrol_v2.txt` (non versionné).
Les deux bleus chargent une **seule instance du même Actor** RLtools et
choisissent chacun leur action avant le pas physique du monde ECS.

Lancer `PPO.exe --example blue-patrol`. Sans modèle, le scénario se charge
mais les bleus restent sans agent ; un fichier invalide fait échouer le
bootstrap plutôt que d'exécuter une politique incompatible.

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
