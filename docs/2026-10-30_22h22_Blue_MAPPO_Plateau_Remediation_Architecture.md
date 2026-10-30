# PPOTrial — PR #39 : relecture et architecture de la correction MAPPO

Date de la revue : 30 octobre 2026. Référence initiale : `a5f11a90cb8996bad940201a0e0c1de1d8189267`.
PR : https://github.com/benoitfragit/PPOTrial/pull/39

## Résultat de la confrontation à l’audit

La première version corrigeait la terminaison au leader et les pénalités de formation, mais conservait une coupure de retour individuel à la destruction. Elle ne réalisait donc pas la correction P0 des retours coopératifs demandée dans `2026-10-30_22h_MAPPO_Diagnostic_Remede.md`. L’affirmation précédente « la coupure GAE à la destruction reste correcte » est rectifiée ici : dans cet objectif coopératif, seules les actions doivent être masquées après une perte ; les conséquences de mission doivent continuer à remonter aux actions antérieures.

| Point vérifié | Comportement retenu |
| --- | --- |
| Perte individuelle | Nouveau canal `OwnLoss`, valeur −1 une seule fois, poids configurable 24 ; indépendant de P. |
| Perte collective | `BlueSurvival = −pertes/P`, poids issu de `Mission outcome` (1,5 dans le JSON actuel). |
| Retour après une perte | Tous les participants conservent les canaux collectifs jusqu’à la fin de mission ; aucune nouvelle pénalité locale ni nouvelle perte propre pour un mort. |
| GAE | La disponibilité physique ne termine plus la récurrence. Un vrai terminal coupe le bootstrap ; une troncature garde V(s′) et coupe la trace au reset. |
| Actor | Gradient et normalisation des avantages limités aux lignes vivantes avant le pas ; un minibatch sans action valide ne met pas l’actor à jour. |
| Critic | Apprend sur toutes les lignes participantes. Pour un mort : bloc propre nul, contexte agrégé des survivants et fractions d’équipe conservés. |
| Leader détruit | La mission continue avec des survivants. Les responsabilités relatives au leader perdu sont suspendues explicitement ; aucun successeur ni contrôleur d’assignation de cibles n’est ajouté. |
| Menaces lointaines | Patrouille conservée si les pistes sont lointaines et non dangereuses pour l’objectif ; la poursuite est pondérée par le voisinage des Rouges et le risque objectif. |
| Basse vitesse | Coût sous 35 % de cruiseSpeed, poids configurable 5 ; la vitesse finale du pas fatal contribue au signal malgré la disparition du mobile. |
| Télémétrie | 15 canaux, valeurs/poids/noms cohérents ; les deux smoke tests n’attendent plus 13 canaux. |

Exemple P=4, une seule perte, avant les autres canaux et le clamp : l’agent détruit reçoit **−24 − 1,5/4 = −24,375** ; chaque autre participant reçoit le coût collectif **−0,375**. Le coût propre est distinct du coût coopératif et n’est plus remplacé par celui-ci.

## Répartition des responsabilités

- Le monde ECS reste responsable de la physique, des destructions, des frontières et des agents rouges historiques `RLMission`.
- L’adaptateur Bleu produit les snapshots, les masques et les transitions de récompense.
- Les fonctions pures calculent les signaux de formation, d’engagement et de sécurité ; elles ne commandent aucun mobile.
- L’agrégateur distingue événements individuels, événements collectifs et responsabilités locales.
- Le learner MAPPO conserve acteur partagé et critic centralisé. Les masques de gradient actor et les cibles de valeur sont séparés.

Les dimensions des observations, des réseaux et les formats des checkpoints sont conservés. L’ordre d’inclusion RLtools est corrigé pour GCC. La lecture du RNG consomme explicitement les espaces avant son état, dans les deux lecteurs de checkpoints Bleus ; le format écrit reste identique. Aucun fichier du contrôleur rouge, de sa configuration ou de la physique partagée n’est modifié.

## Configuration

Dans `resources/settings/Training/BluePatrol.json` :

```json
"generalized": {
  "ownLossWeight": 24.0,
  "lowSpeedRiskWeight": 5.0
}
```

Ce bloc appartient à `rewards`. Son absence conserve ces valeurs par défaut pour les anciennes configurations ; nombres non finis ou négatifs rejetés. Les sept poids historiques gardent leur mapping.

## Vérification

- Relecture des sources modifiées, du diff et des chemins touchés ; contrôle `git diff --check`.
- Compilations et exécutions GCC C++17 : variantes avec symboles `-Og -g` et optimisée `-O2 -DNDEBUG`, avec `-Wall -Wextra -Wpedantic`. Suites : récompenses/évaluation, shaping, état runtime, GAE, domaine d’équipe, dynamique d’équipe, défense, géométrie/configuration, actor et learner.
- Régressions : P=1/4/9, N=0/2/10 selon les suites, perte puis échec tardif, masques actor, contexte critic mort, terminaux/troncatures, permutation, formation, vitesse et aller-retour des checkpoints.
- Quatre suites pures sont désormais intégrées au CTest headless Ubuntu/Windows, sans dépendance graphique, Torch ou RLtools.
- Sur la révision initiale, VS2022 Release compilait mais CTest échouait sur `BlueDefensePatrolRuntimeSmoke` et `BlueDefenseCombatRuntimeSmoke` : attente obsolète de 13 canaux. Les sources de ces tests et les noms du snapshot sont corrigés. Les résultats CI doivent être contrôlés sur le nouveau commit avant fusion.
- Le CMake/CTest complet VS2022 est vérifié par la CI ; il n’a pas été exécuté localement. Des avertissements des en-têtes RLtools épinglés et du backend d’attention préexistant subsistent sous GCC ; aucun nettoyage de ces dépendances n’est inclus.

## Relance et limites

**Repartir de zéro**, conformément au choix de relance exprimé : ne renseigner aucun run de reprise dans la CI. Ni l’ancien critic ni l’ancien état Adam ne correspondent au nouvel objectif. Aucun entraînement n’est lancé par cette revue.

Les tests C++ valident les contrats, pas la convergence. Les poids 24/5 et le seuil de 35 % restent à calibrer sur des graines indépendantes. La suspension de la formation après perte du leader est un choix explicite ; une succession demanderait une évolution séparée des rôles. L’ablation gamma/lambda et les diagnostics étendus KL/saturation recommandés dans l’audit restent à mener ; ils ne sont pas présentés comme réalisés dans cette PR.
