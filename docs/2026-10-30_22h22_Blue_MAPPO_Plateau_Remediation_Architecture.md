# PPOTrial — Blue MAPPO V8: correction du plateau et des pertes à basse vitesse

Date : 2026-10-30, 22 h 22 (Europe/Paris). Périmètre : **MAPPO Bleu généralisé exclusivement**.

## Constat audité

- Dernier run : survie Bleue d'environ 46,9 % à 15,6 % ; 25 pertes finales sur 27 en basse vitesse, et 575 pertes à basse vitesse sur 657.
- À P=4, un Bleu perdu coûtait seulement **-0,375** au scalaire (1,5 × -1/4), tandis qu'une formation médiocre pouvait coûter **-4 par pas**.
- L'agent pouvait avoir intérêt à disparaître pour ne plus subir la formation. La coupure GAE à la destruction est correcte ; les récompenses futures d'un agent mort ne doivent pas être ajoutées comme si celui-ci continuait à agir.
- La perte du leader arrêtait l'épisode même si d'autres Bleus restaient vivants.

## Modifications apportées

1. **Perte d'un Bleu** : le canal `BlueSurvival` conserve une valeur équipe de `-nombre_perdus/P`. Sur la transition de l'agent détruit, ce canal vaut désormais `-1` et son poids est fixé à **24** dans le MAPPO Bleu généralisé (contre 1,5 indirectement). Un coéquipier survivant reçoit encore le coût collectif normalisé. À P=4, la perte coûte donc **-24 à l'agent détruit** et **-6 aux survivants**.
2. **Formation** : remplacer le coût absolu répété par une récompense signée liée à la **variation d'erreur**. Une formation stationnaire imparfaite ne crée plus de dette illimitée. Aucune formation n'est imposée si le leader ou le coéquipier est détruit.
3. **Vitesse** : ajouter `LowSpeedRisk`, canal 14 (append-only), coût borné de risque et d'aggravation sous **35 % de la vitesse de croisière**, poids **5**. Cette mesure préventive ne modifie ni la physique ni les commandes d'action.
4. **Fin d'épisode** : la mort du leader seule ne termine plus l'épisode ; la perte de tous les Bleus, la frontière et la disparition de l'objectif conservent leurs conditions de terminaison.
5. **Tests** : non-régression P=1/N=0 et P=4/N=0, invariance par permutation, pertes de leader, formation stable/améliorée/dégradée, rejet des signaux non finis, vitesse faible et attribution du coût de perte.

## Architecture inchangée

```text
World ECS (physique, collisions, frontière, zones, rouges RLMission)
    -> generalized_patrol_team_agent.cpp (observations et transitions Bleues)
       -> generalized_reward_shaping.hpp (formation et sécurité pures)
       -> generalized_cooperative_reward.hpp (canaux/agrégation)
       -> generalized_patrol_learner.hpp (PPO + GAE, inchangé)
    -> Evaluation, CI et checkpoints, structure inchangée
```

Aucune modification des agents rouges historiques, de `RLMission`, du moteur physique, de l'observation réseau, de l'actor, du critic ou de la disposition des checkpoints. Les 13 indices de canaux précédents sont stables ; `LowSpeedRisk` est ajouté en position 14.

## Vérifications et protocole de relance

- Vérifier CMake **Debug et Release**, puis CTest et les actions CI avant fusion. La compilation n'est pas exécutée par le connecteur GitHub ; vérifier les résultats de la PR.
- Les anciennes métriques de récompense ne sont **pas comparables** ; comparer indépendamment des seeds identiques, la survie, les causes de destruction, le score mission et les commandes d'accélération.
- **Redémarrer à partir des poids actor `prod` du meilleur run #8, score 56,705** : repartir avec nouveau critic, optimizers et compteurs. Ne pas reprendre le `latest` ni le checkpoint entraînement complet du run #9. Le format de poids actor est inchangé.

## Risques restants

La modification change l'objectif d'apprentissage sans garantir la convergence ; 35 %, 24 et 5 sont des points de départ à calibrer. L'apprentissage doit être évalué avec plusieurs seeds avant promotion.
