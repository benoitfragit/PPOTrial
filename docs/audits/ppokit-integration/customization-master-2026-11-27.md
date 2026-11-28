# Configuration et modèles : comparaison avec master

Référence : `origin/master` à `b5ea31967f16d5fb2390268648715da5c6fe1b10`,
comparée à la branche PPOKit le 27 novembre 2026. Les valeurs suivantes sont
celles des fichiers livrés, pas la preuve qu'une politique a convergé.

## Blue Patrol

- `blue_patrol_training.json` a les mêmes valeurs JSON que
  `resources/settings/Training/BluePatrol.json` sur master, y compris les
  poids, le bornage, la géométrie et les réglages P-versus-0. Les fonctions
  `generalized_cooperative_reward.hpp` et `generalized_reward_shaping.hpp`
  sont identiques à master. L'adaptation ECS et le backend d'entraînement ont
  changé ; leur identité comportementale complète n'est pas déduite de cette
  comparaison de fichiers.
- Le scénario est identique à celui de master, hormis le chemin des profils
  d'agents, adapté à l'arborescence des projets. Les curricula V3 et les gates
  adaptatifs CI ont les mêmes valeurs JSON que master.
- `training.json` configure le learner partagé. Les tests de parité comparent
  gamma, lambda, clipping, entropie, nombre d'époques et mini-batch aux
  constantes du learner Blue historique ; les deux fichiers n'ont pas le même
  rôle. `blue_patrol_training.json` reste nécessaire à l'environnement pour
  les récompenses, la géométrie et d'autres réglages.
- Le bouton Train de Studio utilise le scénario configuré et n'exécute pas le
  curriculum des gates CI. Celui-ci reste éditable séparément dans Studio pour
  son lanceur externe ; sa séquence est fixée par son contrat.
- Les fichiers V7 d'entraînement et V10 de production de master ont été
  reconvertis avec `blue_checkpoint_import`. Leurs payloads correspondent aux
  checkpoints historiques livrés. Une seconde conversion avec
  `blue_shared_backend_import` a donné les mêmes payloads que les checkpoints
  partagés livrés : acteur de production et état d'entraînement update 5250.
  L'importateur vérifie aussi les sorties pour 1 à 9 Bleus. Les en-têtes et
  dates diffèrent entre les formats ; les SHA-256 des fichiers convertis ne
  sont donc pas censés être ceux des fichiers sources.
- Les poids modifiés en direct restent limités à [0, 10], comme les curseurs
  historiques. Le poids `OwnLoss` configuré à 24 reste inchangé lorsqu'un
  autre canal est modifié. Le bornage global `rewards.clamp`, réglable en
  direct dans l'ancienne interface de master, se règle actuellement dans le
  fichier de configuration avant lancement ; Studio n'a pas de commande de
  modification en direct pour cette valeur.

## Autres projets

- Les scénarios T-6 et PC-7 correspondent à ceux de master, avec une propriété
  explicite supplémentaire `runtime.centrelineInterceptMeters: 1200` pour le
  contrat PPOKit. Les poids de l'acteur T-6 historique proviennent du BEST 870
  de master ; les tests de parité comparent le checkpoint et les sorties.
- PC-7 expose un modèle BEST 610 issu d'un artefact de run, absent comme tel
  de master. Son empreinte locale correspond à `models.json`, mais il ne faut
  pas le présenter comme une copie vérifiée d'un modèle présent sur master.
- Les empreintes des politiques PROD et EVAL RLMission et RLSNA correspondent
  à leurs manifestes et aux sources de master. Les réglages PPO sont désormais
  déclarés dans les `training.json` de ces projets ; la suppression de la
  section PPO des anciens fichiers de scénario ne signifie pas leur disparition.
  Les scénarios Airspace comparés ne diffèrent que par les chemins des profils
  d'agents, adaptés à la nouvelle arborescence.

Les tests de parité et les contrôles d'empreinte vérifient les contrats et les
conversions couverts ; ils ne remplacent pas une évaluation indépendante sur
des graines distinctes.
