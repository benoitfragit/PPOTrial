# FlightControl3D — curriculum de captures stabilisées sans modification de CI

Date : 13 novembre 2026 (Europe/Paris). Projet : PPOTrial. Issue : [#89](https://github.com/benoitfragit/PPOTrial/issues/89). PR : [#87](https://github.com/benoitfragit/PPOTrial/pull/87). Branche : `fix/flight-approach-only`. Base de cette évolution : `19866f702db81a2cc347eeed519f9833379c37ac`.

## Problème et changement

Les gates LOC et Glide constataient une capture maintenue pendant 2 secondes, puis continuaient le vol jusqu’au crash, à l’arrêt ou au timeout de 900 secondes. Un crash ultérieur invalidait la capture pédagogique. Le début de curriculum demandait donc aussi de savoir atterrir ; les expériences de capture étaient inutilement longues et leurs départs cumulaient des erreurs de vitesse, d’attitude et de hauteur.

Les trois étapes publiques restent `loc → glide → joint`. LOC et Glide terminent maintenant leur épisode dès la capture physiquement stabilisée, avec une durée maximale livrée de **60 secondes**. Joint conserve le toucher sûr puis l’arrêt sur piste, avec la distribution complète des départs et les **900 secondes** du scénario.

## Étapes et départs

Les tolérances et plages ci-dessous sont explicites dans `training.curriculum` du JSON. Distance, côté de départ et repère de piste suivent les réglages de l’approche seule : 4–6,5 km, gauche/droite équilibrés, piste réellement randomisée et mobile.

| Étape | Départ et objectif | Fin de l’épisode |
|---|---|---|
| LOC | Gauche/droite à 4–10°, cap vers le seuil ; hauteur à ±20 m du glide pour éviter une récupération verticale indépendante ; reward Glide éteint | LOC strict, cap/route et pilotage stables pendant 2 s |
| Glide | Sur l’axe LOC, erreur verticale signée de 30–120 m ; 30 % de rappels LOC conservés | LOC et Glide stricts, pente et pilotage stables pendant 2 s ; les rappels utilisent l’objectif LOC |
| Joint | Distribution complète : vitesse 55–80 m/s, roulis ±15°, tangage −4 à +8°, départs latéraux et dans l’axe | Toucher valide avec train sorti, puis arrêt confirmé sur piste |

Les départs des deux premières étapes utilisent une CAS de **56 ±2 m/s**, un roulis de **±3°** et un tangage de **1 à 3°**. Les départs LOC utilisent une pente de **−3 ±0,5°** ; les départs Glide sur l’axe conservent leur pente initiale nominale de −3°. Cette aide n’ajoute ni trim ni pilote automatique et ne change pas les commandes PPO.

## Critère de capture stabilisée

Le critère est évalué à **chaque sous-pas JSBSim de 10 ms**, dans le repère de la piste courante. Le maintien est remis à zéro dès qu’un critère échoue.

| Contrôle commun | Limite livrée |
|---|---|
| Géométrie | Signal ILS valide, avant le seuil, au-dessus de la hauteur d’arrondi, sans contact et avec vitesse longitudinale entrante |
| LOC | Capture confirmée et erreur dans la tolérance active, sans profiter de la seule hystérésis de perte |
| Cap et route | Chacun à ±5° de la piste |
| CAS | À ±5 m/s de la consigne, indépendamment de la tolérance du shaping de vitesse |
| Roulis relatif à la piste | ±8° |
| Vitesses angulaires corps p, q et r | Chacune à ±5°/s |
| Glide, uniquement dans la leçon Glide | Capture confirmée, erreur dans la tolérance active et pente à ±1,5° de la consigne |

L’entraînement conserve son facteur existant de tolérance ILS ; la validation indépendante utilise les tolérances nominales strictes. Le seuil de passage reste **92 % sur 20 épisodes, donc au moins 19/20**. Un crash reste prioritaire sur une capture ; un maintien inachevé au budget reste une troncature.

## Architecture, observations et modèle

- `Core/flight_mission.*` porte le critère commun et son maintien physique. `Outcome::CaptureCompleted` est ajouté après les valeurs existantes : le code CSV de cette issue pédagogique vaut 4 ; les anciens codes sont conservés. Cette issue terminale ne rapporte aucun nouveau bonus, ni `Touchdown` ni `Stop` fictif.
- `Simulator/flight_environment.cpp` applique les plages de départ et la durée des leçons LOC/Glide. Joint utilise le scénario complet. Les épisodes de rappel restent de vrais épisodes LOC.
- `Runtime/flight_session.cpp` compte les issues pédagogiques pour les gates. Il ne déduit plus une réussite d’un ancien drapeau mémorisé pendant un vol qui continue. La validation ne modifie ni les poids, ni Adam, ni le RNG d’entraînement.
- L’observation reste à **56 valeurs**. Le champ 41, auparavant `stopped_hold`, devient `completion_hold` : il expose la fraction du maintien pédagogique en LOC/Glide, puis conserve la fraction d’arrêt en Joint/roulage. Le compteur pertinent est donc visible par la politique sans changer la largeur du réseau.
- `curriculumVersion=2` décrit ce changement de tâche et d’observation. Un ancien latest de même architecture et de même mapping d’actions peut servir de warm start ; un ancien BEST doit être revalidé sur le contrat exact. Seul Joint peut promouvoir un BEST d’atterrissage.

Les dix canaux de reward, leurs poids et les formules LOC/Glide restent inchangés. L’hippodrome demeure désactivé et réactivable comme dans le correctif précédent. Aucun changement BlueDefense/RLMission, CUDA, physique JSBSim ou lissage des actions.

## Contrat CI conservé

Les workflows `.github/workflows/`, le runner `tools/flight/curriculum_ci.py` et ses tests restent identiques. Les options CLI, le format du checkpoint et ses dimensions, les fichiers d’artefact et la ligne `[FLIGHT_GATE] stage=... successes=... episodes=... passed=...` restent compatibles. Le même lancement `stage=progressive` enchaîne donc les trois étapes. Un verdict négatif garde la gate courante ; les limites et mécanismes de reprise existants sont conservés.

## Vérification et limites

- Linux GCC 14, headless CPU : builds Debug et Release réussis ; **15/15 CTest dans chaque configuration**.
- **7/7 tests Python** du runner existant réussis sans modification de son code.
- Régressions : maintien continu observé, remise à zéro sur oscillation ou sortie de tolérance malgré la capture mémorisée, réussite terminale sans faux poser, priorité crash, timeout avec bootstrap, Joint sans arrêt pédagogique anticipé, critères de vitesse/roulis/p/q/r, validation de configuration et warm start de l’ancien contrat.
- JSBSim réel : enveloppes de départ et restauration du scénario Joint vérifiées sur 72 resets ; LOC et Glide terminent une capture contrôlée sans atterrir ; un pas après cette fin exige un reset.
- Test intégré : **runner CI inchangé + vrai FlightControlHeadless**, une update PPO et 20 évaluations LOC sur un scénario volontairement facile (départ à 0,5°, maintien réduit à 0,1 s). Le runner obtient le verdict attendu, écrit `completed`, enregistre `glide` comme étape suivante et ne crée aucun BEST. Ce test vérifie le contrat et la logique, **pas la convergence sur le scénario livré**.
- Diff relu et absence de changement des workflows/runners contrôlée contre la base de l’évolution. Les jobs CI déclenchés par la PR doivent être vérifiés sur le dernier commit ; les validations de révisions antérieures ne lui sont pas attribuées.
