# FlightControl3D — propulsion réelle, commandes utiles et corrections de convergence

Date : 13 novembre 2026, Europe/Paris. Dépôt : PPOTrial. Branche : `fix/flight-approach-only`, [PR #87](https://github.com/benoitfragit/PPOTrial/pull/87), [issue #91](https://github.com/benoitfragit/PPOTrial/issues/91). Analyse démarrée sur `f7720fc`. L’[issue #90](https://github.com/benoitfragit/PPOTrial/issues/90) conserve le suivi de la convergence.

## Résultat et limites

Le défaut principal confirmé est physique : les deux moteurs du C310 s’arrêtaient pendant l’initialisation. La manette des gaz n’avait alors aucune autorité propulsive. Le correctif rétablit les moteurs, initialise les volets dans la configuration de vol courante, simplifie les actions des leçons LOC/Glide et corrige la comparaison temporelle des rewards. Une protection KL limite les mises à jour PPO et le plancher de bruit évite l’effondrement de l’exploration.

Les critères LOC/Glide restent stricts, identiques entre apprentissage et validation : maintien de 2 secondes, CAS 51–61 m/s, cap et route ±5°, roulis ±8°, p/q/r ±5°/s ; LOC ±2,5°, Glide ±0,7°. Le passage exige toujours 19/20. **Les corrections logicielles ne qualifient pas une politique d’atterrissage.**

## 1. Défaut moteur reproduit et corrigé

Le reset appelait `RunIC()` puis `FGPropulsion::InitRunning(-1)`. Cette dernière méthode démarre les moteurs puis lance `GetSteadyState()`, un solveur interne utilisant un pas de 0,5 seconde. Sur le modèle C310 épinglé, ce calcul ramène les hélices à 0 tr/min et les moteurs à `Running=false`, avec une puissance de −1,5 hp dans le cas reproduit, malgré une manette à 0,65.

L’adaptateur prépare maintenant le mélange par `initial.mixture=1.0`, puis appelle `InitRunning()` sur chacun des deux `FGEngine`. La physique ordinaire de 10 ms intègre ensuite la montée en régime. Aucun contrôleur automatique ni modification du XML avion n’est ajouté. Dans l’essai natif de diagnostic, le moteur atteint environ 2 523 tr/min et 90 hp après une seconde ; le fonctionnement des deux moteurs est initialisé par la même boucle bornée.

La régression native compare ralenti, pleine puissance puis ralenti après reset, à conditions initiales identiques. La différence de CAS après 5 secondes dépasse 5 m/s ; les deux essais au ralenti sont identiques à 1e−7 m/s près. Remettre l’ancien appel au solveur fait échouer cette régression. Le test vérifie donc une autorité réelle de la manette, pas seulement une valeur de commande.

Au reset, les volets sont placés à la valeur de la phase : 0 en patrouille, 0,33 en approche/roulage dans le scénario livré. Le modèle épinglé convertit cette valeur normalisée sur sa course physique de 0 à 45°. Le premier pas ne déclenche plus une extension parasite. Leur position réelle est contrôlée par les tests JSBSim.

## 2. Actions et réseau

| Commande | Distribution / application | LOC et Glide | Joint |
|---|---|---|---|
| Profondeur, ailerons, direction | Gaussiennes, clamp linéaire [−1,+1] | Apprises | Apprises |
| Gaz | Gaussienne ; zéro latent correspond à 0,65, autorité physique [0,1] | Appris | Appris |
| Freins | Gaussienne puis clamp [0,1] | Désactivés | Actifs au roulage |
| Train | Bernoulli | Configuration initiale fixe, train rentré par défaut | Appris |
| Volets | Configuration JSON par phase | 0,33 | Selon la phase |

Les composantes inactives sont exclues de l’échantillonnage, de la log-vraisemblance PPO et de l’entropie. Ainsi une sortie de frein inutilisée ne modifie pas le ratio de probabilité d’une action en vol. Les masques sont conservés dans chaque transition jusqu’à son optimisation. Les tests couvrent les quatre combinaisons frein/train, y compris les gradients par différences finies.

Le réseau conserve 56 observations, un acteur `56 → 128 → 128 → 11` (5 moyennes, 5 log-écarts-types et 1 logit train) et un critic `56 → 128 → 128 → 1`. Le format `PPO_FLIGHT_HYBRID_V1 56 5 128` reste compatible avec le runner CI.

## 3. Reward : comparer les vrais états successifs

Le shaping conserve la forme demandée, pour chaque qualité bornée :

`r_i = poids_i × (Q_i(état après, piste après) − Q_i(état avant, piste avant))`.

L’ancien calcul transportait seulement la position antérieure dans le repère de la piste actuelle, puis recalculait la vitesse relative avec la cinématique actuelle. Sur une piste qui tangue, cette position contrefactuelle et cette vitesse ne décrivent pas le vrai état précédent. La pente pouvait ainsi cumuler un reward positif artificiel. Le calcul utilise maintenant chaque instantané avion/piste/mission à son instant propre. Un cycle aller-retour avec piste mobile vérifie que les différences de qualité vitesse/pente se compensent.

Les dix canaux, poids et qualités restent ceux du scénario précédent : `Path=0,002`, `Localizer=30`, `Glide=10`, `Speed=1`, `Slope=1`, entrées LOC/Glide `2`, toucher/arrêt `100`, crash `−100`. Path reste nul en approche. Les protections de frontières et de recul LOC/Glide sont conservées : ces deux canaux ne sont donc pas présentés comme un potentiel global sans exceptions. Vitesse et pente conservent `Q=tolérance/(tolérance+|erreur|)`, avec 15 m/s et 5° pour leur shaping ; les critères de réussite plus stricts restent distincts.

Les essais ajoutant un coût de maintien et un canal de stabilité n’ont pas fourni d’amélioration durable. **Ils ne sont pas livrés.** Aucun bonus ou poids expérimental supplémentaire n’entre dans la correction finale.

## 4. PPO : freiner les changements excessifs

`actorRate` passe de 3e−4 à 1e−4 ; `criticRate` reste 3e−4. `targetKl=0.02` est explicite dans le JSON. À chaque minibatch, le learner mesure la moyenne de `exp(logRatio) − 1 − logRatio`, sur les seules composantes actives. Si elle dépasse la cible, les étapes acteur restantes du rollout sont arrêtées ; le critic poursuit tous ses minibatches. L’âge Adam de l’acteur reste figé pendant cet arrêt et se restaure exactement au checkpoint.

Cette protection est un arrêt sur une estimation de dérive, **pas une borne mathématique stricte de KL après chaque étape**. Le journal publie `approx_kl`, `actor_steps` et `kl_stop`. Un test déterministe force une grande mise à jour : une seule étape acteur est exécutée, tandis que les 32 étapes critic prévues sont réalisées.

Les bornes gaussiennes livrées sont `minLogStd=-3.5`, `initialLogStd=-3`, `maxLogStd=-2`, soit σ≈0,030 / 0,050 / 0,135. Les essais précédents montraient une diminution excessive du bruit ; le nouveau plancher conserve de l’exploration. Aucun changement de tanh/clamp ni réduction d’autorité des gouvernes.

## 5. Curriculum, reprise et BEST

Les étapes publiques restent `loc → glide → joint`, avec 30 % de rappels LOC en Glide. Les workflows et `tools/flight/curriculum_ci.py` restent identiques. Le changement se concentre sur le problème physique, les composantes utiles de l’action et l’apprentissage PPO ; aucun palier supplémentaire n’est prétendu validé.

Le contrat porte `rewardVersion=12`, `spawnVersion=4`, `curriculumVersion=4`, `policyLikelihoodVersion=2`. Les réglages détaillés du curriculum sont inclus dans la signature exacte ; changer une gate impose une revalidation du BEST. Les anciens latest au mapping clampé peuvent servir de warm start, avec un nouveau rollout et de nouvelles vraisemblances. Un ancien champ de mélange absent est interprété comme le démarrage historique plein riche. Les checkpoints tanh ou signatures inconnues restent refusés.

Seul Joint peut promouvoir un BEST après sa validation d’atterrissage et d’arrêt. Les essais LOC n’en créent aucun. Pour mesurer le correctif sans dépendre des mauvaises habitudes apprises moteurs arrêtés, utiliser une campagne neuve ; une reprise reste possible mais doit être jugée séparément.

## 6. Expériences et vérification

Le run CI #7 antérieur terminait à 0/20 après 1 000 updates. Le contrôle local avec **le seul correctif moteur** atteint 6/20 sur seed 4342 à l’update 300. Ce contrôle est distinct de la configuration finale et ne démontre pas sa convergence.

Le banc final part de zéro, seed d’apprentissage 42, avec le scénario livré et 500 updates (1 024 000 décisions). Ses 1 741 épisodes complets comprennent **53 captures stabilisées, 81 crashs et 1 607 timeouts**. À l’update 500, le verdict indépendant est 5/20. Une relecture déterministe du checkpoint final, sans apprentissage ni changement de tolérance, donne :

| Mesure | Seed 4342 | Seed 8342 |
|---|---:|---:|
| Captures stabilisées | 5/20 | 5/20 |
| Crashs | 0/20 | 0/20 |
| Épisodes avec capture géométrique LOC | 8/20 | 8/20 |
| CAS moyenne pendant les échantillons LOC capturé | 57,13 m/s | 57,09 m/s |
| Échantillons LOC capturé dans 51–61 m/s | 542/542 | 550/550 |

La vitesse n’est plus le blocage observé dans ces captures. Cap et route ne satisfont leurs tolérances que sur une partie des échantillons ; le passage dans le faisceau ne constitue pas encore un alignement durable. Les scores intermédiaires fluctuent, et les deux relectures portent sur **une seule graine d’apprentissage**. Ce résultat montre des captures réelles mais ne démontre ni robustesse multi-entraînements, ni passage de gate, ni capacité d’atterrissage. Aucun BEST n’est promu ; #90 reste ouverte. Les essais au coût de maintien et le pré-entraînement sur des départs facilités ne sont pas utilisés pour qualifier ce banc final.

Vérifications locales : GCC 14, CPU headless, builds `build/debug` et `build/release`, 15/15 CTest dans chaque configuration et 7/7 tests Python du runner. Les tests contrôlent la propulsion réelle, les resets, la conservation du train en leçon, les masques et gradients PPO, la protection KL et les âges Adam, la cohérence des instantanés, les reprises et la protection BEST. Une relecture des fonctions, appelants, tests et du diff complet accompagne le correctif. La CI distante doit être associée au SHA exact ; aucun résultat d’une ancienne tête n’est attribué au nouveau code.

Périmètre : FlightControl3D et ses réglages/documentation uniquement. RLMission, BlueDefense, workflows et runner CI sont inchangés. Aucun CUDA, nouveau simulateur, modèle physique modifié ou modèle qualifié artificiellement.

## Sources

- [Run #7 et artefact initial](https://github.com/benoitfragit/PPOTrial/actions/runs/43351754089).
- [JSBSim épinglé : FGPropulsion, démarrage et solveur stationnaire](https://github.com/JSBSim-Team/jsbsim/blob/66f7fafc07fe13e7ede9db20873030fed2881713/src/models/FGPropulsion.cpp).
- [JSBSim épinglé : FGPiston, InitRunning](https://github.com/JSBSim-Team/jsbsim/blob/66f7fafc07fe13e7ede9db20873030fed2881713/src/models/propulsion/FGPiston.cpp).
- [PPO et arrêt sur KL, documentation primaire Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
