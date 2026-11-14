# FlightControl3D — échec LOC du run curriculum #6 et corrections ciblées

Date de vérification : 13 novembre 2026, UTC. Projet : PPOTrial. [Issue #90](https://github.com/benoitfragit/PPOTrial/issues/90), [PR #87](https://github.com/benoitfragit/PPOTrial/pull/87), branche `fix/flight-approach-only`. Révision diagnostiquée : `3cb6950c30634330c89a10e35a94bf125ee2dc81`.

## Verdict sur l’arrêt

Le [run FlightControl3D Curriculum Training #6](https://github.com/benoitfragit/PPOTrial/actions/runs/43343797520) a consommé les **1 000 updates** autorisées : 100 blocs de 10 updates, soit 2 048 000 décisions PPO. La phase d’entraînement a duré environ 11 minutes. Le statut final est `incomplete`, code 2, parce que LOC n’est pas validé. Les limites de 280 minutes de pilotage et de 350 minutes du job n’ont pas été atteintes. Les étapes de compilation et les tests ont réussi ; l’artefact avec `latest` a bien été publié.

Les **100 validations sont toutes à 0/20**. Les 3 514 épisodes d’apprentissage se répartissent en 1 325 crashs, 2 188 timeouts et une seule capture pédagogique stabilisée. Aucun BEST n’est promu. Ce résultat constitue un échec d’apprentissage ; les tests fonctionnels du programme ne prouvent pas la convergence du curriculum.

## Relecture du vrai checkpoint

L’artefact du run contient `PPO_FLIGHT_HYBRID_V1 56 5 128 1000`. La politique est relue sans apprentissage sur les 20 départs indépendants de validation, seed 4342 :

| Mesure | Résultat |
|---|---:|
| Captures pédagogiques terminées | 0/20 |
| Épisodes avec capture géométrique LOC | 4/20 |
| Fins | 6 crashs, 14 timeouts |
| Échantillons avec LOC capturé | 743 |
| CAS pendant ces captures | 34,35–41,41 m/s |
| CAS admissible pour la gate | 51–61 m/s, consigne 56 |
| Maintien stabilisé maximal observé | 0 s |
| Écart-type profondeur / ailerons / direction | 1,6487, au plafond exp(0,5) |

Les 743 échantillons capturés échouent tous au critère vitesse. Certaines trajectoires connaissent aussi des vitesses de tangage importantes. Une capture du faisceau et une capture physiquement stabilisée ne désignent donc pas la même réussite. Le score nul est réel, même si la politique rejoint parfois le LOC.

## Deux défauts corrigés

### Exploration disproportionnée

Les commandes restent des échantillons gaussiens bruts, bornés linéairement à l’entrée des gouvernes, à 10 Hz. Le scénario précédent commençait à σ=exp(−1)=0,368 et pouvait atteindre σ=exp(0,5)=1,649, très large devant la course normalisée [−1,+1]. Le checkpoint atteint effectivement ce plafond. Cette mesure ne suffit pas à attribuer à elle seule tout l’échec LOC.

Le scénario livré utilise maintenant `initialLogStd=-3`, `maxLogStd=-2`, `minLogStd=-5` : σ initial ≈0,050, plafond ≈0,135. Le bonus d’entropie est ramené à 0. Le réseau garde ses moyennes libres, ses 56 observations, ses cinq sorties continues et son Bernoulli de train ; les gouvernes conservent leur pleine autorité. Aucun tanh, lissage, pilote automatique ou changement JSBSim n’est ajouté.

Reproduction physique contrôlée : mêmes moyennes nulles, seed d’action 5243, départ à 60 m/s, 6 km, tangage 2°, pente −3°, piste fixe sans houle, 20 secondes :

| Bruit continu | RMS du delta profondeur | Pic de tangage | Roulis absolu maximal |
|---|---:|---:|---:|
| Ancien σ initial 0,368 | 0,546 | 16,75°/s | 27,72° |
| σ initial 0,0498 retenu | 0,0738 | 2,09°/s | 3,79° |

La baisse de tremblement est mesurée pour cet essai. Les commandes neutres ne constituent toujours pas un vol équilibré : sans trim, le même essai déterministe perd environ 142 m en 20 secondes. La commande Bernoulli de train reste une autre source de variations. Aucune stabilité globale ni capacité d’atterrissage n’est déduite de cette expérience.

### Objectif terminal différent entre apprentissage et validation

Avec `ilsCaptureScale=2`, l’apprentissage LOC pouvait terminer à ±5°, contre ±2,5° en validation ; Glide utilisait ±1,4°, contre ±0,7°. Une fin pédagogique dans l’enveloppe élargie ne démontrait donc pas l’interception nominale attendue.

`Runtime/flight_session.cpp` conserve maintenant les tolérances nominales pendant **les leçons LOC et Glide**, y compris les rappels LOC. Le facteur reste disponible pour Joint. Les exigences de vitesse, roulis, p/q/r, cap, route, maintien continu, priorité du crash et troncature sont inchangées. Le contrat passe à `curriculumVersion=3`.

La ligne informative `[FLIGHT_VALIDATION] success=...` utilise désormais le succès de la gate active. Elle affichait auparavant le taux d’arrêts sur piste, donc 0 même pour une leçon avec des captures réussies. La ligne `[FLIGHT_GATE]`, utilisée par le runner CI, était correcte et reste identique. Cette erreur d’affichage n’explique pas les 100 vrais verdicts nuls du run #6.

## Reprise des modèles et périmètre

Le chargeur autorise le warm start d’un checkpoint complet avec d’anciennes bornes gaussiennes valides, sans réutiliser d’échantillons ni de vraisemblances d’un ancien rollout. Poids, moments Adam, âge et RNG sont conservés ; les nouvelles actions et leurs probabilités sont calculées selon les bornes courantes. Les bornes malformées et les mappings incompatibles restent refusés. Un ancien BEST est conservé mais son score doit être revalidé sur le contrat exact.

La reprise réelle du checkpoint du run #6 a été vérifiée : update 1000 → 1001, sauvegarde valide, validation nominale terminée, aucun BEST artificiel. Son résultat reste 0/20 ; une update n’efface pas les défauts de la politique précédente.

Les workflows, le runner `tools/flight/curriculum_ci.py` et ses tests restent identiques octet pour octet à la base de la PR. L’hippodrome demeure désactivé et réactivable ; aucun point ni reward Path ne revient en approche. Les formules, poids et événements des dix canaux de récompense ne changent pas. BlueDefense et RLMission restent hors périmètre.

## Expériences d’apprentissage et limites

Toutes les expériences suivantes restent distinctes du scénario livré et n’ont pas créé de BEST de production :

| Expérience | Budget et résultat | Interprétation |
|---|---|---|
| Bruit réduit seul, départs 4–10°, ancien terminal d’entraînement ±5° | 300 updates, 124 captures d’apprentissage / 1 180 épisodes, validation nominale 0/20 | Le terminal large facilite des captures insuffisantes ; aucun passage LOC démontré |
| Bruit réduit, terminal nominal, départs facilités 2,6–4°, cap vers le seuil ±0,5° | 600 updates ; 2/20 sur seed 4342 et 1/20 sur seed 8342 | Quelques captures stabilisées apprises, très loin des 19/20 requis |
| Même politique précédente évaluée avec les départs livrés 4–10°, cap ±2° | 0/20 sur 4342 et 0/20 sur 8342 | Pas de généralisation démontrée au scénario réel |
| Priorité vitesse expérimentale, poids Speed=20, reprise de l’update 263 jusqu’à 600 sur les départs facilités | 1/20 en validation | Modification non retenue ; elle n’apporte pas de passage de gate démontré |

Ces résultats ne sont pas des comparaisons multi-seeds d’entraînement ni une preuve causale complète de l’échec. Le correctif traite une source mesurée de secousses et un décalage d’objectif. **La convergence du scénario livré n’est pas validée et l’issue d’apprentissage reste ouverte.** Augmenter aveuglément le budget ou abaisser les critères de réussite ne résoudrait pas ce diagnostic.

## Vérification du correctif

- Linux GCC 14, CPU : builds Debug et Release réussis dans `build/debug` et `build/release` ; **15/15 CTest par configuration**.
- **7/7 tests Python** du runner existant, sans modification de ce runner.
- Régressions : tolérances LOC/Glide identiques avant/après validation, facteur Joint conservé, bornes d’exploration, gradients PPO/entropie par différences finies, reprise d’anciennes distributions, refus des signatures malformées et protection des BEST.
- Vrai runner inchangé + vrai exécutable, scénario contrôlé volontairement facile (0,5°, maintien 0,1 s) : 20/20, ligne informative `success=1`, gate validée, état `completed`, prochaine étape `glide`, aucun BEST. Ce test vérifie l’intégration, pas l’apprentissage du scénario livré.
- Fonctions modifiées, appelants, tests et diff complet contre la base relus. La matrice GitHub du nouveau commit doit être contrôlée séparément ; les résultats de la tête précédente ne sont pas attribués au correctif.

Sources : journaux du [job 131618165229](https://github.com/benoitfragit/PPOTrial/actions/runs/43343797520/job/131618165229), [artefact du run](https://github.com/benoitfragit/PPOTrial/actions/runs/43343797520/artifacts/14002035400), source de la PR #87 et relectures natives JSBSim du checkpoint. Les documents des issues #86 et #89 décrivent les états antérieurs ; ce diagnostic les complète sans les remplacer.
