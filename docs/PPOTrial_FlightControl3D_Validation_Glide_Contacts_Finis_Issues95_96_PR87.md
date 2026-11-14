# FlightControl3D — validation Glide séparée et contacts physiques finis

Date : 13 novembre 2026, Europe/Paris. Branche : `fix/flight-approach-only`, [PR #87](https://github.com/benoitfragit/PPOTrial/pull/87). Corrections [#95](https://github.com/benoitfragit/PPOTrial/issues/95) et [#96](https://github.com/benoitfragit/PPOTrial/issues/96), à partir de `34c7ba9865ac6b49e7cdf5d864634c7d77fef3dd`. Diagnostic du latest du [run 14](https://github.com/benoitfragit/PPOTrial/actions/runs/43421270187), update 11 000, comparé au [run 13](https://github.com/benoitfragit/PPOTrial/actions/runs/43406061312), update 8 000.

## Deux défauts corrigés

| Défaut démontré | Correction | Régression |
|---|---|---|
| Les rappels LOC de l'environnement de validation peuvent compter comme succès Glide | Validation Glide pure, puis campagne LOC séparée et compteurs distincts | Rappel forcé à 100 % pendant l'entraînement ; la validation doit rester Glide. L'ancien code échoue sur cette assertion |
| Un avion hors piste peut être déclaré en contact avec le prolongement infini du pont, alors que JSBSim le voit encore en vol | Contacts physiques natifs prioritaires ; repli géométrique limité au pont fini, puis à la mer hors piste | Quatre caps, avant/après et deux côtés, états natifs/synthétiques. L'ancien calcul échoue avant correction |

Les changements restent dans FlightControl3D et sa documentation. Les rappels LOC restent à 30 % en entraînement. Le réseau, les sorties, PPO, les poids et formules de reward, la durée des leçons, les tolérances et le seuil de passage ne changent pas. La CI existante et le runner Python ne sont pas modifiés.

## Validation de capture et conservation LOC — #95

`EnvironmentConfig` coupe les rappels pour une évaluation. `BeginValidationStage` fixe explicitement la leçon et crée son environnement à graine indépendante. Une campagne de phase Glide comprend :

1. **20 épisodes Glide purs**, chacun devant satisfaire le véritable terminal LOC + Glide stabilisés.
2. **20 épisodes LOC distincts** avec la même politique figée, pour mesurer l'oubli éventuel.

Le premier résultat reste la gate Glide ; les réussites du second échantillon n'y sont jamais ajoutées. Le contrôle LOC est **diagnostique** : il ne modifie pas le seuil ni les règles de progression de la gate Glide, qui exige déjà un LOC stabilisé dans chaque réussite Glide. Ses résultats apparaissent dans `[FLIGHT_RETENTION] stage=loc successes=... episodes=...`. Le format `[FLIGHT_GATE]` attendu par le runner est conservé.

Les deux échantillons passent sans Record/Optimize ; poids, moments et âges Adam, RNG de l'apprenant et rollout restent figés. Aucune capture pédagogique ne crée un BEST d'atterrissage. Les snapshots exposent la leçon effective et les deux compteurs de rétention. La campagne n'est publiée comme terminée qu'après les deux échantillons complets. En LOC et Joint, la validation conserve son chemin normal.

Les tests utilisent aussi une fixture contrôlée où LOC réussit et Glide échoue : toutes les réussites LOC restent dans la rétention, et la gate Glide reste à zéro. Ce test ne qualifie aucune politique apprise.

## Contacts finis et version de reprise — #96

Le callback JSBSim `DeckGround` représente une piste finie surélevée, puis la mer hors de son emprise. L'ancien `AdvanceMission` ajoutait partout le test `g.height <= contactHeightMeters`, où `g.height` est la distance au **plan infini de piste**.

Dans le rejeu, des vols ont ainsi reçu le terminal Crash avec WOW=false et structuralContact=false, hors piste, environ 15 m au-dessus de la mer. Le biais anticipait un impact sans contact physique ; il ne prouve pas que l'avion aurait été récupérable.

`ContactDetected` applique désormais :

- un contact roues ou structure effectivement signalé reste prioritaire ;
- avec les données natives complètes, l'absence de contact physique ne peut pas être remplacée par la distance au plan infini ;
- sans ces données, le repli géométrique utilise la hauteur de pont à l'intérieur de la surface finie, et la hauteur de mer à l'extérieur.

Les vrais impacts continuent de terminer en Crash avec la pénalité existante ; les règles de toucher sûr, arrêt, sortie de piste et timeout sont conservées. Un test JSBSim réel fait voler un avion hors pont à la hauteur du plan prolongé, puis vérifie un vrai contact avec la mer lors d'une chute contrôlée.

Le contrat porte `contactVersion=2`. Un ancien latest sans marqueur peut amorcer l'entraînement avec un nouveau rollout ; un BEST de l'ancien contrat doit être revalidé et reste conservé. Une version de contact inconnue est rejetée. Le warm start réel 11 000 → 11 001 a été exécuté dans une copie isolée via le runner officiel, avec le statut incomplet attendu et aucun BEST fabriqué.

## Mesures sur les mêmes checkpoints, sans optimisation

Les valeurs suivantes sont produites par **FlightControlHeadless officiel**, avec les deux corrections, les mêmes configurations et 20 épisodes par colonne. Le SHA-256 du latest reste identique après chaque validation.

| Graine | Run 13 : Glide pur | Run 13 : LOC séparé | Run 14 : Glide pur | Run 14 : LOC séparé |
|---|---:|---:|---:|---:|
| 4342, CI | 6/20 | 19/20 | 8/20 | 9/20 |
| 4343 | 11/20 | 20/20 | 11/20 | 8/20 |
| 7342 | 10/20 | 17/20 | 13/20 | 7/20 |

Le run 14 améliore un peu le Glide sur deux graines et dégrade nettement la conservation LOC. Les rappels à 30 % n'ont pas empêché cet oubli dans cette campagne. Aucun de ces modèles ne valide le Glide à 19/20 ; le run 13 ne généralise pas non plus un LOC qualifié aux trois graines. Les anciens scores Glide mélangés ne sont pas directement comparables à ces validations pures.

Le checkpoint 11 000 est conservé. Pour une reprise technique depuis le run 14, `resume_run_id=43421270187` garde ses acquis, puis revalide Glide et affiche le LOC séparé avant PPO. Si l'objectif est de repartir d'un LOC mieux conservé, le checkpoint 8 000 du run 13 constitue l'autre point de départ mesuré, sans être un champion Glide démontré. Aucune longue relance identique n'est présentée comme une solution prouvée à l'oubli.

## Diagnostic des crashes et limites

La sonde temporaire enregistre à 10 Hz position, hauteur, vitesse, pente, cap/route, attitude, p/q/r, maintien et commandes. Les terminaux restent ceux de la simulation native à 100 Hz.

Avant le correctif de contact, le latest 11 000 rejoué en Glide pur donne respectivement **3, 2 et 1 crashes** sur les trois graines : tous sont déclenchés par la hauteur géométrique, sans contact natif. Après correction, les mêmes relectures donnent **2, 2 et 1 vrais impacts structurels**. En LOC, les comptes passent de **1/3/2** à **1/3/1**. Les scores de capture restent identiques ; certains anciens impacts anticipés deviennent des timeouts, d'autres aboutissent réellement à la mer.

Le défaut de contact ne résout donc pas les pertes de contrôle. Une trace représentative montre, après dépassement du seuil et perte ILS, une CAS proche de 33 m/s, un cabré d'environ 13° et une pente d'environ −17° : ce comportement est compatible avec une forte incidence et un décrochage. L'incidence aérodynamique exacte n'est pas lue par cette sonde ; cette interprétation reste une inférence. Les trajectoires se terminent ensuite par de vraies pertes d'altitude et des impacts.

Un rejeu stochastique isolé de 100 épisodes avec le latest compte 9 crashes avant et après le correctif, tous physiques après correction. Ce n'est pas la reproduction des 1 401 crashes historiques : les poids évoluaient pendant le run 14 et les checkpoints intermédiaires ne sont pas tous conservés. Les changements de durée affectent aussi la séquence d'actions aléatoires des épisodes suivants. Aucun pourcentage historique de crashes corrigés n'est donc revendiqué.

Il reste à traiter expérimentalement l'oubli LOC et les sorties de trajectoire Glide. Aucun ajustement arbitraire de PPO/reward ou pilotage externe n'est introduit dans ces corrections vérifiables.

## Vérifications et relecture

- CPU, GCC 14, C++17 : builds **Debug et Release**, dans `build/debug` et `build/release`.
- **15/15 CTest** dans chaque configuration, incluant les vrais contacts JSBSim, toucher/arrêt, crashes et reprises.
- **18/18 tests Python** du runner inchangé.
- Régressions de validation et de contact géométrique constatées en échec sur l'ancien comportement.
- Six validations officielles figées : deux checkpoints × trois graines ; modèle original inchangé.
- Reprise native du latest 11 000 jusqu'à 11 001 dans une copie, validation Glide négative et aucun BEST.
- Relecture des fonctions modifiées, appelants, tests et diff contre la base de la PR ; périmètre FlightControl3D vérifié.

La CI distante doit être associée au SHA publié dans la PR ; les résultats d'une ancienne tête ne valident pas ces commits. Aucun merge ni entraînement distant n'a été lancé par ces travaux.

## Références

- [Issue #95](https://github.com/benoitfragit/PPOTrial/issues/95), [issue #96](https://github.com/benoitfragit/PPOTrial/issues/96), [PR #87](https://github.com/benoitfragit/PPOTrial/pull/87).
- [Run 14 et artefact](https://github.com/benoitfragit/PPOTrial/actions/runs/43421270187/artifacts/14039370512), latest SHA-256 `2210885372aef48b81abdf61589799332df7ebadc31d856434736a94dc3b97c0`.
- [Run 13 et artefact](https://github.com/benoitfragit/PPOTrial/actions/runs/43406061312/artifacts/14028944270), latest SHA-256 `ce534dede8750f1ed32f10597489b6d34b62a19f54811a85857ff1422763edbd`.
- [Guide CI](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/docs/PPOTrial_FlightControl3D_Curriculum_ILS.md).
