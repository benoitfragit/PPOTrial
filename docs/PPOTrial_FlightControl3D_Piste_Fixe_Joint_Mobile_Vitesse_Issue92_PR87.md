# FlightControl3D — piste fixe, Joint mobile et vitesse relative physique

Date : 13 novembre 2026, Europe/Paris. Dépôt : PPOTrial. Branche : `fix/flight-approach-only`. [Issue #92](https://github.com/benoitfragit/PPOTrial/issues/92), [PR #87](https://github.com/benoitfragit/PPOTrial/pull/87). État initial inspecté : `d892f289458c0b00fe68a457c2476fc062107fba`.

## Comportement livré

Le curriculum comporte désormais **quatre phases dans la même CI** `FlightControl3D Curriculum Training`, avec le même acteur et le même critic. Il ne passe à la phase suivante qu'après sa validation indépendante. Aucun entraînement n'est lancé par ce correctif.

| Ordre | Phase / valeur `--stage` | Vitesse de piste | Objectif de passage |
|---|---|---|---|
| 1 | LOC / `loc` | 0 m/s | Capture LOC physiquement stabilisée |
| 2 | Glide / `glide` | 0 m/s | Capture Glide stabilisée, avec rappels LOC existants |
| 3 | Joint fixe / `joint` | 0 m/s | Atterrissage sûr puis arrêt |
| 4 | Joint mobile / `joint-mobile` | Tirage dans `[0,3]` m/s | Atterrissage sûr puis arrêt sur piste en translation |

Le seuil reste 19 succès sur 20 épisodes, avec une graine d'évaluation indépendante. Joint fixe ne déclenche pas immédiatement une translation et ne termine pas le curriculum progressif : Joint mobile est une phase séparée. Le mode manuel permet de sélectionner une phase explicitement.

Dans les quatre phases, le scénario livré met les amplitudes de pilonnement, tangage et roulis à `[0,0]`. La houle reste implémentée et testée, mais désactivée. La position nord/est, le cap et la hauteur de piste restent aléatoires à chaque reset ; sa hauteur locale est tirée dans `[14,16]` m. Elle reste constante pendant l'épisode. La translation ultérieure ne change pas cette hauteur.

Le JSON conserve la plage de vitesse mobile. `FlightEnvironment::Reset` impose localement `[0,0]` aux trois premières phases, puis échantillonne **la configuration effective de l'épisode**. Le callback physique JSBSim, le guidage et les observations reçoivent ainsi la même cinématique. Le scénario par défaut reste `joint`, donc fixe aussi en desktop.

## Défaut de vitesse corrigé

L'ancien guidage soustrayait la vitesse rigide `v_seuil + omega × (position_avion - origine_piste)`. Le calcul est une dérivée de coordonnées dans un repère tournant, mais prolonge la piste jusqu'à un point fictif situé plusieurs kilomètres avant son seuil. Il ne représente pas la vitesse d'une surface réelle à cet endroit.

Exemple reproduit avec l'ancienne enveloppe : tangage d'amplitude 0,25°, période 8 s, avion à 6 km avant le seuil. Le seul terme lié à la distance atteint `6000 × (2π/8) × radians(0,25) ≈ 20,56 m/s`. Une descente physique peut alors apparaître comme une montée dans le calcul de pente, les observations et la gate Glide. Ce n'est pas une exagération de la CAS fournie par JSBSim ; c'est une mauvaise référence de vitesse relative.

`RunwaySurfaceVelocity` projette maintenant la position sur la **surface finie** : abscisse bornée entre 0 et la longueur, ordonnée entre les deux bords, hauteur nulle. La vitesse est celle de ce point matériel réel. Avant le seuil sur l'axe, la référence est la vitesse du seuil, indépendamment de la distance d'approche. Au-dessus du pont, elle est celle du point de surface sous l'avion.

`Guide`, `Observe`, les critères de toucher et d'arrêt utilisent cette référence physique, exprimée dans les axes de piste. Elle est explicitement distinguée de la dérivée des coordonnées tournantes. `DeckVelocity` conserve la loi rigide exacte ; le callback de contacts JSBSim continue de l'utiliser aux points de surface réels. Aucun écrêtage arbitraire de vitesse physique n'est ajouté.

## Réseau, récompenses et BEST

Le réseau, les 56 observations, les sorties continues et le train binaire sont conservés. Les deux phases Joint apprennent le train et le frein au roulage ; elles poursuivent jusqu'au toucher et à l'arrêt. Seules LOC/Glide utilisent le terminal de capture et le train initial fixe. Aucun canal, poids ou bonus de reward n'est ajouté.

Le contrat passe à `guidanceVersion=15`, `curriculumVersion=5`. Un ancien latest clampé peut servir de point de départ, avec un nouveau rollout. Le changement des trois amplitudes de houle est accepté pour cette reprise après vérification de leurs bornes. Les autres différences de paramètres physiques restent protégées.

Le manifeste BEST porte `mobileRunway`. Une réussite sur piste fixe ne réutilise pas son score en phase mobile, ni l'inverse. Les anciens modèles restent conservés ; une nouvelle validation dans le mode courant est exigée. La reprise du même acteur/critic ne constitue pas une qualification.

## Même workflow et reprise

Le workflow existant `.github/workflows/flight-control-training.yml` reçoit seulement le choix `joint-mobile` et la sauvegarde de `curriculum_version.txt`. Le runner progressif enchaîne `loc → glide → joint → joint-mobile → done`. Les budgets, l'intervalle de validation, les noms d'artefacts et les règles d'échec pour entraînement incomplet restent en place.

Le marqueur de version vaut `5`. Un ancien `done` sans ce marqueur reprend en Joint fixe, puis doit valider Joint mobile. Un `done` de version 5 reste idempotent. Une interruption ne peut pas transformer le simple passage de Joint fixe en curriculum mobile validé. La sélection manuelle `joint` valide uniquement cette phase et enregistre `joint-mobile` comme suite.

## Vérifications et limites

- Builds locaux GCC 14, CPU headless : Debug et Release réussis dans `build/debug` et `build/release` ; **15/15 CTest dans chaque configuration**.
- Runner : **11/11 tests Python**, couvrant les quatre gates, les échecs répétés, budgets et reprises, Joint fixe sans faux succès mobile, ancien `done`, mode manuel et version inconnue.
- Régression de vitesse : le nouveau test échoue avec l'ancien `RelativeVelocity`, avant l'application du correctif. Après correction, la pente et la gate Glide restent cohérentes à plusieurs distances et caps. La vitesse de contact est comparée à la dérivée numérique de la surface mobile.
- JSBSim natif : vérification de la vitesse nulle pendant les trois premières phases malgré une plage source non nulle, puis translation réelle en quatrième phase ; positions/altitudes variables entre resets, absence de houle, observations cohérentes et commandes de train.
- Reprise et BEST : ancien latest avec houle réutilisable, anciens scores invalidés ; qualification fixe/mobile distincte.
- Exécutable réel : essai court d'une update en `joint-mobile`, deux épisodes de validation, verdict correctement négatif et aucun BEST créé. Cet essai vérifie le chemin d'exécution ; il ne mesure pas la convergence.

Les fonctions modifiées, leurs appelants, tests et diff de PR sont relus. La CI distante est à associer au SHA publié ; les résultats d'une tête antérieure ne valident pas ce correctif. Les mesures d'apprentissage 5/20 documentées pour `d892f28` concernent l'ancien scénario mobile et ne sont pas des résultats de ce nouveau curriculum. Aucune convergence ou politique d'atterrissage n'est déclarée qualifiée.

Périmètre : FlightControl3D, son runner, ses réglages et la sélection supplémentaire dans sa CI existante. BlueDefense et RLMission sont inchangés. Pas de CUDA, de nouveau workflow ni de merge.
