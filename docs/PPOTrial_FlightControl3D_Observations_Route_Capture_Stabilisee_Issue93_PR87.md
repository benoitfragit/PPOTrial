# FlightControl3D — observations de route et réussite stabilisée

Date : 13 novembre 2026, Europe/Paris. Branche : `fix/flight-approach-only`. [Issue #93](https://github.com/benoitfragit/PPOTrial/issues/93), [PR #87](https://github.com/benoitfragit/PPOTrial/pull/87). Complément du [correctif piste fixe/mobile #92](PPOTrial_FlightControl3D_Piste_Fixe_Joint_Mobile_Vitesse_Issue92_PR87.md).

## Audit repris et défauts corrigés

L'[audit d'alignement](https://github.com/benoitfragit/Context/blob/main/PPOTrial/Architecture/PPOTrial_FlightControl3D_Audit_Alignement_Cap_ILS_PR87.md) examine `d892f28`. Il distingue les défauts vérifiés des hypothèses d'apprentissage. Le correctif #92 traite la référence de vitesse distante et livre le curriculum **LOC fixe → Glide fixe → Joint fixe → Joint mobile**, sans houle, dans la CI existante. Ce complément corrige deux autres défauts vérifiés.

| Constat | Correction | Preuve |
|---|---|---|
| `course_error_sin/cos` restent à 0/1 en approche, car le point de guidage est désactivé | Fournir la route physique relative à l'axe piste dans ces deux entrées existantes | Grille déterministe 0°, 90°, 181°, 359°, dérapage, deux côtés, piste fixe/mobile et contrôle JSBSim natif |
| `CaptureCompleted` n'apporte rien au reward ; les bonus géométriques arrivent avant la stabilité | Reporter les bonus existants des leçons LOC/Glide à leur véritable réussite stabilisée | Faux passage, maintien partiel/interrompu, succès complet, crash, timeout et non-répétition |
| Cap/route/roulis/rates sont exigés par la gate, mais l'événement récompensé est seulement géométrique | Le bonus de leçon exige maintenant exactement le terminal contrôlé par cette gate | Conditions de réussite identiques entre entraînement et validation, à chaque sous-pas |

## Entrée de route

La route utilisée est `chi = atan2(v_relative_cross, v_relative_along)`. La vitesse relative est celle de l'avion moins la vitesse de la surface réelle de piste, projetée dans les axes piste. Les observations 27 et 28 contiennent `sin(chi)` et `cos(chi)` en approche, arrondi et roulage. Le passage angulaire ±π est continu dans cette représentation ; `atan2(0,0)` reste fini.

La route n'est pas remplacée par le cap fuselage : un dérapage ou un mouvement du pont peut séparer les deux. Les observations de cap existantes sont conservées. Un cap qui change sans changement de vitesse modifie l'entrée de cap, pas celle de route. Le signal est relatif à la piste réellement tirée, pas au cap nominal de la configuration.

`targetValid` indique uniquement la validité d'un point géométrique. Il reste faux en approche. Les trois premières entrées de point restent nulles et le renderer ne réintroduit aucune ligne HIPP. Lorsque la patrouille optionnelle est activée, son erreur de direction vers le point conserve sa définition historique.

Le nombre d'entrées reste **56** ; aucun changement de taille de réseau, de sortie d'action ou de distribution PPO.

## Bonus aligné sur l'objectif

Les canaux et leurs poids ne changent pas. Leur budget événementiel maximal reste identique :

| Leçon | Événement qui paie | Attribution |
|---|---|---|
| LOC | `CaptureCompleted` après maintien continu | `EnterLocalizer=+2` |
| Glide | `CaptureCompleted` après maintien LOC/Glide complet | `EnterLocalizer=+2`, `EnterGlide=+2` |
| Rappel LOC pendant Glide | Succès de la leçon LOC effective | `EnterLocalizer=+2` |
| Joint fixe et Joint mobile | Entrées géométriques historiques, toucher, arrêt | Inchangé |

Les leçons LOC/Glide ne paient plus ces bonus à la simple entrée dans le faisceau. Il faut maintenir pendant les 2 secondes configurées la géométrie stricte, le cap/route ±5°, la CAS 51–61 m/s, le roulis ±8° et les rates ±5°/s ; Glide impose aussi sa pente. Un écart remet le compteur continu à zéro. Un crash est prioritaire. Un timeout incomplet ne paie aucun bonus de succès. Le terminal empêche toute répétition du paiement.

Le shaping dense conserve ses formules et ses poids. Le succès de leçon ne fabrique ni toucher ni arrêt, et ne promeut aucun BEST d'atterrissage.

## Reprise et validation

Le contrat passe à `guidanceVersion=16`, `rewardVersion=13` ; `curriculumVersion=5` reste celui des quatre phases. Un ancien latest connu peut amorcer l'entraînement avec un nouveau rollout. Le score d'un BEST de l'ancien contrat n'est pas réutilisé ; les générations précédentes restent conservées et une validation exacte est exigée. La qualification fixe/mobile du correctif #92 reste distincte.

Les deux nouvelles régressions ont été exécutées **avant correction** : l'ancienne observation échoue sur la route, l'ancien reward échoue sur le passage géométrique prématurément récompensé. Après correction et relecture des fonctions, appelants, tests et diff :

- GCC 14, CPU headless : builds Debug et Release réussis, **15/15 CTest dans chaque configuration**.
- Les contrôles natifs couvrent les observations de route dans les quatre phases et la conservation des commandes d'atterrissage de Joint mobile.
- Les tests de reprise couvrent l'ancien contrat route/reward et le refus d'une qualification BEST périmée.
- Les **11 tests Python** du runner à quatre phases ont été validés avec le correctif #92 ; ce complément n'en change pas le code.

La CI distante doit être lue pour le SHA final publié. Les tests déterministes vérifient le comportement logiciel ; ils ne démontrent pas la convergence d'une politique. Aucun nouveau résultat d'apprentissage ou BEST qualifié n'est revendiqué.

## Constats qui restent des pistes expérimentales

L'audit ne démontre pas d'inversion systématique du signe LOC/cap. Les régressions de signe et les impulsions JSBSim restent actives. Un shaping continu supplémentaire du cap/roulis/rates n'est pas ajouté arbitrairement : le reward terminal exprime maintenant leur objectif collectif déjà vérifié par la gate. L'horizon GAE caractéristique d'environ 1,68 s et `Slope` pendant LOC restent des paramètres à évaluer par expériences comparables, sans modification de `gamma`, `lambda` ou des poids dans ce correctif.

Périmètre : FlightControl3D et sa documentation uniquement. Pas de nouveau workflow, de modification BlueDefense/RLMission, de CUDA ou de merge.
