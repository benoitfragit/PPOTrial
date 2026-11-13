# FlightControl3D — approche seule et départs proches de l’ILS

Date : 13 novembre 2026 (Europe/Paris). Projet : PPOTrial. Issue : [#86](https://github.com/benoitfragit/PPOTrial/issues/86). Branche : `fix/flight-approach-only`. Base : `e7d2474a739b25851141bc379c0112c9d45c64f2`.

## Comportement livré

`mission.patrolEnabled: false` désactive temporairement l’hippodrome. Son code, ses paramètres, son séquenceur et le poids du canal `Path` sont conservés. Le flightplan publié est vide : aucun circuit ni point hippodrome n’est présenté au rendu ou à la politique. Le canal `Path` vaut exactement zéro, même si son poids est non nul.

L’épisode commence en `Approach`, y compris si une ancienne configuration demande `initial.approach: false`. Une perte LOC ou GLIDE conserve ce mode jusqu’au toucher, au crash ou au timeout ; le retour HIPP et son délai de réactivation sont désactivés. Les captures physiques restent réversibles avec leur hystérésis : perdre GLIDE conserve LOC, perdre LOC libère les deux. Une recapture ne répète pas les bonus d’entrée. Le roulage après un toucher sûr reste actif.

Tous les guidages d’approche, d’arrondi et de roulage gardent `targetValid=false`, avec une cible défensive égale à la position avion. Les trois composantes de l’observation vers le point sont nulles. Les erreurs LOC/GLIDE, les consignes de vitesse/pente et les signaux de sécurité restent disponibles.

## Initialisation

Les distances sont horizontales et mesurées depuis le seuil dans le repère de la piste réellement tirée pour l’épisode, avec sa translation, son orientation et sa houle.

| Réglage | Avant | Désormais |
|---|---|---|
| Distance du seuil | 5–14 km | 4–6,5 km |
| Azimut des départs latéraux | 12–30° de chaque côté | 4–10° de chaque côté |
| Hauteur des départs latéraux au-dessus du plan piste | 400–1 000 m | 180–400 m |
| Écart de cap autour de la direction du seuil | ±3° | ±2° |
| Départs dans l’axe | Un tiers, sur le glide | Conservés, sur le glide de 3° |

Les départs latéraux visent le seuil, avec au plus environ 12° d’écart au cap piste. L’ordonnanceur équilibré gauche/axe/droite est conservé pour Joint ; LOC utilise les deux côtés et Glide utilise l’axe avec ses écarts verticaux et ses rappels LOC. Les plages de vitesse, roulis et tangage restent inchangées. Aucun pilote automatique n’est ajouté.

## Découpage et réversibilité

| Couche | Modification |
|---|---|
| `Core/flight_config.*` | Booléen JSON obligatoire et strict ; contrat `guidanceVersion=14` ; reprise compatible des plages de position initiale et de l’activation hippodrome |
| `Core/flight_mission.cpp` | Flightplan vide lorsque désactivé, protection contre les anciens états/cibles, aucun suivi ni reward Path, maintien Approach après perte ILS |
| `Simulator/flight_environment.cpp` | Reset directement en approche lorsque la patrouille est désactivée |
| JSON scénario | Désactivation et départs rapprochés orientés vers le seuil |
| Tests | Pertes/recaptures répétées, réactivation, observation/rendu, départs JSBSim et reprise de modèle |

Pour réactiver l’ancien fonctionnement, mettre `mission.patrolEnabled: true`. Le retour hippodrome après perte ILS est alors restauré. Ajouter `initial.approach: false` pour commencer par la patrouille. Les réglages géométriques et le poids `Path=0.002` sont conservés pour cette réactivation.

Les 56 observations, les cinq actions gaussiennes et le train Bernoulli conservent leurs dimensions. `rewardVersion=11`, `spawnVersion=2` et `actionMappingVersion=2` restent inchangés. Un `latest` utilisant déjà les actions clampées peut reprendre avec acteur, critic, Adam et RNG ; les anciens checkpoints à action `tanh` restent incompatibles, conformément au correctif #85. Un ancien BEST est conservé mais son score exige une nouvelle validation du contrat exact.

## Vérification

- Relecture des fonctions modifiées, de leurs consommateurs et du diff complet contre la base ; aucun changement RLMission, BlueDefense, physique JSBSim ou formule LOC/GLIDE.
- Linux GCC 14 : builds headless CPU Debug et Release réussis ; **15/15 CTest dans chaque configuration**.
- **7/7 tests Python** du pilote de curriculum réussis.
- **72 resets réels JSBSim** supplémentaires sur LOC/Glide/Joint : 4–6,5 km, secteur ILS valide, cap entrant, piste mobile/inclinée, aucun point ; première action sans Path ni HIPP.
- Trois cycles de pertes LOC et GLIDE prolongées, recaptures et bonus uniques ; tests historiques du circuit, du reward Path et du retour HIPP maintenus avec activation explicite.
- Reprise d’un contrat V13 sans le nouveau champ et avec les anciennes plages de départ ; ancien BEST conservé puis revalidation exigée.

La matrice CI Windows/Linux et les builds desktop VS2022 sont à vérifier sur la PR. Ces tests fonctionnels ne démontrent pas l’apprentissage ni la convergence d’une politique d’atterrissage.
