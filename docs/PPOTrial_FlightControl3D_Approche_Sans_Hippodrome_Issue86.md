# FlightControl3D — approche seule et départs proches de l’ILS

Date : 13 novembre 2026 (Europe/Paris). Projet : PPOTrial. Issues : [#86](https://github.com/benoitfragit/PPOTrial/issues/86), [#88](https://github.com/benoitfragit/PPOTrial/issues/88). PR : [#87](https://github.com/benoitfragit/PPOTrial/pull/87). Branche : `fix/flight-approach-only`. Base : `e7d2474a739b25851141bc379c0112c9d45c64f2`.

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

Les 56 observations, les cinq actions gaussiennes et le train Bernoulli conservent leurs dimensions. `rewardVersion=11` et `actionMappingVersion=2` restent inchangés ; le reset corrigé ci-dessous utilise `spawnVersion=3`. Un `latest` utilisant déjà les actions clampées peut reprendre avec acteur, critic, Adam et RNG ; les anciens checkpoints à action `tanh` restent incompatibles, conformément au correctif #85. Un ancien BEST est conservé mais son score exige une nouvelle validation du contrat exact.

## Stabilité du vol : défaut corrigé et facteurs mesurés

### Tangage initial écrasé par JSBSim — issue #88

`JsbsimAircraft::Reset` réglait le tangage puis appelait `FGInitialCondition::SetClimbRateFpsIC`. Dans la version épinglée de JSBSim, cette fonction recalcule theta/beta pour conserver l’incidence. Le tangage tiré était donc écrasé : avec une pente nulle, les demandes −4°, +2° et +8° donnaient toutes environ 0° ; avec une pente −3°, elles donnaient environ −2,94°.

Le reset initialise maintenant les composantes de vitesse NED, avec une norme cohérente avec la CAS et le taux vertical demandé, puis fixe l’attitude. Les setters NED empêchent cette dernière opération de tourner la vitesse. La direction horizontale vient du vecteur demandé ; le cap est utilisé si ce vecteur est vertical. Une vitesse verticale supérieure à la vitesse vraie est rejetée explicitement. Aucun équilibrage automatique ni pilote automatique n’est ajouté.

Les régressions réelles vérifient 108 combinaisons (quatre caps piste, trois roulis, trois tangages et trois pentes), chacune réinitialisée une seconde fois après une manœuvre : **216 resets**. Roulis/tangage/cap et CAS sont conservés ; les vitesses ne dépendent pas de la manœuvre précédente. Le contrat `spawnVersion=3` permet le warm start de V2, tout en exigeant une nouvelle validation BEST.

### Commandes PPO et oscillations — mesures de reproduction

Un acteur initialisé produit des moyennes nulles et `initialLogStd=-1`, soit sigma = exp(−1) = 0,368. En entraînement, de nouveaux échantillons indépendants sont appliqués toutes les 100 ms ; la profondeur, les ailerons et la direction attaquent directement les commandes du C310, sans limitation de vitesse de déplacement de ces commandes dans l’adaptateur. Le clip borne l’amplitude mais ne lisse pas les à-coups. Le XML C310 épinglé utilise un summer puis un aerosurface_scale pour ces trois axes.

Expérience après correction du reset : même graine d’action 5243, piste fixe sans houle, 60 m/s, départ dans l’axe à 6 km, tangage demandé 2°, pente −3°, commandes maintenues entre deux décisions, 20 secondes sans apprentissage. Les deux lignes gaussiennes utilisent des moyennes nulles constantes ; ce ne sont pas les résultats d’un modèle entraîné de l’utilisateur.

| Commandes | RMS du delta de profondeur sur [−1,+1] | Pic de vitesse de tangage | Roulis maximal absolu |
|---|---:|---:|---:|
| Moyennes déterministes, profondeur nulle | 0 | 1,38°/s | 0,024° |
| Gaussiennes, sigma initial 0,368 | 0,546 | 16,75°/s | 27,72° |
| Gaussiennes, sigma 0,0498, expérience seulement | 0,0738 | 2,09°/s | 3,79° |

La réduction expérimentale du bruit réduit fortement le tremblement ; ce réglage n’est pas modifié dans le scénario livré. Le Bernoulli initial produit aussi 107 inversions de consigne de train en 20 secondes dans ce tirage. Réduire uniquement le bruit continu ne change pas cette composante.

Les commandes neutres ne sont pas un vol équilibré : sans trim, le même essai perd environ 142 m en 20 secondes et sa pente dérive. Un test direct du trim longitudinal natif échoue avec `Trim Failed` / `udot doesn't appear to be trimmable` dans cette configuration ; aucun bénéfice de trim n’est revendiqué. La petite différence entre volets identiques au reset et déploiement des volets au premier pas n’explique pas le tremblement principal de cet essai.

Enfin, le worker desktop avance 100 ms de simulation par décision et attend seulement 2 ms entre deux publications, hors temps de calcul. Le renderer utilise le snapshot courant sans interpolation temporelle. Cela peut amplifier visuellement les secousses et accélérer fortement le temps simulé ; c’est une déduction du code, pas une mesure du FPS de la machine utilisateur.

Le défaut de reset est corrigé. L’exploration stochastique, l’absence d’équilibrage initial et la présentation accélérée restent des facteurs distincts. Ce correctif ne démontre donc pas qu’une politique sait déjà maintenir une altitude ou suivre le glide.

## Vérification

- Relecture des fonctions modifiées, de leurs consommateurs et du diff complet contre la base ; aucun changement RLMission, BlueDefense, physique JSBSim ou formule LOC/GLIDE.
- Linux GCC 14 : builds headless CPU Debug et Release réussis ; **15/15 CTest dans chaque configuration**.
- **7/7 tests Python** du pilote de curriculum réussis.
- **72 resets réels JSBSim** supplémentaires sur LOC/Glide/Joint : 4–6,5 km, secteur ILS valide, cap entrant, piste mobile/inclinée, aucun point ; première action sans Path ni HIPP.
- Trois cycles de pertes LOC et GLIDE prolongées, recaptures et bonus uniques ; tests historiques du circuit, du reward Path et du retour HIPP maintenus avec activation explicite.
- Reprise d’un contrat V13 sans le nouveau champ et avec les anciennes plages de départ ; ancien BEST conservé puis revalidation exigée.
- Après correction du tangage : builds Debug et Release et **15/15 CTest dans chaque configuration** relancés avec succès, dont les 216 resets et la reprise du contrat spawn V2.

La première révision `5d8c7f7` a passé la matrice FlightControl3D Windows/Linux Debug/Release (15 CTest et 7 tests Python par job), avec journaux relus. La CI doit être revérifiée sur le dernier commit de correction du reset. Ces tests fonctionnels ne démontrent pas l’apprentissage ni la convergence d’une politique d’atterrissage.
