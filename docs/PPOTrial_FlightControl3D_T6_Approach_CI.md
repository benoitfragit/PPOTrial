# FlightControl3D T-6 : atterrissage et CI

Le scénario `FlightControl3D_T6_Joint.json` entraîne une seule politique PPO à
poser un T-6 Texan II (JSBSim) sur une piste fixe tirée au hasard à chaque
épisode, puis à s'arrêter. Le succès est un **Stop physique** après un toucher
sûr : ni le passage du seuil ni un survol ne valident l'épisode.

## Organisation du code

| Module | Rôle |
|---|---|
| `Core/flight_config` | Lecture stricte du JSON (format 2) et contrat de checkpoint |
| `Core/runway_geometry` | Repères piste et avion, surface d'atterrissage |
| `Core/flight_spawn` | Pose de piste et départs (approche lointaine, finale courte, roulage) |
| `Core/flight_guidance` | Référence d'approche, vitesse cible et direction commandée |
| `Core/flight_landing` | Détection de contact, limites de toucher, diagnostic de crash |
| `Core/flight_reward` | Potentiel de distance et coût de direction |
| `Core/flight_observation` | Observation normalisée de la politique |
| `Core/flight_mission` | Machine d'état approche → roulage → arrêt |
| `Simulator/` | JSBSim natif et boucle de transition |
| `Training/` | Acteur gaussien PPO (RLtools), checkpoints et BEST |
| `Runtime/` | Session partagée desktop/headless et réglage des récompenses |

## Géométrie d'approche et guidage

| Référence | Position dans le repère piste |
|---|---|
| Seuil | `along = 0`, centre de piste, sur le pont |
| Passage du seuil | `thresholdHeightMeters` (15 m) au-dessus du seuil |
| Point d'aboutissement | `along = thresholdHeightMeters / tan(glideDegrees)` ≈ 286 m, sur l'axe et **sur la piste** |

Le plan de descente est une droite unique à `glideDegrees` (3°) : il passe à la
hauteur de franchissement au-dessus du seuil et touche la piste au point
d'aboutissement, qui doit rester sur la première moitié de la piste. Les départs,
le guidage, l'observation, le potentiel `Path` et la vue de profil utilisent cette
même géométrie.

La direction commandée pointe directement vers le point d'aboutissement tant que
l'avion en est à plus de 300 m. Plus près, puis au-dessus de la piste, la cible
horizontale glisse sur l'axe 300 m devant l'avion : la trajectoire rejoint l'axe
sans virage brusque et continue vers l'avant, sans jamais viser un point derrière
l'avion. Sous `flareHeightMeters`, la descente se raccorde continûment à un taux
de contact doux (0,5 m/s). Au roulage, la direction est horizontale, vers l'axe.

Le potentiel `Path` vaut `w · L / (L + d)`, avec `d` la distance au point
d'aboutissement avant celui-ci, puis l'écart latéral et la hauteur au-delà : il
est continu au point d'aboutissement et ne pénalise ni le passage du seuil ni le
vol au-dessus de la piste. Une fois l'avion au roulage, il reste à sa valeur
maximale `w` : aucun signal de distance ne contredit l'arrêt.

## Actions

L'acteur produit cinq actions gaussiennes : profondeur, ailerons, direction,
gaz et frein. Une sortie nulle commande le **trim de la descente à 3°**
(`initial.throttle` = 0,40 et `initial.elevatorTrim` = −0,21, soit environ
−2,9 m/s à 55 m/s avec les volets d'approche) : une politique non entraînée
suit déjà le plan au lieu de piquer. La profondeur ajoute l'action au trim avec
un gain uniforme ; les gaz vont du trim au ralenti pour les actions négatives et
à pleine puissance pour les positives. Le frein n'agit qu'au sol et couvre
toute la plage brute : −1 relâche, 0 freine à moitié, +1 freine à fond. Un
écrêtage à zéro enverrait toutes les valeurs négatives sur « frein relâché » ;
la politique y dérivait et n'explorait plus le freinage, alors que sans frein
le T-6 sort en bout de piste même au ralenti. Une sortie neutre arrête
désormais l'avion sur la piste. La gouverne de direction braque aussi la roue
avant (±7° dans le modèle JSBSim, même sens de lacet que la gouverne) : sans
elle, la trajectoire au sol ne se corrigeait plus une fois l'avion ralenti. Les départs
en vol démarrent sur ces commandes trimées, les départs au roulage au ralenti.
Ces deux valeurs font partie du contrat de checkpoint.

## Récompenses

| Canal | Nature | Définition |
|---|---|---|
| `Path` | potentiel | `w·L/(L+d)` en vol (voir ci-dessus), `w` au roulage |
| `Direction` | coût courant | `−w·dt·(1 − cos θ)/2`, θ entre vitesse et direction commandée, en vol |
| `VerticalSpeed` | potentiel | `−w·min(|vz − vz_cible|/2 m/s, 1)` en vol, 0 au sol |
| `Speed` | potentiel | en vol `−w·min(|CAS − CAS_cible|/10 m/s, 1)` ; au sol `w·(1 − min(V_sol/V_arrondi, 1))` |
| `Touchdown` | événement | `w × qualité du toucher` |
| `Stop` | événement | `w × facteur de surface d'atterrissage` |
| `Crash` | événement | `w` (négatif) |

Chaque potentiel contribue `γ_sub·Φ(s') − Φ(s)` par sous-pas physique,
actualisé dans l'action pour donner `γ·Φ(s_{t+1}) − Φ(s_t)` à PPO. Un état
terminal (crash, arrêt) a un potentiel nul ; un timeout conserve le sien pour
l'amorçage du critique. Ces termes ne changent pas la politique optimale : ils
guident l'apprentissage sans pouvoir être exploités en boucle.

- `VerticalSpeed` vise la vitesse verticale commandée : environ −2,9 m/s sur le
  plan de 3°, puis −0,5 m/s sous la hauteur d'arrondi.
- `Speed` maintient la vitesse cible en vol (56 → 52 m/s), puis croît au sol à
  mesure que l'avion freine. Les deux définitions coïncident lors d'un toucher à
  la vitesse d'arrondi.
- La qualité du toucher est le produit de quatre facteurs : surface
  d'atterrissage (1 sur la piste, 0 en bord de marge), zone de toucher (1 jusqu'à
  900 m, puis 0 à mi-piste), taux de chute `1 − (V_chute/V_chute_max)²` et vitesse
  (de 1 à 0,5 entre la vitesse d'arrondi et la vitesse maximale de toucher).
  L'arrêt dépend seulement de la surface d'atterrissage.

### Poids et horizon

| Canal | Poids | Rôle |
|---|---:|---|
| `Stop` | 10 | objectif principal |
| `Touchdown` | 5 × qualité | étape intermédiaire, toucher doux |
| `Crash` | −5 | échec ; un essai d'atterrissage reste rentable dès 25 % de réussite |
| `Path` | 4 | progression vers le point d'aboutissement (`pathTolerance` = 1 500 m) |
| `VerticalSpeed` | 2 | taux de chute |
| `Speed` | 2 | vitesse en vol, freinage au sol |
| `Direction` | 0,5 | alignement ; 10° d'écart coûtent moins de 1 sur 200 s |

L'amplitude totale du shaping (8) reste inférieure aux récompenses
d'atterrissage (15) : l'objectif terminal domine. Les récompenses restent de
l'ordre de 10, car les cibles du critique ne sont pas normalisées.

Avec un pas d'action de 0,1 s, `gamma` = 0,999 donne un horizon d'environ 100 s :
depuis un départ à 14 km (≈ 2 500 actions), le `Stop` garde un poids actualisé
de 0,08 (contre 1e-11 avec 0,99). Le `lambda` de GAE vaut 0,98 : avec 0,95,
l'avantage ne portait que sur environ 2 s, si bien qu'un virage loin du sol
semblait éviter le crash avant que le critique ne sache en compter le coût
`Direction`, et l'agent apprenait à faire demi-tour. Les rollouts PPO passent à 8 192 pas
(minibatch 2 048), soit plusieurs épisodes par mise à jour. La limite de temps
de 480 s couvre l'approche la plus longue et le roulage.

## Observations

Les 33 entrées sont nommées par `ObservationIndex` et toutes bornées à [−1, 1].
Positions et vitesses sont exprimées dans le repère piste : l'observation ne
dépend pas du cap ni de la position tirés pour la piste.

| Groupe | Entrées | Normalisation |
|---|---|---|
| Position | distance au seuil, écart latéral, hauteur au-dessus du pont, écart au plan de descente | log signé (résolution 100 / 10 / 5 / 5 m) |
| Piste | piste restante | divisée par la longueur de piste |
| Vitesse | avant, latérale, verticale | / 80, / 20, / 10 m/s |
| Erreurs | vitesse air − cible, vitesse verticale − cible | / 10, / 5 m/s |
| Attitude | sin et cos du roulis, sin du tangage, sin et cos du cap relatif | — |
| Vitesses angulaires | p, q, r | / 0,5 rad/s |
| Guidage | direction commandée et direction de la vitesse, en axes avion | vecteurs unitaires |
| État | roues au sol, roulage, progression de l'arrêt, temps écoulé | 0–1 |
| Commandes | gouvernes, gaz et frein filtrés | plages physiques |

Le log signé `sign(x)·log1p(|x|/r)/log1p(max/r)` garde une résolution métrique
près du pont et sature au-delà des limites d'épisode.

## Curriculum d'entraînement

La section `training.curriculum` règle une difficulté de départ adaptative :

| Réglage | Valeur | Effet |
|---|---|---|
| `firstDistanceMeters` | 1 500–3 000 m | départs du niveau 0 : finale courte, sur l'axe et le plan, cap aligné |
| `levels` | 6 | le dernier niveau est la tâche complète (`initial`) |
| `promotionSuccessRate` | 0,6 | taux d'arrêts sur la fenêtre qui fait passer au niveau suivant |
| `windowEpisodes` | 30 | épisodes en vol par décision |
| `rolloutStartShare` | 0,15 | part des départs déjà posés, pour apprendre le freinage |

Entre les niveaux, la plage de distance est interpolée et les écarts de gisement,
de cap et de hauteur sont mis à l'échelle de la difficulté. Chaque départ en vol
tire sa difficulté entre 0 et le niveau courant, avec une densité croissante
vers ce niveau : la frontière est la plus pratiquée, et les cas faciles restent
révisés. Le niveau est enregistré avec `latest` (`latest/curriculum_level.txt`)
et restauré à la reprise.

Les validations suivent le même mélange que l'entraînement au niveau courant.
Une campagne de `evaluation.episodes` épisodes commence par la part
`rolloutStartShare` de départs au roulage (arrondie, au moins une approche
restant), puis enchaîne des approches à difficultés régulièrement réparties
jusqu'au niveau courant : finales courtes et alignées, puis départs plus
lointains, déportés et très déportés. Chaque épisode a sa graine
`evaluation.seed + i`. La difficulté est enregistrée dans le manifeste du BEST
(protocole d'évaluation 3) : une campagne plus difficile l'emporte toujours, et
à difficulté égale le taux d'arrêt puis la distance d'arrêt départagent. Seule
une campagne sur la tâche complète (difficulté 1) peut réussir le
`[FLIGHT_GATE]` : un niveau facile ne peut pas valider la CI. L'inférence
(`--evaluate`) joue le même mélange à difficulté 1.

## Entraînement et reprise

Les validations indépendantes utilisent la graine d'évaluation. Le checkpoint `latest/flight_policy.txt` (en-tête
`PPO_FLIGHT_POLICY_V3`) contient l'acteur, le critique, Adam et le RNG ; le BEST
validé est conservé séparément.

Un checkpoint peut redémarrer l'entraînement si seuls les poids des récompenses
ou les plages de départ changent (le trim excepté) ; tout autre changement de la signature exige
une nouvelle politique. Un BEST n'est réutilisé que pour une signature exacte.

## CI

**Actions → FlightControl3D Landing Training → Run workflow** compile
`FlightControlHeadless` en Release dans `build/release`, puis
`tools/flight/curriculum_ci.py` enchaîne des blocs de `additional_updates`
mises à jour PPO. Chaque bloc se termine par une validation indépendante
(`[FLIGHT_GATE] successes=… episodes=… difficulty=… passed=…`). Le runner s'arrête à la
première validation réussie sur la tâche complète ; un budget épuisé renvoie le statut `incomplete`
et un code non nul. `resume_run_id` restaure l'artefact `flight-control-training`,
dont le checkpoint est vérifié par le chargeur natif avant reprise.

Ces vérifications ne démontrent aucune convergence PPO : une politique doit être
jugée sur des évaluations à graines indépendantes.
