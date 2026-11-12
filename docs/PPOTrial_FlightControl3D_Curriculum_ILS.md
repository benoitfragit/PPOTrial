# FlightControl3D — curriculum LOC / Glide et CI

Date : 12 novembre 2026. Branche : feat/jsbsim-flight-control. PR : #83.

## Contrat de shaping

La qualité LOC reste une moyenne (0,5 / 0,5) entre la qualité d'écart latéral et la qualité de route d'interception. La route relative à la piste est dérivée de son vecteur vitesse, pas du cap fuselage :
- e_LOC = atan2(y,-x), Q_axe = 1-clamp(|e_LOC|/90°,0,1).
- chi_cible = -clamp(2 e_LOC, -60°, 60°).
- Q_route = 1-clamp(|wrap(chi_reelle-chi_cible)|/60°,0,1).
- Q_LOC = 0,5 Q_axe + 0,5 Q_route.

La qualité Glide combine désormais **la position et la pente de descente** :
- e_G = atan2(z-h_seuil,-x)-3°.
- e_pente = atan2(v_z,hypot(v_x,v_y))-gamma_cible, gamma_cible=-3° en approche.
- Q_GLIDE = 0,5 [2°/(2°+|e_G|) + 5°/(5°+|e_pente|)].

Par sous-pas physique de 0,01 s (sans décadence ni maintien constant) :
- r_LOC = 30 (Q_LOC_après - Q_LOC_avant).
- r_GLIDE = 10 (Q_GLIDE_après - Q_GLIDE_avant).
- r_V = 1 (Q_V_après - Q_V_avant), Q_V = 15/(15+|Vair - Vcible|).
- r_pente = 1 (Q_pente_après - Q_pente_avant).

Le reward total d'une action PPO est la somme de ses sous-pas JSBSim. Une transition hors enveloppe annule le delta de la composante correspondante ; une progression négative vers le seuil interdit un reward LOC/Glide positif. Ces exceptions signifient que le retour n'est pas strictement un potentiel global garanti invariant pour PPO avec gamma<1 : ce choix de façonnage est à évaluer expérimentalement. Les bonus événementiels +2/+2/+100/+100/−100 sont inchangés.

## Curriculum à réseau unique

Le paramétrage est dans training.curriculum du JSON (stage, locReminderProbability, glideOffsetMeters, gateHoldSeconds) et peut être surchargé par --stage pour FlightControlHeadless sans changer le réseau.

- Gate LOC : départs gauche/droite équilibrés, reward Glide et bonus de capture Glide éteints. Signal vitesse/pente de sécurité conservé même avant capture LOC.
- Gate Glide : 70 % d'épisodes démarrant sur le LOC mais ±(0,25..1)*glideOffsetMeters au-dessus ou au-dessous du plan ; 30 % d'épisodes de rappel LOC avec les mêmes règles que Gate 1. LOC reste récompensé à chaque épisode.
- Gate Joint : distributions complètes habituelles, les deux canaux de reward actifs, puis atterrissage et arrêt.

Les validations utilisent le seed d'évaluation indépendant et la tolérance de capture stricte.
Les erreurs de cap sont évaluées par rapport à la piste réellement tirée dans
l'épisode (orientation aléatoire et mobile), et non au cap nominal du JSON.
La capture seule ne suffit pas : le cap de fuselage est contrôlé à ±5° du cap piste,
la route relative à ±5°, la vitesse à ±15 m/s de la consigne (paramètre existant),
et en Gate Glide la pente de trajectoire est contrôlée à ±1,5° de la descente visée.
Les trois tolérances angulaires sont configurables par gateHeadingDegrees, gateRouteDegrees
et gateSlopeDegrees ; un crash annule la réussite pédagogique de l'épisode. LOC et Glide exigent le maintien confirmé du/des faisceaux pendant gateHoldSeconds ; les rappels LOC sont évalués comme LOC. Le succès pédagogique est >=92 % sur 20 épisodes, donc >=19/20. L'évaluation de la gate Joint conserve le contrat strict >=19/20 arrêts sûrs, seul critère de promotion du BEST. Les gates pédagogiques ne promeuvent pas de BEST ; latest est le seul point de passage. Le dernier verdict de validation apparaît sous la forme [FLIGHT_GATE] stage=... successes=... episodes=... passed=true/false.

## Lancement GitHub CI (correctif 12 novembre 2026)

GitHub Actions > **FlightControl3D Curriculum Training** > **Run workflow**, branche contenant le correctif, `stage=progressive`. Le déclencheur par label `flight-curriculum` reste possible sur une PR.

- `additional_updates=10` : **intervalle d'évaluation** (10 updates PPO, puis 20 épisodes indépendants). Ce n'est plus le budget total d'un lancement.
- `max_additional_updates=1000` : budget cumulé maximal de l'exécution CI, partagé entre LOC, Glide et Joint.
- Le workflow répète la gate courante après `passed=false`, sans avancer. Après `passed=true`, il passe automatiquement à la gate suivante. Le critère >=19/20 reste inchangé, sans requalification fictive des BEST.
- Une limite de 280 minutes de pilotage avant chaque bloc protège le job GitHub (`timeout-minutes=350`). Si le budget global ou la durée sont atteints sans achever la cible, le job est indiqué **incomplete** et quitte avec code 2 (pas de faux vert).
- Même si le run est incomplet ou échoue, `if: always()` publie l'artefact `flight-control-training` (30 jours) avec `latest`, `curriculum_stage.txt`, `curriculum_status.txt`, logs et CSV. Reprendre via `resume_run_id` contenant l'identifiant du run, même s'il était rouge pour budget insuffisant.
- Lancement sur une gate explicite (`stage=loc|glide|joint`) : répète cette seule gate jusqu'à validation (ou limite), et enregistre la gate suivante pour une reprise progressive.
- Un état `done` restauré est idempotent : aucun nouvel update ne doit être demandé pour une campagne déjà achevée.

La CI compile toujours FlightControlHeadless en Release, exécute CTest, puis cinq tests de régression Python à simulateur factice. Le vrai entraînement et la convergence des politiques restent **non vérifiés** tant qu'un run CI réel n'est pas inspecté.
