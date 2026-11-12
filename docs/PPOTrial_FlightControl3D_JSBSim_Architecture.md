# FlightControl3D — pilotage JSBSim et PPO hybride

Date : 11 novembre 2026. Projet : PPOTrial. Issue : #82. PR : [#83](https://github.com/benoitfragit/PPOTrial/pull/83). Branche : `feat/jsbsim-flight-control`.

## Livraison et lancement local

`FlightControl3D.exe` est un exécutable graphique dédié. Il utilise le même `Environment`, les mêmes snapshots ECS et le même renderer raylib/ImGui que l'application actuelle. L'exemple apparaît aussi dans le catalogue de `PPO.exe`. Le contrôleur RLMission, BlueDefense et leurs modèles restent hors du diff.

Après compilation de `FlightControl3D`, deux raccourcis Windows sont créés à la racine pour cette copie du dépôt. Ils ciblent les lanceurs correspondants, avec le bon répertoire de travail :

| Raccourci | Exécutable |
|---|---|
| `FlightControl3D_Debug.lnk` | `build/debug/Main/Debug/FlightControl3D.exe` |
| `FlightControl3D_Release.lnk` | `build/release/Main/Release/FlightControl3D.exe` |

Les fichiers `Lancer_FlightControl3D_Debug.bat` et `Lancer_FlightControl3D_Release.bat` permettent aussi le lancement par double-clic. Ils se placent dans le dépôt, acceptent des arguments et affichent une erreur si l'exécutable n'a pas été compilé. Aucun chemin utilisateur absolu n'est embarqué dans ces lanceurs. La création utilise `tools/windows/CreateFlightShortcuts.ps1` en post-build ; les anciens liens contenant uniquement `RelativePath`, sans cible Windows résolue, sont remplacés par des liens générés et ignorés par Git.

Les lanceurs conservent la sortie et les erreurs dans `logs/FlightControl3D_Debug.log` ou `logs/FlightControl3D_Release.log`. En cas d'échec, le code de sortie et le journal restent affichés. Échap ne ferme plus l'UI Flight3D ; la fin d'un budget conserve la fenêtre ouverte.

Dans un répertoire VS2022 déjà configuré pour la bonne variante de LibTorch :

```powershell
cmake --build build/debug --config Debug --target FlightControl3D
cmake --build build/release --config Release --target FlightControl3D
.\Lancer_FlightControl3D_Release.bat
```

L'option CMake `PPO_BUILD_FLIGHT_CONTROL` vaut ON par défaut pour le desktop. Si un ancien cache l'a désactivée, la réactiver explicitement. CMake 3.18 minimum est requis par JSBSim. Les archives LibTorch Debug et Release doivent continuer à correspondre à la configuration du dépôt.

Le programme démarre en pause ; Play lance l'apprentissage. `--auto-run` démarre immédiatement. `--inference` charge exclusivement le BEST validé et lance une évaluation déterministe à graines distinctes. Sans BEST validé, le lancement d'inférence échoue explicitement. `--config FICHIER` sélectionne une autre configuration complète. `--check-example` vérifie le bootstrap sans fenêtre ni entraînement.

### Reprise par Play et après fermeture

Le JSON livré active `training.resume: true` et `checkpointEvery: 1`. En entraînement, un `latest` existant restaure acteur, critic, moments Adam, âges, RNG et numéro d'update. Son absence permet le premier démarrage ; un fichier présent mais corrompu ou incompatible produit une erreur explicite. Chaque update terminé est sauvegardé. Fermer pendant la collecte abandonne seulement le rollout incomplet, dont aucun update n'a encore été appris ; Stop/Play dans le même processus conserve cette collecte.

Le desktop traite `maxUpdates` comme un budget supplémentaire à partir du modèle chargé. Par exemple, un `latest` à l'update 120 et un budget de 10 000 donnent une fin de bloc à 10 120. Après validation finale, un nouveau Play prolonge le budget en conservant le même learner. Le renderer diffère Stop/join et Play hors du verrou de snapshot, puis le callback ECS `Init` prolonge la session terminée. Le headless conserve son budget total existant. L'UI affiche la source du modèle et la limite du bloc courant ; le journal contient `[FLIGHT_SESSION] Resumed latest` et l'update chargé.

La reprise d'entraînement conserve les poids, Adam et le RNG lors d'un changement des intervalles horizontaux, de l'orientation d'interception, de `initial.approach`, de `maxDistanceMeters`, du guidage compatible déclaré ou des réglages de reward. `rewardVersion: 10`, `guidanceVersion: 13` et `spawnVersion: 2` séparent les contrats de shaping, de guidage et d'ordonnancement initial. Les checkpoints de guidage V2 à V13 et de reward V2 à V10, les placements aléatoires antérieurs et les signatures sans ces versions peuvent amorcer l'entraînement lorsque leur largeur de réseau correspond, mais leur BEST n'est jamais réutilisé sans une nouvelle validation. Les changements de géométrie de piste, de physique ou d'autres observations restent incompatibles. Un BEST antérieur est conservé avec ses générations et son contrôle d'intégrité ; l'inférence exige un BEST validé sur le contrat exact.

## Interface FlightControl3D

Le profil graphique est sélectionné explicitement au lancement dédié ou depuis le catalogue, avant le premier snapshot. Le worker publie un aperçu en pause sans effectuer de pas d'apprentissage. La vue d'ensemble cadre l'avion, la piste, le circuit et les références ILS visibles.

| Panneau | Informations et commandes |
|---|---|
| Vue 3D | Avion, piste, hippodrome, références LOC/GLIDE et grand panneau en haut à droite : phase `HIPP`/`APPROACH`/`ROLLOUT`, vitesse, pente et `GEAR` lorsque le train est physiquement sorti |
| FlightControl3D, à droite | Source active Hippodrome/LOC/GLIDE, visibilité du signal et captures actives, temps simulé courant et durée/cause du dernier épisode ; tableau consigne/mesure, écarts ILS et détails du poser |
| Learning | Compteurs d'épisodes, entraînement PPO, validation, BEST et diagnostics ; poids affichés en lecture seule |
| Simulation | Play/Pause/Stop, deux cases LOC et GLIDE indépendantes, sélection de caméra ; distance de poursuite uniquement pour cette caméra |

Les commandes tactiques sans effet sur cet exemple et les informations dupliquées sont masquées. La pente affichée provient de la vitesse relative à la piste, avec le même calcul que le shaping `Slope` ; ce n'est pas le tangage. Les consignes sont celles du guidage courant, y compris pendant l'arrondi et le roulage.

`Renderer/flight_display.hpp` calcule les références ILS sans dépendance graphique. LOC et GLIDE partent du seuil dans le repère courant de la piste, avec la hauteur de franchissement configurée pour GLIDE. Les enveloppes translucides représentent les tolérances de capture du JSON ; elles restent visibles lorsque le prolongement de la piste inclinée traverse le décor marin. Les cases ne modifient que l'affichage.

## Découpage

| Couche | Responsabilité |
|---|---|
| `Core/flight_config.*` | JSON strict, unités SI, validation des intervalles et paramètres |
| `Core/runway_motion.*` | Tirage des épisodes, transformations du repère piste, houle et cinématique du pont |
| `Core/flight_mission.*` | Flightplan hippodrome, guidage ILS, observation, récompenses et fins d'épisodes |
| `Simulator/jsbsim_aircraft.*` | FDM natif, commandes, conversions SI/ECEF, contact sol mobile |
| `Simulator/flight_environment.*` | Réinitialisation et sous-pas physiques bornés |
| `Training/hybrid_policy.hpp` | Distribution gaussienne × Bernoulli, log-probabilités et dérivées PPO |
| `Training/flight_learner.*` | Réseaux RLtools CPU, GAE, Adam et checkpoints |
| `Runtime/flight_session.*` | Boucle commune au desktop et au headless |
| `Runtime/flight_example.*` | Adaptateur ECS et snapshots ; aucune intégration 2D de l'avion |
| `Renderer/flight_view.*` | Caméra avion/vue d'ensemble, piste et instruments |

JSBSim **1.3.3** est figé par FetchContent au commit `66f7fafc07fe13e7ede9db20873030fed2881713`. Les données nécessaires au Cessna 310 et le fichier COPYING de JSBSim sont copiés dans `build/<configuration>/resources/FlightControl3D/jsbsim`. La bibliothèque déclare LGPL ; l'en-tête du modèle C310 déclare séparément GPL, et ses notices originales sont conservées. Le C310 possède un train rétractable avec une course physique de cinq secondes, contrairement au C172 initialement évoqué. Le backend RLtools reste à la révision déjà utilisée dans PPOTrial et utilise le namespace isolé officiellement prévu par l'amont : ses symboles CPU peuvent ainsi coexister avec ceux de BlueDefense dans le même exécutable.

## Épisodes aléatoires et houle

La configuration unique est `resources/settings/Scenarios/FlightControl3D.json`. Tous les intervalles `[minimum, maximum]` sont tirés à chaque reset à partir d'une graine reproductible. Fixer les deux bornes à la même valeur fige un paramètre pour le diagnostic.

| Paramètre avion | Plage initiale livrée |
|---|---|
| Distance horizontale dans le repère piste | 5 à 14 km |
| Azimut latéral gauche/droite | 12 à 30°, côté équilibré |
| Écart autour du cap d'interception | −3 à +3° |
| Hauteur | Glide exact sur l'axe ; 400 à 1 000 m sur les côtés |
| Vitesse calibrée | 55 à 80 m/s |
| Roulis / tangage / pente | ±15° / −4 à +8° / ±3° |

`initial.runwayAlignedProbability` vaut `1/3` par défaut. Cette famille est placée exactement sur le LOC et le glide, avant le seuil, cap piste et pente de descente configurée. Le complément est partagé à parts égales entre la gauche et la droite ; chaque cap vise le seuil avec seulement le jitter configuré. Un ordonnanceur à déficit choisit la famille la plus en retard sur sa part cible et utilise le RNG uniquement pour départager les égalités. Avec `1/3`, chaque bloc de trois épisodes contient donc exactement une fois les trois familles, dans un ordre pseudo-aléatoire reproductible. Le validateur interdit un azimut ou un jitter qui permettrait de partir dos à la piste ou hors de l'enveloppe de cap de capture. Le repère est celui de la piste réellement tirée, translation, tangage et roulis compris.

Cette part cible finie doit appartenir à [0, 1] : `0` alterne les deux interceptions latérales et `1` produit uniquement le LOC/glide exact. L'ordonnancement persiste pendant la session et borne l'écart entre les proportions demandées et réalisées. Modifier la distribution conserve les poids et Adam comme warm start, mais exige une nouvelle validation du BEST. Les récompenses et les dimensions du réseau restent identiques.

La position de la piste varie de ±2 km sur les deux axes ; son orientation varie de ±35°. La surface peut avancer de 0 à 3 m/s. La houle tire une amplitude verticale de 0 à 0,8 m, une période de 8 à 14 s, des amplitudes de tangage de 0 à 0,25° et de roulis de 0 à 0,4°, avec phases indépendantes. Le pivot de rotation est au centre de la piste. Ces valeurs sont des réglages JSON modifiables ; le schéma refuse les plages incohérentes et les départs sous la mer.

Le callback `FGGroundCallback` fournit réellement à JSBSim la surface finie, sa normale et sa vitesse au point de contact. Le pont bouge dans la physique, le guidage et le rendu. Hors de son rectangle, le contact correspond au plan de la mer. Il s'agit d'une surface rigide cinématique avec mouvements sinusoïdaux ; aucun modèle hydrodynamique de navire n'est revendiqué.

Le scénario livré démarre directement en phase approche (`initial.approach: true`) afin d'apprendre l'interception ILS. L'hippodrome reste le guidage secondaire de récupération : une perte LOC ou GLIDE confirmée avant l'engagement final lui rend la priorité. Une nouvelle capture LOC maintenue pendant `captureHoldSeconds`, dans le secteur avant et avec un cap compatible, redonne ensuite la priorité à l'approche, même sans tour terminé. Terminer le nombre de tours configuré ne change jamais à lui seul la phase ou le point de guidage.

Le flightplan hippodrome est lié à la piste courante. Sa branche entrante est exactement superposée à l'axe LOC et parcourue vers le seuil ; à l'altitude de patrouille livrée, sa partie éloignée traverse aussi la fenêtre de capture du glide. Le suivi d'hippodrome peut donc présenter naturellement l'avion à l'ILS sans contrôleur d'affectation externe. Pendant le ralliement, la position est projetée sur le segment 3D le plus proche et la cible est avancée de `patrolSpeed × patrolLookaheadSeconds` sur le circuit. Après acquisition, les portes séquentielles restent fixes et autoritaires pour la validation du tour. Hors corridor ou après une porte ratée, le ralliement anticipé reprend et toute progression incomplète est invalidée. Une réacquisition ne valide ni porte ni tour fictif ; un avion immobile n'accumule aucun progrès.

Comme BlueAgent, Flight3D sépare désormais un ralliement par projection/anticipation d'un suivi séquentiel physiquement validé. Flight3D conserve son implémentation 3D distincte et `2 × (straightSamples + turnSamples)`, soit 26 portes avec 3 par ligne droite et 10 par demi-tour dans le JSON livré, recalculées avec le mouvement de la piste. Le point anticipé ne remplace pas les portes de FOLLOW. Les composants BlueAgent et RLMission ne sont pas modifiés.

Le signal ILS brut est visible uniquement devant le seuil, à moins de `ilsRangeMeters` (15 km), au-dessus du plan de piste et dans le secteur horizontal `±ilsSectorDegrees` (±35°). Cela ne constitue pas une capture. Le LOC exige en plus une erreur `±localizerCaptureDegrees` (±2,5°), un cap compatible à ±35° et un maintien de 0,5 s. Le glide exige ensuite `±glideCaptureDegrees` (±0,7°) pendant 0,5 s.

Les captures LOC/GLIDE sont actives et réversibles avant l'engagement final. Une sortie au-delà des tolérances multipliées par `captureHysteresis`, ou une perte de signal, doit durer 0,5 s avant libération. Une perte confirmée de l'un des deux axes réactive le ralliement projeté/anticipé vers l'hippodrome, libère les deux captures, remet à zéro le crédit du tour partiel et exige une nouvelle capture LOC puis GLIDE stable. Une fois les deux axes capturés, le point géométrique au seuil, son vecteur et son erreur de direction sont neutralisés : la politique suit directement les erreurs LOC/GLIDE, la pente et la vitesse. Le franchissement du seuil avec les deux captures engage définitivement l'approche jusqu'au toucher, car le signal avant devient naturellement invalide derrière son origine. L'arrondi et le roulage utilisent leurs consignes et la géométrie de piste sans cible au seuil. À faible hauteur après capture glide, la vitesse cible diminue et la pente cible devient nulle pour l'arrondi.

## Observation, actions et PPO

L'observation contient **56 valeurs normalisées dans [−1, 1]**, dont la liste ordonnée est `ObservationNames` : vecteur vers le guidage courant, géométrie piste, hauteur, vitesse calibrée, vecteur vitesse, attitude et vitesses angulaires, pente, erreurs LOC/glide et validité du signal, consignes, train réel, contact roues, phase, temps restant implicite, événements mémorisés, progression des tours et maintiens de capture, état cinématique de la surface. Les échelles sont explicites dans le JSON. Une observation terminale issue d'un FDM non fini est neutralisée ; la transition devient un crash.

L'acteur et le critic reçoivent le même vecteur ; le correctif de shaping ne change pas ses dimensions.

| Groupe | Nombre | Contenu |
|---|---:|---|
| Guidage | 6 | Cible devant/droite/haut, sinus/cosinus de l'erreur de direction, écart à la trajectoire |
| Piste | 4 | Position longitudinale, écart latéral, distance au seuil, hauteur |
| Vol | 15 | Vitesse air, vitesse N/E/U, roulis/tangage/cap relatif en sinus/cosinus, vitesses angulaires, pente en sinus/cosinus |
| ILS et consignes | 7 | Erreurs LOC/glide, validité, vitesse et pente cibles, durées de maintien LOC/glide |
| Atterrissage | 6 | Train réel, contact roues, toucher mémorisé, distance du toucher, piste restante, maintien à l'arrêt |
| Mission | 9 | Trois phases, temps écoulé, deux captures actives, progression des portes, tours terminés, qualification du tour |
| Piste mobile | 9 | Vitesse N/E/U, normale du pont, vitesses de rotation |

L'erreur LOC reste observable dans tout le volume avant non terminal, même au-delà de la portée radio. Elle est encodée par `sin(erreur_LOC)` : son signe distingue les deux côtés et sa magnitude reste strictement croissante de 0° à 90°, sans plateau à ±35°. L'erreur GLIDE reste observable dans la portée ILS avant ; toutes deux peuvent donc être présentes hors secteur alors que le flag `ilsValid` vaut zéro. Derrière le seuil, le LOC vaut zéro ; hors de la portée ILS, le GLIDE vaut zéro. Le vecteur géométrique reste disponible pendant l'hippodrome et l'acquisition. Pendant le suivi LOC+GLIDE capturé, ses trois composantes sont nulles et l'erreur de direction est neutre afin de ne pas envoyer le point du seuil à la politique ; les coordonnées de la piste restent disponibles.

L'acteur produit cinq moyennes, cinq log-écarts-types et un logit Bernoulli. Les commandes continues sont profondeur, ailerons, direction, gaz symétriques et freinage symétrique. Les échantillons gaussiens latents sont transformés par `tanh` ; gaz et freinage sont ensuite convertis vers [0, 1]. Le train est un tirage binaire réel, pas une gaussienne seuillée.

PPO conserve l'échantillon latent et sa log-probabilité jointe avec le Bernoulli. La transformation étant fixe, le Jacobien s'annule dans le ratio PPO ; les calculs utilisent donc la densité latente. La régularisation d'entropie est explicitement celle de la gaussienne latente × Bernoulli. Les dérivées des moyennes, log-écarts-types et du logit incluent le clipping et l'entropie.

Acteur et critic utilisent deux couches cachées tanh, de largeur sélectionnable 32, 64 ou 128 ; la configuration livrée emploie 128. Les dimensions de l'observation, des sorties et la largeur sont des contrats structurels : un ancien checkpoint de largeur 64 ne peut pas être chargé dans le réseau 128. Les matrices et buffers RLtools sont fixes ; le stockage du rollout est alloué au démarrage. Les hyperparamètres PPO/Adam, tailles de rollout/minibatch et budgets sont configurables.

Un crash ou un arrêt réussi est terminal. La limite de distance livrée vaut 18 km depuis le seuil, soit 3 km au-delà de la portée radio ILS : le LOC de shaping reste visible dans cette zone, puis franchir la limite provoque un crash pénalisé et déclenche immédiatement l'épisode suivant, sans bootstrap. La limite de temps est une troncature : la valeur suivante est conservée pour le bootstrap, mais le GAE ne franchit jamais le reset. Le clipping de norme et le contrôle de finitude arrêtent l'utilisation d'un optimiseur défaillant.

## Récompenses, toucher et arrêt

Les dix canaux propres à FlightControl3D sont `Path`, `Localizer`, `Glide`, `Speed`, `Slope`, `EnterLocalizer`, `EnterGlide`, `Touchdown`, `Stop`, `Crash`. Aucune modification des canaux Blue MAPPO n'est effectuée.

GLIDE utilise la qualité lisse rationnelle `Q(e) = tolérance / (tolérance + abs(e))`. LOC emploie une qualité composite bornée. Sa moitié géométrique vaut `1 − clamp(abs(erreur_LOC) / 90°, 0, 1)` : son gradient reste constant sur tout le demi-plan avant et ne s'effondre donc plus entre les départs à 12° et 30°, ni hors du secteur radio. Sa moitié trajectoire compare le cap sol relatif à la piste avec un cap d'interception `−clamp(2 × erreur_LOC, −60°, +60°)`. Ce cap est plus intérieur que la ligne directe vers le seuil et récompense le début du bon virage avant une variation latérale importante. `localizerCourseBlend=0.5`, `localizerInterceptGain=2` et `localizerInterceptLimitDegrees=60` rendent ces choix explicites dans le JSON.

La contribution générique reste `poids * [dt * Q_après * progrès + convergenceSeconds * (Q_après − Q_avant)]`. Pour LOC seulement, `localizerSustainScale=0.25` multiplie le premier terme et `localizerConvergenceScale=5` multiplie le second. La qualité composite reste dans `[0,1]` : ce changement ne relève pas la borne maximale du canal. Le potentiel correctif exige que les deux états successifs appartiennent déjà à l'enveloppe avant ; franchir sa limite géométrique ne crée aucun saut de reward.

La différence signée récompense le rapprochement et pénalise l'éloignement. Elle n'est pas clippée : une amélioration lente suivie d'une régression rapide ne doit pas créer de bonus net. Le facteur de temps vaut 20 secondes dans le JSON. La comparaison avant/après utilise la même phase, le même waypoint et les deux repères successifs de la piste, sans gain artificiel dû au seul mouvement rigide du pont.

Avant le seuil, le shaping LOC et l'observation `localizer_sin` sont denses dans tout le volume non terminal, y compris au-delà de la portée et hors du secteur radio, et agissent dès la phase HIPP. En APPROACH, GLIDE reste dense avant sa capture mais seulement dans la portée ILS ; en HIPP, il attend la capture du LOC. Cette géométrie d'apprentissage ne modifie ni `ilsValid`, ni l'UI, ni les captures : le vrai signal conserve sa portée et son secteur configurés. Le progrès est l'avancement longitudinal relatif à la piste, normalisé par la vitesse cible et borné entre −1 et +1. Un déplacement qui s'éloigne du seuil ne peut rapporter de contribution LOC/glide positive, même si l'erreur angulaire diminue avec la distance. Une géométrie stationnaire ne reçoit pas de maintien répété. Derrière le seuil ou au-delà de la limite terminale, le shaping LOC reste nul. `Path` et son point restent strictement réservés à HIPP ; ils sont nuls et invalides pendant toute la phase APPROACH.

En patrouille, `Path` reprend le shaping borné de BlueAgent : coût rationnel composé à 50 % de l'écart à la courbe, 30 % de l'écart de vitesse et 20 % de l'écart de cap, amélioration signée, progression physique projetée et pénalité de coût courant. Le résultat est borné dans `[-1, 1]` avant le poids existant `0,002`. Stationner, reculer, s'écarter ou tourner autour d'une porte ne rapporte rien de positif. Le tour complet ne rapporte aucun nouveau bonus dédié. Après capture LOC confirmée seulement, les canaux `Speed` et `Slope` utilisent la même qualité rationnelle et une amélioration signée, avec un maintien décroissant ; ils restent disponibles hors tolérance sans pouvoir exploiter un aller-retour fermé. Ils valent exactement zéro avant capture et dès une perte LOC confirmée. Les captures restent payées une seule fois par épisode après maintien confirmé.

| Canal | Poids livré | Amplitude et rôle |
|---|---:|---|
| Localizer | 0,30 | Maintien maximal +0,075 par seconde ; potentiel signé au plus 30 en valeur absolue par transition |
| Glide | 0,25 | Maintien maximal +0,25 par seconde ; potentiel signé au plus 5 par transition |
| Path | 0,002 | 150 fois moins pondéré que LOC ; convergence élémentaire au plus 0,04 |
| Speed / Slope | 0,01 chacun | Guidages secondaires actifs uniquement sous capture LOC confirmée, chacun 30 fois moins pondéré que LOC |
| EnterLocalizer / EnterGlide | +2 chacun | Événement unique confirmé |
| Touchdown / Stop | +100 chacun | Poser sûr puis arrêt confirmé |
| Crash | −100 | Échec terminal |

Dans le volume avant, la borne non actualisée du maintien et du potentiel LOC/GLIDE vaut environ 80,503 avec la décroissance livrée. Après capture LOC seulement, vitesse et pente ajoutent au plus 3,2, puis les captures 4, soit environ 87,7 avant le poser. L'extension de portée du LOC ne change pas cette borne car sa qualité reste dans `[0,1]`. Les récompenses de réussite restent donc dominantes dans cette comparaison ; cela ne garantit pas la convergence d'une politique PPO.

Après chaque rollout optimisé, `[FLIGHT_UPDATE]` journalise les pertes, l'entropie,
le reward total moyen et la contribution moyenne pondérée de chacun des dix
canaux par action de l'acteur. Ces moyennes portent sur les 2 048 transitions de
l'update, pas sur le seul dernier sous-pas affiché par l'IHM.

Le toucher nécessite un contact physique des roues, un train suffisamment sorti, une position sur la piste et une vitesse, un taux de descente et une attitude acceptables. La première abscisse de toucher provient du point physique de roue JSBSim ; les coordonnées des trois trains sont contrôlées dans le rectangle de piste. La vitesse verticale avant impact est vérifiée pour éviter que le solveur de contact ne masque un poser dur. Les vitesses et attitudes de sécurité sont relatives à la piste mobile. Un contact structurel ou un poser sans train provoque un crash sans récompense de toucher. La télémétrie garde aussi l'impact non valide : position signée, erreur latérale, vitesse verticale/horizontale et position réelle du train.

Le toucher ouvre la phase de roulage et rapporte le bonus demandé. L'épisode se poursuit jusqu'à un arrêt confirmé pendant la durée réglée. Une sortie latérale ou un dépassement du bout de piste reste un échec. La télémétrie distingue distance du premier toucher depuis le seuil, abscisse d'arrêt depuis le seuil et longueur parcourue sur le pont après le toucher. L'arrêt est relatif à la surface, pas à la Terre.

## Headless et checkpoints

Le binaire `FlightControlHeadless` se compile aussi sans desktop, Torch ni rendu :

```powershell
cmake -S . -B build/release -DPPO_BUILD_DESKTOP=OFF -DPPO_BUILD_FLIGHT_CONTROL=ON -DPPO_BUILD_CORE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --config Release
ctest --test-dir build/release -C Release --output-on-failure
```

Utiliser les répertoires `build/debug` et `build/release` correspondant à la configuration souhaitée ; ne pas reconfigurer un cache desktop existant en headless pour un simple lancement graphique.

`FlightControlHeadless --updates 1 --output CHEMIN_TEST` exécute un court smoke d'apprentissage suivi de sa validation. `--evaluate --episodes 20 --output CHEMIN_TEST` recharge uniquement le BEST validé pour l'évaluer. Un modèle presque vierge ne fournit normalement aucun BEST admissible. `--config` sélectionne un autre JSON. L'évaluation n'écrit jamais les checkpoints.

Le checkpoint `flight_hybrid_v1.txt` contient acteur, critic, moments Adam, âges des optimiseurs, RNG et compteur d'updates. Son en-tête vérifie dimensions, largeur et signature sémantique de la configuration. La lecture est transactionnelle et les matrices RLtools sont copiées avec leurs opérations dédiées, jamais par une copie C++ de leurs pointeurs internes. Une sauvegarde valide remplace le fichier en conservant `.previous`. La reprise commence un nouvel épisode ; elle ne prétend pas restaurer l'état du FDM en plein vol.

### Politique BEST et validation

Le lancement standard écrit `resources/model/FlightControl3D/latest/flight_hybrid_v1.txt`. Le chemin JSON est relatif à la racine `resources` déduite du scénario ; `--output` permet de choisir un autre répertoire en headless. Tous les dix updates et à la fin du budget, le candidat est figé et évalué sur un environnement séparé, avec 20 épisodes et la graine 4342, distincte de l'entraînement. Paramètres et fréquence sont dans `evaluation` du JSON. La validation n'ajoute aucun échantillon au rollout et ne modifie ni Adam ni le RNG de l'entraînement ; le dernier état physique d'entraînement reste en attente.

Le fichier `latest/flight_hybrid_v1.txt` est le seul artefact FlightControl3D
commitable sous `resources/model`. Il contient acteur, critic, moments Adam,
âges des optimiseurs, RNG et compteur d'updates ; `.previous`, les temporaires
et les générations BEST restent locaux. Lorsqu'il est présent dans le dépôt,
le test de session CI le charge par le même chemin que le desktop et exige une
reprise propre à la frontière d'un update avec un budget additionnel. Le workflow
FlightControl3D est également déclenché par toute modification de ce checkpoint.
Il expose aussi `workflow_dispatch` : depuis l'onglet GitHub Actions, **Run workflow**
relance manuellement toute la matrice Linux/Windows en Debug/Release.

La réussite exige un poser sûr puis l'arrêt confirmé sur la piste. Par défaut, le seuil est **92 %**, donc au moins **19 succès sur 20**. Un candidat ne devient production qu'après la campagne complète et le franchissement de ce seuil. Le score est le pourcentage de réussites ; à score identique, une abscisse moyenne d'arrêt plus courte départage les candidats. Un candidat identique, moins bon, incomplet ou sous le seuil conserve le BEST précédent.

Chaque génération admise est conservée sous `best/flight_<update>_<checksum>.txt`. Le pointeur transactionnel `best/current.json` contient score, compteurs, graine, contrat, update et checksum d'intégrité FNV-1a64 du fichier. Une incohérence est rejetée ; aucun retour silencieux vers `latest`. La sauvegarde d'entraînement et une évaluation de production ne remplacent jamais le BEST. Reprendre un budget déjà consommé termine sa validation avant de considérer la session finie.

## Vérification et limites

Correctif des départs dans l'axe : builds GCC 14 Debug et Release, 15/15 CTest dans chaque configuration. Les régressions couvrent 3 000 tirages déterministes aux probabilités 0/0,5/1 sur quatre caps de piste avec translation et houle, les départs aléatoires des deux côtés, le cap vers la piste, les paramètres qui restent aléatoires, le refus des probabilités invalides, 32 resets JSBSim réels alignés, la reprise des checkpoints sans ce champ et la revalidation après changement de probabilité. La CI Windows/Linux du nouveau commit doit être vérifiée séparément ; ces tests ne démontrent aucune convergence d'atterrissage.

Les régressions déterministes couvrent géométrie, captures uniques, transitions, crash train/impact/attitude, toucher, arrêt relatif, limites de temps, 1 000 tirages aléatoires, dérivée cinématique de la houle, callback réellement consulté par JSBSim, mouvement des commandes, contact des vraies roues, gradients hybrides par différences finies, mise à jour RLtools, reprise des moments/RNG et intégrité en évaluation.

La révision d'interface ajoute les références ILS sur quatre caps avec inclinaison et mouvement, la pente relative distincte du tangage, le transport des tolérances vers le snapshot et la proximité des 1 000 départs avec le circuit. Un test ImGui sans fenêtre vérifie les deux cases par clics simulés, les valeurs consigne/mesure et l'absence des compteurs d'apprentissage dans les instruments. Localement, les 14 CTest headless passent en Debug et Release ; le test ImGui est aussi exécuté séparément. La compilation complète du desktop reste vérifiée par la CI VS2022 de la PR.

La reprise est couverte par comparaison octet pour octet des checkpoints avant/après prolongation et relance, conservation du collector actif, refus d'un `latest` corrompu, migration des intervalles horizontaux, des rewards et de la phase initiale, et conservation du BEST antérieur. Les modifications de géométrie ou vitesse initiale non autorisées restent refusées. Le test desktop ECS vérifie le vrai worker en aperçu sans physique, un bloc complet, Stop/Play vers le bloc suivant et une nouvelle instance reprenant l'update sauvegardé. Le test Windows des raccourcis recharge les `.lnk` par COM et contrôle cible et répertoire de travail. Le rendu interactif et le double-clic Explorer restent des vérifications distinctes sur la machine cible.

Les régressions spécifiques de shaping vérifient le rapprochement hors des anciennes tolérances, le signe inverse en éloignement, l'absence de gain stationnaire ou sur excursion fermée, l'indépendance de la subdivision physique pour la convergence, la progression vers le seuil et le refus du sens inverse, les amplitudes JSON, le guidage hors secteur, la réacquisition d'une porte manquée sans faux tour et le mouvement de piste. La matrice headless contient désormais 15 CTest.

Correctif LOC/glide du 11 novembre : GCC 14 headless, builds Debug et Release et 15/15 CTest dans chaque configuration. Le smoke CLI réel exécute un update, sauvegarde `latest` et termine 20 épisodes de validation indépendante ; ce candidat presque vierge obtient 0 succès et ne remplace aucun BEST. Les résultats de la CI Windows au commit livré restent à contrôler séparément.

Validation locale au premier état relu : GCC 14, Debug et Release, CTest ; smoke d'un update puis deux épisodes indépendants. L'échec des deux épisodes d'évaluation de ce modèle presque vierge ne démontre aucune convergence. La CI Windows du desktop et les quatre configurations headless Windows/Linux doivent être vérifiées avant de déclarer la livraison validée. Le lancement des raccourcis par l'explorateur Windows doit encore être confirmé sur la machine cible.

Sources amont : [JSBSim 1.3.3](https://github.com/JSBSim-Team/jsbsim/tree/66f7fafc07fe13e7ede9db20873030fed2881713), [modèle C310](https://github.com/JSBSim-Team/jsbsim/blob/66f7fafc07fe13e7ede9db20873030fed2881713/aircraft/c310/c310.xml), [contrat FGGroundCallback](https://github.com/JSBSim-Team/jsbsim/blob/66f7fafc07fe13e7ede9db20873030fed2881713/src/input_output/FGGroundCallback.h).

## Renforcement du shaping LOC / glide (11 novembre 2026)

Le profil de reward JSBSim augmente le feedback ILS sans changer la physique, l'actor, le critic, les observations, les actions hybrides, ni les composants BlueAgent/RLMission.

- Poids `Localizer=0.25` et `Glide=0.25`, contre 0.05 : pour une progression parfaite sur les deux axes, **+0.05 par action de 100 ms** au début au lieu de +0.01.
- `convergenceSeconds=20` contre 10 : la contribution de réduction signée d'erreur est initialement dix fois plus intense (poids x5, durée x2). Une régression reste pénalisée; aucun reward positif pour un éloignement du seuil.
- `ilsDecaySeconds=140` : facteur `exp(-elapsed/140)` appliqué **uniquement au terme de progression vers le seuil**. La différence signée de qualité LOC/glide n'est pas atténuée : un retour tardif à l'erreur initiale annule exactement le gain de convergence initial. Paramètre configurable dans le JSON.
- La borne théorique pour une progression continue **dans le secteur ILS** est d'environ 80.005 sur la durée d'épisode (maintien ILS et potentiel télescopique) ; vitesse + pente au plus 3.6, captures 4, soit moins de 88 avant le bonus de poser +100. Cette borne ne couvre pas les phases d'interception géométrique hors ILS et ne garantit pas la convergence PPO.
- `Path/Speed/Slope=0.002` restent secondaires; capture LOC et glide +2, toucher +100, arrêt +100, crash -100 inchangés. Le mode hippodrome et le point de guidage visuel ne sont pas modifiés dans ce correctif.
- `rewardVersion=3` distingue la nouvelle sémantique pour BEST; les modèles latest de version précédente restent compatibles pour amorcer l'entraînement, mais BEST exige une nouvelle validation 20 épisodes (92 %).
- Tests de régression : amplitude initiale, décroissance du progrès, sorties de secteur, trajectoire stationnaire, aller-retour sans gain net même lorsque le retour survient après la constante de décroissance, et validation JSON des paramètres invalides.

**Validation :** revue du diff et vérifications numériques des bornes ; compilation C++ et CI à confirmer avant fusion. Le comportement appris doit être évalué séparément.

## Départs symétriques et priorité LOC (11 novembre 2026)

Le contrat `guidanceVersion=2` introduit les trois familles de départ équilibrées,
le ralliement anticipé vers l'hippodrome, la priorité irréversible donnée à une
capture LOC compatible avec le cap et la capture GLIDE postérieure. La source du
point de guidage est transportée dans le snapshot uniquement pour le rendu ; les
56 observations et les dix canaux de reward conservent leurs dimensions. Les
tests déterministes couvrent les côtés gauche/droit, le départ LOC/glide, le refus
d'une capture perpendiculaire, le maintien de l'hippodrome après un tour sans
capture, la priorité LOC et les durées affichées.

Validation locale demandée pour ce correctif : compilation de la seule cible
`FlightControl3D` en Release dans `build/release`, réussie. Aucun CTest, test
unitaire, build Debug ou campagne de convergence n'a été exécuté, conformément à
la consigne explicite ; ces vérifications restent donc non validées.

## Capture LOC strictement autoritaire (11 novembre 2026)

Le contrat `guidanceVersion=3` supprime la transition historique déclenchée par
la seule combinaison d'un tour terminé et d'une large fenêtre d'approche. La
phase, le point de guidage et le shaping restent ceux de l'hippodrome tant que le
LOC n'a pas satisfait sa géométrie, son cap compatible et son maintien de
capture. Après cette confirmation seulement, l'approche devient irréversible et
le maintien GLIDE peut commencer. Les checkpoints V2 restent utilisables comme
warm start, mais leur BEST doit être revalidé sur ce nouveau contrat.

Validation locale : compilation Release réussie de `FlightControl3D` et des
exécutables `FlightControl_core_tests`, `FlightControl_reward_tests` et
`FlightControl_session_tests`. Aucun de ces tests n'a été exécuté et aucun build
Debug n'a été lancé, conformément à la consigne.

## Initialisation visuellement diversifiée (11 novembre 2026)

Le contrat `spawnVersion=2` remplace les tirages indépendants des trois familles
par un ordonnanceur fixe en mémoire, équilibré sur chaque préfixe de la suite.
La configuration livrée étend aussi la distance à 5–14 km et l'azimut latéral à
12–30°. Les départs restent devant la piste, dans les 15 km et 35° du secteur
ILS, et les caps latéraux visent toujours le seuil. Cette évolution évite les
séries visuellement répétitives sans abandonner la reproductibilité par graine.

## Suivi ILS sans basculement parasite (11 novembre 2026)

Le contrat `guidanceVersion=4` sépare visibilité brute, capture active et bonus
d'entrée. Une perte persistante libère LOC et GLIDE avec la même durée et la
même hystérésis que l'acquisition ; un simple franchissement instantané de la
limite ne fait donc pas bagotter le guidage. La réinterception utilise un point
continu sur le LOC/glide et ne repasse en suivi qu'après LOC puis GLIDE stables.
En suivi, le point du seuil est absent du rendu et neutralisé dans l'observation.
La branche entrante de l'hippodrome est placée sur l'axe LOC et orientée vers le
seuil afin que le guidage secondaire fournisse une présentation ILS cohérente.
Les bonus `EnterLocalizer` et `EnterGlide` restent uniques sur l'épisode, même
après une recapture. Les checkpoints V2/V3 peuvent amorcer l'entraînement, mais
un BEST doit être revalidé sur V4.

Validation locale de cette évolution : non exécutée à la demande de l'utilisateur ;
le build Release et les tests restent à vérifier séparément.

## Suppression définitive de la cible seuil (11 novembre 2026)

Le contrat `guidanceVersion=5` neutralise aussi la cible géométrique pendant
`Flare` et `Rollout`. Sa valeur défensive est la position courante de l'avion et
`targetValid` reste faux : aucun consommateur ne peut donc tracer une ligne vers
le seuil, même s'il oublie de vérifier le drapeau. Après l'hippodrome, la seule
ligne autorisée est désormais celle de réinterception LOC/GLIDE.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

## Priorité LOC puis GLIDE sans point concurrent V6/V10 (11 novembre 2026)

`rewardVersion=6` conserve le signal LOC dense de V5 dans toute la portée avant,
mais conditionne désormais le signal GLIDE dense à `capturedLocalizer`. Avant la
capture LOC, GLIDE vaut donc exactement zéro. Dès le passage en APPROACH, le canal
`Path` vaut également zéro : ni reward hippodrome ni reward de point d'interception
ne concurrence les deux faisceaux.

`guidanceVersion=10` invalide tout point géométrique pendant APPROACH, y compris
entre les captures LOC et GLIDE. Après une perte confirmée, la chaîne perdue est
libérée mais APPROACH reste seul actif pendant une grâce supplémentaire égale à
`captureHoldSeconds`. Une perte LOC exige sa recapture ; une perte GLIDE conserve
le LOC et exige la recapture GLIDE. Sans cette recapture, le ralliement hippodrome
redevient actif à l'expiration. Les 56 observations,
les dimensions du réseau et le format du checkpoint ne changent pas. Les checkpoints
V9/reward V5 conservent donc leurs poids, critic, moments Adam, âges et RNG comme
warm start ; seul le BEST doit être revalidé sur le nouveau contrat.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

## Guidage hippodrome sans point périmé (11 novembre 2026)

Le contrat `guidanceVersion=8` sépare désormais complètement le point de contrôle
et les portes de validation du tour. En phase HIPP, le point présenté à la
politique et au rendu est toujours obtenu par projection sur le segment le plus
proche, puis anticipation de `patrolSpeed × patrolLookaheadSeconds` dans le sens
du circuit. C'est le principe de projection/preview employé par le guidage de
ralliement BlueAgent.

Les portes ordonnées restent autoritaires pour refuser raccourcis, marche arrière
et tours incomplets, mais elles ne deviennent plus des cibles de pilotage fixes.
Un passage manqué ne peut donc plus maintenir la ligne de guidage sur un point de
100 m déjà dépassé et provoquer une orbite autour de celui-ci. Les checkpoints
de guidage V2 à V7 restent utilisables comme warm start ; le BEST doit être
revalidé sur V8.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

Cette tentative V8 a été remplacée par V9 : conserver la projection mobile en
FOLLOW ne reproduisait pas le séquenceur BlueAgent et pouvait faire sauter la
cible entre des segments voisins.

## Séquenceur hippodrome RALLY/FOLLOW V9 (11 novembre 2026)

V9 reprend les invariants du séquenceur BlueAgent. RALLY projette l'avion sur le
circuit et place un point anticipé. Deux échantillons consécutifs dans le corridor
confirment FOLLOW. FOLLOW vise alors une porte fixe parmi 26, dans l'ordre ; le
passage physique avance d'une seule porte et une porte manquée invalide le tour
partiel avant de revenir en RALLY. Les portes utilisent un rayon de 200 m avec
trois échantillons par droite et dix par demi-tour.

Le canal existant `Path` passe à `rewardVersion=4` et utilise également le shaping borné de BlueAgent : coût
position/vitesse/cap `50/30/20`, amélioration signée, progression projetée et
pénalité du coût courant. Aucun canal, bonus ni poids supplémentaire n'est créé.
Le checkpoint V8 reste un warm start compatible ; le BEST doit être revalidé.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

## Shaping ILS dense avant capture V5 (11 novembre 2026)

Le contrat `rewardVersion=5` calcule les potentiels LOC et GLIDE dès HIPP sur
toute la géométrie située avant le seuil, au-dessus du plan de piste et dans la
portée ILS. La qualité rationnelle reste strictement positive loin des faisceaux,
donc une amélioration produit un gradient avant la capture et même hors du
secteur radio. Le recul ne rapporte jamais de contribution positive et une
géométrie stationnaire ne reçoit aucun maintien répété.

Ce shaping privilégié n'élargit pas le signal physique : `ilsValid`, les
observations radio, l'affichage, les captures LOC/GLIDE et leurs bonus conservent
le secteur et les seuils configurés. Aucun canal ni poids supplémentaire n'est
créé.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

## Tolérance ILS temporaire d'entraînement (11 novembre 2026)

`training.ilsCaptureScale` multiplie uniquement les tolérances de capture LOC et
GLIDE de l'environnement qui collecte les rollouts. La valeur locale `2.0` donne
donc `LOC ±5°` et `GLIDE ±1,4°` pendant l'entraînement. Les valeurs nominales
`±2,5°` et `±0,7°` restent utilisées sans modification par les campagnes de
validation et par l'inférence. Le rendu affiche l'enveloppe réellement active et
revient automatiquement à l'enveloppe nominale pendant une validation.

Le réglage appartient à la section `training`, exclue de la signature sémantique
du modèle. Il peut être ramené progressivement à `1.0` et le checkpoint `latest`
reprend alors acteur, critic et moments Adam sans repartir de zéro. Le validateur
refuse un facteur inférieur à `1` ou assez grand pour dépasser le secteur LOC ou
la pente nominale du glide. Les rewards, observations et bonus de capture ne sont
pas modifiés.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

## Point d'interception ILS fixe (11 novembre 2026)

Le contrat `guidanceVersion=6` remplace la cible d'acquisition mobile, auparavant
calculée à mi-distance du seuil, par l'unique entrée configurée à
`-switchDistanceMeters`. La ligne ne dérive plus jusqu'à la piste et ne peut plus
sauter vers l'entrée ILS lors d'un survol du seuil. Elle reste fixe pendant toute
l'acquisition et disparaît après les captures LOC puis GLIDE.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

## Retour hippodrome après perte ILS (11 novembre 2026)

Le contrat `guidanceVersion=7` abandonne l'approche après une perte LOC ou GLIDE confirmée
par les 0,5 s d'hystérésis. Le guidage revient alors au ralliement projeté et
anticipé de l'hippodrome, sans créditer le tour partiel. Les deux captures sont
libérées et une nouvelle séquence LOC puis GLIDE redonne la priorité à l'approche. Si LOC et GLIDE sont encore capturés au
franchissement du seuil, l'approche reste engagée jusqu'au toucher afin de ne pas
interpréter la disparition géométrique normale du signal comme une remise de gaz.

La caméra affiche en grand `HIPP`, `APPROACH` ou `ROLLOUT`, la vitesse réelle et
la pente de trajectoire en haut à droite. `GEAR` apparaît uniquement lorsque la
position physique du train atteint `gearDownMinimum`. Les
poids de shaping livrés sont LOC `0,30`, GLIDE `0,25`, Speed/Slope `0,01`
chacun et Path `0,002` ; le tableau principal reflète ces valeurs.

Validation locale : ni compilation ni test exécuté, conformément à la consigne.

## Renforcement du potentiel correctif LOC V8 (12 novembre 2026)

Le contrat `rewardVersion=8` sépare l'intensité du potentiel correctif LOC de
son maintien récurrent. `localizerConvergenceScale=5` multiplie uniquement la
différence signée de qualité latérale ; `localizerSustainScale=0.25` réduit
uniquement le terme reçu en restant aligné et en progressant. GLIDE, Speed et
Slope conservent leurs formules et leurs échelles. Les checkpoints V2 à V7
restent utilisables comme warm start, mais leur BEST exige une nouvelle
validation sur le contrat complet.

Les tests déterministes couvrent le facteur correctif LOC sans amplification de
GLIDE, le maintien sur l'axe, l'absence de saut à l'entrée de l'enveloppe,
l'annulation d'un aller-retour, la borne cumulée avant le poser et le refus des
échelles non finies ou hors intervalle.

Validation locale : contrôle statique du JSON et du diff uniquement ; compilation
et CTest non exécutés, conformément à la consigne.

## Qualité LOC d'interception et réseau 128 V9 (12 novembre 2026)

Le contrat `rewardVersion=9` remplace la qualité LOC rationnelle par le mélange
borné de l'erreur latérale linéaire et du cap d'interception. Le mélange conserve
exactement la plage `[0,1]` utilisée par la formule V8, donc sa borne cumulative,
mais fournit un gradient plus fort loin du faisceau et un retour dès que la
trajectoire s'oriente vers une vraie interception. GLIDE, Speed, Slope et les
événements terminaux sont inchangés. Les versions V2 à V8 restent acceptées pour
un warm start de même architecture ; le passage livré de 64 à 128 impose en
pratique un nouveau modèle et empêche de reprendre l'ancien critic et son Adam.

Les régressions couvrent le gradient géométrique constant, les trajectoires
intérieure et extérieure à position identique, le comportement réel JSBSim sous
les deux directions d'aileron, les bornes de configuration et les moyennes de
reward par rollout.

Validation locale : contrôle statique du JSON et du diff uniquement ; compilation
et CTest non exécutés, conformément à la consigne.

## Portée LOC et objectifs secondaires V10 (12 novembre 2026)

Le contrat `rewardVersion=10` active `Speed` et `Slope` uniquement sous une
capture LOC confirmée. Les deux canaux sont exactement nuls pendant le
ralliement, avant capture et dès une perte LOC confirmée ; ils ne peuvent plus
être optimisés loin de l'approche. Le contrat `guidanceVersion=12` étend en
parallèle le shaping LOC et l'observation `localizer_error` jusqu'à la limite
terminale du demi-plan avant. La portée radio réelle, `ilsValid`, les transitions
de capture et le GLIDE restent inchangés.

Les régressions déterministes couvrent le LOC au-delà de la portée radio,
l'absence de GLIDE dans cette zone, le silence Speed/Slope avant capture et leur
coupure sur une perte LOC confirmée. Les versions reward V2 à V9 et guidance V2
à V11 restent utilisables comme warm start de même architecture ; un BEST exige
toujours une validation sur le contrat exact.

Validation locale : contrôle statique du JSON et du diff uniquement ; compilation
et CTest non exécutés, conformément à la consigne.

## Observation LOC signée V13 (12 novembre 2026)

Le contrat `guidanceVersion=13` remplace la normalisation LOC saturée à ±35° par
`sin(erreur_LOC)`. Le réseau reçoit une valeur positive du côté `RunwayRight` et
négative du côté opposé. Sa magnitude continue d'augmenter jusqu'à ±90° au lieu
de rester bloquée à ±1 hors du secteur radio. Le champ JSON devenu sans effet
`normalization.localizerDegrees` est supprimé. Le reward demeure symétrique : un
état et son miroir associés à des actions miroir reçoivent la même valeur, sans
introduire de préférence fixe pour un côté.

Les tests couvrent les observations à ±45° et ±60°, ainsi que les trajectoires
intérieure et extérieure depuis chacun des deux côtés. Guidance V2 à V12 reste
acceptée pour un warm start de même architecture ; le BEST doit être revalidé
sur V13.

Validation locale : contrôle statique du diff uniquement ; compilation et CTest
non exécutés, conformément à la consigne.
