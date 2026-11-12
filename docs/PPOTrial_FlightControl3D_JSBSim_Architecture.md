# FlightControl3D — pilotage JSBSim et PPO hybride

Date : 11 novembre 2026. Projet : PPOTrial. Issue : #82. Branche : `feat/jsbsim-flight-control`.

## Livraison et lancement local

`FlightControl3D.exe` est un exécutable graphique dédié. Il utilise le même `Environment`, les mêmes snapshots ECS et le même renderer raylib/ImGui que l'application actuelle. L'exemple apparaît aussi dans le catalogue de `PPO.exe`. Le contrôleur RLMission, BlueDefense et leurs modèles restent hors du diff.

À la racine, deux raccourcis relatifs appellent les lanceurs correspondants :

| Raccourci | Exécutable |
|---|---|
| `FlightControl3D_Debug.lnk` | `build/debug/Main/Debug/FlightControl3D.exe` |
| `FlightControl3D_Release.lnk` | `build/release/Main/Release/FlightControl3D.exe` |

Les fichiers `Lancer_FlightControl3D_Debug.bat` et `Lancer_FlightControl3D_Release.bat` permettent aussi le lancement par double-clic. Ils se placent dans le dépôt, acceptent des arguments et affichent une erreur si l'exécutable n'a pas été compilé. Aucun chemin utilisateur absolu n'est embarqué dans ces lanceurs.

Dans un répertoire VS2022 déjà configuré pour la bonne variante de LibTorch :

```powershell
cmake --build build/debug --config Debug --target FlightControl3D
cmake --build build/release --config Release --target FlightControl3D
.\Lancer_FlightControl3D_Release.bat
```

L'option CMake `PPO_BUILD_FLIGHT_CONTROL` vaut ON par défaut pour le desktop. Si un ancien cache l'a désactivée, la réactiver explicitement. CMake 3.18 minimum est requis par JSBSim. Les archives LibTorch Debug et Release doivent continuer à correspondre à la configuration du dépôt.

Le programme démarre en pause ; Play lance l'apprentissage. `--auto-run` démarre immédiatement. `--inference` charge le checkpoint FlightControl3D et lance une évaluation déterministe à graines distinctes. `--config FICHIER` sélectionne une autre configuration complète. `--check-example` vérifie le bootstrap sans fenêtre ni entraînement.

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

JSBSim **1.3.3** est figé par FetchContent au commit `66f7fafc07fe13e7ede9db20873030fed2881713`. Les données nécessaires au Cessna 310 et sa licence sont copiées dans `build/<configuration>/resources/FlightControl3D/jsbsim`. Le C310 possède un train rétractable avec une course physique de cinq secondes, contrairement au C172 initialement évoqué. Le backend RLtools reste à la révision déjà utilisée dans PPOTrial.

## Épisodes aléatoires et houle

La configuration unique est `resources/settings/Scenarios/FlightControl3D.json`. Tous les intervalles `[minimum, maximum]` sont tirés à chaque reset à partir d'une graine reproductible. Fixer les deux bornes à la même valeur fige un paramètre pour le diagnostic.

| Paramètre avion | Plage initiale livrée |
|---|---|
| Distance horizontale dans le repère piste | 7 à 14 km |
| Azimut autour de la piste / décalage de cap | −180 à +180° |
| Hauteur au-dessus du plan courant de la piste | 400 à 1 000 m |
| Vitesse calibrée | 55 à 80 m/s |
| Roulis / tangage / pente | ±15° / −4 à +8° / ±3° |

La position de la piste varie de ±2 km sur les deux axes ; son orientation varie de ±35°. La surface peut avancer de 0 à 3 m/s. La houle tire une amplitude verticale de 0 à 0,8 m, une période de 8 à 14 s, des amplitudes de tangage de 0 à 0,25° et de roulis de 0 à 0,4°, avec phases indépendantes. Le pivot de rotation est au centre de la piste. Ces valeurs sont des réglages JSON modifiables ; le schéma refuse les plages incohérentes et les départs sous la mer.

Le callback `FGGroundCallback` fournit réellement à JSBSim la surface finie, sa normale et sa vitesse au point de contact. Le pont bouge dans la physique, le guidage et le rendu. Hors de son rectangle, le contact correspond au plan de la mer. Il s'agit d'une surface rigide cinématique avec mouvements sinusoïdaux ; aucun modèle hydrodynamique de navire n'est revendiqué.

Le flightplan hippodrome est référencé à la piste courante. Le prochain waypoint est choisi près du départ aléatoire. La transition vers l'approche est irréversible lorsque la distance configurée est franchie du côté approche. Le LOC et le glide ne produisent pas de capture à l'arrière de la piste ou hors du secteur de service. À faible hauteur, la vitesse cible diminue et la pente cible devient nulle pour l'arrondi.

## Observation, actions et PPO

L'observation contient **51 valeurs normalisées dans [−1, 1]**, dont la liste ordonnée est `ObservationNames` : vecteur vers le guidage courant, géométrie piste, hauteur, vitesse calibrée, vecteur vitesse, attitude et vitesses angulaires, pente, erreurs LOC/glide et validité du signal, consignes, train réel, contact roues, phase, temps restant implicite, événements mémorisés et état cinématique de la surface. Les échelles sont explicites dans le JSON. Une observation terminale issue d'un FDM non fini est neutralisée ; la transition devient un crash.

L'acteur produit cinq moyennes, cinq log-écarts-types et un logit Bernoulli. Les commandes continues sont profondeur, ailerons, direction, gaz symétriques et freinage symétrique. Les échantillons gaussiens latents sont transformés par `tanh` ; gaz et freinage sont ensuite convertis vers [0, 1]. Le train est un tirage binaire réel, pas une gaussienne seuillée.

PPO conserve l'échantillon latent et sa log-probabilité jointe avec le Bernoulli. La transformation étant fixe, le Jacobien s'annule dans le ratio PPO ; les calculs utilisent donc la densité latente. La régularisation d'entropie est explicitement celle de la gaussienne latente × Bernoulli. Les dérivées des moyennes, log-écarts-types et du logit incluent le clipping et l'entropie.

Acteur et critic utilisent deux couches cachées tanh, de largeur sélectionnable 32, 64 ou 128. Les dimensions de l'observation et des sorties sont des contrats structurels. Les matrices et buffers RLtools sont fixes ; le stockage du rollout est alloué au démarrage. Les hyperparamètres PPO/Adam, tailles de rollout/minibatch et budgets sont configurables.

Un crash ou un arrêt réussi est terminal. La limite de temps est une troncature : la valeur suivante est conservée pour le bootstrap, mais le GAE ne franchit jamais le reset. Le clipping de norme et le contrôle de finitude arrêtent l'utilisation d'un optimiseur défaillant.

## Récompenses, toucher et arrêt

Les dix canaux propres à FlightControl3D sont `Path`, `Localizer`, `Glide`, `Speed`, `Slope`, `EnterLocalizer`, `EnterGlide`, `Touchdown`, `Stop`, `Crash`. Aucune modification des canaux Blue MAPPO n'est effectuée.

Les shapings sont des qualités triangulaires `max(0, 1 − abs(erreur)/tolérance)`, pondérées par le temps physique écoulé. Les captures sont payées une seule fois par épisode. Les seuils, tolérances et poids sont dans le JSON.

Le toucher nécessite un contact physique des roues, un train suffisamment sorti, une position sur la piste et une vitesse, un taux de descente et une attitude acceptables. La vitesse verticale avant impact est vérifiée pour éviter que le solveur de contact ne masque un poser dur. Les vitesses et attitudes de sécurité sont relatives à la piste mobile. Un contact structurel ou un poser sans train provoque un crash sans récompense de toucher.

Le toucher ouvre la phase de roulage et rapporte le bonus demandé. L'épisode se poursuit jusqu'à un arrêt confirmé pendant la durée réglée. Une sortie latérale ou un dépassement du bout de piste reste un échec. La télémétrie distingue distance du premier toucher depuis le seuil, abscisse d'arrêt depuis le seuil et longueur parcourue sur le pont après le toucher. L'arrêt est relatif à la surface, pas à la Terre.

## Headless et checkpoints

Le binaire `FlightControlHeadless` se compile aussi sans desktop, Torch ni rendu :

```powershell
cmake -S . -B build/release -DPPO_BUILD_DESKTOP=OFF -DPPO_BUILD_FLIGHT_CONTROL=ON -DPPO_BUILD_CORE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --config Release
ctest --test-dir build/release -C Release --output-on-failure
```

Utiliser les répertoires `build/debug` et `build/release` correspondant à la configuration souhaitée ; ne pas reconfigurer un cache desktop existant en headless pour un simple lancement graphique.

`FlightControlHeadless --updates 1 --output CHEMIN_TEST` exécute un court smoke d'apprentissage. `--evaluate --episodes 20 --output CHEMIN_TEST` recharge le checkpoint pour l'évaluer. `--config` sélectionne un autre JSON. L'évaluation n'écrit jamais le checkpoint.

Le checkpoint `flight_hybrid_v1.txt` contient acteur, critic, moments Adam, âges des optimiseurs, RNG et compteur d'updates. Son en-tête vérifie dimensions, largeur et signature sémantique de la configuration. La lecture est transactionnelle et les matrices RLtools sont copiées avec leurs opérations dédiées, jamais par une copie C++ de leurs pointeurs internes. Une sauvegarde valide remplace le fichier en conservant `.previous`. La reprise commence un nouvel épisode ; elle ne prétend pas restaurer l'état du FDM en plein vol.

## Vérification et limites

Les régressions déterministes couvrent géométrie, captures uniques, transitions, crash train/impact/attitude, toucher, arrêt relatif, limites de temps, 1 000 tirages aléatoires, dérivée cinématique de la houle, callback réellement consulté par JSBSim, mouvement des commandes, contact des vraies roues, gradients hybrides par différences finies, mise à jour RLtools, reprise des moments/RNG et intégrité en évaluation.

Validation locale au premier état relu : GCC 14, Debug et Release, CTest ; smoke d'un update puis deux épisodes indépendants. L'échec des deux épisodes d'évaluation de ce modèle presque vierge ne démontre aucune convergence. La CI Windows du desktop et les quatre configurations headless Windows/Linux doivent être vérifiées avant de déclarer la livraison validée. Le lancement des raccourcis par l'explorateur Windows doit encore être confirmé sur la machine cible.

Sources amont : [JSBSim 1.3.3](https://github.com/JSBSim-Team/jsbsim/tree/66f7fafc07fe13e7ede9db20873030fed2881713), [modèle C310](https://github.com/JSBSim-Team/jsbsim/blob/66f7fafc07fe13e7ede9db20873030fed2881713/aircraft/c310/c310.xml), [contrat FGGroundCallback](https://github.com/JSBSim-Team/jsbsim/blob/66f7fafc07fe13e7ede9db20873030fed2881713/src/input_output/FGGroundCallback.h).
