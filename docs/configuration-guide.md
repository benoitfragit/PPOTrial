# PPOTrial — Configuration unique V2

La configuration de l'application utilise **un seul contrat** : chaque JSON commence par **"formatVersion": 2**. Le code C++17 de production conserve exclusivement le format JSON V2 ; il accepte cependant les anciens noms de types d'entités et les dossiers historiques de plateformes personnalisées. Les valeurs physiques, les paramètres PPO, les poids des récompenses et les modèles de réseau sont conservés.

## Organisation

~~~text
resources/settings/
├── AgentProfiles.json            profils logiques et modes d'agent
├── Environment.json              monde de simulation principal
├── Renderer.json                 affichage et caméra
├── Training/
│   ├── BluePatrol.json           MAPPO RLTools des deux défenseurs
│   ├── RLMission.json            PPO, récompenses et version du modèle
│   └── RLSNA.json
├── Scenarios/
│   ├── BluePatrol.json           curriculum de patrouille sans rouge
│   ├── Playground.json           composition du monde principal
│   ├── TwoVsTwo.json             défense MAPPO face à RLMission
│   └── Evaluation/
│       ├── RLMission.json        composition et limites des tests
│       └── RLSNA.json
├── Platforms/
│   ├── Mobile/                   paramètres des mobiles
│   └── RestrictedZone/           paramètres des zones fixes
└── Evaluation/
    ├── Cases/
    │   ├── RLMission.json        100 cas d'évaluation
    │   └── RLSNA.json            100 cas d'évaluation
    └── InitialStates.json        2 collections de 100 états initiaux nommés
~~~

Aucun doublon d'ancienne configuration n'est conservé, y compris parmi les fixtures de test.

## Démarrer avec trois fichiers

1. **Scenarios/Playground.json** : qui est dans le monde, quelle équipe, quelle cible et quel agent pilote chaque entité.
2. **AgentProfiles.json** : quel agent C++ est associé à chaque profil logique et à chaque mode.
3. **Training/RLMission.json**, **Training/RLSNA.json** ou
   **Training/BluePatrol.json** : réglages PPO et pondération des récompenses.

L'évaluation est indépendante : sa composition se trouve dans **Scenarios/Evaluation/**, ses scénarios de référence dans **Evaluation/Cases/**.

Les types canoniques sont `Mobile`, `RestrictedZone` et `Objective`.
`"Aircraft"` instancie toujours un `Mobile`, et `"SAM"` une
`RestrictedZone`. Les archétypes `aircraft`, `trainingaircraft`,
`evalaircraft`, `sam` conservent leur nom afin de préserver les références
des scénarios existants. Les nouveaux dossiers sont `Platforms/Mobile/`
et `Platforms/RestrictedZone/`.

## Déclarer les agents sans exposer leurs classes dans les scénarios

La partie pertinente de Playground.json se présente ainsi :

~~~json
{
  "formatVersion": 2,
  "agentProfiles": "../AgentProfiles.json",
  "entities": {
    "blue": {
      "type": "Aircraft",
      "platform": "trainingaircraft",
      "side": "Friend",
      "target": "obj2",
      "agents": [
        {"profile": "RLMission", "mode": "training"}
      ]
    }
  }
}
~~~

Le fichier de profils centralise le lien avec les fabriques C++ :

~~~json
{
  "formatVersion": 2,
  "profiles": {
    "RLMission": {
      "training": "TargetSelectorTrainingAgent",
      "production": "TargetSelectorProdAgent"
    },
    "RLSNA": {
      "training": "SNATrainingAgent",
      "production": "SNAProdAgent"
    }
  }
}
~~~

**profile** désigne un contrat et un comportement applicatif ; **mode** sélectionne son exécution. Les réseaux, les observations et les actions restent définis par les adaptateurs C++ existants ; le Core PPO est indépendant des noms métier. Le même registre de fabriques reste celui de l'application.

Pour ajouter un profil, déclarer le nouvel adaptateur C++ dans la couche applicative, enregistrer ses fabriques, compléter AgentProfiles.json et référencer le profil depuis un scénario. Il n'est pas nécessaire de modifier le cœur PPO.

## Paramètres d'apprentissage

~~~json
{
  "formatVersion": 2,
  "patrol": {
    "racetrack": {
      "lengthMeters": { "min": 50000.0, "max": 60000.0 },
      "widthMeters": { "min": 9000.0, "max": 13000.0 },
      "headingDegrees": { "min": 80.0, "max": 100.0 }
    },
    "formation": {
      "separationDistanceMeters": { "min": 6000.0, "max": 9000.0 },
      "separationAngleDegrees": { "min": 10.0, "max": 25.0 }
    },
    "defense": {
      "activationRisk": 0.55,
      "deactivationRisk": 0.35,
      "minimumAssignmentSeconds": 8.0,
      "recoverySeconds": 15.0
    }
  },
  "training": {
    "rolloutSteps": 2021,
    "checkpointEvery": 5,
    "modelVersion": 125
  },
  "rewards": {
    "clamp": 60,
    "weights": {
      "Main target destruction": 1,
      "Secondary target destruction": 0.15
    }
  },
  "ppo": {
    "gamma": 0.99,
    "gaeLambda": 0.95,
    "clipEpsilon": 0.2,
    "valueCoefficient": 0.5,
    "entropyCoefficient": 0.02,
    "learningRate": 0.0003,
    "epochs": 4,
    "batchSize": 64,
    "targetKL": 0.025999999
  }
}
~~~

Cet exemple est **abrégé** : le fichier RLMission réel contient 12 récompenses nommées, RLSNA en contient 8. Aucune valeur opérationnelle n'a été supprimée.

Le convertisseur typé de **Serialization/training_settings.cpp** reconstruit les coefficients dans l'ordre imposé par **Model::REWARDS**, quelle que soit la présentation des clés JSON. Une récompense absente ou inconnue est une erreur, pas un coefficient implicite.

**checkpointEvery** mesure les mises à jour PPO entre deux demandes d'évaluation ; **rolloutSteps** mesure les pas d'apprentissage par rollout. Ils ne représentent ni des secondes, ni le nombre de cas d'évaluation. **modelVersion** est un état évoluant avec l'entraînement.

### Patrouille bleue MAPPO/RLTools

`Training/BluePatrol.json` possède un contrat volontairement plus petit que
les profils LibTorch historiques :

~~~json
{
  "formatVersion": 2,
  "training": {
    "rolloutSteps": 2048,
    "maxUpdates": 300,
    "checkpointEvery": 10,
    "seed": 42
  },
  "evaluation": {
    "episodes": 8,
    "maxSteps": 4096,
    "seed": 4242,
    "minimumTrackingFraction": 0.8
  },
  "rewards": {
    "clamp": 100.0,
    "weights": {
      "Patrol tracking": 1.0,
      "Formation safety": 1.0,
      "Control effort": 1.0,
      "Zone protection": 1.0,
      "Interception": 1.0,
      "Mission outcome": 1.0,
      "Timeout": 1.0
    }
  }
}
~~~

Ici, `rolloutSteps` compte les pas ECS de l'équipe. Chaque pas fournit deux
transitions au buffer, une par défenseur. Les valeurs applicatives proposées
sont 512, 1024, 2048 et 4096 ; 128 reste la taille interne d'un mini-batch PPO.
Les modifications faites dans l'IHM s'appliquent à la session en cours. Pour
un nouveau démarrage reproductible, modifier le JSON lorsque l'application est
arrêtée.

Les coefficients s'appliquent à des canaux conditionnés par la mission. Sans
menace crédible, `Patrol tracking` évalue les deux bleus. Une menace visible
mais lointaine ou en éloignement ne suffit pas. Avec une menace crédible,
l'affectation stable choisit un intercepteur et conserve l'autre en réserve ;
avec deux menaces crédibles, les deux bleus reçoivent des cibles distinctes.
Les contributions patrouille/défense suivent le poids continu `q`, y compris
pendant la récupération. `Formation safety` combine le wedge pondéré et une
marge anticollision toujours active. `Timeout` vaut -15 si une menace active
ou encore planifiée reste irrésolue à l'horizon ; une patrouille sans menace
n'est pas punie pour avoir atteint sa durée normale.

`checkpointEvery` lance réellement l'évaluateur intégré après le nombre de
mises à jour indiqué. L'acteur agit alors de façon déterministe pendant
`evaluation.episodes` cas de `evaluation.maxSteps` pas au maximum. En 2v2, ces
cas conservent les rouges RLMission de production et alternent 2v2/2v1, côtés
et phases. Le score et sa progression sont visibles dans l'IHM ; seul un
candidat meilleur remplace le checkpoint de production.

Le dossier `resources/model/BluePatrol/latest/` reçoit deux fichiers après la
première mise à jour : `blue_patrol_mappo_v5.txt` destiné à l'inférence et
`blue_patrol_training_v2.txt` destiné à la reprise. Ce second fichier conserve
le critique, les états Adam, le compteur de mises à jour et le RNG. Les données
du rollout ne sont pas conservées ; le buffer repart vide au redémarrage. Le
dossier `prod/` reste volontairement limité au meilleur acteur évalué et à son
score.

La version v5 doit être entraînée depuis des poids neufs : la dimension de
l'observation reste identique, mais les entrées de patrouille représentent
désormais la projection sur une courbe analytique continue, et non une cible
ou une progression temporelle. Les noms et signatures v5/v2 empêchent le
chargement silencieux des anciens fichiers v4/v1.

### Géométrie aléatoire du 2v2

`Scenarios/TwoVsTwo.json` règle les conditions initiales sans modifier les
agents RLMission. `redSpawn.minimumWorldScale` et `maximumWorldScale` placent
les attaquants au-delà de la limite rectangulaire ; cette limite reste un
repère et ne bloque pas leur mouvement. Les angles, exprimés en degrés, sont
mesurés autour de la direction opposée à l'objectif : 0° est frontal et ±90°
est latéral. `minimumSeparation` est une distance en mètres garantie entre les
deux rouges.

`threatCurriculum` ne modifie que les épisodes d'entraînement 2v2. Les champs
`noThreatProbability` et `singleThreatProbability` définissent respectivement
les cas à zéro et une menace ; la probabilité restante crée deux menaces.
`delayedFirstThreatProbability` applique la plage `minimumFirstDelaySteps` à
`maximumFirstDelaySteps` à la première activation. Pour deux menaces,
`delayedSecondThreatProbability` ajoute la seconde plage après la première.
Une entité non encore activée reste détruite et sa piste est masquée dans
l'attention. Les campagnes d'évaluation désactivent le curriculum stochastique
et conservent les cas immédiats 2v1/2v2.

`restrictedZoneSpawn.enabledProbability` contrôle la présence de la zone. Ses
quatre fractions définissent une boîte de tirage relative aux demi-dimensions
du monde ; les fractions est sont orientées vers le côté défendu. Avec les
valeurs distribuées, la zone est active dans 75 % des épisodes. Une même
graine reproduit les positions et le calendrier d'entraînement ; les campagnes
d'évaluation utilisent leur propre calendrier déterministe.

## Scénarios et cas d'évaluation

Les scènes de **Scenarios/Evaluation/RLMission.json** et **RLSNA.json** utilisent les mêmes déclarations d'entités que Playground, plus une section d'évaluation :

~~~json
"evaluation": {
  "evaluatedEntity": "blue",
  "casesFile": "../../Evaluation/Cases/RLMission.json",
  "nbTests": 100,
  "nbSteps": 1024,
  "timeStep": 0.2,
  "maxNorth": 45000,
  "maxEast": 60000
}
~~~

**casesFile** est un fichier JSON d'états initiaux, pas un checkpoint de réseau. Son format est :

~~~json
{
  "formatVersion": 2,
  "tests": [
    {
      "name": "mission_eval_001",
      "entities": {
        "blue": {
          "north": -16000,
          "east": -28500,
          "heading": 0.255,
          "speed": 310,
          "accpsi": -0.01,
          "acc": -3
        }
      }
    }
  ]
}
~~~

Il s'agit d'un extrait de structure ; dans le vrai fichier, chaque test renseigne toutes les entités nécessaires. **Les 100 tests de chaque campagne sont conservés**, avec les mêmes états initiaux.

Les modèles de réseau restent dans **resources/model/NOM_DE_PROFIL/**, dont les sous-dossiers **prod/** et **eval/**. Aucune migration des fichiers **model.pt** n'est effectuée.

## Plateformes et états initiaux nommés

Les paramètres des plateformes sont séparés des algorithmes :

~~~json
{
  "formatVersion": 2,
  "parameters": {
    "FlightModel": {}
  }
}
~~~

Le profil réel contient l'intégralité des sections nécessaires à la plateforme ; le chemin **Platforms/Mobile/trainingaircraft.json** identifie la plateforme, sans répéter son nom à la racine.

Pour les défenseurs MAPPO, `Platforms/Mobile/defender_car.json` contient aussi
`FlightModel.CruiseSpeed`. Cette vitesse nominale est utilisée au spawn et par
la référence de patrouille ; `MaxSpeed`, `MaxAcceleration` et
`MaxRotAcceleration` restent les limites physiques. La section `SNA` règle la
zone sans échappement et l'angle d'engagement des bleus indépendamment de la
plateforme rouge `attacker_car`.

**Evaluation/InitialStates.json** contient des collections de points initiaux nommés, au format **presets.blueeval[]** et **presets.redeval[]**, avec les coordonnées et commandes dans des champs explicites **north**, **east**, **heading**, **speed**, **turnAcceleration**, **acceleration**. L'initialiseur C++ peut les utiliser comme auparavant, sans liste opaque de six nombres ni ancien chemin. Le champ **evaluationSettings** conserve les paramètres numériques associés à ces jeux d'initialisation.

## Environnement et affichage

Ces réglages ont le même schéma V2 et le même style de nommage que les autres configurations :

~~~json
{
  "formatVersion": 2,
  "environment": {
    "maxNorth": 44658.75,
    "maxEast": 71454,
    "timeStep": 0.2
  }
}
~~~

**Renderer.json** contient un objet **renderer**. L'ancien fichier V1 ne peut
pas être copié tel quel : `schemaVersion: 1` devient `formatVersion: 2`,
`Renderer` devient `renderer`, et les clés sont renommées en lowerCamelCase.

| Ancienne clé V1 (dans `Renderer`) | Clé V2 (dans `renderer`) |
| --- | --- |
| `ScaleLevel`, `AutoAttachCamera` | `scaleLevel`, `autoAttachCamera` |
| `ShowGrid`, `ShowDomains`, `ShowDestroyed`, `ShowMetaCatalog` | `showGrid`, `showDomains`, `showDestroyed`, `showMetaCatalog` |
| `ShowPseudoLidar`, `FocusSelectionInMinimap` | `showPseudoLidar`, `focusSelectionInMinimap` |
| `CameraDistance`, `CameraHeight`, `TrailDuration` | `cameraDistance`, `cameraHeight`, `trailDuration` |
| `DomainHeight`, `MinimapZoom` | `domainHeight`, `minimapZoom` |
| `Camera.Position`, `Camera.Target`, `Camera.Up`, `Camera.Fovy` | `camera.position`, `camera.target`, `camera.up`, `camera.fovy` |
| `ShowMinimap`, `ShowEntities`, `ShowInspection`, `ShowSimulation` (si présents) | `showMinimap`, `showEntities`, `showInspection`, `showSimulation` |

Les quatre derniers paramètres de panneaux ont été ajoutés avec l'interface
d'apprentissage. Lorsqu'ils sont absents de l'ancien fichier, les valeurs par
défaut du code sont `false`, `false`, `false` et `true`, respectivement.
Le fichier V2 distribué les renseigne explicitement. Conserver les valeurs
numériques (notamment celles de la caméra) lors de la conversion ; ne pas
réinitialiser `renderer` aux valeurs d'exemple.

**Fichiers versionnés :** `resources/settings/Renderer.json` est déjà au format V2
dans le dépôt. Les valeurs du fichier historique (dont les coordonnées de caméra
avec leur précision complète) y sont reprises et les quatre paramètres
de panneaux ajoutés sur `master` sont explicitement renseignés. Aucune
conversion ni action supplémentaire n'est nécessaire pour utiliser les
configurations fournies sur cette branche.

Les configurations personnelles qui ne sont pas présentes dans le dépôt ne
peuvent pas être migrées automatiquement. Le runtime lit uniquement V2 ;
les champs inconnus ne sont pas reconnus et doivent être examinés avant tout
remplacement.

## Sauvegarde et erreurs

**Environment.json**, **Renderer.json** et **Training/*.json** sont chargés et sauvés par le Serializer. Chaque fichier doit porter **formatVersion = 2** : un document incorrect, manquant ou impossible à charger est **préservé sans écrasement** par des valeurs par défaut à l'arrêt de l'application. Éviter les modifications manuelles pendant l'exécution.

Les campagnes, les profils et les jeux de tests sont des données d'entrée. Les tailles des tableaux de récompenses, les valeurs de contrôle, les noms de profils et les références d'entités sont contrôlés au chargement par les composants concernés.

## Vérifications de non-régression

- Les scénarios V2 contrôlent le nombre d'entités, les fabriques résolues, la composition des évaluations et le nombre de cas.
- Les paramètres PPO V2 sont comparés à des constantes numériques de référence, notamment la cadence des checkpoints et l'ordre exact des récompenses.
- La sérialisation assure un cycle lecture-écriture-lecture sans modification des paramètres et refuse les poids nommés invalides.
- Les tests de sûreté vérifient qu'une configuration mal formée n'est pas écrasée.
- Les builds et CTest Windows VS2022 Debug/Release, ainsi que les tests de simulation/évaluation et la lecture des modèles existants, doivent être validés avant fusion.

Il n'y a **aucun lecteur V1 ni second format en production**. Une éventuelle conversion ponctuelle doit être exécutée hors application et ne fait pas partie du runtime.
