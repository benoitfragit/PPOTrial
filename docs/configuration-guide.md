# PPOTrial — Configuration unique V2

La configuration de l'application utilise **un seul contrat** : chaque JSON commence par **"formatVersion": 2**. Le code C++17 de production conserve exclusivement le format JSON V2 ; il accepte cependant les anciens noms de types d'entités et les dossiers historiques de plateformes personnalisées. Les valeurs physiques, les paramètres PPO, les poids des récompenses et les modèles de réseau sont conservés.

## Organisation

~~~text
Examples/Shared/settings/          couches communes à tous les exemples
├── AgentProfiles.json            profils logiques et modes d'agent
├── Environment.json              monde de simulation principal
└── Platforms/
    ├── Mobile/                   paramètres des mobiles
    └── RestrictedZone/           paramètres des zones fixes

Examples/Airspace/settings/        réglages de l'exemple Airspace (agents rouges)
├── Training/
│   ├── RLMission.json            récompenses et taille du rollout
│   └── RLSNA.json
├── Scenarios/
│   ├── Playground.json           composition du monde principal
│   └── Evaluation/
│       ├── RLMission.json        composition et limites des tests
│       └── RLSNA.json
└── Evaluation/
    ├── Cases/
    │   ├── RLMission.json        100 cas d'évaluation
    │   └── RLSNA.json            100 cas d'évaluation
    └── InitialStates.json        2 collections de 100 états initiaux nommés

Examples/BlueDefense/projects/BluePatrol/   projet PPOKit de la défense MAPPO P/N
├── scenario.json                 composition (P=4, N=4 par défaut)
├── blue_patrol_training.json     géométrie et récompenses de l'environnement
├── training.json                 PPO et rollout du learner partagé
└── checkpoints/                  latest/, prod/, télémétrie
~~~

Le chargeur de scénario Airspace (`LoadExternalScenario`, utilisé par les DLL
d'environnement `rlmission_environment` et `rlsna_environment` des projets
`Examples/Airspace/projects/RLMission` et `RLSNA`) cherche un fichier de
réglage d'abord dans `Examples/Airspace/settings`, puis dans
`Examples/Shared/settings` (`Serializer::SetSearchPaths`).

Les projets historiques et les fixtures de parité conservés pour la CI peuvent
utiliser des configurations distinctes du projet principal.

## Démarrer avec trois fichiers

1. **Scenarios/Playground.json** : qui est dans le monde, quelle équipe, quelle cible et quel agent pilote chaque entité.
2. **AgentProfiles.json** : quel agent C++ est associé à chaque profil logique et à chaque mode.
3. **Training/RLMission.json**, **Training/RLSNA.json** ou
   **blue_patrol_training.json** du projet BluePatrol : pondération des récompenses
   et géométrie. Les hyperparamètres PPO et le rollout sont dans le `training.json`
   de chaque projet PPOKit, notamment BluePatrol, RLMission et RLSNA.

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

`Examples/Airspace/settings/Training/RLMission.json` (et `RLSNA.json`) :

~~~json
{
  "formatVersion": 2,
  "training": {
    "rolloutSteps": 2021
  },
  "rewards": {
    "clamp": 60,
    "weights": {
      "Main target destruction": 1,
      "Secondary target destruction": 0.15
    }
  }
}
~~~

Cet exemple est **abrégé** : le fichier RLMission réel contient 12 récompenses nommées, RLSNA en contient 8. Aucune valeur opérationnelle n'a été supprimée.

Le convertisseur typé de **Serialization/training_settings.cpp** reconstruit les coefficients dans l'ordre imposé par **Model::REWARDS**, quelle que soit la présentation des clés JSON. Une récompense absente ou inconnue est une erreur, pas un coefficient implicite.

**rolloutSteps** mesure les pas d'apprentissage par rollout ; sous `ppokit-run`, c'est aussi la limite de pas d'un épisode, qui tronque l'épisode (l'échec de mission du dernier pas, `IsLastStep()`, n'est pas payé). Ce n'est ni une durée en secondes, ni un nombre de cas d'évaluation. Les épisodes d'entraînement de RLMission (scénario `Playground.json`) et de RLSNA (scénario `RLSNATraining.json` : l'intercepteur en mode entraînement et sa cible) sont tirés de la graine que `ppokit-run` donne à l'environnement (`seed` de `training.json` du projet, plus 1 pour `train`) : un entraînement est reproductible. Le `TrainingAgent` ne lit de ce fichier que `rolloutSteps`, le clamp et les poids des récompenses : il n'a ni learner, ni sauvegarde de checkpoint, ni évaluateur. Les hyperparamètres PPO (section `ppo`) sont dans `Examples/Airspace/projects/RLMission/training.json` (et `RLSNA`), et la fréquence des évaluations pendant l'entraînement dans la suite du projet (`eval/campaign.json`, `during_training.every_updates` : 5 pour RLMission et RLSNA), jouée par `ppokit-run`.

### Patrouille bleue MAPPO/RLTools

`blue_patrol_training.json` du projet BluePatrol
(`Examples/BlueDefense/projects/BluePatrol/`) décrit aussi la géométrie de la
patrouille (`patrol`) ; extrait :

~~~json
{
  "formatVersion": 2,
  "patrol": {
    "racetrack": {
      "lengthMeters": { "min": 40000.0, "max": 45000.0 },
      "widthMeters": { "min": 16000.0, "max": 19000.0 },
      "headingDegrees": { "min": 80.0, "max": 100.0 }
    },
    "formation": {
      "separationDistanceMeters": { "min": 3000.0, "max": 3500.0 },
      "separationAngleDegrees": { "min": 30.0, "max": 35.0 }
    },
    "defense": {
      "activationRisk": 0.25,
      "deactivationRisk": 0.10,
      "activationDistanceMeters": 90000.0,
      "deactivationDistanceMeters": 110000.0,
      "protectedRadiusMeters": 10000.0,
      "timeToIntrusionSeconds": 300.0,
      "teammateThreatDistanceMeters": 30000.0,
      "teammateVulnerabilityWeight": 0.75
    }
  },
  "training": {
    "rolloutSteps": 2048,
    "maxUpdates": 4000,
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
    "noThreatPatrol": {
      "rallyEnterDistanceMeters": 2500.0,
      "rallyExitDistanceMeters": 3000.0,
      "maxEpisodeSeconds": 480.0
    },
    "weights": {
      "Patrol tracking": 1.0,
      "Formation safety": 4.0,
      "Control effort": 1.0,
      "Zone protection": 2.0,
      "Interception": 2.0,
      "Mission outcome": 1.5,
      "Timeout": 1.5
    }
  }
}
~~~

Ici, `rolloutSteps` compte les pas ECS synchronisés de l'équipe :
chaque pas produit P transitions, une par Bleu initialement présent.
Les valeurs usuelles sont 512, 1024, 2048 et 4096.
Les paramètres sont chargés depuis `blue_patrol_training.json` ; l'extrait omet
les sections `rewards.generalized` et `rewards.shaping` et une partie de
`patrol.racetrack` et de `rewards.noThreatPatrol`.
`rallyEnterDistanceMeters` et `rallyExitDistanceMeters` définissent le couloir
à hystérésis acquis avant de figer les dix waypoints. Un premier échantillon sous
le seuil d'entrée arme l'acquisition ; l'observation suivante la confirme si
l'avion reste sous le seuil de sortie, y compris dans la bande entre les seuils.
Une sortie du couloir avant confirmation annule l'acquisition. La transition
vers FOLLOW est ensuite irréversible pour l'épisode.
`maxEpisodeSeconds` ne s'applique qu'à N=0 et
est converti en pas depuis `dt` : 480 s à 0,2 s donnent 2400 pas. L'expiration
est une troncature, pas une terminaison réussie ni une destruction.

Le MAPPO actuel utilise des canaux séparés pour la progression sur
l'hippodrome (leader), le maintien du wedge (équipiers), la poursuite des
Rouges, les pertes Bleues/Rouges, la protection de l'objectif, le coût de
commande et l'issue de mission. Les sept poids nommés du JSON sont
convertis explicitement en quinze canaux de reward dans l'exécution.
Une formation correctement tenue ne génère pas un bonus positif
permanent : l'écart hors tolérance est pénalisé. Les 61 rayons LiDAR
restent disponibles, y compris pour la prévention Bleu/Bleu.

L'acteur ne reçoit aucune attribution externe de cible rouge.
Le signal de poursuite tient les positions des Rouges fixes lors du
calcul du progrès causé par le mouvement Bleu ; les menaces
influencent continûment la priorité patrouille/formation/défense.
Cela n'établit pas, à lui seul, la convergence de la politique.

`checkpointEvery` donne le rythme de la campagne d'évaluation déterministe : la suite
`eval/validation.json` du projet, jouée par `ppokit-run` au même rythme
(`during_training.every_updates`).
Les checkpoints courants sont `policy.ppokit` pour l'acteur
et `latest.ppokit` pour l'état complet (artefacts PPOKit), sous
`Examples/BlueDefense/projects/BluePatrol/checkpoints/latest/`. Les poids V9 ne sont pas repris : le
guidage figé modifie la sémantique de l'observation de patrouille.

`ppokit-run train` et `ppokit-run play` sur le projet BluePatrol utilisent
exactement la scène `scenario.json` du projet : `train` entraîne le MAPPO
généralisé, tandis que `play --policy checkpoints/prod/policy.ppokit` exécute
l'acteur de production de façon déterministe. Ce second mode ne possède ni
rollout, ni optimiseur, ni campagne d'évaluation périodique. La progression
et les métriques vont au journal du run (`checkpoints/latest/metrics.jsonl`,
suivi en direct par ppokit-studio) ; le monde et les quinze rewards du pas
sont envoyés à ppokit-viewer par le Visualiseur `blue_visualiser`. Les tests
de simulation et la CI Desktop doivent être validés avant fusion.

### Géométrie aléatoire P/N

`scenario.json` du projet BluePatrol règle les conditions initiales sans modifier les
agents RLMission. `redSpawn.minimumWorldScale` et `maximumWorldScale` placent
les attaquants au-delà de la limite rectangulaire ; cette limite reste un
repère et ne bloque pas leur mouvement. Les angles, exprimés en degrés, sont
mesurés autour de la direction opposée à l'objectif : 0° est frontal et ±90°
est latéral. `minimumSeparation` est une distance en mètres garantie entre
toutes les paires de rouges.

Le contrat d'effectif `blueTeam`/`redTeam` est obligatoire. Le
`threatCurriculum` historique et les probabilités spécialisées
sur exactement deux attaquants sont rejetés. Les activations
retardées sont désormais définies par les champs génériques
`redTeam.delayedActivationProbability`,
`minimumActivationDelaySteps` et `maximumActivationDelaySteps`.
Une piste non encore activée demeure masquée jusqu'à son
activation effective.

Le bloc historique `restrictedZoneSpawn` continue à positionner
`blue_restricted_zone` avec les mêmes fractions et le même ordre de tirage RNG.
Cette zone reste consommée par RLMission et n'est pas incluse dans la nouvelle
collection.

`restrictedZones` est une collection dédiée, indépendante de `entities`. Son
booléen `enabled` désactive toutes les zones. Chaque élément fournit un `id`
unique, `type="RestrictedZone"`, un `platform`, `side="Hostile"`, `enabled`,
une `enabledProbability` dans `[0,1]` et exactement un mode de position :
`fixedPosition` ou `randomPosition`. Toutes les coordonnées sont en mètres.
`oppositeObjective=true` interprète la coordonnée est comme une magnitude et
la reflète automatiquement du côté opposé à l'objectif à chaque reset.
Le mode aléatoire exige deux intervalles nord/est ordonnés et finis. Un champ
inconnu dans ce contrat, un doublon, un mode ambigu ou une valeur invalide fait
échouer explicitement le chargement. Une même graine reproduit positions et
activations.

Le profil de plateforme reste la source du rayon létal
`SNA.MaxNoEscapeZoneRange`. Le rayon LiDAR vaut 110 % de ce rayon. Dans
BlueDefense, seul le capteur ECS des `Mobile/Friend` est rafraîchi par ces règles;
le contrôleur, l'observation 152 et les checkpoints RLMission rouges ne sont
pas modifiés. La règle SNA probabiliste de la zone bleue historique reste elle
aussi active contre les rouges ; seule l'arme des nouvelles zones hostiles est
remplacée par la pénétration déterministe exacte pour les bleus.

### Équipes P/N — contrôleur MAPPO unique

`ppokit-run train` et `ppokit-run play` chargent tous deux le `scenario.json`
du projet BluePatrol, avec P=4/N=4 par défaut. Dans les deux cas,
`GeneralizedPatrolMappoAgent` pilote les Bleus dans le monde depuis la DLL
`blue_environment` ; le premier entraîne le learner MAPPO de `blue_network`,
le second utilise uniquement son acteur de production, sans capacité
d'apprentissage. Les effectifs sont
déclarés sous `blueDefense` :

~~~json
{
  "blueTeam": {"activeCount": 5},
  "redTeam": {
    "activeCount": 7,
    "maximumPlacementAttempts": 512,
    "delayedActivationProbability": 0.25,
    "minimumActivationDelaySteps": 256,
    "maximumActivationDelaySteps": 768
  }
}
~~~

Les blocs `blueTeam` et `redTeam` sont **obligatoires**.
P est dans [1,9], N dans [0,10], les tentatives de placement dans
[1,4096] et les bornes temporelles sont strictement positives et ordonnées.
Les anciennes clés `red2InitiallyDestroyedProbability` et
`threatCurriculum` sont refusées.

Le scénario contient un template `blue_template` sans agent et, en
présence de Rouges, `red_template` avec le profil
`RLMission/production`. Les entités physiques `blue_i` et `red_i`
sont instanciées pour les effectifs choisis. Aucun pilote PPO à deux
Bleus n'est instancié ; les Rouges conservent leur RLMission d'origine.

L'acteur commun lit les pistes masquées de 0 à 10 Rouges, les équipiers
et les 61 rayons LiDAR ; les P actions sont exécutées de manière
synchrone et les P transitions sont agrégées par le critic individualisé.
Le runtime publie **15 canaux de récompense** (journal du run et port
`rewards` du Visualiseur). Le mapping exact des
sept clés JSON vers ces quinze canaux est décrit dans le
[guide d'adaptation de l'agent Blue](adapter-agent-blue.md#mapping-du-json-vers-les-15-poids).

Les checkpoints courants sont l'acteur V10 et l'état d'entraînement V7 dans
`Examples/BlueDefense/projects/BluePatrol/checkpoints/latest`. Les fichiers historiques archivés ne
définissent pas un second chemin d'exécution et ne sont pas acceptés lorsque
leur contrat d'observation est incompatible.

Voir [BlueDefense — MAPPO P/N](blue-defense-example.md) pour le
démarrage, la visualisation et les commandes `ppokit-run`.

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

Les modèles des agents rouges sont dans **Examples/Airspace/models/NOM_DE_PROFIL/**, dont les sous-dossiers **prod/** et **eval/** (`policy.ppokit`, sinon `policy.txt`).

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
plateforme rouge `attacker_car`. La configuration d'entraînement actuelle fixe
`CruiseSpeed` et `MaxSpeed` à 700 m/s. `MaxRotAcceleration` vaut
0,700000596 rad/s et `SNA.MaxShootAngle` 3,147198 rad. Ces limites reprennent
`trainingaircraft` et restent propres aux bleus ; `attacker_car` conserve
400 m/s, 0,40 rad/s et 1,047198 rad.

**Evaluation/InitialStates.json** contient des collections de points initiaux nommés, au format **presets.blueeval[]** et **presets.redeval[]**, avec les coordonnées et commandes dans des champs explicites **north**, **east**, **heading**, **speed**, **turnAcceleration**, **acceleration**. L'initialiseur C++ peut les utiliser comme auparavant, sans liste opaque de six nombres ni ancien chemin. Le champ **evaluationSettings** conserve les paramètres numériques associés à ces jeux d'initialisation.

## Environnement

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

Les configurations personnelles qui ne sont pas présentes dans le dépôt ne
peuvent pas être migrées automatiquement. Le runtime lit uniquement V2 ;
les champs inconnus ne sont pas reconnus et doivent être examinés avant tout
remplacement.

## Sauvegarde et erreurs

**Environment.json** et **Training/*.json** sont chargés par le Serializer, qui ne fait que les lire (il n'a plus de sauvegarde). Chaque fichier doit porter **formatVersion = 2** : un document incorrect, manquant ou impossible à charger est signalé et ses fonctions de chargement ne sont pas appelées, sans valeurs implicites. Les programmes PPOKit actuels ne réécrivent pas ces fichiers, contrairement à l'ancien PPO.exe. Éviter les modifications manuelles pendant l'exécution.

Les campagnes, les profils et les jeux de tests sont des données d'entrée. Les tailles des tableaux de récompenses, les valeurs de contrôle, les noms de profils et les références d'entités sont contrôlés au chargement par les composants concernés.

## Vérifications de non-régression

- Les scénarios V2 contrôlent le nombre d'entités, les fabriques résolues, la composition des évaluations et le nombre de cas.
- Les paramètres d'entraînement V2 (rollout, clamp, poids) sont comparés à des constantes numériques de référence, dans l'ordre exact des récompenses ; les poids nommés invalides sont refusés.
- Les tests du chargeur (`serializer_load_tests`) vérifient qu'un fichier qui ne se charge pas n'appelle pas ses fonctions de chargement, l'ordre des dossiers de recherche et `Unregister`.
- Les builds et CTest Windows VS2022 Debug/Release, ainsi que les tests de simulation/évaluation et la lecture des modèles existants, doivent être validés avant fusion.

Il n'y a **aucun lecteur V1 ni second format en production**. Une éventuelle conversion ponctuelle doit être exécutée hors application et ne fait pas partie du runtime.
