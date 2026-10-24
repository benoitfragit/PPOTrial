# PPOTrial — Configuration unique V2

La configuration de l'application utilise **un seul contrat** : chaque JSON commence par **"formatVersion": 2**. Le code C++17 de production ne contient aucun lecteur ni chemin de repli pour les anciennes configurations. Les valeurs physiques, les paramètres PPO, les poids des récompenses et les modèles de réseau sont conservés.

## Organisation

~~~text
resources/settings/
├── AgentProfiles.json            profils logiques et modes d'agent
├── Environment.json              monde de simulation principal
├── Renderer.json                 affichage et caméra
├── Training/
│   ├── RLMission.json            PPO, récompenses et version du modèle
│   └── RLSNA.json
├── Scenarios/
│   ├── Playground.json           composition du monde principal
│   └── Evaluation/
│       ├── RLMission.json        composition et limites des tests
│       └── RLSNA.json
├── Platforms/
│   ├── Aircraft/                 paramètres des mobiles
│   └── Ground/                   paramètres des plateformes fixes
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
3. **Training/RLMission.json**, ou **Training/RLSNA.json** : réglages PPO et pondération des récompenses.

L'évaluation est indépendante : sa composition se trouve dans **Scenarios/Evaluation/**, ses scénarios de référence dans **Evaluation/Cases/**.

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

Le profil réel contient l'intégralité des sections nécessaires à la plateforme ; le chemin **Platforms/Aircraft/trainingaircraft.json** identifie la plateforme, sans répéter son nom à la racine.

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

Renderer.json contient un objet **renderer** (scaleLevel, autoAttachCamera,
showGrid, showPseudoLidar, camera avec position, target, up et fovy), sans les
anciens noms de champs en majuscule. Le rendu et les valeurs enregistrées restent
identiques : seule la représentation de configuration a changé.

## Sauvegarde et erreurs

**Environment.json**, **Renderer.json** et **Training/*.json** sont chargés et sauvés par le Serializer. Chaque fichier doit porter **formatVersion = 2** : un document incorrect, manquant ou impossible à charger est **préservé sans écrasement** par des valeurs par défaut à l'arrêt de l'application. Éviter les modifications manuelles pendant l'exécution.

Les campagnes, les profils et les jeux de tests sont des données d'entrée. Les tailles des tableaux de récompenses, les valeurs de contrôle, les noms de profils et les références d'entités sont contrôlés au chargement par les composants concernés.

## Vérifications de non-régression

- Les scénarios V2 contrôlent le nombre d'entités, les fabriques résolues, la composition des évaluations et le nombre de cas.
- Les paramètres PPO V2 sont comparés à des constantes numériques de référence, notamment la cadence des checkpoints et l'ordre exact des récompenses.
- La sérialisation assure un cycle lecture-écriture-lecture sans modification des paramètres et refuse les poids nommés invalides.
- Les tests de sûreté vérifient qu'une configuration mal formée n'est pas écrasée.
- Les builds et CTest Windows VS2022 Debug/Release, ainsi que les tests de simulation/évaluation et la lecture des modèles existants, doivent être validés avant fusion.

Il n'y a **ni outil de conversion externe, ni lecteur V1, ni second format en production**.
