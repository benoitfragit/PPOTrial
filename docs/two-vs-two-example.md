# Exemple TwoVsTwo

## Lancement

L'exemple complet se lance avec :

~~~powershell
PPO.exe --example two-vs-two
~~~

Le contrôle de chargement sans interface graphique est :

~~~powershell
PPO.exe --example two-vs-two --check-example
~~~

Le curriculum sans ennemi est disponible avec `--example blue-patrol`.

## Composition du scénario

Le monde est le rectangle centré défini par `Environment.json`. Avec les
valeurs distribuées, l'axe est/ouest est plus long que l'axe nord/sud. La
patrouille bleue suit donc l'axe est/ouest.

Le scénario contient :

- deux défenseurs bleus pilotés ensemble par `PatrolMappoAgent` et RLTools ;
- un objectif bleu à protéger ;
- la zone bleue historique observée par le LiDAR RLMission rouge ;
- zéro ou plusieurs zones restreintes rouges configurables ;
- deux entités rouges pilotées par les agents `RLMission` de production
  existants, dont l'activation dépend du curriculum d'entraînement.

Les rouges apparaissent sur un arc extérieur devant l'objectif, pouvant aller
jusqu'aux entrées latérales nord et sud. Leur distance au centre vaut entre
1,05 et 1,35 fois la limite du rectangle dans cette direction et une séparation
initiale minimale de 10 km est imposée. Ils visent ensuite l'objectif avec une
petite variation de cap. Les limites du monde ne sont pas des murs et ne les
arrêtent pas : les agents RLMission de production continuent à les piloter.

Le second rouge peut toujours être retiré avec `red_2.enabled=false`. Lorsque
`threatCurriculum.enabled` vaut `false`, le réglage historique
`red2InitiallyDestroyedProbability` choisit les épisodes 2v1. Lorsqu'il vaut
`true`, les deux entités restent chargées mais commencent masquées jusqu'à leur
éventuelle étape d'activation.

Chaque zone est instanciée comme `RestrictedZone/Hostile`, sans dépendre de son
nom. Son activation et sa position sont échantillonnées une fois au reset et
sont reproductibles à graine identique. `restrictedZones.enabled` coupe la
collection entière; `enabled` et `enabledProbability` pilotent chaque entrée.

Ces valeurs sont configurables dans `Scenarios/TwoVsTwo.json` :

~~~json
"redSpawn": {
  "minimumWorldScale": 1.15,
  "maximumWorldScale": 1.45,
  "minimumBearingDegrees": -90.0,
  "maximumBearingDegrees": 90.0,
  "minimumSeparation": 10000.0
},
"restrictedZoneSpawn": {
  "enabledProbability": 0.75,
  "minimumNorthFraction": -0.45,
  "maximumNorthFraction": 0.45,
  "minimumEastFraction": 0.15,
  "maximumEastFraction": 0.65
},
"threatCurriculum": {
  "enabled": true,
  "noThreatProbability": 0.075,
  "singleThreatProbability": 0.15,
  "delayedFirstThreatProbability": 0.0,
  "delayedSecondThreatProbability": 0.0,
  "minimumFirstDelaySteps": 256,
  "maximumFirstDelaySteps": 768,
  "minimumSecondDelaySteps": 384,
  "maximumSecondDelaySteps": 1024
},
"restrictedZones": {
  "enabled": true,
  "items": [{
    "id": "red_restricted_zone_north",
    "type": "RestrictedZone",
    "platform": "restrictedzone",
    "side": "Hostile",
    "enabled": true,
    "enabledProbability": 0.75,
    "randomPosition": {
      "north": { "minimum": 14000.0, "maximum": 20500.0 },
      "east": { "minimum": -25000.0, "maximum": 25000.0 }
    }
  }]
}
~~~

La probabilité restante après `noThreatProbability` et
`singleThreatProbability` correspond aux épisodes à deux menaces. Le premier
intervalle est le délai absolu de la première activation ; le second est un
délai supplémentaire appliqué à la seconde. Les évaluations désactivent ce
tirage et conservent des cas immédiats 2v1/2v2 reproductibles.

Un angle de 0° place un rouge directement en face de l'objectif ; -90° et
+90° correspondent aux deux côtés. Les positions des zones sont en mètres.
Chaque entrée exige exactement `fixedPosition` (`north`, `east`) ou
`randomPosition` (plages `minimum`/`maximum` nord et est). Les deux modes, les
champs inconnus, les doublons d'identifiant et les probabilités hors de `[0,1]`
sont refusés avec un diagnostic.

`restrictedZoneSpawn` reste le contrat historique de la zone bleue
`blue_restricted_zone`. Il n'est pas remplacé par la nouvelle collection :
son ordre de tirage aléatoire, son LiDAR rouge et sa règle SNA restent
inchangés. `restrictedZones` décrit uniquement les nouvelles zones rouges.

Le rayon létal est exactement `SNA.MaxNoEscapeZoneRange` du profil. Le test de
segment empêche le tunneling entre deux pas. Seuls les `Mobile/Friend` sont
détruits et `m_destroyedById` reçoit l'identifiant de la zone. Les rouges ne
sont jamais concernés par ces zones rouges. Le pseudo-LiDAR bleu réutilise sans changement les 61
rayons RLMission/SAM et son rayon d'avertissement de 110 %; les zones détruites,
désactivées ou du même camp sont ignorées.

## Responsabilités des modules

`Examples/Shared/World` fournit les entités 2D, la cinématique, les commandes
SNA et le pseudo-LiDAR. `Examples/TwoVsTwo` compose le scénario, les positions
initiales, les règles et le contrôleur bleu. `Examples/Airspace` conserve les
agents RLMission rouges et le renderer partagé.

Le contrôleur bleu utilise un acteur partagé, une récompense d'équipe et un
critique centralisé. Les observations locales contiennent les deux pistes
rouges avec leur masque vivant/disponible ; un encodeur partagé, une attention
masquée et les poolings moyen/max masqués agrègent ces pistes. Le détail du
contrat se trouve dans [blue-ppo-patrol.md](blue-ppo-patrol.md).

Le format, le réseau, le LiDAR et les checkpoints RLMission ne changent pas.
Les modèles rouges restent sous `resources/model/RLMission/`. L'acteur bleu
reste en v7 avec exactement 156 valeurs, dont les mêmes 61 emplacements LiDAR.

## Affichage et entraînement

Le renderer montre les entités dans la vue tactique 3D et dans la minimap. Le
panneau `Blue MAPPO (RLTools)` affiche la progression du rollout, la récompense
totale, les sept canaux de récompense, leurs coefficients et le nombre de mises
à jour PPO. Il affiche aussi les campagnes d'évaluation automatiques, avec
progression, score, succès, survie et timeouts.

Le rollout bleu vaut 2048 pas d'équipe par défaut et se configure dans
`resources/settings/Training/BluePatrol.json`. Les 128 échantillons visibles
dans le backend correspondent à la taille d'un mini-batch PPO, pas à un arrêt
de l'entraînement.

Toutes les dix mises à jour par défaut, le candidat courant est testé sans
exploration sur des cas reproductibles 2v2 et 2v1. Les rouges restent les
agents RLMission de production pendant ces tests. Seul un score meilleur est
promu vers le checkpoint `prod` ; la collecte MAPPO reprend ensuite.

## Build et tests

RLTools est intégré en C++17 et utilisé sur CPU. LibTorch reste le backend des
rouges. La validation principale couvre :

- le spawn 2v2 ;
- le chargement des deux scénarios ;
- l'avancement conjoint des deux défenseurs ;
- la présence des observations rouges et des récompenses dans les snapshots ;
- une mise à jour MAPPO complète en smoke test ;
- la patrouille sur l'axe long et les règles de terminaison.
