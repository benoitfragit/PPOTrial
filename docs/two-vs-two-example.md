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

Sans contrat P/N optionnel, le scénario contient :

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
"redTeam": {
  "activeCount": 4,
  "maximumPlacementAttempts": 1024,
  "delayedActivationProbability": 0.25,
  "minimumActivationDelaySteps": 256,
  "maximumActivationDelaySteps": 768
},
"restrictedZones": {
  "enabled": true,
  "items": [{
    "id": "red_restricted_zone_outside_north",
    "type": "RestrictedZone",
    "platform": "restrictedzone",
    "side": "Hostile",
    "enabled": true,
    "enabledProbability": 0.65,
    "oppositeObjective": true,
    "randomPosition": {
      "north": { "minimum": 3000.0, "maximum": 15000.0 },
      "east": { "minimum": 36000.0, "maximum": 42000.0 }
    }
  }]
}
~~~

Chaque rouge possède son propre RLMission. La probabilité de délai est appliquée
indépendamment, sans changer ses observations ni ses poids.

Un angle de 0° place un rouge directement en face de l'objectif ; -90° et
+90° correspondent aux deux côtés. Les positions des zones sont en mètres.
Chaque entrée exige exactement `fixedPosition` (`north`, `east`) ou
`randomPosition` (plages `minimum`/`maximum` nord et est). Les deux modes, les
champs inconnus, les doublons d'identifiant et les probabilités hors de `[0,1]`
sont refusés avec un diagnostic.
Avec `oppositeObjective=true`, la valeur est est une magnitude automatiquement
miroir du côté opposé à l'objectif. Le profil livré définit uniquement deux
zones de bordure dont les centres restent hors des limites du monde ; aucune
zone rouge n'est centrée dans la zone défendue et l'objectif ne peut jamais
être couvert.

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

Les zones restent hors des dix pistes classées de RLMission : seuls les
`Mobile` et `Objective` opposés peuvent consommer ces slots. Les zones restent
observables par le chemin SAM/pseudo-LiDAR dédié ; l'observation RLMission de
152 valeurs et ses checkpoints ne dépendent donc jamais du nombre de zones.

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
Les modèles rouges restent sous `resources/model/RLMission/`. Le chemin bleu
legacy reste en v7 avec exactement 156 valeurs, dont les mêmes 61 emplacements
LiDAR. Le chemin P/N est décrit ci-dessous.

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

## Runtime MAPPO généralisé unique

L'entrée de catalogue existante `two-vs-two` utilise directement le runtime
généralisé. Aucune entrée `p-vs-n` et aucun second MAPPO ne sont créés.
`Scenarios/TwoVsTwo.json` démarre à P=2/N=4 ; `blueTeam.activeCount`
dimensionne les bleus dans la capacité 1..9 et `redTeam.activeCount` les rouges
dans la capacité 0..10.

À chaque reset, un unique leader est tiré avec le seed d'entraînement. Les
autres bleus gardent un slot distinct et stable en formation V. Les rouges
restent chacun pilotés par leur agent RLMission de production.

Les objets `blueTeam` et `redTeam` restent optionnels dans le parseur.
Le fichier livré renseigne les deux sur l'unique entrée `two-vs-two`. Les quatre
rouges conservent chacun leur RLMission de production ; un quart de leurs
activations est retardé de façon seedée. Le même objet `twoVsTwo` accepte :

~~~json
"blueTeam": {
  "activeCount": 5
},
"redTeam": {
  "activeCount": 7,
  "maximumPlacementAttempts": 512,
  "delayedActivationProbability": 0.25,
  "minimumActivationDelaySteps": 256,
  "maximumActivationDelaySteps": 768
}
~~~

`blueTeam.activeCount` accepte les entiers `[1,9]`. Le chargeur clone un seul
template `Mobile/Friend` sans agent sous les noms `blue_1..blue_P`, puis crée
un `GeneralizedPatrolMappoAgent` partagé. `redTeam.activeCount` accepte
`[0,10]`; un seul template `Mobile/Hostile` ciblant `blue_objective` et portant
exactement `RLMission/production` produit `red_1..red_N`. N=0 ne crée aucun
rouge. `redTeam` et le `threatCurriculum` legacy actif sont mutuellement
exclusifs.

Le placement de chaque rouge respecte `redSpawn.minimumSeparation`. Il essaie
au plus `maximumPlacementAttempts` candidats, valeur elle-même bornée à
`[1,4096]`, puis échoue avec un diagnostic si la géométrie est impossible. La
probabilité de délai reste dans `[0,1]`; les bornes d'activation sont des pas
d'équipe strictement positifs, finis et ordonnés. Aucun de ces contrôles
n'introduit une boucle non bornée.

`BlueTeamRoles` choisit exactement un leader à chaque reset. Les P-1 autres
indices physiques obtiennent des slots uniques `0..P-2`, alternés
intérieur/extérieur puis reculés par rangées pour former le V. Leur slot
normalisé se trouve dans `[-1,1]`. Le contrôleur lit P observations d'un même
instant, produit P actions avant la physique et enregistre exactement P
transitions portant le même numéro de pas après la résolution ECS. Une entité
détruite conserve son slot stable mais son masque d'apprentissage devient nul.

L'observation brute v8 contient 295 flottants : état propre, exactement 61
rayons `sPseudoLidarState` ECS, rôle/slot, dix pistes rouges, huit autres bleus,
un objectif séparé, patrouille, données globales et P/N normalisés. Le dernier
champ de chaque piste est un masque explicite. L'attention, la moyenne et le
maximum masqués produisent ensuite le tronc compact de 156 valeurs consommé
par le MLP ; le padding n'est jamais interprété comme une entité.

Le runtime P/N écrit `blue_patrol_generalized_v8.txt` et
`blue_patrol_generalized_training_v5.txt`. Sans v8/v5, il peut migrer un acteur
v7. L'API explicite `MigrateV4TrainingState` reprend depuis v4 uniquement
l'acteur et les premiers/deuxièmes moments Adam compatibles : le critique 2v2,
les états scalaires des optimiseurs, le RNG, le compteur et le rollout sont
réinitialisés. Les fichiers `blue_patrol_mappo_v7.txt` et
`blue_patrol_training_v4.txt` restent nécessaires au chemin legacy et ne
doivent jamais être supprimés.

Aux frontières d'update, une campagne P/N distincte joue des actions
déterministes dans le vrai ECS sans alimenter le rollout. Ses resets seedés et
miroirs sont reproductibles. Elle publie un score composé de métriques
normalisées par P/N (survie bleue, neutralisation rouge, objectif, succès,
leader et slots) et promeut uniquement un meilleur acteur vers
`prod/blue_patrol_generalized_v8.txt`. Cette promotion appartient à l'unique
MAPPO exposé par `two-vs-two` et ne modifie jamais les checkpoints rouges.

Boucle rapide pendant le développement :

~~~powershell
cmake --build out/build/x64-Debug --config Debug --target TwoVsTwoSpawnTests TwoVsTwoTeamDomainTests TwoVsTwoTeamDynamicsTests TwoVsTwoGeneralizedActorTests TwoVsTwoGeneralizedRuntimeStateTests TwoVsTwoGeneralizedLearnerTests TwoVsTwoGeneralizedRewardEvaluationTests MissionTrackSelectionTests --parallel 4
ctest --test-dir out/build/x64-Debug -C Debug -R "^(TwoVsTwoSpawnTests|TwoVsTwoTeamDomainTests|TwoVsTwoTeamDynamicsTests|TwoVsTwoGeneralized.*|MissionTrackSelectionTests)$" --output-on-failure
~~~

Les smoke tests `TwoVsTwoPatrolRuntimeSmoke`, `TwoVsTwoCombatRuntimeSmoke` et
`TwoVsTwoEmbeddedLearnerSmoke`, puis `--check-example`, sont réservés à la
validation finale plus lourde avant fusion.
