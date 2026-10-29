# Prompt d'implémentation — zones rouges puis P bleus vs N rouges

Tu travailles dans le dépôt C++ `PPOTrial`. Tu es expert C++17, architecture
logicielle, apprentissage multi-agent MAPPO et RLTools. Le travail doit être
réalisé en deux phases strictement séparées :

1. ajouter des zones d'interdiction rouges configurables et observables par les
   bleus ;
2. généraliser ensuite le scénario vers `P` bleus contre `N` rouges.

Ne commence pas la phase 2 tant que la phase 1 n'est pas terminée, documentée,
testée et validée.

## Référence immuable à préserver

La référence fonctionnelle 2-vs-2 est le commit :

`072c6c7862a7f17d0efd347135f7201409f4e622`

Elle contient les deux checkpoints suivis par Git :

- acteur : `resources/model/BluePatrol/latest/blue_patrol_mappo_v7.txt`
  - SHA-256 : `E216F78D55CBA90DF889F68D84C08374903AAFD1C64F0DE4B84F850E26DE8FA7`
- état complet : `resources/model/BluePatrol/latest/blue_patrol_training_v4.txt`
  - SHA-256 : `4C31937260534D3F423D215BA3A8EA62667033DFDF72BED0A2E9F44B1CDDDE68`

Le checkpoint complet contient l'acteur partagé, le critique centralisé, les
moments Adam des deux réseaux, les états scalaires des deux optimiseurs, le
compteur d'updates et le RNG.

Contraintes impératives :

- ne jamais écraser, renommer ni supprimer ces artefacts v7/v4 ;
- ne jamais modifier silencieusement leur contrat ;
- introduire de nouveaux noms et de nouvelles versions pour les formats
  généralisés ;
- fournir une migration explicite des poids compatibles ;
- conserver un chemin de retour vers le scénario et le modèle 2-vs-2.

## Règles générales de réalisation

- Utiliser exclusivement C++17 pour le code C++.
- Avant toute modification, auditer le code existant et présenter le plan, les
  contrats de données et les incompatibilités de checkpoint pour approbation.
- Ne pas créer de monolithe. Séparer clairement configuration, domaine pur,
  adaptation ECS, apprentissage, checkpoints, évaluation et rendu.
- Utiliser des types métier explicites et des responsabilités uniques. Ne pas
  exposer d'API publique inutile.
- Centraliser les capacités maximales et supprimer les constantes magiques
  dispersées.
- Préserver les allocations bornées dans les boucles critiques ; préférer des
  tableaux de capacité fixe avec masques lorsque RLTools exige des dimensions
  statiques.
- Documenter toutes les nouvelles API avec Doxygen (`@brief`, `@param`,
  `@return`, invariants et unités).
- Mettre à jour `README.md`, la documentation d'architecture, le guide de
  configuration et les exemples JSON.
- Ajouter des tests unitaires, des tests de contrat de checkpoint et des smoke
  tests ECS. Si un build est nécessaire, privilégier Debug Win32.
- Relire les changements, exécuter `git diff --check`, vérifier les JSON et ne
  jamais inclure de fichiers utilisateur sans rapport.

## État actuel à ne pas mal interpréter

Le MAPPO bleu possède déjà un encodeur partagé, une attention masquée, un
pooling moyen et un pooling maximum pour les pistes rouges et amies. En
revanche, les capacités brutes sont encore figées à deux pistes rouges et deux
pistes amies. L'attention est invariante à l'ordre des pistes présentes, mais
le stockage d'entrée n'est pas encore extensible.

Les 61 valeurs appelées pseudo-LiDAR dans l'observation bleue sont actuellement
recalculées depuis les limites rectangulaires du monde dans
`patrol_core.hpp`. Le composant ECS `sPseudoLidarState`, pourtant alimenté par
les règles du monde, n'est pas consommé par le MAPPO bleu. Il faut corriger ce
point pendant la phase 1 sans augmenter la dimension de l'acteur v7.

Le sélecteur `RLMission` expose au maximum dix pistes classées. Il accepte
actuellement toutes les entités adverses hors missiles, y compris les
`RestrictedZone`. Cette règle doit être traitée explicitement pendant la phase
2, car neuf mobiles bleus plus l'objectif occupent déjà les dix pistes.

# Phase 1 — zones d'interdiction rouges

## Objectif fonctionnel

Permettre de déclarer dans le scénario zéro, une ou plusieurs zones rouges :

- de type `RestrictedZone` ;
- appartenant au camp `Hostile` ;
- positionnables librement depuis le JSON du scénario ;
- désactivables globalement et individuellement ;
- activées ou non aléatoirement à chaque épisode ;
- visibles dans le LiDAR des bleus ;
- détruisant immédiatement un mobile bleu qui pénètre dans leur rayon létal.

Les zones rouges ne doivent pas détruire les rouges. Les règles doivent reposer
sur les camps et les types d'entités, pas sur des noms codés en dur.

## Configuration attendue

Concevoir un schéma JSON typé, validé strictement et documenté. Chaque zone doit
au minimum fournir :

- un identifiant stable ;
- un profil de plateforme `RestrictedZone` ;
- un booléen `enabled` ;
- une probabilité `enabledProbability` dans `[0, 1]` ;
- soit une position fixe en mètres, soit des plages aléatoires ;
- des plages nord/est permettant de placer la zone partout où le scénario
  l'autorise, à l'intérieur ou à l'extérieur de la zone défendue bleue.

Les modes fixe et aléatoire doivent être mutuellement exclusifs et validés. Un
JSON invalide doit faire échouer explicitement le chargement avec un diagnostic
utile. Après cette implémentation, modifier positions, probabilités et
activation ne doit plus nécessiter de recompilation.

Ne pas limiter le chargeur à `blue_restricted_zone`, `red_1` et `red_2`. Faire
la découverte par métadonnées typées (`side`, `type`, rôle de scénario) ou par
une structure de configuration dédiée, sans dépendre d'une cascade de noms.

## Géométrie, destruction et LiDAR

Créer une source de vérité unique et testable pour la géométrie des zones :

- rayon létal documenté, exprimé en mètres ;
- fonction pure de test de pénétration ;
- éventuel rayon d'avertissement LiDAR documenté séparément ;
- mêmes conventions de coordonnées dans le domaine, l'ECS, le rendu et les
  tests.

La destruction demandée est déterministe : dès que le centre d'un mobile bleu
est dans le rayon létal d'une zone rouge active, le mobile est détruit et
`m_destroyedById` référence la zone. Détecter aussi une traversée entre deux
pas afin d'éviter le tunneling à grande vitesse. Publier l'explosion et mettre
fin à l'épisode selon les règles MAPPO existantes.

Ne pas s'appuyer implicitement sur le seuil probabiliste `pk > 0.95` si ce
seuil produit un rayon différent de celui annoncé. Si les règles SNA existantes
sont conservées pour les autres scénarios, isoler la règle déterministe du
scénario dans un composant clair et documenté.

Conserver exactement 61 rayons dans l'observation bleue afin de ne pas casser
la dimension de l'acteur v7. Fusionner pour chaque rayon :

- la distance à la frontière du monde ;
- la distance aux zones rouges actives ;
- toute autre obstruction déjà officiellement couverte par ce capteur.

La valeur la plus proche doit gagner. Les zones désactivées, détruites ou du
même camp doivent être ignorées. Le runtime MAPPO doit réellement consommer le
résultat ECS ; il ne suffit pas de publier `sPseudoLidarState` pour le rendu.

## Tests d'acceptation de la phase 1

Ajouter au minimum les tests suivants :

1. aucune zone configurée : comportement 2-vs-2 inchangé ;
2. zone `enabled=false` : absente des collisions et du LiDAR ;
3. probabilité `0` puis `1` : activation déterministe ;
4. position fixe : coordonnées exactes après reset ;
5. position aléatoire : valeurs dans les plages et reproductibles avec le seed ;
6. mobile bleu hors zone : vivant ;
7. mobile bleu entrant ou traversant la zone : détruit par la bonne entité ;
8. mobile rouge dans une zone rouge : non détruit ;
9. rayon LiDAR face à la zone : distance attendue ;
10. acteur v7 toujours chargeable, avec une observation de même dimension ;
11. récompense et terminaison MAPPO cohérentes après destruction d'un bleu ;
12. snapshot/rendu montrant correctement l'état actif ou désactivé.

Créer un commit autonome pour cette phase avant de commencer la suivante.

# Phase 2 — P bleus contre N rouges

## Capacités et invariants

Généraliser le domaine avec les limites initiales suivantes :

- `1 <= P <= 9` mobiles bleus actifs ;
- exactement un leader bleu par épisode ;
- `P - 1` équipiers bleus ;
- `0 <= N <= 10` mobiles rouges actifs pour le premier contrat généralisé ;
- un objectif bleu fixe ;
- un nombre de zones bleues et rouges indépendant des pistes mobiles.

Centraliser ces limites dans un contrat de capacité unique. `P` et `N` sont des
comptages actifs de runtime validés par rapport aux capacités statiques. Aucun
tableau `std::array<..., 2>` ne doit subsister dans le domaine généralisé pour
représenter une équipe.

## Sémantique des rôles bleus

Il doit toujours exister un et un seul leader. Son identité physique peut être
permutée de façon reproductible au reset. Les autres bleus reçoivent chacun un
emplacement de formation distinct et stable pendant l'épisode.

Ne pas donner la même consigne de formation aux `P - 1` équipiers. Introduire
un identifiant ou un encodage de slot normalisé, avec une géométrie de formation
testable pour toutes les valeurs de `P`. Le leader conserve la responsabilité
de progression sur l'hippodrome ; les équipiers reçoivent leurs responsabilités
de tenue de formation, tandis que les objectifs de défense restent coopératifs.

## Acteur partagé et observations locales

Conserver un acteur décentralisé unique avec paramètres partagés entre les
`P` bleus. Chaque acteur local doit recevoir :

- son état propre normalisé ;
- son rôle leader/équipier et son slot de formation ;
- les mobiles rouges actifs sous forme d'ensemble masqué ;
- les autres mobiles bleus sous forme d'ensemble masqué ;
- l'objectif et les données de patrouille utiles ;
- le LiDAR corrigé pendant la phase 1.

Utiliser les encodeurs partagés et l'attention/pooling invariants à l'ordre.
Séparer le stockage brut des pistes de la représentation compacte fournie au
MLP. Le nombre de pistes ne doit pas déplacer arbitrairement les caractéristiques
sémantiques dans l'entrée compacte.

Réutiliser les poids v7 compatibles : encodeurs de pistes, requêtes d'attention,
tronc acteur, sortie et écart-type. Fournir un test de migration démontrant que
le nouveau modèle reproduit la sortie v7 dans un cas 2-vs-2 équivalent, à une
tolérance numérique documentée.

## Critique centralisé et collecte MAPPO

Le critique doit accepter des équipes de cardinalités variables sans dépendre
de l'ordre des entités. Utiliser des ensembles masqués et un pooling explicite,
pas une concaténation figée de deux gardes et deux attaquants.

Chaque pas d'équipe doit produire exactement `P` transitions cohérentes, une
par acteur bleu, avec le même instant physique. Le rollout reste compté en pas
d'équipe et la documentation doit expliquer la différence entre pas d'équipe
et transitions individuelles.

Généraliser :

- les récompenses coopératives ;
- les récompenses spécifiques au leader et aux slots de formation ;
- la détection balayée des collisions pour toutes les paires bleues ;
- la survie de l'objectif ;
- l'évaluation et les statistiques de survie normalisées par `P` et `N`.

## Spawns et curriculum rouges

Remplacer les deux emplacements `red_1`/`red_2` par une collection bornée et
configurable. Générer `N` spawns avec séparation minimale vérifiée entre toutes
les paires, activation immédiate ou retardée, masques cohérents et seed
reproductible. Un échec de placement doit produire un diagnostic et ne jamais
boucler indéfiniment.

Chaque rouge conserve son contrôleur de production `RLMission`. Les poids
rouges existants doivent rester chargeables.

## Contrat des dix pistes RLMission

Avec `P = 9`, chaque rouge doit pouvoir observer exactement les neuf mobiles
bleus et l'objectif, soit dix pistes. Les `RestrictedZone` ne doivent donc pas
consommer ces dix emplacements mobiles/objectif.

Modifier proprement la sélection afin que les zones soient traitées par le
pseudo-LiDAR et l'évaluation de menace dédiés, mais exclues du bloc des dix
pistes classées. Vérifier que cela ne change ni la dimension 152 ni le format
du checkpoint `RLMission`. Ajouter un test avec neuf bleus, un objectif et
plusieurs zones : les dix pistes attendues doivent être présentes, aucune zone
ne doit évincer un mobile ou l'objectif.

## Checkpoints et migration

Créer de nouvelles versions pour :

- l'acteur généralisé ;
- l'état complet d'entraînement généralisé ;
- le contrat d'observation ;
- les métadonnées `P`, `N`, capacités et version de layout.

La migration doit :

- lire les formats v7/v4 sans les modifier ;
- copier tous les paramètres compatibles ;
- copier les moments Adam uniquement lorsque leurs formes et sémantiques sont
  identiques ;
- initialiser explicitement les nouvelles branches ;
- réinitialiser et documenter les états incompatibles, notamment les parties
  nouvelles du critique ;
- refuser un fichier tronqué, non fini ou de mauvaise version ;
- écrire atomiquement les nouveaux checkpoints.

Ne jamais présenter une migration partielle comme une reprise exacte de
l'optimiseur. Produire un rapport de migration lisible indiquant ce qui a été
copié et ce qui a été réinitialisé.

## Tests d'acceptation de la phase 2

Couvrir au minimum :

1. `P=1, N=0` ;
2. `P=2, N=2`, équivalent au scénario historique ;
3. `P=9, N=10`, capacités maximales ;
4. exactement un leader pour chaque `P` ;
5. slots de formation uniques pour les équipiers ;
6. invariance aux permutations des bleus et des rouges ;
7. masques des entités absentes ou détruites ;
8. `P` actions et `P` transitions par pas d'équipe ;
9. collisions balayées entre toutes les paires bleues ;
10. spawn de `N` rouges respectant toutes les séparations ;
11. dix pistes RLMission pour neuf bleus plus l'objectif ;
12. zones exclues des pistes RLMission mais visibles au pseudo-LiDAR ;
13. chargement inchangé des checkpoints rouges ;
14. migration v7 vers le nouvel acteur ;
15. rejet des checkpoints incompatibles ou corrompus ;
16. reprise du nouvel état complet d'entraînement ;
17. évaluation déterministe et métriques normalisées ;
18. absence de régression sur le scénario 2-vs-2 archivé.

## Livrables attendus

- code C++17 modulaire et documenté ;
- schémas et exemples JSON ;
- migration de checkpoint explicite ;
- tests unitaires et smoke tests ;
- documentation et `README.md` à jour ;
- rapport final listant les fichiers modifiés, les commandes de validation,
  les résultats, les limites restantes et la stratégie de reprise
  d'entraînement.

À chaque phase, commencer par présenter le plan détaillé et demander validation
avant d'écrire le code.
