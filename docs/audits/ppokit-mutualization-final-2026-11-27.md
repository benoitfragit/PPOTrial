# Audit final de mutualisation PPOKit — 27 novembre 2026

## Conclusion

Les chemins de production des cinq projets historiques convergent sur
`ppokit-run`, `ppokit_graph_network`, les DLL d’environnement et
`CurriculumEngine`. Les implémentations historiques restantes ne sont plus
enregistrées comme programmes PPOKit et ne sont pas construites par défaut.
Elles sont conservées uniquement lorsqu’un convertisseur ou un oracle de
non-régression les appelle explicitement.

## Inventaire PPO/MAPPO et véritables appelants

| Composant | Classement | Appelants autorisés |
|---|---|---|
| `ppokit_graph_network` + PPOKit Training/Core | moteur de production unique | RLMission, RLSNA, BluePatrol, T6, PC7, projets `ppokit-new`, Studio, CLI et CI |
| `ppokit-run` | orchestration de production | Studio, commandes utilisateur et workflows GitHub |
| DLL `rlmission_environment`, `rlsna_environment`, `blue_environment`, `flight_environment` | adaptateurs d’environnement | `ppokit-run`, viewer et tests de contrat ; aucun learner embarqué |
| `blue_network` et `project.legacy.json` | compatibilité et oracle | tests de parité/visualisation explicitement dépendants ; cible `EXCLUDE_FROM_ALL` |
| `FlightControlLearner`, `FlightControlRuntime`, `FlightControlHeadless` | oracle et convertisseur | tests numériques et `flight_checkpoint_import` ; toutes ces cibles et le convertisseur sont `EXCLUDE_FROM_ALL` |
| lecteurs Blue V7/V10, Flight et Airspace | convertisseurs de checkpoints | migrations et tests de compatibilité |
| `mappo_curriculum.py`, `progressive_gate.py`, profils historiques | oracles de non-régression | tests Python et outil local de qualification, jamais un workflow de production |
| ancien bootstrap `branch_checkpoint_seed.py` et tests cross-variant associés | code mort | supprimés après retrait du workflow qui reconstruisait l’état externe |
| exécutable PPO propre du scaffold `reach` | code mort | supprimé ; la DLL générée est chargée par `ppokit-run` |

Les recherches d’appelants couvrent CMake, manifestes de projets, DLL,
exécutables, Studio, tests, scripts et workflows. Les cibles historiques
restantes exigent donc un nom de cible explicite ou sont tirées par une cible de
qualification ; aucun build normal ni workflow d’entraînement ne les sélectionne.

## Curriculum commun

Le moteur commun accepte trois sources de promotion : fenêtre d’épisodes,
résultat d’une suite d’évaluation indépendante et quota d’updates. Sa
configuration canonique porte les payloads de stage, critères, seuils,
répétitions et possibilité d’avancement automatique. Les payloads restent
interprétés par l’adaptateur d’environnement ; PPOKit ne dépend donc d’aucune
stratégie Blue ou Flight.

L’état versionné contient stage de promotion, stage actif de répétition, fenêtre,
frontière d’updates, complétion, dernière évaluation et raison. Son empreinte
couvre également les quotas et poids de répétition, `auto_advance` et les
budgets. Une reprise avec une autre configuration ou un stage actif incohérent
est refusée. Les publications transactionnelles conservent cet état avec le
learner et l’environnement.

Les workflows Blue génèrent éventuellement un plan canonique, puis appellent
le même `ppokit-run` que le Studio. Le workflow manuel de reprise transmet une
génération publiée complète ; il ne fabrique plus de progression parallèle.

## Graphes RLMission et RLSNA

`networkEditable` n’était pas seulement un verrou d’interface : les paramètres
de couches partagées pouvaient diverger entre le propriétaire acteur et les
références critique, et certains paramètres de masquage/pooling n’étaient pas
éditables. L’éditeur synchronise désormais les représentations partagées dans
une transaction atomique et valide les valeurs finies, dimensions, masques,
encodeurs, attention et pooling.

Les tests ouvrent les manifestes réels RLMission/RLSNA, sauvegardent et
rechargent leurs graphes sans changer l’empreinte, vérifient la compatibilité
des checkpoints historiques, appliquent une modification structurelle et
constatent le refus, puis restaurent la compatibilité par Undo. Les architectures,
poids, observations, actions et récompenses livrés n’ont pas été modifiés.

## Suppressions et isolation

- suppression de l’ABI de décision de curriculum appartenant à
  l’environnement ; seule l’application d’un payload de stage subsiste ;
- suppression du `main.cpp` PPO du scaffold et remplacement par une DLL
  d’environnement générique ;
- suppression du bootstrap CI qui reconstruisait
  `adaptive-curriculum-progress.json` et de son code/test mort ;
- retrait des enregistrements de programmes historiques et passage des
  learners historiques nécessaires en `EXCLUDE_FROM_ALL` ;
- remplacement dans les workflows des DLL/manifestes historiques par
  `ppokit_graph_network` et `project.json`.

## Validation et limites

Le détail des commandes Debug, Release, CTest et Python figure dans la
description de la PR et dans `PPOKit/PROGRESS.md`. Les tests C++ ne valent pas
preuve de convergence : aucune politique n’a été réentraînée et les campagnes
indépendantes restent requises pour qualifier les poids.
