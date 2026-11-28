# PPOKit — état de mutualisation

Mise à jour : 27 novembre 2026, branche `design/ppokit-framework`, PR #105.

## État fonctionnel

- Le moteur `CurriculumEngine` est l’unique machine d’état de curriculum pour
  les promotions par épisodes, évaluations indépendantes ou budgets d’updates.
  Il gère les répétitions déterministes, les raisons de décision, l’avancement,
  la complétion et un état versionné lié par empreinte à toute la configuration.
- `ppokit-run` exécute ce moteur pour Studio, CLI et CI. Une génération publie
  ensemble learner, politique, état d’environnement, état de curriculum et
  poids de récompense. `--fresh`, `--resume`, Pause, Resume et Stop empruntent
  ce même chemin.
- Les workflows Blue sont des clients du moteur. `common_plan.py` traduit les
  profils historiques en configuration canonique sans prendre de décision au
  runtime. Le workflow de reprise ne réécrit plus un état externe.
- RLMission et RLSNA utilisent l’éditeur générique pour leurs graphes acteur et
  critique. Les nœuds partagés acteur/critique sont synchronisés dans une seule
  transaction Undo/Redo ; encodeurs, ensembles, masques, attention et pooling
  sont validés. Un changement structurel rend explicitement incompatible un
  checkpoint historique, tandis qu’un graphe inchangé conserve son empreinte.
- RLMission, RLSNA, BluePatrol, T6 et PC7 déclarent le learner de production
  commun `ppokit_graph_network`. `ppokit-new` génère une DLL d’environnement et
  un projet consommé par `ppokit-run`, sans executable PPO propre.

## Qualification historique (27 novembre, avant nettoyage)

Les tests déterministes couvrent les décisions et reprises du curriculum, les
profils Blue historiques, l’édition/rechargement RLMission et RLSNA, Undo/Redo,
la compatibilité de checkpoints, les projets du dépôt et le scaffold `reach`.

- Configuration CMake régénérée dans les seuls répertoires x64 autorisés
  `build/debug` et `build/release`.
- Build Release de la solution complète : toutes les cibles produites sauf le
  remplacement du `ppokit-run.exe` nominal, maintenu ouvert par une autre tâche.
  Le même projet a été lié sans erreur sous
  `build/release/PPOKit/Validation/ppokit-run-final.exe`. Les cibles modifiées
  ont également été reconstruites en Debug avec un runner de validation isolé.
- CTest Release : 113/113 tests sélectionnés passent, plus
  `BlueDefenseSharedBackendImportSmoke` exécuté séparément hors sandbox (le
  test crée volontairement un lien physique), soit 114 tests qualifiés ; le
  golden trace Flight sans fixture locale reste marqué `Skipped` comme avant.
- CTest Debug : les tests 1–85 passent après reconstruction des DLL ABI 7 ;
  l’oracle `BlueDefenseGeneralizedLearnerTests` passe directement et la tranche
  87–113 est exécutée séparément. Les trois tests touchés par la relecture
  finale (`Curriculum`, `RepositoryProjects`, `Run`) repassent en Debug et
  Release avec les binaires finaux.
- Python 3.12 : 80/80 tests `tools/ci`, 93/93 tests
  `tools/ci/adaptive_curriculum` et 6/6 tests de déclenchement de workflows.
- Le scénario Studio headless réel passe avec Train, Pause/Resume, Stop,
  évaluation et inférence via le runner commun.

## Compatibilité conservée

- `project.legacy.json`, `blue_network`, `FlightControlLearner`,
  `FlightControlRuntime` et `FlightControlHeadless` ne sont plus des chemins de
  production. Ils sont `EXCLUDE_FROM_ALL` et construits seulement par les
  convertisseurs, tests de parité et qualifications qui les nomment.
- Les anciens modèles Airspace au format texte et les outils de qualification et
  de conversion Flight ont été retirés. Les modèles PPOKit livrés n’ont pas été
  modifiés. Les chemins Blue historiques encore utilisés par la CI et les
  lecteurs nécessaires aux reprises restent présents.
- Les anciens algorithmes Python de gates sont conservés comme oracles de
  non-régression ; aucun workflow de production ne les appelle pour décider un
  stage ou une promotion.

## Limite intentionnelle

Les budgets `max_seconds` sont des limites de client par invocation ; ils
n’entrent pas dans la décision mathématique de promotion et ne prétendent pas
mesurer une convergence. La qualification d’un modèle reste une évaluation
indépendante multi-seed.
