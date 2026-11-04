# PPOTrial — correction de la reprise du BEST run 62

Date : 2026-11-04. Incident : run 64, GitHub Actions ID `41883733716`, job `126762551085`.

## Symptôme observé
Le bootstrap `41883666492` a réussi à importer le BEST évalué de `4v1` du run 62 (`41865674608`), puis à préparer une nouvelle gate `2v2`. Le run d'entraînement Windows a téléchargé le checkpoint, configuré et compilé PPO.exe Release, puis échoué avant le premier update PPO sur `ValueError: Incompatible adaptive checkpoint provenance` dans `progressive_gate.initialize_state()`.

## Cause et correctif
Le bootstrap Ubuntu calcule `gateConfigSha256` sur les octets LF du contrat `tools/ci/adaptive_curriculum/gates.json`. Un checkout Git Windows peut convertir le même contenu en CRLF, produisant un SHA-256 différent. La vérification stricte précédente rejetait cette divergence bénigne.

La fonction `equivalent_gate_config_hashes()` autorise exactement deux encodages du même fichier, LF et CRLF. Le validateur contrôle l'empreinte fournie par le code courant et celle du checkpoint contre cet ensemble. Après validation, l'empreinte persistée est mise au format du checkout courant. Les différences réelles de `gates.json` restent interdites.

## Invariants préservés
- Aucune modification du modèle Bleu V10, de son checkpoint d'entraînement V7, des optimisateurs Adam ou du RNG.
- Conservation du BEST évalué historique 4v1, des BEST par gate, de la règle de promotion, des rappels tous les huit blocs et du curriculum courant.
- 20 évaluations sur des graines différentes de l'entraînement, sans modification de leur configuration.
- Aucun changement de PPO C++, de RLMission Rouge, de la physique ni des rewards.
- Aucun entraînement dans le run 64 : le checkpoint bootstrap est une source utilisable.

## Test de non-régression
`test_run62_linux_bootstrap_resumes_2v2_on_windows_crlf` simule la restauration d'un BEST 4v1 évalué et du checkpoint complet sur un bootstrap LF, la conversion CRLF sur Windows, deux reprises successives, la préservation exacte des octets acteur/optimiseur, des rappels et de la provenance. Il rejette une modification réelle d'un seuil et une empreinte inventée.

Commande CI : `python -m unittest discover -s tools/ci/adaptive_curriculum -p "test_*.py" -v`. Vérification C++17 : pipeline `Core C++17 tests` sur Ubuntu et Windows.

## Relance après validation et fusion
Depuis `Progressive Blue Defense (Windows Release)` avec `resume_run_id=41883666492`, `progressive_gate=2v2`, `success_percent=90` (valeur effectivement transmise par le bootstrap 64), `same_threshold_all_gates=true`, `auto_advance=false` (le dispatcher unique du `restartCycle` conserve l'option auto-advance originale `true`), `max_seconds=14400` et `save_every_updates=1`. Ne pas relancer le workflow de bootstrap ni reprendre le run 63 annulé.

**Validation** : la PR et les tests doivent réussir avant fusion ; aucune exécution de reprise après correction n'est attestée ici.
