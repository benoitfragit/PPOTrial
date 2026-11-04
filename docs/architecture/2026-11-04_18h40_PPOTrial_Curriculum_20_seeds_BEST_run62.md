# PPOTrial — Curriculum ciblé, 20 évaluations et reprise BEST 62

Date : 2026-11-04. Livraison : PR #53, correction de la gate 4v1 en PR #54.

## Décision

- Gates obligatoires uniquement, ordre : 1v0 → 2v0 → 3v0 → 4v0 → 1v1 → 2v1 → 4v1 → 2v2 → 3v2 → 3v3 → 4v3 → 4v4.
- Gates définitivement supprimées du curriculum progressif : 3v1, 1v2, 4v2, 1v3, 2v3. Aucun mode facultatif, aucun scénario parallèle.
- `4v1` est conservée explicitement, comme validé précédemment : le BEST 4v1 du run 62 reste réutilisable pour la suite et les rappels.
- Rappel inchangé : `rehearsalEveryBlocks=8`, uniquement parmi les gates effectivement validées du cycle.
- Validation du curriculum GitHub uniquement : **20 épisodes déterministes par campagne**, seeds `1004242..1004261`, distinctes de la seed d'entraînement `42` et des anciennes évaluations `4242..4249`. Le scénario BlueDefense autonome conserve ses paramètres.
- Seuil demandé par défaut : **92 %** pour toutes les gates ; avec 20 essais, le passage exige 19/20 = **95 %**. Ne pas confondre le seuil demandé avec la résolution de l'échantillon.
- Aucun changement de modèle Rouge RLMission, guidage, physique, actions, rewards, réseau MAPPO ni production 4v4.

## Migration sûre depuis le run 62 (pas 63)

1. L'action existante **Restart Progressive Blue Defense (checkpoint)** démarre avec `source_run_id=41865674608`, `start_gate=2v2`, `success_percent=92`, `same_threshold_all_gates=true`.
2. Vérifier que le run source a terminé en succès, puis lire son `gates.json` au **SHA exact du commit source** depuis l'API GitHub et comparer son SHA-256 au manifeste du checkpoint source (LF/CRLF acceptés).
3. Le run 62 a terminé en `4v1` (gate conservée dans le parcours). **Restaurer le BEST évalué de 4v1**, couple V10 acteur + V7 entraînement complet (critic, Adam, RNG, compteur). Si le BEST 4v1 manque ou n'est pas évalué, refuser la migration.
4. Créer une graine **non évaluée** de `2v2` depuis ce BEST, réinitialiser la progression de la gate principale et mettre à jour son empreinte avec le nouveau `gates.json`. Conserver les anciennes gates authentiquement validées et présentes dans le nouveau parcours pour le **rappel**, sans prétendre qu'elles ont été certifiées sur 20 scénarios. Les modèles antérieurs restent dans les archives BEST. Ne jamais passer `2v2` à tort.
5. Lancement d'un nouveau run de l'entraînement progressif existant ; son mécanisme de BEST par gate reste strictement inchangé et seul un champion évalué et associé au bon état de l'optimiseur peut être promu.
6. L'action ne prend **aucune donnée du run 63 annulé**. Une nouvelle gate a toujours son premier taux de validation mesuré sur 20 seeds.
7. Le mécanisme de rappels continue sur les gates précédemment validées, y compris historiques sous leurs évaluations à huit cas. Il ne valide jamais la gate principale à la place d'une vraie évaluation correspondante.

## Sécurité et validation

- Brancher depuis master, conserver un seul commit de PR et merger en `squash` après vérification CI.
- Tests de régression : séquence/menus identiques, exclusion des six gates, rappel périodique, seeds distinctes, seuil à 20 cas, source/empreinte historique vérifiés, et reprise BEST.
- Le premier run sur un nouveau SHA peut devoir reconstruire l'exécutable avant l'usage du cache.
- Build VS2022 x64 Debug/Release et CTest attendus par AGENTS.md ; ne déclarer réussis que les tests réellement exécutés.
