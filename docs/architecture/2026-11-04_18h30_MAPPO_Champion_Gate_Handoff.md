# PPOTrial — Architecture des checkpoints MAPPO « champion »

Date : 2026-11-04. Périmètre : GitHub Actions Progressive Blue Defense, orchestration Python, **aucun changement C++ ni RLMission**.

## Défaut racine

Dans l'ancien curriculum, `latest/blue_patrol_generalized_training_v7.txt`
continue à progresser après qu'un acteur a réussi une évaluation. Les fichiers
`prod/` sauvegardent éventuellement un acteur seul, selon un **score composite**
C++ et une logique de sauvegarde différente de celle du taux de réussite des
gates. De plus, l'ancien pilote effaçait temporairement `prod/` pour les
gates hors 4v4, réinitialisant donc le meilleur score interne C++ à chaque bloc.
Le passage de gate restaurait `latest`, pas l'époque évaluée gagnante.

## Invariants

1. `latest/` reste le checkpoint de progression continue : acteur V10, critique,
   Adam acteur et critique, RNG, compteur dans le V7. Il peut régresser.
2. `best/<gate>/` contient **exactement la paire évaluée** :
   `blue_patrol_generalized_v10.txt`,
   `blue_patrol_generalized_training_v7.txt`,
   `champion.json` (époque, gate, provenance, taux de réussite, score, SHA-256).
3. Les couples sont publiés uniquement après la **confirmation de la
   sauvegarde acteur + V7** puis une **campagne déterministe terminée** au
   même update exact. Un bloc se termine à la prochaine frontière
   `checkpointEvery`, pour empêcher l'update suivant d'écraser l'évalué.
   Si plusieurs threads entrelacent le texte de démarrage, le message de
   sauvegarde réussi sert de référence d'update.
4. Un couple n'est remplacé que si le résultat de la **même gate** améliore
   (priorité : **taux de réussite de mission** ; départage : score composite).
   Une mauvaise évaluation ne modifie jamais le champion.
5. À la transition, la gate suivante reçoit une **copie des deux fichiers**
   du champion précédent. Le champion précédent reste intact. Si une gate
   reprise possède déjà une évaluation BEST au-dessus du seuil demandé, elle
   peut passer sans relancer PPO ; aucun acteur plus récent non évalué ne peut
   prévaloir. L'initialisation
   de la nouvelle gate est marquée *seed*, **pas réussite certifiée** sur la
   nouvelle configuration P/N. La première évaluation sur cette gate constitue
   sa première référence comparable.
6. Dans chaque bloc, le `prod/` C++ temporaire est initialisé depuis le
   `best/<selected_gate>` correspondant. Son score n'est chargé que s'il a été
   mesuré sur cette même gate. Le `prod/` externe 4v4 reste protégé et est
   restauré après chaque bloc. La promotion de curriculum n'est **pas un
   déploiement global 4v4**.
7. Checkpoint manquant, hash invalide, acteur non V10, V7 absent, époque
   incohérente ou campagne incomplète : refus de promotion/transition.
8. Les artefacts GitHub Actions exportent `best/**` pour franchir les jobs
   et runs indépendants ; le dispatcher vérifie le champion avant l'action
   suivante. Aucun changement de récompense, physique, observation ou modèle
   rouge.

## Reprise des runs antérieurs à la correction

- Un run déjà en cours, non encore validé, peut **continuer depuis son V7
  `latest`** : il ne repart pas de zéro ; la prochaine évaluation complète
  créera le premier champion fiable.
- Une ancienne gate marquée **passée** sans dossier `best/` peut
  importer son **dernier checkpoint évalué et validé**, sous réserve que
  l'update du V7 corresponde au compteur enregistré dans l'artefact. L'ancien
  orchestrateur s'arrêtait après la dernière campagne ayant validé la gate.
  La provenance est marquée `legacy-v1-gate-pass`.
- Si ce run avait obtenu 100 % à un update antérieur, puis 87,5 % lors de la
  dernière évaluation validante, **seul le 87,5 % est récupérable** depuis son
  artefact final. On ne prétend pas restaurer les 100 % perdus, mais on n'impose
  pas de recommencer depuis zéro. Si le V7 est plus récent que l'update
  enregistré, la transition est refusée : aucune promotion fictive.
- L'action de redémarrage d'un cycle rétablit le meilleur couple complet si
  l'artefact source dispose de `best/`. Sinon elle conserve le V7 legacy
  comme **warm-start non certifié** ; cela n'implique pas un réentraînement
  depuis zéro.

## Critères d'acceptation

- Évaluation 87,5 %, puis 25 % à un update plus récent : `latest` avance,
  `best/<gate>` conserve l'acteur ET le V7 de 87,5 %.
- Transition de gate après ces mises à jour : acteur et V7 de départ provenant
  **tous les deux du `best`**, avec compteur identique.
- Rehearsal sur une gate antérieure : aucune contamination de la production
  4v4 ni de la validation de la gate courante.
- Corruption d'un des deux fichiers BEST : échec contrôlé, pas de promotion.
- CI : tests déterministes Python + lint workflows ; builds C++/CTest restent
  nécessaires avant tout merge. La convergence de la politique n'est pas
  déduite de ces tests techniques.
