# PPOTrial — Évaluation précoce des gates MAPPO

Date : 2026-11-04 12:28 (Europe/Paris).

## Objectif

Éviter que la première réussite d'une nouvelle gate soit perdue par des mises à jour PPO ultérieures, tout en garantissant que le checkpoint déclaré gagnant est **celui effectivement transmis** à la gate suivante.

## Contrat conservé

- Aucun changement du C++17, des observations/actions, des récompenses ou de RLMission.
- `blue_patrol_generalized_training_v7.txt` conserve acteur, critique, états Adam, RNG et compteur global d'updates.
- `resume_run_id`, `adaptive-curriculum-progress.json`, seuil de réussite, `auto_advance` et restauration des modèles 4v4 demeurent inchangés.
- Le rollout partiel n'est pas repris (comportement existant).

## Adaptation Python

1. Lors de la première entrée dans une gate avec un checkpoint restauré (`--resume`, `blockIndex == 0`), limiter le premier bloc à **une mise à jour PPO** pour provoquer immédiatement la campagne déterministe déjà disponible en C++.
2. Pour les blocs suivants, fixer `maxUpdates` à la prochaine frontière `checkpointEvery` (sans dépasser `block_updates`).
3. N'utiliser pour la validation que le résultat de la campagne terminée qui correspond au checkpoint final de ce bloc. Ne pas déclarer un modèle gagnant sur une évaluation antérieure si ses poids ont été modifiés ensuite.
4. Si la réussite satisfait le seuil, publier l'état complet et arrêter cette gate ; sinon poursuivre avec l'optimiseur existant.

## Exemple

Avec `checkpointEvery=10`, `block_updates=20` et reprise à l'update 2560 sur 2v0 :

- Première vérification à l'update 2561 (une seule update).
- Si elle échoue : nouvelle campagne à 2570 (9 updates supplémentaires).
- Si elle réussit : la gate s'arrête avec exactement le checkpoint qui a été évalué, sans continuer jusqu'à 2580.

## Limites connues

L'interface C++ actuelle ne permet pas une campagne avant **toute** update. La vérification est donc à +1 update et non à +0 ; aucune affirmation inverse ne doit être faite. Le seuil de 85 % avec huit épisodes nécessite au moins 7 succès sur 8 (87,5 %) et présente une variance statistique significative. Les tests Python vérifient le contrôle des frontières et du transfert de checkpoint, mais ne constituent pas une démonstration de convergence sur plusieurs seeds.

## Vérifications

Tests de régression Python : reprise 2560 → 2561, arrêt sur chaque frontière d'évaluation, validation de configuration, continuité du compteur PPO, non-régression de l'ancien orchestrateur. Construction C++ Debug/Release et CTest : non exécutés localement, à confirmer en CI.
