# PPOTrial — Auto-revue corrective PR #47
Date : 2026-11-03, 18h28 (Paris)
Branche : `fix/pvs0-physical-wedge-speed-reward-run26`
Base : `97125c5cbbab9f77eff14969ad0bc9b13ed594d9`
État attendu : PR en brouillon, un commit squash, aucun merge.

## Modèle de décision
Le MAPPO bleu conserve un acteur partagé. Les cibles géométriques et le wedge physique restent des **observations**, sans sélection scriptée PATROL/DEFEND ou attribution des Rouges. Actions et physique inchangées.

## Corrections rewards
- En présence d'au moins un Rouge réellement actif avant ou après le pas, aucune récompense de rapprochement de l'hippodrome ne concurrence l'interception.
- La poursuite de la cible Rouge conserve son normaliseur et son amplitude existants, même à grande distance, sans coefficient d'engagement qui l'annule.
- Les équipiers reçoivent la récompense existante de progrès de formation en combat, pondérée par la coopération existante; sans Rouge actif, ils apprennent le wedge et le guidage physique.
- Aucun canal, poids, bonus ou normalisation nouveau. Les conditions ne commandent pas l'acteur, elles qualifient les objectifs de récompense.
- Tests : un Rouge/multiples Rouges, distant, apparition/destruction, absence de Rouge, P=1.

## Curriculum progressif GitHub
- Gate ouverte jusqu'à réussite évaluée, sans nombre maximal d'épisodes ni quota d'updates; les blocs de 20 updates ne sont que les points de checkpoint/évaluation.
- Une répétition des gates antérieures tous les huit blocs complétés, déterministe et persistée via `blockIndex` au travers des runs. La répétition ne peut pas valider la gate principale.
- Sur expiration : même gate, même modèle et état optimiseur V7, nouveau budget au run GitHub suivant. Contrôle de version et compteur monotone lors de la reprise.
- Sauvegarde/restauration systématique du candidat de production 4v4 à chaque bloc non-4v4, y compris lors des répétitions à la gate 4v4.
- En cas d'absence d'update sur un run repris, conservation du checkpoint valide restauré plutôt que fausse erreur de fraîcheur.
- Seuil, temps maximal et auto-enchaînement configurables; dispatch de l'étape suivante seulement après publication et validation d'une campagne de réussite.

## Points à vérifier impérativement
- Tester l'intégration GitHub Actions réelle, notamment le dispatch et la restauration cross-run; les tests Python utilisent des simulations.
- Évaluer séparément le combat 1v1 et 4v4, l'objectif, les destructions, les collisions, et le guidage Pvs0 sur des seeds indépendantes; les CTest ne prouvent aucune convergence.
- Interface `workflow_dispatch` : les nouveaux champs peuvent ne pas apparaître avant présence du workflow sur la branche par défaut.
- Ne pas modifier RLMission, les modèles Rouges, la physique ni les voies de build; ne pas merger sans validation explicite.

## Références
- `.github/workflows/mappo-training-windows.yml`
- `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp`
- `Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_reward_shaping.hpp`
- `tools/ci/adaptive_curriculum/progressive_gate.py`
- `tools/ci/adaptive_curriculum/test_progressive_gate.py`
