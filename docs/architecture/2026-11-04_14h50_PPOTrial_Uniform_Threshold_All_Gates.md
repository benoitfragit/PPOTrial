# PPOTrial — Seuil de réussite uniforme par défaut pour toutes les gates
Date : 2026-11-04. Référence : correctif du curriculum progressif GitHub Actions.

## Besoin
Dans l'action **Progressive Blue Defense (Windows Release)**, le champ
`success_percent` doit s'appliquer à **toutes** les gates automatiquement
enchaînées, pas seulement à la première. L'option est cochée par défaut.
Par défaut, le seuil est **85 %**, modifiable dans l'interface.

## Architecture
- Entrée GitHub Actions `same_threshold_all_gates: true` par défaut.
- `progressive_gate.py` persiste dans
  `adaptive-curriculum-progress.json` :
  - `sameThresholdAllGates` (booléen) ;
  - `uniformThresholdPercent` (valeur demandée dans le run).
- Le résultat `adaptive-gate-result.json` reprend les mêmes champs.
- `dispatch_next_gate.py` vérifie la cohérence état/résultat et transmet :
  - `success_percent = uniformThresholdPercent` en mode uniforme ;
  - `success_percent = thresholdPercent` de la prochaine gate si désactivé ;
  - `same_threshold_all_gates` explicite, afin que la prochaine action
    ne revienne pas involontairement au défaut.
- La gate choisie, son temps maximum, les flags `auto_advance` et
  `resume_run_id` ainsi que la logique d'archivage des champions restent
  inchangés.
- L'action **Restart Progressive Blue Defense** continue d'utiliser son
  `restartCycle` et sa logique d'enchaînement unique. Son appel transmet
  explicitement l'option du formulaire. Aucun déclenchement double.
- Le curriculum V2/V3 classique, les rewards, RLMission, les exécutables
  C++ et les modèles ne sont pas modifiés.

## Comportement attendu
| Action | Seuil |
|---|---|
| Nouveau départ 1v0, 85 %, option cochée | 85 % |
| Passage automatique à 2v0 | 85 % |
| Passage à 1v1 ou 4v4 | 85 % |
| Option décochée | seuil configuré individuellement dans `gates.json` |
| `auto_advance=false` | aucune transition automatique |
| Reprise via `resume_run_id` | conserve le champion PROD + V7 complet |

## Validation
- Tests déterministes de propagation sur plusieurs gates.
- Validation du rejet de métadonnées de seuil incohérentes.
- Test de l'option par défaut et de l'opt-out dans l'interface GitHub.
- Tests existants de reprise, d'enchaînement du restart et de préservation
  des checkpoints.
- Validation CI `actionlint`, CTest, Python curriculum V3.
