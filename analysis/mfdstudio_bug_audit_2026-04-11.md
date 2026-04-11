# Bugs MFDStudio — confirmés / partiellement confirmés (analyse code C++)

Date: 2026-04-11
Portée: revue des implémentations C++ (`.cpp/.hpp`) + cohérence API/doc associée.

## Liste filtrée (uniquement confirmés ou partiellement confirmés)

### BUG-04 — collision sémantique `fill` (booléen vs couleur) — **CONFIRMÉ**
- **Preuve code**: le parseur accepte `fill` comme booléen (`ParseFilledFlag`) et aussi comme couleur (`ParseFillColorField`).
- **Impact**: ambiguïté forte côté JSON auteur (même clé, deux significations selon type).
- **Correctif court**:
  1. garder `filled` (bool) comme source canonique,
  2. introduire `fillColor` (color) canonique,
  3. déprécier `fill` avec warning de compatibilité.

### BUG-07 — `bezier` avec 0 point — **PARTIELLEMENT CONFIRMÉ**
- **Preuve code**: la chaîne de rendu ne crash pas, mais le cas vide est accepté et passe dans le pipeline (`EvaluateBezier`/échantillonnage), ce qui peut produire un rendu non voulu (points nuls / trait parasite).
- **Impact**: comportement silencieux non déterministe pour l’auteur de scène.
- **Correctif court**:
  1. valider `controlPoints.size() >= 2` au parse JSON,
  2. lever une erreur claire sinon,
  3. ajouter un test négatif dédié.

### BUG-11 — alias `rotation` sans unité explicite — **CONFIRMÉ**
- **Preuve code**: le parseur mappe plusieurs alias (`rotationDegrees`, `rotation`, `angle`) vers la même propriété.
- **Impact**: ambiguïté API (unités/priority des alias).
- **Correctif court**:
  1. standardiser sur `rotationDegrees`,
  2. déprécier `rotation`/`angle`,
  3. documenter l’unité en degrés partout.

### BUG-13 — primitive `ring` absente du README — **CONFIRMÉ (doc/runtime mismatch)**
- **Preuve code**: `Ring` est traité dans le rendu, mais omis de la liste synthétique README.
- **Impact**: capacité runtime existante mais mal communiquée.
- **Correctif court**:
  1. ajouter `ring` au README,
  2. ajouter un test de cohérence doc ↔ enum runtime.

## Bugs C++ additionnels identifiés pendant l’analyse

### CPP-01 — Bezier vide accepté sans validation stricte — **CONFIRMÉ**
- **Preuve code**: absence de garde bloquante explicite au chargement pour `controlPoints` vide.
- **Impact**: rendu parasite possible au lieu d’une erreur de validation.
- **Correctif court**:
  1. rejet parse-time si `< 2` points,
  2. message d’erreur explicite,
  3. test unitaire parseur.

### CPP-02 — Validation cardinalité trop tardive (rendu au lieu du parseur) — **PARTIELLEMENT CONFIRMÉ**
- **Preuve code**: certaines protections existent surtout côté rendu (retours précoces) plutôt qu’au chargement JSON.
- **Impact**: scènes invalides acceptées silencieusement, diagnostics faibles.
- **Correctif court**:
  1. déplacer les garde-fous cardinalité au parseur,
  2. conserver protections runtime en filet de sécurité,
  3. ajouter tests négatifs (`triangle`, `polyline`, `bezier`).

## Résumé exécutable (priorités)
1. **P1**: corriger `fill` (BUG-04).
2. **P1**: validation stricte des cardinalités au parseur (BUG-07, CPP-01, CPP-02).
3. **P2**: normaliser les alias de rotation (BUG-11).
4. **P3**: aligner README avec capacités runtime (`ring`, BUG-13).
