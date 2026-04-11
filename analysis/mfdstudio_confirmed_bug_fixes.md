# Correctifs propres — bugs confirmés (MFDStudio)

> Note: le dépôt source `benoitfragit/MFDStudio` n'est pas présent dans ce workspace, donc ce fichier contient un **plan de patch exact** (code + doc) à appliquer dans ce dépôt.

## 1) BUG-04 — collision `fill` (bool / color)

### Objectif
- Canonique: `filled` (bool) et `fillColor` (color)
- Legacy temporaire: `fill` (bool ou color) avec warning de dépréciation

### Correctif code (JsonLoader)
1. Parser d'abord `filled` (bool)
2. Parser ensuite `fillColor` (color)
3. Si clé `fill` existe:
   - bool -> map vers `filled`, warning
   - color -> map vers `fillColor`, warning
4. Si conflit (`filled` + `fill` bool, ou `fillColor` + `fill` color), priorité au champ canonique + warning

### Correctif doc
- `docs/reference/common_json_syntax.md`
  - remplacer les exemples `fill` par `filled`/`fillColor`
  - ajouter encart “legacy alias `fill` (deprecated)”

---

## 2) BUG-07 (+ CPP-01/CPP-02) — validation cardinalité au parseur

### Objectif
- Rejeter tôt les primitives invalides (bezier/triangle/polyline)

### Correctif code (JsonLoader)
- `bezier`: exiger `controlPoints.size() >= 2`
- `triangle`: exiger `points.size() == 3`
- `polyline`: exiger `points.size() >= 2`

En cas d'échec: erreur parse explicite avec nom de primitive + cardinalité attendue.

### Correctif doc
- `docs/reference/primitive_reference.md`
  - ajouter section "Validation guarantees"
  - préciser cardinalités minimales/maximales et message d'erreur attendu

---

## 3) BUG-11 — alias rotation ambigu

### Objectif
- Canonique unique: `rotationDegrees`
- Alias legacy: `rotation`, `angle` (warning)

### Correctif code
- Mapping parse:
  1. lire `rotationDegrees` (prioritaire)
  2. fallback `rotation` / `angle` avec warning
- En cas de présence multiple, priorité à `rotationDegrees` + warning “conflicting aliases ignored”.

### Correctif doc
- `docs/reference/common_json_syntax.md`
  - “Rotation unit = degrees”
  - exemples uniquement avec `rotationDegrees`

---

## 4) BUG-13 — `ring` manquant du README

### Correctif doc
- `README.md`
  - ajouter `ring` dans “Supported Primitive Types”
- ajouter un test de cohérence doc/runtime (script CI simple):
  - comparer liste README vs enum/registry primitives

---

## 5) Tests à ajouter (minimum)

1. `fill` legacy bool/color + champs canoniques (priorité et warnings)
2. `bezier` avec 0/1/2 points
3. `triangle` avec 2/3/4 points
4. `polyline` avec 0/1/2 points
5. rotation avec `rotationDegrees`, `rotation`, `angle`, et collisions
