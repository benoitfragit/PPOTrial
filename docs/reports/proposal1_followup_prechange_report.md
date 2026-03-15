# Rapport pré-modification — Correctif de suivi point 1

## Contexte
Le premier lot sur le point 1 a introduit une couche robuste de chargement/schéma dans `Serializer`, mais a aussi modifié massivement des fichiers JSON de ressources.

## Problème visé
- Diff trop large sur les données (`resources/settings`), susceptible de masquer les changements métier.
- Besoin de minimiser le risque de régression en gardant des ressources inchangées.

## Proposition de correction
1. Conserver les changements C++ de robustesse dans `Serialization/*`.
2. Restaurer `resources/settings/*.json` à l'état antérieur au commit `e28c474`.
3. Conserver la compatibilité via le fallback déjà implémenté (`schemaVersion` absent => warning + fallback version 1).
4. Refaire les vérifications (validité JSON + check build applicable à l'environnement).

## Impact attendu
- Même robustesse au chargement qu'avant correction.
- Patch plus propre et plus sûr (moins de bruit sur les assets de données).
- Alignement renforcé avec l'exigence de non-régression.
