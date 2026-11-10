# Blue MAPPO — export ONNX (Actor et Critic)

Outil **hors entraînement**, limité aux Bleus RLTools. Il ne charge ni
`RLMission` ni LibTorch et ne modifie jamais les checkpoints existants.

## Choix du langage

Le parseur et le générateur sont en **Python**, avec NumPy et ONNX.
Le format texte V10/V7 est déjà suffisamment spécifié dans les lecteurs C++.
Aucun nouveau code C++ n'est nécessaire pour exporter les matrices. Le seul
travail de comparaison à conserver côté C++ est la **validation de parité**
sur les mêmes entrées réelles que le modèle RLTools.

## Entrées / sorties

- V10 : `BLUE_TEAM_MAPPO_V10`, acteur partagé et attention masquée → `blue_actor.onnx`.
- V7 : `BLUE_TEAM_TRAINING_V7`, acteur V10 encapsulé + critic centralisé →
  `blue_actor.onnx` et `blue_critic.onnx`.
- `manifest.json` : SHA256 du checkpoint d'origine et version/metadata P/N.
- Dimensions : Actor `[batch,295] → [batch,2]`, plus `log_std [1,2]`.
  Critic `[batch,470] → [batch,1]`.
- Un Actor unique est partagé par tous les Bleus : aucun modèle par pilote.

La construction ONNX reproduit le compactage des 295 observations en 156
valeurs et les **deux attentions masquées** du
`generalized_attention_actor.hpp` (query `tanh`, encodeur `tanh`,
softmax masqué, pooling context/mean/max). Les pistes absentes, y compris
l'ensemble vide, contribuent zéro.

## Utilisation locale

Installer les dépendances **dans un environnement Python séparé** :

```bash
python -m pip install "numpy>=1.26,<3" "onnx>=1.16,<2" "onnxruntime>=1.19,<2"
python tools/blue_onnx/export.py \
    --checkpoint resources/model/BluePatrol/latest/blue_patrol_generalized_v10.txt \
    --output-dir build/blue-onnx/local
```

Pour exporter le Critic également, sélectionner
`blue_patrol_generalized_training_v7.txt` comme checkpoint.
Puis ouvrir `blue_actor.onnx` ou `blue_critic.onnx` sur
[Netron](https://netron.app/).

**Il n'y a ni compilation C++ ni nouvelle dépendance du simulateur.**

## GitHub Actions

Workflow `Blue MAPPO - Export BEST ONNX (read-only)` :

- **Manuel** : fournir l'ID d'un run Progressive Blue Defense, la gate BEST
  (ex. 4v4) et la variante de scénario : `fixed`, `fixed-zones-on`,
  `randomized` ou `randomized-zones-on`.
- **Automatique** : à la fin d'un run Progressive Blue Defense **réussi**
  dont la branche source est la branche par défaut, exporter les couples BEST
  évalués dans tous les espaces de noms de l'artefact
  `blue-mappo-checkpoints`, y compris les variantes imbriquées.
- Le script réutilisable `export_best.py` effectue exactement les mêmes
  validations en CI et en local. Exemple d'export de tous les BEST évalués :

```bash
python tools/blue_onnx/export_best.py \
    --best-root resources/model/BluePatrol/best \
    --output-dir build/blue-onnx
```

- Vérification du manifeste `champion.json`, des SHA256, du compteur d'updates
  et de l'identité des poids V10 / V7.
- Publication séparée `blue-best-onnx` (30 jours) : jamais d'écriture
  dans `latest`, `prod` ou les fichiers BEST.

## Tests et limites

```bash
python -m unittest discover -s tools/blue_onnx -p "test_*.py" -v
```

- Tests de format V10/V7 et de rejet des données erronées.
- Tests du traitement des BEST : quatre espaces de noms, sélection manuelle,
  intégrité SHA256, correspondance des poids et lecture seule.
- `onnx.checker.check_model`, ONNX Runtime contre référence NumPy pour
  les deux réseaux, pistes masquées, ensembles vides et permutation.
- **Non certifié à ce stade** : égalité numérique RLTools C++ versus ONNX
  sur des observations d'inférence issues du simulateur réel.
  Tant que cette validation n'est pas réalisée, l'ONNX est destiné
  à la **visualisation et au diagnostic**, pas au remplacement du moteur
  d'inférence du jeu.

Source de référence : `docs/architecture/blue-network-architecture.md`.
Respecter `AGENTS.md`, conserver le chemin de production C++ et le
format historique des checkpoints inchangés.
