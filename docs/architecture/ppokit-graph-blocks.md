# PPOKit — graphes de blocs (encodeurs, attention, pooling)

> Statut : conception validée par l'utilisateur, implémentation en cours (voir
> `PPOKit/PROGRESS.md`, étape 9).
> Complète la §5.2 de [`ppokit-framework-design.md`](ppokit-framework-design.md).

## Pourquoi

`ppokit.graph/1` ne décrit que des MLP. Les réseaux réellement utiles, comme
l'acteur à attention de BlueDefense, combinent des blocs : encodeur par
entité, attention masquée, pooling, concaténation. L'éditeur doit montrer ces
blocs comme des boîtes reliées par des connecteurs typés. Sans eux, le
framework ne peut pas porter les modèles des exemples.

## Décision : un moteur de graphe PPOKit, chargé à l'exécution

- `ppokit.graph/2` décrit des **nœuds** et des **arêtes entre ports**.
- Le graphe est **construit au chargement** par le moteur de PPOKit, en C++,
  sans recompilation. Les dimensions sont déduites et vérifiées une fois ;
  tous les buffers sont alloués à ce moment ; un pas d'exécution n'alloue
  rien (AGENTS.md).
- Chaque bloc fournit `Forward`, `Backward` (gradient exact) et ses
  paramètres. L'optimiseur (Adam) est commun.
- La voie `ppokit.graph/1` (MLP générés pour rl_tools) reste disponible :
  elle est bit-exacte avec les learners existants de Flight et de Blue.
- Conséquence pour l'éditeur : une boîte ajoutée ou rebranchée est
  utilisable tout de suite, et la même bibliothèque vérifie les formes en
  direct.

## Ports et valeurs

| Type de valeur | Forme | Exemple |
|---|---|---|
| `vector` | `[n]` | features propres, sortie d'un dense |
| `set` | `[capacité, n]` + masque `[capacité]` | pistes rouges, coéquipiers |

Un ensemble transporte toujours son masque : les blocs qui le consomment
ignorent les entités absentes.

## Blocs de la première version

| Bloc | Entrées | Sortie | Paramètres |
|---|---|---|---|
| `input` | groupe de `task.json` (`from`) | `vector`, ou `set` si le groupe est un ensemble masqué | — |
| `dense` | `vector[n]` | `vector[width]` | `width`, `activation` (`tanh`, `relu`, `identity`) |
| `mlp` | `vector[n]` | `vector[width]` | `layers`, `width`, `activation` |
| `entity_encoder` | `set[c, n]` | `set[c, width]` | `width`, `activation` ; mêmes poids pour chaque entité ; `encode_absent` (booléen, défaut faux) : encode aussi les entités absentes (sinon lignes nulles) |
| `masked_attention` | `query: vector[d]`, `set: set[c, d]` (+ `keys: set[c, d]`) | `vector[d]` | `keys` (booléen, défaut faux) ; score `q·k/√d` (`k` = `keys` s'il existe, sinon l'entité), softmax sur les entités présentes, sortie = moyenne pondérée de `set`, zéro si l'ensemble est vide ; `mask_fill` (nombre fini) : le score d'une entité absente reçoit cette valeur en plus, et le softmax porte sur toutes les entités |
| `masked_pool` | `set[c, n]` | `vector[n]` | `mode` : `mean` ou `max` ; zéro si l'ensemble est vide ; en cas d'égalité au max, le gradient est partagé ; `mask_fill` (`max` seulement) : une entité absente compte pour cette valeur (l'ensemble vide donne la valeur) |

`encode_absent` et `mask_fill` reproduisent les réseaux LibTorch qui masquaient par
`masked_fill` (les agents rouges RLMission : −1e9) : sans aucune entité présente, la
valeur se propage comme dans l'ancien réseau au lieu de donner des zéros ; avec une
entité présente, les absentes ne pèsent rien. Ils font partie de l'identité du réseau.
| `concat` | `in1 … inK: vector` | `vector[Σ]` | — |
| `select` | `vector[n]` | `vector[k]` | `indices` : caractéristiques retenues, dans l'ordre (ex. un scalaire d'un groupe) |
| `concat_sets` | `in1 … inK: set[c_i, n]` | `set[Σc, n]` | — ; masques concaténés (ex. pistes amies puis objectif, vues comme un seul ensemble) |
| `gaussian_head` | `vector[n]` | moyennes `[n]` | `log_std` : `parameter` (défaut : appris, indépendant de l'état, valeur initiale `PpoSettings::initialLogStd`) ou `state` (couche dense de la même entrée, bornée par `log_std_min` et `log_std_max`, obligatoires ; pas de gradient hors bornes) |
| `bernoulli_head` | `vector[n]` | logits `[n]` | — |
| `value` | `vector[1]` | valeur | — |
| `component:<nom>@<version>` | ports du manifeste | ports du manifeste | DLL boîte noire (§12 de la conception) |

## Exemple : l'acteur de BlueDefense

```json
{
  "schema": "ppokit.graph/2",
  "actor": {
    "nodes": {
      "own":        { "type": "input", "from": "observation.own" },
      "red":        { "type": "input", "from": "observation.red_tracks" },
      "red_embed":  { "type": "entity_encoder", "width": 8, "activation": "tanh" },
      "red_query":  { "type": "dense", "width": 8, "activation": "tanh" },
      "red_attend": { "type": "masked_attention" },
      "red_mean":   { "type": "masked_pool", "mode": "mean" },
      "red_max":    { "type": "masked_pool", "mode": "max" },
      "features":   { "type": "concat" },
      "trunk":      { "type": "mlp", "layers": 2, "width": 128, "activation": "tanh" },
      "policy":     { "type": "gaussian_head" }
    },
    "edges": [
      ["red", "red_embed"], ["own", "red_query"],
      ["red_query", "red_attend.query"], ["red_embed", "red_attend.set"],
      ["red_embed", "red_mean"], ["red_embed", "red_max"],
      ["own", "features.in1"], ["red_attend", "features.in2"],
      ["red_mean", "features.in3"], ["red_max", "features.in4"],
      ["features", "trunk"], ["trunk", "policy"]
    ]
  }
}
```

Une arête relie `nœud` (sortie par défaut) à `nœud.port`. Le nom seul d'un
nœud vise son unique entrée.

Les attributs `keys`, `log_std: state` et `share` n'entrent dans l'identité
du réseau (texte canonique) que lorsqu'ils sont utilisés : les réseaux et
checkpoints existants gardent leur identité.

## Critique qui partage des couches de l'acteur

Un nœud `dense`, `mlp` ou `entity_encoder` du critique peut porter
`"share": "<nœud de l'acteur>"` : il utilise alors les paramètres de ce nœud
(même type, mêmes tailles, même activation, mêmes formes d'entrée ; sinon
l'erreur le dit). C'est le tronc commun acteur-critique des anciens agents
torch (RLMission, RLSNA). L'apprentissage devient joint, comme
`ppo.cpp` : un seul Adam sur les paramètres de l'acteur et ceux, non
partagés, du critique ; perte = acteur + `value_coefficient` × MSE −
`entropy_coefficient` × entropie ; les gradients de la copie du critique
s'ajoutent à ceux de l'acteur avant le pas, puis la copie reprend les valeurs
de l'acteur. `critic_learning_rate` doit alors valoir `actor_learning_rate`.
Le checkpoint écrit une section `adam` (au lieu de `actor_adam` et
`critic_adam`) ; la garde KL saute le pas entier. `max_gradient_norm`
(comme le `clip_grad_norm_(…, 0.5)` de l'ancien PPO torch) borne la norme
globale des gradients de cet optimiseur unique, après la somme des gradients
partagés. Dans le studio : case
« Separate keys » de l'attention, choix du log-std de la tête gaussienne,
et sélecteur « Share » sur les couches du critique.

## MAPPO : mêmes blocs, pour l'acteur et le critique

- **Acteur partagé** : un seul graphe, appliqué à l'observation de chaque
  agent (rôle compris, via un groupe d'entrée). Les agents détruits sont
  exclus de l'apprentissage par l'indicateur `learnable` du rollout.
- **Critique centralisé** : un graphe dont les entrées peuvent être des
  **ensembles d'agents** (alliés, adversaires) avec masque des vivants. Un
  `entity_encoder`, puis une `masked_attention` (requête issue de l'agent
  évalué) et des `masked_pool`, le rendent invariant à l'ordre et au nombre
  d'agents (capacités 1..9 Bleus et 0..10 Rouges, AGENTS.md).
- Le critique de Blue, aujourd'hui `own`, `mean` et `max` concaténés (470
  entrées), s'écrit ainsi avec `masked_pool` : c'est la même idée,
  généralisée et apprise de bout en bout.

## Vérifications à la construction

Chaque erreur nomme le nœud et le port, comme pour les autres fichiers :
- type de nœud inconnu, paramètre manquant ou hors domaine ;
- port inexistant, port d'entrée non relié ou relié deux fois ;
- `vector` branché sur un port qui attend un `set`, et inversement ;
- dimensions incompatibles (`masked_attention` : requête et entités de
  même largeur ; tête de la bonne taille pour les actions de `task.json`) ;
- cycle ;
- nœud dont la sortie n'est utilisée par personne.

## Preuves attendues

1. Chaque bloc : gradients comparés aux différences finies, masques
   (entités absentes, ensemble vide), absence d'allocation par pas.
2. Graphe complet : gradient de bout en bout aux différences finies.
3. Apprentissage : le couloir jouet, puis une tâche jouet à ensemble
   d'entités où l'attention est nécessaire.
4. Blue : le graphe ci-dessus, chargé avec les poids d'un checkpoint de
   l'acteur écrit à la main, doit donner les mêmes actions à une tolérance
   fixée. Ce n'est pas du bit à bit, car l'ordre des calculs diffère
   (décision B). Ensuite seulement, l'acteur écrit à la main pourra être
   remplacé. **Fait** : `Examples/BlueDefense/Agents/BluePatrol/graph_blocks.json`
   décrit l'acteur réel (entrée compacte v7 comprise, grâce à `select` et
   `concat_sets`) ; `graph_actor_parity_tests` copie les poids rl_tools et
   compare moyennes et gradients sur un lot complet : écart relatif ≤ 4e-7.
   **Remplacement fait** : l'acteur de production de BlueDefense est calculé
   par ce graphe (stockage rl_tools inchangé, donc checkpoints, migrations et
   modèle de production compatibles sans conversion) ; l'ancien code est
   gardé comme oracle de test dans `Examples/BlueDefense/tests/reference/`.
5. Éditeur : imnodes affiche ces nœuds et ports, et la même validation
   s'exécute en direct.

## Éditeur (PPOKit studio)

`ppokit-studio <task.json> <graph.json>` ouvre le graphe dans une fenêtre :
palette des blocs (infobulle sur chacun), un onglet par réseau, fils tirés
d'une sortie vers une entrée (carré orange : ensemble d'entités), formes
inférées affichées sous chaque nœud. La validation tourne après chaque
modification avec le même code que l'entraînement : la barre d'état dit si
le graphe s'entraîne tel quel, sinon le nœud, le port ou le fil fautif passe
en rouge avec le message, et « Show » le centre. Les positions vont dans
`<graph>.layout.json`, à part, pour que les diffs de `graph.json` ne montrent
que le réseau. `ppokit-studio --check` valide sans fenêtre.

Le modèle (`PPOKit::Editor`, `GraphDocument`) ne dépend pas d'ImGui ; la vue
(`GraphEditorView`) le dessine et ne contient aucune logique de réseau.

## Ordre de réalisation

1. Moteur et blocs vectoriels (`input`, `dense`, `mlp`, `concat`, têtes),
   avec les tests de gradient.
2. Ensembles : `entity_encoder`, `masked_attention`, `masked_pool`.
3. Schéma `ppokit.graph/2`, inférence des formes, messages d'erreur.
4. Learner PPO sur le moteur de graphe (même interface que
   `GaussianMlpLearner`), têtes gaussiennes et Bernoulli.
5. Parité avec l'acteur Blue, puis éditeur.
