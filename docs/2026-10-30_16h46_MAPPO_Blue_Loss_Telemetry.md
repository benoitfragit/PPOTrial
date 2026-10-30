# PPOTrial — MAPPO Bleu : diagnostic non intrusif des pertes

Date : 30 octobre 2026. Portée : BlueDefense, MAPPO généralisé uniquement.

## Objectif

Déterminer si les pertes de Bleus, auparavant classées `unknown` faute
d'identifiant destructeur ECS, correspondent à une vitesse trop faible ou
à un épuisement du carburant. Aucun comportement de simulation ou de RL n'est
modifié.

## Architecture conservée

```text
MAPPO partagé + critic centralisé (inchangés)
      |
      v
Actions Bleu (rotation, accélération) — inchangées
      |
      v
World2D / FlightModelSystem — inchangés
      |
      v
Snapshot ECS avant / après un pas de simulation
      |
      +-- entraînement : aucun diagnostic supplémentaire
      |
      +-- évaluation : transition vivant -> détruit
             |
             +-- [MAPPO_EVAL_LOSS] sur stdout (une ligne par perte)
             +-- [MAPPO_EVAL_EPISODE] existant, conservé
                     |
                     v
           GitHub Actions capture mappo-training.log
                     |
                     v
           tools/ci/mappo_metrics.py
                     |
                     +-- résumé de la dernière campagne
                     +-- mappo-losses.csv dans blue-mappo-telemetry
```

## Mesures par perte en évaluation

La ligne `[MAPPO_EVAL_LOSS]` contient les clés suivantes :

- Corrélation : `update`, `campaign`, `case`, `seed`, `step`, `blue`.
- Attribution ECS inchangée : `source` (`unknown`, `blue_collision`, etc.).
- Cinématique : `speed_before`, `speed`, `accel_cmd`.
- Modèle de vol : `flight_model`, `min_speed`,
  `low_speed_threshold`, `fuel_remaining`, `fuel_capacity`.
- Indicateurs : `low_speed` et `fuel_empty` (0/1).

Les indicateurs sont évalués avec **les conditions exactes de
FlightModelSystem** sur l'état du pas où le mobile disparaît :

```cpp
low_speed = (2.F * speed) < min_speed;
fuel_empty = fuel < std::numeric_limits<float>::epsilon();
```

La plateforme `defender_car` définit `min_speed = 10 m/s` : le seuil
de destruction associé est donc **strictement inférieur à 5 m/s**.

**Important :** les indicateurs ne remplacent pas `source` et ne sont pas
des causes mutuellement exclusives. Une perte avec `source=unknown` et
`low_speed=1` fournit une preuve forte de l'activation de la condition de
vitesse du modèle de vol ; `low_speed=0` ne prouve pas à lui seul l'origine
d'une perte. Si les deux conditions sont vraies, les deux indicateurs valent 1.
La mesure intervient après l'évolution physique du pas, sans modifier son
ordre d'exécution.

## Exécution et exploitation

1. Utiliser la workflow `MAPPO training (Windows Release)`, scénario
   `blue-defense`, sur la branche contenant cet outillage.
2. Lire `Blue flight-model loss diagnostics` dans le résumé GitHub Actions.
3. Dans l'artefact `blue-mappo-telemetry`, consulter
   `mappo-losses.csv` et `mappo-training.log`.
4. Filtrer le CSV sur `source=unknown`, examiner `low_speed`,
   `fuel_empty`, les vitesses et `accel_cmd`.
5. Comparer plusieurs campagnes d'évaluation ; ne pas extrapoler les
   pertes d'évaluation à toutes les transitions d'entraînement.

## Frontières et validation

- Physique, vitesses, carburant, récompenses, actor, critic, checkpoints,
  observations, affectation de rôles et contrôleur rouge RLMission :
  **strictement inchangés**.
- Aucune allocation ou sortie supplémentaire dans les boucles PPO ;
  les lignes sont émises uniquement lors d'une destruction en évaluation.
- Parsing tolérant aux lignes de log tronquées ; CSV de diagnostic produit
  même si aucune perte n'est enregistrée.
- Tests unitaires Python couvrant vitesse trop faible, carburant vide,
  collision, données incomplètes et export CSV.
- Vérification des builds C++ et des contrôles CI à effectuer avant merge,
  conformément à `AGENTS.md`.

## Limite actuelle

Les pertes survenues **pendant l'entraînement** ne sont pas journalisées
individuellement : l'instrumentation couvre seulement les évaluations
déterministes et n'ajoute aucune charge aux épisodes de training.
