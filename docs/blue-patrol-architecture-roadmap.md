# Architecture actuelle de patrouille et défense bleue

Le projet a convergé vers une architecture MAPPO **P/N unique** ; les
anciens contrôleurs à deux Bleus et les outils de simulation dédiés ont été
supprimés du build.

## Modules

- `Examples/BlueDefense/Defenders/Core` : géométrie d'hippodrome, formation,
  capacité des équipes, observation et récompenses coopératives.
- `Examples/BlueDefense/Defenders/Training` : acteur partagé, attention masquée,
  critic centralisé conditionné par agent et optimiseur MAPPO.
- `Examples/BlueDefense/Defenders/Runtime` : contrôleur ECS unique,
  adaptation des observations et des récompenses, liaison avec l'IHM.
- `Examples/BlueDefense/Defenders/Evaluation` : campagnes déterministes
  et métriques de validation.
- `Examples/BlueDefense` : génération des P Bleus et N Rouges, initialisation,
  zones interdites, collisions et scénarios.
- `Examples/src/examplecatalog.cpp` : déclaration des cartes du sélecteur IHM.

Une même scène `BlueDefense.json` alimente `blue-defense` pour l'entraînement
et `blue-defense-inference` pour l'exécution déterministe de l'acteur de
production. L'acteur dispose d'entrées masquées indépendantes du nombre
d'agents. Le learner gère simultanément toutes les transitions Bleues dans le
seul mode entraînement.

La migration V4/V7 reste disponible en lecture mais ne constitue pas
un second contrôleur.

Pour la configuration et la prise en main, consulter
[BlueDefense — MAPPO P/N](blue-defense-example.md) et
[MAPPO à effectifs variables](mappo-variable-teams.md).

## Reste à démontrer

Les tests déterministes valident les contrats logiciels ; les métriques de
convergence sur plusieurs seeds et compositions P/N restent une validation
d'apprentissage à conduire séparément.
