# Entraînement de la patrouille bleue — MAPPO P/N

Ce document décrit uniquement le **MAPPO généralisé**.

Le pilote à deux Bleus, les anciennes interfaces d'entraînement et le scénario
`two-vs-two` ont été retirés. L'entrée actuelle du sélecteur est
`blue-defense` ; le scénario de patrouille sans Rouge reste `blue-patrol`.
Ces deux exemples utilisent une unique instance de
`GeneralizedPatrolMappoAgent`, P compris entre 1 et 9 et N entre 0 et 10.

Pour les fichiers de configuration, les observations, les rewards et le pilotage
de l'IHM, consulter [le guide BlueDefense](blue-defense-example.md).
Pour le critic centralisé, les checkpoints et la reprise de l'entraînement,
consulter [l'architecture MAPPO P/N](mappo-variable-teams.md).

Les Rouges continuent à employer RLMission production sans modification.
Les anciens lecteurs de checkpoints restent uniquement pour la migration
des poids V4/V7. Aucun entraînement fixed-2 n'est compilé.

## Validation requise

La CI contrôle la compilation Windows/Linux, les tests P/N, le snapshot à
10 canaux, le chargement des deux exemples et le démarrage effectif via Play.
La convergence en formation et en défense nécessite une campagne de tests
multi-seed distincte de la compilation.
