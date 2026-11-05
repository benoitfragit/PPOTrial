# Documentation PPOTrial

Ce dossier sépare la documentation courante des sources Doxygen et des prompts.
Les anciennes notes datées, les comptes rendus de PR et les versions remplacées
ont été retirés : Git conserve leur historique.

## Commencer ici

- [Architecture visuelle des réseaux Blue](architecture/blue-network-architecture.md)
- [Adapter l'agent Blue](adapter-agent-blue.md) : entrées, sorties, réseaux,
  rewards, configuration et points d'entrée du code
- [Configurer PPOTrial](configuration-guide.md)
- [Lancer BlueDefense](blue-defense-example.md)

## Architecture et contrats

- [MAPPO à effectifs variables](mappo-variable-teams.md)
- [Séquenceur de patrouille à dix waypoints](architecture/blue-patrol-waypoint-sequencer.md)
- [Curriculum MAPPO V3 et observation V10](blue-mappo-v3.md)
- [Monde dédié à l'évaluation MAPPO](mappo-evaluation-world-architecture.md)
- [Contrats génériques des agents](agent-contracts.md)
- [Noyau RL générique](generic-rl-core.md)
- [Entités partagées du monde](shared-world-entities.md)
- [Catalogue des exemples](example-catalog.md)

## Exploitation

- [Entraînement MAPPO distant](remote-mappo-training.md)
- [Prompt d'implémentation des zones rouges et équipes P/N](prompts/red-zones-and-p-vs-n.md)

## Référence Doxygen

Les sources de la documentation Doxygen sont dans [`doxygen/`](doxygen/).
La sortie HTML locale `generated/` n'est pas versionnée. Depuis la racine du
dépôt, elle se régénère avec :

```powershell
tools/generatedoc.ps1 -NoOpen
```

