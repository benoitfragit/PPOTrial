# Documentation PPOTrial

Ce dossier rassemble la documentation courante et les sources Doxygen. Git
conserve l'historique des documents remplacés.

Le dépôt présente trois démonstrateurs : **RLMission**, PPO en 2D ; **Blue
MAPPO**, MA-PPO en 2D ; et **FlightControl3D**, PPO en 3D.

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
- [Conception PPOKit : framework commun aux trois exemples, backend rl_tools, codegen et studio](architecture/ppokit-framework-design.md)
- [Contrats génériques des agents](agent-contracts.md)
- [Noyau RL générique](generic-rl-core.md)
- [Entités partagées du monde](shared-world-entities.md)

## Exploitation

- [Entraînement MAPPO distant](remote-mappo-training.md)

## Référence Doxygen

Les sources de la documentation Doxygen sont dans [`doxygen/`](doxygen/).
La sortie HTML locale `generated/` n'est pas versionnée. Depuis la racine du
dépôt, elle se régénère avec :

```powershell
tools/generatedoc.ps1 -NoOpen
```

