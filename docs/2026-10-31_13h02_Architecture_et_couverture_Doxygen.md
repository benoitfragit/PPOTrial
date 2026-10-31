# PPOTrial — architecture documentaire et couverture Doxygen

Date : 2026-10-31 13h02 (Europe/Paris)  
Révision du code analysée : `e003f0c5235057470db824b5d9bf12b23d6e64c8`.

## Périmètre

Référence des **165 fichiers C++** actuels du dépôt, comprenant **66 `.cpp`** et **99 `.hpp`**. Les changements portent sur `docs/doxygen` et sur les **commentaires Doxygen du code source** : 128 fichiers `.cpp`/`.hpp` reçoivent des annotations (443 descriptions de fonctions, 58 descriptions de types). Les 37 sources qui possédaient déjà une documentation suffisante sont conservées. **Aucune instruction C++, aucun test fonctionnel, JSON opérationnel, script CI ou checkpoint n'est modifié.**

## Diagnostic initial

Le Doxyfile désignait des anciens dossiers (`Agents`, `Entities`, etc.), n'extrayait pas tous les symboles (`EXTRACT_ALL=NO`), masquait l'API sans commentaires et ne montrait pas les corps des fonctions. Certaines pages JSON documentaient l'ancien scénario `TwoVsTwo` et un nombre obsolète de canaux.

## Architecture réelle

- `Main/` : application, exemples, exécution GUI/headless, suivi télémétrique.
- `Framework/RL/Core` : contrats et orchestration génériques, indépendante du domaine.
- `Framework/RL/PPO` et `Bindings` : PPO historique LibTorch.
- `Framework/Simulation` : moteur ECS, physique, règles de monde et snapshots.
- `Examples/Airspace` : RLMission et RLSNA historiques; RLMission rouge reste inchangé.
- `Examples/BlueDefense` : agent bleu MAPPO partagé RLTools pour P=1..9; jusqu'à N=10 adversaires rouges; formation, lidar, rewards, évaluation et checkpoints.
- `resources/settings` : scénarios, plates-formes, patrouille, entraînement et curriculum V1/V2.
- `tools/ci` et `.github/workflows` : entraînement Release et orchestration V2.

## Cycle d'un pas

Commandes -> règles BeforeStep -> reset si nécessaire -> choix de toutes les actions -> physique ECS une seule fois -> résolution des pertes -> retour de tous les agents -> AfterStep -> snapshot. Dans BlueDefense, `SetNewState` décompose les récompenses, enregistre P transitions puis déclenche éventuellement une mise à jour PPO.

Les 15 canaux MAPPO sont référencés dans `docs/doxygen/pages/blue-mappo-execution.md`. Le critère de neutralisation rouge est conditionné à N>0 et à l'absence de menaces différées; pour N=0, le terme de patrouille ne s'arrête pas sur une fausse « victoire zéro rouge » mais à horizon/échec.

## Génération

Sous Windows, lancer `generatedoc.bat`, ou `tools/generatedoc.ps1 -NoOpen`. Résultat attendu : `docs/generated/html/index.html`. Le graphe d'appel est optionnel et dépend de Graphviz. Les avertissements Doxygen sont exportés dans `docs/generated/doxygen-warnings.log`.

**Limite** : extraction exhaustive ne signifie pas que chaque fonction anciennement dépourvue de commentaire obtient automatiquement une spécification sémantique; la référence expose néanmoins sa signature et son implémentation, et fournit un audit des lacunes. La génération Doxygen et les builds C++ ne sont pas exécutés dans cet environnement.

## Vérification d'intégrité des sources

Les 165 sources ont été traitées individuellement par un outil temporaire d'insertion de commentaires. Pour chaque fichier annoté, le texte d'origine est reconstitué **octet pour octet** par retrait des blocs ajoutés; cette assertion a été vérifiée avant la création des objets Git. L'outil temporaire ne fait pas partie du diff final. Tous les nouveaux commentaires sont du type `/** ... */` et les sources historiques déjà commentées restent inchangées.

La PR est regroupée en **un seul commit** construit à partir de la branche `master`; aucun historique de commits intermédiaires n'est conservé dans la PR. La validation Doxygen / CMake / CTest reste à exécuter sur un runner équipé.
