# AGENTS.md

Contrat de contribution pour PPOTrial. Objectif : du C++ explicite, modulaire,
testable et maintenable. Adapte des principes de `benoitfragit/MFDStudio/AGENTS.md`
(revision source `77ce8c0c0d916d803c46641feffc96d07728a20f`).

## Workflow

- Lire les fichiers et leurs appelants avant toute modification.
- Annoncer le perimetre concret et les incompatibilites previsibles.
- Travailler sur une branche dediee ; ne jamais modifier directement `master`.
- Privilegier les extractions locales, conserver les parties fonctionnelles.
- Ne pas modifier les algorithmes, parametres ou comportements au passage.
- Compiler, executer les tests pertinents, relire le diff complet et synchroniser
  les documents avant une PR. Indiquer exactement les validations impossibles.
- Ne pas merger automatiquement ; ne pas ecraser les changements d'autrui.

## Architecture et dependances

- `RL/include` et `RL/src` portent le contrat et le runtime generiques,
  sans LibTorch, EnTT, rendu, entites ou scenarios. `RL/torch` isole LibTorch.
- `RL/torch` contient le backend extrait `ppo_torch`. `Examples/Navigation2D`
  implemente l'environnement. `Main` orchestre l'application unique `PPO` ;
  `Renderer` conserve la presentation ImGui/raylib. Aucun chemin legacy, shim
  ou deuxieme application. Voir `docs/rl-contract.md`.
- Le framework ne doit connaitre aucun type metier. Un environnement produit
  observations, actions, rewards et metriques par le contrat public.
- Les adaptateurs dependent des contrats ; les contrats ne dependent jamais
  des adaptateurs. Une implementation distante implementera `IEnvironment`.
- Garder les conversions tensor et le device dans le backend LibTorch.
- Garder les protocoles Protobuf et GameNetworkingSockets hors du coeur.
- Ne pas dupliquer les regles de simulation entre entrainement et evaluation.
- L'exemple conserve un registre EnTT avec N entites (mobiles et zones), des
  identifiants stables et des controles par mobile. Ne jamais le reduire a une
  paire codee en dur. EnTT reste dans l'exemple et le rendu, jamais dans `RL/`.
- Les recompenses doivent rester decomposables et parametrables par mobile :
  eviter les coefficients non documentes disperses dans la logique de simulation.
- Tous les agents decident sur le meme etat avant l'avancement commun du monde.
  Les retours sont propres a chaque mobile ; seuls les learners attaches recoivent
  du feedback. Un reset global tronque les episodes non termines des autres agents.
- Aucun score specifique a un exemple dans un evaluateur generique.

## C++ et nommage

- C++17 standard, sans extensions compilateur ; RAII et const-correctness.
- Preferer la composition, `std::unique_ptr` pour une propriete exclusive,
  `std::shared_ptr` seulement pour un partage reel et documente.
- Pas de `new`/`delete` manuels hors adaptation encapsulee d'une API externe.
- Une responsabilite par classe ; API publique minimale et documentee.
- Details non templates dans les `.cpp`. Helpers nommes plutot que lambdas
  complexes, captures larges ou astuces de metaprogrammation.
- Pas de casts dangereux, suppressions artificielles de variables inutilisees,
  noms factices, classes locales ou membres `mutable` sans justification.
- Dans `RL/` : namespace `ppo::rl`, types en PascalCase, fonctions et champs
  en snake_case, membres prives avec suffixe `_`. Conserver les conventions
  historiques dans les fichiers existants ; pas de renommage massif.
- Documenter les preconditions et exceptions. `noexcept` seulement si garanti.
- Verifier tailles, bornes, overflow et NaN/Inf aux frontieres publiques.

## CMake et dependances

- Cibles explicites, sources explicites, proprietes au niveau des targets.
- `PRIVATE` pour les details ; `PUBLIC` seulement pour une exigence de l'API.
- Pas de chemin machine, flags globaux ou choix de generateur dans une librairie.
- Le contrat doit se compiler et se tester sans telecharger LibTorch ou le rendu.
- Ajouter une dependance seulement si necessaire ; versionner ses references.
- Ne pas modifier les versions de dependances dans une extraction structurelle.

## LibTorch et PPO

- Conserver les valeurs par defaut, distributions, log-probabilites, clipping,
  normalisations et formats de checkpoints lors d'un refactoring.
- Expliciter shape, dtype, device et ownership de chaque buffer ; aucune vue
  `from_blob` ne doit survivre a sa memoire source.
- L'inference et l'evaluation ne doivent pas construire de graphe de gradients.
- Ne pas partager une policy mutable entre optimisation et evaluation.
- Distinguer `terminated` et `truncated`. Pour un futur raccordement GAE,
  une troncature permet le bootstrap sur l'observation finale mais coupe la
  recurrence entre episodes ; une terminaison coupe aussi le bootstrap.
- Ne pas substituer l'observation du reset a l'observation finale.
- Toute evolution numerique exige ses propres tests ; ne pas la dissimuler
  dans une migration d'API. Ne pas lancer d'entrainement sans demande.

## Tests et revue

- Tests rapides, deterministes, independants des ressources graphiques.
- Tester les contrats invalides et les limites, pas seulement le chemin heureux.
- Une correction de bug doit avoir un test de non-regression pertinent.
- Verifier que les assertions de test fonctionnent aussi en Release.
- Relire inclusions, dependances, ownership, boucles, exceptions et documentation.
- Examiner dans le diff : casts, lambdas complexes, `mutable`, `TODO`, `FIXME`,
  allocations manuelles et templates inutiles. Justifier les exceptions.
- Une revue distingue faits confirmes et risques ; pour un finding : severite,
  fichier, fonction, scenario, cause, impact, correction et test attendu.
- Pour ce contrat : `cmake -S RL -B build/rl`,
  `cmake --build build/rl`, `ctest --test-dir build/rl --output-on-failure`.
- Le desktop exige LibTorch et raylib/ImGui/ImPlot. Ne pas annoncer sa
  validation a partir du seul build du contrat ou d'un controle syntaxique.
- Le portage Navigation2D remplace explicitement les anciens noms metier, et non
  leurs capacites : preservation fonctionnelle de la simulation 2D, perception,
  controle multi-agent, initialisations, evaluation et rendu. Toute fonction
  deliberement simplifiee doit etre repertoriee avec un test de non-regression
  et l'accord explicite du mainteneur. Ne pas reintroduire les anciens noms
  d'entites, ni de shim pour les checkpoints neuronaux historiques.

## Definition de termine

- Le perimetre livre est explicite, l'API minimale et le code relu.
- Les tests pertinents passent et leurs limites sont indiquees.
- Documentation, CMake et code sont coherents.
- La PR distingue les travaux livres des etapes futures.
