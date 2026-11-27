# PPOKit — audit d’intégration du 25 novembre 2026

> Cet audit conserve l’état et les limites observés le 25 novembre. L’inventaire
> d’appelants et la finalisation architecturale du 27 novembre sont dans
> [`ppokit-mutualization-final-2026-11-27.md`](../ppokit-mutualization-final-2026-11-27.md).

L’inférence et les campagnes d’atterrissage ont été comparées sur les archives réelles du T‑6 et du PC‑7. La sauvegarde des tenseurs est exacte pour les trois checkpoints. La parité comportementale globale de la branche reste ouverte, en particulier pour les modèles Airspace et BlueDefense.

## Références

| Référence | Identifiant | Usage |
|---|---|---|
| Deuxième audit | `2b4372895e31be671fafdb7dbdf5deb0305cc0c8` | Point de départ demandé |
| Intégration lue avant correction | `42ccd52ae9be0aee7fb2df8a6a526ae77d2908b1` | Incluait déjà la comparaison d’inférence T‑6 et le contrat d’avion |
| Master historique | `dfeb2b43a857c35b79f575dd52184c199fc0dcb6` | Interception à 2 000 m, dernier niveau 11, BEST T‑6 340 |
| Master après la PR 112 | `c8282b8b66de55bc31880a508d5ecf1e29d938b6` | Interception à 1 200 m, gate supplémentaire `runway-heading-offset`, dernier niveau 12, BEST 870 |
| Dernier master vérifié | `b5ea31967f16d5fb2390268648715da5c6fe1b10` | La PR 113 modifie uniquement le test de départ sur la dernière gate. Poids, scénarios, physique et guidage sont les mêmes que pour `c8282b8`. Son test renforcé est repris sur la branche. |
| Intégration | `design/ppokit-framework`, PR 105 | Branche rebasée sur `b5ea319`. Les changements de guidage et de gate proviennent de la PR 112 |

Les corrections déjà présentes ont été conservées. Les commits `c7d94de`, `5bb86b7` et `00c74f1` concernaient les masques historiques, le digest des BEST et la persistance de l’état d’environnement. Les commits `2b43728`, `a01d1c1` et `42ccd52` concernaient respectivement le branchement du learner RLSNA, l’inférence T‑6 et la distinction T‑6/PC‑7.

## FlightControl3D

Deux exécutables indépendants ont piloté deux simulateurs. Le côté historique a été compilé contre les bibliothèques de chaque SHA de master. Le côté PPOKit est passé par `ppokit-run eval` et ses DLL de projet. Chaque campagne comprend vingt seeds successives. La première série reproduit la sélection historique. La seconde commence à 90001. Le résultat est issu de l’épisode réellement simulé, jusqu’à sa terminaison ou sa limite de temps.

| Checkpoint | Guidage / niveau | Seeds | Arrêts master | Arrêts PPOKit | Pas par moteur | Épisodes avec un écart de durée, d’issue ou de retour |
|---|---|---|---:|---:|---:|---:|
| T‑6 340 | 2 000 m / 11 | 4342–4361 | 19/20 | 19/20 | 43 730 | 0 |
| T‑6 340 | 2 000 m / 11 | 90001–90020 | 18/20 | 18/20 | 48 378 | 0 |
| PC‑7 610 | 2 000 m / 11 | 4342–4361 | 19/20 | 19/20 | 29 331 | 0 |
| PC‑7 610 | 2 000 m / 11 | 90001–90020 | 17/20 | 17/20 | 31 146 | 0 |
| T‑6 870 | 1 200 m / 12 | 4342–4361 | 19/20 | 19/20 | 38 905 | 0 |
| T‑6 870 | 1 200 m / 12 | 90001–90020 | 20/20 | 20/20 | 31 429 | 0 |
| PC‑7 610, évaluation explicite sur le nouveau profil | 1 200 m / 12 | 4342–4361 | 19/20 | 19/20 | 30 639 | 0 |
| PC‑7 610, évaluation explicite sur le nouveau profil | 1 200 m / 12 | 90001–90020 | 18/20 | 18/20 | 31 332 | 0 |

Sur ces 160 paires, les distances d’arrêt et de toucher des événements effectivement survenus ont également un écart maximal nul sur ce build Linux. Pour un événement absent, le CSV préserve la valeur brute historique et le champ vide du rapport PPOKit. Un contact hors critères pendant un crash peut laisser une position dans le simulateur historique. Le PC‑7 reste un checkpoint du run 45460292649, y compris dans l’évaluation explicite au niveau 12. Aucun poids d’archive n’a été réentraîné.

| Archive | SHA‑256 du V3 historique | Âges Adam V3 acteur / critique | Pas Adam PPOKit acteur / critique | Niveau source |
|---|---|---:|---:|---:|
| T‑6 340 | `42d56137c19630f4420c5270639ddd46a33b97e765a81db9f1f246ab1d50ca72` | 10850 / 10881 | 10849 / 10880 | 11 |
| T‑6 870 | `d7bff9e3249f03facc237d22f1b66d4bba62259210c0cbf640b1b80644f4a4e8` | 27773 / 27841 | 27772 / 27840 | 12 |
| PC‑7 610 | `5cc6337bdd215701c22f28536a796d6aaeb3a294a6083d101290d15587f1a2d4` | 19490 / 19521 | 19489 / 19520 | 11 |

Pour chaque archive, 15 tenseurs d’acteur et de critique et leurs deux moments donnent 45 comparaisons. Tous les éléments correspondent bit à bit après lecture en float32. Le convertisseur sépare les cinq lignes gaussiennes et la ligne de train, et applique la même séparation aux biais et aux moments. La sixième case du log-std historique correspond à la sortie Bernoulli sans échelle. Sa valeur et ses deux moments valent exactement zéro dans les trois archives V3. Les âges RLtools commencent à un, le compteur PPOKit enregistre les pas appliqués. Le test reconstruit l’état MT19937 complet et vérifie 1 248 tirages successifs. Le V3 et le fichier de niveau ne contiennent ni la fenêtre complète de promotion ni le générateur des futurs départs de ce run historique.

La récupération PC‑7 a utilisé l’artefact GitHub `14960752977`, SHA‑256 `1c1c99c63f33893847271d55c64f810b705f6770c4345695140eb8eda7190c60`. Son manifeste désigne `flight_610_0b6a880567ae4791.txt`. Le moteur historique a effectivement chargé ces poids. Le projet PC‑7 livré dans le dépôt conserve son démarrage vierge. Le paquet historique livré avec l’audit contient le projet, le scénario d’origine et les fichiers convertis.

| Génération | Tenseurs avant update | Une update PPO figée | Épisodes indépendants |
|---|---|---|---|
| T‑6 340 | [Par tenseur](t6-tensors.csv) | [Par tenseur](t6-update.csv) | [40 résultats](t6-episodes.csv) |
| T‑6 870 | [Par tenseur](t6-870-tensors.csv) | [Par tenseur](t6-870-update.csv) | [40 résultats](t6-870-episodes.csv) |
| PC‑7 610 | [Par tenseur](pc7-tensors.csv) | [Par tenseur](pc7-update.csv) | [40 historiques](pc7-episodes.csv) · [40 sur master actuel](pc7-610-current-master-episodes.csv) |

## Audit mathématique

Le test figé charge séparément les deux moteurs au même état, construit 64 transitions de valeurs et d’actions représentables, inclut une terminaison et une truncation, puis exécute quatre epochs de minibatches de 16 avec les réglages de l’avion. Les deux moteurs effectuent au moins un pas acteur et seize pas critique. Les sorties de cet essai sont des copies temporaires. La comparaison porte sur les poids et moments résultants, l’âge réel, l’état aléatoire et le numéro d’update. Les tailles du rollout et des minibatches sont réduites pour le test.

| Après l’update | Max absolu poids | Max absolu premier moment | Max absolu second moment |
|---|---:|---:|---:|
| T‑6 340 → 341 | 5,96 × 10⁻⁸ | 2,62 × 10⁻⁶ | 3,81 × 10⁻⁶ |
| T‑6 870 → 871 | 0 | 0 | 0 |
| PC‑7 610 → 611 | 0 | 0 | 0 |

Les maxima relatifs sont calculés élément par élément par rapport à la valeur historique et figurent dans chaque CSV. Sur le T‑6 340, le premier moment atteint 0,00707214 en relatif près de zéro. Les traces de collecte et les poids importés sont identiques avant l’update. Les différences apparaissent dans le calcul numérique des gradients et de la mise à jour par les deux backends. Cet essai ne démontre pas l’identité sur tous les rollouts, paramètres ou compilateurs.

| Sujet | Contrôle et résultat | Limite |
|---|---|---|
| GAE, bootstrap, frontières | Le dernier pas utilise la valeur de l’observation suivante. Un état absorbant annule le bootstrap. Une truncation l’autorise et interrompt la récursion. Les tests GAE, buffers d’équipe et le rollout figé passent. | La parité de tous les canaux de récompense des applications reste distincte. |
| Ratio, clipping, KL | Différences finies dans les quatre cas signe de l’avantage / saturation. Le gradient de la partie exponentielle est nul au-delà de sa garde numérique. Le garde KL peut sauter l’acteur pendant que le critique continue. | Une comparaison inter-moteurs sur le BEST MAPPO reste ouverte. |
| Normes, Adam, entropie | Validation des états restaurés, du pas Adam candidat, des gradients et de la finitude. Les tests exercent le masquage, la dispersion, le KL et les gradients du graphe. | Une erreur dans le dernier minibatch restaure l’update complet du learner de graphe, y compris le tronc partagé. Le test compare ensuite une relance au témoin. |
| Actions | Actions latentes utilisées pour les log-probabilités, commandes physiques comparées après conversion. Frein et train sous masque. Le seuil au logit zéro est explicite dans le contrat de l’avion. | Les commandes des applications Airspace et BlueDefense ne sont pas certifiées par ces vols. |
| Réseaux, paramètres partagés | Tests par différences finies sur les entrées et poids, couches denses, encodeurs, attention et pooling. Essais des ensembles vides et masques. Le test de reprise du tronc partagé reproduit le prochain update sur la tâche de test. | Un test du bloc générique ne qualifie pas le comportement d’un modèle déployé. |

## Registre des anomalies

Les chemins commençant par `src/`, `include/`, `studio/` ou `tools/ppokit_run` dans cette table sont relatifs à `PPOKit/`. Les SHA des premiers contrôles identifient les commits d’origine. Les commits rejoués indiquent cette origine dans `Rebased-from`. Les entrées ouvertes ne sont pas comptées parmi les corrections.

| ID | Fichier et ligne de référence | Observation, cause, impact | Correction et essai déterministe | Résultat / commit |
|---|---|---|---|---|
| P0‑04 | `Examples/FlightControl3D/Tools/flight_checkpoint_import.cpp:346`, `tests/master_checkpoint_parity_tests.cpp:210` dans le même exemple | L’import historique réinitialisait le RNG. La parité de l’acteur masquait la divergence de reprise. Le PC‑7 n’avait aucune référence sur master. | Restaurer MT19937, séparer les têtes et leurs moments, vérifier V3 et curriculum, sorties et commandes. Trois archives vérifiées. | CSV ci-dessus. `07e30a81`, `f3b92fbb`, `d15e3b2f`. La fenêtre de promotion historique complète n’est pas dans les archives. |
| P0‑05 / P2‑05 | `Examples/Airspace/Agents/Bindings/data/trainingagent.hpp:78`, `Agents/Mission/targetselector.cpp:588`, `Agents/Interception/interception.cpp:154` | La fin de rollout est constante sur la branche. Master évalue la position dans le rollout à `Framework/RL/Bindings/data/trainingagent_def.hpp:397`. Les deux branches de récompense n’ont plus la même accessibilité. | Constat statique. Les tests existants d’Airspace ne qualifient pas la parité de cette récompense. | **Ouvert**, aucune correction de la récompense liée aux engagements armés. |
| P0‑06 | `src/schema/task_schema.cpp:769`, `src/artifacts/artifact.cpp:222`, `src/network/graph_network_library.cpp:55`, `Examples/FlightControl3D/Core/flight_config.cpp:800` | Deux contrats de même forme pouvaient échanger des modèles malgré des mappings ou masques différents. Le V3 ne versionnait pas l’anticipation latérale. | Identité sémantique versionnée, contrôle à la DLL réseau, migration explicite, contrats physiques distincts pour les deux guidages. Tests des mutations à forme identique et des chargements croisés. | `9c77b34e`, `d15e3b2f`. Les modules natifs d’environnement et de réseau utilisés par `ppokit-run` exportent désormais leur contrat compilé. Un hash différent ou un export manquant interrompt le chargement. `15326a07`. |
| P0‑07 | `tools/ppokit_run.cpp:722`, `src/eval/best_store.cpp:146` | Un ancien BEST pouvait hériter du contexte courant, et une modification de poids des rewards pouvait être perdue à l’évaluation. | Le BEST conserve état et poids effectifs. L’évaluation valide update/digest, accepte un état explicite et enregistre seeds, portée de simulation, empreinte de suite, scénario et paramètres. Test d’un BEST antérieur, checkpoint récent et contexte forcé. | Tests runner et BEST réussis. `5d612cba`, `d08ebbe6`. Les anciens BEST sans contexte nécessitent une sélection explicite. Pour la génération courante, `play` et `eval` vérifient le SHA-256 et reprennent l’état publié avec le fichier sélectionné, y compris si l’acteur est lu depuis `latest.ppokit`. `18c6022a`, `638de7f3`. |
| P0‑08 | `src/runtime/writer_lock.cpp:27`, `src/runtime/run_state.cpp:159`, `tools/ppokit_run.cpp:1096` | Deux projets partageant des sorties pouvaient écraser modèles et journaux. Un fresh pouvait emporter la référence déployée. | Verrous OS sur emplacements canoniques, politique de travail distincte, Stop privé, archivage avec restauration en cas d’erreur. Deux vrais runners et deux projets avec chemins relatifs aliasés. | `27821197`, `98b17bd8`. La panne OS empêchant aussi la restauration d’un fichier d’archive nécessite une récupération depuis le chemin signalé. |
| P1‑05 | `src/editor/process.cpp:190`, `studio/project_window.cpp:281`, `studio/studio_app.cpp:71` | Un descendant gardant le pipe ouvert empêchait le lecteur de finir. | Lecture interruptible, drainage borné, confinement des descendants Windows/Linux, attente coopérative de sauvegarde à la fermeture. Le helper laisse un descendant hériter du pipe, puis teste la relance et le code de sortie. | CTest Linux réussi, test commun Windows. `239a4fc9`. Une fermeture de fenêtre native pendant une évaluation réelle n’a pas été jouée manuellement. |
| P1‑07 | `include/ppokit/graph/graph_learner.hpp:197` | Un update pouvait consommer un buffer vide ou incomplet. | Contrat contrôlé avant GAE et avant toute mutation. Test vide, partiel, plein et comparaison du pas suivant. | Réussi. `ec95271c`. |
| P1‑08 | `include/ppokit/graph/adam.hpp:64` | Un moment mal formé ou négatif pouvait être installé avant l’échec du critique. | Taille, finitude, second moment et âge validés. Les deux états sont contrôlés avant installation. Injection d’un dernier tenseur critic corrompu. | Réussi. `060f925c`. |
| P1‑09 | `include/ppokit/graph/adam.hpp:131` | Un gradient non fini ou un dépassement de la norme pouvait contaminer les poids. | Le pas candidat est préparé dans des buffers réutilisés, contrôlé intégralement puis appliqué. Tests NaN, Inf et valeurs extrêmes. | Réussi au niveau du pas Adam (`af923a03`) et de l’update complet (`35d46dda`). Échec injecté après un premier minibatch, avec comparaison des poids, moments, compteurs et RNG avant la relance. |
| P1‑10 | `include/ppokit/core/ppo_settings.hpp:82`, `src/network/graph_network_library.cpp:55` | Les accès natifs et les frontières JSON / DLL ne contrôlaient pas tous les mêmes domaines. | Catalogue commun, finitude, intervalles et ordre des log-std. Essais natifs, configuration et DLL de graphe. | Réussi. `5598c04c`. |
| P1‑11 | `include/ppokit/core/ppo_surrogate.hpp:56` | Le gradient traversait la saturation numérique alors que l’objectif y est constant. | Dérivée alignée sur l’objectif dans la zone saturée. Différences finies pour les deux signes d’avantage et de saturation. | Réussi. `ad367121`. |
| P1‑12 | `include/ppokit/graph/graph_learner.hpp:171` | Le logit de train exactement nul ne donnait pas la même commande déterministe. | Règle de départage portée par la tâche de vol. Tests sur logits négatif, positif, nul et composant inactif. | Réussi. `7f27dfb7`. |
| P1‑13 | `src/runtime/publication.cpp:56`, `tools/ppokit_run.cpp:1090` | Une interruption entre les sauvegardes pouvait mélanger deux générations. | Préparation commune, décision par renommage, publication du manifeste, rejeu idempotent. Interruption pendant la préparation et à cinq états d’installation, refus en lecture et vérification des digests. | Réussi pour interruption de processus. `98b17bd8`. La résistance à une coupure électrique sans flush du volume n’a pas été éprouvée. |
| P1‑14 | `tools/ppokit_run.cpp:1174`, `Examples/FlightControl3D/Environment/flight_environment_library.cpp:266` | Un Stop pendant la collecte pouvait sauver une policy avant intégration de ses transitions. JSBSim ne sérialise pas ici son intégrateur en vol. | Arrêt à l’update complet, prochain reset différé. La reprise crée un nouvel épisode. Tests du prochain départ, du curriculum, de l’arrêt et du learner sur tâche déterministe. | `6764827d`. La reprise à l’identique en plein épisode JSBSim reste ouverte. |
| P2‑01 | `Examples/FlightControl3D/Training/flight_learner.hpp:106`, `Runtime/flight_session.hpp:161`, `include/ppokit/graph/graph_learner.hpp:82` | Des moteurs et clients ont des responsabilités proches. | Analyse des liaisons CMake, workflows et points d’entrée. Les références de validation ont des appelants réels. | Conservés, voir inventaire ci-dessous. L’examen de tous les moteurs MAPPO reste partiel. |
| P2‑02 / P2‑07 | `PPOKit/tests/repository_projects_tests.cpp:178`, `Examples/FlightControl3D/Core/flight_config.cpp:358` | Le projet et le scénario historique répètent des réglages. La suite pouvait masquer une différence de durée de vie de JSBSim. | Test des onze clés PPO communes des modèles d’avion, contrat de physique et audit de la campagne effective. | Tests des projets réussis. `cb816f64`, `d08ebbe6`. L’audit de tous les paramètres du Studio et des DLL tierces reste ouvert. |
| P2‑03 | `tools/ppokit_artifact.cpp:83`, `src/artifacts/artifact.cpp:222`, `Examples/Airspace/Agents/Interception/interception.cpp:171` | Des références historiques et des chemins de compatibilité coexistent. | Migration des en-têtes avec tâche/graphe source, refus de l’identité incompatible. Les lecteurs historiques encore appelés ont été gardés. | `9c77b34e`. L’audit de tous les launchers et initialisations de secours reste ouvert. |
| P2‑03, copie sous Windows | `tools/ppokit_artifact.cpp`, `src/artifacts/artifact.cpp`, `Examples/Airspace/Tools/airspace_checkpoint_import.cpp` | La réécriture de l’en-tête ouvrait la sortie en mode texte. Sous Windows, chaque LF du corps recevait un CR supplémentaire. | Migration, transfert de contrat et remplacement d’en-tête utilisent une sortie binaire. Le test de migration vérifie séparément des archives LF et CRLF. Le test de transfert compare tous les octets après l’en-tête. | Reproduit dans les jobs Windows de la PR. Les deux formats ont aussi été vérifiés sur un artefact Airspace en Debug et Release. |
| P2‑04 | `include/ppokit/graph/graph.hpp:50`, `include/ppokit/graph/policy_layout.hpp:103`, `src/schema/task_schema.cpp:181` | Une multiplication ou l’addition d’un offset pouvait reboucler et accepter une vue hors limites. | Contrôles avant multiplication et liaison, cas limites de taille et d’offset. | Réussi. `6594ab5b`. La revue exhaustive de toutes les allocations, ABI et extensions reste ouverte. |
| P2‑06 | `Examples/BlueDefense/Environment/blue_environment.cpp:254` | L’argument seed du constructeur est ignoré au profit de la configuration Blue. La réinitialisation d’évaluation reçoit séparément sa seed à la ligne 272. | Deux sources identifiées. Le changement de priorité modifierait le flux historique et demande une décision de contrat. | **Ouvert**. |
| Évaluation PC‑7, P1 | `tools/ppokit_run.cpp:722`, `src/eval/evaluation.cpp:381` | Recréer JSBSim à chaque seed déplaçait l’arrêt du PC‑7 sur la seed 90004 de 5,93 m et de huit pas. L’ancien moteur garde une instance pendant la campagne. | Portée de campagne explicite pour les deux avions et persistée par l’éditeur. Test de schéma, refus pour les factories incompatibles, 160 comparaisons d’épisodes. | Réussi. `cb816f64`, `f3b92fbb`, `d15e3b2f`. |

## Matrice de non-régression

« Référence » désigne ici les poids du fichier source réellement lu. « Sondes » désigne les observations de test communes aux implémentations. Une campagne indépendante couvre les seeds annoncées, sans extrapolation aux autres scénarios.

| Modèle | Structure | Poids | Sorties | Commandes physiques | Rewards | Comportement | Reprise | Adam | RNG |
|---|---|---|---|---|---|---|---|---|---|
| RLMission prod | Graphe chargé | 14 tenseurs acteur exacts depuis son `model.pt` | Sondes d’inférence | Ouvert | P0‑05 ouvert | Ouvert | Checkpoint de reprise fourni issu d’eval | 7453 dans prod, 8197 dans le checkpoint fourni | Historique non démontré |
| RLMission eval | Graphe chargé | Acteur et critique exacts | Sondes d’inférence | Ouvert | P0‑05 ouvert | Ouvert | État des poids et Adam vérifié | Moments exacts, pas 8197 | Historique non démontré |
| RLSNA prod | Graphe chargé | 8 tenseurs acteur exacts depuis son `model.pt` | Sondes d’inférence | Ouvert | P0‑05 ouvert | Ouvert | Checkpoint de reprise fourni issu d’eval | 3187 dans prod, 3340 dans le checkpoint fourni | Historique non démontré |
| RLSNA eval | Graphe chargé | Acteur et critique exacts | Sondes d’inférence | Ouvert | P0‑05 ouvert | Ouvert | État des poids et Adam vérifié | Moments exacts, pas 3340 | Historique non démontré |
| FlightControl3D T‑6 340 / 870 | Exacte sur les contrats versionnés | Exacts | Sondes, acteur et critique sur les vols historiques | Comparées après `ToControls` | Retours identiques sur les campagnes, décomposition par canal ouverte | 40 paires par modèle | Poids et optimiseur vérifiés, nouvel épisode à la reprise | Exact à l’import, voir update figé | Learner vérifié, ancien flux environnement incomplet |
| FlightControl3D PC‑7 610 | Exacte | Exacts | Sondes, acteur et critique sur le profil historique | Comparées après `ToControls` sur le profil historique | Retours identiques sur les campagnes, décomposition par canal ouverte | 40 paires historiques et 40 sous le nouveau profil | Même contrat que le T‑6 | Exact à l’import et sur l’update figé testé | Learner vérifié, ancien flux environnement incomplet |
| BlueDefense MAPPO 4v4 | V10/V7 et artefacts PPOKit rapprochés | Acteur et critique exacts | Parité sur le BEST en exécution ouverte | Ouvert | Ouvert | Évaluation indépendante ouverte | État sérialisé vérifié, reprise déroulée ouverte | Paramètres, âges et deux moments exacts | 624 mots identiques dans les deux archives |

Les comparaisons Airspace ont lu directement les quatre `model.pt` du master actuel et les fichiers PPOKit du dépôt. Pour chaque version, la policy d’inférence correspond à son acteur. Chaque agent possède un seul artefact d’entraînement, issu de sa variante eval. Les nombres de pas différents indiquent précisément pourquoi il ne peut pas représenter aussi l’optimiseur prod. [Détail Airspace](airspace-artifacts.csv)

### BEST BlueDefense

Le véritable BEST 4v4 de production a été retrouvé dans le **run 101**, artefact `13699304800`, dossier `best/4v4`, update **5250**. Le manifeste historique rapporte 100 % et un score de 80,1073. Ces deux valeurs proviennent de l’archive. Les SHA‑256 bruts sont `cfd2f2c1b221cc61a6bfd2d5a7085da0fe9a33d67ca946c4cf96ce9494afb931` pour l’acteur et `24d84192d0e3f3468d174466cf741bd0ffb1a2238445abe26ff802f7dc3a524e` pour le checkpoint. Les deux fichiers correspondent intégralement à master après normalisation CRLF/LF. Les 15 tenseurs de la policy prod PPOKit représentent 37 152 scalaires identiques. Le checkpoint latest PPOKit conserve l’acteur, les six tenseurs du critique, les deux moments de chaque tenseur, les réglages Adam, les deux âges 99361, l’update et le flux MT enregistré. [Run 101](https://github.com/benoitfragit/PPOTrial/actions/runs/42639806607) · [Comparaison de l’état](blue-production-integrity.csv)

Le workflow de validation historique de master utilise aussi, par défaut, le **run 73**, artefact `13384201451`, dossier `best/2v2`, update **1140**. L’acteur porte l’empreinte `bae468f5ab9f36c60b9c6b895cd7e6693675bdc0c441f3a36f46fc8df9f70cc3`. Ses paramètres diffèrent de ceux du 4v4 de production. La campagne indépendante multi-agent demandée, y compris les transitions de rôles, collisions et engagements, reste ouverte. Aucune mesure de succès, de perte ou de collision PPOKit n’est déduite des scores historiques. [Run 73](https://github.com/benoitfragit/PPOTrial/actions/runs/41948019589)

### Responsabilités et compatibilité

| Entrée | Chemin effectif et décision |
|---|---|
| Studio et CI Flight | Les deux workflows d’entraînement Flight chargent le projet avec `ppokit-run`. Le processus héberge les DLL environnement et réseau, le journal, les campagnes et la publication. |
| `GraphLearner` | Exécute les graphes de la DLL réseau, le PPO et les têtes. Les paramètres PPO effectifs du réseau de graphe viennent de `training.json`. |
| `FlightSession`, `FlightLearner`, `FlightControlHeadless` | Clients historiques encore compilés. Ils portent le chemin RLtools, la lecture V3 et la référence exécutable de comparaison. Une suppression compromettrait la vérification du portage. |
| MAPPO natif | L’interface équipe délègue la collecte et l’update au moteur d’équipe. Une mutualisation avec le learner mono-agent demanderait des preuves distinctes sur les rôles, les masques et les paramètres partagés. |
| `project.json`, `task.json`, `network.json` | Respectivement points d’entrée / chemins, contrat des observations-actions-rewards, graphes exécutés et observateurs. Le `graph.json` source Flight génère aussi les réseaux RLtools de référence. Les projets ont des tests d’équivalence avec ces sources. |
| `scenario.json`, `training.json`, `eval/*.json` | Physique, guidage, niveaux et paramètres du moteur historique dans le scénario. Réglages du learner de graphe dans training. Seeds, nombre d’épisodes, critères et durée de vie du simulateur dans la suite. Une édition locale des deux représentations doit garder leur provenance explicite. |
| Modèles et fallbacks | `policy.ppokit` sert l’inférence, `latest.ppokit` la reprise. Le chemin de policy déployée peut être distinct du chemin de publication. Les V3, V7, V10, `policy.txt` et archives Torch ont des lecteurs ou des usages d’oracle vérifiables. Les factories d’Airspace possèdent encore des branches de compatibilité. |
| Code mort retiré | `Adam::SetLearningRate` était un membre interne du header de l’optimiseur du graphe. La recherche de ses références dans les sources, les tests et les tables ABI ne retourne aucun appelant. Les réglages du constructeur restent utilisés. Aucun export de DLL n’a été supprimé. |
| Maintenance ouverte | `src/schema/block_graph_schema.cpp`, `studio/graph_editor_view.cpp`, `tools/ppokit_run.cpp` et `include/ppokit/graph/graph_learner.hpp` restent des unités importantes. Une absence d’appelant C++ ne démontre pas qu’un symbole chargé par une DLL est inutilisé. Aucun découpage spéculatif n’a été appliqué. |

### Validation et reproduction

Les builds locaux utilisent C++17, GCC 14, Ninja, sans CUDA, dans `build/debug` et `build/release`. Le périmètre compilé est headless, avec bibliothèques de projets, tests de processus, FlightControl, Airspace et BlueDefense. Release compte 96 CTest inscrits, 95 exécutés avec succès. L’unique skip est `FlightControl_golden_trace_tests`, dont l’empreinte est gelée sur une autre plateforme. Le contrôle séparé sous GCC 13 avec la libm Ubuntu 24.04 vérifie en Debug et Release l’empreinte historique `0x20c6e13722644e9d` du scénario T-6 340. Les jobs Flight Ubuntu exigent désormais cette comparaison.

La suite Debug GCC 14 a également exécuté 95 tests avec succès. Son test long `BlueDefenseGeneralizedLearnerTests` a terminé en 611,79 s. Les CTest du runner et des checkpoints ont été rejoués après la correction de la sélection explicite de l’acteur de `latest.ppokit`. Sur le T-6 870, les deux fichiers ont reproduit les mêmes 20 épisodes, retours, métriques et 38 905 pas (graines 4342–4361, 19 arrêts).

Les suites Python ont exécuté 11 tests Flight, 80 tests CI et 110 tests de curriculum adaptatif, tous réussis. Le vérificateur d’artefacts legacy et la migration V3 sont également exercés depuis CTest. Les journaux des jobs déjà présents sur la PR ont été lus. Le job Blue de validation comportementale était **skipped**. Les jobs qui ont lancé CTest ou les tests Python n’apportent aucune mesure comportementale Blue supplémentaire.

Pour reproduire le contrôle de migration T‑6, `ctest --test-dir build/release -R FlightControl_v3_migration --output-on-failure` importe le 870, compare toutes ses valeurs, rejoue une update, refuse un contrat incompatible et rejoue aussi le 340 avec son scénario. Les tests de publication et de processus sont `PPOKitPublicationTests` et `PPOKitProcessTests`. La campagne indépendante est reproductible avec le petit exécutable d’audit compilé contre les bibliothèques de chaque master, puis en confrontant son CSV au JSON de `ppokit-run eval`. [Pilote de référence](../../../tools/ppokit/flight_master_campaign.cpp) · [Comparateur d’épisodes](../../../tools/ppokit/compare_flight_campaign.py) · [Comparateur de tenseurs](../../../tools/ppokit/compare_flight_checkpoint.py)

La discipline de rewards, les architectures réseau et les références de production ont été préservées pendant la comparaison. La reprise JSBSim intra-épisode, la qualification comportementale d’Airspace et BlueDefense, les essais manuels du Studio, l’examen complet des extensions ABI et les lignes ouvertes du registre empêchent une déclaration de non-régression globale.


## Suivi de stabilisation Studio — 26 novembre 2026

Références relues : intégration `b834f5d157aff916f5e696b95b6ad1cac2e3387e`, master `b5ea31967f16d5fb2390268648715da5c6fe1b10`. Comparaison GitHub : 155 commits d'avance, aucun retard sur master. La PR 105 reste ouverte à la validation, sans merge.

| Sujet | Constat et modification | Preuve attendue | Statut |
|---|---|---|---|
| P1 — aides Overview | Les dix cartes disposent déjà de textes d'aide dans `HelpOf`. `DrawBox` dépendait de la temporisation de `ItemTooltip`. Les cartes demandent maintenant explicitement un survol sans temporisation ; les autres appelants conservent la politique de survol existante. La cause exacte du défaut signalé sous Windows n'est pas encore reproduite. | `PPOKitStudioProjectTests` : survol de chaque carte en trois frames, présence d'une fenêtre tooltip visible et disparition hors diagramme, clics de navigation existants. | NON VÉRIFIÉ : test ajouté, compilation et exécution non effectuées dans cette session. |
| P1-05 — processus | `ProjectWindow::RequestStop` utilise encore un fichier ; `Poll` constate la sortie du programme sans fermer le viewer lancé automatiquement. | Protocole de contrôle, arrêt borné, gestion de la propriété des viewers et tests d'intégration. | OUVERT |
| P0/P2 et entraînement | Aucun changement de modèle, de contrat ou de moteur dans ce lot. | Voir les contrôles de parité et les points ouverts précédents ; les résultats historiques ne constituent pas une validation du nouveau HEAD. | OUVERT |

Conformité AGENTS.md (version du HEAD de référence, blob `91b1004d67f1e55f6fe8dc953c3586501a33e78d`) : règles Before editing / Git — branche, master et instructions racine relus, aucun AGENTS.md imbriqué dans l'arbre inspecté ; modifications limitées au Studio et à son test. C++17 et documentation — API interne documentée et comportement par défaut conservé ; relecture statique effectuée. Testing and review / Fixed x64 build directories — **BLOQUÉ** : pas de clone ni de dépendances compilables dans la session ; pas de build Debug/Release, de CTest ni de validation graphique Windows. À exécuter dans les répertoires imposés : `cmake --build build/debug --config Debug --target PPOKitStudioProjectTests`, `ctest --test-dir build/debug -C Debug -R ^PPOKitStudioProjectTests$ --output-on-failure`, puis équivalents dans `build/release` avec Release. Aucun merge recommandé. Cette matrice partielle ne vaut pas déclaration de conformité complète.

## Studio, entraînement et modèles livrés — validation du 26 novembre 2026

Le point de départ de ce lot est `960ff4f2ac1d32c956022d8801bc661557b2ab2f`. Le code testé ci-dessous correspond à `9e2a73f837ba167b7562ea4f5f30739687a07e71`. Le master de comparaison est `b5ea31967f16d5fb2390268648715da5c6fe1b10`. Les chemins et lignes de cette section se rapportent au SHA du code. La mention « ouvert » antérieure sur les tooltips concerne désormais uniquement le rendu interactif Windows.

| Mode rejoué | Acteur réellement ouvert | Profil, seeds 4342–4361 | Arrêts | Pas | Distance moyenne d’arrêt |
|---|---|---|---:|---:|---:|
| T‑6, sélection BEST du Studio | `references/t6_best_870.ppokit`, SHA-256 `00d13803d680a3f36b2b620889bf15e87c6855524bd471ef96b083567024ade7` | actuel, niveau 12 | 19/20 | 38 905 | 546,87 m |
| PC‑7, archive du run 45460292649 | `checkpoints/policy.ppokit`, SHA-256 `b3574df581a8ce89c6344d8817bf361f80a5ad5c3ac71bd9c50181c497e7293c` | historique, niveau 11 | 19/20 | 29 331 | 731,67 m |
| PC‑7, sélection BEST du Studio | `references/pc7_best_610.ppokit`, SHA-256 `74a7e2851672af08c8bfac8dd71c4a317c0b4948b1084d137ac69390afab4673` | actuel, niveau 12 | 19/20 | 30 639 | 744,89 m |

Pour chaque campagne, les 20 durées, issues et retours ont été confrontés aux lignes sources des CSV ci-dessus. Les issues et durées coïncident, l’écart absolu de retour est inférieur à `1e-12`. Le transfert PC‑7 par `ppokit-run transfer` a changé le champ de contrat dans l’en-tête. La comparaison binaire du reste de l’artefact est exacte. Les références immuables des deux avions sont hors du dossier de publication des entraînements. Le démarrage de l’entraînement PC‑7 reste vierge. Les catalogues Airspace séparent les fichiers PROD et EVAL. Quand l’archive d’origine ne donne aucune update, le runner affiche `unknown`. L’acteur de production Blue conserve la valeur conventionnelle 0 de son en-tête et l’origine documentée à 5250 du run 101.

| ID et priorité | Cause vérifiée et correction | Preuve au SHA du code | Résultat et commit |
|---|---|---|---|
| TR-01, P0 budget | Le runner utilisait son budget générique. L’ABI optionnelle du fournisseur déclare la cible absolue du scénario. `--updates` reste additionnel, `--until-update` fixe une cible absolue. | `PPOKit/tools/ppokit_run.cpp:1147`, `Examples/FlightControl3D/tests/t6_evaluation_smoke.cmake:43` | Vérifié à la reprise 870 sans update supplémentaire. `f70dc3f0` |
| TR-02, P0 cadence | La valeur de sauvegarde par défaut ne venait pas du scénario Flight. Le profil fournit `checkpointEvery` et une option explicite garde priorité. | `PPOKit/include/ppokit/environment/run_profile.h:15`, `Examples/FlightControl3D/Environment/flight_environment_library.cpp:744` | Cadence 3 et publication finale vérifiées. `f70dc3f0` |
| TR-03, P0 seeds | L’inférence reprenait le reset d’entraînement. Play et le fournisseur Flight utilisent le plan d’épisodes et les seeds historiques. Le profil expose aussi la seed effective de l’environnement d’entraînement. | `Examples/FlightControl3D/Environment/flight_environment_library.cpp:289`, `PPOKit/tools/ppokit_run.cpp:1195` | Première seed T‑6, même acteur, 5 805 pas avant correction et 63 après. Les 20 longueurs et fins de play ont été comparées à eval. `f70dc3f0` |
| TR-04, P0 configuration | Task, fournisseur de scénario et réseau étaient confondus dans la vue. Les chemins des modules ouverts, le contrat, les stages réels et la provenance du modèle sont exposés. | `PPOKit/studio/scenario_view.cpp:101`, `PPOKit/studio/launch_settings.cpp:35`, `PPOKit/tools/ppokit_run.cpp:1377` | Vues ImGui, arguments et contrôles d’intégrité exécutés. L’audit de chaque hyperparamètre natif reste partiel. `6facf8af`, `09e47264` |
| TR-05, P1 arrêt / P1-05 processus | La demande par fichier n’interrompait pas un épisode. La session UDP lie l’enfant au Studio, sépare acquittement, sauvegarde et sortie OS. Le viewer ouvert pour le run est arrêté avec lui. | `PPOKit/src/ipc/control.cpp:212`, `PPOKit/studio/project_window.cpp:330`, `PPOKit/tests/session_control_tests.cpp:39`, `PPOKit/studio/tests/project_window_tests.cpp:151` | Stop en pause, terminal attaché, paquets rejetés, répétitions et sessions concurrentes couverts. Le protocole historique est une option explicite. Rendu interactif et crash réel du Studio Windows à qualifier. `4889dc34`, `9e2a73f8` |
| TR-06, P0 reprise | Une reprise explicite pouvait démarrer vierge et les poids courants dépendaient du journal. `--resume` exige le fichier et la publication v2 associe acteur, learner, état et poids de récompense. | `PPOKit/src/runtime/publication.cpp:96`, `PPOKit/tools/ppokit_run.cpp:1147` | T‑6 et PC‑7, une update depuis zéro puis une après reprise sur deux copies, rollouts natifs de 32 768 pas. Compteur 1 → 2, deux empreintes distinctes et sept coefficients conservés. La reprise repart sur un nouvel épisode JSBSim. `f70dc3f0`, `446affc4` |
| TR-07, P0 rewards | Une génération sans son journal pouvait perdre des coefficients modifiés. Restauration depuis le manifeste v2 et compatibilité v1. | `PPOKit/src/runtime/publication.cpp:113`, `PPOKit/tests/publication_tests.cpp`, `PPOKit/tests/ppokit_run_tests.cpp` | Zéro, valeur négative, corruption, redémarrage et rechargement testés. Parité par canal Airspace et BlueDefense ouverte. `446affc4`, `4889dc34` |
| TR-08, P0 maths | Trois imports et updates Flight comparés au moteur de référence. Les tests de graphe valident aussi GAE, masques et clipping. | Section « Audit mathématique », `Examples/FlightControl3D/tests/master_checkpoint_parity_tests.cpp` | T‑6 870 et PC‑7 610 identiques sur l’update figée publiée. Pour le T‑6 340 les écarts restent ceux indiqués dans le tableau historique. La comparaison exhaustive des gradients entre les backends reste ouverte. |
| TR-09, P0 frontière | Le reset natif d’inférence ignorait la distribution du scénario. Le runner différencie arrêt en simulation, terminaison et truncation. | `PPOKit/tools/ppokit_run.cpp:1514`, `Examples/FlightControl3D/tests/t6_evaluation_smoke.cmake:27` | 20 épisodes Flight vérifiés et absence de rapport partiel après interruption. État intra-épisode JSBSim non sérialisé. `f70dc3f0`, `4889dc34` |
| TR-10, P0 curriculum / BEST | Une sélection pouvait charger l’acteur courant sans identifier l’archive. Les rôles déclarés sont vérifiés par digest, les références Flight fixent le niveau d’évaluation et le Studio affiche la source native des stages. | `PPOKit/tools/ppokit_run.cpp:1437`, `Examples/FlightControl3D/tests/t6_evaluation_smoke.cmake:63` | Les deux acteurs livrés passent la campagne ci-dessus. Un BEST publié endommagé provoque une erreur. Une promotion de suite est prioritaire sur la référence de départ. `6facf8af`, `09e47264` |

**Mesure IPC locale.** Sur dix processus PC‑7 indépendants en Release Linux, du premier envoi STOP en état Paused à la terminaison système observée, la médiane est de 85,75 ms et le maximum de 92,7 ms. Le client de mesure attendait les datagrammes avec un timeout de 65 ms entre deux contrôles du processus. Cette mesure inclut ce délai d’observation. Un FlightControlHeadless natif a aussi terminé son update 1, écrit son checkpoint et quitté avec le code 0 après pause puis Stop.

**Hypothèses réfutées.** La divergence d’inférence T‑6 à la première seed provenait du reset malgré des poids identiques. Le PC‑7 possède une archive historique authentifiée. Le rectangle Task provient du contrat déclaratif `task.json`. Les DLL sont les fournisseurs compilés de simulation et de réseau.

**P2-01 à P2-04.** Le moteur Flight compilé et le SDK de graphes coexistent. Leur remplacement par une DLL universelle reste conditionné par la comparaison numérique des chemins repris. Les doubles déclarations de configuration sont désormais rapprochées pour le profil Flight, leurs autres consommateurs restent à examiner. Les catalogues et les refus explicites traitent une partie des anciens chemins. Les extensions ABI et les anciens chemins encore appelés gardent le statut du registre précédent. La comparaison exhaustive en boucle fermée d’Airspace et BlueDefense ainsi que la parité des mises à jour sur les cinq projets restent ouvertes.

### Contrôles reproductibles et conformité

Les trois commandes publiques à vérifier sur des **copies de projet** sont `ppokit-run eval --project DIR --best --suite validation`, `ppokit-run train --project DIR --fresh --updates 1` puis `ppokit-run train --project DIR --resume --updates 1`. Les deux dernières ont été exécutées sur chaque avion avec ses paramètres PPO du dépôt. L’archive PC‑7 d’origine se rejoue dans son projet extrait avec `ppokit-run eval --project DIR --suite validation`.

| Règles de l’AGENTS.md racine, blob `91b1004d67f1e55f6fe8dc953c3586501a33e78d` | Application à ce lot |
|---|---|
| Lignes 6–11, portée et séparation | Master, tête d’intégration, audits et appelants relus. Seul AGENTS.md racine s’applique. Modifications aux points d’entrée, au stockage, aux contrats déclarés et à la supervision. |
| Lignes 13–37, C++17 et architecture | ABI optionnelle versionnée, sessions RAII, sockets bornées et non bloquantes, formats de données explicites. La physique et le contrôleur rouge ne font l’objet d’aucun changement. |
| Lignes 39–49 et 51–73, preuves et rewards | Tests de régression du runner, de la publication, de la reprise, des sessions UDP et des vues. Comparaisons sur campagnes indépendantes et contrôles d’empreintes. Aucun canal ni formule de reward Blue modifié. |
| Lignes 75–83, répertoires | `cmake --build build/debug` et `cmake --build build/release`, puis `ctest --test-dir build/debug --output-on-failure` et équivalent Release en headless. Le test de trace Flight utilise son empreinte sur la plateforme GCC 13 prise en charge et se déclare skipped sous le GCC 14 local. Six suites Studio ont été liées et exécutées avec ImGui en mode headless. |
| Lignes 85–88, Git et validation | Branche d’intégration de la PR 105. Le nouvel intervalle passe `git diff --check`. Le différentiel historique complet contient un saut de ligne final antérieur dans `Examples/BlueDefense/Agents/BluePatrol/CMakeLists.txt:150`. Les jobs déclenchés par le SHA final sont à consulter dans les checks de la PR. |

Les comparaisons en boucle fermée et les essais graphiques Windows encore ouverts excluent une recommandation de merge globale.

## Intégrité des campagnes, des archives et du retour d’épisode

Les trois corrections partent de `a3d150231985ff73ddfa60967cad8e94318c0dfa`. Les lignes ci-dessous se rapportent au code `8f9fec5872bb46a6477415dcb275568a90158f09`.

| Anomalie | Reproduction avant correction | Correction et régression déterministe | Commit |
|---|---|---|---|
| TR-10, P0, promotion du mauvais acteur | Une campagne réussit avec les poids A. Le fichier courant est remplacé par B avant la promotion. Le manifeste précédent attribuait à B le score de A, même quand l’update restait identique. | La campagne ouvre l’acteur de la génération vérifiée et inscrit son SHA-256 dans le rapport. Le stockage vérifie la suite, le candidat et sa copie avant publication. L’ancien BEST reste lisible en cas de refus. `PPOKit/tools/ppokit_run.cpp:908`, `PPOKit/src/eval/best_store.cpp:219`, `PPOKit/tests/best_store_tests.cpp:187`. Les anciens manifests sans empreinte dans le rapport restent lisibles. | `7be38dcc` |
| TR-06, P1, démarrage vierge après modification des suites | Avec `approach` et `approach_wind`, l’archive planifiait deux fois le même déplacement et `train --fresh` échouait. Le BEST d’une suite retirée pouvait réapparaître dans un autre run. | L’inventaire tient compte du manifeste de chaque suite, du numéro de génération et de l’extension de ses poids. Le test retire une suite, change son extension historique et conserve un dossier de champions ainsi qu’une note opérateur nommée `approach_1.txt`. `PPOKit/src/eval/best_store.cpp:128`, `PPOKit/tests/ppokit_run_tests.cpp:434`. | `0459ed62` |
| TR-07, P1, retour incorrect dans le journal du learner | Sur une tâche qui plafonne la récompense totale, le dashboard affichait la somme des contributions avant plafond. | Le runner, le collecteur SDK et la session Flight fournissent au journal le scalaire envoyé au learner. Sur deux flux, dont un épisode à cheval sur deux rollouts, le test obtient un retour moyen de −0,375 pour une somme des moyennes de canaux de −1. Un autre test confronte le collecteur à son rollout. `PPOKit/include/ppokit/runtime/reward_ledger.hpp:114`, `PPOKit/tests/run_log_tests.cpp:207`, `PPOKit/tests/runtime_tests.cpp:157`. | `8f9fec58` |

### Essais par les points d’entrée publics

Sur deux copies des projets du dépôt, les commandes `train --fresh --updates 1` puis `train --resume --updates 1` ont publié les updates 1 et 2. Pour chaque publication, les SHA-256 du learner et de l’acteur ont été recalculés et comparés au manifeste. Le sidecar d’état de la même update, les coefficients de récompense et l’intégrité de l’acteur historique ont également été vérifiés. Chaque copie a ensuite servi à `play --model LATEST --episodes 1`, puis à `eval --model HISTORICAL_BEST --suite validation`.

| Projet | Vierge, reprise et inférence LATEST | Rapport HISTORICAL_BEST, profil livré et seeds 4342–4361 | Distance moyenne d’arrêt |
|---|---|---|---:|
| T‑6 | PASS, updates 1 → 2, un épisode joué | PASS, acteur 870, empreinte conforme, 19 arrêts sur 20 | 546,87 m |
| PC‑7 | PASS, updates 1 → 2, un épisode joué | PASS, acteur 610, empreinte conforme, 19 arrêts sur 20 | 744,89 m |
| BlueAgent MAPPO | NON EXÉCUTÉ sur une copie de projet pour ce lot | NON EXÉCUTÉ | — |
| RLMission | NON EXÉCUTÉ sur une copie de projet pour ce lot | NON EXÉCUTÉ | — |
| RLSNA | NON EXÉCUTÉ sur une copie de projet pour ce lot | NON EXÉCUTÉ | — |

La relecture porte sur le runner et ses deux types de learner, les appels du SDK et de Flight, la lecture et l’écriture des rapports, et leurs consommateurs Studio. Le calcul des récompenses et les paramètres d’apprentissage restent ceux des projets. Le test de télémétrie utilise la tâche synthétique `team_line`. Une comparaison de chaque canal historique par transition sur les cinq projets reste ouverte.

| Règles AGENTS.md du lot | Vérification |
|---|---|
| Lignes 6–37 | Instructions racine au blob `91b1004d`, appelants et contrats relus. Calcul des totaux par trajectoire dans un vecteur alloué à la construction. L’inventaire du disque et les contrôles de fichiers ont lieu au démarrage ou à la publication et à l’ouverture des modèles. |
| Lignes 39–73 | Cas périmé à update identique, changement de suite, fichiers d’opérateur, retour plafonné et épisodes inter-rollouts couverts. `PPOKitBestStoreTests`, `PPOKitRunTests`, `PPOKitRuntimeTests` et `PPOKitRunLogTests` reproduisent les chemins corrigés. |
| Lignes 75–88 | Builds C++17 et CTest dans `build/debug` et `build/release`, sans CUDA. Le test de trace historique nécessite la plateforme GCC 13 de la CI Flight. Les six tests ImGui s’exécutent sans écran. La vérification visuelle Windows et la parité globale des cinq projets restent ouvertes. |

## Vérification du choix par défaut et couverture par projet

Le contrôle final sur un répertoire PC‑7 vierge a révélé une seconde cause de modèle manquant. Le Studio transmettait son rôle, mais un `ppokit-run eval --project DIR` direct cherchait encore le dernier acteur d’entraînement. Le commit `816f94706f260acef2b6ee4e889c3597c7a266c9` partage la sélection déclarée dans `project.json` entre les deux points d’entrée. `--model LATEST` sélectionne explicitement le dernier acteur. Les sélecteurs Evaluation et Inference du Studio sont indépendants. Ces comportements sont vérifiés dans `PPOKit/src/schema/project.cpp:373`, `PPOKit/tools/ppokit_run.cpp:361`, `PPOKit/studio/launch_settings.cpp:34` et le test de dépôt `Examples/FlightControl3D/tests/t6_evaluation_smoke.cmake:63`.

| Contrôle et portée exacte | T‑6 | PC‑7 | BlueAgent MAPPO | RLMission | RLSNA |
|---|---|---|---|---|---|
| SHA-256 des acteurs déclarés et rattachement aux sources | PASS, 870 | PASS, 610 | PASS, PROD 5250 | PASS, PROD et EVAL | PASS, PROD et EVAL |
| Comparaison des poids importés aux archives | PASS | PASS | PASS, périmètre du tableau historique | PASS, acteur et checkpoint EVAL | PASS, acteur et checkpoint EVAL |
| Boucle fermée indépendante, comparaison épisode par épisode à master | PASS | PASS sur les deux profils | NON EXÉCUTÉ | NON EXÉCUTÉ à ce niveau de détail | NON EXÉCUTÉ à ce niveau de détail |
| Reprise courte et écriture de l’update suivante | PASS | PASS | NON EXÉCUTÉ contre le run historique | PASS, 127 → 128 dans le smoke existant | PASS, 888 → 889 dans le smoke existant |
| Entraînement depuis zéro avec les réglages réels puis reprise | PASS | PASS | NON EXÉCUTÉ pour cette qualification | NON EXÉCUTÉ pour cette qualification | NON EXÉCUTÉ pour cette qualification |
| Une update sur données figées avec comparaison finale des poids et moments | PASS, 870 | PASS, 610 | NON EXÉCUTÉ entre les deux moteurs | NON EXÉCUTÉ entre les deux moteurs | NON EXÉCUTÉ entre les deux moteurs |
| Rewards par canal à chaque transition du BEST, seeds 4342–4361 et 90001–90020 | PASS, 70 334 transitions | PASS, 61 971 transitions | NON EXÉCUTÉ | NON EXÉCUTÉ | NON EXÉCUTÉ |
| Comparaison exhaustive des gradients intermédiaires | NON EXÉCUTÉ | NON EXÉCUTÉ | NON EXÉCUTÉ | NON EXÉCUTÉ | NON EXÉCUTÉ |
| BEST embarqué chargé puis campagne du projet | PASS | PASS | NON EXÉCUTÉ en boucle fermée | NON APPLICABLE au catalogue qui distingue PROD et EVAL | NON APPLICABLE au catalogue qui distingue PROD et EVAL |
| GUI Windows exercée manuellement, du lancement à la fermeture du viewer | NON EXÉCUTÉ | NON EXÉCUTÉ | NON EXÉCUTÉ | NON EXÉCUTÉ | NON EXÉCUTÉ |

La seconde série (seeds 90001–90020) a été rejouée avec les références livrées et comparée aux CSV historiques. Le T‑6 870 obtient 20/20 en 31 429 pas. Le PC‑7 610 obtient 18/20 en 31 332 pas au niveau 12 et 17/20 en 31 146 pas dans son scénario initial. Sur les 120 épisodes des deux séries, les durées, les indicateurs de crash, de toucher et d’arrêt et les retours sont identiques aux valeurs des CSV de master. Les deux scores PC‑7 de seconde série sont sous le seuil de 92 %.

En Release Linux, sur quatre paires alternées de campagnes d’inférence T‑6 de 38 905 pas, le temps médian observé était de 3,877 s sans superviseur et de 4,142 s avec superviseur UDP. Ces mesures de processus complet incluent le lancement et l’observation de la sortie. Elles décrivent cette machine et cette charge.

## Collecte PPO, reprise des métriques et curriculum natif

Base du lot `90ee30328819556909b01d3f14eb437d2e88c813`. Le master de comparaison est `b5ea31967f16d5fb2390268648715da5c6fe1b10`.

| Anomalie | Reproduction | Correction vérifiée |
|---|---|---|
| TR-08 / TR-09, valeur suivante et masque | Deux variantes de Reach fournissent la même observation à la décision suivante. L’une actualise l’horloge dans `Step`, l’autre dans `BeginStep`. Après deux updates à seed identique, 40 lignes de la charge numérique du checkpoint différaient. Dans `team_line`, le retrait d’un agent entre deux décisions donnait aussi un masque périmé au learner. | Le runner transmet l’observation préparée et son masque au critique avant la décision suivante. La terminaison conserve une valeur suivante nulle, la troncature utilise l’observation finale. Les observateurs reçoivent le pas physique. Désormais, toute la charge numérique et les champs d’en-tête déterministes coïncident. Le test équipe compare l’entrée de la décision à celle reçue pour le bootstrap. `PPOKit/tools/ppokit_run.cpp:478`, `:1007`, `:1041`, commit `2158aef4`. |
| TR-06 / TR-07, cumul à la reprise | Sur une update Reach suivie d’une reprise, le canal Time affichait −5,12 à l’update 2 au lieu de −10,24. Goal affichait 1 au lieu de 2. | Le journal restaure le cumul pondéré de chaque canal à l’update exacte du checkpoint. Si cette ligne manque, le cumul historique reste inconnu et les statistiques de la nouvelle fenêtre restent disponibles. Reprise après retour à un ancien checkpoint et changement de coefficient couverts. `PPOKit/src/runtime/run_log.cpp:359`, `PPOKit/tests/run_log_tests.cpp:269`, commit `fb18bcd3`. |
| TR-04 / TR-10, navigation du Studio | Sur un projet natif sans `training.json`, cliquer sur Curriculum ne montrait pas les stages. Un pointeur de curriculum absent masquait également les paramètres lisibles du scénario. | La carte et la navigation ouvrent les stages déclarés. Le panneau affiche la dernière update publiée, le niveau et la fenêtre de succès. Les erreurs de source sont affichées dans la vue concernée. Test ImGui par clic sur la carte et lecture du contenu affiché. `PPOKit/studio/scenario_view.cpp:74`, `PPOKit/studio/tests/project_window_tests.cpp:399`, commit `2d3d6c37`. |

Sur une copie de chaque projet Flight, les commandes publiques de démarrage vierge et de reprise ont publié les updates 1 et 2. Sur les sept canaux, l’écart maximal entre le cumul publié et le cumul précédent augmenté de la contribution de la nouvelle fenêtre atteint 2,51 × 10⁻¹⁰ pour le T‑6 et 1,43 × 10⁻¹⁰ pour le PC‑7. Les deux générations ont passé le contrôle SHA-256 du learner et de l’acteur. Un épisode avec LATEST et la campagne de 20 épisodes avec HISTORICAL_BEST ont ensuite été exécutés pour chaque appareil.

### Récompenses Flight comparées à chaque pas

Le commit `c593bd30` capture les actions déterministes des BEST via les modules désignés par le projet. Un second processus, compilé contre les sources Core et Simulator du master, rejoue exactement ces actions avec ses propres observations. Chaque bloc de 20 seeds ouvre une nouvelle instance JSBSim et la conserve pour toute la campagne. Les deux processus utilisent JSBSim 1.3.3, les mêmes données avion et le niveau 12. Remplacer une contribution par sa valeur augmentée de 1 ou retirer la fin du fichier fait échouer le comparateur.

| Mesure sur les seeds 4342–4361 et 90001–90020 | T‑6, BEST 870 | PC‑7, BEST 610 |
|---|---:|---:|
| Épisodes et transitions | 40 / 70 334 | 40 / 61 971 |
| Arrêts sur chaque série | 19/20 et 20/20 | 19/20 et 18/20 |
| Écart absolu maximal sur les 35 observations de décision | 0 | 0 |
| Écart absolu maximal sur chacun des sept canaux | 0 | 0 |
| Écart absolu maximal sur le scalaire de récompense | 0 | 0 |
| Fins d’épisode divergentes, pas ou type | 0 | 0 |
| Transitions avec une contribution Crash non nulle | 0 | 3 |
| SHA-256 de la capture x64 Release locale | `7d1fb9fcaf77e60418409e27f9d198d9ff6f61c47608bfd37cb77c6d3a0d93ae` | `94c29b12edb3f031849ae53e3ccf8c035d377779a0a2855f81ba892a17c77677` |

Pour reproduire sous Linux x64, utiliser un checkout de master à la révision ci-dessus, avec `FlightControlSimulator` compilé en Release. Exécuter ces commandes depuis le dépôt d’intégration, configuré avec `PPO_BUILD_FLIGHT_CONTROL=ON` et `PPO_BUILD_CORE_TESTS=ON`. Donner à `ref` le chemin absolu du checkout historique.

```sh
ref=/chemin/vers/PPOTrial-master
src="$ref/Examples/FlightControl3D"
native="$ref/build/release/Examples/FlightControl3D"
out=build/release/manual_checks/flight
kit=build/release/Examples/FlightControl3D
mkdir -p "$out"
cmake --build build/release --target FlightControl_reward_capture
c++ -std=c++17 -O3 -DNDEBUG -I"$src/Core" -I"$src/Simulator" -I"$src/Training" \
    tools/ppokit/flight_reward_replay.cpp "$native/libFlightControlSimulator.a" \
    "$native/libFlightControlCore.a" "$ref/build/release/_deps/jsbsim-build/src/libJSBSim.a" \
    -pthread -o "$out/flight_reward_replay"
for aircraft in T6 PC7
do
    "$kit/FlightControl_reward_capture" "Examples/FlightControl3D/projects/$aircraft" "$kit" "$out/$aircraft.trace" || exit 1
    "$out/flight_reward_replay" "$ref/resources/settings/Scenarios/FlightControl3D_${aircraft}_Joint.json" "$out/$aircraft.trace" || exit 1
done
```

| Règles du fichier AGENTS.md racine, blob `91b1004d` | Vérification du lot |
|---|---|
| Lignes 6–37 | Lecture des deux chemins de collecte, des appelants du journal et des vues. Le runner réutilise ses tampons de décision et de masque. Le rejeu de référence lit une transition à la fois. |
| Lignes 39–73 | Régressions avant et après correction, captures indépendantes, vérification de la terminaison et de la troncature. Comparaison des sept canaux existants sur les deux avions. |
| Lignes 75–88 | Builds C++17 CPU en `build/debug` et `build/release`. Dans chaque configuration locale, 95 CTest réussis et une trace de référence GCC 13 ignorée sous GCC 14. Six suites ImGui réussies. Le lot passe `git diff --check`. L’essai manuel Windows et la parité complète des cinq projets restent ouverts. |

## Campagnes manuelles dans le Studio et contrôle de chaque seed

La révision de départ est `688101685d54a3d321d00a004d662393daaddc25`. Les deux corrections concernent les résultats que voit l’opérateur après la commande Evaluate.

| Anomalie | Preuve avant correction | Résultat vérifié |
|---|---|---|
| TR-09, comparaison des campagnes | Permuter les retours des seeds 4342 et 4343 dans le rapport réel T‑6 conservait la moyenne et le taux d’arrêt. Le comparateur produisait son CSV et quittait avec le code 0. Même reproduction sur le PC‑7. | Le mode `--check` sort avec le code 1 et indique la seed, le champ et les deux valeurs. Il vérifie les pas, les issues, les retours, les distances aux événements et la distinction terminaison/troncature. Sans option, la commande produit le tableau des différences. Le comparateur accepte le CSV natif et les colonnes master des tableaux d’audit. `tools/ppokit/compare_flight_campaign.py:35`, commit `47d6449e`. |
| TR-04 / TR-10, résultats du Studio | Avec un fichier `checkpoints/evaluation_validation.json` valide et sans journal d’entraînement, la vue ne montrait ni le verdict ni l’acteur évalué. Sur une copie vierge du PC‑7, elle ignorait aussi le BEST livré avec le projet. | Evaluation affiche le verdict enregistré, le nombre d’épisodes, le rôle, l’update disponible, le retour moyen, les gates et le SHA-256 de l’acteur évalué. Le BEST promu est prioritaire sur l’archive de la même suite. Un acteur endommagé produit une erreur. Le panneau identifie un rapport de la mauvaise suite ou une campagne incomplète. Les anciens rapports sans attribution restent lisibles. `PPOKit/studio/evaluation_view.cpp:467`, `PPOKit/src/eval/evaluation.cpp:603`, commit `7d300359`. |

Les deux séries de 20 épisodes des deux avions ont passé le comparateur strict face aux références historiques. Sur les fichiers des seeds 90001–90020, la fenêtre de projet ImGui a affiché les valeurs suivantes après une commande réelle `ppokit-run eval`.

| Projet | Acteur affiché | Taux d’arrêt | Seuil | Verdict affiché |
|---|---|---:|---:|---|
| T‑6 | HISTORICAL_BEST 870 | 20/20 | 92 % | passed |
| PC‑7 | HISTORICAL_BEST 610 | 18/20 | 92 % | failed |

Le processus d’évaluation PC‑7 a rendu le code 3 pour le seuil manqué. La comparaison avec master a rendu 0 car les vingt épisodes correspondent à la référence. Le contrôle suivant vérifie la campagne de validation du PC‑7 sur une copie du projet.

```sh
python3 tools/ppokit/compare_flight_campaign.py \
    docs/audits/ppokit-integration/pc7-610-current-master-episodes.csv \
    "$project/checkpoints/evaluation_validation.json" --check --seed-range 4342 4361
```

La régression Python est enregistrée dans CTest sous `FlightControl_campaign_comparison`. La vue ImGui exerce l’arrivée et le retrait du rapport, la restauration d’une archive modifiée, la promotion d’un BEST et le rejet de poids corrompus. Les essais utilisent les répertoires Debug et Release prescrits. Les comparaisons utilisent les deux acteurs historiques et leurs profils de validation déclarés.
