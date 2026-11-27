# PPOKit — stabilisation corrective du 27 novembre 2026

Référence auditée : `69c09196c79b57ef189d24b532adddc655d5c433`, branche
`design/ppokit-framework`, PR 105. Aucun modèle n'a été réentraîné et aucun
hyperparamètre nominal n'a été modifié. L'ancien journal de migration
`PPOKit/PROGRESS.md`, sans rôle dans le build ni le runtime, a été retiré à la
demande du mainteneur.

## Anomalies confirmées et corrections

| ID | Cause racine | Correction | Test de non-régression | Validation actuelle |
|---|---|---|---|---|
| P0-01 | Le fichier officiel était déplacé avant l'installation de son successeur ; le temporaire était partagé entre writers. | Temporaire unique, verrou interprocessus, flush fichier/répertoire, remplacement atomique natif, backup installé sans retirer la dernière génération valide. | `PPOKitAtomicFileTests` : concurrence, interruption, backup, Unicode. | Debug et Release Windows réussis. |
| P0-02 | Publication et promotion BEST protégeaient les fichiers séparément mais pas toute la décision transactionnelle. | Verrou de génération, staging vérifié par SHA-256, manifeste relu avant/après, promotion BEST sérialisée. | `PPOKitPublicationTests`, `PPOKitBestStoreTests` : interruptions, récupération idempotente, publishers concurrents. | Debug et Release Windows réussis ; publication répétée 10 fois en Debug. |
| P1-01 | Un chemin explicite manquant pouvait retomber sur un ancien binaire de même nom ; registre et PATH ambigus choisissaient le premier ; la ligne Windows repassait par une chaîne étroite. | Résolution absolue stricte, refus des ambiguïtés, registre relatif à son fichier, PATH UTF-16 et ligne `CreateProcessW` UTF-16. | `PPOKitProjectTests`, `PPOKitProcessTests` avec espaces, accents et caractères Unicode. | Debug Windows réussi. |
| P1-02 | Seul le processus enfant local verrouillait une partie des éditeurs. Les sauvegardes directes réseau/évaluation et une session CLI attachée contournaient la protection. | Politique de lecture seule commune aux quatre éditeurs persistants et sauvegardes directes refusées ; le canal live des rewards reste disponible. | `PPOKitStudioProjectTests`. | Debug Windows réussi. |
| P1-03 | Le runtime contrôlait déjà l'identité sémantique, mais le Studio ne prévenait qu'après création du processus. | Préflight des artefacts de graphe sélectionnés avant lancement ; le layout reste hors identité. | Tests d'identité existants et `PPOKitStudioProjectTests`. | Debug Windows réussi. |
| P1-04 | `Kill` attendait jusqu'à une seconde sur le thread appelant et l'escalade ne distinguait pas Saving. | Kill non bloquant, observation par polling, reap POSIX différé, délai borné propre à Saving et conservation du kill forcé. | `PPOKitProcessTests`, `PPOKitStudioProjectTests`. | Debug Windows réussi. |
| P1-05 | Un ACK de réception pouvait laisser Pause/Resume/rewards en attente sans limite ; Stop n'expirait jamais et un nouveau port ne pouvait pas se reconnecter. | États Pending/Received/Applied/Rejected/Indeterminate, deadline d'application, réponse tardive corrélée, reconnexion après silence et retries Stop prioritaires. | `PPOKitSessionControlTests` : pertes pendant Pause/Saving/rewards, retard et reconnexion. | Debug Windows réussi. |
| P1 numérique | Somme et carrés des avantages pouvaient déborder ; observations/actions non finies entraient dans les rollouts. | Chemin nominal inchangé bit à bit, fallback mis à l'échelle uniquement après débordement, refus précoce des valeurs non finies. | `PPOKitGaeTests`, `PPOKitTeamRolloutTests` et suites learners/update/hosted. | Six suites Debug réussies. |
| P1-06 | La reprise générique persistait uniquement le nom de stage et effaçait la fenêtre de promotion. | État versionné stage/episodes/successes ; lecture compatible du format historique. Le RNG reste celui du checkpoint de l'appelant. | `PPOKitCurriculumTests` : interruption à 3/4 puis résultat identique à l'exécution continue. | Debug Windows réussi. |
| P2 CLI | `std::stod` acceptait un préfixe numérique et laissait ses exceptions sortir comme erreur runtime. | Consommation complète, finitude et diagnostic uniforme d'usage. | `PPOKitRunTests` : suffixe, NaN, Inf et espaces. | Debug Windows réussi. |
| P2 récents | La liste des projets récents était tronquée avant de connaître le succès de l'écriture. | Écriture atomique durable, ancienne liste préservée sur erreur, APPDATA UTF-16. | `PPOKitRecentProjectsTests` et couverture commune d'atomic file. | Debug Windows réussi. |

## Protections examinées et conservées

- GAE et surrogate PPO : formules inchangées ; terminal/troncature et bootstrap
  possèdent déjà des tests bit à bit.
- Learners graphe PPO/MAPPO : sauvegarde préalable des poids, moments Adam et RNG,
  rollback complet sur erreur, minibatches partiels, rollout vide et acteur sans
  échantillon déjà couverts.
- BEST/LATEST : sélection, contexte d'évaluation, digest, annulation et références
  historiques restent couverts par les manifestes existants ; les critères de classement
  n'ont pas changé.
- Identité graphe : blocs, paramètres, câblage, composants, contrat sémantique et masques
  sont hashés ; le sidecar de layout n'entre volontairement pas dans cette identité.

## Validation effectuée et limites

- Build complet Debug Windows : réussi.
- Build complet Release Windows : réussi. Un avertissement C4244 préexistant reste présent
  dans `Examples/BlueDefense/Environment/blue_environment.cpp`.
- Matrice corrective Release : 15/15 tests réussis (`AtomicFile`, `Publication`, `BestStore`,
  `Process`, `SessionControl`, `Project`, `StudioProject`, `GAE`, learners PPO/MAPPO,
  update engine, hosted trainer, curriculum, CLI et projets récents).
- `PPOKitPublicationTests` : 10 exécutions Debug successives réussies après ajout du retry
  borné pour les renommages de répertoires momentanément verrouillés par Windows.
- CTest Debug global : 86 tests atteints sur 114 avant interruption manuelle du très long
  `BlueDefenseGeneralizedLearnerTests`. Les tests 1 à 15 et 17 à 86 avaient réussi ; le
  test 16, qui avait exposé le verrouillage Windows intermittent, a ensuite été corrigé et
  répété avec succès. Le CTest global n'est donc pas déclaré complet.
- CTest Release global et logs CI de la PR : non exécutés. La branche n'a pas été poussée
  par cette validation locale ; aucune conclusion CI n'est revendiquée.
