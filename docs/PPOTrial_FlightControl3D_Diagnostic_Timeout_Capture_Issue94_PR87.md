# FlightControl3D — diagnostic causal et correction du délai de capture

Date : 13 novembre 2026, Europe/Paris. [Issue #94](https://github.com/benoitfragit/PPOTrial/issues/94), [PR #87](https://github.com/benoitfragit/PPOTrial/pull/87), branche `fix/flight-approach-only`. Diagnostic à partir du latest du [run 13](https://github.com/benoitfragit/PPOTrial/actions/runs/43406061312), update 8 000, code initial `67057d9660834b0297f54e9a72b2095d0e76190f`.

## Résultat et cause démontrée

La limite de **60 secondes par épisode LOC/Glide** coupait des interceptions valides avant leur capture stabilisée. Le modèle du run 13 avait appris davantage que ne le laissaient apparaître les validations arrêtées à cette borne. Le correctif porte `training.curriculum.captureMaxSeconds` à **120**, sans changer les poids du modèle, le reward, les seuils physiques ou le seuil de passage.

Le même latest, en politique déterministe et sans optimisation, donne :

| Graine d'évaluation | Limite 60 s | Limite 120 s | Verdict à 120 s |
|---|---:|---:|---|
| 4342, celle de la CI | 8/20 | 19/20 | LOC validé |
| 4343 | 6/20 | 20/20 | LOC validé |
| 7342 | 8/20 | 17/20 | LOC insuffisant |

Les trois scores à 120 s ont été reproduits avec le **FlightControlHeadless officiel**, et pas uniquement avec la sonde diagnostique temporaire. Le SHA-256 du latest est identique avant et après chaque validation : `ce534dede8750f1ed32f10597489b6d34b62a19f54811a85857ff1422763edbd`.

Cela corrige le diagnostic initial : le plafond apparent à 55 % mesuré sur 60 s ne démontrait pas une incapacité générale du réseau à capturer le LOC. En revanche, **la généralisation complète et l'atterrissage ne sont pas qualifiés** : la graine 7342 reste sous 19/20, et les vols prolongés révèlent encore des crashes (1, 0 et 3 selon les trois graines). Aucun BEST d'atterrissage n'est promu par ces captures.

## Méthode de diagnostic

1. Rejouer exactement le modèle du run 13, avec les mêmes sources et la graine 4342 : résultat 8/20 reproduit.
2. Enregistrer à 10 Hz le LOC, le cap relatif, la route, la vitesse, le roulis, p/q/r, le maintien et les actions. Le terminal officiel reste calculé à 100 Hz par les sous-pas physiques.
3. Parmi les 12 timeouts de cette campagne, neuf n'atteignent pas le faisceau capturé avant 60 s. Dans les échantillons déjà dans le faisceau, les principaux défauts supplémentaires concernent le cap, la route et les rates ; les colonnes se chevauchent.
4. Faire varier uniquement la limite de leçon dans le JSON, sans apprentissage : le score passe à 19/20. Les captures réussies de cette campagne se terminent entre environ 45 et 93 s.
5. Contrôler le changement de normalisation de l'observation `elapsed` : une seconde sonde prolonge le vol à 120 s tout en conservant l'ancienne observation `min(t/60,1)`. Elle donne aussi 19/20. Cette ablation soutient le rôle de la coupure temporelle, indépendamment du seul changement d'horloge observée.
6. Refaire les validations sur deux autres graines, puis vérifier les mêmes résultats avec le headless officiel et contrôler l'intégrité du checkpoint.

La sonde temporaire ne remplace aucun pilotage de production et ne modifie pas le modèle. Les compteurs d'échecs échantillonnés à 10 Hz n'établissent pas à eux seuls la durée exacte des violations entre deux actions ; le verdict terminal officiel reste la référence.

## Hypothèse PPO contrôlée

Deux continuations locales de **200 updates**, chacune depuis le même latest 8 000, utilisent les anciennes leçons de 60 s, les mêmes graines, des blocs de 10 et une validation après chaque bloc. Le seul paramètre varié est le taux acteur :

| Mesure | actorRate = 0,0001 | actorRate = 0,00001 |
|---|---:|---:|
| Updates avec arrêt KL | 200/200 | 96/200 |
| Pas acteur moyens par update | 4,36 | 94,24 |
| Moyenne des 20 taux de validation | 40,5 % | 41,0 % |
| Maximum LOC | 11/20 | 10/20 |
| LOC final | 9/20 | 8/20 |

Réduire le taux a bien changé la dynamique d'optimisation, mais cet essai court n'a pas démontré de gain utile sur la gate. Il n'est donc pas retenu dans le correctif. `targetKl=0.02`, les taux acteur/critic, gamma/lambda, la distribution et les dix canaux de reward restent ceux du code initial. L'effet éventuel de PPO sur les défauts résiduels reste une question distincte.

## Correction et architecture de reprise

Le délai LOC/Glide est réglé dans le JSON partagé par l'interface, le headless et la CI. `ConfigureCaptureLesson` applique déjà `min(mission.maxSeconds, captureMaxSeconds)`. Les deux phases Joint gardent leur limite de 900 s. La succession reste **LOC fixe → Glide fixe → Joint fixe → Joint mobile**, houle désactivée.

Un second problème apparaît lors de la reprise : le runner entraînait systématiquement avant de revalider. Dans l'essai réel, le modèle qualifié à 19/20 retombe à 18/20 après une seule update. Le runner réévalue maintenant un latest repris en LOC/Glide, avec `training.resume=true`, **avant tout pas PPO** :

- copie temporaire du latest, validation au même compteur d'updates, puis comparaison octet pour octet ;
- conservation du journal `gate_recheck_*.log`, déjà couvert par l'artefact du workflow existant ;
- préservation des modèles, BEST et CSV originaux, y compris les CSV hérités du run précédent ;
- passage de gate uniquement sur un nouveau verdict indépendant réussi ; aucun budget d'updates consommé par ce contrôle ;
- en cas d'échec, entraînement avec le budget complet ;
- Joint fixe/mobile gardent le chemin normal de validation et de promotion BEST.

Le test natif manuel LOC termine avec `Updates: 8000 -> 8000`, 19/20 et prochaine phase Glide ; le latest reste strictement identique. Le test progressif natif passe ensuite réellement en Glide : une update jusqu'à 8 001 donne **13/20**, reste sous le seuil, sauvegarde la phase Glide et retourne correctement `incomplete` lorsque son budget d'une update est épuisé. C'est un essai local de chemin de reprise, pas une qualification du Glide.

## Régressions et relecture

La nouvelle régression C++ intègre une référence d'interception cinématique idéale sur les extrêmes de distance/azimut du scénario, les deux côtés et quatre caps piste (0/90/181/359°). Elle fait passer le véritable automate de capture et son maintien de 2 s. Elle échoue avec le JSON à 60 s et réussit à 120 s ; un contrôle négatif conserve explicitement l'échec de l'ancien cas éloigné. Ce test vérifie une condition nécessaire de cohérence temporelle, sans qualifier un avion ou une politique.

Validation locale CPU GCC 14, C++17 :

- builds headless Debug et Release ;
- **15/15 CTest en Debug et 15/15 en Release** ;
- **18/18 tests Python du runner** : reprise, réussite/échec du contrôle, budget, conservation modèle/BEST/télémétrie, mutation interdite, verdict mal formé, reprise désactivée, et chemin de promotion des atterrissages ;
- régression de délai et cinq nouvelles régressions du runner observées en échec avant correction ;
- rejeux JSBSim réels, empreintes de checkpoints et tests de progression décrits ci-dessus.

Relecture des fonctions modifiées, de leurs appelants, des tests, du chemin de validation et du diff de la branche contre sa base. Le correctif ne touche pas RLMission ni BlueDefense et n'ajoute aucun workflow ou CUDA. La CI du nouveau commit est à contrôler dans les journaux de la PR ; elle n'est pas déclarée passée au moment de la rédaction.

## Reproduire et continuer

Avec l'artefact du run 13 extrait dans `build/flight-training`, sur la branche corrigée :

```bash
build/release/Examples/FlightControl3D/FlightControlHeadless \
  --config resources/settings/Scenarios/FlightControl3D.json \
  --output build/flight-training --stage loc --updates 8000
```

La borne absolue 8 000 égale le compteur du latest : cette commande passe directement en validation. Elle ne réalise aucune update PPO. Son CSV courant est celui de la commande ; pour conserver toute la télémétrie d'origine, utiliser une copie du dossier ou le runner, qui isole automatiquement son contrôle initial.

Pour reprendre dans la même **FlightControl3D Curriculum Training**, choisir :

| Champ | Valeur |
|---|---|
| Branche | fix/flight-approach-only |
| stage | progressive |
| resume_run_id | 43406061312 |
| additional_updates | 10 |
| max_additional_updates | 10000 |

Le contrôle initial LOC est refait à 120 s avant l'entraînement, puis la progression dépend des nouveaux verdicts. Aucun entraînement distant n'a été déclenché par ces travaux ; les expériences d'apprentissage ont été locales et isolées des artefacts de référence.

## Sources

- [Run 13 et son artefact](https://github.com/benoitfragit/PPOTrial/actions/runs/43406061312/artifacts/14028944270).
- [Configuration du scénario](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/resources/settings/Scenarios/FlightControl3D.json), [runner](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/tools/flight/curriculum_ci.py), [régressions Python](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/tools/flight/test_curriculum_ci.py).
- [Automate de capture](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/Examples/FlightControl3D/Core/flight_mission.cpp), [environnement](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/Examples/FlightControl3D/Simulator/flight_environment.cpp), [régression de délai](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/Examples/FlightControl3D/tests/core_tests.cpp), [validation native](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/Examples/FlightControl3D/Runtime/flight_session.cpp).
- [Guide CI actualisé](https://github.com/benoitfragit/PPOTrial/blob/fix/flight-approach-only/docs/PPOTrial_FlightControl3D_Curriculum_ILS.md).
