# PPOTrial — Blue MAPPO V9 : reward de convergence calibré et rapport CI

Date : 2026-11-02. Branche : `fix/blue-v9-patrol-convergence-reward`.
PR : #45. Base : master. Curriculum : V3 inchangé.

## Contrat

- 0 Rouge visible : leader guidé vers l'hippodrome, équipiers vers leur slot
  wedge. La récompense se calcule avec une cible **figée au début du pas**.
- 1 Rouge visible : leader seul en poursuite; équipiers en formation derrière
  le leader. Comportement de combat conservé, sauf extraction d'une formule
  mathématiquement identique dans un helper partagé.
- 2+ Rouges : aucun changement des signaux de poursuite ou coopération.
- Ne change ni les observations/poids V9, ni la physique, ni RLMission.

## Échelle et formule du cas sans Rouge

Vitesse de croisière = 300 m/s, max = 750 m/s, dt = 0,2 s.
Les deux modes utilisent désormais la même normalisation **maxSpeed*dt = 150 m**
pour la progression physique entre deux positions du Bleu vers une référence
figée.

```text
p(T) = clamp((distance(B_avant,T) - distance(B_après,T)) / 150, -1, 1)
g = clamp(1 - error_position_normalized, 0, 1)
closing = g*p(guidancePose_avant) + (1-g)*p(trackPose_avant)
speed_error = abs(speed_après - speed_cible)/max(0.2*speed_cible, 1)
speed_penalty = 0.4*speed_error/(1+speed_error)
patrol_reward = clamp(2*closing - speed_penalty
  + 0.004*(forward*g*alignment - 0.15*bounded_error), -1, 1)
```

Le coefficient 2 correspond au poids effectif de `Pursuit progress` avec un
Rouge; la normalisation et le helper sont communs.

| Condition (piste droite, valeurs indicatives) | Signal par pas |
| --- | ---: |
| Convergence 60 m, vitesse 300 m/s, aligné | ~ +0,80 |
| Convergence 150 m, vitesse 750 m/s | ~ +0,65 |
| Déplacement latéral 50 m vers le circuit, hors tolérance | ~ +0,67 |
| Vol parallèle à 2500 m du circuit | ≤ 0 |
| Éloignement 50 m du circuit, hors tolérance | ~ −0,67 |

Une convergence directe à vitesse maximale n'est donc pas plus avantageuse
qu'un suivi à vitesse de croisière. En dehors du circuit, la cible projetée
prend le relais du point anticipé afin de ne pas récompenser le vol parallèle
permanent.

La sortie `Patrol tracking` / `Formation keeping` reste bornée dans [-1,1]
avant agrégation; les poids configurés restent inchangés. Ce shaping reste
une hypothèse de correction, non une démonstration de convergence PPO.

## Pipeline de diagnostic

À la fin du workflow Windows Release, le module Python de diagnostic
préexistant calcule `diagnostic.html` à partir des fichiers **locaux**
`build/mappo-telemetry` et du CSV Blue. Les checkpoints restent référencés
depuis leur chemin d'origine (pas de copie coûteuse).

- Synthèse Markdown dans `GITHUB_STEP_SUMMARY` avec indicateurs de patrouille,
  erreurs et progression.
- Artefact autonome `blue-mappo-diagnostic` contenant `diagnostic.html`.
- Exécution et publication `if: always()` même si la phase train échoue.
- Aucun téléchargement GitHub ni dépendance pip ajoutée.
- Pas de visualisation HTML directement dans le Summary GitHub Actions.

## Tests et précautions

Tests déterministes C++ : échelle 60 m, pénalité à 750 m/s, retour latéral,
vol parallèle hors circuit, sens de rotation P=1/3, maintien des cas Rouge.
Tests Python : génération HTML et Summary depuis un dossier de runner,
dégradation maîtrisée en cas de CSV absent.

**Exécutions à confirmer sur CI** : compilation VS2022 x64 Debug/Release,
CTest, Python unittest. La convergence du nouveau modèle doit être évaluée
sur des seeds indépendantes et sur Pvs0, 1 Rouge et plusieurs Rouges.
