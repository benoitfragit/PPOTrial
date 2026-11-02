# Blue Training Diagnostics

Application Windows autonome pour télécharger, diagnostiquer et comparer les
entraînements Blue MAPPO produits par GitHub Actions.

L’outil utilise uniquement la bibliothèque standard de Python : Tkinter pour
l’interface, `urllib` pour GitHub, `csv`/`json` pour les données et des SVG
intégrés au rapport HTML pour les courbes. Il ne nécessite ni `pip install`, ni
pandas, matplotlib, requests, Qt ou serveur web.

## Lancement

Double-cliquer sur :

```text
tools\blue_training_diagnostics\Lancer_Diagnostic_Blue.bat
```

Prérequis : Windows et Python 3.7 ou plus récent avec Tkinter. Le lanceur
Windows `py` et Tkinter sont déjà présents sur le poste de développement ayant
servi à valider cet outil.

L’application détecte automatiquement le dépôt `origin`. Les fichiers sont
placés dans le dossier Windows connu :

```text
Téléchargements\PPOTrial-CI\
├── runs\run-<identifiant GitHub>\
│   ├── artifacts\
│   ├── reports\diagnostic.html
│   └── run.json
└── comparisons\<run A>-vs-<run B>\comparaison.html
```

## Authentification Git / Git Extensions

La liste d’un dépôt public peut être consultée sans connexion. GitHub exige en
revanche une authentification pour télécharger un artefact Actions. L’outil
réutilise automatiquement l’identification déjà configurée par Git et Git
Extensions via la commande standard `git credential fill`.

L’identifiant fourni par Git Credential Manager reste uniquement en mémoire le
temps de la requête. L’application ne le montre pas, ne le journalise pas et ne
l’écrit ni dans le dépôt, ni dans les téléchargements ou les rapports. Le
stockage protégé et la session GitHub restent entièrement gérés par Git
Credential Manager et le gestionnaire d’identifiants Windows.

Si aucune session n’est disponible, Git Credential Manager peut ouvrir la page
de connexion GitHub. Il suffit sinon d’effectuer une opération Git authentifiée
dans Git Extensions, puis de relancer le téléchargement.

Documentation GitHub :

- <https://docs.github.com/en/rest/actions/artifacts#download-an-artifact>

## Parcours utilisateur

1. Cliquer sur **Actualiser les runs**.
2. Sélectionner un ou plusieurs runs puis **Télécharger + diagnostiquer**.
3. Le rapport HTML s’ouvre automatiquement dans le navigateur.
4. Sélectionner exactement deux lignes avec `Ctrl+clic`, puis cliquer sur
   **Comparer 2 runs**.

Un double-clic sur un run ouvre sa page GitHub. **Analyser un dossier local**
accepte aussi un ancien artefact déjà extrait.

## Diagnostic produit

Le rapport autonome contient notamment :

- validité du CSV, valeurs finies, progression PPO et stagnations ;
- évolution par campagne du succès, de la survie Blue, de la neutralisation Red,
  de la survie de l’objectif et du leader, ainsi que de la durée des épisodes ;
- tableau des campagnes par update, composition P/N et causes terminales ;
- patrouilles : distance qualifiée, circuit déclaré, resets, violations de suivi,
  marge d’espacement et tolérances position/vitesse/cap par rôle avec moyenne,
  médiane, p95 et pire valeur ;
- séquence de patrouille : mode RALLY/FOLLOW, waypoint actif, nombre validé,
  distance au waypoint, écart à la piste, erreur de cap, vitesse, abscisse
  curviligne et cause de terminaison ;
- décomposition de chaque canal de récompense par rôle ;
- pertes Blue par cause et agent, avec vitesse, variation de vitesse, carburant et
  commande d’accélération au moment de la perte ;
- récompense lissée, distribution, volatilité et débit d’updates ;
- temps médian, p95 et maximal de l’optimiseur, du critic, des avantages et de
  PPO, ainsi que le débit d’évaluation ;
- policy loss, value loss, entropie, KL approximative, fraction clippée,
  variance expliquée et écart-type des actions pour les nouveaux runs ;
- présence, nombre et taille des checkpoints ;
- erreurs MAPPO explicites trouvées dans les journaux ;
- segments et blocs de curriculum analysés séparément.

Les seuils KL `> 0,05`, clipping `> 50 %` et variance expliquée `< 0` sont des
heuristiques signalées comme telles. Ils servent à orienter l’inspection et ne
constituent pas des critères automatiques de promotion.

## Diagnostic automatique dans GitHub Actions

À chaque fin du workflow `mappo-training-windows.yml`, même après un échec
d'entraînement, l'action `python -m blue_diagnostics.ci_report` réutilise
**localement** les journaux, CSV, fichiers curriculum, manifeste et checkpoints
du runner. Aucun téléchargement ni identifiant GitHub n'est nécessaire.

- **Run → Summary** : tableau Markdown compact (score, PPO, patrouille, erreurs).
- **Run → Artifacts → `blue-mappo-diagnostic`** : téléchargement du fichier
  `diagnostic.html`, consultable dans le navigateur hors connexion.
- `blue-mappo-telemetry` et `blue-mappo-checkpoints` restent inchangés.

GitHub Actions ne permet pas d'intégrer une véritable page HTML interactive
directement dans le Summary. Une visualisation HTML publique dans le navigateur
nécessiterait un hébergement distinct, tel que GitHub Pages (non activé ici).

## Comparaison de deux runs

Le comparateur superpose les courbes de qualité, scores, récompenses et durées
d’optimisation. Il compare aussi la patrouille, les causes de perte et toutes
les métriques d’évaluation. Une section appariée ne conserve que les quartets
`P/N + seed + cas` présents dans les deux runs ; elle montre les changements de
terminaison, de survivants et de durée sans mélanger des scénarios différents.

L’outil ne désigne un avantage observé que si les métadonnées disponibles
indiquent des protocoles compatibles. Un scénario, curriculum ou dernier
contexte P/N différent force une comparaison descriptive sans verdict qualité.

Les anciens runs sans `run-manifest.json` restent lisibles. La compatibilité est
alors inférée des journaux et le rapport indique explicitement les informations
manquantes.

Une comparaison de CI ne démontre jamais la convergence. La promotion d’une
politique exige une campagne séparée sur plusieurs graines indépendantes et
plusieurs compositions P/N.

## Sécurité et intégrité

- seuls `blue-mappo-checkpoints` et `blue-mappo-telemetry` sont téléchargés ;
- l’identifiant GitHub en mémoire n’est pas transmis à l’URL de stockage signée ;
- l’empreinte SHA-256 publiée par GitHub est vérifiée lorsqu’elle existe ;
- les chemins sortant du dossier cible et les liens symboliques ZIP sont
  refusés ;
- les téléchargements partiels ne remplacent pas un artefact validé.

## Tests

Depuis la racine du dépôt :

```powershell
$env:PYTHONPATH = "tools/blue_training_diagnostics"
py -3 -m unittest discover -s tools/blue_training_diagnostics/tests -p "test_*.py" -v
py -3 -m unittest discover -s tools/ci -p "test_mappo_metrics.py" -v
```

Les tests couvrent le parsing GitHub, les flux de télémétrie détaillés, les
agrégats par rôle, les comparaisons appariées, la compatibilité P/N, les rapports
autonomes et le refus d’un ZIP tentant une traversée de dossier.

