# PPOKit

Framework commun des exemples PPOTrial : PPO/MAPPO, observations,
récompenses, artefacts, évaluation, inférence et studio.

Il héberge trois démonstrateurs : RLMission (PPO 2D), Blue MAPPO
(MA-PPO 2D) et FlightControl3D (PPO 3D). RLSNA est un exemple PPO 2D
complémentaire du monde Airspace.

> Conception :
> [`docs/architecture/ppokit-framework-design.md`](../docs/architecture/ppokit-framework-design.md).
> Les constats de migration et leurs validations sont conservés dans
> [`docs/audits/ppokit-integration/`](../docs/audits/ppokit-integration/README.md).
> La phase corrective courante est consignée dans
> [`docs/audits/ppokit-stabilization-2026-11-27.md`](../docs/audits/ppokit-stabilization-2026-11-27.md).
> La qualification finale de la mutualisation est décrite dans
> [`docs/audits/ppokit-mutualization-final-2026-11-27.md`](../docs/audits/ppokit-mutualization-final-2026-11-27.md)
> et son état synthétique dans [`PROGRESS.md`](PROGRESS.md).

## Disponible aujourd'hui

### Un projet = un dossier

Un dossier de projet PPOKit décrit l'entraînement, l'évaluation et l'inférence.
Il peut aussi référencer des DLL, des paramètres ou des modèles partagés hors
de ce dossier, comme le font les démonstrateurs du dépôt.

```
reach/
  project.json          nom du projet et programme qui l'exécute (train / eval / play)
  task.json             observation, actions, canaux de récompense, scénario
  network.json          acteur et critique (blocs et fils) ; network.layout.json : positions
  training.json         réglages PPO et GAE, sondes
  curriculum.json       stages, critères, évaluations et répétitions (facultatif)
  eval/*.json           évaluations sur seeds indépendants
  components/           DLL utilisateur (.dll / .so) offertes dans l'éditeur de réseau
  checkpoints/          latest.ppokit, policy, metrics.jsonl (ignoré par git)
```

### Le studio en un clic

`Lancer_PPOKit_Studio.bat` (ou `ppokit-studio` sans argument) ouvre l'écran
de départ : « Open a project » (sélecteur de dossier natif sous Windows, ou un
chemin tapé), « New project » (nom + emplacement), et les projets récents.
`ppokit-studio <dossier>` ouvre directement un projet. Le studio ne parcourt
pas le disque : c'est l'utilisateur qui choisit le dossier.

Dans un projet :

- **Overview** : la boucle acteur-critique en un schéma (environnement,
  observation, acteur, actions, critique, V(s), récompenses, GAE, update PPO,
  curriculum). Survoler une boîte l'explique ; cliquer ouvre l'endroit où elle
  se règle. Sur une fenêtre étroite, la barre horizontale donne accès à tout le schéma.
- **Network** : l'éditeur de blocs (zoom à la molette ou par les loupes, « Fit »,
  déplacement au bouton du milieu). Les DLL de `components/` apparaissent dans
  la palette ; « Add a DLL... » en copie une dans le projet.
- **Training** : PPO et GAE, curriculum.
- **Task** : ce que voit l'agent, ce qu'il décide, ses récompenses.
- **Run** : santé de l'entraînement et courbes, suivies en direct.
- **Console** : la sortie du programme lancé.

Les boutons **Train**, **Evaluate** et **Inference** enregistrent le projet puis démarrent la commande de `project.json`. Le Studio cherche l’exécutable dans le projet, dans le registre du build puis dans le PATH. Avec `ppokit-run`, **Pause**, **Resume** et **Stop** passent par une session UDP locale. L’entraînement termine l’update en cours et publie atomiquement le learner, la politique, l’état de l’environnement et l’état du curriculum. En évaluation et en inférence, Stop prend effet au pas de simulation et une campagne inachevée ne publie aucun résultat. Le Studio observe la sortie réelle du processus, ferme le viewer qu’il a ouvert pour ce run et permet de régler le délai avant arrêt forcé (30 s par défaut). **Kill** termine immédiatement le processus.

Le curriculum affiché dans **Training** est celui du moteur commun : stage de
promotion et stage actif de répétition, progression de la fenêtre ou du budget,
critère, dernière évaluation indépendante et raison du maintien ou de la
promotion. Les mêmes décisions sont prises depuis Studio, CLI et CI. `--fresh`
repart du stage demandé ou du premier stage ; `--resume` restaure exactement la
génération publiée et refuse une configuration de curriculum différente.

Pour superviser une commande lancée dans un terminal, ouvrir **Run → Connect a command-line run**, copier l’argument privé `--control` et le joindre à la commande. Le terminal reste propriétaire du processus. Les projets générés par le SDK déclarent `"controlProtocol": "udp/1"`. Le runner d’exemple et FlightControlHeadless acceptent `--control`. Un client historique qui dépend des fichiers ou de l’annonce TCP peut activer `--legacy-control` en ligne de commande. Pour un ancien exécutable externe, la déclaration explicite `"controlProtocol": "legacy-files"` réactive le comportement antérieur du Studio.

#### Protocole de contrôle UDP v1

Les deux sockets sont non bloquantes et liées à `127.0.0.1`. Le Studio réserve son port avant le lancement, génère un identifiant aléatoire de 128 bits et fixe le port du worker au premier HELLO valide. L’argument est omis du journal de la console. Un datagramme contient au plus 1 200 octets, dont un en-tête de 44 octets en ordre réseau.

| Octets | Champ |
|---|---|
| 0–3, 4–5, 6–7 | Magic `PPKC`, version 1, type |
| 8–23 | Identifiant de session |
| 24–31, 32–39, 40–43 | Numéro de séquence, séquence acquittée, longueur exacte de la charge |

Les types 1 à 12 sont HELLO, HELLO_ACK, HEARTBEAT, STATUS, PAUSE, RESUME, STOP, SET_REWARDS, ACK, NACK, STOPPED et ERROR. ERROR contient le diagnostic terminal. STATUS transporte un octet représentant Starting, Running, Paused, Stopping, Saving ou Stopped (0 à 5). SET_REWARDS transporte un tableau JSON de couples `[indice_du_canal, poids]`. L’indice commence à zéro et suit l’ordre des récompenses du contrat de tâche. ACK distingue la réception de l’application effective. NACK donne le motif du refus. Une commande en vol est retransmise toutes les 120 ms et dédupliquée par séquence. STOP est prioritaire. Le heartbeat est émis toutes les 250 ms. Après 3 s sans superviseur, un worker en pause reprend. Chaque tick lit au plus 16 datagrammes. L’état du processus système confirme la fin, même si le dernier datagramme se perd.

Un run montre d'abord s'il va bien : son état (« training, last update 12 s
ago » ou « stopped »), son temps d'entraînement total (reprises comprises) et
des cartes de santé colorées avec une phrase claire chacune : retour par
épisode et sa tendance, précision du critic (variance expliquée), changement
de politique comparé à la cible KL, échantillons clippés, pas d'acteur sautés.
Ce sont des repères pour savoir où regarder, pas des preuves : seule une
évaluation sur des seeds indépendants dit qu'une politique a appris
(`editor/run_health.hpp`).

Les sondes marchent pour les trois : FlightControl3D et Blue MAPPO sondent
leurs réseaux toutes les 10 updates (Blue : blocs du graphe acteur, attention
comprise, et couches du critic ; Flight : couches des MLP rl_tools), les
exemples selon `training.json`. Elles ne font que lire : une update sondée
donne exactement les mêmes poids (testé au bit près). Blue accepte les poids
changés depuis le studio, avec les règles des curseurs du jeu (0 à 10, pour
la session).

### FlightControl3D et Blue MAPPO sont des projets

`Examples/FlightControl3D/projects/T6`, `PC7` et `Examples/BlueDefense/projects/BluePatrol`
s'ouvrent dans le studio comme les autres et tournent sous `ppokit-run`. Ils lisent
`scenario.json` dans le projet (`"settings"` de `project.json`) et gardent leurs
checkpoints dans `checkpoints/` ; Blue lit aussi `blue_patrol_training.json`.
Son projet principal utilise le réseau partagé éditable `ppokit_graph_network` et
les checkpoints convertis dans `checkpoints/shared/`.

Dans Studio, les réglages numériques et booléens de l'environnement Blue sont
modifiables et validés par son chargeur avant sauvegarde. Son curriculum V3 et
le calendrier progressif de la CI sont des configurations du même moteur
`CurriculumEngine`. La CI ne décide plus des promotions : elle génère au besoin
une configuration canonique puis appelle `ppokit-run`, comme le Studio et la CLI.

BluePatrol s'entraîne et s'évalue avec `ppokit-run` : DLL d'environnement
`blue_environment` (le monde de BlueDefense, P bleus contre N rouges), DLL réseau
d'équipe `ppokit_graph_network` (le learner partagé et les checkpoints convertis),
suite `eval/validation.json` (la campagne de son fichier d'entraînement),
Visualiseur `blue_visualiser` (vue 2D de dessus dans ppokit-viewer : ce que la
mini-carte tactique de l'ancien PPO.exe montrait, plus les lidars et les explosions de sa vue 3D).
Son inférence est un `ppokit-run play` de l'acteur de production, épisode après
épisode ; le studio ouvre alors ppokit-viewer sur le monde, et son Stop interrompt l’inférence au prochain pas de simulation.

**Launch settings...** ouvre un panneau défilant par-dessus la zone de travail. Les choix restent actifs pour les lancements de cette fenêtre ; **Done** ou un clic à l’extérieur ferme le panneau. Pour Train, choisir le mode du projet, **Resume checkpoint** ou **From scratch**, puis le budget du projet, un nombre d’updates supplémentaires ou une update cible absolue. Une reprise explicite exige un checkpoint. From scratch archive le run précédent. Les choix d’évaluation et d’inférence sont indépendants ; les références détaillées sont regroupées sous **Policy reference details**. Pour un projet muni d’un Visualiseur, Train ouvre automatiquement sa vue. Celle-ci montre les épisodes collectés à la vitesse de l’entraînement ; elle peut rester figée pendant l’optimisation et les sauvegardes, puis reprendre à la collecte suivante. Le bouton **View** permet de la rouvrir pendant le run.

Les sélecteurs **Evaluation policy** et **Inference policy** mémorisent chacun le choix du lancement dans la fenêtre du projet. Le choix par défaut suit `project.json`, y compris depuis une commande CLI sans sélecteur. Un `--model` ou un `--policy` explicite peut le remplacer. **LATEST** ouvre l’acteur du dernier entraînement. **BEST** ouvre la promotion de la suite demandée ou, en l’absence de promotion, sa référence `HISTORICAL_BEST` lorsqu’elle est déclarée. **Custom file** ouvre le chemin choisi. Les rôles immuables supplémentaires sont déclarés dans `models.json`. Le runner vérifie le digest de chaque référence, y compris lorsqu’on la sélectionne par son chemin. La console affiche le rôle, le chemin résolu, l’update connue, le SHA-256, le contrat et les modules réellement chargés. Le rapport d’évaluation conserve l’origine de la référence. En ligne de commande, utiliser `--model LATEST`, `--model HISTORICAL_BEST` ou un autre rôle déclaré.

Le T-6 et le PC-7 (`Examples/FlightControl3D/projects/T6`, `PC7`) sont des projets
`ppokit-run` : DLL d'environnement `flight_environment`, DLL réseau `ppokit_graph_network`,
Visualiseur `flight_visualiser` (la scène 3D, le profil et les panneaux de
FlightControl3D dans ppokit-viewer), suite `eval/validation.json` (validation et BEST).
Le projet T-6 embarque le BEST 870 de master `c8282b8` au niveau 12.
Son contrat `t6texan2@8-centreline1200` conserve le guidage de cette génération.
Les scénarios historiques du T-6 340 et du PC-7 610 utilisent l'interception à 2 000 m et le
niveau 11. L’entraînement du PC‑7 démarre vierge. Le Studio peut immédiatement jouer le BEST 610 issu du run 45460292649. Le fichier de référence pour le scénario actuel porte explicitement le contrat à 1 200 m. Les octets des tenseurs correspondent au paquet historique à 2 000 m.
Seul
`train` fait avancer leur curriculum (API d'environnement 5, `set_learning`). Son
état (niveau, fenêtre de promotion, flux aléatoire des départs) est sauvegardé avec
chaque checkpoint (`checkpoints/environment_state/<update>.json`, API
`training_state` / `restore_training_state`) et repris avec lui ; `train --fresh`
(ou un entraînement sans checkpoint) archive l'ancien run dans
`checkpoints/archive/run-<date>/` et repart du niveau 0.

```
ppokit-run train --project Examples/FlightControl3D/projects/T6            # reprend le T-6
ppokit-run eval  --project Examples/FlightControl3D/projects/T6
ppokit-run play  --project Examples/FlightControl3D/projects/T6 --best --realtime
ppokit-run train --project Examples/BlueDefense/projects/BluePatrol --updates 100
ppokit-run eval  --project Examples/BlueDefense/projects/BluePatrol --model PROD
ppokit-run train --project Examples/FlightControl3D/projects/PC7           # premier entraînement du PC-7
ppokit-run play --project Examples/BlueDefense/projects/BluePatrol --model PROD --realtime --step-seconds 0.004
ppokit-viewer --view <build>/Examples/BlueDefense/Release/blue_visualiser.dll --port 47001
```

Les projets T6 et PC7 déclarent `"runProfile": "environment/1"`. Leur DLL lit
le scénario avec le même parseur que FlightControl3D. En entraînement,
`maxUpdates` donne la cible absolue et `checkpointEvery` la fréquence de
publication. `--updates N` ajoute N mises à jour, `--until-update N` fixe une
cible absolue et `--save-every N` remplace la fréquence. `--resume` exige le
checkpoint du projet. `--fresh` archive le run précédent. En inférence, le
profil choisit les seeds, la taille de la campagne et les départs sur toute la
difficulté. L'évaluation conserve la frontière du curriculum enregistré.

L'extension optionnelle est décrite dans
`include/ppokit/environment/run_profile.h`. Un projet qui la déclare exige
les deux symboles versionnés de sa DLL. Les projets sans profil conservent
leurs valeurs de lancement.

### Démarrer un projet : `ppokit-new`

```
ppokit-new lander --into Examples        # écrit Examples/lander/, qui s'entraîne déjà
```

Ajouter `add_subdirectory(Examples/lander)` au build (le studio connaît alors
sa DLL `lander_environment`), puis ouvrir `Examples/lander` dans le studio,
ou en ligne de commande :

```
ppokit-run check --project Examples/lander   # valide task, network, training, curriculum, eval/
ppokit-run train --project Examples/lander   # 100 updates ; relancer continue (--fresh repart)
ppokit-run eval  --project Examples/lander   # seeds indépendants, code 0 si les seuils tiennent
ppokit-run play  --project Examples/lander   # rejoue la politique et affiche chaque épisode
```

Le projet généré fournit une petite DLL d'environnement conforme au SDK, un
graphe acteur/critique éditable et un `curriculum.json`. Il ne crée aucun
learner ou exécutable PPO parallèle : `ppokit-run` fournit entraînement,
évaluation, inférence, journal, reprise exacte et contrôle Studio.
`PPOKit/examples/reach` est la sortie de `ppokit-new reach`, gardée identique
par un test.

### `PPOKit::Artifacts` — fichiers d'artefacts transactionnels

Un lecteur ne voit jamais un checkpoint à moitié écrit :

```cpp
#include "ppokit/artifacts/atomic_file.hpp"

struct PolicyWriter
{
    Policy const& policy;
    void operator()(std::ostream& stream) const { policy.Write(stream); }
};

// `model.txt.tmp` est écrit, puis remplace `model.txt` ; en cas d'échec,
// l'ancien fichier reste intact et une ArtifactWriteError décrit la cause.
ppokit::artifacts::WriteFileAtomically("models/model.txt", PolicyWriter{policy},
    ppokit::artifacts::AtomicWriteOptions{}, "MyExample checkpoint");

// Pour garder la génération précédente à côté (`model.txt.previous`) :
auto const options = ppokit::artifacts::AtomicWriteOptions::KeepingPrevious(".previous");
```

CMake : `target_link_libraries(<cible> PRIVATE PPOKit::Artifacts)`.

### `PPOKit::Core` — briques PPO (header-only)

```cpp
#include "ppokit/core/gae.hpp"
#include "ppokit/core/ppo_surrogate.hpp"
#include "ppokit/heads/gaussian.hpp"

// GAE, en remontant la trajectoire : `terminated` coupe le bootstrap,
// `truncated` (limite de temps) le garde mais arrête la trace.
advantage = ppokit::core::GaeStep(reward, value, nextValue, advantage, gamma, lambda, terminated, truncated);

// Politique gaussienne, axe par axe (un axe inactif est simplement sauté).
logPi += ppokit::heads::GaussianAxisLogDensity(sample, mean, logStd);

// Surrogate PPO clippé, puis règle de chaîne vers les sorties de l'acteur.
auto const surrogate = ppokit::core::EvaluateClippedSurrogate(logPi, oldLogPi, advantage, clip);
auto const gradient = ppokit::heads::GaussianAxisLogDensityGradient(sample, mean, logStd);
double const dLossDMean = surrogate.logProbabilityGradient * gradient.mean;
```

CMake : `target_link_libraries(<cible> PUBLIC PPOKit::Core)`.

FlightControl3D et BlueDefense utilisent exactement ces fonctions.

### `PPOKit::Schema` et `ppokit-codegen` — décrire une tâche dans `task.json`

Un fichier `task.json` décrit la forme de ce que l'agent observe et fait :
groupes d'observation (champs nommés, vecteurs, ensembles d'entités masqués),
entrée du critique, groupes d'actions, canaux de récompense. Exemples complets :
[`Examples/FlightControl3D/task.json`](../Examples/FlightControl3D/task.json),
[`Examples/BlueDefense/Agents/BluePatrol/task.json`](../Examples/BlueDefense/Agents/BluePatrol/task.json).

```json
{
  "schema": "ppokit.task/1",
  "name": "my_task",
  "observation": {
    "own":    { "fields": ["speed", "heading_error"] },
    "lidar":  { "size": 61 },
    "tracks": { "set": { "capacity": 10, "features": 13, "mask": "last" } }
  },
  "actions": { "motion": { "head": "gaussian", "names": ["turn", "acceleration"], "bounds": [-1, 1], "map": "clip" } },
  "rewards": { "Progress": {}, "Crash": { "label": "Crash penalty", "weight": 10 } }
}
```

CMake génère un en-tête de layout à chaque modification du fichier :

```cmake
ppokit_generate_task_layout(TARGET my_target TASK task.json HEADER my_task_layout.hpp)
```

```cpp
#include "my_task_layout.hpp"
using Layout = ppokit::tasks::my_task::TaskLayout;
float const lidarFirst = observation[Layout::Observation::Lidar::Offset];
static_assert(Layout::Action::Size == 2);
```

Une faute dans le fichier est signalée avec son chemin exact :

```
ppokit-codegen: task.json: observation.tracks.set.capacty: unknown key (accepted: capacity, features, fields, mask)
```

### Réseaux décrits dans `graph.json`

`graph.json` décrit l'acteur et le critique. Leurs tailles d'entrée et de sortie
viennent de `task.json`, donc les deux fichiers ne peuvent pas se contredire.
Un réseau fourni par ailleurs reste possible (`"model": "external"`). C'est
le cas de l'acteur à attention de BlueDefense, décrit en blocs dans
`graph_blocks.json` (`ppokit.graph/2`, voir `docs/architecture/ppokit-graph-blocks.md`).

```json
{
  "schema": "ppokit.graph/1",
  "actor":  { "model": "mlp", "hidden": { "layers": 2, "width": 128 }, "activation": "tanh",
              "log_std": "state_independent" },
  "critic": { "model": "mlp", "input": "observation", "hidden": { "layers": 2, "width": 128 },
              "activation": "tanh" }
}
```

```cmake
ppokit_generate_networks(TARGET my_target TASK task.json GRAPH graph.json HEADER my_networks.hpp)
```

```cpp
#include <rl_tools/operations/cpu_mux.h>   // d'abord les opérations rl_tools
#include "my_networks.hpp"
using Actor = ppokit::tasks::my_task::networks::Actor::Network<Index, BatchSize>;
```

Les types générés sont exactement ceux que les exemples écrivaient à la main :
le MLP de FlightControl3D chargé depuis un checkpoint de `FlightLearner` donne
les mêmes sorties au bit près, et le critique de BlueDefense est le même type
rl_tools.

Équipe (MAPPO) : le backend commun `ppokit_graph_network` entraîne aussi une
tâche d'équipe (`"agents": {"max": N}` avec N > 1). La section `team_critic`
de `network.json` dit comment il construit l'entrée du critique centralisé à
partir d'un bloc vecteur de l'acteur, calculé pour chaque membre de l'équipe :
sa ligne (nulle une fois détruit), la moyenne et le maximum sur les vivants,
puis les vivants et les membres rapportés à N. Chaque partie remplit un groupe
de `"critic"` de `task.json` :

```json
"team_critic": { "features": "critic_features", "own": "critic.own", "mean": "critic.team_mean",
                 "max": "critic.team_max", "living_fraction": "critic.living_fraction",
                 "team_fraction": "critic.team_fraction" }
```

Les membres d'un épisode sont les agents présents à son premier pas ; une
ligne jamais présente est ignorée (ni action ni mise à jour). Exemple :
`tests/fixtures/team_graph`.

### Entraîner une tâche : collecte, rollout et learner PPO

En mode *Owned*, l'exemple écrit seulement sa tâche (`Reset`, `Step`) et
PPOKit mène la boucle. Le learner rl_tools s'appuie sur les réseaux générés
depuis `graph.json` :

```cpp
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include "my_networks.hpp"
#include "ppokit/backend/rltools/gaussian_mlp_learner.hpp"

using Learner = ppokit::backend::rltools::GaussianMlpLearner<networks::Actor, networks::Critic>;
Learner learner(settings);                      // PpoSettings : gamma, lambda, clip, learning rates…
Learner::Buffer buffer(2048);                   // alloué une fois
ppokit::runtime::RolloutCollector<MyTask, Learner, Learner::Buffer> collector(task, learner);
for (int update = 0; update < 100; ++update)
{
    collector.Collect(buffer);                  // épisodes enchaînés, troncatures bootstrapées
    auto const statistics = learner.Update(buffer);   // pertes, KL, clip fraction, variance expliquée
}
auto const action = learner.Mean(observation);  // politique déterministe
```

Exemple complet : `PPOKit/tests/backend_learner_tests.cpp` (couloir jouet
résolu en 30 updates, de façon reproductible à partir du seed).

### Régler PPO : section `ppo` de `train.json`

Les hyper-paramètres PPO se règlent dans `train.json`, sans recompiler. Un
paramètre absent garde sa valeur par défaut :

```json
{
  "schema": "ppokit.train/1",
  "ppo": { "gamma": 0.995, "rollout_steps": 4096, "actor_learning_rate": 1e-4 }
}
```

```cpp
#include "ppokit/schema/train_schema.hpp"
auto const plan = ppokit::schema::LoadTrainingPlan("train.json", task);
Learner learner(plan.ppo.Settings());            // PpoSettings
Learner::Buffer buffer(plan.ppo.RolloutSteps());
```

Paramètres : `gamma`, `lambda`, `rollout_steps`, `minibatch`, `epochs`,
`actor_learning_rate`, `critic_learning_rate`, `clip`, `target_kl` (0 : coupé),
`entropy_coefficient`, `value_coefficient` (poids de la perte de valeur,
utilisé quand le critique partage des couches de l'acteur),
`max_gradient_norm` (0 : coupé ; norme globale maximale des gradients d'un pas
d'optimiseur, comme `clip_grad_norm_` de torch ; réseaux en graphe seulement,
les autres learners refusent une valeur non nulle),
`initial_log_std`, `seed` (catalogue unique, avec
bornes et défauts : `ppokit/core/ppo_parameters.hpp`). Une erreur nomme la
clé et dit quoi corriger : `train.json: ppo.gama: unknown PPO parameter (did
you mean 'gamma'?)`, `ppo.clip: 3 is outside [0.01, 1]`.

Arithmétique d'Adam (réseaux en graphe) : section `optimizer`, facultative.
`"adam": "reference"` (défaut, epsilon 1e-8) divise les moments par
1 - beta^t ; `"adam": "rltools"` fait les opérations dans l'ordre de
`rl_tools::nn::optimizers::Adam` (epsilon 1e-7 par défaut), pour qu'un
learner en graphe reprenne bit à bit les pas d'un learner rl_tools.
`"epsilon"` remplace l'epsilon par défaut :

```json
{ "schema": "ppokit.train/1", "optimizer": { "adam": "rltools" } }
```

Dans le studio : section « Training » d'un projet (fichier `training.json`).
Paramètres groupés (retours, données par update, taille de
pas, exploration, reproductibilité), infobulle sur chacun (rôle, valeurs
typiques, symptômes, défaut et bornes), curseur logarithmique pour les
learning rates, boutons /2 et x2 pour les tailles, bouton « Use default »,
annuler/rétablir (Ctrl+Z / Ctrl+Y), Ctrl+S. Sous les paramètres, des
remarques en clair : horizon impliqué par gamma, travail d'une update,
minibatch plus grand que le rollout, learning rate très grand… ; un
paramètre concerné par un avertissement est surligné. Le reste du fichier
(curriculum) est conservé tel quel ; seuls les paramètres choisis sont
écrits.

### Composants DLL boîte noire

Un composant est un bloc calculé par une DLL : PPOKit ne connaît que ses ports
et ses capacités. On l'écrit en C++ avec le SDK (un seul en-tête, aucune
dépendance) :

```cpp
#include "ppokit/component/sdk.hpp"
class Scale final : public ppokit::sdk::Component
{
public:
    static ppokit::sdk::Manifest Describe()
    {
        ppokit::sdk::Manifest manifest("scale", 1);
        manifest.Input("x", {"N"}).Output("y", {"N"});   // "N" : fixé par le câblage
        manifest.Trainable();                           // l'optimiseur de PPOKit met à jour ses paramètres
        manifest.Number("initial", 1.0, -10.0, 10.0);   // réglage de graph.json, avec défaut et plage
        return manifest;
    }
    Scale(ppokit::sdk::Config const& config, ppokit::sdk::Shapes const& shapes, std::uint64_t seed);
    void Forward(ppokit::sdk::Call const& call) override;
    void Backward(ppokit::sdk::GradientCall const& call) override;   // car Trainable()
    float* Parameters(std::size_t& count) override;
};
PPOKIT_COMPONENTS(Scale)                                // exporte la table C
```

CMake : `add_library(scale MODULE scale.cpp)` +
`target_link_libraries(scale PRIVATE PPOKit::ComponentSdk)`. Dans
`graph.json` :

```json
"components": ["scale"],
"actor": { "nodes": { "scaled": { "type": "component:scale@1", "config": { "initial": 2 } }, ... },
           "edges": [ ["state", "scaled"], ["scaled", "trunk"], ... ] }
```

`"scale"` trouve `scale.dll` ou `libscale.so` à côté de `graph.json` ; dans un
projet, la référence est `"components/scale.dll"`. Un
ensemble (`[N, F]`) est accompagné de son masque, déclaré par `Mask(...)` et
relié tout seul. Règles vérifiées à la lecture, avec un message qui dit quoi
faire : un composant sans gradient doit venir avant tout bloc entraînable ;
port, réglage ou version inconnus listent ce qui existe ; une exception de
la DLL remonte avec son texte (« forward failed: sensor offline »). Pour
l'instant, dans un graphe : une sortie float par composant, et pas de
composant à état (leurs sorties devraient être stockées dans le rollout).
Studio : section « Components » de la palette. Les bibliothèques de
`components/` y sont listées (« Use »), « Add a DLL... » en copie une dans le
projet ; réglages du manifeste dans l'inspecteur.

### Éditer le curriculum dans le studio

Dans un projet, « Training > Curriculum » (ou la boîte Curriculum de
l'Overview) :

- fenêtre de promotion, seuil (en % de succès) et part de répétition des
  étapes précédentes, avec une infobulle chacun ;
- à gauche, les étapes dans l'ordre d'entraînement : Add (copie de l'étape
  choisie, juste après), Remove, Up, Down ;
- à droite, l'étape choisie : nom, fenêtre et seuil propres si besoin, et
  une ligne par paramètre de scénario du `task.json` avec sa règle (Task
  default, Fixed, Uniform, Normal, Choice) et les champs de cette règle
  (case pour un booléen, liste pour une énumération, bornes du paramètre
  respectées) ;
- en haut, « Valid » ou le message exact que donnerait l'entraînement
  (`curriculum.stages[1].name: ...`) ;
- en bas, un aperçu : 2000 scénarios tirés de l'étape par le moteur de
  curriculum lui-même, un histogramme par paramètre.

Save, Undo (Ctrl+Z) et Redo (Ctrl+Y) valent pour tout le fichier ; un
glisser compte pour un seul changement.

### Suivre un run : `metrics.jsonl` et le studio

Le journal de run garde, update après update, les statistiques PPO et un
résumé des récompenses par canal (par pas brut et pondéré, par épisode,
cumul). Pour les nouveaux journaux, `episode_return_mean` est la moyenne des
retours effectivement transmis au learner sur les épisodes terminés de chaque
agent. Les contributions par canal restent mesurées avant un éventuel bornage
du total par la tâche. Une ligne JSON est écrite par événement.

```cpp
#include "ppokit/runtime/run_log.hpp"
ppokit::runtime::RewardLedger ledger(weights, 1);          // un flux par agent
collector.Track(&ledger);                                  // la tâche peut exposer LastRewardChannels()
ppokit::runtime::RunLogWriter journal("run/metrics.jsonl", {"my_task", channels, PpoSettingsValues(settings)});
// à chaque update :
journal.WriteUpdate({update, elapsedSeconds, ppokit::runtime::PpoMetrics(statistics), ledger.Summarize()});
// au changement de poids, à une frontière d'update :
journal.WriteWeights({update + 1, previous, weights});
```

L’onglet **Reward weights** envoie les changements au prochain point de publication de l’entraînement. Avec `ppokit-run`, le transport est la session UDP. Le worker valide les poids, les applique à la frontière d’update et retourne l’update d’application. Le journal conserve l’historique et le manifeste de génération conserve le vecteur effectif. Pour un ancien client utilisant la boîte aux lettres sur disque, l’API de transition reste disponible.

Pour BluePatrol, chaque poids modifié doit rester entre 0 et 10, comme dans
l'interface historique. Un poids configuré supérieur à 10, notamment `OwnLoss`
à 24 par défaut, est conservé lorsqu'un autre canal est modifié.

```cpp
#include "ppokit/runtime/weight_request.hpp"
header.acceptsWeightRequests = true;                       // le studio propose l'édition
ppokit::runtime::WeightRequestInbox inbox("run/metrics.jsonl", channelNames);
// après chaque update, buffer vide :
if (auto request = inbox.Take()) { /* valider, appliquer, journal.WriteWeights(...) */
    inbox.Answer({request->id, true, "Applied from update 13."}); }
```

Flight le fait quand il est lancé avec un fichier de scénario (les poids
sont aussi enregistrés dans ce fichier, comme avec les curseurs du bureau).
Blue garde son réglage en direct existant ; il ne lit pas ces requêtes.

Sondes du réseau : avec `"probes": { "every": 10 }` dans `train.json` (ou la
case « Probe the network blocks » des réglages du studio), une update sur
dix mesure chaque bloc de l'acteur et du critique : dispersion et moyenne
des sorties, part d'unités mortes (ReLU à zéro), part d'unités saturées
(|x| ≥ 0,99), taille du gradient reçu. L'onglet « Probes » du tableau de
bord trace ces mesures par bloc et dit en clair ce qui cloche (« 90 % of
the outputs of 'trunk' are exactly zero… », « Almost no gradient reaches
'encoder'… »). Sondes coupées : coût nul (un test de pointeur par
échantillon, aucune allocation) ; elles ne changent jamais l'entraînement.

```cpp
#include "ppokit/graph/probes.hpp"
ppokit::graph::GraphProbes actorProbes(learner.Actor()), criticProbes(learner.Critic());
if (ppokit::schema::ProbesDue(plan.probeEvery, update)) learner.SetProbes(&actorProbes, &criticProbes);
auto const statistics = learner.Update(buffer);
learner.SetProbes(nullptr, nullptr);
journal.WriteProbes({update, "actor", actorProbes.Summary()});
```

`RunLogWriter::Resume(path, header, updates)` continue le journal d'un run
repris (l'historique au-delà de l'update reprise est retiré). Flight et Blue
écrivent ce journal pendant l'entraînement. Le studio le montre dans la section
« Run » d'un projet (`checkpoints/metrics.jsonl` : courbes par canal,
statistiques PPO, paramètres, avec une infobulle par nom).

### Checkpoints versionnés

Un checkpoint PPOKit commence par une ligne d'identité (`ppokit.artifact/1`) :
type (`training` pour reprendre, `policy` pour l'inférence), tâche et
réseaux (hashes et résumé lisible), nombre d'updates, seed, date.

```cpp
auto const identity = ppokit::schema::ArtifactIdentityOf(task, graphs);
learner.SaveTraining("model/latest.ppokit", identity);   // à une frontière d'update, atomique
learner.LoadTraining("model/latest.ppokit", identity);   // la reprise donne les mêmes poids, au bit près
learner.SavePolicy("model/policy.ppokit", identity);     // l'acteur seul
```

Un checkpoint d'une autre tâche ou d'un autre réseau est refusé en disant
ce qui a changé (« the actor had blocks state:input, trunk:mlp,
policy:gaussian_head; graph.json now has ... (added: extra:dense) »). Un
fichier abîmé laisse le learner intact.

```
ppokit-artifact inspect model/latest.ppokit          # ce qu'il contient
ppokit-artifact check model/latest.ppokit task.json graph.json   # se charge-t-il ?
```

Flight et Blue gardent leurs formats actuels, qu'ils lisent directement
(les modèles en cours se reprennent tels quels).

### Évaluer une politique sur des seeds indépendants

Une suite d'évaluation est un fichier :

```json
{
  "schema": "ppokit.eval/1",
  "suite": "corridor_smoke",
  "seeds": [1001, 1002, 1003],
  "episodes": 2,
  "policy": "deterministic",
  "gates": { "success_rate": ">= 1.0", "mean_length": "<= 8" }
}
```

```cpp
#include "ppokit/eval/evaluator.hpp"
auto const suite = ppokit::eval::LoadEvaluationSuite("eval/smoke.json");
ppokit::eval::RequireDisjointSeeds(suite, trainingSeeds);   // refuse un seed d'entraînement
auto const report = ppokit::eval::Evaluate(suite, MakeTask, learner, maximumSteps);
std::cout << report.ToJson();                               // métriques, gates, épisodes
```

Métriques intégrées : `mean_return`, `success_rate` (épisodes terminés),
`truncation_rate`, `mean_length`. La politique n'est jamais échantillonnée
pendant l'évaluation, ce qui laisse intact le générateur aléatoire de
l'entraînement. CMake : `PPOKit::Evaluation`.

### Scénarios et curriculum

`task.json` déclare ce qui varie d'un épisode à l'autre :

```json
"scenario": {
  "wind_kt":   { "type": "float", "range": [0, 25], "default": 0 },
  "start":     { "type": "enum", "values": ["rolling", "airborne"], "default": "airborne" },
  "gusts":     { "type": "bool", "default": false },
  "opponents": { "type": "int", "range": [0, 10] }
}
```

`train.json` décrit le curriculum, du plus facile au plus difficile. Un
paramètre non mentionné par une étape prend sa valeur par défaut :

```json
{
  "schema": "ppokit.train/1",
  "curriculum": {
    "window": 200, "promote_at": 0.8, "rehearse": 0.1,
    "stages": [
      { "name": "calm",  "sample": { "opponents": 0, "start": "rolling" } },
      { "name": "windy", "sample": { "opponents": { "uniform": [1, 3] }, "wind_kt": { "uniform": [5, 15] },
                                     "start": { "choice": { "rolling": 1, "airborne": 3 } } } },
      { "name": "full",  "window": 400, "sample": { "opponents": { "normal": { "mean": 8, "std": 4 } } } }
    ]
  }
}
```

Une étape est promue quand le taux de succès sur une fenêtre d'épisodes de
cette étape atteint `promote_at`. La fenêtre repart à zéro après chaque
décision. Une part `rehearse` des épisodes rejoue une étape précédente sans
compter pour la promotion.

```cpp
ppokit::runtime::Curriculum curriculum(*ppokit::schema::LoadTrainingPlan("train.json", task).curriculum);
ppokit::runtime::LoadCurriculumStage("latest/stage.txt", curriculum);   // reprise
auto const draw = curriculum.Sample(rng, values);
auto const scenario = Layout::Scenario::From(values);                  // champs typés générés
// ... épisode ...
if (curriculum.Record(draw, success)) { ppokit::runtime::SaveCurriculumStage("latest/stage.txt", curriculum); }
```

## Organisation

```
PPOKit/
  include/ppokit/<module>/   en-têtes publics, sans rl_tools ni GUI
  src/<module>/              implémentations
  tests/                     tests sans dépendance, lancés par CTest
```
