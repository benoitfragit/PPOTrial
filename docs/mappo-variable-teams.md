# MAPPO defender teams: architecture and migration

This document describes the generalized Blue P/N learner introduced on 2026-10-29.
The legacy RLMission and historical 2-vs-2 PPO training interfaces are separate
and should not be confused with the generalized team runtime.

## Team sizes and shared model

The generalized learner accepts 1-9 blue mobiles and 0-10 red mobiles.
Each episode has immutable initial P/N membership, stable slot assignments,
and changing alive masks. The actor is one **shared** 128-wide model: it
consumes the observation of one blue, with masked red/blue track attention.
The same actor weights evaluate every active blue and any permitted team size.

The **centralized critic** is also one shared model, but it predicts a distinct
value for each surviving blue. The critic has two hidden layers of 256 units,
one scalar output and a 470-feature fixed-width input:

| Input portion | Features | Meaning |
| --- | ---: | --- |
| Own compact observation | 156 | Evaluated blue's state, role, patrol, LiDAR and attention contexts |
| Mean of alive blue compact observations | 156 | Permutation-invariant collective context |
| Maximum of alive blue compact observations | 156 | Additional masked summary of the entire team |
| Living blue count fraction | 1 | Current survivors / maximum capacity (9) |
| Initial blue count fraction | 1 | Episode participants / maximum capacity (9) |

The critic prediction is `V(s_team, blue_i)`; per-blue rewards and GAE
targets correspond to the same selected blue. Dead and absent rows cannot
contribute to either aggregate or optimization. If no blue is alive, the
aggregates and critic row inputs are zero.

The **network input dimensions never change** with P/N. A V6 full training
state may now be resumed with different valid team counts because rollout
buffers are constructed for the new episode and the shared weights and
optimizer state retain their shape. A checkpoint records its source P/N,
not a model cardinality constraint. This enables experiments across team
sizes, but does not by itself prove policy generalization: evaluate with
multiple P/N configurations and independent seeds.

## Shared pseudo-LiDAR

World boundaries, hostile restricted-zone circles and living friendly mobiles
are combined into the existing 61-ray 360-degree pseudo-LiDAR. Same-side
mobiles other than the observer are treated as **nonlethal**, dynamic circular
safety envelopes. The TwoVsTwo rules enable these at 2,000 metres by default,
using `Parameters::cautionDistance`; other existing callers retain the
default radius 0 and the previous sensor behavior.

Friendly envelopes are **not** RestrictedZone entities and do not themselves
destroy any entity. Physical mobile-mobile collisions are still evaluated
separately at the existing 1,000-metre threshold. A ray starting inside an
envelope reports the exit intersection to preserve an escape direction.

Historical dense separation penalty shaping was removed. The sensor tells
the policy where the danger is, but avoiding collisions still has to be learned
from the mission-failure and lost-blue rewards; this is not a guarantee
of collision-free behavior.

## Defending the objective

The pure `ObjectiveThreatRisk` function gives each active red a normalized
risk using objective distance, heading/closing speed, protected-radius
penetration and time-to-intrusion. It reuses the existing defense JSON
settings and supports arbitrary P/N team sizes.

The generalized observation's threat pressure uses the maximum objective
risk across active reds. The generalized cooperative reward includes a
dense `ThreatContainment` channel equal to the **change in mean danger
across original red participants** from before to after a step. Unavailable
or neutralized threats contribute zero. Since the denominator is the initial
red count, neutralizing the most dangerous red gives a larger immediate
risk-reduction reward than neutralizing a distant red. If N=0 it is zero.
This channel shares the existing `Zone protection` weight.

The previous sparse `RedNeutralization` reward, objective-loss penalty,
mission outcome and control/role channels remain present. Training logs
should monitor the new channel independently so the reward scales can be
verified instead of inferring convergence from motion alone.

## Checkpoint compatibility

- **Inference actor**: keeps the existing generalized V8 binary/text
  parameter dimensions (128-wide). It can be loaded with its old weights.
  The observation semantics and rewards have changed, so retraining and
  evaluation are recommended.
- **Full generalized training state**: V6
  (`BLUE_TEAM_TRAINING_V6`) contains the new 470-input/256-wide critic.
  Same architecture can resume across different P/N episode counts.
- **Historical V5 full training state**: the runtime keeps an untouched
  `.v5.bak` and migrates the compatible V8 actor, while creating a fresh
  critic and optimizer state. The update counter restarts at zero. The
  migration is logged explicitly. Incompatible or malformed files must fail
  rather than being silently accepted.
- **Historical V4/v7 actor migrations**: retain their earlier strict
  contracts and do not modify the legacy model classes.

## Regression coverage and open evaluation

Automated tests cover 1, 2, 3, 4 and 9 blue slots; 0, 1 and multiple
red threats; permutation-invariance of team context; individual critic
inputs; dead-agent masking; cross-team-size V6 restore; sensor nearest
hit, inside envelope, 360-degree coverage, destroyed/wrong-side exclusion;
and objective-risk ordering.

Still needed to establish learning quality: multi-seed training comparisons,
per-blue explained critic variance, success and objective survival, collision
frequency, minimum blue separation, risk-weighted red neutralization,
advantage/gradient diagnostics and deterministic evaluations.

## Scientific context

Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
Games*, NeurIPS 2022 (MAPPO); Zaheer et al., *Deep Sets*, NeurIPS 2017
(permutation-invariant set functions); Iqbal & Sha, *Actor-Attention-Critic
for Multi-Agent Reinforcement Learning*, ICML 2019 (agent-dependent value
context). The specific mean/max pooling and 256 hidden size here are
engineering choices, not a claim of an optimal published architecture.
