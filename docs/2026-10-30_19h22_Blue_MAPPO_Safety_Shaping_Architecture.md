# PPOTrial — Blue MAPPO V8: anticipatory safety shaping

Date: 2026-10-30. Scope: BlueDefense generalized MAPPO only.

## Decision and scope

Retain the 61-ray pseudo-LiDAR geometry, actor observation layout, model checkpoints,
flight-model commands, red RLMission agent, and ECS destruction rules. Add three
bounded per-blue reward channels for early feedback rather than a scripted
avoidance controller or target assignment. The source remains strictly C++17
with fixed-capacity buffers.

## Data flow

1. BlueDefense world rules continue to update 61 LiDAR beams and resolve
   Blue/Blue collisions (1,000 m), penetrations into hostile restricted
   zones (physical NEZ), and hard-world failure separately.
2. The blue ECS adapter reads each aircraft and computes a reward-only
   `restrictedZoneRisk` from hostile active zone positions (zone danger
   1.10 NEZ; caution 1.75 NEZ). This field is deliberately absent from the
   actor observations and checkpoints.
3. The pure shaping layer computes `GeneralizedBlueSeparationRisk`
   using minimum predicted pairwise separation under constant velocity
   over five seconds; and `GeneralizedBoundaryRisk` from present and
   five-second projected positions relative to the nominal world limits.
4. All three risks are reduced by
   `clamp(0.15 * max(risk_before, risk_after)
   + 0.35 * max(risk_after - risk_before, 0), 0, 1)`.
   An unavailable blue receives no new safety shaping.
5. The cooperative reward aggregator exposes three new channels, appended
   after the previous ten: `BlueSeparation`,
   `RestrictedZoneClearance` and `BoundaryClearance`. Each contributes
   to its blue's reward, while team telemetry contains its mean over P.
   Their initial independent weights are 1.0; old ten channel IDs are stable.

## Risk boundaries and limitations

- These new terms **train** the actor to avoid future collisions; they do
  not guarantee physical avoidance or override its selected actions.
- Constant-velocity CPA ignores future turning and acceleration, deliberately
  as a lightweight approximate safety assessment.
- Mixing obstacle-specific normalizations in the existing 61-ray LiDAR is not
  changed in this PR: altering the policy input scaling during training would
  invalidate comparability with earlier actor checkpoints. Investigate that
  separately after independent-seed safety evaluation.
- Added shaping changes the reward objective. Continue from a compatible
  policy only with awareness that training metrics are not directly comparable
  to the previous reward configuration.

## Regression coverage

`BlueDefenseGeneralizedRewardShapingTests`: P=1/N=0, head-on and parallel
Blue/Blue motion, dead peers, early world-boundary approach, penetration and
caution of red zones, worsening versus retreating risk.

`BlueDefenseGeneralizedRewardEvaluationTests`: three safety channels,
per-blue assignment and team averages, non-finite signal rejection, loss of
one blue, P=1/N=0.

`BlueDefenseGeneralizedRuntimeStateTests`: invalid zone risk and indexed
diagnostics; retains older negative-speed/destroyed-mask checks.

## Verification gate

Run the repository's documented CMake Debug and Release configurations,
CTest core tests and CI checks before merging. A build/test pass is not
evidence of trained-policy convergence: compare independent evaluation seeds
and monitor collisions by source, zone penetrations, boundary exits, survival,
and overall mission score.

No red RLMission code or shared LiDAR geometry is modified.
