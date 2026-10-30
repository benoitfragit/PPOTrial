# MAPPO V8 — reward scaling and evaluation-loss diagnostics

Date: 2026-10-30. Target: BlueDefense generalized MAPPO only.
Base: master f97d40d6e32a77c99c469005e91af9400f6cc3cc.

## Problem observed

GitHub Actions training run 41046789452 completed 194 PPO updates and
published actor/training checkpoints. The final deterministic 4v4 campaign
reported score=29.6875, success=0, blue-survival=0, red-neutralization=0.3125,
objective-survival=1. The telemetry summary step failed on a Windows CP1252
UnicodeEncodeError. These are distinct issues.

## Changes

1. **Reward parameter fix (not a PPO/network change).** In the generalized
   BluePatrol runtime, the control-effort channel now applies
   `Parameters::controlEffortWeight` exactly once. The JSON already defines
   `rewards.shaping.controlEffortWeight=0.002`; before this correction the
   generalized runtime had omitted that multiplier. The existing
   `ControlEffort` channel weight remains intact.
2. **Evaluation-only loss records.** Each completed evaluation episode prints
   a bounded `[MAPPO_EVAL_EPISODE]` line: update, campaign, case, seed,
   steps, P/N, survivors, leader and objective status, termination reason and
   the last known source for each blue loss.
3. **Physical loss provenance.** Source identification reads the existing ECS
   `sEntityState::m_destroyedById`. Recognized source classes:
   `blue_collision`, `hostile_mobile`, `hostile_zone`, `other_zone`,
   `other_source`. If no valid source exists the value is `unknown`;
   no cause is fabricated.
4. **CI diagnostics.** Windows workflow writes the headless stdout to
   `build/mappo-telemetry/mappo-training.log` using PowerShell Tee-Object,
   retaining the native process exit-code check. The offline Python script
   reads those episode rows, summarizes the latest campaign and includes
   the log in the telemetry artifact. Stdout uses UTF-8 even on a CP1252
   Windows console.

## Architecture and invariants

```text
BlueDefense ECS (physics and existing impact attribution)
    |
    +--- RLMission red controllers (UNCHANGED)
    |
    +--- BluePatrol generalized MAPPO V8
          +--- RLtools actor / centralized critic (UNCHANGED)
          +--- training rollout and PPO (UNCHANGED)
          +--- cooperative reward
          |      +--- per-agent controlEffortWeight now applied
          +--- deterministic evaluation
                 +--- existing episode metrics (UNCHANGED)
                 +--- one log record per finished episode (NEW)
                       |
                       +--- GitHub Actions captured stdout (NEW)
                             +--- offline Python summary (NEW)
```

No serialization layout, checkpoint version, action space, observation
contract, role assignment, physics resolution, simulator world, RLMission,
or scoring formula is modified. The evaluation remains synchronous and
uses the existing ECS; this patch does **not** implement a threaded
evaluation World.

## Verification

- The generalized reward-shaping unit test covers zero, half-range, full
  range, saturation and zero weight.
- Python tests cover per-episode cause aggregation and CP1252 stdout.
- A new PR must pass the `Core C++17 tests` matrix for Linux/Windows.
- Before training production use, inspect all CI checks and confirm
  the first evaluation reports per-episode reasons.
- The previous evaluation plateau is **not** proven solved until a new
  campaign shows improved independent-seed survival and mission success.

## Running after merge

1. Launch GitHub Actions workflow `MAPPO training (Windows Release)`.
2. Use `scenario=blue-defense`. For a fresh experimental comparison,
   leave `resume_run_id` empty; a resumed Adam optimizer has history
   under the old reward scaling and is not an isolated ablation.
3. Follow the `MAPPO training telemetry` summary and open artifact
   `blue-mappo-telemetry` for `mappo-training.log`, CSV and SVG.
4. Inspect `[MAPPO_EVAL_EPISODE]` outcomes, especially
   `leader_lost`, `blue_collision`, `hostile_zone`, `unknown`.
5. Compare mission success and blue survival against the unchanged
   baseline run 41046789452. Do not use the composite score alone.

## Change boundaries

Do not alter legacy RLMission, old 2v2 agent, PPO hyperparameters,
network architecture, or autonomous role attribution as part of this
specific fix.
