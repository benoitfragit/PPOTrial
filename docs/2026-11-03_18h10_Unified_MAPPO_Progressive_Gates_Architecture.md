# Unified Blue MAPPO and progressive GitHub gates — 2026-11-03

## Decision ownership
One shared MAPPO actor owns every Blue action in P=1..9, N=0..10. No scripted PATROL/DEFEND selector or N=0/N>0 training branch chooses actions. The policy receives live Red masks/tracks, objective, pressure, curve/formation references, own state and LiDAR. All missions consume the same physical leader/slot guidance. The 10-waypoint sequencer remains an N=0 *measurement and completion* mechanism, not a control policy.

## Reward contract
The existing frozen-before-step convergence reward applies only when *both* snapshots contain no active Red; delayed and already destroyed Reds retain a navigation learning signal. When Reds are active, the existing team pursuit and teammate formation-progress rewards are used without a patrol objective that competes with interception. This is reward relevance, **not a hard-coded decision about the actor's actions**. No engagement-distance gate suppresses pursuit of distant Reds. Objective survival, Red neutralization, safety, boundary, terminal outcome and role-specific channels are preserved. No new reward channel, normalization or weight. This permits the actor to learn the trade-off; it does **not** establish improved combat until independent evaluation.

## Checkpoints
Transport headers remain V10 actor/V7 full optimizer, with strict `UNIFIED_PHYSICAL_GUIDANCE_V1` semantic on load for every P/N. The former Pvs0-specific tags and compatibility behavior branches have been removed, not silently migrated. Old run-19/26/27 models remain untouched as reference artifacts, but cannot be resumed as new unified-policy weights without an explicit validated migration. Across the *new* gates, the same actor, critic, optimizer state and RNG are passed unmodified.

## Historical V3 balance
Within 1200 PPO updates, the revised default V3 distributes 140 to auxiliary Pvs0 (previously 560), 1060 to defense cases, including 460 at 4v4 (previously 160); rehearsal changes from 24:6 to 27:3. Existing V3 progress uses a config fingerprint: no silent resume from a changed schedule.

## Outcome-based progressive gate flow
The existing `.github/workflows/mappo-training-windows.yml` provides scenario `progressive-blue-defense` with `progressive_gate`, `success_percent`, `max_seconds`, `auto_advance`, plus the preexisting `resume_run_id`. Gate definitions are in `tools/ci/adaptive_curriculum/gates.json`, 1v0 → 4v4. Every gate is independently configurable (threshold, time, automation). **No gate has an update or episode quota**: an internal bounded block establishes periodic optimizer checkpoint and evaluation opportunities only. A gate passes on a *completed, matching P/N mission-success evaluation*, never PPO score, elapsed updates or the mere end of time.

Every eighth completed training block rehearses an earlier gate in deterministic round-robin order, rotating across all previous P/N scenarios without imposing update quotas; only a completed evaluation of the *primary* gate can pass it. Production 4v4 checkpoint files are restored after every non-4v4 rehearsal, including in the 4v4 phase.

When time expires without success, persist the same gate in `adaptive-curriculum-progress.json` and the full V7 training checkpoint; resuming via the preceding GitHub run ID grants a fresh wall-clock budget and keeps the optimizer update counter. When the optimizer made no new update, a resumed run may carry forward the validated existing checkpoint rather than treating it as a failed gate. After success, another GitHub run uses that same checkpoint and advances exactly one gate. The `advance` job dispatches the same existing training workflow only after successful artifact publication and validated gate handoff; it never builds. The chain can exceed one GitHub run's wall-time limit.

## Invariants and validation
No new Pvs0 build workflow; remove `.github/workflows/pvs0-msvc-x64-tests.yml`. Existing VS2022 Debug/Release `desktop-windows.yml`, CTest and Python CI tests remain authoritative. RLMission Red production models, simulator physics, action dimensions, PPO actor/critic dimensions and Red target assignment are unchanged. The PR remains draft; no merge or training dispatch is performed automatically by submitting this code. Validate deterministic C++/Python tests and independent-seed combat evaluation before promoting candidate weights.

## Deployment constraint
The `workflow_dispatch` form is sourced from the workflow on the repository default branch. The feature branch remains a draft PR; the new GUI fields are not guaranteed to be exposed until the workflow is present on the default branch. No training job is triggered by this PR and no master edit is performed.
