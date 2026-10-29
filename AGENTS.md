# AGENTS.md

> Engineering contract for humans and AI agents working on PPOTrial.
> Adapted from the MFDStudio AGENTS.md: favor maintainable, explicit, testable C++17 over cleverness.

## Mandatory workflow

1. Read this file, the existing implementation, relevant configs and tests before editing.
2. State the exact intended behavior, invariants and compatibility impact.
3. Prefer narrow changes; do not relocate, rename, reformat or rewrite unrelated files.
4. Work in a dedicated branch, never commit directly to master.
5. Re-read modified functions for correctness, ownership, state transitions and C++17 readability.
6. Run available builds and tests; explicitly disclose any checks not actually executed.
7. Update documentation, tests and checkpoint contracts alongside behavior changes.

## C++17 readability rules

- Strict C++17. Prefer RAII, const-correctness, clear ownership and named helpers.
- Use std::unique_ptr for exclusive ownership. Avoid raw new/delete, unsafe casts,
  unexplained reinterpret_cast, arbitrary mutable state and broad lambda captures.
- Keep one primary responsibility per class and helper. Prefer composition to inheritance.
- Avoid speculative templating, dummy unused variables, immediately invoked lambdas,
  nested mutations and noisy compatibility wrappers.
- Prefer static/fixed-capacity arrays in per-step paths where maximum capacity is known.
- Validate JSON, type/range/count invariants, finite numeric inputs, and all indices.
- Document units, error behavior, and nontrivial public APIs with Doxygen.
- Follow the surrounding module's formatting and naming conventions.

## Module boundaries

- SharedWorld: reusable ECS components, sensors, geometry and physics.
- Defenders/Core: pure domain logic, team roles, rewards and patrol/defense geometry.
- Defenders/Training: RLTools neural models, PPO/GAE, optimizers and checkpoints.
- Defenders/Runtime: ECS-to-domain adapters, actions and training coordination.
- Examples: demonstration scenarios consuming shared modules, never the reverse.
- Preserve existing Airspace and RLMission agents, historical 2-vs-2, GUI, launcher,
  example selection and configuration behavior unless the task explicitly requires a change.

## Multiagent RL contract

- Generalized P/N teams have P in [1, 9] and N in [0, 10].
- Distinguish initial episode membership from current alive/available status.
- Preserve stable physical identities and use masked fixed-capacity observations.
- Permuting the order of interchangeable tracks must not change the pooled context.
- Centralized critics using per-agent returns must be conditioned on the selected
  agent's own state and role; shared critic parameters do not imply shared targets.
- GAE must distinguish death, mission termination, truncation and masked transitions.
- Train/test multiple team sizes; do not infer generalization from tensor shape alone.
- Reward channels must remain measurable, normalized by active counts where appropriate,
  and aligned with mission goals; avoid per-step survival bonuses or timeout exploits.
- No hard-coded red-target assignments or predetermined policy actions by default.

## World and pseudo-LiDAR contract

- Preserve the historical 61-ray / 360-degree ordering, normalizations and bounds.
- The nearest valid hit may come from a hostile restricted zone, world boundary,
  or an explicitly enabled nonlethal friendly-mobile safety envelope.
- Never self-detect, sense destroyed peers, or confuse an envelope with an ECS
  RestrictedZone. Only collision/world resolution destroys a mobile.
- Preserve the existing Airspace behavior when dynamic-envelope sensing is disabled.
- Keep geometry, safety radii, timestep and reward units consistent and documented.
- Recompute sensing at the correct time, and never allow NaN/Inf in observations.

## Checkpoints and compatibility

- Inference-only actor checkpoints and full training-state checkpoints differ.
- Changing trainable dimensions, critic inputs, observation semantics or serialization
  requires a deliberate versioning and migration decision.
- Never reinterpret old checkpoint bytes or silently reset pretrained weights.
- Keep deterministic seeds and honest update counters.
- Test save/load/resume and document any actor-only migration with critic reset.

## Tests, review and documentation

- Add regression tests for every changed domain behavior and public contract.
- Test 1/2/3/4/9 blue participants, 0/1/multiple reds, inactive/dead masks,
  permutation invariance, finite values and nondegenerate PPO gradients.
- Test LiDAR self exclusion, nearest hit, rear obstacle, overlap, disabled sensing,
  destroyed mobiles, hostile exclusion and boundary/zone behavior.
- Test threat ranking, danger reduction, neutrality when no red and reward scaling.
- Audit loops for progression, zero/invalid steps and bounds.
- Findings must distinguish proven code defects from hypotheses needing training data.
- Update README/docs for observable changes and checkpoint migration semantics.
- Keep pull requests focused; report changed files and actual CI/test outcomes.

## Definition of done

- [ ] Implementation is coherent C++17, finite and bounded.
- [ ] Tests and documentation reflect updated behavior.
- [ ] No unrelated behavior, relocation or formatting churn.
- [ ] Checkpoint compatibility and training reset behavior are explicit.
- [ ] Build/tests/CI have been checked or unavailable execution disclosed.
