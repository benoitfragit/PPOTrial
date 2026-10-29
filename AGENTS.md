# AGENTS.md — PPOTrial

Instructions for all contributors and automated agents modifying this C++17
repository. Adapted from the MFDStudio AGENTS.md engineering conventions.

## Before editing
- Read the existing code, actual branch, configuration and tests first.
- Keep change sets scoped, reviewable, and behaviorally justified.
- Do not modify the red RLMission controller while evolving the generalized MAPPO runtime.
- Keep simulator physics, policies, reward computation and ECS adaptation
  separated. Do not introduce an external target-assignment controller.

## C++17 and architecture
- Strict, portable C++17, RAII, const-correctness, explicit names and noexcept
  for pure non-throwing helpers.
- Prefer named, documented functions to broad captures, hidden lambdas,
  dummy variables, unsafe casts and overly clever templates.
- Keep data and interfaces minimal. Use fixed-capacity storage in hot paths.
- Avoid allocations in per-step actor, critic, observation and reward loops.
- Preserve 1..9 blue and 0..10 red capacity, masking and permutation behavior.

## Testing and review
- Every functional fix needs deterministic regression tests, including P=1,
  N=0, multiple tracks, collisions and critical edge cases when applicable.
- Review changes to reward channel names and count in aggregation, snapshots,
  JSON mapping, tests and documentation together.
- Check finite values, valid ranges, normalization and terminal-vs-truncation
  semantics. Inspect loops for bounded execution.
- Build Debug and Release, run CTest, inspect CI results before merging.
- Do not claim convergence from C++ unit tests; require independent-seed
  evaluation of learned policies.
- Re-read all changed files and keep diffs free of unrelated formatting edits.

## Git
- Work on the current feature branch and existing PR; never force-push
  unrelated changes or overwrite master.
- State exactly which builds/tests ran and which remain unverified.
