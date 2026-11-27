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
- Write clear, clean and readable C++17: use consistent formatting, descriptive
  names, explicit control flow and small functions with one understandable
  responsibility. Avoid dense expressions, nested ternaries, surprising side
  effects, magic values and unnecessary abstractions.
- Keep classes and translation units cohesive. Split long functions when a
  named operation clarifies the algorithm; do not create monoliths or expose
  public interfaces that have no caller. Use familiar design patterns only
  when they make ownership and responsibilities clearer.
- Document interfaces and non-trivial implementation helpers with Doxygen
  (`@brief`, `@param`, and `@return` when applicable). In `.cpp` files, add
  concise comments explaining the purpose of important steps, algorithms,
  physical assumptions and non-obvious decisions. Keep comments accurate and
  useful; do not merely repeat the code or comment every obvious statement.
- Re-read modified functions and their callers for clarity as well as
  correctness. Simplify confusing existing code in the scope of the change and
  remove obsolete branches, settings, data and comments instead of leaving dead
  compatibility paths behind.
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

## Mandatory self-review and reward discipline
- After **every** code or test change, re-read the complete modified functions,
  callers, affected tests and final diff. Check role-to-physical-index mapping,
  before/after-step snapshots, control-flow and success/failure/timeout semantics.
  Do not rely on a successful build alone as a substitute for code review.
- Before updating an existing PR, inspect the full diff against its base,
  verify that no unrelated change entered it, and correct every identified
  defect. Check the actual CI job logs, not only the workflow headline; rerun
  relevant checks after fixes. Never declare a change validated while required
  checks are failing, pending or not executed.
- Do not invent or add Blue MAPPO reward channels, bonuses, weights, shaping
  formulas or normalizations unless the user **explicitly requests** them.
  Reuse the existing leader patrol, wingman formation, separation, individual
  loss and mission-outcome channels. Any requested redistribution of an existing
  channel must retain its established magnitude and have role-specific tests.
- For no-Red patrol (N=0), the living leader must be able to earn its existing
  completed-lap reward by following a physically validated complete racetrack,
  **independently of wingmen formation errors or losses**. Wingmen learn from
  their existing formation and safety signals; collision physics and individual
  loss penalties remain authoritative. Do not award completion for shortcuts,
  stationary orbits, invalidated laps or a destroyed leader.
- Keep unfinished training evaluations independent of development branches;
  never merge their behavioral changes into master without explicit approval.

## Fixed x64 build directories
- The only x64 Debug build directory is `build/debug`.
- The only x64 Release build directory is `build/release`.
- Never configure or compile x64 artifacts in another directory. Reuse these
  destinations for every incremental build so the root launchers (`.bat`) stay
  valid: they run the Release programs of `build/release` (ppokit-studio,
  ppokit-run, ppokit-viewer).
- Do not relocate an existing build merely to apply this convention. State any
  unverified configuration instead of creating a temporary x64 build tree.

## Git
- Work on the current feature branch and existing PR; never force-push
  unrelated changes or overwrite master.
- State exactly which builds/tests ran and which remain unverified.
