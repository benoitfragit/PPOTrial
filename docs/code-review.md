# Review against AGENTS.md and the agreed scope

Review baseline: `9852f9e`, PR #13. This review covers the environment contract,
runtime, backend adapters and extracted PPO/network sources, Navigation2D,
application controller, renderer, CMake, protocol draft and documentation.

## Corrected findings

| Priority | Location / scenario | Cause and impact | Correction / regression coverage |
| --- | --- | --- | --- |
| P1 | `Application::pause`, evaluation active | Only the training flag was cleared; evaluation continued stepping and could promote while shown as paused. | Explicit evaluation pause/resume. Fake-backend tests assert no actions or learner updates while paused, resume in bounded batches, cancellation without promotion. |
| P1 | `Application::evaluate_candidate`, different scenario/criteria in the same store | Best scores had no evaluation provenance; incomparable runs could replace each other. | Opaque context in generic promotion, canonical scenario/criteria, episode count and seed sequence in the application. Context mismatch is rejected before copying/promoting. Synthetic checkpoint and controller tests cover this. |
| P2 | `Main/src/main.cpp`, executable moved away from checkout | A compile-time absolute source path was needed to open the scenario. | Resources copied beside the executable and resolved using its directory. A CTest checks packaged files; custom scenario/store paths remain selectable. |
| P2 | `aggregate`, extreme finite values with standard deviation | Overflow in variance accumulation could be clamped to zero, producing a false valid metric. | Nonfinite variance accumulation raises an exception. Regression with opposite-sign values of magnitude 1e308. This is a metric validation fix, not a PPO update. |
| P2 | Application/controller tests | Direct construction of Torch objects prevented testing training/evaluation commands without a real model. | Injected `IPolicyBackend`; the Torch binding is a separate source. Headless controller tests use only fake policies and synthetic checkpoint bytes. |
| P2 | `Renderer` inspection and heading display | Presentation decoded hardcoded policy observation offsets and normalization. | Environment exposes a read-only physical state for presentation; observation packing remains inside the environment. |
| P3 | `RLAgent::SetNewState` | Inherited throwaway bindings included a reserved C++ identifier (`__`), contrary to agent naming rules. | Select tuple element with `std::get<2>`; the same call and RNG consumption are retained. |
| P3 | Source comments / validation docs / CI | Stale file paths, stale validation text and duplicated push/PR builds. | Correct source paths and documentation; branch builds use the PR event once. |

## Compliance and scope

- One application and one Navigation2D dynamics implementation. The generic core
  includes neither the example nor graphics, EnTT or Torch headers.
- Algorithm equations, hyperparameters and network remain unchanged by this review.
  The tuple-access cleanup does not change computation. No neural model is created,
  loaded, evaluated or trained by the regression suites.
- Ownership is explicit. Controller sessions are destroyed before their policies
  and environments; the injected backend must outlive the application.
- Scenario serialization moved into the example; metric serialization belongs to
  the generic runtime. Criteria saved with a candidate come from the parsed values
  actually evaluated, rather than rereading a potentially changed source file.
- Existing naming conventions are retained in extracted PPO files, as AGENTS.md
  permits; this review does not rename the entire historical backend.
- Historical resources remain unchanged. They are not an application/compatibility
  path, and their deletion is outside this correction.

## Limits and explicit migrations

This is not a behavior-identical port of the initial simulator: the agreed
Navigation2D example now has a scripted moving goal mobile and geometric zones. It replaces
old observations, rewards and scenario semantics, as documented in the PR.

The desktop keeps its original shell and major panels; exact interactive parity
has not been demonstrated. Visual behavior, camera interaction and manual controls
still require an interactive smoke test. Compilation alone is not that test.

Stop still completes one final action and marks it truncated so the existing
learner receives a valid final transition. Pause executes no extra action. PPO
updates remain synchronous and may block a UI frame. These are documented behavior
limits, not claims of a finished production scheduler.

Best references now include context. Old two-line references must not be reused;
choose a new store. The UI exposes its checkpoint root through the second command
line argument. No automatic data deletion or migration occurs.

Discrete spaces and the transport protocol are preparatory APIs. The current Torch
adapter remains continuous Box; GNS and the Python bridge are not implemented.
The initial request explicitly allowed those integrations to remain future work.

## Validation

Direct GCC Release: 4 contract, 6 Navigation2D/runtime, 2 fake-backend controller
test groups. Main/Renderer/Torch binding syntax checked using the actual dependency
headers, without running a model. Full CMake builds and tests are run in PR CI:
Linux Debug/Release and Windows Release with LibTorch. The PR records the outcome
for the correction commit; previous green checks do not validate a later revision.

## Subsequent correction: moving goal

The fixed-only goal in the review baseline did not preserve the user's requirement
to reach another mobile. The example now supports a constant-velocity goal mobile,
current position/velocity observations, relative swept meeting detection and live
display. This is an environment schema change (v2), not a change to PPO. Old v1
policies are incompatible. See the contract and PR for the new verification results.
