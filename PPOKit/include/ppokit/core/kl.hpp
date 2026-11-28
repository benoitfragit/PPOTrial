/**
 * @file PPOKit/include/ppokit/core/kl.hpp
 * @brief KL estimates of a PPO minibatch and the guard that skips an actor step.
 * @details Two estimates of KL(old || new) per sample, from the log ratio
 * log r = log pi_new(a|s) - log pi_old(a|s):
 * - sampled: -log r, unbiased but noisy, can be slightly negative;
 * - approximate: r - 1 - log r, always non-negative (log r bounded to +/-20).
 * Their minibatch means are what the learners report. The guard skips the actor step of a
 * minibatch whose sampled KL reaches KlSkipFactor x target: the update has moved the
 * policy too far from the data. The critic still learns from that minibatch.
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace ppokit::core {

/** @brief The actor step is skipped when the sampled minibatch KL reaches this multiple of the target. */
inline constexpr double KlSkipFactor = 1.5;
/** @brief Bound of |log r| in the approximate KL, so one sample cannot overflow it. */
inline constexpr double MaximumKlLogRatio = 20.0;

/** @brief Sampled KL term of one sample: -log r. */
template<typename Scalar>
constexpr Scalar SampledKl(Scalar logRatio) noexcept
{
    return -logRatio;
}

/** @brief Approximate KL term of one sample: r - 1 - log r, never negative. */
template<typename Scalar>
Scalar ApproximateKl(Scalar logRatio) noexcept
{
    Scalar const bounded = std::clamp(logRatio, Scalar(-MaximumKlLogRatio), Scalar(MaximumKlLogRatio));
    return std::max(Scalar(0), std::expm1(bounded) - bounded);
}

/**
 * @brief Whether the actor may step after a minibatch of mean sampled KL `sampledKl`.
 * @param targetKl Target KL; 0 turns the guard off.
 */
template<typename Scalar>
constexpr bool ActorStepAllowed(Scalar sampledKl, Scalar targetKl) noexcept
{
    return targetKl <= Scalar(0) || sampledKl < Scalar(KlSkipFactor) * targetKl;
}

} // namespace ppokit::core
