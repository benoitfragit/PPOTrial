/**
 * @file PPOKit/include/ppokit/core/ppo_surrogate.hpp
 * @brief Clipped PPO surrogate of one sample and its derivative with respect to log pi.
 * @details PPO maximizes, for each collected sample,
 *
 *     objective = min(r * A, clip(r, 1 - epsilon, 1 + epsilon) * A),  r = pi_new(a|s) / pi_old(a|s)
 *
 * where A is the (normalized) advantage. The clipping removes any incentive to
 * move the policy further than epsilon from the collection policy in one update.
 * The learner minimizes `-objective`; its derivative with respect to
 * log pi_new is `-A * r` where the unclipped term is the minimum, and 0 where
 * the clipped term is (the sample then no longer pushes the policy).
 * Chain it with the head gradients (for instance GaussianAxisLogDensityGradient)
 * to obtain the derivatives of the actor outputs.
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace ppokit::core {

/**
 * @brief Default bound of |log r| before exponentiation.
 * @details exp(20) ~ 4.9e8 is finite even in single precision; the surrogate is
 * flat far beyond 1 +/- epsilon anyway, so the bound only prevents overflow.
 */
template<typename Scalar>
inline constexpr Scalar DefaultMaximumAbsoluteLogRatio = Scalar(20);

/** @brief Values of the clipped surrogate for one sample. */
template<typename Scalar>
struct ClippedSurrogate final
{
    /** @brief log r after bounding to +/- maximumAbsoluteLogRatio. */
    Scalar logRatio;
    /** @brief r = exp(logRatio). */
    Scalar ratio;
    /** @brief min(r A, clip(r) A); the loss to minimize is -objective. */
    Scalar objective;
    /** @brief d(-objective) / d log pi_new: -A r while unclipped, else 0. */
    Scalar logProbabilityGradient;
    /** @brief True when |r - 1| > epsilon (counted by the clip-fraction diagnostic). */
    bool outsideClipRange;
};

/**
 * @brief Evaluates the clipped surrogate of one sample.
 * @param logProbability log pi_new(a|s) under the policy being optimized.
 * @param oldLogProbability log pi_old(a|s) recorded at collection time.
 * @param advantage Advantage estimate of the sample.
 * @param clip PPO epsilon, typically 0.1 to 0.3.
 * @param maximumAbsoluteLogRatio Overflow guard on |log r|.
 */
template<typename Scalar>
ClippedSurrogate<Scalar> EvaluateClippedSurrogate(Scalar logProbability, Scalar oldLogProbability,
    Scalar advantage, Scalar clip,
    Scalar maximumAbsoluteLogRatio = DefaultMaximumAbsoluteLogRatio<Scalar>) noexcept
{
    ClippedSurrogate<Scalar> result{};
    result.logRatio = std::clamp(logProbability - oldLogProbability, -maximumAbsoluteLogRatio,
        maximumAbsoluteLogRatio);
    result.ratio = std::exp(result.logRatio);
    Scalar const clippedRatio = std::clamp(result.ratio, Scalar(1) - clip, Scalar(1) + clip);
    Scalar const unclippedTerm = result.ratio * advantage;
    Scalar const clippedTerm = clippedRatio * advantage;
    result.objective = std::min(unclippedTerm, clippedTerm);
    result.logProbabilityGradient = unclippedTerm <= clippedTerm ? -advantage * result.ratio : Scalar(0);
    result.outsideClipRange = std::abs(result.ratio - Scalar(1)) > clip;
    return result;
}

} // namespace ppokit::core
