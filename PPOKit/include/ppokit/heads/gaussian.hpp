/**
 * @file PPOKit/include/ppokit/heads/gaussian.hpp
 * @brief Diagonal Gaussian action head: log-density, entropy and gradients, axis by axis.
 * @details A continuous action is sampled per axis from N(mean, exp(logStd)^2).
 * PPO needs, for every axis taking part in the action:
 * - its log-density, to build the probability ratio of the new and old policies;
 * - its entropy, for the exploration bonus;
 * - the gradient of the log-density with respect to the mean and the log-std.
 *
 * The helpers work on one axis at a time, so they serve fixed arrays, rl_tools
 * matrices or masked layouts alike: an inactive axis (for instance a brake
 * that only exists on the ground) is simply skipped by the caller.
 *
 * Every PPOKit learner uses these exact expressions, so two examples with the
 * same weights and samples compute the same numbers.
 */
#pragma once

#include <cmath>

namespace ppokit::heads {

/** @brief log(2 pi) / 2, the normalization constant of a unit Gaussian. */
template<typename Scalar>
inline constexpr Scalar HalfLogTwoPi = Scalar(0.918938533204672741780329736405617639L);

/** @brief log(2 pi e) / 2, the entropy of a unit Gaussian. */
template<typename Scalar>
inline constexpr Scalar UnitGaussianEntropy = Scalar(1.418938533204672741780329736405617639L);

/**
 * @brief Log-density of `sample` on one axis: -(z^2 / 2 + logStd + log(2 pi) / 2).
 * @param sample Raw (unclipped) sampled action of the axis.
 * @param mean Gaussian mean produced by the actor.
 * @param logStd Natural logarithm of the standard deviation.
 * @details z = (sample - mean) / sigma is computed as (sample - mean) * exp(-logStd).
 * Clipping or squashing applied afterwards to the physical command does not
 * enter the density: PPO scores the latent sample.
 */
template<typename Scalar>
Scalar GaussianAxisLogDensity(Scalar sample, Scalar mean, Scalar logStd) noexcept
{
    Scalar const standardized = (sample - mean) * std::exp(-logStd);
    return -(Scalar(0.5) * standardized * standardized + logStd + HalfLogTwoPi<Scalar>);
}

/** @brief Entropy of one axis: logStd + log(2 pi e) / 2; its derivative in logStd is 1. */
template<typename Scalar>
constexpr Scalar GaussianAxisEntropy(Scalar logStd) noexcept
{
    return logStd + UnitGaussianEntropy<Scalar>;
}

/** @brief Gradient of the log-density of one axis. */
template<typename Scalar>
struct GaussianAxisGradient final
{
    /** @brief d log p / d mean = (sample - mean) / sigma^2. */
    Scalar mean;
    /** @brief d log p / d logStd = (sample - mean)^2 / sigma^2 - 1. */
    Scalar logStd;
};

/** @brief Gradient of GaussianAxisLogDensity with respect to the mean and the log-std. */
template<typename Scalar>
GaussianAxisGradient<Scalar> GaussianAxisLogDensityGradient(Scalar sample, Scalar mean, Scalar logStd) noexcept
{
    Scalar const inverseVariance = std::exp(Scalar(-2) * logStd);
    Scalar const difference = sample - mean;
    Scalar const scaledDifference = difference * inverseVariance;
    return {scaledDifference, difference * scaledDifference - Scalar(1)};
}

} // namespace ppokit::heads
