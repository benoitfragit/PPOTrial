/**
 * @file PPOKit/include/ppokit/heads/bernoulli.hpp
 * @brief Bernoulli action head: a yes/no decision drawn with probability sigmoid(logit).
 * @details The actor outputs one logit l per decision. PPO needs its log-probability,
 * its entropy and their derivatives with respect to l. Every expression goes through
 * softplus(x) = log(1 + exp(x)), written so that it never overflows, even for logits
 * far beyond +/-700. Example: a landing gear extension request (FlightControl3D).
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace ppokit::heads {

/** @brief log(1 + exp(x)) without overflow. */
template<typename Scalar>
Scalar Softplus(Scalar x) noexcept
{
    return std::max(x, Scalar(0)) + std::log1p(std::exp(-std::abs(x)));
}

/** @brief 1 / (1 + exp(-x)) without overflow. */
template<typename Scalar>
Scalar Sigmoid(Scalar x) noexcept
{
    Scalar const decay = std::exp(-std::abs(x));
    if (x >= Scalar(0))
    {
        return Scalar(1) / (Scalar(1) + decay);
    }
    return decay / (Scalar(1) + decay);
}

/** @brief log p(drawn | logit): -softplus(-l) for a "true" draw, -softplus(l) for a "false" one. */
template<typename Scalar>
Scalar BernoulliLogProbability(Scalar logit, bool drawn) noexcept
{
    if (drawn)
    {
        return -Softplus(-logit);
    }
    return -Softplus(logit);
}

/** @brief d log p(drawn | logit) / d logit = drawn - sigmoid(logit). */
template<typename Scalar>
Scalar BernoulliLogProbabilityGradient(Scalar logit, bool drawn) noexcept
{
    Scalar const outcome = drawn ? Scalar(1) : Scalar(0);
    return outcome - Sigmoid(logit);
}

/** @brief Entropy of the decision: softplus(l) - sigmoid(l) l, at most log 2 for l = 0. */
template<typename Scalar>
Scalar BernoulliEntropy(Scalar logit) noexcept
{
    return Softplus(logit) - Sigmoid(logit) * logit;
}

/** @brief d entropy / d logit = -l sigmoid(l) (1 - sigmoid(l)). */
template<typename Scalar>
Scalar BernoulliEntropyGradient(Scalar logit) noexcept
{
    Scalar const probability = Sigmoid(logit);
    return -logit * probability * (Scalar(1) - probability);
}

} // namespace ppokit::heads
