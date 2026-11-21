/**
 * @file flight_policy.hpp
 * @brief Diagonal Gaussian and Bernoulli policy: sampling, likelihood, KL, entropy, clipped PPO
 *        loss and GAE delta.
 *
 * Role in the chain: observation -> actor network (flight_learner.cpp) -> **policy** ->
 * physical controls -> environment. The actor outputs one mean per continuous action axis
 * and one gear logit; the standard deviations are global, state-independent parameters
 * (log-std) bounded to [MinimumLogStandardDeviation, MaximumLogStandardDeviation].
 *
 * Continuous axes: elevator, aileron, rudder, throttle, brake. The brake axis is "active"
 * (sampled, scored and optimized) only during the ground roll; in flight it is excluded
 * from the likelihood, the KL and the entropy.
 *
 * Gear: a Bernoulli extension request with probability sigmoid(logit). It is active only
 * while the extension is not yet commanded; the environment latches the first request, so
 * the gear is never retracted and a latched gear contributes nothing to PPO.
 *
 * Likelihoods are those of the latent Gaussian sample and of the Bernoulli draw, not of
 * the clipped physical control: clipping happens afterwards in ToControls.
 *
 * Not here: the networks and the optimizer (flight_learner.cpp).
 *
 * Read next: flight_learner.cpp (how these terms are used), Core/flight_types.hpp.
 */
#pragma once

#include "flight_config.hpp"
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace ppo::examples::flight {

/** @brief Actor outputs: one Gaussian mean per continuous control, then the gear logit. */
inline constexpr std::size_t PolicyOutputs = ContinuousActions + 1;
/** @brief Index of the gear extension logit in the actor output. */
inline constexpr std::size_t GearOutput = ContinuousActions;
/** @brief Index of the brake action, active only on the ground. */
inline constexpr std::size_t BrakeAction = 4;
/** @brief Index of the throttle action. */
inline constexpr std::size_t ThrottleAction = 3;
using PolicyVector = std::array<double, PolicyOutputs>;
using LogStandardDeviations = std::array<double, ContinuousActions>;
/** @brief Lower log-std bound: sigma = exp(-3.5) ~ 0.03, keeps a minimum exploration. */
inline constexpr double MinimumLogStandardDeviation = -3.5;
/** @brief Initial log-std: sigma = exp(-2) ~ 0.14 around the trimmed approach. */
inline constexpr double InitialLogStandardDeviation = -2.0;
/** @brief Upper log-std bound: sigma = exp(-1) ~ 0.37, avoids saturating the controls. */
inline constexpr double MaximumLogStandardDeviation = -1.0;
/**
 * @brief Bound of |log(pi_new / pi_old)| before exponentiation.
 *
 * exp(20) ~ 4.9e8 stays finite in double precision; the clipped surrogate is flat far
 * beyond 1 +/- clip anyway, so the bound only guards against overflow.
 */
inline constexpr double MaximumAbsoluteLogRatio = 20.0;
inline constexpr std::array<char const*, ContinuousActions> ActionNames{
    "elevator", "aileron", "rudder", "throttle", "brake"};
inline constexpr std::array<char const*, PolicyOutputs> PolicyOutputNames{
    "elevator_mean", "aileron_mean", "rudder_mean", "throttle_mean", "brake_mean", "gear_logit"};

/** @brief Numerically stable softplus log(1 + exp(x)). */
inline double Softplus(double x) noexcept
{
    return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
}

/** @brief Numerically stable logistic function 1 / (1 + exp(-x)). */
inline double Sigmoid(double x) noexcept
{
    double const decay = std::exp(-std::abs(x));
    if (x >= 0)
    {
        return 1 / (1 + decay);
    }
    return decay / (1 + decay);
}

/** @brief State-dependent actor means and global Gaussian exploration parameters. */
struct PolicyDistribution final
{
    /** @brief Gaussian means of the raw actions. */
    PolicyVector output{};
    /** @brief Natural logarithm of the standard deviation of each raw action. */
    LogStandardDeviations logStandardDeviations{};
};

inline bool operator==(PolicyDistribution const& left, PolicyDistribution const& right) noexcept
{
    return left.output == right.output && left.logStandardDeviations == right.logStandardDeviations;
}

inline bool operator!=(PolicyDistribution const& left, PolicyDistribution const& right) noexcept
{
    return !(left == right);
}

/** @brief One sampled (or deterministic) action and the data PPO needs to score it later. */
struct PolicyAction final
{
    /** @brief Raw latent Gaussian sample per axis, before clipping by ToControls. */
    std::array<double, ContinuousActions> raw{};
    /** @brief Log-density of raw under the collection policy, active axes only. */
    double logProbability{};
    /** @brief The brake is sampled and optimized only while rolling on the ground. */
    bool brakeActive{true};
    /** @brief Bernoulli gear draw: true requests the extension. */
    bool gearDown{};
    /** @brief The gear is sampled and optimized only while its extension is not yet commanded. */
    bool gearActive{};
    /** @brief Frozen collection distribution used by likelihood ratios and analytic KL tests. */
    PolicyDistribution policy{};
};

/** @brief Whether an action axis takes part in sampling, likelihood and optimization. */
inline bool ActiveAxis(std::size_t axis, bool brakeActive) noexcept
{
    return axis != BrakeAction || brakeActive;
}

/**
 * @brief Log-probability of a Bernoulli gear draw: -softplus(-l) for an extension, -softplus(l) otherwise.
 * @param logit Gear logit l; the extension probability is sigmoid(l).
 * @param gearDown Drawn value.
 */
inline double GearLogProbability(double logit, bool gearDown) noexcept
{
    if (gearDown)
    {
        return -Softplus(-logit);
    }
    return -Softplus(logit);
}

/**
 * @brief Log-density of the latent Gaussian sample and of an active gear draw, not of the
 *        clipped physical control.
 *
 * log p = sum over active axes of -(z^2 / 2 + log sigma + log(2 pi) / 2), z = (x - mu) / sigma,
 * plus GearLogProbability when the gear is active.
 */
inline double LogProbability(PolicyDistribution const& policy, PolicyAction const& action) noexcept
{
    double logProbability = 0;
    if (action.gearActive)
    {
        logProbability += GearLogProbability(policy.output[GearOutput], action.gearDown);
    }
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        if (!ActiveAxis(axis, action.brakeActive))
        {
            continue;
        }
        double const logStd = policy.logStandardDeviations[axis];
        double const standardized = (action.raw[axis] - policy.output[axis]) * std::exp(-logStd);
        logProbability -= 0.5 * standardized * standardized + logStd + 0.5 * std::log(2 * Pi);
    }
    return logProbability;
}

/**
 * @brief Exact old-to-new KL of two gear Bernoulli distributions given by their logits.
 *
 * KL = p_old (softplus(-l_new) - softplus(-l_old)) + (1 - p_old) (softplus(l_new) - softplus(l_old)).
 */
inline double GearKl(double previousLogit, double currentLogit) noexcept
{
    double const previousProbability = Sigmoid(previousLogit);
    return previousProbability * (Softplus(-currentLogit) - Softplus(-previousLogit))
        + (1 - previousProbability) * (Softplus(currentLogit) - Softplus(previousLogit));
}

/**
 * @brief Exact old-to-new KL on the physically active controls.
 *
 * Per Gaussian axis, with s = log sigma_old - log sigma_new:
 * KL = (exp(2 s) - 1) / 2 - s + (mu_old - mu_new)^2 / (2 sigma_new^2); plus GearKl when the
 * gear is active.
 */
inline double PolicyKl(PolicyDistribution const& previous, PolicyDistribution const& current,
    PolicyAction const& action) noexcept
{
    double result = 0;
    if (action.gearActive)
    {
        result += GearKl(previous.output[GearOutput], current.output[GearOutput]);
    }
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        if (!ActiveAxis(axis, action.brakeActive))
        {
            continue;
        }
        double const logScale = previous.logStandardDeviations[axis] - current.logStandardDeviations[axis];
        double const difference = previous.output[axis] - current.output[axis];
        result += 0.5 * std::expm1(2 * logScale) - logScale
            + 0.5 * difference * difference * std::exp(-2 * current.logStandardDeviations[axis]);
    }
    return std::max(0.0, result);
}

/** @brief Entropy of the gear Bernoulli distribution: softplus(l) - sigmoid(l) l. */
inline double GearEntropy(double logit) noexcept
{
    return Softplus(logit) - Sigmoid(logit) * logit;
}

/**
 * @brief Analytic entropy of the active axes: sum of log(2 pi e) / 2 + log sigma per active
 *        Gaussian axis, plus GearEntropy when the gear is active.
 */
inline double Entropy(PolicyDistribution const& policy, bool brakeActive = true, bool gearActive = false) noexcept
{
    double entropy = 0;
    if (gearActive)
    {
        entropy += GearEntropy(policy.output[GearOutput]);
    }
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        if (ActiveAxis(axis, brakeActive))
        {
            entropy += 0.5 * std::log(2 * Pi * std::exp(1.0)) + policy.logStandardDeviations[axis];
        }
    }
    return entropy;
}

/**
 * @brief Maps the raw ground-roll brake action linearly over its whole range.
 *
 * Raw -1 releases the brakes, zero applies half braking and +1 full braking.
 * Clamping a Gaussian at zero instead would map every negative sample to the
 * same released brake: the policy then drifts into that dead zone and never
 * explores braking. A neutral output now stops the T-6 on the runway.
 * @param raw Raw sampled or deterministic brake action.
 * @return Brake command within [0, 1].
 */
inline double BrakeCommand(double raw) noexcept
{
    return 0.5 * (std::clamp(raw, -1.0, 1.0) + 1.0);
}

/**
 * @brief Maps clipped raw Gaussian actions to physical controls around the trimmed approach.
 *
 * A zero actor output flies the trimmed 3-degree approach: the elevator adds the
 * action to its trim with a uniform gain, and the throttle spans from its trim
 * to idle for negative actions and to full power for positive ones. The brake
 * follows BrakeCommand during the ground roll and stays released in flight. The gear lever
 * requests the extension only when an active gear draw asks for it; the environment keeps a
 * latched extension.
 * @param action Raw sampled or deterministic actor action.
 * @param initial Trimmed approach controls.
 * @return Physically bounded controls.
 */
inline Controls ToControls(PolicyAction const& action, InitialSettings const& initial) noexcept
{
    double const neutralThrottle = initial.throttle;
    double const throttleCommand = std::clamp(action.raw[ThrottleAction], -1.0, 1.0);
    // Piecewise-linear throttle: [-1, 0] -> [0, trim], [0, 1] -> [trim, 1].
    double throttleSpan = neutralThrottle;
    if (throttleCommand >= 0)
    {
        throttleSpan = 1.0 - neutralThrottle;
    }
    double const throttle = neutralThrottle + throttleCommand * throttleSpan;
    Controls controls{};
    controls.elevator = std::clamp(initial.elevatorTrim + action.raw[0], -1.0, 1.0);
    controls.aileron = std::clamp(action.raw[1], -1.0, 1.0);
    controls.rudder = std::clamp(action.raw[2], -1.0, 1.0);
    controls.throttle = std::clamp(throttle, 0.0, 1.0);
    controls.brake = 0.0;
    if (action.brakeActive)
    {
        controls.brake = BrakeCommand(action.raw[BrakeAction]);
    }
    if (action.gearActive && action.gearDown)
    {
        controls.gear = 1.0;
    }
    return controls;
}

/**
 * @brief Samples a raw action from the policy, or returns its means.
 *
 * Draws one std::normal_distribution sample per active axis, in axis order, then one
 * std::bernoulli_distribution sample for an active gear; inactive axes draw nothing, so the
 * RNG stream depends on the phase and on the gear latch. A deterministic gear requests the
 * extension when its logit is non-negative.
 * @param policy Actor means, gear logit and log-std.
 * @param rng Random engine (the learner RNG).
 * @param deterministic True to return the means without drawing.
 * @param brakeActive True during the ground roll.
 * @param gearActive True while the gear extension is not yet commanded.
 * @return Raw action with its log-probability and the frozen collection distribution.
 * @throws std::runtime_error for non-finite means or deviations.
 */
template<typename Engine>
PolicyAction Sample(PolicyDistribution const& policy, Engine& rng, bool deterministic,
    bool brakeActive = true, bool gearActive = false)
{
    for (double const value : policy.output)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("FlightControl3D: non-finite actor output");
        }
    }
    for (double const value : policy.logStandardDeviations)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("FlightControl3D: non-finite global log standard deviation");
        }
    }
    PolicyAction action{};
    action.policy = policy;
    action.brakeActive = brakeActive;
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        if (!ActiveAxis(axis, brakeActive))
        {
            continue;
        }
        double const deviation = std::exp(policy.logStandardDeviations[axis]);
        if (!std::isfinite(deviation) || deviation <= 0)
        {
            throw std::runtime_error("FlightControl3D: invalid global Gaussian deviation");
        }
        std::normal_distribution<double> distribution(policy.output[axis], deviation);
        // The ternary keeps the draw conditional: a deterministic action consumes no random number.
        action.raw[axis] = deterministic ? policy.output[axis] : distribution(rng);
    }
    action.gearActive = gearActive;
    if (gearActive)
    {
        double const logit = policy.output[GearOutput];
        if (deterministic)
        {
            action.gearDown = logit >= 0;
        }
        else
        {
            action.gearDown = std::bernoulli_distribution(Sigmoid(logit))(rng);
        }
    }
    action.logProbability = LogProbability(policy, action);
    return action;
}

/** @brief Clipped PPO loss of one sample and its derivatives. */
struct PolicyLoss final
{
    /** @brief -min(r A, clip(r) A) - entropy coefficient * entropy. */
    double loss{};
    /** @brief Entropy of the current distribution on the active axes. */
    double entropy{};
    /** @brief Likelihood ratio r = pi_new / pi_old, computed from the bounded log-ratio. */
    double ratio{};
    /** @brief d loss / d mean, per axis. */
    PolicyVector outputDerivative{};
    /** @brief d loss / d log-std, per axis. */
    LogStandardDeviations logStandardDeviationDerivative{};
};

/**
 * @brief Exact clipped PPO surrogate derivative, including the log-std entropy bonus.
 *
 * r = exp(clamp(log pi_new - log pi_old, -20, 20)). The surrogate -min(r A, clip(r, 1 - eps,
 * 1 + eps) A) has a non-zero gradient only where the unclipped term is the active minimum:
 * A >= 0 and r <= 1 + eps, or A < 0 and r >= 1 - eps. There d loss / d log pi = -A r, and
 * d log pi / d mu = (x - mu) / sigma^2, d log pi / d log sigma = (x - mu)^2 / sigma^2 - 1.
 * The entropy term adds -entropy coefficient to each active log-std derivative. An active
 * gear adds d log p / d l = gearDown - sigmoid(l) and its Bernoulli entropy derivative.
 */
inline PolicyLoss ClippedLoss(PolicyDistribution const& policy, PolicyAction const& action,
    double advantage, TrainingSettings const& settings) noexcept
{
    PolicyLoss result{};
    double const logRatio = LogProbability(policy, action) - action.logProbability;
    result.ratio = std::exp(std::clamp(logRatio, -MaximumAbsoluteLogRatio, MaximumAbsoluteLogRatio));
    double const clipped = std::clamp(result.ratio, 1 - settings.clip, 1 + settings.clip);
    result.entropy = Entropy(policy, action.brakeActive, action.gearActive);
    result.loss = -std::min(result.ratio * advantage, clipped * advantage) - settings.entropy
        * result.entropy;
    bool unclippedActive = result.ratio >= 1 - settings.clip;
    if (advantage >= 0)
    {
        unclippedActive = result.ratio <= 1 + settings.clip;
    }
    double logProbabilityDerivative = 0;
    if (unclippedActive)
    {
        logProbabilityDerivative = -advantage * result.ratio;
    }
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        if (!ActiveAxis(axis, action.brakeActive))
        {
            continue;
        }
        double const inverseVariance = std::exp(-2 * policy.logStandardDeviations[axis]);
        double const difference = action.raw[axis] - policy.output[axis];
        result.outputDerivative[axis] = logProbabilityDerivative * difference * inverseVariance;
        result.logStandardDeviationDerivative[axis] =
            logProbabilityDerivative * (difference * difference * inverseVariance - 1) - settings.entropy;
    }
    if (action.gearActive)
    {
        // d log p / d l = gearDown - sigmoid(l); d(-c H) / d l = c l sigmoid(l) (1 - sigmoid(l)).
        double const logit = policy.output[GearOutput];
        double const probability = Sigmoid(logit);
        double const drawn = action.gearDown ? 1.0 : 0.0;
        result.outputDerivative[GearOutput] = logProbabilityDerivative * (drawn - probability)
            + settings.entropy * logit * probability * (1 - probability);
    }
    return result;
}

/**
 * @brief One-step TD error of GAE: bootstrap on time limits, but never carry value across a reset.
 *
 * delta = r + gamma * V(s') - V(s), with V(s') = 0 only for a terminated (absorbing)
 * transition; a truncated transition keeps its bootstrap.
 */
inline double AdvantageDelta(double reward, double value, double nextValue, bool terminated,
    double gamma) noexcept
{
    return reward + (terminated ? 0 : gamma * nextValue) - value;
}

} // namespace ppo::examples::flight
