#pragma once

#include "flight_config.hpp"
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace ppo::examples::flight {

/** @brief Actor outputs: one Gaussian mean per continuous control. */
inline constexpr std::size_t PolicyOutputs = ContinuousActions;
/** @brief Index of the brake action, active only on the ground. */
inline constexpr std::size_t BrakeAction = 4;
using PolicyVector = std::array<double, PolicyOutputs>;
using LogStandardDeviations = std::array<double, ContinuousActions>;
inline constexpr double MinimumLogStandardDeviation = -3.5;
inline constexpr double InitialLogStandardDeviation = -2.0;
inline constexpr double MaximumLogStandardDeviation = -1.0;
inline constexpr std::array<char const*, ContinuousActions> ActionNames{
    "elevator", "aileron", "rudder", "throttle", "brake"};
inline constexpr std::array<char const*, PolicyOutputs> PolicyOutputNames{
    "elevator_mean", "aileron_mean", "rudder_mean", "throttle_mean", "brake_mean"};

/** @brief State-dependent actor means and global Gaussian exploration parameters. */
struct PolicyDistribution final
{
    PolicyVector output{};
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

struct PolicyAction final
{
    std::array<double, ContinuousActions> raw{};
    double logProbability{};
    /** @brief The brake is sampled and optimized only while rolling on the ground. */
    bool brakeActive{true};
    /** @brief Frozen collection distribution used by likelihood ratios and analytic KL tests. */
    PolicyDistribution policy{};
};

/** @brief Whether an action axis takes part in sampling, likelihood and optimization. */
inline bool ActiveAxis(std::size_t axis, bool brakeActive) noexcept
{
    return axis != BrakeAction || brakeActive;
}

/** Density of the latent Gaussian sample, not of the clipped physical control. */
inline double LogProbability(PolicyDistribution const& policy, PolicyAction const& action) noexcept
{
    double logp = 0;
    for (std::size_t i = 0; i < ContinuousActions; ++i)
    {
        if (!ActiveAxis(i, action.brakeActive)) continue;
        double const logStd = policy.logStandardDeviations[i];
        double const z = (action.raw[i] - policy.output[i]) * std::exp(-logStd);
        logp -= 0.5 * z * z + logStd + 0.5 * std::log(2 * Pi);
    }
    return logp;
}

/** @brief Exact old-to-new Gaussian KL on the physically active controls. */
inline double PolicyKl(PolicyDistribution const& previous, PolicyDistribution const& current,
    PolicyAction const& action) noexcept
{
    double result = 0;
    for (std::size_t i = 0; i < ContinuousActions; ++i)
    {
        if (!ActiveAxis(i, action.brakeActive)) continue;
        double const logScale = previous.logStandardDeviations[i] - current.logStandardDeviations[i];
        double const difference = previous.output[i] - current.output[i];
        result += 0.5 * std::expm1(2 * logScale) - logScale
            + 0.5 * difference * difference * std::exp(-2 * current.logStandardDeviations[i]);
    }
    return std::max(0.0, result);
}

/** Analytic entropy of the active latent Gaussian axes. */
inline double Entropy(PolicyDistribution const& policy, bool brakeActive = true) noexcept
{
    double entropy = 0;
    for (std::size_t i = 0; i < ContinuousActions; ++i)
        if (ActiveAxis(i, brakeActive))
            entropy += 0.5 * std::log(2 * Pi * std::exp(1.0)) + policy.logStandardDeviations[i];
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
 * follows BrakeCommand during the ground roll and stays released in flight.
 * @param action Raw sampled or deterministic actor action.
 * @param initial Trimmed approach controls.
 * @return Physically bounded controls.
 */
inline Controls ToControls(PolicyAction const& action, InitialSettings const& initial) noexcept
{
    double const neutralThrottle = initial.throttle;
    double const throttleCommand = std::clamp(action.raw[3], -1.0, 1.0);
    double const throttle = neutralThrottle
        + throttleCommand * (throttleCommand >= 0 ? 1.0 - neutralThrottle : neutralThrottle);
    Controls controls{};
    controls.elevator = std::clamp(initial.elevatorTrim + action.raw[0], -1.0, 1.0);
    controls.aileron = std::clamp(action.raw[1], -1.0, 1.0);
    controls.rudder = std::clamp(action.raw[2], -1.0, 1.0);
    controls.throttle = std::clamp(throttle, 0.0, 1.0);
    controls.brake = action.brakeActive ? BrakeCommand(action.raw[BrakeAction]) : 0.0;
    return controls;
}

template<typename Engine>
PolicyAction Sample(PolicyDistribution const& policy, Engine& rng, bool deterministic, bool brakeActive = true)
{
    for (double value : policy.output)
        if (!std::isfinite(value)) throw std::runtime_error("FlightControl3D: non-finite actor output");
    for (double value : policy.logStandardDeviations)
        if (!std::isfinite(value)) throw std::runtime_error("FlightControl3D: non-finite global log standard deviation");
    PolicyAction action{};
    action.policy = policy;
    action.brakeActive = brakeActive;
    for (std::size_t i = 0; i < ContinuousActions; ++i)
    {
        if (!ActiveAxis(i, brakeActive)) continue;
        double const deviation = std::exp(policy.logStandardDeviations[i]);
        if (!std::isfinite(deviation) || deviation <= 0)
            throw std::runtime_error("FlightControl3D: invalid global Gaussian deviation");
        std::normal_distribution<double> distribution(policy.output[i], deviation);
        action.raw[i] = deterministic ? policy.output[i] : distribution(rng);
    }
    action.logProbability = LogProbability(policy, action);
    return action;
}

struct PolicyLoss final
{
    double loss{}, entropy{}, ratio{};
    PolicyVector outputDerivative{};
    LogStandardDeviations logStandardDeviationDerivative{};
};

/** Exact clipped PPO surrogate derivative, including the log-std entropy bonus. */
inline PolicyLoss ClippedLoss(PolicyDistribution const& policy, PolicyAction const& action,
    double advantage, TrainingSettings const& t) noexcept
{
    PolicyLoss result{};
    double const logRatio = LogProbability(policy, action) - action.logProbability;
    result.ratio = std::exp(std::clamp(logRatio, -20.0, 20.0));
    double const clipped = std::clamp(result.ratio, 1 - t.clip, 1 + t.clip);
    result.entropy = Entropy(policy, action.brakeActive);
    result.loss = -std::min(result.ratio * advantage, clipped * advantage) - t.entropy * result.entropy;
    bool const active = advantage >= 0 ? result.ratio <= 1 + t.clip : result.ratio >= 1 - t.clip;
    double const dlogp = active ? -advantage * result.ratio : 0;
    for (std::size_t i = 0; i < ContinuousActions; ++i)
    {
        if (!ActiveAxis(i, action.brakeActive)) continue;
        double const inverseVariance = std::exp(-2 * policy.logStandardDeviations[i]);
        double const difference = action.raw[i] - policy.output[i];
        result.outputDerivative[i] = dlogp * difference * inverseVariance;
        result.logStandardDeviationDerivative[i] = dlogp * (difference * difference * inverseVariance - 1) - t.entropy;
    }
    return result;
}

/** GAE: bootstrap on time limits, but never carry advantage across a reset. */
inline double AdvantageDelta(double reward, double value, double nextValue, bool terminated, double gamma) noexcept
{
    return reward + (terminated ? 0 : gamma * nextValue) - value;
}

} // namespace ppo::examples::flight
