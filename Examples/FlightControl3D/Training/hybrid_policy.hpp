#pragma once

#include "flight_config.hpp"
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace ppo::examples::flight {

using PolicyVector = std::array<double, PolicyOutputs>;
struct PolicyAction final
{
    std::array<double, ContinuousActions> raw{};
    bool gearDown{};
    double logProbability{};
    bool brakeActive{true}, gearActive{true};
    /** @brief Frozen collection distribution for the exact rollout KL guard. */
    PolicyVector policy{};
};

inline double Softplus(double x) noexcept { return std::max(x,0.0)+std::log1p(std::exp(-std::abs(x))); }
inline double Sigmoid(double x) noexcept
{
    double const e = std::exp(-std::abs(x));
    return x >= 0 ? 1/(1+e) : e/(1+e);
}

/** Density of the latent Gaussian sample and Bernoulli gear, not of a clipped action. */
inline double LogProbability(PolicyVector const& output, PolicyAction const& action, TrainingSettings const& t) noexcept
{
    double logp = action.gearActive
        ? (action.gearDown ? -Softplus(-output.back()) : -Softplus(output.back())) : 0;
    for (std::size_t i=0; i<ContinuousActions; ++i)
    {
        if(i==4 && !action.brakeActive) continue;
        double const logStd = std::clamp(output[ContinuousActions+i],t.minLogStd,t.maxLogStd);
        double const z = (action.raw[i]-output[i])*std::exp(-logStd);
        logp -= 0.5*z*z + logStd + 0.5*std::log(2*Pi);
    }
    return logp;
}

/** @brief Exact old-to-new Gaussian/Bernoulli KL on the physically active controls. */
inline double PolicyKl(PolicyVector const& previous,PolicyVector const& current,
    PolicyAction const& action,TrainingSettings const& t) noexcept
{
    double result=0;
    for(std::size_t i=0;i<ContinuousActions;++i)
    {
        if(i==4 && !action.brakeActive) continue;
        double const oldStd=std::clamp(previous[ContinuousActions+i],t.minLogStd,t.maxLogStd);
        double const newStd=std::clamp(current[ContinuousActions+i],t.minLogStd,t.maxLogStd);
        double const logScale=oldStd-newStd;
        double const difference=previous[i]-current[i];
        result+=0.5*std::expm1(2*logScale)-logScale
            +0.5*difference*difference*std::exp(-2*newStd);
    }
    if(action.gearActive)
    {
        double const probability=Sigmoid(previous.back());
        result+=probability*(Softplus(-current.back())-Softplus(-previous.back()))
            +(1-probability)*(Softplus(current.back())-Softplus(previous.back()));
    }
    return std::max(0.0,result);
}

/** Analytic entropy of the latent Gaussian x Bernoulli policy. */
inline double Entropy(PolicyVector const& output, TrainingSettings const& t,
    bool brakeActive=true,bool gearActive=true) noexcept
{
    double const p = Sigmoid(output.back());
    double entropy = gearActive ? Softplus(output.back())-p*output.back() : 0;
    for (std::size_t i=0; i<ContinuousActions; ++i)
        if(i!=4 || brakeActive)
            entropy += 0.5*std::log(2*Pi*std::exp(1.0)) + std::clamp(output[ContinuousActions+i],t.minLogStd,t.maxLogStd);
    return entropy;
}

/** BlueAgent-style clipping of raw Gaussian actions; throttle zero holds the configured reference. */
inline Controls ToControls(PolicyAction const& action, double neutralThrottle) noexcept
{
    double const throttleCommand=std::clamp(action.raw[3],-1.0,1.0);
    double const throttle=neutralThrottle+throttleCommand
        *(throttleCommand>=0 ? 1.0-neutralThrottle : neutralThrottle);
    return {std::clamp(action.raw[0],-1.0,1.0),std::clamp(action.raw[1],-1.0,1.0),
        std::clamp(action.raw[2],-1.0,1.0),std::clamp(throttle,0.0,1.0),
        std::clamp(action.raw[4],0.0,1.0),action.gearDown};
}

template<typename Engine>
PolicyAction Sample(PolicyVector const& output, TrainingSettings const& t, Engine& rng, bool deterministic,
    bool brakeActive=true,bool gearActive=true,bool fixedGearDown=false)
{
    for (double value : output)
        if (!std::isfinite(value)) throw std::runtime_error("FlightControl3D: non-finite actor output");
    PolicyAction action{};
    action.policy=output;
    action.brakeActive=brakeActive;action.gearActive=gearActive;
    for (std::size_t i=0; i<ContinuousActions; ++i)
    {
        if(i==4 && !brakeActive) continue;
        std::normal_distribution<double> distribution(output[i],std::exp(std::clamp(output[ContinuousActions+i],t.minLogStd,t.maxLogStd)));
        action.raw[i] = deterministic ? output[i] : distribution(rng);
    }
    std::bernoulli_distribution gear(Sigmoid(output.back()));
    action.gearDown = gearActive ? (deterministic ? output.back() >= 0 : gear(rng)) : fixedGearDown;
    action.logProbability = LogProbability(output,action,t);
    return action;
}

struct PolicyLoss final { double loss{}, entropy{}, ratio{}; PolicyVector derivative{}; };

/** Exact clipped PPO surrogate derivative, including Bernoulli and log-std entropy. */
inline PolicyLoss ClippedLoss(PolicyVector const& output, PolicyAction const& action,
    double advantage, TrainingSettings const& t) noexcept
{
    PolicyLoss result{};
    double const logRatio = LogProbability(output,action,t)-action.logProbability;
    // Limit only exponential overflow; outside this range fail-stop is handled by the learner.
    result.ratio = std::exp(std::clamp(logRatio,-80.0,80.0));
    double const clipped = std::clamp(result.ratio,1-t.clip,1+t.clip);
    result.entropy = Entropy(output,t,action.brakeActive,action.gearActive);
    result.loss = -std::min(result.ratio*advantage,clipped*advantage)-t.entropy*result.entropy;
    bool const active = (advantage >= 0 ? result.ratio <= 1+t.clip : result.ratio >= 1-t.clip)
        && logRatio > -80 && logRatio < 80;
    double const dlogp = active ? -advantage*result.ratio : 0;
    for (std::size_t i=0; i<ContinuousActions; ++i)
    {
        if(i==4 && !action.brakeActive) continue;
        double const rawLogStd = output[ContinuousActions+i];
        double const invVariance = std::exp(-2*std::clamp(rawLogStd,t.minLogStd,t.maxLogStd));
        double const difference = action.raw[i]-output[i];
        result.derivative[i] = dlogp*difference*invVariance;
        if (rawLogStd > t.minLogStd && rawLogStd < t.maxLogStd)
            result.derivative[ContinuousActions+i] = dlogp*(difference*difference*invVariance-1)-t.entropy;
    }
    double const p = Sigmoid(output.back());
    if(action.gearActive)
        result.derivative.back() = dlogp*(static_cast<double>(action.gearDown)-p) + t.entropy*output.back()*p*(1-p);
    return result;
}

/** GAE: bootstrap on time limits, but never carry advantage across a reset. */
inline double AdvantageDelta(double reward, double value, double nextValue, bool terminated, double gamma) noexcept
{
    return reward + (terminated ? 0 : gamma*nextValue)-value;
}

} // namespace ppo::examples::flight
