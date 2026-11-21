/**
 * @file PPOKit/tests/heads_tests.cpp
 * @brief Gaussian head and clipped PPO surrogate: values, gradients and parity with the former formulas.
 */
#include "ppokit/core/ppo_surrogate.hpp"
#include "ppokit/heads/bernoulli.hpp"
#include "ppokit/heads/gaussian.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {

using ppokit::core::EvaluateClippedSurrogate;
using ppokit::heads::GaussianAxisEntropy;
using ppokit::heads::GaussianAxisLogDensity;
using ppokit::heads::GaussianAxisLogDensityGradient;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief Relative difference with an absolute floor, for values that may be near zero. */
template<typename Scalar>
Scalar RelativeDifference(Scalar left, Scalar right) noexcept
{
    Scalar const scale = std::max({std::abs(left), std::abs(right), Scalar(1)});
    return std::abs(left - right) / scale;
}

/** @brief FlightControl3D log-density per axis before PPOKit (double). */
double FormerFlightLogDensity(double sample, double mean, double logStd) noexcept
{
    constexpr double Pi = 3.14159265358979323846;
    double const standardized = (sample - mean) * std::exp(-logStd);
    return -(0.5 * standardized * standardized + logStd + 0.5 * std::log(2 * Pi));
}

/** @brief BlueDefense log-density per axis before PPOKit (float). */
float FormerBlueLogDensity(float sample, float mean, float logStd) noexcept
{
    constexpr float TwoPi = 6.28318530717958647692F;
    float const normalized = (sample - mean) / std::exp(logStd);
    return -0.5F * normalized * normalized - logStd - 0.5F * std::log(TwoPi);
}

void GaussianValues()
{
    double const unit = GaussianAxisLogDensity(0.0, 0.0, 0.0);
    Check(std::abs(unit + 0.5 * std::log(2 * 3.14159265358979323846)) < 1e-15, "unit Gaussian density at its mean");
    Check(std::abs(GaussianAxisEntropy(0.0) - 0.5 * std::log(2 * 3.14159265358979323846 * std::exp(1.0))) < 1e-15,
        "unit Gaussian entropy is log(2 pi e) / 2");
    Check(GaussianAxisEntropy(-1.5) == -1.5 + GaussianAxisEntropy(0.0), "entropy grows one-for-one with log-std");
}

void GaussianGradientMatchesFiniteDifferences()
{
    constexpr double Step = 1e-6;
    for (double const sample : {-1.3, 0.2, 2.7})
    {
        for (double const mean : {-0.4, 0.9})
        {
            for (double const logStd : {-3.0, -1.0, 0.5})
            {
                auto const gradient = GaussianAxisLogDensityGradient(sample, mean, logStd);
                double const byMean = (GaussianAxisLogDensity(sample, mean + Step, logStd)
                    - GaussianAxisLogDensity(sample, mean - Step, logStd)) / (2 * Step);
                double const byLogStd = (GaussianAxisLogDensity(sample, mean, logStd + Step)
                    - GaussianAxisLogDensity(sample, mean, logStd - Step)) / (2 * Step);
                Check(RelativeDifference(gradient.mean, byMean) < 1e-6, "d log p / d mean");
                Check(RelativeDifference(gradient.logStd, byLogStd) < 1e-6, "d log p / d log-std");
            }
        }
    }
}

/** @brief The common formula differs from the former ones only by rounding. */
void GaussianParityWithFormerFormulas()
{
    std::mt19937 rng(17U);
    std::uniform_real_distribution<double> value(-3.0, 3.0);
    std::uniform_real_distribution<double> logStd(-3.5, 1.0);
    double worstDouble = 0;
    float worstFloat = 0;
    for (int sample = 0; sample < 100000; ++sample)
    {
        double const x = value(rng);
        double const mean = value(rng);
        double const scale = logStd(rng);
        worstDouble = std::max(worstDouble,
            RelativeDifference(GaussianAxisLogDensity(x, mean, scale), FormerFlightLogDensity(x, mean, scale)));
        auto const xf = static_cast<float>(x);
        auto const meanf = static_cast<float>(mean);
        auto const scalef = static_cast<float>(scale);
        worstFloat = std::max(worstFloat,
            RelativeDifference(GaussianAxisLogDensity(xf, meanf, scalef), FormerBlueLogDensity(xf, meanf, scalef)));
    }
    std::cout << "Gaussian log-density, worst relative change: double " << worstDouble << ", float " << worstFloat
              << '\n';
    Check(worstDouble < 1e-13, "double log-density equals the former FlightControl3D formula up to rounding");
    Check(worstFloat < 2e-5F, "float log-density equals the former BlueDefense formula up to rounding");
}

void BernoulliHead()
{
    using ppokit::heads::BernoulliEntropy;
    using ppokit::heads::BernoulliEntropyGradient;
    using ppokit::heads::BernoulliLogProbability;
    using ppokit::heads::BernoulliLogProbabilityGradient;
    Check(std::abs(BernoulliEntropy(0.0) - std::log(2.0)) < 1e-15, "a fair decision has entropy log 2");
    Check(std::abs(std::exp(BernoulliLogProbability(0.7, true)) + std::exp(BernoulliLogProbability(0.7, false)) - 1.0)
            < 1e-15,
        "the two outcomes sum to probability one");
    for (double const logit : {-1000.0, 1000.0})
    {
        Check(std::isfinite(BernoulliLogProbability(logit, true)) && std::isfinite(BernoulliLogProbability(logit, false))
                && std::isfinite(BernoulliEntropy(logit)),
            "extreme logits stay finite");
    }
    constexpr double Step = 1e-6;
    for (double const logit : {-3.0, -0.2, 0.4, 2.5})
    {
        for (bool const drawn : {false, true})
        {
            double const numerical = (BernoulliLogProbability(logit + Step, drawn)
                - BernoulliLogProbability(logit - Step, drawn)) / (2 * Step);
            Check(RelativeDifference(BernoulliLogProbabilityGradient(logit, drawn), numerical) < 1e-6,
                "d log p / d logit");
        }
        double const entropySlope = (BernoulliEntropy(logit + Step) - BernoulliEntropy(logit - Step)) / (2 * Step);
        Check(RelativeDifference(BernoulliEntropyGradient(logit), entropySlope) < 1e-6, "d entropy / d logit");
    }
}

void SurrogateRegions()
{
    constexpr double Clip = 0.2;
    // Inside the trust region the unclipped term drives the gradient.
    auto const inside = EvaluateClippedSurrogate(std::log(1.1), 0.0, 2.0, Clip);
    Check(std::abs(inside.objective - 1.1 * 2.0) < 1e-12, "unclipped objective inside the range");
    Check(std::abs(inside.logProbabilityGradient + 2.0 * 1.1) < 1e-12, "gradient -A r inside the range");
    Check(!inside.outsideClipRange, "1.1 is inside 1 +/- 0.2");
    // A positive advantage gains nothing beyond 1 + clip.
    auto const tooFar = EvaluateClippedSurrogate(std::log(1.5), 0.0, 2.0, Clip);
    Check(std::abs(tooFar.objective - 1.2 * 2.0) < 1e-12, "objective capped at (1 + clip) A");
    Check(tooFar.logProbabilityGradient == 0.0 && tooFar.outsideClipRange, "no gradient past 1 + clip for A > 0");
    // A negative advantage keeps pushing a ratio that grew, but not one already below 1 - clip.
    auto const negativeHigh = EvaluateClippedSurrogate(std::log(1.5), 0.0, -2.0, Clip);
    Check(negativeHigh.logProbabilityGradient == 2.0 * 1.5, "A < 0 still corrects a ratio above 1 + clip");
    auto const negativeLow = EvaluateClippedSurrogate(std::log(0.5), 0.0, -2.0, Clip);
    Check(negativeLow.logProbabilityGradient == 0.0, "no gradient below 1 - clip for A < 0");
    // A zero advantage contributes neither objective nor push.
    auto const neutral = EvaluateClippedSurrogate(std::log(1.7), 0.0, 0.0, Clip);
    Check(neutral.objective == 0.0 && neutral.logProbabilityGradient == 0.0, "zero advantage is neutral");
    // The log-ratio is bounded before exp, so extreme likelihoods stay finite.
    auto const extreme = EvaluateClippedSurrogate(5000.0F, -5000.0F, 1.0F, 0.2F);
    Check(extreme.logRatio == 20.0F && std::isfinite(extreme.ratio), "bounded log-ratio in single precision");
}

void SurrogateGradientMatchesFiniteDifferences()
{
    constexpr double Step = 1e-7;
    for (double const advantage : {-1.7, 0.6})
    {
        for (double const logRatio : {-0.5, -0.05, 0.05, 0.5})
        {
            auto const center = EvaluateClippedSurrogate(logRatio, 0.0, advantage, 0.2);
            double const plus = EvaluateClippedSurrogate(logRatio + Step, 0.0, advantage, 0.2).objective;
            double const minus = EvaluateClippedSurrogate(logRatio - Step, 0.0, advantage, 0.2).objective;
            double const numerical = -(plus - minus) / (2 * Step);
            Check(RelativeDifference(center.logProbabilityGradient, numerical) < 1e-6,
                "d(-objective) / d log pi matches finite differences");
        }
    }
}

} // namespace

int main()
{
    GaussianValues();
    GaussianGradientMatchesFiniteDifferences();
    GaussianParityWithFormerFormulas();
    BernoulliHead();
    SurrogateRegions();
    SurrogateGradientMatchesFiniteDifferences();
    if (failures != 0)
    {
        std::cerr << failures << " head check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit head tests passed\n";
    return EXIT_SUCCESS;
}
