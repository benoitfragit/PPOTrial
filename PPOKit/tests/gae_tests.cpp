/**
 * @file PPOKit/tests/gae_tests.cpp
 * @brief Semantics and bit-exactness of the shared GAE helpers.
 */
#include "ppokit/core/gae.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

namespace {

using ppokit::core::GaeStep;
using ppokit::core::TemporalDifference;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

template<typename Scalar>
bool SameBits(Scalar left, Scalar right) noexcept
{
    return std::memcmp(&left, &right, sizeof(Scalar)) == 0;
}

/** @brief Formula of BlueDefense GeneralizedMissionAdvantage before the migration (float). */
float BlueReference(float reward, float value, float nextValue, float followingAdvantage, float gamma,
    float lambda, bool terminated, bool truncated) noexcept
{
    float const bootstrap = terminated ? 0.F : nextValue;
    float const trace = terminated || truncated ? 0.F : followingAdvantage;
    return reward + gamma * bootstrap - value + gamma * lambda * trace;
}

/** @brief Formula of FlightControl3D ComputeAdvantages before the migration (double). */
double FlightReference(double reward, double value, double nextValue, double followingAdvantage, double gamma,
    double lambda, bool terminated, bool truncated) noexcept
{
    double const delta = reward + (terminated ? 0 : gamma * nextValue) - value;
    return delta + ((terminated || truncated) ? 0 : gamma * lambda * followingAdvantage);
}

void EpisodeEndSemantics()
{
    constexpr double Gamma = 0.9;
    constexpr double Lambda = 0.5;
    // Running transition: bootstrap and trace both used.
    double const running = GaeStep(1.0, 2.0, 3.0, 4.0, Gamma, Lambda, false, false);
    Check(std::abs(running - (1.0 + Gamma * 3.0 - 2.0 + Gamma * Lambda * 4.0)) < 1e-12,
        "running transition bootstraps and continues the trace");
    // Truncation (time limit): bootstrap kept, trace cut.
    double const truncated = GaeStep(1.0, 2.0, 3.0, 4.0, Gamma, Lambda, false, true);
    Check(std::abs(truncated - (1.0 + Gamma * 3.0 - 2.0)) < 1e-12,
        "truncation keeps the bootstrap but stops the trace");
    // Termination: no bootstrap, no trace.
    double const terminated = GaeStep(1.0, 2.0, 3.0, 4.0, Gamma, Lambda, true, false);
    Check(std::abs(terminated - (1.0 - 2.0)) < 1e-12, "termination drops bootstrap and trace");
    Check(TemporalDifference(1.0, 2.0, 1e300, true, Gamma) == -1.0,
        "a terminal next value is never read");
    // A non-finite advantage of the next episode must not leak across a reset.
    double const isolated = GaeStep(1.0, 2.0, 3.0, std::nan(""), Gamma, Lambda, false, true);
    Check(std::isfinite(isolated), "next-episode advantage cannot cross a truncation");
}

template<typename Scalar, typename Reference>
void MatchesReferenceBitForBit(Reference reference, char const* description)
{
    std::mt19937 rng(20261121U);
    std::uniform_real_distribution<Scalar> magnitude(Scalar{-50}, Scalar{50});
    std::array<Scalar, 3> const gammas{Scalar(0.99), Scalar(0.995), Scalar(1)};
    std::array<Scalar, 3> const lambdas{Scalar(0.9), Scalar(0.95), Scalar(0)};
    bool identical = true;
    for (int sample = 0; sample < 200000; ++sample)
    {
        Scalar const reward = magnitude(rng);
        Scalar const value = magnitude(rng);
        Scalar const nextValue = magnitude(rng);
        Scalar const following = magnitude(rng);
        Scalar const gamma = gammas[static_cast<std::size_t>(sample) % gammas.size()];
        Scalar const lambda = lambdas[static_cast<std::size_t>(sample / 3) % lambdas.size()];
        bool const terminated = sample % 7 == 0;
        bool const truncated = sample % 5 == 0;
        Scalar const expected = reference(reward, value, nextValue, following, gamma, lambda, terminated, truncated);
        Scalar const actual = GaeStep(reward, value, nextValue, following, gamma, lambda, terminated, truncated);
        identical = identical && SameBits(expected, actual);
    }
    // Signed zeros are the classic place where a reordered expression diverges.
    for (Scalar const zero : {Scalar(0), -Scalar(0)})
    {
        for (bool const terminated : {false, true})
        {
            for (bool const truncated : {false, true})
            {
                Scalar const expected = reference(zero, zero, zero, zero, Scalar(0.99), Scalar(0.95), terminated, truncated);
                Scalar const actual = GaeStep(zero, zero, zero, zero, Scalar(0.99), Scalar(0.95), terminated, truncated);
                identical = identical && SameBits(expected, actual);
            }
        }
    }
    Check(identical, description);
}

} // namespace

int main()
{
    EpisodeEndSemantics();
    MatchesReferenceBitForBit<float>(BlueReference, "GaeStep<float> reproduces the BlueDefense formula bit for bit");
    MatchesReferenceBitForBit<double>(FlightReference, "GaeStep<double> reproduces the FlightControl3D formula bit for bit");
    if (failures != 0)
    {
        std::cerr << failures << " GAE check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit GAE tests passed\n";
    return EXIT_SUCCESS;
}
