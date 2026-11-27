/**
 * @file flight_reward_replay.cpp
 * @brief Replay a Flight ABI capture against a separately built historical FlightEnvironment.
 * @details Usage: flight_reward_replay MASTER_SCENARIO TRACE. Use the same x64 host and JSBSim
 * data for both processes. Zero exit requires equality at every recorded decision and boundary.
 */
#include "flight_environment.hpp"
#include "flight_policy.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace ppo::examples::flight;

namespace {

constexpr std::uint32_t CampaignSize = 20;
constexpr std::uint32_t MaximumSteps = 12000;
static_assert(ObservationSize == 35 && ContinuousActions + 1 == 6 && RewardCount == 7, "unsupported Flight trace");

/** @brief Decode one unpadded field from the x64 little-endian trace. */
template<class T>
T Read(std::istream& input)
{
    static_assert(std::is_trivially_copyable_v<T>, "trace fields must have a fixed representation");
    T value{};
    if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) throw std::runtime_error("truncated Flight trace");
    return value;
}

/** @brief Absolute error on finite simulator outputs. */
double Error(double expected, double recorded)
{
    if (!std::isfinite(expected) || !std::isfinite(recorded)) throw std::runtime_error("non-finite Flight trace");
    return std::abs(expected - recorded);
}

struct Differences final
{
    std::array<double, RewardCount> channels{};
    std::array<std::uint64_t, RewardCount> nonzero{};
    double observation{};
    double reward{};
    std::uint64_t transitions{};
    std::uint32_t stops{};
};

/** @brief Replay identical controls and compare the seven native reward channels before aggregation. */
void ReplayEpisode(std::istream& input, FlightEnvironment& environment, Config const& config,
    std::uint32_t seed, std::uint32_t index, std::uint32_t steps, Differences& differences)
{
    auto const start = EvaluationStart(index, CampaignSize, config.training.curriculum.rolloutStartShare, 1.0);
    environment.Reset(seed, start);
    for (std::uint32_t step = 1; step <= steps; ++step)
    {
        auto const expected = environment.ObservationNow();
        auto const recorded = Read<Observation>(input);
        auto const actions = Read<std::array<float, ContinuousActions + 1>>(input);
        auto const channels = Read<std::array<double, RewardCount>>(input);
        auto const reward = Read<double>(input);
        auto const end = Read<std::uint32_t>(input);
        for (std::size_t k = 0; k < expected.size(); ++k)
            differences.observation = std::max(differences.observation, Error(expected[k], recorded[k]));
        PolicyAction policy;
        std::copy_n(actions.begin(), ContinuousActions, policy.raw.begin());
        policy.brakeActive = expected[RolloutObservation] > 0.5F;
        policy.gearActive = expected[GearCommandObservation] <= 0.5F;
        policy.gearDown = actions.back() > 0.5F;
        auto const result = environment.Step(ToControls(policy, config.initial));
        for (std::size_t k = 0; k < RewardCount; ++k)
        {
            differences.channels[k] = std::max(differences.channels[k], Error(result.rewards[k], channels[k]));
            if (result.rewards[k] != 0.0) ++differences.nonzero[k];
        }
        differences.reward = std::max(differences.reward, Error(result.reward, reward));
        std::uint32_t nativeEnd = 0;
        if (result.terminated) nativeEnd = 1;
        else if (result.truncated) nativeEnd = 2;
        if (end != nativeEnd || (end != 0) != (step == steps))
            throw std::runtime_error("different episode boundary at seed " + std::to_string(seed)
                + " step " + std::to_string(step));
        ++differences.transitions;
    }
    if (environment.Mission().outcome == Outcome::Stopped) ++differences.stops;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    try
    {
        auto const config = LoadConfig(argv[1]);
        std::ifstream input(argv[2], std::ios::binary);
        auto const episodes = Read<std::uint32_t>(input);
        if (episodes != 2 * CampaignSize) throw std::runtime_error("expected two 20-episode campaigns");
        std::unique_ptr<FlightEnvironment> environment;
        Differences differences;
        for (std::uint32_t i = 0; i < episodes; ++i)
        {
            auto const seed = Read<std::uint32_t>(input);
            auto const index = Read<std::uint32_t>(input);
            auto const size = Read<std::uint32_t>(input);
            auto const steps = Read<std::uint32_t>(input);
            auto const first = i < CampaignSize ? 4342U : 90001U;
            if (index != i % CampaignSize || size != CampaignSize || seed != first + index
                || steps == 0 || steps > MaximumSteps) throw std::runtime_error("invalid Flight episode header");
            if (index == 0)
            {
                // The native simulator lives for one campaign, as specified by scope=campaign.
                environment = std::make_unique<FlightEnvironment>(config, DefaultDataRoot(), seed);
                environment->SetDifficulty(1.0);
            }
            ReplayEpisode(input, *environment, config, seed, index, steps, differences);
        }
        if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing Flight records");
        std::cout << std::setprecision(17) << "episodes " << episodes << " transitions " << differences.transitions
                  << " stops " << differences.stops << " max_observation " << differences.observation
                  << " max_reward " << differences.reward << '\n';
        for (std::size_t k = 0; k < RewardCount; ++k)
            std::cout << RewardNames[k] << " max " << differences.channels[k] << " nonzero "
                      << differences.nonzero[k] << '\n';
        if (differences.observation != 0.0 || differences.reward != 0.0
            || *std::max_element(differences.channels.begin(), differences.channels.end()) != 0.0) return 1;
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
