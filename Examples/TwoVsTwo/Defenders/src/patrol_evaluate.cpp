#include "patrol_evaluator.hpp"
#include "patrol_policy.hpp"
#include "patrol_config.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace patrol = ppo::examples::twovstwo::defenders;

namespace {
std::size_t ParseCount(char const* value)
{
    std::string const input{value};
    std::size_t consumed = 0U;
    auto const parsed = std::stoull(input, &consumed, 10);
    if (consumed != input.size() || parsed == 0U
        || parsed > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Expected positive integer");
    return static_cast<std::size_t>(parsed);
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc > 4)
        {
            std::cerr << "Usage: PPOPatrolEvaluate [checkpoint.txt] [episodes] [steps]\n";
            return 2;
        }
        std::filesystem::path const checkpoint = argc >= 2
            ? std::filesystem::path{argv[1]}
            : std::filesystem::path{PPO_BLUE_MODEL_PATH};
        patrol::PatrolPolicy policy{checkpoint};
        patrol::PatrolEvaluationSettings settings{};
        if (argc >= 3)
            settings.episodes = ParseCount(argv[2]);
        if (argc >= 4)
            settings.maxSteps = ParseCount(argv[3]);

        patrol::Parameters parameters{};
        // Evaluation uses the same aircraft and world parameters as training.
        parameters = patrol::LoadPatrolParameters(
            std::filesystem::path{PPO_BLUE_SETTINGS_DIR});
        patrol::PatrolEvaluator evaluator{parameters, settings};
        auto const report = evaluator.Evaluate(
            [&](patrol::Observation const& observation) {
                return policy.Predict(observation);
            });
        for (std::size_t i = 0; i < report.episodes.size(); ++i)
        {
            auto const& episode = report.episodes[i];
            std::cout << "test=" << i << " side=" << episode.side
                      << " steps=" << episode.steps
                      << " outcome=" << static_cast<int>(episode.outcome)
                      << " failure=" << static_cast<int>(episode.failure)
                      << " tracking=" << episode.trackingFraction << '\n';
        }
        std::cout << "Patrol evaluation: success=" << report.successes
                  << " failure=" << report.failures
                  << " truncated=" << report.truncations << '\n';
        return report.Passed() ? 0 : 1;
    }
    catch (std::exception const& error)
    {
        std::cerr << "Patrol evaluation failed: " << error.what() << '\n';
        return 2;
    }
}
