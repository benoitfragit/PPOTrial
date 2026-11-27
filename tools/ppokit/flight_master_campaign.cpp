/**
 * @file tools/ppokit/flight_master_campaign.cpp
 * @brief Read-only historical campaign. Compile against the archived master FlightControl libraries.
 * @details FlightControlHeadless binds BEST evaluation to its promotion seed. This harness uses
 * the same FlightLearner, simulator, reset schedule and action mapping for an independent seed block.
 */
#include "flight_curriculum.hpp"
#include "flight_environment.hpp"
#include "flight_learner.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ppo::examples::flight;

int main(int argc, char** argv)
{
    if (argc != 4) return 2;
    try
    {
        constexpr std::uint32_t episodes = 20;
        auto const base = std::stoull(argv[3]);
        if (base > std::numeric_limits<std::uint32_t>::max() - (episodes - 1)) throw std::out_of_range("seed");
        auto const config = LoadConfig(argv[1]);
        FlightLearner learner(config);
        learner.Load(argv[2]);
        FlightEnvironment environment(config);
        std::cout << std::setprecision(17) << "seed,steps,reward,stop,touchdown,crash,stop_m,touchdown_m\n";
        for (std::uint32_t index = 0; index < episodes; ++index)
        {
            auto const seed = static_cast<std::uint32_t>(base + index);
            auto const start = EvaluationStart(index, episodes, config.training.curriculum.rolloutStartShare, 1.0);
            auto observation = environment.Reset(seed, start);
            double episodeReturn = 0;
            for (int step = 1; ; ++step)
            {
                if (step > 12000) throw std::runtime_error("episode exceeded its bound");
                auto const result = environment.Step(ToControls(learner.Choose(observation, true), config.initial));
                episodeReturn += result.reward;
                observation = environment.ObservationNow();
                if (result.terminated || result.truncated)
                {
                    auto const& mission = environment.Mission();
                    std::cout << seed << ',' << step << ',' << episodeReturn << ',' << (mission.outcome == Outcome::Stopped)
                              << ',' << mission.touchedDown << ',' << (mission.outcome == Outcome::Crash)
                              << ',' << mission.stopFromThreshold << ',' << mission.touchdownFromThreshold << '\n';
                    break;
                }
            }
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
