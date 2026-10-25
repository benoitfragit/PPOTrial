#include "patrol_training_agent.hpp"
#include "patrol_config.hpp"
#include "patrol_checkpoint.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace blue = ppo::examples::twovstwo::defenders;
namespace rlt = rl_tools;

namespace {
unsigned ParseUnsigned(char const* input)
{
    std::string const text{input};
    std::size_t consumed = 0;
    unsigned long const number = std::stoul(text, &consumed, 10);
    if (consumed != text.size() || number > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Expected unsigned integer argument");
    return static_cast<unsigned>(number);
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        // Optional seed and output path. No dependency on LibTorch or GPU.
        if (argc > 4)
        {
            std::cerr << "Usage: PPOBlueTrain [seed] [checkpoint.txt] [updates]\n";
            return 2;
        }
        unsigned const seed = argc >= 2 ? ParseUnsigned(argv[1]) : 1U;
        std::filesystem::path const output = argc >= 3
            ? std::filesystem::path{argv[2]}
            : std::filesystem::path{PPO_BLUE_MODEL_PATH};

        unsigned const updates = argc >= 4 ? ParseUnsigned(argv[3]) :
            static_cast<unsigned>(blue::TrainingParameters::STEP_LIMIT);
        if (updates == 0U)
            throw std::invalid_argument("PPO update count must be positive");

        blue::Parameters const params = blue::LoadPatrolParameters(
            std::filesystem::path{PPO_BLUE_SETTINGS_DIR});
        if (!blue::Valid(params))
            throw std::invalid_argument("Invalid patrol environment parameters");

        blue::PatrolTrainingAgent training{params, seed};
        while (training.StepCount() < updates)
        {
            if (training.Step())
                break;
            if (training.StepCount() % 10U == 0U)
                std::cout << "PPO update " << training.StepCount()
                          << " / " << updates << '\n';
        }
        training.WithActor([&](auto const& actor) { blue::SavePatrolPolicy(output, actor); });
        std::cout << "Saved actor: " << output.string() << '\n';
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Blue PPO training failed: " << e.what() << '\n';
        return 1;
    }
}
