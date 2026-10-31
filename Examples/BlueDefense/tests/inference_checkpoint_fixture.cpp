/**
 * @file inference_checkpoint_fixture.cpp
 * @brief Test-only deterministic V8 actor fixture for desktop inference smoke.
 *
 * Never overwrites an existing promoted model. The generated policy is
 * deliberately untrained and must never be interpreted as a candidate.
 * CTest creates it only when no production checkpoint is present and cleans
 * up only if this fixture created it.
 */
#include "generalized_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace fs = std::filesystem;
namespace blue = ppo::examples::bluedefense::defenders;
namespace rlt = rl_tools;

int main(int argc, char** argv)
{
    if (argc != 4)
        return 2;
    std::string_view const action{argv[1]};
    fs::path const model{argv[2]};
    fs::path const marker{argv[3]};
    try
    {
        if (action == "cleanup")
        {
            if (fs::is_regular_file(marker))
            {
                fs::remove(model);
                fs::remove(marker);
            }
            return 0;
        }
        if (action != "prepare")
            return 2;

        // Respect real user/CI checkpoints and never mark them for removal.
        if (fs::exists(model))
        {
            fs::remove(marker);
            return fs::is_regular_file(model) ? 0 : 3;
        }
        fs::create_directories(model.parent_path());
        fs::create_directories(marker.parent_path());

        blue::Device device{};
        using Actor = blue::GeneralizedAttentionActor<
            blue::TrainingConfig::NN::ACTOR_SPEC>;
        Actor actor{};
        rlt::malloc(device, actor);
        try
        {
            auto rng = rlt::random::default_engine(
                typename blue::Device::SPEC::RANDOM{}, 918273U);
            rlt::init_weights(device, actor, rng);
            blue::SaveGeneralizedPolicy(model, actor, {4U, 4U});
        }
        catch (...)
        {
            rlt::free(device, actor);
            throw;
        }
        rlt::free(device, actor);
        std::ofstream stamp{marker};
        stamp << "CTest-only untrained actor fixture\n";
        if (!stamp)
            throw std::runtime_error("Failed to mark inference fixture");
        std::cout << "Generated temporary test actor: "
                  << model.string() << '\n';
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "Inference fixture: " << error.what() << '\n';
        return 1;
    }
}
