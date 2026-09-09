#include "application.hpp"
#include "environmentrenderer.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    try
    {
        const std::filesystem::path resources = std::filesystem::path(GetApplicationDirectory()) / "scenarios";
        const auto scenario = navigation::load_scenario(argc > 1 ? std::filesystem::path(argv[1]) : resources / "default.nav");
        const std::filesystem::path checkpoint_root = argc > 2 ? argv[2] : "checkpoints/navigation2d";
        navigation::TorchBackend backend;
        navigation::Application application(scenario, resources / "score.metrics", checkpoint_root, backend);
        ppo::renderer::EnvironmentRenderer renderer(application);
        while (renderer.Run()) {}
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "PPOTrial: " << error.what() << '\n';
        return 1;
    }
}
