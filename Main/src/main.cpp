#include "application.hpp"
#include "environmentrenderer.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    try
    {
        const std::filesystem::path resources = NAVIGATION_RESOURCE_PATH;
        const auto scenario = navigation::load_scenario(argc > 1 ? std::filesystem::path(argv[1]) : resources / "default.nav");
        navigation::Application application(scenario, resources / "score.metrics", "checkpoints/navigation2d");
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
