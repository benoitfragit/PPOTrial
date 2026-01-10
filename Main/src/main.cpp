#include <SFML/Graphics.hpp>

#include "utils.hpp"
#include "environment.hpp"
#include "serializer.hpp"
#include "Interception.hpp"

#include "environmentrenderer.hpp"

int main()
{
    ppo::serialization::Serializer& serializer{ppo::serialization::Serializer::GetInstance()};
    std::filesystem::path serializationPath{RESOURCE_TOML_PATH};
    serializer.SetPath(serializationPath);

    ppo::model::Environment environment{};

    auto [blue, blueAgent] = ppo::interception::GetNewTrainingMissile("blue",  ppo::model::ePlayerSide::Friend,  environment);
    auto [red, redAgent]   = ppo::interception::GetNewProductionMissile("red", ppo::model::ePlayerSide::Hostile, environment);

    // create a first player instance
    blue->SetTarget(red->GetPlayerId());
    red->SetTarget(blue->GetPlayerId());

    serializer.Load();

    // Get the environment renderer instance
    ppo::environment::EnvironmentRenderer& environmentRenderer { ppo::environment::EnvironmentRenderer::GetInstance( 1280, 960, "PLAYGROUND") };
    
    // attach the renderer to the environment
    environmentRenderer.SetEnvironment(environment);
    
    // Append renderers for each player
    environmentRenderer.Display(*blue.get(), blueAgent.get());
    environmentRenderer.Display(*red.get());

    // Main loop
    while (environmentRenderer.Run())
    {

    }

    serializer.Save();

    return 0;
}