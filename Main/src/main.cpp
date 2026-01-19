#include <SFML/Graphics.hpp>

#include "utils.hpp"
#include "environment.hpp"
#include "serializer.hpp"
#include "objective.hpp"
#include "Interception.hpp"

#include "environmentrenderer.hpp"

int main()
{
    ppo::serialization::Serializer& serializer{ppo::serialization::Serializer::GetInstance()};
    std::filesystem::path serializationPath{RESOURCE_TOML_PATH};
    serializer.SetPath(serializationPath);

    ppo::model::Environment environment{};

    std::shared_ptr<ppo::model::Objective> objective{ std::make_shared<ppo::model::Objective>("red", ppo::model::ePlayerSide::Hostile, environment) };
    environment.AddAgent(objective, nullptr);

    auto [blue1, blueAgent1] = ppo::interception::GetNewTrainingMissile("blue",  ppo::model::ePlayerSide::Friend,  environment);

    auto [red1, redAgent1]   = ppo::interception::GetNewProductionMissile("red", ppo::model::ePlayerSide::Hostile, environment);

    // create a first player instance
    blue1->SetTarget(red1->GetPlayerId());
    red1->SetTarget(blue1->GetPlayerId());

    serializer.Load();

    // Get the environment renderer instance
    ppo::environment::EnvironmentRenderer& environmentRenderer { ppo::environment::EnvironmentRenderer::GetInstance( 1280, 960, "PLAYGROUND") };
    
    // attach the renderer to the environment
    environmentRenderer.SetEnvironment(environment);
    
    // Append renderers for each player
    environmentRenderer.Display(*blue1.get(), blueAgent1.get());
    environmentRenderer.Display(*red1.get());
    environmentRenderer.Display(*objective.get());

    // Main loop
    while (environmentRenderer.Run())
    {

    }

    serializer.Save();

    return 0;
}