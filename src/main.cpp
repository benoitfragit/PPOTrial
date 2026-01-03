#include <SFML/Graphics.hpp>

#include "player.hpp"
#include "agent.hpp"
#include "utils.hpp"
#include "environment.hpp"
#include "serializer.hpp"

#include "environmentrenderer.hpp"

int main()
{
    ppo::serialization::Serializer& serializer{ppo::serialization::Serializer::GetInstance()};
    std::filesystem::path serializationPath{RESOURCE_TOML_PATH};
    serializer.SetPath(serializationPath);

    ppo::model::Environment environment{};

    // create a first player instance
    ppo::model::Player red{ "Rafale", 1};
    red.SetTarget(2);

    // crate a new player instance
    ppo::model::Player blue{ "F35", 2 };
    blue.SetTarget(1);

    // Create an agent to control the red player
    using ppoAgent = ppo::model::Agent<ppo::model::sEnvironmentState, ppo::model::Player, ppo::model::sAgentState>;
    ppoAgent redAgent{"FirstGeneration", RESOURCE_PPO_PATH };
    
    serializer.Load();

    // Append this agent into the environment
    environment.AddAgent(&red, &redAgent);

    // Append the blue player as a non trainable agent
    environment.AddAgent(&blue);

    // Get the environment renderer instance
    ppo::environment::EnvironmentRenderer& environmentRenderer { ppo::environment::EnvironmentRenderer::GetInstance( 1280, 960, "PLAYGROUND") };
    
    // attach the renderer to the environment
    environmentRenderer.SetEnvironment(environment);
    
    // Append renderers for each player
    environmentRenderer.Display(red , sf::Color::Red, &redAgent);
    environmentRenderer.Display(blue, sf::Color::Blue);

    // Main loop
    while (environmentRenderer.Run())
    {

    }

    serializer.Save();

    return 0;
}