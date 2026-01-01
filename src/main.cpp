#include <SFML/Graphics.hpp>

#include "player.hpp"
#include "agent.hpp"
#include "utils.hpp"
#include "environment.hpp"

#include "environmentrenderer.hpp"

int main()
{
    ppo::model::Environment environment{};

    // create a first player instance
    ppo::model::Player red{ "Rafale", 1};
    red.LoadFromFile(RESOURCE_TOML_PATH);
    red.SetTarget(2);

    // crate a new player instance
    ppo::model::Player blue{ "F35", 2 };
    blue.LoadFromFile(RESOURCE_TOML_PATH);
    blue.SetTarget(1);

    // Create an agent to control the red player
    using ppoAgent = ppo::model::Agent<ppo::model::sEnvironmentState, ppo::model::Player, ppo::model::sAgentState>;
    ppoAgent redAgent{"FirstGeneration"};

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

    red.SaveToFile(RESOURCE_TOML_PATH);
    blue.SaveToFile(RESOURCE_TOML_PATH);

    return 0;
}