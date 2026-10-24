/**
* @file Main/src/main.cpp
* @brief Desktop application entry point for registered examples.
*/
#include "example.hpp"
#include "existingexample.hpp"
#include "environment.hpp"
#include "environmentrenderer.hpp"
#include "serializer.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv)
{
    // The application composition root selects the compiled example.
    // No example-specific simulation, rendering or learning logic belongs here.
    std::array<ppo::app::ExampleDefinition, 1> const examples{
        ppo::examples::existing::Definition()
    };

    bool checkExample = false;
    std::string_view requested = examples.front().id;
    for (int i = 1; i < argc; ++i)
    {
        std::string_view const option{argv[i]};
        if (option == "--list-examples")
        {
            for (auto const& example : examples)
                std::cout << example.id << '\n';
            return 0;
        }
        if (option == "--check-example")
        {
            checkExample = true;
            continue;
        }
        if (option == "--example")
        {
            if (++i >= argc)
            {
                std::cerr << "Missing id after --example\n";
                return 2;
            }
            requested = argv[i];
        }
    }

    auto const selected = std::find_if(
        examples.begin(), examples.end(),
        [requested](ppo::app::ExampleDefinition const& example)
        {
            return example.id == requested;
        });

    if (selected == examples.end() || !ppo::app::IsValid(*selected))
    {
        std::cerr << "Unknown or invalid example: " << requested << '\n';
        return 2;
    }

    // Preserve the original initialization sequence and the default scenario.
    selected->registerTypes();

    auto& serializer = ppo::serialization::Serializer::GetInstance();
    std::filesystem::path const resourcePath{RESOURCE_PATH};
    serializer.SetPath(resourcePath);

    // Headless CI startup exercises the exact same example factories and
    // config/model loading without allocating graphics resources.
    ppo::renderer::EnvironmentRenderer* renderer = nullptr;
    if (!checkExample)
        renderer = &ppo::renderer::EnvironmentRenderer::GetInstance(
            std::string(selected->windowTitle));

    ppo::model::Environment environment{};
    selected->loadScenario(resourcePath / selected->defaultScenario, environment);

    serializer.Load();
    if (checkExample)
    {
        std::cout << "Bootstrapped example: " << selected->id << '\n';
        return 0;
    }

    renderer->SetEnvironment(environment);
    while (renderer->Run())
    {
    }
    serializer.Save();
    return 0;
}
