/**
* @file Main/src/main.cpp
* @brief Desktop application entry point for registered examples.
*/
#include "example.hpp"
#include "examplecatalog.hpp"
#include "exampleselector.hpp"
#include "environment.hpp"
#include "environmentrenderer.hpp"
#include "serializer.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv)
{
    // Main only selects a compiled example; each example owns its startup.
    ppo::app::ExampleCatalog const examples = ppo::app::CreateExampleCatalog();

    bool checkExample = false;
    bool explicitExample = false;
    std::string requested{examples.Default().id};

    for (int i = 1; i < argc; ++i)
    {
        std::string_view const option{argv[i]};
        if (option == "--list-examples")
        {
            for (auto const& example : examples.Entries())
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
            explicitExample = true;
        }
    }

    // No model, scenario or simulation window is created before the choice.
    // Explicit --example and headless CI checks retain direct startup.
    if (ppo::app::ShouldShowExampleSelector(explicitExample, checkExample))
    {
        auto const chosen = ppo::app::SelectExample(examples);
        if (!chosen)
            return 0; // User closed the selector.
        requested = *chosen;
    }

    auto const* selected = examples.Find(requested);
    if (selected == nullptr || !ppo::app::IsValid(*selected))
    {
        std::cerr << "Unknown or invalid example: " << requested << '\n';
        return 2;
    }

    selected->registerTypes();

    auto& serializer = ppo::serialization::Serializer::GetInstance();
    std::filesystem::path const resourcePath{RESOURCE_PATH};
    serializer.SetPath(resourcePath);

    // Headless CI must not allocate graphics resources.
    ppo::renderer::EnvironmentRenderer* renderer = nullptr;
    if (!checkExample)
        renderer = &ppo::renderer::EnvironmentRenderer::GetInstance(
            std::string(selected->windowTitle));

    ppo::model::Environment environment{};
    bool const scenarioLoaded = selected->loadScenario(
        resourcePath / selected->defaultScenario, environment);

    serializer.Load();
    if (checkExample)
    {
        if (!scenarioLoaded)
        {
            std::cerr << "Failed to bootstrap example: " << selected->id << '\n';
            return 3;
        }
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
