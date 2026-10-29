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

#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

int main(int argc, char** argv)
{
    // Main only selects a compiled example; each example owns its startup.
    ppo::app::ExampleCatalog const examples = ppo::app::CreateExampleCatalog();

    bool checkExample = false;
    bool explicitExample = false;
    bool noGui = false;
    bool durationSpecified = false;
    unsigned int maxSeconds = 14400U;
    std::string requested{examples.Default().id};

    for (int i = 1; i < argc; ++i)
    {
        std::string_view const option{argv[i]};
        if (option == "--no-gui")
        {
            noGui = true;
            continue;
        }
        if (option == "--max-seconds")
        {
            if (++i >= argc)
            {
                std::cerr << "Missing value after --max-seconds\n";
                return 2;
            }
            std::string_view const value{argv[i]};
            unsigned int seconds{};
            auto const parsed = std::from_chars(
                value.data(), value.data() + value.size(), seconds);
            if (parsed.ec != std::errc{} ||
                parsed.ptr != value.data() + value.size() ||
                seconds == 0U || seconds > 86400U)
            {
                std::cerr << "--max-seconds expects an integer in [1, 86400]\n";
                return 2;
            }
            maxSeconds = seconds;
            durationSpecified = true;
            continue;
        }
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

    if ((durationSpecified && !noGui)
        || (noGui && (checkExample || !explicitExample
            || (requested != "blue-patrol" && requested != "blue-defense"))))
    {
        std::cerr << "--no-gui requires --example blue-patrol|blue-defense; "
                     "--max-seconds requires --no-gui\n";
        return 2;
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
    if (!checkExample && !noGui)
        renderer = &ppo::renderer::EnvironmentRenderer::GetInstance(
            std::string(selected->windowTitle));

    ppo::model::Environment environment{};
    bool const scenarioLoaded = selected->loadScenario(
        resourcePath / selected->defaultScenario, environment);

    if (noGui && !scenarioLoaded)
    {
        std::cerr << "Failed to load training scenario: " << requested << '\n';
        return 3;
    }
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

    if (noGui)
    {
        // Run the same ECS physics and production red policies, without
        // rendering or throttling the simulation to human viewing speed.
        environment.SetHeadlessMode(true);
        auto const deadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(maxSeconds);
        bool observedController = false;
        bool completed = false;
        std::cout << "[HEADLESS] Running " << requested
                  << " (budget " << maxSeconds << " seconds)\n";
        environment.Play();
        while (std::chrono::steady_clock::now() < deadline)
        {
            environment.ReadSnapshot([&](entt::registry const& snapshot)
            {
                snapshot.view<ppo::model::sAgentStateSnaphot>().each(
                    [&](ppo::model::sAgentStateSnaphot const& state)
                    {
                        if (state.m_name != "Blue generalized shared MAPPO")
                            return;
                        observedController = true;
                        if (!state.m_common.m_training
                            && !state.m_common.m_evaluationinprogress)
                            completed = true;
                    });
            });
            if (completed)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        environment.Stop();
        // No configuration file changes are written in batch mode.
        // The MAPPO controller owns all training checkpoint writes.
        if (!observedController)
        {
            std::cerr << "[HEADLESS] Missing MAPPO controller snapshot\n";
            return 4;
        }
        auto const latest = resourcePath.parent_path() / "model"
            / "BluePatrol" / "latest"
            / "blue_patrol_generalized_training_v5.txt";
        if (!std::filesystem::is_regular_file(latest))
        {
            std::cerr << "[HEADLESS] No resumable training checkpoint\n";
            return 5;
        }
        std::cout << (completed ? "[HEADLESS] Training completed\n"
                                : "[HEADLESS] Time budget reached; saved checkpoint available\n");
        return 0;
    }

    renderer->SetEnvironment(environment);
    while (renderer->Run())
    {
    }
    serializer.Save();
    return 0;
}
