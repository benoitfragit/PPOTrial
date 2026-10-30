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
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <iostream>
#include <optional>
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

        // Append-only, flushed, read-only sampling of existing ECS snapshots.
        // This monitor never calls the learner or its optimizer.
        auto const telemetryPath = resourcePath.parent_path() / "model"
            / "BluePatrol" / "telemetry" / "mappo-training.csv";
        std::ofstream telemetry;
        try
        {
            std::filesystem::create_directories(telemetryPath.parent_path());
            telemetry.open(telemetryPath, std::ios::trunc);
            if (telemetry)
            {
                telemetry << "elapsed_seconds,updates,rollout_steps,rollout_target,"
                             "evaluating,evaluation_progress,evaluation_cases_done,"
                             "evaluation_cases_total,evaluation_steps,"
                             "evaluation_steps_per_second,training,"
                             "eval_completed,latest_score,best_score,"
                             "eval_success_rate,instant_reward\n";
                telemetry.flush();
            }
            else
                std::cerr << "[MAPPO_STATUS] Cannot open telemetry CSV: "
                          << telemetryPath.string() << '\n';
        }
        catch (std::filesystem::filesystem_error const& error)
        {
            std::cerr << "[MAPPO_STATUS] Cannot create telemetry directory: "
                      << error.what() << '\n';
        }

        using Clock = std::chrono::steady_clock;
        auto const started = Clock::now();
        auto nextSample = started;
        auto nextHeartbeat = started;
        std::uint64_t lastUpdates = 0U;
        bool lastEvaluating = false;
        bool haveLastState = false;
        std::cout << "[HEADLESS] Running " << requested
                  << " (budget " << maxSeconds << " seconds)\n";
        environment.Play();
        while (Clock::now() < deadline)
        {
            auto const sampledAt = Clock::now();
            if (sampledAt >= nextSample)
            {
                auto const elapsedSeconds = std::chrono::duration<double>(
                    sampledAt - started).count();
                bool const heartbeat = sampledAt >= nextHeartbeat;
                // Hold the ECS snapshot mutex only long enough to copy
                // scalars; disk I/O and console output happen after unlock.
                std::optional<ppo::model::sCommonAgentState> stateCopy;
                environment.ReadSnapshot([&](entt::registry const& snapshot)
                {
                    snapshot.view<ppo::model::sAgentStateSnaphot>().each(
                        [&](ppo::model::sAgentStateSnaphot const& state)
                        {
                            if (state.m_name == "Blue generalized shared MAPPO")
                                stateCopy = state.m_common;
                        });
                });
                if (stateCopy)
                {
                    observedController = true;
                    auto const& common = *stateCopy;
                    if (!common.m_training && !common.m_evaluationinprogress)
                        completed = true;
                    auto const updates =
                        static_cast<std::uint64_t>(common.m_version);
                    if (telemetry)
                    {
                        telemetry << std::fixed << std::setprecision(3)
                                  << elapsedSeconds << ',' << updates << ','
                                  << common.m_step << ',' << common.m_maxStep << ','
                                  << (common.m_evaluationinprogress ? 1 : 0) << ','
                                  << common.m_evaluationprogress << ','
                                  << common.m_evaluationTestsFinished << ','
                                  << common.m_latestEvaluationTestCount << ','
                                  << common.m_evaluationStepsExecuted << ','
                                  << common.m_evaluationStepsPerSecond << ','
                                  << (common.m_training ? 1 : 0) << ','
                                  << (common.m_hasLatestEvaluation ? 1 : 0) << ','
                                  << common.m_latestEvaluationNotation << ','
                                  << common.m_notation << ','
                                  << common.m_latestEvaluationSuccessRate << ','
                                  << common.m_reward << '\n';
                        telemetry.flush();
                    }
                    if (heartbeat || !haveLastState
                        || updates != lastUpdates
                        || common.m_evaluationinprogress != lastEvaluating)
                    {
                        std::clog << "[MAPPO_STATUS] elapsed_s="
                                  << static_cast<std::uint64_t>(elapsedSeconds)
                                  << " updates=" << updates
                                  << " rollout=" << common.m_step << '/'
                                  << common.m_maxStep
                                  << " evaluation="
                                  << (common.m_evaluationinprogress ? 1 : 0)
                                  << " eval_progress="
                                  << std::fixed << std::setprecision(3)
                                  << common.m_evaluationprogress
                                  << " eval_cases="
                                  << common.m_evaluationTestsFinished << '/'
                                  << common.m_latestEvaluationTestCount
                                  << " eval_steps=" << common.m_evaluationStepsExecuted
                                  << std::endl;
                    }
                    lastUpdates = updates;
                    lastEvaluating = common.m_evaluationinprogress;
                    haveLastState = true;
                }
                nextSample = sampledAt + std::chrono::seconds(2);
                if (heartbeat)
                    nextHeartbeat = sampledAt + std::chrono::seconds(15);
            }
            if (completed)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        environment.Stop();
        if (telemetry)
            telemetry.flush();
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
