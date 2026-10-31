/**
 * @file scenario_config_tests.cpp
 * @brief Verify the single V2 scenario/catalog/cases contract.
 */
#include "scenarioconfig.hpp"
#include "serializer.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ppo::config::ScenarioDefinition;
using Json = ppo::serialization::json;

namespace {
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
void Require(bool ok, std::string const& what)
{
    if (!ok)
        throw std::runtime_error(what);
}

void CheckScenario(fs::path const& path, std::size_t entityCount,
                   std::string const& agentName, bool evaluation, int steps)
{
    auto const scenario = ppo::config::LoadScenarioDefinition(path);
    Require(scenario.m_entities.size() == entityCount,
            path.string() + ": entity count changed");
    Require(scenario.m_evaluation.m_enabled == evaluation,
            path.string() + ": evaluation enabled flag changed");

    bool blueFound = false;
    for (auto const& entity : scenario.m_entities)
    {
        if (entity.m_name != "blue")
            continue;
        blueFound = true;
        Require(entity.m_side == ppo::model::eEntitySide::Friend,
                "blue side changed");
        Require(entity.m_agents.size() == 1 && entity.m_agents.front() == agentName,
                "resolved agent changed");
    }
    Require(blueFound, "blue entity disappeared");

    if (!evaluation)
        return;

    Require(scenario.m_evaluation.m_evaluatedEntity == "blue",
            "evaluated entity changed");
    Require(scenario.m_evaluation.m_nbTests == 100
        && scenario.m_evaluation.m_nbSteps == steps
        && scenario.m_evaluation.m_timeStep == 0.2F,
        "evaluation budgets changed");
    Require(fs::is_regular_file(scenario.m_evaluation.m_casesFile),
            "evaluation case file missing");
    auto const cases = ppo::config::LoadEvaluationBufferDefinition(
        scenario.m_evaluation.m_casesFile);
    Require(cases.m_tests.size() == 100, "evaluation case count changed");
    for (auto const& test : cases.m_tests)
        Require(test.m_entities.size() == entityCount,
                "evaluation case entity count changed");
}

/**
 * @brief Sérialise ou écrit les données associées à WriteJson selon les garanties implémentées.
 * @details Utilise des fichiers ou chemins.
 */
void WriteJson(fs::path const& path, Json const& doc)
{
    std::ofstream out(path);
    out << doc.dump(2);
}


void CheckAdditionalAssets(fs::path const& root)
{
    auto const world = ppo::serialization::Serializer::Parse(root / "Environment.json");
    Require(world.at("formatVersion") == 2, "world formatVersion changed");
    auto const& environment = world.at("environment");
    Require(environment.at("maxNorth").get<float>() == 44658.75F
        && environment.at("maxEast").get<float>() == 71454.F
        && environment.at("timeStep").get<double>() == 0.20000000298023224,
        "environment state changed during migration");

    auto const display = ppo::serialization::Serializer::Parse(root / "Renderer.json");
    Require(display.at("formatVersion") == 2, "renderer formatVersion changed");
    auto const& renderer = display.at("renderer");
    Require(renderer.at("scaleLevel").get<int>() == 1
        && renderer.at("autoAttachCamera").get<bool>()
        && renderer.at("multipleCameras").get<bool>()
        && !renderer.at("showGrid").get<bool>()
        && renderer.at("showPseudoLidar").get<bool>()
        && renderer.at("camera").at("fovy").get<float>() == 45.F,
        "renderer settings changed during migration");

    // The V2 file must retain the original camera data and current workspace
    // panel visibility, not just a small subset of renderer preferences.
    auto const& camera = renderer.at("camera");
    Require(camera.at("position").size() == 3
        && camera.at("position").at(0).get<double>() == 36.58390808105469
        && camera.at("position").at(1).get<double>() == 24.686992645263672
        && camera.at("position").at(2).get<double>() == 23.742538452148438
        && camera.at("target").at(0).get<double>() == 7.42773962020874
        && camera.at("target").at(1).get<double>() == 0.34999987483024597
        && camera.at("target").at(2).get<double>() == -12.805513381958008
        && renderer.at("cameraDistance").get<double>() == 45.35300064086914
        && renderer.at("cameraHeight").get<double>() == 24.687000274658203
        && renderer.at("domainHeight").get<double>() == 2.318000078201294
        && renderer.at("trailDuration").get<double>() == 3.0999999046325684,
        "renderer V2 lost original camera precision");
    Require(!renderer.at("showMinimap").get<bool>()
        && !renderer.at("showEntities").get<bool>()
        && !renderer.at("showInspection").get<bool>()
        && renderer.at("showSimulation").get<bool>(),
        "renderer V2 workspace panel defaults changed");

    auto const profiles = ppo::serialization::Serializer::Parse(root / "AgentProfiles.json");
    Require(profiles.at("formatVersion").get<int>() == 2, "wrong agent profile format");
    auto const& mapping = profiles.at("profiles");
    Require(mapping.at("RLMission").at("training") == "TargetSelectorTrainingAgent"
        && mapping.at("RLMission").at("production") == "TargetSelectorProdAgent"
        && mapping.at("RLSNA").at("training") == "SNATrainingAgent"
        && mapping.at("RLSNA").at("production") == "SNAProdAgent",
        "agent profile bindings changed");

    for (auto const& file : {
            "Platforms/Mobile/aircraft.json",
            "Platforms/Mobile/evalaircraft.json",
            "Platforms/Mobile/trainingaircraft.json",
            "Platforms/RestrictedZone/sam.json",
            "Platforms/Mobile/mobile.json",
            "Platforms/RestrictedZone/restrictedzone.json"})
    {
        auto const profile = ppo::serialization::Serializer::Parse(root / file);
        Require(profile.at("formatVersion").get<int>() == 2
            && profile.at("parameters").is_object(),
            std::string(file) + ": invalid platform format");
        auto const& parameters = profile.at("parameters");
        std::string const filename = fs::path(file).filename().string();
        if (filename == "aircraft.json" || filename == "evalaircraft.json"
            || filename == "mobile.json")
        {
            Require(parameters.at("FlightModel").at("MaxRotAcceleration").get<double>()
                    == 0.4000000059604645
                && parameters.at("SNA").at("MaxShootAngle").get<double>()
                    == 1.0471980571746826,
                filename + ": V2 platform numerical precision changed");
        }
        else if (filename == "trainingaircraft.json")
        {
            Require(parameters.at("FlightModel").at("MaxRotAcceleration").get<double>()
                    == 0.70000059604645
                && parameters.at("SNA").at("MaxShootAngle").get<double>()
                    == 3.1471980571746827,
                filename + ": V2 training platform numerical precision changed");
        }
        else if (filename == "sam.json" || filename == "restrictedzone.json")
        {
            Require(parameters.at("SNA").at("MaxShootAngle").get<double>()
                    == 3.14159178,
                filename + ": V2 ground platform numerical precision changed");
        }
    }

    auto const presets = ppo::serialization::Serializer::Parse(
        root / "Evaluation" / "InitialStates.json");
    Require(presets.at("formatVersion").get<int>() == 2, "invalid initial-state format");
    for (auto const& name : {"blueeval", "redeval"})
    {
        auto const& samples = presets.at("presets").at(name);
        Require(samples.is_array() && samples.size() == 100,
                std::string(name) + ": initial-state count changed");
        for (auto const& sample : samples)
        {
            Require(sample.is_object() && sample.size() == 6,
                    std::string(name) + ": incomplete initial-state sample");
            for (auto const& field : {"north", "east", "heading",
                                      "speed", "turnAcceleration", "acceleration"})
                Require(std::isfinite(sample.at(field).get<float>()),
                        std::string(name) + ": nonfinite initial-state sample");
        }
    }
    auto const& first = presets.at("presets").at("blueeval").at(0);
    Require(first.at("north").get<float>() == 14905.041015625F
        && first.at("east").get<float>() == -9068.5419921875F
        && first.at("speed").get<float>() == 355.4700012207031F,
        "initial-state golden sample changed");

    auto const& settings = presets.at("evaluationSettings");
    Require(settings.at("tests").get<int>() == 100
        && settings.at("steps").get<int>() == 512
        && settings.at("timeStep").get<float>() == 0.2F,
        "initial-state evaluation settings changed");
}
} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    fs::path temporary{};
    try
    {
        fs::path const root{PPO_SETTINGS_ROOT};
        CheckAdditionalAssets(root);
        CheckScenario(root / "Scenarios" / "Playground.json",
                      7, "TargetSelectorTrainingAgent", false, 0);
        CheckScenario(root / "Scenarios" / "Evaluation" / "RLMission.json",
                      7, "TargetSelectorProdAgent", true, 1024);
        CheckScenario(root / "Scenarios" / "Evaluation" / "RLSNA.json",
                      2, "SNAProdAgent", true, 512);

        // Reject invalid format rather than guessing at an alternate schema.
        temporary = fs::temp_directory_path() /
            ("ppo_scenario_v2_test_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(temporary);
        Json original = ppo::serialization::Serializer::Parse(
            root / "Scenarios" / "Playground.json");

        Json invalidVersion = original;
        invalidVersion["formatVersion"] = 1;
        WriteJson(temporary / "invalid_version.json", invalidVersion);
        Require(ppo::config::LoadScenarioDefinition(temporary / "invalid_version.json")
                    .m_entities.empty(),
                "unsupported scenario format was accepted");

        Json badAgent = original;
        badAgent["agentProfiles"] = (root / "AgentProfiles.json").string();
        badAgent["entities"]["blue"]["agents"][0]["profile"] = "nonexistent";
        WriteJson(temporary / "invalid_agent.json", badAgent);
        Require(ppo::config::LoadScenarioDefinition(temporary / "invalid_agent.json")
                    .m_entities.empty(),
                "unregistered profile was accepted");

        std::error_code ignored;
        fs::remove_all(temporary, ignored);
        std::cout << "PASS: V2 scenarios, logical agents, evaluations and 100-case suites\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::error_code ignored;
        if (!temporary.empty()) fs::remove_all(temporary, ignored);
        std::cerr << "Scenario configuration regression: " << error.what() << '\n';
        return 1;
    }
}
