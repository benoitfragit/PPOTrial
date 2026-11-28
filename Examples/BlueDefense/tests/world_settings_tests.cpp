/**
 * @file Examples/BlueDefense/tests/world_settings_tests.cpp
 * @brief The world of the environment DLL (LoadExternalScenario) steps with the time step and
 *        limits of the shared Environment.json.
 * @details Without them, the world keeps the defaults of sEnvironmentState (0.02 s instead of
 * 0.2 s): the mobiles would cover a tenth of their path per step.
 */
#include "airspacemeta.hpp"
#include "bluedefenseexample.hpp"
#include "components.hpp"
#include "environment.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

Json ReadJson(fs::path const& path)
{
    std::ifstream stream(path);
    return Json::parse(stream);
}

/** @brief The BluePatrol project's scenario in `folder`, with its agent profiles' absolute path. */
fs::path CopyScenario(fs::path const& folder)
{
    fs::path const original(PPO_BLUEDEFENSE_PROJECT);
    fs::create_directories(folder);
    Json scenario = ReadJson(original / "scenario.json");
    // The scenario names its agent profiles relative to its folder.
    scenario["agentProfiles"] = (fs::path(PPO_SHARED_SETTINGS) / "AgentProfiles.json").generic_string();
    std::ofstream(folder / "scenario.json") << scenario.dump(2) << '\n';
    fs::copy_file(original / "blue_patrol_training.json", folder / "blue_patrol_training.json",
        fs::copy_options::overwrite_existing);
    return folder / "scenario.json";
}

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path() / "blue_world_settings";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    try
    {
        ppo::examples::airspace::RegisterAirspaceMeta();
        ppo::model::Environment world;
        Check(ppo::examples::bluedefense::LoadExternalScenario(CopyScenario(root), world) != nullptr,
            "the environment DLL's loader reads the scenario");
        Json const settings = ReadJson(fs::path{PPO_SHARED_SETTINGS} / "Environment.json").at("environment");
        auto const& state = world.SteppedWorld().ctx().get<ppo::model::sEnvironmentState>();
        Check(state.m_dt == settings.at("timeStep").get<float>()
                && state.m_maxNorth == settings.at("maxNorth").get<float>()
                && state.m_maxEast == settings.at("maxEast").get<float>(),
            "the DLL's world steps with the time step and limits of Environment.json");
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " BlueDefense world settings check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "BlueDefense world settings tests passed\n";
    return EXIT_SUCCESS;
}
