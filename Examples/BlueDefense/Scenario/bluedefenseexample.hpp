/**
 * @file Examples/BlueDefense/Scenario/bluedefenseexample.hpp
 * @brief Loads a BlueDefense scenario for the environment DLL (blue_environment): its
 *        entities, red agents, world rules and the blue team's controller.
 */
#pragma once

#include <filesystem>
#include <memory>

namespace ppo::model {
class Environment;
}

namespace ppo::examples::bluedefense {
namespace defenders {
class GeneralizedPatrolMappoAgent;
}
/**
 * @brief Loads a BlueDefense scenario whose blue team is driven from outside the world:
 *        the environment DLL of ppokit-run.
 * @param path BlueDefense scenario document (scenario.json of a BluePatrol project).
 * @param environment Destination ECS environment, stepped by its caller.
 * @return The blue team's controller, or null when the scenario is invalid.
 * @details Register the Airspace reflection (RegisterAirspaceMeta) first. The world then
 *          has the time step and limits of the shared Environment.json.
 */
std::shared_ptr<defenders::GeneralizedPatrolMappoAgent> LoadExternalScenario(
    std::filesystem::path const& path, ppo::model::Environment& environment);
}
