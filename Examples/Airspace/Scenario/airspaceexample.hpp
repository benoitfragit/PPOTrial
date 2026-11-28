/**
 * @file Examples/Airspace/Scenario/airspaceexample.hpp
 * @brief Loads an Airspace scenario for the environment DLL (rlmission_environment): its
 *        entities, agents and world rules, with its training agent driven from outside.
 */
#pragma once

#include "external_training.hpp"

#include "entt/entt.hpp"

#include <filesystem>
#include <memory>

namespace ppo::model {
class Environment;
}

namespace ppo::examples::airspace {

/** @brief The training agent of an external scenario and the entity it controls. */
struct ExternalScenario final
{
    std::shared_ptr<ppo::model::IExternalTrainingAgent> agent;
    entt::entity entity{ entt::null };
};

/**
 * @brief Loads an Airspace scenario whose one training agent is driven from outside the
 *        world (ppokit-run's environment DLL).
 * @details The training agent reads its settings (Training/<agent>.json: reward weights,
 * clamp, rollout size); the world's Environment.json is applied.
 * @param scenarioPath The scenario file.
 * @param environment The world that receives the scenario.
 * @return The training agent and its entity.
 * @throws std::runtime_error When the scenario does not load, or holds no training agent or
 *         more than one.
 */
ExternalScenario LoadExternalScenario(
    std::filesystem::path const& scenarioPath, ppo::model::Environment& environment);

} // namespace ppo::examples::airspace
