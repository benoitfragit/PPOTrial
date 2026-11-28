/**
 * @file Examples/Airspace/Scenario/airspaceexample.cpp
 * @brief Construction, données et arbitrage du scénario airspaceexample.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "airspaceexample.hpp"
#include "airspaceworldrules.hpp"
#include "randominitializersystem.hpp"

#include "components.hpp"
#include "airspacecomponents.hpp"
#include "entity.hpp"
#include "environment.hpp"
#include "metaregistry.hpp"
#include "scenarioconfig.hpp"
#include "serializer.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppo::examples::airspace {
namespace {


/**
* @brief Tells whether one scenario entity type requires a platform file.
* @param entityType Scenario type string.
* @return True when the type needs a platform JSON file.
*/
bool
RequiresPlatform(std::string const& entityType) noexcept
{
    // Preserve the historical platform-loading behavior.
    return entityType == "Mobile" || entityType == "RestrictedZone"
        || entityType == "Aircraft" || entityType == "SAM";
}

/** @brief An agent of the scenario and the entity it controls. */
struct ScenarioAgent final
{
    std::shared_ptr<ppo::model::IAgent> agent;
    entt::entity entity{ entt::null };
};

/**
* @brief Instantiates entities and agents from one scenario file.
*
* The loader creates each entity, stores its metadata, instantiates the agents
* declared in the scenario, and resolves target names once all entities exist.
*
* @param scenarioPath Path to the scenario JSON file.
* @param environment Environment that receives the instantiated content.
* @return True when all declared entities and agents were created.
*/
bool
LoadScenarioAgents(std::filesystem::path const& scenarioPath, ppo::model::Environment& environment,
                   std::vector<ScenarioAgent>& agents)
{
    if (!environment.SetEntityInitializer(&ppo::model::RandomInitiliazerSystem::Init)
        || !environment.SetWorldRules(std::make_unique<AirspaceWorldRules>()))
        return false;

    auto const scenario = ppo::config::LoadScenarioDefinition(scenarioPath);
    bool complete = !scenario.m_entities.empty();
    std::map<ppo::model::Entity*, std::string> targetMapping{};
    std::map<std::string, ppo::model::Entity*> entityMapping{};

    for (auto const& entityDef : scenario.m_entities)
    {
        if (RequiresPlatform(entityDef.m_type) && entityDef.m_platform.empty())
        {
            complete = false;
            continue;
        }

        auto entity = ppo::reflection::CreateEntity(
            entityDef.m_type,
            environment,
            entityDef.m_platform,
            entityDef.m_side);

        if (entity == nullptr)
        {
            complete = false;
            continue;
        }

        entity->add<ppo::model::sEntityMetadata>(ppo::model::sEntityMetadata{
            entityDef.m_name,
            entityDef.m_platform.empty() ? entityDef.m_type : entityDef.m_platform
        });

        for (auto const& agentName : entityDef.m_agents)
        {
            std::shared_ptr<ppo::model::IAgent> agent =
                ppo::reflection::CreateAgent(agentName, environment, *entity.get());
            if (agent == nullptr)
                complete = false;
            else
                agents.push_back({ std::move(agent), entity->uniqid() });
        }

        if (!entityDef.m_target.empty())
            targetMapping[entity.get()] = entityDef.m_target;

        entityMapping[entityDef.m_name] = entity.get();
    }

    for (auto const& [entity, target] : targetMapping)
    {
        if (entity == nullptr
            || !entity->has<ppo::model::sSNAState>())
        {
            continue;
        }

        if (auto iter = entityMapping.find(target);
            iter != entityMapping.end()
            && iter->second != nullptr)
        {
            auto [snaState] = entity->get<ppo::model::sSNAState>();
            snaState.m_target = iter->second->uniqid();
        }
        else
        {
            complete = false;
        }
    }

    return complete;
}

} // namespace

ExternalScenario LoadExternalScenario(
    std::filesystem::path const& scenarioPath, ppo::model::Environment& environment)
{
    // The agents register their settings files against these folders when they are created.
    auto& serializer = ppo::serialization::Serializer::GetInstance();
    serializer.SetSearchPaths({std::filesystem::path{PPO_AIRSPACE_SETTINGS},
                               std::filesystem::path{PPO_SHARED_SETTINGS}});
    std::vector<ScenarioAgent> agents;
    if (!LoadScenarioAgents(scenarioPath, environment, agents))
    {
        throw std::runtime_error(scenarioPath.string() + ": not a valid Airspace scenario (see the messages above)");
    }
    ExternalScenario trained{};
    for (ScenarioAgent const& candidate : agents)
    {
        auto external = std::dynamic_pointer_cast<ppo::model::IExternalTrainingAgent>(candidate.agent);
        if (external == nullptr)
            continue;
        if (trained.agent != nullptr)
            throw std::runtime_error(scenarioPath.string() + ": more than one agent in training mode");
        trained = { std::move(external), candidate.entity };
    }
    if (trained.agent == nullptr)
        throw std::runtime_error(scenarioPath.string() + ": no agent in training mode");
    // The training agent's settings (rollout size, reward clamp and weights).
    serializer.Load();
    environment.LoadSettings(std::filesystem::path{PPO_SHARED_SETTINGS} / "Environment.json");
    return trained;
}

} // namespace ppo::examples::airspace
