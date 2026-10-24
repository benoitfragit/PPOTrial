/**
* @file Main/src/main.cpp
* @brief Application entry point and scenario bootstrap.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "utils.hpp"
#include "environment.hpp"
#include "serializer.hpp"
#include "objective.hpp"
#include "Interception.hpp"
#include "targetselector.hpp"
#include "sam.hpp"

#include "environmentrenderer.hpp"
#include "metaregistry.hpp"
#include "scenarioconfig.hpp"

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
    return entityType == "Aircraft" || entityType == "SAM";
}

/**
* @brief Instantiates entities and agents from one scenario file.
*
* The loader creates each entity, stores its metadata, instantiates the agents
* declared in the scenario, and resolves target names once all entities exist.
*
* @param scenarioPath Path to the scenario JSON file.
* @param environment Environment that receives the instantiated content.
*/
void
LoadScenario(std::filesystem::path const& scenarioPath, ppo::model::Environment& environment)
{
    auto const scenario = ppo::config::LoadScenarioDefinition(scenarioPath);
    std::map<ppo::model::Entity*, std::string> targetMapping{};
    std::map<std::string, ppo::model::Entity*> entityMapping{};

    for (auto const& entityDef : scenario.m_entities)
    {
        if (RequiresPlatform(entityDef.m_type) && entityDef.m_platform.empty())
            continue;

        auto entity = ppo::reflection::CreateEntity(
            entityDef.m_type,
            environment,
            entityDef.m_platform,
            entityDef.m_side);

        if (entity == nullptr)
            continue;

        entity->add<ppo::model::sEntityMetadata>(ppo::model::sEntityMetadata{
            entityDef.m_name,
            entityDef.m_platform.empty() ? entityDef.m_type : entityDef.m_platform
        });

        for (auto const& agentName : entityDef.m_agents)
        {
            auto agent = ppo::reflection::CreateAgent(agentName, environment, *entity.get());
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
    }
}

} // namespace

/**
* @brief Runs the desktop application loop.
* @param argc Command-line argument count.
* @param argv Command-line argument vector.
* @return Process exit code.
*/
int
main(int argc, char** argv)
{
    ppo::reflection::RegisterAllMeta();

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    std::filesystem::path serializationPath{ RESOURCE_PATH };
    serializer.SetPath(serializationPath);

    ppo::renderer::EnvironmentRenderer& environmentRenderer{
        ppo::renderer::EnvironmentRenderer::GetInstance("PLAYGROUND")
    };

    ppo::model::Environment environment{};

    std::filesystem::path const scenarioPath =
        std::filesystem::path(RESOURCE_PATH) / "Scenarios" / "Playground.json";
    LoadScenario(scenarioPath, environment);

    serializer.Load();
    environmentRenderer.SetEnvironment(environment);

    // Drive the whole application from the renderer until the window closes.
    while (environmentRenderer.Run())
    {
    }

    serializer.Save();

    return 0;
}
