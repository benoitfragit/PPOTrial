#include "utils.hpp"
#include "environment.hpp"
#include "serializer.hpp"
#include "objective.hpp"
#include "Interception.hpp"
#include "targetselector.hpp"
#include "sam.hpp"

#include "environmentrenderer.hpp"
#include "metaregistry.hpp"

namespace {

ppo::model::eEntitySide
ParseEntitySide(std::string const& name) noexcept
{
    if (name == "Hostile")
        return ppo::model::eEntitySide::Hostile;

    if (name == "Friend")
        return ppo::model::eEntitySide::Friend;

    return ppo::model::eEntitySide::Unknown;
}

bool
RequiresPlatform(std::string const& entityType) noexcept
{
    // Preserve the historical behavior:
    // - Aircraft and SAM require a platform file
    // - Objective does not
    return entityType == "Aircraft" || entityType == "SAM";
}

void
LoadScenario(std::filesystem::path const& scenarioPath, ppo::model::Environment& environment)
{
    if (!std::filesystem::exists(scenarioPath))
        return;

    std::map<ppo::model::Entity*, std::string>  targetMapping{};
    std::map<std::string, ppo::model::Entity*>  entityMapping{};

    auto ar = ppo::serialization::Serializer::Parse(scenarioPath);

    auto scenarioSection = ar["Scenario"];
    auto entitiesList = scenarioSection["entities"];

    for (auto [name, entitySection] : entitiesList.items())
    {
        std::string const entityType{ entitySection.value("type", "") };
        std::string const sideName{ entitySection.value("side", "") };
        std::string const targetName{ entitySection.value("target", "") };
        std::string const platform{ entitySection.value("platform", "") };
        std::vector<std::string> const agents{ entitySection.value("agents", std::vector<std::string>{}) };

        if (entityType.empty())
            continue;

        if (RequiresPlatform(entityType) && platform.empty())
            continue;

        auto entity = ppo::reflection::CreateEntity(
            entityType,
            environment,
            platform,
            ParseEntitySide(sideName));

        if (entity == nullptr)
            continue;

        entity->add<ppo::model::sEntityMetadata>(ppo::model::sEntityMetadata{
            std::string(name),
            entityType
        });

        // Agents are now instantiated via the reflection registry.
        // This removes hard-coded string-to-class dispatch while keeping
        // the current JSON schema and behavior unchanged.
        for (auto const& agentName : agents)
            auto agent = ppo::reflection::CreateAgent(agentName, environment, *entity.get());

        if (!targetName.empty())
            targetMapping[entity.get()] = targetName;

        entityMapping[std::string(name)] = entity.get();
    }

    for (auto const& [entity, target] : targetMapping)
    {
        if (entity == nullptr
        ||  !entity->has<ppo::model::sSNAState>())
            continue;

        if (auto Iter = entityMapping.find(target);
            Iter != entityMapping.end()
        &&  Iter->second != nullptr)
        {
           auto [snaState] = entity->get<ppo::model::sSNAState>();
           snaState.m_target = Iter->second->uniqid();
        }
    }
}

} // namespace

int 
main(int argc, char** argv)
{
    ppo::reflection::RegisterAllMeta();

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    std::filesystem::path serializationPath{ RESOURCE_PATH };
    serializer.SetPath(serializationPath);

    ppo::model::Environment environment{};

    std::filesystem::path scenarioPath{ std::filesystem::path(RESOURCE_PATH) / "Scenarios" / "Scenario1.json" };
    LoadScenario(scenarioPath, environment);

    serializer.Load();

    ppo::renderer::EnvironmentRenderer& environmentRenderer { ppo::renderer::EnvironmentRenderer::GetInstance("PLAYGROUND") };
    
    environmentRenderer.SetEnvironment(environment);
    
    // Main loop
    while (environmentRenderer.Run())
    {

    }

    serializer.Save();

    return 0;
}
