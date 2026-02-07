#include <SFML/Graphics.hpp>

#include "utils.hpp"
#include "environment.hpp"
#include "serializer.hpp"
#include "objective.hpp"
#include "Interception.hpp"
#include "sam.hpp"

#include "environmentrenderer.hpp"

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
        ppo::model::eEntityType  type{ ppo::model::eEntityType::None };
        ppo::model::eEntitySide  side{ ppo::model::eEntitySide::Unknown };
        std::string              target{ "" };
        std::string              agent{""};

        // champs simples
        std::string tmp{ entitySection["type"] };
        if      (tmp == "Aircraft")  type = ppo::model::eEntityType::Aircraft;
        else if (tmp == "Objective") type = ppo::model::eEntityType::Objective;
        else if (tmp == "SAM")       type = ppo::model::eEntityType::SAM;

        tmp = entitySection["side"];
        if      (tmp == "Hostile")   side = ppo::model::eEntitySide::Hostile;
        else if (tmp == "Friend")    side = ppo::model::eEntitySide::Friend;

        target = entitySection["target"];
        agent  = entitySection["agent" ];

        if (type == ppo::model::eEntityType::Aircraft)
        {
            auto entity = environment.CreateEntity<ppo::model::Aircraft>(name, side);

            if (agent == "SNAProdAgent")
                auto agent = environment.CreateAgent<rl::SNAProdAgent>(*entity.get());
            else if (agent == "SNATrainingAgent")
                auto agent = environment.CreateAgent<rl::SNATrainingAgent>(*entity.get());

            if (target.compare("") != 0)
                targetMapping[entity.get()] = target;

            entityMapping[name] = entity.get();
        }
        else if (type == ppo::model::eEntityType::Objective)
        {
            auto entity = environment.CreateEntity<ppo::model::Objective>(side);

            entityMapping[name] = entity.get();
        }
        else if (type == ppo::model::eEntityType::SAM)
        {
            auto entity = environment.CreateEntity<ppo::model::SAM>(name, side);
            entityMapping[name] = entity.get();
        }
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

int 
main(int argc, char** argv)
{
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