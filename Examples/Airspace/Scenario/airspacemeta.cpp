/**
 * @file Examples/Airspace/Scenario/airspacemeta.cpp
 * @brief The entity and agent factories of the Airspace scenarios.
 */
#include "airspacemeta.hpp"
#include "metaregistry.hpp"
#include "mobile.hpp"
#include "restrictedzone.hpp"
#include "objective.hpp"
#include "interception.hpp"
#include "targetselector.hpp"

#include <mutex>
#include <string>

namespace ppo::examples::airspace {
namespace {

/**
* @brief Creates one Mobile entity through the environment factory.
* @param environment Target environment.
* @param platform Platform archetype name.
* @param side Tactical side assigned to the entity.
* @return Created Mobile entity.
*/
std::shared_ptr<ppo::model::Entity>
CreateMobile(ppo::model::Environment& environment,
               std::string_view platform,
               ppo::model::eEntitySide side)
{
    return environment.CreateEntity<ppo::model::Mobile>(std::string(platform), side);
}

/**
* @brief Creates one RestrictedZone entity through the environment factory.
* @param environment Target environment.
* @param platform Platform archetype name.
* @param side Tactical side assigned to the entity.
* @return Created RestrictedZone entity.
*/
std::shared_ptr<ppo::model::Entity>
CreateRestrictedZone(ppo::model::Environment& environment,
          std::string_view platform,
          ppo::model::eEntitySide side)
{
    return environment.CreateEntity<ppo::model::RestrictedZone>(std::string(platform), side);
}

/**
* @brief Creates one objective entity through the environment factory.
* @param environment Target environment.
* @param side Tactical side assigned to the entity.
* @return Created objective entity.
*/
std::shared_ptr<ppo::model::Entity>
CreateObjective(ppo::model::Environment& environment,
                std::string_view,
                ppo::model::eEntitySide side)
{
    return environment.CreateEntity<ppo::model::Objective>(side);
}

/**
* @brief Creates the interception training agent.
* @param environment Target environment.
* @param entity Controlled entity.
* @return Created training agent.
*/
std::shared_ptr<ppo::model::IAgent>
CreateSNATrainingAgent(ppo::model::Environment& environment,
                       ppo::model::BaseEntity& entity,
                       std::string_view)
{
    return environment.CreateAgent<rl::SNATrainingAgent>(entity);
}

/**
* @brief Creates the interception production agent.
* @param environment Target environment.
* @param entity Controlled entity.
* @param version Requested model version tag.
* @return Created production agent.
*/
std::shared_ptr<ppo::model::IAgent>
CreateSNAProdAgent(ppo::model::Environment& environment,
                   ppo::model::BaseEntity& entity,
                   std::string_view version)
{
    return environment.CreateAgent<rl::SNAProdAgent>(entity, std::string(version));
}

/**
* @brief Creates the mission training agent.
* @param environment Target environment.
* @param entity Controlled entity.
* @return Created training agent.
*/
std::shared_ptr<ppo::model::IAgent>
CreateTargetSelectorTrainingAgent(ppo::model::Environment& environment,
                                  ppo::model::BaseEntity& entity,
                                  std::string_view)
{
    return environment.CreateAgent<rl::TargetSelectorTrainingAgent>(entity);
}

/**
* @brief Creates the mission production agent.
* @param environment Target environment.
* @param entity Controlled entity.
* @param version Requested model version tag.
* @return Created production agent.
*/
std::shared_ptr<ppo::model::IAgent>
CreateTargetSelectorProdAgent(ppo::model::Environment& environment,
                              ppo::model::BaseEntity& entity,
                              std::string_view version)
{
    return environment.CreateAgent<rl::TargetSelectorProdAgent>(entity, std::string(version));
}


/** @brief The entity factories of the Airspace scenarios (with their historical names). */
void RegisterAirspaceEntities()
{
    ppo::reflection::RegisterEntityFactory("Mobile", &CreateMobile);
    ppo::reflection::RegisterEntityFactory("Aircraft", &CreateMobile);
    ppo::reflection::RegisterEntityFactory("RestrictedZone", &CreateRestrictedZone);
    ppo::reflection::RegisterEntityFactory("SAM", &CreateRestrictedZone);
    ppo::reflection::RegisterEntityFactory("Objective", &CreateObjective);
}

/** @brief The agent factories of the Airspace scenarios. */
void RegisterAirspaceAgents()
{
    ppo::reflection::RegisterAgentFactory("SNATrainingAgent", &CreateSNATrainingAgent);
    ppo::reflection::RegisterAgentFactory("SNAProdAgent", &CreateSNAProdAgent);
    ppo::reflection::RegisterAgentFactory("TargetSelectorTrainingAgent", &CreateTargetSelectorTrainingAgent);
    ppo::reflection::RegisterAgentFactory("TargetSelectorProdAgent", &CreateTargetSelectorProdAgent);
}
} // namespace

/**
 * @brief Registers the Airspace factories, once.
 */
void RegisterAirspaceMeta()
{
    static std::once_flag once;
    std::call_once(once, []()
    {
        RegisterAirspaceEntities();
        RegisterAirspaceAgents();
    });
}
} // namespace ppo::examples::airspace
