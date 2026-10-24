#include "airspacemeta.hpp"
#include "metaregistry.hpp"
#include "mobile.hpp"
#include "restrictedzone.hpp"
#include "objective.hpp"
#include "components.hpp"
#include "airspacecomponents.hpp"
#include "interception.hpp"
#include "targetselector.hpp"

#include <mutex>
#include <string>

namespace ppo::examples::airspace {
namespace {
using namespace entt::literals;
using ppo::reflection::eMetaCategory;
using ppo::reflection::sMetaDescriptor;
using ppo::reflection::EntityFactory;
using ppo::reflection::AgentFactory;

template<typename Type>
/**
* @brief Registers one reflected entity type and its scenario factory.
* @param scenarioName Scenario-visible type name.
* @param factory Factory used to instantiate the entity.
*/
void
RegisterEntityType(char const* scenarioName, EntityFactory factory)
{
    entt::meta_factory<Type>{}
        .type(scenarioName)
        .custom<sMetaDescriptor>(eMetaCategory::Entity, scenarioName);

    ppo::reflection::RegisterEntityFactory(scenarioName, factory);
}

template<typename Type>
/**
* @brief Registers one reflected agent type and its scenario factory.
* @param scenarioName Scenario-visible type name.
* @param factory Factory used to instantiate the agent.
*/
void
RegisterAgentType(char const* scenarioName, AgentFactory factory)
{
    entt::meta_factory<Type>{}
        .type(scenarioName)
        .custom<sMetaDescriptor>(eMetaCategory::Agent, scenarioName);

    ppo::reflection::RegisterAgentFactory(scenarioName, factory);
}


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


void RegisterAirspaceEnums()
{
    entt::meta_factory<ppo::model::eEntityType>{}
        .data<ppo::model::eEntityType::Objective>("Objective"_hs)
        .data<ppo::model::eEntityType::Missile>("Missile"_hs)
        .data<ppo::model::eEntityType::Mobile>("Mobile"_hs)
        .data<ppo::model::eEntityType::Aircraft>("Aircraft"_hs)
        .data<ppo::model::eEntityType::RestrictedZone>("RestrictedZone"_hs)
        .data<ppo::model::eEntityType::SAM>("SAM"_hs);
}

void RegisterAirspaceComponents()
{
    entt::meta_factory<ppo::model::sSNAState>{}
        .type("SNAState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "SNAState")
        .data<&ppo::model::sSNAState::m_target>("m_target"_hs)
        .data<&ppo::model::sSNAState::m_pk>("m_pk"_hs)
        .data<&ppo::model::sSNAState::m_target_x>("m_target_x"_hs)
        .data<&ppo::model::sSNAState::m_target_y>("m_target_y"_hs)
        .data<&ppo::model::sSNAState::m_target_dx>("m_target_dx"_hs)
        .data<&ppo::model::sSNAState::m_target_dy>("m_target_dy"_hs)
        .data<&ppo::model::sSNAState::m_closingspeed>("m_closingspeed"_hs)
        .data<&ppo::model::sSNAState::m_target_los>("m_target_los"_hs)
        .data<&ppo::model::sSNAState::m_target_los_rate>("m_target_los_rate"_hs)
        .data<&ppo::model::sSNAState::m_maxNoescapeZone>("m_maxNoescapeZone"_hs)
        .data<&ppo::model::sSNAState::m_escapezone>("m_escapezone"_hs)
        .data<&ppo::model::sSNAState::m_noescapezone>("m_noescapezone"_hs)
        .data<&ppo::model::sSNAState::m_maxShootAngle>("m_maxShootAngle"_hs)
        .data<&ppo::model::sSNAState::m_nbmissiles>("m_nbmissiles"_hs)
        .data<&ppo::model::sSNAState::m_maxnbmissiles>("m_maxnbmissiles"_hs);

    entt::meta_factory<ppo::model::sSNAControlState>{}
        .type("SNAControlState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "SNAControlState")
        .data<&ppo::model::sSNAControlState::m_accCmd>("m_accCmd"_hs)
        .data<&ppo::model::sSNAControlState::m_rotSpeedCmd>("m_rotSpeedCmd"_hs);

    entt::meta_factory<ppo::model::sPseudoLidarState>{}
        .type("PseudoLidarState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "PseudoLidarState")
        .data<&ppo::model::sPseudoLidarState::m_nearestCollisionRanges>("m_nearestCollisionRanges"_hs)
        .data<&ppo::model::sPseudoLidarState::m_nearestCollisionDistances>("m_nearestCollisionDistances"_hs)
        .data<&ppo::model::sPseudoLidarState::m_angleSpanRadians>("m_angleSpanRadians"_hs);

    entt::meta_factory<ppo::model::sFlightModelState>{}
        .type("FlightModelState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "FlightModelState")
        .data<&ppo::model::sFlightModelState::m_acc>("m_acc"_hs)
        .data<&ppo::model::sFlightModelState::m_accpsi>("m_accpsi"_hs)
        .data<&ppo::model::sFlightModelState::m_maxAcc>("m_maxAcc"_hs)
        .data<&ppo::model::sFlightModelState::m_maxAccRot>("m_maxAccRot"_hs)
        .data<&ppo::model::sFlightModelState::m_maxSpeed>("m_maxSpeed"_hs)
        .data<&ppo::model::sFlightModelState::m_minSpeed>("m_minSpeed"_hs)
        .data<&ppo::model::sFlightModelState::m_fuel>("m_fuel"_hs)
        .data<&ppo::model::sFlightModelState::m_fuelconsumption>("m_fuelconsumption"_hs)
        .data<&ppo::model::sFlightModelState::m_maxfuel>("m_maxfuel"_hs);


    entt::meta_factory<rl::StatesAndActions>{}
        .type("InterceptionStateModel"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "InterceptionStateModel")
        .data<&rl::StatesAndActions::m_states>("m_states"_hs)
        .data<&rl::StatesAndActions::m_actions>("m_actions"_hs);

    entt::meta_factory<rl::StatesAndActionsForMission>{}
        .type("MissionStateModel"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "MissionStateModel");
}

void RegisterAirspaceEntities()
{
    RegisterEntityType<ppo::model::Mobile>("Mobile", &CreateMobile);
    // Keep historical scenario and reflection names as aliases.
    ppo::reflection::RegisterEntityFactory("Aircraft", &CreateMobile);
    ppo::reflection::RegisterTypeAlias("Aircraft", "Mobile");
    entt::meta_factory<ppo::model::Mobile>{}.base<ppo::model::Entity>();

    RegisterEntityType<ppo::model::RestrictedZone>("RestrictedZone", &CreateRestrictedZone);
    ppo::reflection::RegisterEntityFactory("SAM", &CreateRestrictedZone);
    ppo::reflection::RegisterTypeAlias("SAM", "RestrictedZone");
    entt::meta_factory<ppo::model::RestrictedZone>{}.base<ppo::model::Entity>();

    RegisterEntityType<ppo::model::Objective>("Objective", &CreateObjective);
    entt::meta_factory<ppo::model::Objective>{}.base<ppo::model::Entity>();

}

void RegisterAirspaceAgents()
{
    RegisterAgentType<rl::SNATrainingAgent>("SNATrainingAgent", &CreateSNATrainingAgent);
    entt::meta_factory<rl::SNATrainingAgent>{}.base<ppo::model::IAgent>();

    RegisterAgentType<rl::SNAProdAgent>("SNAProdAgent", &CreateSNAProdAgent);
    entt::meta_factory<rl::SNAProdAgent>{}.base<ppo::model::IAgent>();

    RegisterAgentType<rl::TargetSelectorTrainingAgent>("TargetSelectorTrainingAgent", &CreateTargetSelectorTrainingAgent);
    entt::meta_factory<rl::TargetSelectorTrainingAgent>{}.base<ppo::model::IAgent>();

    RegisterAgentType<rl::TargetSelectorProdAgent>("TargetSelectorProdAgent", &CreateTargetSelectorProdAgent);
    entt::meta_factory<rl::TargetSelectorProdAgent>{}.base<ppo::model::IAgent>();

}
} // namespace

void RegisterAirspaceMeta()
{
    ppo::reflection::RegisterAllMeta();
    static std::once_flag once;
    std::call_once(once, []()
    {
        RegisterAirspaceEnums();
        RegisterAirspaceComponents();
        RegisterAirspaceEntities();
        RegisterAirspaceAgents();
    });
}
} // namespace ppo::examples::airspace
