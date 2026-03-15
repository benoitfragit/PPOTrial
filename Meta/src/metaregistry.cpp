#include "metaregistry.hpp"

#include "aircraft.hpp"
#include "sam.hpp"
#include "objective.hpp"
#include "components.hpp"
#include "entitytypes.hpp"

#include "interception.hpp"
#include "targetselector.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace ppo::reflection {

using EntityFactory = std::shared_ptr<ppo::model::Entity>(*)(
    ppo::model::Environment&,
    std::string_view,
    ppo::model::eEntitySide);

using AgentFactory = std::shared_ptr<ppo::model::IAgent>(*)(
    ppo::model::Environment&,
    ppo::model::BaseEntity&,
    std::string_view);

namespace {

using namespace entt::literals;

std::unordered_map<entt::id_type, EntityFactory>&
EntityFactories(void)
{
    static std::unordered_map<entt::id_type, EntityFactory> factories{};
    return factories;
}

std::unordered_map<entt::id_type, AgentFactory>&
AgentFactories(void)
{
    static std::unordered_map<entt::id_type, AgentFactory> factories{};
    return factories;
}

bool&
MetaRegisteredFlag(void)
{
    static bool registered{ false };
    return registered;
}

entt::id_type
ToTypeId(std::string_view name) noexcept
{
    return entt::hashed_string::value(name.data(), name.size());
}

std::once_flag&
MetaRegistrationFlag(void)
{
    static std::once_flag once{};
    return once;
}

template<typename Type>
void
RegisterEntityType(char const* scenarioName, EntityFactory factory)
{
    entt::meta_factory<Type>{}
        .type(scenarioName)
        .custom<sMetaDescriptor>(eMetaCategory::Entity, scenarioName);

    EntityFactories()[entt::hashed_string{ scenarioName }.value()] = factory;
}

template<typename Type>
void
RegisterAgentType(char const* scenarioName, AgentFactory factory)
{
    entt::meta_factory<Type>{}
        .type(scenarioName)
        .custom<sMetaDescriptor>(eMetaCategory::Agent, scenarioName);

    AgentFactories()[entt::hashed_string{ scenarioName }.value()] = factory;
}

template<typename Enum>
void
RegisterEnumType(char const* name)
{
    entt::meta_factory<Enum>{}
        .type(name)
        .custom<sMetaDescriptor>(eMetaCategory::Enum, name);
}

std::shared_ptr<ppo::model::Entity>
CreateAircraft(ppo::model::Environment& environment,
               std::string_view platform,
               ppo::model::eEntitySide side)
{
    return environment.CreateEntity<ppo::model::Aircraft>(std::string(platform), side);
}

std::shared_ptr<ppo::model::Entity>
CreateSAM(ppo::model::Environment& environment,
          std::string_view platform,
          ppo::model::eEntitySide side)
{
    return environment.CreateEntity<ppo::model::SAM>(std::string(platform), side);
}

std::shared_ptr<ppo::model::Entity>
CreateObjective(ppo::model::Environment& environment,
                std::string_view,
                ppo::model::eEntitySide side)
{
    return environment.CreateEntity<ppo::model::Objective>(side);
}

std::shared_ptr<ppo::model::IAgent>
CreateSNATrainingAgent(ppo::model::Environment& environment,
                       ppo::model::BaseEntity& entity,
                       std::string_view)
{
    return environment.CreateAgent<rl::SNATrainingAgent>(entity);
}

std::shared_ptr<ppo::model::IAgent>
CreateSNAProdAgent(ppo::model::Environment& environment,
                   ppo::model::BaseEntity& entity,
                   std::string_view version)
{
    return environment.CreateAgent<rl::SNAProdAgent>(entity, std::string(version));
}

std::shared_ptr<ppo::model::IAgent>
CreateTargetSelectorTrainingAgent(ppo::model::Environment& environment,
                                  ppo::model::BaseEntity& entity,
                                  std::string_view)
{
    return environment.CreateAgent<rl::TargetSelectorTrainingAgent>(entity);
}

std::shared_ptr<ppo::model::IAgent>
CreateTargetSelectorProdAgent(ppo::model::Environment& environment,
                              ppo::model::BaseEntity& entity,
                              std::string_view version)
{
    return environment.CreateAgent<rl::TargetSelectorProdAgent>(entity, std::string(version));
}

void
RegisterEnums(void)
{
    RegisterEnumType<eMetaCategory>("MetaCategory");
    entt::meta_factory<eMetaCategory>{}
        .data<eMetaCategory::Unknown>("Unknown"_hs)
        .data<eMetaCategory::Enum>("Enum"_hs)
        .data<eMetaCategory::Entity>("Entity"_hs)
        .data<eMetaCategory::Component>("Component"_hs)
        .data<eMetaCategory::Agent>("Agent"_hs);

    RegisterEnumType<ppo::model::eEntitySide>("EntitySide");
    entt::meta_factory<ppo::model::eEntitySide>{}
        .data<ppo::model::eEntitySide::Unknown>("Unknown"_hs)
        .data<ppo::model::eEntitySide::Hostile>("Hostile"_hs)
        .data<ppo::model::eEntitySide::Friend>("Friend"_hs);

    RegisterEnumType<ppo::model::eEntityType>("EntityType");
    entt::meta_factory<ppo::model::eEntityType>{}
        .data<ppo::model::eEntityType::None>("None"_hs)
        .data<ppo::model::eEntityType::Objective>("Objective"_hs)
        .data<ppo::model::eEntityType::Missile>("Missile"_hs)
        .data<ppo::model::eEntityType::Aircraft>("Aircraft"_hs)
        .data<ppo::model::eEntityType::SAM>("SAM"_hs);
}

void
RegisterComponents(void)
{
    entt::meta_factory<ppo::model::sEnvironmentState>{}
        .type("EnvironmentState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "EnvironmentState")
        .data<&ppo::model::sEnvironmentState::m_maxNorth>("m_maxNorth"_hs)
        .data<&ppo::model::sEnvironmentState::m_maxEast>("m_maxEast"_hs)
        .data<&ppo::model::sEnvironmentState::m_dt>("m_dt"_hs);

    entt::meta_factory<ppo::model::sEntityState>{}
        .type("EntityState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "EntityState")
        .data<&ppo::model::sEntityState::m_entityId>("m_entityId"_hs)
        .data<&ppo::model::sEntityState::m_x>("m_x"_hs)
        .data<&ppo::model::sEntityState::m_y>("m_y"_hs)
        .data<&ppo::model::sEntityState::m_dx>("m_dx"_hs)
        .data<&ppo::model::sEntityState::m_dy>("m_dy"_hs)
        .data<&ppo::model::sEntityState::m_speed>("m_speed"_hs)
        .data<&ppo::model::sEntityState::m_rotation>("m_rotation"_hs)
        .data<&ppo::model::sEntityState::m_side>("m_side"_hs)
        .data<&ppo::model::sEntityState::m_type>("m_type"_hs)
        .data<&ppo::model::sEntityState::m_prevDestroyed>("m_prevDestroyed"_hs)
        .data<&ppo::model::sEntityState::m_destroyed>("m_destroyed"_hs)
        .data<&ppo::model::sEntityState::m_destroyedById>("m_destroyedById"_hs);

    entt::meta_factory<ppo::model::sEntityMetadata>{}
        .type("EntityMetadata"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "EntityMetadata")
        .data<&ppo::model::sEntityMetadata::m_name>("m_name"_hs)
        .data<&ppo::model::sEntityMetadata::m_archetype>("m_archetype"_hs);

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
        .data<&ppo::model::sPseudoLidarState::m_nearestCollisionRanges>("m_nearestCollisionRanges"_hs);

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

    entt::meta_factory<ppo::model::sEvaluatorState>{}
        .type("EvaluatorState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "EvaluatorState")
        .data<&ppo::model::sEvaluatorState::m_nbstep>("m_nbstep"_hs)
        .data<&ppo::model::sEvaluatorState::m_nbtests>("m_nbtests"_hs)
        .data<&ppo::model::sEvaluatorState::m_notation>("m_notation"_hs)
        .data<&ppo::model::sEvaluatorState::m_progress>("m_progress"_hs)
        .data<&ppo::model::sEvaluatorState::m_inprogress>("m_inprogress"_hs);

    entt::meta_factory<ppo::model::PPOParamters>{}
        .type("PPOParameters"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "PPOParameters")
        .data<&ppo::model::PPOParamters::m_gamma>("m_gamma"_hs)
        .data<&ppo::model::PPOParamters::m_lambda_gae>("m_lambda_gae"_hs)
        .data<&ppo::model::PPOParamters::m_clip_epsilon>("m_clip_epsilon"_hs)
        .data<&ppo::model::PPOParamters::m_value_coef>("m_value_coef"_hs)
        .data<&ppo::model::PPOParamters::m_entropy_coef>("m_entropy_coef"_hs)
        .data<&ppo::model::PPOParamters::m_learningReate>("m_learningReate"_hs)
        .data<&ppo::model::PPOParamters::m_K_epochs>("m_K_epochs"_hs)
        .data<&ppo::model::PPOParamters::m_batch_size>("m_batch_size"_hs)
        .data<&ppo::model::PPOParamters::m_target_kl>("m_target_kl"_hs);

    entt::meta_factory<ppo::model::sCommonAgentState>{}
        .type("CommonAgentState"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "CommonAgentState")
        .base<ppo::model::PPOParamters>()
        .data<&ppo::model::sCommonAgentState::m_step>("m_step"_hs)
        .data<&ppo::model::sCommonAgentState::m_maxStep>("m_maxStep"_hs)
        .data<&ppo::model::sCommonAgentState::m_reward>("m_reward"_hs)
        .data<&ppo::model::sCommonAgentState::m_rewardClamp>("m_rewardClamp"_hs)
        .data<&ppo::model::sCommonAgentState::m_loss>("m_loss"_hs)
        .data<&ppo::model::sCommonAgentState::m_actorloss>("m_actorloss"_hs)
        .data<&ppo::model::sCommonAgentState::m_entropyloss>("m_entropyloss"_hs)
        .data<&ppo::model::sCommonAgentState::m_valueloss>("m_valueloss"_hs)
        .data<&ppo::model::sCommonAgentState::m_checkpoint>("m_checkpoint"_hs)
        .data<&ppo::model::sCommonAgentState::m_version>("m_version"_hs)
        .data<&ppo::model::sCommonAgentState::m_notation>("m_notation"_hs)
        .data<&ppo::model::sCommonAgentState::m_evaluationprogress>("m_evaluationprogress"_hs)
        .data<&ppo::model::sCommonAgentState::m_evaluationinprogress>("m_evaluationinprogress"_hs)
        .data<&ppo::model::sCommonAgentState::m_done>("m_done"_hs)
        .data<&ppo::model::sCommonAgentState::m_training>("m_training"_hs);

    entt::meta_factory<ppo::model::sAgentStateSnaphot>{}
        .type("AgentStateSnapshot"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "AgentStateSnapshot")
        .data<&ppo::model::sAgentStateSnaphot::m_common>("m_common"_hs)
        .data<&ppo::model::sAgentStateSnaphot::m_scalings>("m_scalings"_hs)
        .data<&ppo::model::sAgentStateSnaphot::m_rewards>("m_rewards"_hs)
        .data<&ppo::model::sAgentStateSnaphot::m_rewardnames>("m_rewardnames"_hs)
        .data<&ppo::model::sAgentStateSnaphot::m_name>("m_name"_hs)
        .data<&ppo::model::sAgentStateSnaphot::m_id>("m_id"_hs);

    entt::meta_factory<rl::StatesAndActions>{}
        .type("InterceptionStateModel"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "InterceptionStateModel")
        .data<&rl::StatesAndActions::m_states>("m_states"_hs)
        .data<&rl::StatesAndActions::m_actions>("m_actions"_hs);

    entt::meta_factory<rl::sRealThreat>{}
        .type("RealThreat"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "RealThreat")
        .data<&rl::sRealThreat::m_x>("m_x"_hs)
        .data<&rl::sRealThreat::m_y>("m_y"_hs)
        .data<&rl::sRealThreat::m_noescapeZone>("m_noescapeZone"_hs)
        .data<&rl::sRealThreat::m_sam>("m_sam"_hs);

    entt::meta_factory<rl::StatesAndActionsForMission>{}
        .type("MissionStateModel"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Component, "MissionStateModel")
        .data<&rl::StatesAndActionsForMission::m_threats>("m_threats"_hs);
}

void
RegisterEntities(void)
{
    entt::meta_factory<ppo::model::BaseEntity>{}
        .type("BaseEntity"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Entity, "BaseEntity");

    entt::meta_factory<ppo::model::Entity>{}
        .type("Entity"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Entity, "Entity")
        .base<ppo::model::BaseEntity>();

    RegisterEntityType<ppo::model::Aircraft>("Aircraft", &CreateAircraft);
    entt::meta_factory<ppo::model::Aircraft>{}.base<ppo::model::Entity>();

    RegisterEntityType<ppo::model::SAM>("SAM", &CreateSAM);
    entt::meta_factory<ppo::model::SAM>{}.base<ppo::model::Entity>();

    RegisterEntityType<ppo::model::Objective>("Objective", &CreateObjective);
    entt::meta_factory<ppo::model::Objective>{}.base<ppo::model::Entity>();
}

void
RegisterAgents(void)
{
    entt::meta_factory<ppo::model::IAgent>{}
        .type("IAgent"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Agent, "IAgent");

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

void
RegisterAllMeta(void)
{
    std::call_once(MetaRegistrationFlag(), []()
    {
        RegisterEnums();
        RegisterComponents();
        RegisterEntities();
        RegisterAgents();
        MetaRegisteredFlag() = true;
    });
}

bool
IsMetaRegistered(void) noexcept
{
    return MetaRegisteredFlag();
}

entt::meta_type
ResolveType(std::string_view name) noexcept
{
    RegisterAllMeta();
    return entt::resolve(ToTypeId(name));
}

std::shared_ptr<ppo::model::Entity>
CreateEntity(std::string_view name,
             ppo::model::Environment& environment,
             std::string_view platform,
             ppo::model::eEntitySide side)
{
    RegisterAllMeta();

    auto const iter = EntityFactories().find(ToTypeId(name));
    if (iter == EntityFactories().end() || iter->second == nullptr)
        return nullptr;

    return iter->second(environment, platform, side);
}

std::shared_ptr<ppo::model::IAgent>
CreateAgent(std::string_view name,
            ppo::model::Environment& environment,
            ppo::model::BaseEntity& entity,
            std::string_view version)
{
    RegisterAllMeta();

    auto const iter = AgentFactories().find(ToTypeId(name));
    if (iter == AgentFactories().end() || iter->second == nullptr)
        return nullptr;

    return iter->second(environment, entity, version);
}

} /* ppo::reflection */
