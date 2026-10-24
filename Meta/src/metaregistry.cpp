/**
* @file Meta/src/metaregistry.cpp
* @brief Implementation for metaregistry.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "metaregistry.hpp"

#include "components.hpp"
#include "entitytypes.hpp"


#include <mutex>
#include <string>
#include <unordered_map>

namespace ppo::reflection {

namespace {

using namespace entt::literals;

/**
* @brief Returns the singleton map of reflected entity factories.
* @return Mutable entity-factory map.
*/
std::unordered_map<entt::id_type, EntityFactory>&
EntityFactories(void)
{
    static std::unordered_map<entt::id_type, EntityFactory> factories{};
    return factories;
}

/**
* @brief Returns the singleton map of reflected agent factories.
* @return Mutable agent-factory map.
*/
std::unordered_map<entt::id_type, AgentFactory>&
AgentFactories(void)
{
    static std::unordered_map<entt::id_type, AgentFactory> factories{};
    return factories;
}

std::mutex&
FactoryMutex(void)
{
    static std::mutex mutex{};
    return mutex;
}

/**
* @brief Returns the shared registration-complete flag.
* @return Mutable registration flag.
*/
bool&
MetaRegisteredFlag(void)
{
    static bool registered{ false };
    return registered;
}

/**
* @brief Converts one scenario-visible name to the EnTT hashed identifier.
* @param name Scenario-visible name.
* @return Hashed identifier used by EnTT meta.
*/
entt::id_type
ToTypeId(std::string_view name) noexcept
{
    return entt::hashed_string::value(name.data(), name.size());
}

/**
* @brief Returns the once flag guarding meta registration.
* @return Shared once flag.
*/
std::once_flag&
MetaRegistrationFlag(void)
{
    static std::once_flag once{};
    return once;
}

template<typename Enum>
/**
* @brief Registers one reflected enumeration type.
* @param name Scenario-visible enum name.
*/
void
RegisterEnumType(char const* name)
{
    entt::meta_factory<Enum>{}
        .type(name)
        .custom<sMetaDescriptor>(eMetaCategory::Enum, name);
}

/**
* @brief Registers reflected enumerations exposed by the project.
*/
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
        .data<ppo::model::eEntityType::None>("None"_hs);
}

/**
* @brief Registers reflected ECS components and lightweight state structures.
*/
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


}

/**
* @brief Registers reflected entity base types and concrete entity classes.
*/
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

}

/**
* @brief Registers reflected agent base types and concrete agent classes.
*/
void
RegisterAgents(void)
{
    entt::meta_factory<ppo::model::IAgent>{}
        .type("IAgent"_hs)
        .custom<sMetaDescriptor>(eMetaCategory::Agent, "IAgent");

}

} // namespace

/**
* @brief Populates the global meta registry once.
*/
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


// Concrete examples register their factories before any simulation worker starts.
void RegisterEntityFactory(std::string_view name, EntityFactory factory)
{
    RegisterAllMeta();
    std::lock_guard<std::mutex> lock(FactoryMutex());
    EntityFactories()[ToTypeId(name)] = factory;
}

void RegisterAgentFactory(std::string_view name, AgentFactory factory)
{
    RegisterAllMeta();
    std::lock_guard<std::mutex> lock(FactoryMutex());
    AgentFactories()[ToTypeId(name)] = factory;
}

/**
* @brief Returns whether meta registration already happened.
* @return True when the registry is populated.
*/
bool
IsMetaRegistered(void) noexcept
{
    return MetaRegisteredFlag();
}

/**
* @brief Resolves one reflected type by its scenario-visible name.
* @param name Scenario-visible type name.
* @return Resolved EnTT meta type.
*/
entt::meta_type
ResolveType(std::string_view name) noexcept
{
    RegisterAllMeta();
    return entt::resolve(ToTypeId(name));
}

/**
* @brief Creates one entity instance from the reflected factory table.
* @param name Scenario-visible entity type name.
* @param environment Environment that owns the entity.
* @param platform Platform or archetype name.
* @param side Tactical side assigned to the entity.
* @return Created entity instance, or null when unknown.
*/
std::shared_ptr<ppo::model::Entity>
CreateEntity(std::string_view name,
             ppo::model::Environment& environment,
             std::string_view platform,
             ppo::model::eEntitySide side)
{
    RegisterAllMeta();

    EntityFactory factory = nullptr;
    {
        std::lock_guard<std::mutex> lock(FactoryMutex());
        auto const iter = EntityFactories().find(ToTypeId(name));
        if (iter == EntityFactories().end())
            return nullptr;
        factory = iter->second;
    }
    return factory != nullptr ? factory(environment, platform, side) : nullptr;
}

/**
* @brief Creates one agent instance from the reflected factory table.
* @param name Scenario-visible agent type name.
* @param environment Environment that owns the agent.
* @param entity Entity controlled by the agent.
* @param version Requested model version tag.
* @return Created agent instance, or null when unknown.
*/
std::shared_ptr<ppo::model::IAgent>
CreateAgent(std::string_view name,
            ppo::model::Environment& environment,
            ppo::model::BaseEntity& entity,
            std::string_view version)
{
    RegisterAllMeta();

    AgentFactory factory = nullptr;
    {
        std::lock_guard<std::mutex> lock(FactoryMutex());
        auto const iter = AgentFactories().find(ToTypeId(name));
        if (iter == AgentFactories().end())
            return nullptr;
        factory = iter->second;
    }
    return factory != nullptr ? factory(environment, entity, version) : nullptr;
}

} /* ppo::reflection */
