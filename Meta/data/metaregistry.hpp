/**
* @file Meta/data/metaregistry.hpp
* @brief Reflection registry helpers for entities, agents, and components.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entt/meta/factory.hpp"
#include "entt/meta/resolve.hpp"

#include "environment.hpp"
#include "entitytypes.hpp"
#include "iagent.hpp"

#include <memory>
#include <string_view>

namespace ppo::reflection {

/**
* @brief Categories used to organize reflected types in the meta catalog.
*/
enum class eMetaCategory : unsigned char
{
    /** @brief Fallback category for unclassified reflected types. */
    Unknown = 0,
    /** @brief Reflected enumeration type. */
    Enum,
    /** @brief Reflected entity type. */
    Entity,
    /** @brief Reflected ECS component type. */
    Component,
    /** @brief Reflected agent type. */
    Agent
};

/**
* @brief Small descriptor attached to reflected types.
*/
struct sMetaDescriptor
{
    /**
    * @brief Builds one default descriptor.
    */
    sMetaDescriptor(void) = default;

    /**
    * @brief Builds one descriptor with an explicit category and scenario name.
    * @param value Meta category attached to the reflected type.
    * @param name Scenario-facing type name.
    */
    sMetaDescriptor(eMetaCategory value, char const* name) :
        category{ value },
        scenario_name{ name }
    {
    }

    /** @brief Category used to group the reflected type in the UI. */
    eMetaCategory category{ eMetaCategory::Unknown };
    /** @brief Scenario-visible name associated with the reflected type. */
    char const* scenario_name{ "" };
};

/**
* @brief Registers all reflected enums, components, entities, and agents.
*/
void RegisterAllMeta(void);

/**
* @brief Tells whether the meta registry has already been populated.
* @return True when registration already happened.
*/
bool IsMetaRegistered(void) noexcept;

/**
* @brief Resolves one reflected type by its scenario-visible name.
* @param name Scenario-visible type name.
* @return Resolved EnTT meta type, or an invalid handle when unknown.
*/
entt::meta_type ResolveType(std::string_view name) noexcept;

/**
* @brief Creates one entity instance from the reflected entity registry.
* @param name Scenario-visible entity type name.
* @param environment Environment that owns the new entity.
* @param platform Platform or archetype name used by the factory.
* @param side Tactical side assigned to the entity.
* @return Newly created entity instance, or null when the type is unknown.
*/
std::shared_ptr<ppo::model::Entity>
CreateEntity(std::string_view name,
             ppo::model::Environment& environment,
             std::string_view platform,
             ppo::model::eEntitySide side);

/**
* @brief Creates one agent instance from the reflected agent registry.
* @param name Scenario-visible agent type name.
* @param environment Environment that owns the new agent.
* @param entity Entity controlled by the agent.
* @param version Version tag used to select prod or eval assets.
* @return Newly created agent instance, or null when the type is unknown.
*/
std::shared_ptr<ppo::model::IAgent>
CreateAgent(std::string_view name,
            ppo::model::Environment& environment,
            ppo::model::BaseEntity& entity,
            std::string_view version = "prod");

} /* ppo::reflection */
