/**
* @file Meta/data/metaregistry.hpp
* @brief The factory tables of the scenarios: entity and agent types by scenario name.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "environment.hpp"
#include "entitytypes.hpp"
#include "ppo/runtime/iagent.hpp"

#include <memory>
#include <string_view>

namespace ppo::reflection {

/**
* @brief Creates one entity of a scenario type.
*/
using EntityFactory = std::shared_ptr<ppo::model::Entity>(*)(
    ppo::model::Environment&, std::string_view, ppo::model::eEntitySide);
/**
* @brief Creates one agent of a scenario type.
*/
using AgentFactory = std::shared_ptr<ppo::model::IAgent>(*)(
    ppo::model::Environment&, ppo::model::BaseEntity&, std::string_view);

/**
* @brief Registers the factory of a scenario entity type (examples register their own).
* @param name Scenario type name (several names may share a factory).
* @param factory Entity factory.
*/
void RegisterEntityFactory(std::string_view name, EntityFactory factory);
/**
* @brief Registers the factory of a scenario agent type.
* @param name Scenario agent name.
* @param factory Agent factory.
*/
void RegisterAgentFactory(std::string_view name, AgentFactory factory);

/**
* @brief Creates one entity from the factory of its scenario type.
* @param name Scenario entity type name.
* @param environment Environment that owns the entity.
* @param platform Platform or archetype name.
* @param side Tactical side assigned to the entity.
* @return Created entity instance, or null when the name is unknown.
*/
std::shared_ptr<ppo::model::Entity>
CreateEntity(std::string_view name,
             ppo::model::Environment& environment,
             std::string_view platform,
             ppo::model::eEntitySide side);

/**
* @brief Creates one agent from the factory of its scenario name.
* @param name Scenario agent name.
* @param environment Environment that owns the agent.
* @param entity Entity controlled by the agent.
* @param version Requested model version tag.
* @return Created agent instance, or null when the name is unknown.
*/
std::shared_ptr<ppo::model::IAgent>
CreateAgent(std::string_view name,
            ppo::model::Environment& environment,
            ppo::model::BaseEntity& entity,
            std::string_view version = "prod");

} /* ppo::reflection */
