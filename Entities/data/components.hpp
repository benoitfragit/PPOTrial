/**
* @file Entities/data/components.hpp
* @brief Declarations for components.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <array>
#include <string>
#include <vector>

#include "entt/entt.hpp"

#include "entitytypes.hpp"
#include "ppo/core/evaluator_state.hpp"

namespace ppo::model {

/**
* @brief Global environment parameters shared through the registry context.
*/
struct sEnvironmentState
{
    /**
    * @brief North-axis world size in meters.
    */
    float m_maxNorth{ 112500.F };
    /**
    * @brief East-axis world size in meters.
    */
    float m_maxEast{ 150000.F };
    /**
    * @brief Fixed simulation time step in seconds.
    */
    float m_dt{ 0.02F };
};

/**
* @brief Core kinematic and ownership state of one entity.
*/
struct sEntityState
{
    /**
    * @brief Unique ECS identifier of the entity.
    */
    EntityId m_entityId{ entt::null };
    /**
    * @brief North position in meters.
    */
    float m_x{ 0.F };
    /**
    * @brief East position in meters.
    */
    float m_y{ 0.F };
    /**
    * @brief North velocity component in meters per second.
    */
    float m_dx{ 0.F };
    /**
    * @brief East velocity component in meters per second.
    */
    float m_dy{ 0.F };
    /**
    * @brief Scalar speed in meters per second.
    */
    float m_speed{ 0.F };
    /**
    * @brief Heading angle in radians.
    */
    float m_rotation{ 0.F };
    /**
    * @brief Team side owning the entity.
    */
    eEntitySide m_side{ eEntitySide::Unknown };
    /**
    * @brief Functional type of the entity.
    */
    eEntityType m_type{ eEntityType::Objective };
    /**
    * @brief Destroyed state captured on the previous simulation step.
    */
    bool m_prevDestroyed{ false };
    /**
    * @brief Current destroyed state.
    */
    bool m_destroyed{ false };
    /**
    * @brief Identifier of the destroying entity when known.
    */
    EntityId m_destroyedById{ entt::null };
};

/**
* @brief Human-readable metadata used by tooling and rendering.
*/
struct sEntityMetadata
{
    /**
    * @brief Scenario-level entity name.
    */
    std::string m_name{};
    /**
    * @brief Archetype or platform name used to instantiate the entity.
    */
    std::string m_archetype{};
};

} /* ppo::model */
