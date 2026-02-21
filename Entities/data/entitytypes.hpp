/**
* @file Entities/data/entitytypes.hpp
* @brief Declarations for entitytypes.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entt/entity/fwd.hpp"

namespace ppo::model {

/**
* @brief Alias used to store ECS entity identifiers in gameplay components.
*/
using EntityId = entt::entity;

/**
* @brief Team side assigned to one entity.
*/
enum class eEntitySide :
    unsigned char
{
    /**
    * @brief Side is unknown or neutral.
    */
    Unknown = 0,
    /**
    * @brief Entity belongs to the hostile side.
    */
    Hostile = 1,
    /**
    * @brief Entity belongs to the friendly side.
    */
    Friend = 2
};

/**
* @brief Runtime category assigned to one entity.
*/
enum class eEntityType :
    unsigned char
{
    /**
    * @brief Undefined type.
    */
    None = 0,
    /**
    * @brief Static mission objective.
    */
    Objective = 1,
    /**
    * @brief Missile-like object.
    */
    Missile = 2,
    /**
    * @brief Aircraft platform.
    */
    Aircraft = 3,
    /**
    * @brief Surface-to-air missile battery.
    */
    SAM = 4
};

} /* ppo::model */
