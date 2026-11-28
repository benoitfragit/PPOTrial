/**
* @file Entities/data/objective.hpp
* @brief Declarations for objective.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entity.hpp"

namespace ppo::model {

/**
* @brief Static mission objective entity.
*/
class Objective : public Entity
{
public:
    /**
    * @brief Creates a simple objective marker.
    * @param registry ECS registry owning the entity.
    * @param side Team side associated with the objective.
    */
    Objective(entt::registry* registry, eEntitySide side) : 
        Entity(registry, side, eEntityType::Objective) {};
};

} /* ppo::model */
