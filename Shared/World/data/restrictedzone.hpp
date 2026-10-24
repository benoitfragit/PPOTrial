/**
* @file Entities/data/restrictedzone.hpp
* @brief Declarations for restrictedzone.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <string>

#include "entity.hpp"

namespace ppo::model {

/**
* @brief Stationary restricted zone with the historical range-only elimination behavior.
*/
class RestrictedZone: public Entity {
public:
    /**
    * @brief Creates a RestrictedZone entity from one platform file.
    * @param registry ECS registry owning the entity.
    * @param name Platform configuration file name.
    * @param side Team side assigned to the RestrictedZone.
    */
    RestrictedZone(entt::registry* registry, std::string const& name, eEntitySide side);

    /**
    * @brief Executes one RestrictedZone simulation step.
    */
    void Run    (void) noexcept final;

private:
    /**
    * @brief Initializes the RestrictedZone-specific runtime systems.
    */
    void OnInit(void) noexcept final;
};

}
