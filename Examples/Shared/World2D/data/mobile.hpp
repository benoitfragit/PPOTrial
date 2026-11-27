/**
* @file Entities/data/mobile.hpp
* @brief Declarations for mobile.
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
* @brief Mobile entity retaining the historical motion, guidance and control systems.
*/
class Mobile: public Entity {
public:
    /**
    * @brief Creates a mobile entity from one mobile platform file.
    * @param registry ECS registry owning the entity.
    * @param name Platform configuration file name.
    * @param side Team side assigned to the mobile.
    */
    Mobile(entt::registry* registry, std::string const& name, eEntitySide side);

    /**
    * @brief Executes one mobile simulation step.
    */
    void Run    (void) noexcept final;

private:
    /**
    * @brief Initializes the mobile-specific runtime systems.
    */
    void OnInit(void) noexcept final;
};

}
