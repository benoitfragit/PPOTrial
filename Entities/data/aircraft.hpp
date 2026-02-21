/**
* @file Entities/data/aircraft.hpp
* @brief Declarations for aircraft.
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
* @brief Aircraft entity combining flight, guidance, and control systems.
*/
class Aircraft: public Entity {
public:
    /**
    * @brief Creates an aircraft entity from one aircraft platform file.
    * @param registry ECS registry owning the entity.
    * @param name Platform configuration file name.
    * @param side Team side assigned to the aircraft.
    */
    Aircraft(entt::registry* registry, std::string const& name, eEntitySide side);

    /**
    * @brief Executes one aircraft simulation step.
    */
    void Run    (void) noexcept final;

private:
    /**
    * @brief Initializes the aircraft-specific runtime systems.
    */
    void OnInit(void) noexcept final;
};

}
