/**
* @file Entities/data/sam.hpp
* @brief Declarations for sam.
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
* @brief Surface-to-air missile battery entity.
*/
class SAM: public Entity {
public:
    /**
    * @brief Creates a SAM entity from one platform file.
    * @param registry ECS registry owning the entity.
    * @param name Platform configuration file name.
    * @param side Team side assigned to the SAM.
    */
    SAM(entt::registry* registry, std::string const& name, eEntitySide side);

    /**
    * @brief Executes one SAM simulation step.
    */
    void Run    (void) noexcept final;

private:
    /**
    * @brief Initializes the SAM-specific runtime systems.
    */
    void OnInit(void) noexcept final;
};

}
