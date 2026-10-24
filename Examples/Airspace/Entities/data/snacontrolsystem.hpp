/**
* @file Entities/data/snacontrolsystem.hpp
* @brief Declarations for snacontrolsystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <array>

#include "serializer.hpp"
#include "baseentity.hpp"
#include "components.hpp"

namespace ppo::model {

/**
* @brief Control system converting agent commands into platform actuation.
*/
class SNAControlSystem
{
public:
    /**
    * @brief Initializes command state for one entity.
    * @param entity Entity receiving the command state.
    */
    static void Init(BaseEntity& entity);
    /**
    * @brief Applies control commands during one simulation step.
    * @param entity Entity receiving the update.
    */
    static void Run(BaseEntity& entity);
    /**
    * @brief Creates control-related components from JSON configuration.
    * @param entity Entity receiving the components.
    * @param ar JSON configuration archive.
    */
    static void Create(BaseEntity& entity, ppo::serialization::json& ar);
};

} /* ppo::model */
