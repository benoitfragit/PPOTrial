/**
* @file Entities/data/flightmodelsystem.hpp
* @brief Declarations for flightmodelsystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "serializer.hpp"
#include "components.hpp"

#include "baseentity.hpp"

namespace ppo::model {

/**
* @brief Helper system responsible for flight-model setup and integration.
*/
class FlightModelSystem
{
public:
    /**
    * @brief Integrates one simulation step for the flight model.
    * @param entity Entity receiving the update.
    */
    static void Run(BaseEntity& entity);
    /**
    * @brief Resets flight-model state before an episode starts.
    * @param entity Entity receiving the initialization.
    */
    static void Init(BaseEntity& entity);
    /**
    * @brief Creates flight-model components from JSON configuration.
    * @param entity Entity receiving the components.
    * @param ar JSON configuration archive.
    */
    static void Create(BaseEntity& entity, ppo::serialization::json& ar);
};

} /* ppo::model */
