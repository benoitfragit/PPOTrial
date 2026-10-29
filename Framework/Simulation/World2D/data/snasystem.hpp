/**
* @file Entities/data/snasystem.hpp
* @brief Declarations for snasystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "serializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include "baseentity.hpp"

namespace ppo::model {

/**
* @brief Situational awareness and engagement system.
*/
class SNASystem
{
public:
    /**
    * @brief Initializes SNA state before an episode starts.
    * @param entity Entity receiving the initialization.
    */
    static void Init(BaseEntity& entity);
    /**
    * @brief Updates engagement geometry and kill estimates for one step.
    * @param entity Entity receiving the update.
    */
    static void Run(BaseEntity& entity);
    /**
    * @brief Creates SNA components from JSON configuration.
    * @param entity Entity receiving the components.
    * @param ar JSON configuration archive.
    */
    static void Create(BaseEntity& entity, ppo::serialization::json& ar);
    /**
    * @brief Evaluates the destruction probability of one entity against another.
    * @param entity Shooter or threat entity.
    * @param OtherEntity Potential victim entity.
    * @return Estimated kill probability.
    */
    static float EvaluateDestruction(BaseEntity& entity, BaseEntity& OtherEntity);
};

} /* ppo::model */
