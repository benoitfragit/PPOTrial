/**
* @file Initializer/data/randominitializersystem.hpp
* @brief Declarations for randominitializersystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entity.hpp"

namespace ppo::model {

/**
* @brief Default initializer creating randomized initial entity states.
*/
class RandomInitiliazerSystem
{
public:
    /**
    * @brief Applies randomized initialization to one entity.
    * @param entity Entity receiving the initialization.
    */
    static void Init(Entity& entity);
};

} /* ppo::model */
