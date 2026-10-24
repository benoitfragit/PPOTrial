/**
* @file Initializer/data/checkpointinitializersystem.hpp
* @brief Declarations for checkpointinitializersystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entity.hpp"

namespace ppo::model {

/**
* @brief Number of scalar values stored per initial-state sample.
*/
static constexpr int const CHECKPOINT_DIMENSION = 6;
/**
* @brief Compact internal representation of a named initial-state sample.
*/
using CheckPointBuffer = std::array<float, CHECKPOINT_DIMENSION>;

/**
* @brief Component storing checkpoint data for one entity.
*/
struct sEntityCheckPoint
{
    /**
    * @brief Logical checkpoint buffer name.
    */
    std::string m_name{ "" };
    /**
    * @brief Current checkpoint index.
    */
    int m_index{ 0 };
    /**
    * @brief Serialized checkpoint state list.
    */
    std::vector<ppo::model::CheckPointBuffer> m_buffers{};
};

/**
* @brief Context component storing the active evaluator file name.
*/
struct sEvaluatorName
{
    /**
    * @brief Evaluator-related file name.
    */
    std::string m_filename{ "" };
};

/**
* @brief Initializer loading an entity state from named JSON presets.
*/
class CheckpointInitializerSystem
{
public:
    /**
    * @brief Loads checkpoint data for one entity.
    * @param entity Entity receiving the checkpoint component.
    */
    static void Load(Entity& entity);
    /**
    * @brief Applies one checkpoint state to one entity.
    * @param entity Entity receiving the initialization.
    */
    static void Init(Entity& entity);
};

} /* ppo::model */
