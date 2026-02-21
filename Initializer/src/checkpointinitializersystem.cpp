/**
* @file Initializer/src/checkpointinitializersystem.cpp
* @brief Implementation for checkpointinitializersystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "checkpointinitializersystem.hpp"
#include "randominitializersystem.hpp"
#include "utils.hpp"
#include "serializer.hpp"

namespace ppo::model {

/**
* @brief Loads or lazily generates checkpoint buffers for one entity.
*
* The first call reads serialized checkpoints when available. If no explicit
* data exists for the requested checkpoint name, the method falls back to the
* random initializer to generate a buffer of initial states.
*
* @param entity Entity that owns the checkpoint component.
*/
void
CheckpointInitializerSystem::Load(Entity& entity)
{
    if (!entity.has<sEntityState, sFlightModelState, sEntityCheckPoint>())
        return;

    entt::registry* registry{ entity.registry() };

    if (registry == nullptr
        || !registry->ctx().contains<sEvaluatorState>()
        || !registry->ctx().contains<sEvaluatorName>()
        || !registry->ctx().contains<sEnvironmentState>())
        return;

    auto& evalState = registry->ctx().get<sEvaluatorState>();
    auto const& evalName = registry->ctx().get<sEvaluatorName>();
    auto const& envState = registry->ctx().get<sEnvironmentState>();

    auto [entityState, flightModelState, checkpoint] = entity.get<sEntityState, sFlightModelState, sEntityCheckPoint>();

    // Load or generate the checkpoint buffer only once per entity.
    if (checkpoint.m_buffers.empty())
    {
        std::filesystem::path filepath{ RESOURCE_PATH };

        filepath = filepath / evalName.m_filename;
        filepath.make_preferred();

        if (!std::filesystem::exists(filepath))
            return;

        auto tbl = ppo::serialization::Serializer::Parse(filepath);

        bool generation{ true };

        if (tbl.contains("Data"))
        {
            auto checkpointsection = tbl["Data"];

            if (checkpointsection.contains(checkpoint.m_name))
            {
                checkpoint.m_buffers = checkpointsection[checkpoint.m_name].get<std::vector<CheckPointBuffer>>();
                generation = false;
            }
        }

        if (generation)
        {
            for (auto i = 0; i < evalState.m_nbtests; ++i)
            {
                CheckPointBuffer  buf{};

                RandomInitiliazerSystem::Init(entity);

                buf[0] = entityState.m_x;
                buf[1] = entityState.m_y;
                buf[2] = entityState.m_rotation;
                buf[3] = entityState.m_speed;
                buf[4] = flightModelState.m_accpsi;
                buf[5] = flightModelState.m_acc;

                checkpoint.m_buffers.push_back(buf);
            }
        }

        checkpoint.m_index = 0;
    }
}

/**
* @brief Applies the current checkpoint sample to one entity.
* @param entity Entity that owns the checkpoint component.
*/
void
CheckpointInitializerSystem::Init(Entity& entity)
{
    if (!entity.has<sEntityState, sFlightModelState, sEntityCheckPoint>())
        return;

    auto [entityState, flightModelState, checkpoint] = entity.get<sEntityState, sFlightModelState, sEntityCheckPoint>();

    if (checkpoint.m_index < checkpoint.m_buffers.size())
    { 
        entityState.m_x           = checkpoint.m_buffers[checkpoint.m_index][0];
        entityState.m_y           = checkpoint.m_buffers[checkpoint.m_index][1];
        entityState.m_rotation    = checkpoint.m_buffers[checkpoint.m_index][2];
        entityState.m_speed       = checkpoint.m_buffers[checkpoint.m_index][3];
        flightModelState.m_accpsi = checkpoint.m_buffers[checkpoint.m_index][4];
        flightModelState.m_acc    = checkpoint.m_buffers[checkpoint.m_index][5];
    }
}

}
