/**
* @file Entities/src/snacontrolsystem.cpp
* @brief Implementation for snacontrolsystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "snacontrolsystem.hpp"

#include "utils.hpp"

namespace ppo::model {

/**
* @brief Creates the control component from serialized platform data.
* @param entity Entity that receives the control state.
* @param ar Serialized platform section.
*/
void 
SNAControlSystem::Create(BaseEntity& entity, ppo::serialization::json& ar)
{
    if (ar.contains("SNAControl"))
    {
        auto section = ar["SNAControl"];

        sSNAControlState& state{ entity.add<sSNAControlState>() };

        state.m_accCmd = 0.F;
        state.m_rotSpeedCmd = 0.F;

        bool available = section["Available"];
    }
}

/**
* @brief Resets control commands to a neutral state.
* @param entity Entity that owns the control state.
*/
void
SNAControlSystem::Init(BaseEntity& entity)
{
    if (!entity.has<sSNAControlState>())
        return;

    auto [snaControlState] = entity.get<sSNAControlState>();

    snaControlState.m_accCmd = 0.F;
    snaControlState.m_rotSpeedCmd = 0.F;
}

/**
* @brief Converts normalized commands into flight-model accelerations.
* @param entity Entity that owns the control and flight-model states.
*/
void
SNAControlSystem::Run(BaseEntity& entity)
{
    if (!entity.has<sFlightModelState, sSNAControlState>())
        return;

    auto& [flightModelState, snaControlState] = entity.get<sFlightModelState, sSNAControlState>();
    flightModelState.m_accpsi = flightModelState.m_maxAccRot * snaControlState.m_rotSpeedCmd;
    flightModelState.m_acc    = flightModelState.m_maxAcc    * snaControlState.m_accCmd;
}

}
