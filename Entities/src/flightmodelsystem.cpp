/**
* @file Entities/src/flightmodelsystem.cpp
* @brief Implementation for flightmodelsystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "flightmodelsystem.hpp"

#include "utils.hpp"

namespace ppo::model {

/**
* @brief Creates the flight-model component from serialized platform data.
* @param entity Entity that receives the flight-model state.
* @param ar Serialized platform section.
*/
void
FlightModelSystem::Create(BaseEntity& entity, ppo::serialization::json& ar)
{
    if (ar.contains("FlightModel"))
    {
        auto section = ar["FlightModel"];

        // Attach one flight-model component configured from the platform file.
        sFlightModelState& flightmodelstate{ entity.add<sFlightModelState>() };

        flightmodelstate.m_maxfuel  = section["MaxFuel"];           
        flightmodelstate.m_maxSpeed = section["MaxSpeed"];          
        flightmodelstate.m_minSpeed = section["MinSpeed"];          
        flightmodelstate.m_maxAccRot= section["MaxRotAcceleration"];
        flightmodelstate.m_maxAcc   = section["MaxAcceleration"];   
    }
}

/**
* @brief Restores fuel to the platform maximum before an episode starts.
* @param entity Entity that owns the flight-model state.
*/
void 
FlightModelSystem::Init(BaseEntity& entity)
{
    if (!entity.has<sFlightModelState>())
        return;

    auto& [state] = entity.get<sFlightModelState>();

    state.m_fuel        = state.m_maxfuel;
}

/**
* @brief Propagates speed, heading, and fuel consumption for one time step.
*
* The algorithm applies control inputs, clamps the resulting speed, estimates
* fuel burn from cruise and turn-rate terms, and destroys the entity when fuel
* is exhausted or speed drops below the minimum sustainable value.
*
* @param entity Entity that owns the flight-model state.
*/
void
FlightModelSystem::Run(BaseEntity& entity)
{
    entt::registry* registry{ entity.registry() };

    if (registry == nullptr
    || !registry->ctx().contains<sEnvironmentState>()
    || !entity.has<sEntityState, sFlightModelState>())
        return;

    auto const& envState{ registry->ctx().get<sEnvironmentState>() };

    auto& [entityState, state] = entity.get<sEntityState, sFlightModelState>();

    if (entity.has<sSNAControlState>())
    {
        auto const& [snaControlState] = entity.get<sSNAControlState>();
        state.m_accpsi = state.m_maxAccRot  * snaControlState.m_rotSpeedCmd;
        state.m_acc    = state.m_maxAcc     * snaControlState.m_accCmd;
    }

    if (!entityState.m_destroyed)
    {
        float const prevRotation{ entityState.m_rotation };
        entityState.m_rotation += state.m_accpsi * envState.m_dt;
        entityState.m_speed += state.m_acc * envState.m_dt;

        if (state.m_maxSpeed < entityState.m_speed)
            entityState.m_speed = state.m_maxSpeed;

        float const diffRotation{ std::fabs(static_cast<float>(wrapMinusPi2Pi(entityState.m_rotation - prevRotation))) };
        float const w{ diffRotation / envState.m_dt };
        float const consumption{ static_cast<float>(C0
                                                  + CV3 * entityState.m_speed * entityState.m_speed * entityState.m_speed
                                                  + CROT * w * w) };

        state.m_fuelconsumption = consumption * envState.m_dt;

        state.m_fuel -= state.m_fuelconsumption;
        state.m_fuel = std::max(state.m_fuel, 0.F);

        if (state.m_fuel < std::numeric_limits<float>::epsilon()
            || (2.F * entityState.m_speed) < state.m_minSpeed)
            entityState.m_destroyed = true;
    }
}

}
