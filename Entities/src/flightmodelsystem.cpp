#include "flightmodelsystem.hpp"

#include "utils.hpp"

namespace ppo::model {

void
FlightModelSystem::Create(BaseEntity& entity, ppo::serialization::json& ar)
{
    if (ar.contains("FlightModel"))
    {
        auto section = ar["FlightModel"];

        // add a flightmodel to this aircraft
        sFlightModelState& flightmodelstate{ entity.add<sFlightModelState>() };

        flightmodelstate.m_maxfuel  = section["MaxFuel"];           
        flightmodelstate.m_maxSpeed = section["MaxSpeed"];          
        flightmodelstate.m_minSpeed = section["MinSpeed"];          
        flightmodelstate.m_maxAccRot= section["MaxRotAcceleration"];
        flightmodelstate.m_maxAcc   = section["MaxAcceleration"];   
    }
}

void 
FlightModelSystem::Init(BaseEntity& entity)
{
    if (!entity.has<sFlightModelState>())
        return;

    auto& [state] = entity.get<sFlightModelState>();

    state.m_fuel        = state.m_maxfuel;
}

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