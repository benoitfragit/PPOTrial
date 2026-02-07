#include "snasystem.hpp"
#include "utils.hpp"
#include "aircraft.hpp"

namespace ppo::model { 

inline static float
sigmoid(float x)
{
    return 1.F / (1.F + std::exp(-x));
}

void    
SNASystem::Create(BaseEntity& entity, ppo::serialization::json& ar)
{
    if (ar.contains("SNA"))
    {
        auto section = ar["SNA"];

        // add a SNA
        sSNAState& snastate{ entity.add<sSNAState>() };
                             
         snastate.m_maxNoescapeZone = section["MaxNoEscapeZoneRange"];
         snastate.m_maxnbmissiles   = section["MaxMissile"          ];
         snastate.m_maxShootAngle   = section["MaxShootAngle"       ];
    }
}

void 
SNASystem::Run(BaseEntity& entity)
{
    if (!entity.has<sEntityState, sSNAState>())
        return;

    auto& [entityState, state] = entity.get<sEntityState, sSNAState>();

    state.m_target_x = 0.F;
    state.m_target_y = 0.F;
    state.m_target_dx = 0.F;
    state.m_target_dy = 0.F;

    state.m_pk = 0.F;

    if (!entityState.m_destroyed
    &&  state.m_target != entt::null)
    {
        if (entt::registry* registry{ entity.registry() };
            registry != nullptr
        &&  registry->any_of<sEntityState>(state.m_target))
        {
            auto const& [targetState] = registry->view<sEntityState>().get(state.m_target);

            state.m_target_x  = targetState.m_x;
            state.m_target_y  = targetState.m_y;
            state.m_target_dx = targetState.m_dx;
            state.m_target_dy = targetState.m_dy;
        }

        float const dx{ state.m_target_x - entityState.m_x };
        float const dy{ state.m_target_y - entityState.m_y };

        float const d{ std::sqrt(dx * dx + dy * dy) };

        float const ux{ dx / (d + 1e-6F) };
        float const uy{ dy / (d + 1e-6F) };

        // 1. computing the kill probability
        float az = atan2(uy, ux);
        az = ppo::model::wrapMinusPi2Pi(az);
        az = ppo::model::wrapMinusPi2Pi(az - entityState.m_rotation);

        float const k0{ 5.F / state.m_maxShootAngle };
        float const pkangle{ sigmoid(k0 * (state.m_maxShootAngle - std::fabs(az))) };

        float const dmid{ (state.m_noescapezone + state.m_escapezone) / 2.F };
        float const deltad{ state.m_escapezone - state.m_noescapezone + 1e-6F };
        float const kd{ 8.F / deltad };
        float const pkrange{ sigmoid(kd * (dmid - d)) };

        state.m_pk = pkrange * pkangle;

        // 2. computing the closing speed
        float const vrx{ state.m_target_dx - entityState.m_dx };
        float const vry{ state.m_target_dy - entityState.m_dy };

        state.m_closingspeed = -(vrx * ux + vry * uy);

        // 3. compute new LOS value
        float const LOS{ (ux * vry - uy * vrx) / d };

        // 4. update the LOS rate
        state.m_target_los_rate = LOS - state.m_target_los;

        // 5. compute the LOS
        state.m_target_los = LOS;
    }
    else
    {
        //state.m_target = entt::null;
        state.m_target_dx = 0.F;
        state.m_target_dy = 0.F;
        state.m_target_los = 0.F;
        state.m_target_los_rate = 0.F;
        state.m_target_x = 0.F;
        state.m_target_y = 0.F;
        state.m_closingspeed = 0.F;
    }
}

void
SNASystem::Init(BaseEntity& entity)
{
    if (!entity.has<sSNAState>())
        return;

    auto& [state] = entity.get<sSNAState>();

   state.m_escapezone    = 2.F * state.m_maxNoescapeZone;
   state.m_noescapezone  = state.m_maxNoescapeZone;
   state.m_nbmissiles    = state.m_maxnbmissiles;
   state.m_pk = 0.F;
   state.m_target_x = 0.F;
   state.m_target_y = 0.F;
   state.m_closingspeed = 0.F;
   state.m_target_los = 0.F;
   state.m_target_los_rate = 0.F;
}

}