/**
* @file Entities/src/snasystem.cpp
* @brief Implementation for snasystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "snasystem.hpp"
#include "airspacecomponents.hpp"
#include "utils.hpp"
#include "aircraft.hpp"

namespace ppo::model { 

/**
* @brief Smooth logistic helper used for kill-probability shaping.
* @param x Input value.
* @return Logistic output in the [0, 1] interval.
*/
inline static float
sigmoid(float x)
{
    return 1.F / (1.F + std::exp(-x));
}

/**
* @brief Creates the weapon-system component from serialized platform data.
* @param entity Entity that receives the SNA state.
* @param ar Serialized platform section.
*/
void    
SNASystem::Create(BaseEntity& entity, ppo::serialization::json& ar)
{
    if (ar.contains("SNA"))
    {
        auto section = ar["SNA"];

        // Attach one weapon model to the entity.
        sSNAState& snastate{ entity.add<sSNAState>() };
                             
         snastate.m_maxNoescapeZone = section["MaxNoEscapeZoneRange"];
         snastate.m_maxnbmissiles   = section["MaxMissile"          ];
         snastate.m_maxShootAngle   = section["MaxShootAngle"       ];
    }
}

/**
* @brief Updates target geometry, line-of-sight data, and kill probability.
*
* The algorithm reads the current target state, computes range and relative
* bearing, evaluates range and angle gates, then derives closing speed and LOS
* rate terms used by guidance and reward shaping.
*
* @param entity Entity that owns the SNA state.
*/
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

        // Step 1: compute the instantaneous kill probability.
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

        // Step 2: compute the scalar closing speed along the line of sight.
        float const vrx{ state.m_target_dx - entityState.m_dx };
        float const vry{ state.m_target_dy - entityState.m_dy };

        state.m_closingspeed = -(vrx * ux + vry * uy);

        // Step 3: derive the current line-of-sight value.
        float const LOS{ (ux * vry - uy * vrx) / d };

        // Step 4: update the LOS rate by finite differencing.
        state.m_target_los_rate = LOS - state.m_target_los;

        // Step 5: store the latest LOS value for the next step.
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

/**
* @brief Restores missile inventory and derived range values before an episode.
* @param entity Entity that owns the SNA state.
*/
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

/**
* @brief Evaluates the destruction probability against one candidate target.
*
* The method enforces side and type constraints, then evaluates range and angle
* gates using the same simplified weapon geometry used by the runtime SNA
* system.
*
* @param entity Attacking entity.
* @param otherEntity Candidate target entity.
* @return Estimated kill probability in the [0, 1] interval.
*/
float 
SNASystem::EvaluateDestruction(BaseEntity& entity, BaseEntity& otherEntity)
{
    if (!entity.has<sEntityState, sSNAState>()
    ||  !otherEntity.has<sEntityState>()
    ||  entity.uniqid() == otherEntity.uniqid())
        return 0.F;

    auto [entityState, snaState] = entity.get<sEntityState, sSNAState>();
    auto [otherEntityState] = otherEntity.get<sEntityState>();

    if (otherEntityState.m_side == entityState.m_side
    ||  (entityState.m_type == eEntityType::SAM && otherEntityState.m_type != eEntityType::Aircraft))
        return 0.F;

    float pkrange{ 0.F };
    float pkangle{ 0.F };

    float const dx{ otherEntityState.m_x - entityState.m_x };
    float const dy{ otherEntityState.m_y - entityState.m_y };

    float const d{ std::sqrt(dx * dx + dy * dy) };

    float const ux{ dx / (d + 1e-6F) };
    float const uy{ dy / (d + 1e-6F) };

    float az = atan2(uy, ux);
    az = ppo::model::wrapMinusPi2Pi(az);
    az = ppo::model::wrapMinusPi2Pi(az - entityState.m_rotation);

    float const deltad{ snaState.m_escapezone - snaState.m_noescapezone + 1e-6F };
    pkrange = std::clamp((snaState.m_escapezone - d) / deltad, 0.F, 1.F);

    if (entityState.m_type != eEntityType::SAM)
    {
        if (d <= snaState.m_noescapezone && std::fabs(az) < HALFPI)
            return 1.F;

        pkangle = std::clamp(1.F - std::fabs(az)/(snaState.m_maxShootAngle+1e-6F), 0.F, 1.F);
    }
    else
    {
        pkangle = 1.F;
    }

    return pkrange * pkangle;
}

}
