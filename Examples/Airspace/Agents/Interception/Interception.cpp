/**
* @file Agents/Interception/src/Interception.cpp
* @brief Implementation for Interception.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "Interception.hpp"
#include "airspacecomponents.hpp"
#include "entity.hpp"
#include "utils.hpp"

#include <array>

namespace rl {

/**
* @brief Applies the normalized action vector to SNA control commands.
* @param action Action vector sampled by the policy.
*/
void 
StatesAndActions::SetNextAction(ACTIONS const& action) noexcept
{
    if (!m_entity.has<ppo::model::sSNAControlState>())
        return;

    auto [snaControlState] = m_entity.get<ppo::model::sSNAControlState>();

    snaControlState.m_rotSpeedCmd   = std::clamp(action[0], -1.F, 1.F);
    snaControlState.m_accCmd        = std::clamp(action[1], -1.F, 1.F);
}

/**
* @brief Builds the observation vector consumed by the interception policy.
* @param state Output observation buffer.
*/
void 
StatesAndActions::SetNewState(STATES& state) noexcept
{
    if (!m_entity.has<ppo::model::sEntityState, ppo::model::sSNAState>())
        return;

    auto [entityState, snaState] = m_entity.get<ppo::model::sEntityState, ppo::model::sSNAState>();

    float const dx{ snaState.m_target_x - entityState.m_x };
    float const dy{ snaState.m_target_y - entityState.m_y };
    float const d{ sqrt(dx * dx + dy * dy) };
    float const ux{ dx / (d + 1e-6F) };
    float const uy{ dy / (d + 1e-6F) };
    float az = atan2(uy, ux);
    az = ppo::model::wrapMinusPi2Pi(az);
    az = ppo::model::wrapMinusPi2Pi(az - entityState.m_rotation);

    state[0] = dx / std::max(d, 1e-6F);
    state[1] = dy / std::max(d, 1e-6F);
    state[2] = 1.0F - std::exp(-d / 24000.F);
    state[3] = std::clamp(snaState.m_closingspeed / 300.F, -1.F, 1.F);
    state[4] = cos(az);
    state[5] = sin(az);
}

/**
* @brief Computes the interception reward and episode termination flag.
*
* The reward favors target destruction, positive closing geometry, and reduced
* fuel usage while strongly penalizing ownship destruction and mission timeout.
*
* @return True when the current episode is finished.
*/
bool
SNATrainingAgent::RewardFunction(void) noexcept
{
    bool ret{ false };

    entt::registry* registry{ m_entity.registry() };

    if (registry == nullptr
    || !registry->ctx().contains<ppo::model::sEnvironmentState>()
    ||  m_entity.uniqid() == entt::null
    || !m_entity.has<ppo::model::sEntityState, 
                     ppo::model::sFlightModelState, 
                     ppo::model::sSNAState>())
        return ret;
                   
    auto const& [entityState, flightModelState, snaState] =
        m_entity.get<ppo::model::sEntityState, ppo::model::sFlightModelState, ppo::model::sSNAState>();

    auto const&             envState        { registry->ctx().get<ppo::model::sEnvironmentState>() };
    float                   dt              { envState.m_dt };

    float                   x               { entityState.m_x };
    float                   y               { entityState.m_y };

    float                   fuelconsumption { flightModelState.m_fuelconsumption };
    float                   maxSpeed        { flightModelState.m_maxSpeed };

    float                   closingSpeed    { snaState.m_closingspeed};
    ppo::model::EntityId    targetId        { snaState.m_target };

    float                   LOS             { snaState.m_target_los };
    float                   LOSrate         { snaState.m_target_los_rate};
    float                   pkTarget        { snaState.m_pk };
    bool                    targetDestroyed { 0.95F <= pkTarget };
    float const             vref            { 300.F };
    float const             fuelref         { ppo::model::C0 + ppo::model::CV3 * vref * vref * vref };
    float                   R_fuel          { std::clamp(-fuelconsumption / (fuelref * dt), -1.F, 0.F) };
    float                   R_close         { tanh(closingSpeed / 300.F) };
    float                   R_navpn         { -std::fabs(LOS) };
    float                   R_navpn_act     { 0.F };
    float                   R_navapn_act    { 0.F };
    float                   R_navapn        { -std::fabs(LOSrate) };
    bool                    done            { false };
    float                   pkEntity        { 0.F };

    if (registry->all_of<ppo::model::sEntityState>(snaState.m_target))
    {
        auto const& [targetState] = registry->view<ppo::model::sEntityState>().get(snaState.m_target);

        float const tx { targetState.m_x };
        float const ty { targetState.m_y };
        float const dtx{ targetState.m_dx };
        float const dty{ targetState.m_dy };
        float const d  { static_cast<float>(sqrt(pow(x - tx, 2.F) + pow(y - ty, 2.F))) };

        R_navpn_act = std::exp(-d / 20000.F);
        R_navapn_act = std::exp(-d / 7000.F);
    }

    // Strong terminal reward for destroying the assigned target.
    if (targetDestroyed)
        AddReward(StatesAndActions::RewardType::TargetDestruction, 10.F);

    // Strong penalty for losing the ownship before the objective is met.
    if (entityState.m_destroyed)
    {
        AddReward(StatesAndActions::RewardType::OwnshipDestruction, -10.F);
    }
    else
    {
        AddReward(StatesAndActions::RewardType::TargetInEZ, pkTarget);

        AddReward(StatesAndActions::RewardType::Closing, R_close);

        AddReward(StatesAndActions::RewardType::NavPN, R_navpn * R_navpn_act);

        AddReward(StatesAndActions::RewardType::NavAPN, R_navapn * R_navapn_act);

        AddReward(StatesAndActions::RewardType::Fuel, R_fuel);
    }

    if (IsLastStep() && !targetDestroyed)
        AddReward(StatesAndActions::RewardType::MissionFailure, -5.F);

    done = entityState.m_destroyed || targetDestroyed;

    SetDone(done);

    return done;
}


}  /* rl */
