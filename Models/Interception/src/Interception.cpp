#include "Interception.hpp"
#include "entity.hpp"
#include "utils.hpp"

#include <array>

namespace ppo::interception {

float
RLSNA::StateNorthRange(void) const noexcept
{
    if (p_aircraft == nullptr)
        return 0.F;

    auto const& entityState{ p_aircraft->GetEntityState() };

    float       dx{ GetTargetNorthPosition() - entityState.m_x};
    float const dy{ GetTargetEastPosition() - entityState.m_y};
    float const d{ sqrt(dx * dx + dy * dy) };
    dx /= std::max(d, 1e-6F);

    return dx;
}

float
RLSNA::StateEastRange(void) const noexcept
{
    if (p_aircraft == nullptr)
        return 0.F;

    auto const& entityState{ p_aircraft->GetEntityState() };

    float const dx{ GetTargetNorthPosition() - entityState.m_x };
    float       dy{ GetTargetEastPosition() - entityState.m_y };
    float const d{ sqrt(dx * dx + dy * dy) };
    dy /= std::max(d, 1e-6F);

    return dy;
}

float
RLSNA::StateRange(void) const noexcept
{
    if (p_aircraft == nullptr)
        return 0.F;

    auto const& entityState{ p_aircraft->GetEntityState() };

    float const dx{ GetTargetNorthPosition() - entityState.m_x };
    float const dy{ GetTargetEastPosition() - entityState.m_y };

    float d = std::sqrt(dx * dx + dy * dy);

    return 1.0F - std::exp(-d / 24000.F);
}

float
RLSNA::StateClosingSpeed(void) const noexcept
{
    return std::clamp(GetCLosingSpeed() / 300.F, -1.F, 1.F);
}

float
RLSNA::StateCosRotation(void) const noexcept
{
    if (p_aircraft == nullptr)
        return 0.F;

    auto const& entityState{ p_aircraft->GetEntityState() };

    float const dx{ GetTargetNorthPosition() - entityState.m_x };
    float const dy{ GetTargetEastPosition() - entityState.m_y };

    float const d{ std::sqrt(dx * dx + dy * dy) };

    float const ux{ dx / (d + 1e-6F) };
    float const uy{ dy / (d + 1e-6F) };

    // 1. computing the kill probability
    float az = atan2(uy, ux);
    az = ppo::model::wrapMinusPi2Pi(az);
    az = ppo::model::wrapMinusPi2Pi(az - entityState.m_rotation);

    return cos(az);
}

float
RLSNA::StateSinRotation(void) const noexcept
{
    if (p_aircraft == nullptr)
        return 0.F;

    auto const& entityState{ p_aircraft->GetEntityState() };

    float const dx{ GetTargetNorthPosition() - entityState.m_x };
    float const dy{ GetTargetEastPosition() - entityState.m_y };

    float const d{ std::sqrt(dx * dx + dy * dy) };

    float const ux{ dx / (d + 1e-6F) };
    float const uy{ dy / (d + 1e-6F) };

    // 1. computing the kill probability
    float az = atan2(uy, ux);
    az = ppo::model::wrapMinusPi2Pi(az);
    az = ppo::model::wrapMinusPi2Pi(az - entityState.m_rotation);

    return sin(az);
}

void
RLSNA::ActionSetRotSpeed(float value)                 noexcept
{
    m_state.m_rotSpeedCmd = std::clamp(value, -1.F, 1.F);
}

void
RLSNA::ActionSetAcceleration(float value) noexcept
{
    m_state.m_accCmd = std::clamp(value, -1.F, 1.F);
}

bool
SNATrainingAgent::RewardFunction(void) noexcept
{
    bool ret{ false };

    if (p_model == nullptr
    ||  p_environment == nullptr)
        return ret;
                   
    ppo::model::Aircraft const* const aircraft{ p_model->GetAircraft() };
    if (aircraft == nullptr)
        return ret;

    ppo::model::FlightModel const& flightModel{ aircraft->GetFlightModel() };

    float                   dt{ p_environment->GetTimeStep() };
    float                   fuelconsumption{ flightModel.GetFuelConsumption() };
    float                   closingSpeed{ p_model->GetCLosingSpeed() };
    ppo::model::EntityId    targetId{ p_model->GetTarget() };
    float                   x{ aircraft->GetNorthPosition() };
    float                   y{ aircraft->GetEastPosition() };
    float                   LOS{ p_model->GetTargetLOS() };
    float                   LOSrate{ p_model->GetTargetLOSRate() };
    float                   pkTarget{ p_model->GetKillProbability() };
    float                   maxSpeed{ flightModel.GetMaxSpeed() };
    bool                    targetDestroyed{ 0.95F <= pkTarget };
    float const             vref{ 300.F };
    float const             fuelref{ ppo::model::C0 + ppo::model::CV3 * vref * vref * vref };
    float                   R_fuel{ std::clamp(-fuelconsumption / (fuelref * dt), -1.F, 0.F) };
    float                   R_close{ tanh(closingSpeed / 300.F) };
    float                   R_navpn{ -std::fabs(LOS) };
    float                   R_navpn_act{ 0.F };
    float                   R_navapn_act{ 0.F };
    float                   R_navapn{ -std::fabs(LOSrate) };
    bool                    done{ false };
    float                   pkEntity{ 0.F };

    if (auto const target{ dynamic_cast<ppo::model::Entity*>(p_environment->GetIEntityById(targetId)) };
        target != nullptr)
    {
        float const tx{ target->GetNorthPosition() };
        float const ty{ target->GetEastPosition() };
        float const dtx{ target->GetNorthSpeed() };
        float const dty{ target->GetEastSpeed() };
        float const d{ static_cast<float>(sqrt(pow(x - tx, 2.F) + pow(y - ty, 2.F))) };

        R_navpn_act = std::exp(-d / 20000.F);
        R_navapn_act = std::exp(-d / 7000.F);
    }

    // big positive reward mission succeed
    if (targetDestroyed)
        AddReward(RLSNA::RewardType::TargetDestruction, 10.F);

    // big negative reward on mission failure
    if (aircraft->GetDestroyed())
    {
        AddReward(RLSNA::RewardType::OwnshipDestruction, -10.F);
    }
    else
    {
        AddReward(RLSNA::RewardType::TargetInEZ, pkTarget);

        AddReward(RLSNA::RewardType::Closing, R_close);

        AddReward(RLSNA::RewardType::NavPN, R_navpn * R_navpn_act);

        AddReward(RLSNA::RewardType::NavAPN, R_navapn * R_navapn_act);

        AddReward(RLSNA::RewardType::Fuel, R_fuel);
    }

    if (IsLastStep() && !targetDestroyed)
        AddReward(RLSNA::RewardType::MissionFailure, -5.F);

    done = aircraft->GetDestroyed() || targetDestroyed;

    SetDone(done);

    return done;
}


}