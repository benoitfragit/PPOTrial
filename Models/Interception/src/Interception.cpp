#include "Interception.hpp"
#include "player.hpp"
#include "utils.hpp"

#include <array>

namespace ppo::interception {

void 
UpdateEnv(ppo::model::Environment& env, ppo::model::IPlayer* player)
{
    if (player == nullptr)
        return;

    Missile* missile{ dynamic_cast<Missile*> (player) };
    if (missile == nullptr)
        return;

    float  pkTarget{ missile->GetKillProbability() };
    if (ppo::model::Player* const target{ dynamic_cast<ppo::model::Player*>(env.GetIPlayerById(missile->GetTarget())) };
        target != nullptr
        && 0.95F < pkTarget)
        target->SetDestroyed(true);
}

float
Missile::StateNorthRange(void) const noexcept
{
    float       dx{ GetTargetNorthPosition() - GetNorthPosition() };
    float const dy{ GetTargetEastPosition() - GetEastPosition() };
    float const d{ sqrt(dx * dx + dy * dy) };
    dx /= std::max(d, 1e-6F);

    return dx;
}

float
Missile::StateEastRange(void) const noexcept
{
    float const dx{ GetTargetNorthPosition() - GetNorthPosition() };
    float       dy{ GetTargetEastPosition() - GetEastPosition() };
    float const d{ sqrt(dx * dx + dy * dy) };
    dy /= std::max(d, 1e-6F);

    return dy;
}

float
Missile::StateRange(void) const noexcept
{
    float const dx{ GetTargetNorthPosition() - GetNorthPosition() };
    float const dy{ GetTargetEastPosition() - GetEastPosition() };

    float d = std::sqrt(dx * dx + dy * dy);

    return 1.0F - std::exp(-d / 24000.F);
}

float
Missile::StateClosingSpeed(void) const noexcept
{
    return std::clamp(GetCLosingSpeed() / 300.F, -1.F, 1.F);
}

float
Missile::StateCosRotation(void) const noexcept
{
    float const dx{ GetTargetNorthPosition() - GetNorthPosition() };
    float const dy{ GetTargetEastPosition() - GetEastPosition() };

    float const d{ std::sqrt(dx * dx + dy * dy) };

    float const ux{ dx / (d + 1e-6F) };
    float const uy{ dy / (d + 1e-6F) };

    // 1. computing the kill probability
    float az = atan2(uy, ux);
    az = ppo::model::wrapMinusPi2Pi(az);
    az = ppo::model::wrapMinusPi2Pi(az - GetRotation());

    return cos(az);
}

float
Missile::StateSinRotation(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    float const dx{ GetTargetNorthPosition() - GetNorthPosition() };
    float const dy{ GetTargetEastPosition() - GetEastPosition() };

    float const d{ std::sqrt(dx * dx + dy * dy) };

    float const ux{ dx / (d + 1e-6F) };
    float const uy{ dy / (d + 1e-6F) };

    // 1. computing the kill probability
    float az = atan2(uy, ux);
    az = ppo::model::wrapMinusPi2Pi(az);
    az = ppo::model::wrapMinusPi2Pi(az - GetRotation());

    return sin(az);
}

void
Missile::ActionSetRotSpeed(float value)                 noexcept
{
    SetRotSpeed(GetMaxRotSpeed() * std::clamp(value, -1.F, 1.F));
}

void
Missile::ActionSetAcceleration(float value) noexcept
{
    if (p_impl == nullptr)
        return;

    SetLongitudinalAcc(GetMaxLongitudinalAcc() * std::clamp(value, -1.F, 1.F));
}

bool
RewardingFunction(ppo::model::Environment& environment, ppo::model::Player* const player, MissileTrainingAgent& pAgent)
{
    bool ret{ false };

    if (player == nullptr)
        return ret;


    float                   dt{ environment.GetTimeStep() };
    float                   fuelconsumption{ player->GetFuelConsumption() };
    float                   closingSpeed{ player->GetCLosingSpeed() };
    ppo::model::PlayerId    targetId{ player->GetTarget() };
    float                   x{ player->GetNorthPosition() };
    float                   y{ player->GetEastPosition() };
    float                   LOS{ player->GetTargetLOS() };
    float                   LOSrate{ player->GetTargetLOSRate() };
    float                   pkTarget{ player->GetKillProbability() };
    float                   maxSpeed{ player->GetMaxSpeed() };
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
    float                   pkPlayer{ 0.F };

    if (auto const target{ dynamic_cast<ppo::model::Player*>(environment.GetIPlayerById(targetId)) };
        target != nullptr)
    {
        float const tx{ target->GetNorthPosition() };
        float const ty{ target->GetEastPosition() };
        float const dtx{ target->GetNorthSpeed() };
        float const dty{ target->GetEastSpeed() };
        float const d{ static_cast<float>(sqrt(pow(x - tx, 2.F) + pow(y - ty, 2.F))) };

        R_navpn_act = std::exp(-d / 20000.F);
        R_navapn_act = std::exp(-d / 7000.F);
        pkPlayer = target->GetKillProbability();
    }

    // big positive reward mission succeed
    if (targetDestroyed)
        pAgent.AddReward(sAgentState::RewardType::TargetDestruction, 10.F);

    // big negative reward on mission failure
    if (player->GetDestroyed())
    {
        pAgent.AddReward(sAgentState::RewardType::OwnshipDestruction, -10.F);
    }
    else
    {
        pAgent.AddReward(sAgentState::RewardType::TargetInEZ, pkTarget);

        // penalty when entering opponent escape zone
        pAgent.AddReward(sAgentState::RewardType::OwnshipInEZ, -pkPlayer);

        pAgent.AddReward(sAgentState::RewardType::Closing, R_close);

        pAgent.AddReward(sAgentState::RewardType::NavPN, R_navpn * R_navpn_act);

        pAgent.AddReward(sAgentState::RewardType::NavAPN, R_navapn * R_navapn_act);

        pAgent.AddReward(sAgentState::RewardType::Fuel, R_fuel);
    }

    if (pAgent.IsLastStep() && !targetDestroyed)
        pAgent.AddReward(sAgentState::RewardType::MissionFailure, -5.F);

    done = player->GetDestroyed() || targetDestroyed;

    pAgent.SetDone(done);

    return done;
}


}