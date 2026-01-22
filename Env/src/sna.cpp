#include "sna.hpp"
#include "utils.hpp"

namespace ppo::model { 

inline static float
sigmoid(float x)
{
    return 1.F / (1.F + std::exp(-x));
}

void 
SNA::OnRun(void)
{
    Player::OnRun();

    m_pk = 0.F;

    if (p_env == nullptr ||  m_destroyed)
        return;

    if (Player* const target{ p_env->GetPlayerById(m_target) };
        target != nullptr)
    {
        m_target_x = target->GetNorthPosition();
        m_target_y = target->GetEastPosition();
        m_target_dx = target->GetNorthSpeed();
        m_target_dy = target->GetEastSpeed();

        float const dx{ m_target_x - m_x };
        float const dy{ m_target_y - m_y };

        float const d{ std::sqrt(dx * dx + dy * dy) };

        float const ux{ dx / (d + 1e-6F) };
        float const uy{ dy / (d + 1e-6F) };

        // 1. computing the kill probability
        float az = atan2(uy, ux);
        az = ppo::model::wrapMinusPi2Pi(az);
        az = ppo::model::wrapMinusPi2Pi(az - m_rotation);

        float const k0{ 5.F / m_maxShootAngle };
        float const pkangle{ sigmoid(k0 * (m_maxShootAngle - std::fabs(az))) };

        float const dmid{ (m_noescapezone + m_escapezone) / 2.F };
        float const deltad{ m_escapezone - m_noescapezone + 1e-6F };
        float const kd{ 8.F / deltad };
        float const pkrange{ sigmoid(kd * (dmid - d)) };

        m_pk = pkrange * pkangle;

        // 2. computing the closing speed
        float const vrx{ m_target_dx - m_dx };
        float const vry{ m_target_dy - m_dy };

        m_closingspeed = -(vrx * ux + vry * uy);

        // 3. compute new LOS value
        float const LOS{ (ux * vry - uy * vrx) / d };

        // 4. update the LOS rate
        m_target_los_rate = LOS - m_target_los;

        // 5. compute the LOS
        m_target_los = LOS;
    }
    else
    {
        m_target = 0;
        m_target_dx = 0.F;
        m_target_dy = 0.F;
        m_target_los = 0.F;
        m_target_los_rate = 0.F;
        m_target_x = 0.F;
        m_target_y = 0.F;
        m_closingspeed = 0.F;
    }
}

void
SNA::OnInit(void)
{
    m_pk = 0.F;
    m_target_x = 0.F;
    m_target_y = 0.F;
    m_closingspeed = 0.F;
    m_target_los = 0.F;
    m_target_los_rate = 0.F;

    Player::OnInit();
}

void
SNA::UpdateEnv(void) const noexcept
{
    if (p_env == nullptr)
        return;
    
    // very simple effect for now
    float const pkTarget{ GetKillProbability() };
    if (ppo::model::Player* const target{ dynamic_cast<ppo::model::Player*>(p_env->GetPlayerById(GetTarget())) };
        target != nullptr
        && 0.95F < pkTarget)
        target->SetDestroyed(true);
}

float
SNA::GetKillProbability(void) const noexcept
{
    return m_pk;
}

void
SNA::SetTargetPosition(float north, float east) noexcept
{
    m_target_x = north;
    m_target_y = east;
}

void
SNA::SetTarget(PlayerId targetId)       noexcept
{
    m_target = targetId;
}

ppo::model::PlayerId
SNA::GetTarget(void) const noexcept
{
    return m_target;
}

float
SNA::GetCLosingSpeed(void) const noexcept
{
    return m_closingspeed;
}

float
SNA::GetTargetLOS(void) const noexcept
{
    return m_target_los;
}

float
SNA::GetTargetLOSRate(void) const noexcept
{
    return m_target_los_rate;
}

void
SNA::SetTargetSpeed(float north, float east) noexcept
{
    m_target_dx = north;
    m_target_dy = east;
}


}