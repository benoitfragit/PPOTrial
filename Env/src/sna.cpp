#include "sna.hpp"
#include "utils.hpp"

namespace ppo::model { 

inline static float
sigmoid(float x)
{
    return 1.F / (1.F + std::exp(-x));
}

void    
SNA::SetEntityState(sEntityState const* pState) noexcept
{
    p_entityState = pState;
}

void 
SNA::Run(void)
{
    m_state.m_pk = 0.F;

    if (p_entityState ==nullptr  ||
        p_entityState->m_destroyed)
        return;

    if (0 < m_state.m_target
    &&  m_state.m_target < ULLONG_MAX)
    {
        float const dx{ m_state.m_target_x - p_entityState->m_x };
        float const dy{ m_state.m_target_y - p_entityState->m_y };

        float const d{ std::sqrt(dx * dx + dy * dy) };

        float const ux{ dx / (d + 1e-6F) };
        float const uy{ dy / (d + 1e-6F) };

        // 1. computing the kill probability
        float az = atan2(uy, ux);
        az = ppo::model::wrapMinusPi2Pi(az);
        az = ppo::model::wrapMinusPi2Pi(az - p_entityState->m_rotation);

        float const k0{ 5.F / m_state.m_maxShootAngle };
        float const pkangle{ sigmoid(k0 * (m_state.m_maxShootAngle - std::fabs(az))) };

        float const dmid{ (m_state.m_noescapezone + m_state.m_escapezone) / 2.F };
        float const deltad{ m_state.m_escapezone - m_state.m_noescapezone + 1e-6F };
        float const kd{ 8.F / deltad };
        float const pkrange{ sigmoid(kd * (dmid - d)) };

        m_state.m_pk = pkrange * pkangle;

        // 2. computing the closing speed
        float const vrx{ m_state.m_target_dx - p_entityState->m_dx };
        float const vry{ m_state.m_target_dy - p_entityState->m_dy };

        m_state.m_closingspeed = -(vrx * ux + vry * uy);

        // 3. compute new LOS value
        float const LOS{ (ux * vry - uy * vrx) / d };

        // 4. update the LOS rate
        m_state.m_target_los_rate = LOS - m_state.m_target_los;

        // 5. compute the LOS
        m_state.m_target_los = LOS;
    }
    else
    {
        m_state.m_target = 0;
        m_state.m_target_dx = 0.F;
        m_state.m_target_dy = 0.F;
        m_state.m_target_los = 0.F;
        m_state.m_target_los_rate = 0.F;
        m_state.m_target_x = 0.F;
        m_state.m_target_y = 0.F;
        m_state.m_closingspeed = 0.F;
    }

    UpdateDoubleBuffer();
}

void
SNA::Init(void)
{
    m_state.m_escapezone    = 2.F * m_state.m_maxNoescapeZone;
    m_state.m_noescapezone  = m_state.m_maxNoescapeZone;
    m_state.m_nbmissiles    = m_state.m_maxnbmissiles;
    m_state.m_pk = 0.F;
    m_state.m_target_x = 0.F;
    m_state.m_target_y = 0.F;
    m_state.m_closingspeed = 0.F;
    m_state.m_target_los = 0.F;
    m_state.m_target_los_rate = 0.F;
    m_state.m_accCmd= 0.F ;
    m_state.m_rotSpeedCmd= 0.F ;

    UpdateDoubleBuffer();
}

float
SNA::GetKillProbability(void) const noexcept
{
    return m_state.m_pk;
}

void
SNA::SetTargetPosition(float north, float east) noexcept
{
    m_state.m_target_x = north;
    m_state.m_target_y = east;
}

void
SNA::SetTarget(EntityId targetId)       noexcept
{
    m_state.m_target = targetId;
}

ppo::model::EntityId
SNA::GetTarget(void) const noexcept
{
    return m_state.m_target;
}

float
SNA::GetCLosingSpeed(void) const noexcept
{
    return m_state.m_closingspeed;
}

float
SNA::GetTargetLOS(void) const noexcept
{
    return m_state.m_target_los;
}

float 
SNA::GetTargetNorthPosition(void)                    const noexcept
{
    return m_state.m_target_x;
}

float 
SNA::GetTargetEastPosition(void)                    const noexcept
{
    return m_state.m_target_y;
}

float
SNA::GetTargetLOSRate(void) const noexcept
{
    return m_state.m_target_los_rate;
}

void
SNA::SetTargetSpeed(float north, float east) noexcept
{
    m_state.m_target_dx = north;
    m_state.m_target_dy = east;
}

void
SNA::UpdateDoubleBuffer(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);
    memcpy(&m_buffers[next], &m_state, sizeof(sSNAState));
    m_index.store(next, std::memory_order_release);
}

sSNAState 
SNA::GetSnapshot(void) const noexcept
{
    return m_buffers[m_index.load(std::memory_order_acquire)];
}


float
SNA::GetAccCmd(void)                    const noexcept
{
    return m_state.m_accCmd;
}

float
SNA::GetRotSpeedCmd(void)                    const noexcept
{
    return m_state.m_rotSpeedCmd;
}

}