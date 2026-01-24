#include "flightmodel.hpp"

#include "utils.hpp"

namespace ppo::model {

void 
FlightModel::Init(void)
{
    RandomGenerator generator{};

    generator.registerAttr(m_entityState.m_rotation, 0.F, static_cast<float>(2.0 * PI));

    if (m_state.m_minSpeed < m_state.m_maxSpeed)
        generator.registerAttr(m_entityState.m_speed, m_state.m_minSpeed + 50.F, m_state.m_maxSpeed);

    generator.registerAttr(m_state.m_accpsi, -m_state.m_maxAccRot / 2.F, m_state.m_maxAccRot / 2.F);
    generator.registerAttr(m_state.m_acc, -m_state.m_maxAcc / 2.F, m_state.m_maxAcc / 2.F);

    m_state.m_fuel        = m_state.m_maxfuel;
    m_entityState.m_dx    = m_entityState.m_speed * cos(m_entityState.m_rotation);
    m_entityState.m_dy    = m_entityState.m_speed * sin(m_entityState.m_rotation);

    UpdateDoubleBuffer();
}

void
FlightModel::Run(float dt)
{
    if (!m_entityState.m_destroyed)
    {
        float const prevRotation{ m_entityState.m_rotation };
        m_entityState.m_rotation += m_state.m_accpsi * dt;
        m_entityState.m_speed += m_state.m_acc * dt;

        if (m_state.m_maxSpeed < m_entityState.m_speed)
            m_entityState.m_speed = m_state.m_maxSpeed;

        float const diffRotation{ std::fabs(static_cast<float>(wrapMinusPi2Pi(m_entityState.m_rotation - prevRotation))) };
        float const w{ diffRotation / dt };
        float const consumption{ static_cast<float>(C0
                                                  + CV3 * m_entityState.m_speed * m_entityState.m_speed * m_entityState.m_speed
                                                  + CROT * w * w) };

        m_state.m_fuelconsumption = consumption * dt;

        m_state.m_fuel -= m_state.m_fuelconsumption;
        m_state.m_fuel = std::max(m_state.m_fuel, 0.F);

        if (m_state.m_fuel < std::numeric_limits<float>::epsilon()
            || (2.F * m_entityState.m_speed) < m_state.m_minSpeed)
            m_entityState.m_destroyed = true;
    }

    UpdateDoubleBuffer();
}

void
FlightModel::UpdateDoubleBuffer(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);
    memcpy(&m_buffers[next], &m_state, sizeof(sFlightModelState));
    m_index.store(next, std::memory_order_release);
}

float
FlightModel::GetFuelConsumption(void)                const noexcept
{
    return m_state.m_fuelconsumption;
}

float
FlightModel::GetMaxSpeed(void)    const noexcept
{
    return m_state.m_maxSpeed;
}

float
FlightModel::GetMaxRotSpeed(void)                                     const noexcept
{
    return m_state.m_maxAccRot;
}

float
FlightModel::GetMaxLongitudinalAcc(void)                              const noexcept
{
    return m_state.m_maxAcc;
}

void
FlightModel::SetRotSpeed(float value) noexcept
{
    m_state.m_accpsi = value;
}

void
FlightModel::SetLongitudinalAcc(float value) noexcept
{
    m_state.m_acc = value;
}

void
FlightModel::FillCheckpoint(CheckPointBuffer& buffer) noexcept
{
    RandomGenerator generator{};

    generator.registerAttr(buffer[2], 0.F, static_cast<float>(2.0 * PI));
    if (m_state.m_minSpeed + 50.F < m_state.m_maxSpeed)
        generator.registerAttr(buffer[3], m_state.m_minSpeed + 50.F, m_state.m_maxSpeed);
    generator.registerAttr(buffer[4], -m_state.m_maxAccRot / 2.F, m_state.m_maxAccRot / 2.F);
    generator.registerAttr(buffer[5], -m_state.m_maxAcc / 2.F, m_state.m_maxAcc / 2.F);

    generator.generate();
}

void
FlightModel::SetCheckpoint(CheckPointBuffer const& buffer) noexcept
{
    m_entityState.m_rotation= buffer[2];
    m_entityState.m_speed   = buffer[3];
    m_state.m_accpsi        = buffer[4];
    m_state.m_acc           = buffer[5];
    m_state.m_fuel          = m_state.m_maxfuel;

    UpdateDoubleBuffer();
}

sFlightModelState
FlightModel::GetSnapshot() const noexcept
{
    return m_buffers[m_index.load(std::memory_order_acquire)];
}

}