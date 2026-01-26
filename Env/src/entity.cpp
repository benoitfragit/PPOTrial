#include "entity.hpp"
#include "utils.hpp"

#include <array>
#include <chrono>
#include <atomic>
#include <thread>
#include <type_traits>
#include <functional>

namespace ppo::model {

Entity::Entity( std::string const& name, 
                eEntitySide side, 
                eEntityType type, 
                EntityId entityId, 
                std::shared_ptr<IEnvironment> pEnv)   :
    p_env{ pEnv },
    m_name{ name }
{
    m_state.m_entityId  = entityId ;
    m_state.m_side      = side ;
    m_state.m_type      = type ;
}

Entity::~Entity(void) = default;

void 
Entity::Init(void) noexcept
{
    if (p_env == nullptr)
        return;

    RandomGenerator generator{};

    generator.registerAttr(m_state.m_x, -p_env->GetMaxNorth() / 2.F, p_env->GetMaxNorth() / 2.F);
    generator.registerAttr(m_state.m_y, -p_env->GetMaxEast() / 2.F, p_env->GetMaxEast() / 2.F);

    generator.generate();

    OnInit();
}

void
Entity::UpdateDoubleBuffer(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);
    memcpy(&m_buffers[next], &m_state, sizeof(sEntityState));
    m_index.store(next, std::memory_order_release);
}

void
Entity::OnInit(void)
{
    m_state.m_destroyed     = false;
    m_state.m_dx = m_state.m_speed * cos(m_state.m_rotation);
    m_state.m_dy = m_state.m_speed * sin(m_state.m_rotation);

    UpdateDoubleBuffer();
}

void
Entity::Run(void) noexcept
{
    if (p_env == nullptr)
        return;

    if (std::numeric_limits<float>::epsilon() < m_state.m_speed)
    {
        m_state.m_dx = m_state.m_speed * cos(m_state.m_rotation);
        m_state.m_dy = m_state.m_speed * sin(m_state.m_rotation);

        // new position according to rotation and speeds
        float const dt{ p_env->GetTimeStep() };

        m_state.m_x += m_state.m_dx * dt;
        m_state.m_y += m_state.m_dy * dt;
    }

    UpdateDoubleBuffer();
}

void 
Entity::UpdateEnv(void) const noexcept 
{
}

float
Entity::GetNorthSpeed(void) const noexcept
{
    return m_state.m_dx;
}

float
Entity::GetEastSpeed(void)   const noexcept
{
    return m_state.m_dy;
}

float
Entity::GetRotation(void)                const noexcept
{
    return m_state.m_rotation;
}

void
Entity::SetDestroyed(bool destroyed)                           noexcept 
{
    m_state.m_destroyed = destroyed;
}

std::string const&
Entity::GetName(void) const noexcept
{
    return m_name;
}

bool
Entity::GetDestroyed(void)                               const noexcept
{
    return m_state.m_destroyed;
}

float
Entity::GetNorthPosition(void)               const noexcept
{
    return m_state.m_x;
}

float
Entity::GetEastPosition(void)               const noexcept
{
    return m_state.m_y;
}

EntityId 
Entity::GetEntityId(void) const noexcept
{
    return m_state.m_entityId;
}

sEntityState
Entity::GetSnapshot() const noexcept
{
    return m_buffers[m_index.load(std::memory_order_acquire)]; 
}

CheckPointBuffer
Entity::GetCheckpoint(void) noexcept
{
    CheckPointBuffer buffer{};
    if (p_env == nullptr)
        return buffer;

    RandomGenerator generator{};

    generator.registerAttr(buffer[0], -p_env->GetMaxNorth() / 2.F, p_env->GetMaxNorth()/2.F);
    generator.registerAttr(buffer[1], -p_env->GetMaxEast()  / 2.F, p_env->GetMaxEast() /2.F);
    generator.generate();
   
    return buffer;
}

void  
Entity::SetCheckpoint(CheckPointBuffer const& buffer) noexcept
{
    m_state.m_x         = buffer[0];
    m_state.m_y         = buffer[1];

    OnInit();
}

sEntityState& 
Entity::GetEntityState(void)   noexcept
{ 
    return m_state; 
}


}