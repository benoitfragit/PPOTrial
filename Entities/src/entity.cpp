#include "entity.hpp"
#include "utils.hpp"

#include <array>
#include <chrono>
#include <atomic>
#include <thread>
#include <type_traits>
#include <functional>

namespace ppo::model {

Entity::Entity(entt::registry* registry, eEntitySide side, eEntityType type) :
    BaseEntity(registry)
{
    sEntityState& state{ add<sEntityState>() };
    state.m_side = side;
    state.m_type = type;
    state.m_entityId = m_entity;
}

void
Entity::SetInitializerSystem(InitializerSystem system)
{
    if (system == nullptr)
        return;

    p_initializer = system;
}

void
Entity::OnInit(void) noexcept {}

void
Entity::Init(void) noexcept
{
    if (!has<sEntityState>())
        return;

    OnInit();

    auto& [state] = get<sEntityState>();

    state.m_destroyed = false;
    state.m_prevDestroyed = false;
    state.m_destroyedById = entt::null;

    if (p_initializer != nullptr)
        p_initializer(*this);

    state.m_dx = state.m_speed * cos(state.m_rotation);
    state.m_dy = state.m_speed * sin(state.m_rotation);
}

void
Entity::Run(void) noexcept
{
    if (!p_registry
    || !p_registry->ctx().contains<sEnvironmentState>()
    || !has<sEntityState>())
        return;

    auto const& envState{ p_registry->ctx().get<sEnvironmentState>() };

    auto& [state] = get<sEntityState>();

    if (std::numeric_limits<float>::epsilon() < state.m_speed)
    {
        state.m_dx = state.m_speed * cos(state.m_rotation);
        state.m_dy = state.m_speed * sin(state.m_rotation);

        // new position according to rotation and speeds
        state.m_x += state.m_dx * envState.m_dt;
        state.m_y += state.m_dy * envState.m_dt;
    }
}

}