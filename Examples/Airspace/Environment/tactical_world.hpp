/**
 * @file Examples/Airspace/Environment/tactical_world.hpp
 * @brief The world row of the Airspace environment DLLs: what the former PPO.exe's tactical minimap
 *        draws, for the Airspace Visualiseur (airspace_visualiser).
 * @details One row (task.json "world" of rl_mission and rl_sna), written after each reset
 * and step from the ECS registry the agent plays in (the training scenario's world, or the
 * evaluation campaign's):
 *
 * - "entities": EntityCapacity entities, EntityValues each, in entity id order: present,
 *   side (0 unknown, 1 hostile, 2 friend), type (1 objective, 2 missile, 3 mobile,
 *   4 restricted zone), north, east, heading (clockwise from north), destroyed, no-escape
 *   radius and half shooting angle (0 without an engagement state), role (0 other,
 *   1 the controlled entity, 2 its primary target);
 * - "scene": work area half extents north and east (sEnvironmentState, as the former PPO.exe's
 *   minimap), entities of the world (may exceed the capacity), the controlled entity's
 *   probability of kill and missiles left.
 */
#pragma once

#include "airspacecomponents.hpp"
#include "components.hpp"

#include "entt/entt.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace ppo::examples::airspace::tactical {

/** @brief Entities of a row: the scenarios hold at most seven. */
constexpr std::size_t EntityCapacity = 16;
/** @brief Values of an entity in "entities". */
constexpr std::size_t EntityValues = 10;
/** @brief Values of "scene". */
constexpr std::size_t SceneValues = 5;
/** @brief The row: "entities" then "scene". */
constexpr std::size_t WorldSize = EntityCapacity * EntityValues + SceneValues;

/** @brief Role of an entity in "entities". */
enum class Role : int
{
    Other = 0,
    Controlled = 1,
    PrimaryTarget = 2
};

/** @brief The controlled entity's primary target (its engagement state's), or null. */
inline entt::entity PrimaryTarget(entt::registry const& registry, entt::entity controlled) noexcept
{
    if (controlled == entt::null || !registry.valid(controlled)
        || !registry.all_of<ppo::model::sSNAState>(controlled))
    {
        return entt::null;
    }
    return registry.get<ppo::model::sSNAState>(controlled).m_target;
}

/** @brief The role of `entity` for the controlled entity. */
inline Role RoleOf(entt::entity entity, entt::entity controlled, entt::entity target) noexcept
{
    if (entity == controlled)
    {
        return Role::Controlled;
    }
    if (entity == target)
    {
        return Role::PrimaryTarget;
    }
    return Role::Other;
}

/** @brief Writes one entity's values. */
inline void WriteEntity(entt::registry const& registry, entt::entity entity, Role role, float* values) noexcept
{
    auto const& state = registry.get<ppo::model::sEntityState>(entity);
    values[0] = 1.F;
    values[1] = static_cast<float>(state.m_side);
    values[2] = static_cast<float>(state.m_type);
    values[3] = state.m_x;
    values[4] = state.m_y;
    values[5] = state.m_rotation;
    values[6] = state.m_destroyed ? 1.F : 0.F;
    values[7] = 0.F;
    values[8] = 0.F;
    if (registry.all_of<ppo::model::sSNAState>(entity))
    {
        auto const& engagement = registry.get<ppo::model::sSNAState>(entity);
        values[7] = engagement.m_noescapezone;
        values[8] = engagement.m_maxShootAngle;
    }
    values[9] = static_cast<float>(role);
}

/** @brief Writes the scene's values. */
inline void WriteScene(entt::registry const& registry, entt::entity controlled, std::size_t entities,
    float* values) noexcept
{
    std::fill(values, values + SceneValues, 0.F);
    if (registry.ctx().contains<ppo::model::sEnvironmentState>())
    {
        auto const& environment = registry.ctx().get<ppo::model::sEnvironmentState>();
        values[0] = environment.m_maxNorth;
        values[1] = environment.m_maxEast;
    }
    values[2] = static_cast<float>(entities);
    if (controlled != entt::null && registry.valid(controlled) && registry.all_of<ppo::model::sSNAState>(controlled))
    {
        auto const& engagement = registry.get<ppo::model::sSNAState>(controlled);
        values[3] = engagement.m_pk;
        values[4] = static_cast<float>(engagement.m_nbmissiles);
    }
}

/**
 * @brief Writes the row of `registry`, seen from `controlled`.
 * @param registry The world the agent plays in.
 * @param controlled The agent's entity (null: no role).
 * @param world WorldSize values; null writes nothing (ppokit-run asks for no world).
 */
inline void WriteWorld(entt::registry const& registry, entt::entity controlled, float* world) noexcept
{
    if (world == nullptr)
    {
        return;
    }
    // Entity id order keeps an entity in its slot from one step to the next (trails).
    std::array<entt::entity, EntityCapacity> slots{};
    std::size_t kept = 0;
    std::size_t entities = 0;
    for (entt::entity const entity : registry.view<ppo::model::sEntityState>())
    {
        ++entities;
        if (kept < EntityCapacity)
        {
            slots[kept++] = entity;
        }
    }
    std::sort(slots.begin(), slots.begin() + static_cast<std::ptrdiff_t>(kept));
    std::fill(world, world + EntityCapacity * EntityValues, 0.F);
    entt::entity const target = PrimaryTarget(registry, controlled);
    for (std::size_t slot = 0; slot < kept; ++slot)
    {
        WriteEntity(registry, slots[slot], RoleOf(slots[slot], controlled, target), world + slot * EntityValues);
    }
    WriteScene(registry, controlled, entities, world + EntityCapacity * EntityValues);
}

} // namespace ppo::examples::airspace::tactical
