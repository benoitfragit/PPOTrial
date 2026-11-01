/**
 * @file tactical_role.hpp
 * @brief Allocation-free mapping from canonical blue names to display roles.
 */
#pragma once

#include <cstddef>
#include <string_view>

namespace ppo::renderer {

/** @brief Role marker displayed by the tactical renderer. */
enum class TacticalRole
{
    None,
    Leader,
    Teammate
};

/**
 * @brief Resolves a canonical `blue_1..blue_9` name against episode roles.
 * @param entityName Scenario-level entity name.
 * @param activeBlueCount Number of active blue mobiles.
 * @param leaderPhysicalIndex Zero-based physical leader index.
 * @return Leader, teammate, or none for an invalid/non-blue entity.
 */
constexpr TacticalRole ResolveTacticalRole(
    std::string_view const entityName, std::size_t const activeBlueCount,
    std::size_t const leaderPhysicalIndex) noexcept
{
    constexpr std::string_view prefix{"blue_"};
    if (activeBlueCount == 0U || activeBlueCount > 9U
        || leaderPhysicalIndex >= activeBlueCount
        || entityName.size() != prefix.size() + 1U
        || entityName.substr(0U, prefix.size()) != prefix)
    {
        return TacticalRole::None;
    }

    char const digit = entityName.back();
    if (digit < '1' || digit > '9')
        return TacticalRole::None;

    std::size_t const physicalIndex = static_cast<std::size_t>(digit - '1');
    if (physicalIndex >= activeBlueCount)
        return TacticalRole::None;
    return physicalIndex == leaderPhysicalIndex
        ? TacticalRole::Leader : TacticalRole::Teammate;
}

/**
 * @brief Returns the single-character badge associated with a tactical role.
 * @param role Resolved display role.
 * @return `L`, `E`, or an empty string for no role.
 */
constexpr char const* TacticalRoleMarker(TacticalRole const role) noexcept
{
    return role == TacticalRole::Leader ? "L"
        : role == TacticalRole::Teammate ? "E" : "";
}

} // namespace ppo::renderer
