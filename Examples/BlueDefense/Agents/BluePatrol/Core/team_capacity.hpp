/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Core/team_capacity.hpp
 * @brief Interface ou implémentation du composant team capacity.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include <cstddef>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Static capacities of the generalized BlueDefense team domain.
 *
 * Runtime team sizes are validated against this single contract before they
 * are used to index fixed-capacity storage required by RLTools.
 */
struct TeamCapacity final
{
    /** @brief Minimum number of active blue mobiles in an episode. */
    static constexpr std::size_t MinimumBlueMobiles = 1U;
    /** @brief Fixed storage capacity for active blue mobiles. */
    static constexpr std::size_t MaximumBlueMobiles = 9U;
    /** @brief Minimum number of active red mobiles in an episode. */
    static constexpr std::size_t MinimumRedMobiles = 0U;
    /** @brief Fixed storage capacity for active red mobiles. */
    static constexpr std::size_t MaximumRedMobiles = 10U;
};

/**
 * @brief Active mobile counts for one episode.
 *
 * Zones and the fixed blue objective are intentionally excluded: they do not
 * consume blue or red mobile capacity.
 */
struct ActiveTeamCounts final
{
    /** @brief Number of active blue mobiles, excluding the objective. */
    std::size_t blueMobiles{};
    /** @brief Number of active red mobiles. */
    std::size_t redMobiles{};
};

/**
 * @brief Checks active counts against the centralized static capacities.
 * @param counts Episode-level active mobile counts.
 * @return True when `1 <= blueMobiles <= 9` and
 * `0 <= redMobiles <= 10`.
 */
constexpr bool Valid(ActiveTeamCounts const counts) noexcept
{
    return counts.blueMobiles >= TeamCapacity::MinimumBlueMobiles
        && counts.blueMobiles <= TeamCapacity::MaximumBlueMobiles
        && counts.redMobiles >= TeamCapacity::MinimumRedMobiles
        && counts.redMobiles <= TeamCapacity::MaximumRedMobiles;
}

} // namespace ppo::examples::bluedefense::defenders
