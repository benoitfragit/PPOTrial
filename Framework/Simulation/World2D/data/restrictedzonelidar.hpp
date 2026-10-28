#pragma once
// Shared pseudo-LiDAR geometry for stationary restricted zones and world bounds.
// The calculation and 61-bin layout are unchanged from historical RLMission.
#include "baseentity.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <array>
#include <cstddef>

namespace ppo::model {

inline constexpr std::size_t kRestrictedZoneLidarBins = 61U;

void BuildRestrictedZonePseudoLidar(
    entt::registry& registry,
    BaseEntity& ownEntity,
    sEntityState const& ownState,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionRanges,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionDistances) noexcept;

void BuildZoneBoundaryPseudoLidar(
    sEntityState const& ownState,
    sEnvironmentState const& envState,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionRanges,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionDistances) noexcept;

} // namespace ppo::model
