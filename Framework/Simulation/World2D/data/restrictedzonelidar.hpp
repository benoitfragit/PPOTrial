/**
 * @file Framework/Simulation/World2D/data/restrictedzonelidar.hpp
 * @brief Échantillonnage des rayons de pseudo-LiDAR à proximité des zones.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once
// Shared pseudo-LiDAR geometry for stationary zones, friendly safety envelopes and world bounds.
// The calculation and 61-bin layout are unchanged from historical RLMission.
#include "baseentity.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <array>
#include <cstddef>

namespace ppo::model {

inline constexpr std::size_t kRestrictedZoneLidarBins = 61U;

/**
 * @brief Refreshes pseudo-LiDAR components for every living mobile.
 * @param registry ECS registry containing mobiles, zones and world bounds.
 * @param mobileSide Only mobiles owned by this side are refreshed.
 * @param friendlyMobileSafetyRadius Nonlethal safety radius in metres;
 *        zero preserves legacy zone-and-boundary-only sensing.
 * @param boundaryScale Scale of sensed rectangle relative to work bounds;
 *        1 retains legacy behavior, 2 observes the MAPPO hard boundary.
 *
 * Each beam keeps the nearest hit between hostile active restricted zones and
 * the selected rectangular boundary and optional nearby same-side mobiles.
 * No heap allocation occurs in this update.
 */
void UpdateRestrictedZonePseudoLidars(entt::registry& registry,
                                      eEntitySide mobileSide,
                                      float friendlyMobileSafetyRadius = 0.F,
                                      float boundaryScale = 1.F) noexcept;

/**
 * @brief Construit, transforme ou échantillonne les données produites par BuildRestrictedZonePseudoLidar.
 * @details Consulte ou modifie des composants ECS.
 */
void BuildRestrictedZonePseudoLidar(
    entt::registry& registry,
    BaseEntity& ownEntity,
    sEntityState const& ownState,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionRanges,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionDistances) noexcept;

/** @brief Adds nonlethal circular safety envelopes around live teammates. */
void BuildFriendlyMobilePseudoLidar(
    entt::registry& registry,
    BaseEntity& ownEntity,
    sEntityState const& ownState,
    float safetyRadius,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionRanges,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionDistances) noexcept;

/**
 * @brief Construit, transforme ou échantillonne les données produites par BuildZoneBoundaryPseudoLidar.
 */
void BuildZoneBoundaryPseudoLidar(
    sEntityState const& ownState,
    sEnvironmentState const& envState,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionRanges,
    std::array<float, kRestrictedZoneLidarBins>& nearestCollisionDistances,
    float boundaryScale = 1.F) noexcept;

} // namespace ppo::model
