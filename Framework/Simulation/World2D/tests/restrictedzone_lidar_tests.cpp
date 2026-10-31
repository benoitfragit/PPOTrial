/**
 * @file Framework/Simulation/World2D/tests/restrictedzone_lidar_tests.cpp
 * @brief Tests et assertions déterministes pour restrictedzone lidar tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "restrictedzonelidar.hpp"
#include "restrictedzonegeometry.hpp"
#include "snasystem.hpp"
#include "entitytypes.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace {
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Consulte ou modifie des composants ECS. Peut signaler un état invalide par exception.
 */
void RequireNear(float actual, float expected, char const* message)
{
    if (std::fabs(actual - expected) > 1.0e-4F)
        throw std::runtime_error(message);
}
}

int main()
{
    using namespace ppo::model;
    static_assert(kRestrictedZoneLidarBins == 61U,
                  "Legacy RLMission uses exactly 61 pseudo-LiDAR beams");
    static_assert(static_cast<unsigned>(eEntityType::Mobile) == 3U);
    static_assert(static_cast<unsigned>(eEntityType::Aircraft) == 3U);
    static_assert(static_cast<unsigned>(eEntityType::RestrictedZone) == 4U);
    static_assert(static_cast<unsigned>(eEntityType::SAM) == 4U);

    entt::registry registry;
    auto const ownId = registry.create();
    auto& ownState = registry.emplace<sEntityState>(ownId);
    ownState.m_entityId = ownId;
    ownState.m_side = eEntitySide::Friend;
    ownState.m_type = eEntityType::Mobile;
    ownState.m_rotation = 0.F;

    auto const zoneId = registry.create();
    auto& zoneState = registry.emplace<sEntityState>(zoneId);
    zoneState.m_entityId = zoneId;
    zoneState.m_side = eEntitySide::Hostile;
    zoneState.m_type = eEntityType::RestrictedZone;
    zoneState.m_x = 100.F;
    zoneState.m_y = 0.F;
    auto& zoneSense = registry.emplace<sSNAState>(zoneId);
    zoneSense.m_noescapezone = 20.F;

    BaseEntity own{&registry, ownId};
    std::array<float, kRestrictedZoneLidarBins> ranges{};
    std::array<float, kRestrictedZoneLidarBins> distances{};
    BuildRestrictedZonePseudoLidar(registry, own, ownState, ranges, distances);
    RequireNear(distances[30], 78.F, "Near intersection changed");
    RequireNear(ranges[30], 78.F / (78.F + 22.F), "Legacy normalization changed");
    RequireNear(RestrictedZoneWarningRadius(20.F), 22.F,
                "warning radius no longer differs explicitly from lethal radius");
    if (IsInsideRestrictedZone({0.F, 0.F}, {30.F, 0.F}, 20.F)
        || !IsInsideRestrictedZone({10.F, 0.F}, {30.F, 0.F}, 20.F)
        || !CrossesRestrictedZone({0.F, 0.F}, {60.F, 0.F},
                                  {30.F, 0.F}, 5.F))
        throw std::runtime_error("pure restricted-zone geometry changed");

    auto& environment = registry.ctx().emplace<sEnvironmentState>();
    environment.m_maxNorth = 400.F;
    environment.m_maxEast = 400.F;
    auto& published = registry.emplace<sPseudoLidarState>(ownId);
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend);
    RequireNear(published.m_nearestCollisionDistances[30], 78.F,
                "ECS pseudo-lidar publication changed");

    zoneState.m_side = eEntitySide::Friend;
    BuildRestrictedZonePseudoLidar(registry, own, ownState, ranges, distances);
    RequireNear(ranges[30], 1.F, "Friendly restricted zone must be ignored");
    RequireNear(distances[30], 0.F, "Ignored zone must have no measured distance");

    zoneState.m_side = eEntitySide::Hostile;
    zoneState.m_destroyed = true;
    BuildRestrictedZonePseudoLidar(registry, own, ownState, ranges, distances);
    RequireNear(ranges[30], 1.F, "Destroyed restricted zone must be ignored");

    zoneState.m_destroyed = false;
    ownState.m_x = 100.F;
    BuildRestrictedZonePseudoLidar(registry, own, ownState, ranges, distances);
    RequireNear(ranges[30], 0.F, "Inside restricted zone must read zero");
    RequireNear(distances[30], 0.F, "Inside restricted zone distance must be zero");

    ownState.m_x = 0.F;
    sEnvironmentState bounds{};
    bounds.m_maxNorth = 400.F;
    bounds.m_maxEast = 400.F;
    ranges.fill(1.F);
    distances.fill(0.F);
    BuildZoneBoundaryPseudoLidar(ownState, bounds, ranges, distances);
    RequireNear(distances[30], 200.F, "Forward world boundary hit changed");
    auto const halfDiagonal = std::sqrt(200.F * 200.F + 200.F * 200.F);
    RequireNear(ranges[30], 200.F / (200.F + halfDiagonal),
                "World boundary normalization changed");

    // Historical sensing is unchanged at scale=1. MAPPO senses the
    // outer failure boundary instead, including its native normalization.
    ranges.fill(1.F);
    distances.fill(0.F);
    BuildZoneBoundaryPseudoLidar(ownState, bounds, ranges, distances, 2.F);
    RequireNear(distances[30], 400.F,
                "MAPPO must detect the hard North boundary");
    RequireNear(ranges[30], 400.F / (400.F + 2.F * halfDiagonal),
                "Hard-boundary normalization changed");
    ownState.m_rotation = 1.5707963267948966F;
    ranges.fill(1.F);
    distances.fill(0.F);
    BuildZoneBoundaryPseudoLidar(ownState, bounds, ranges, distances, 2.F);
    RequireNear(distances[30], 400.F,
                "MAPPO must detect the hard East boundary");
    ownState.m_rotation = 0.F;
    ranges.fill(1.F);
    distances.fill(0.F);
    BuildZoneBoundaryPseudoLidar(ownState, bounds, ranges, distances, 0.F);
    RequireNear(distances[30], 0.F,
                "Invalid boundary scale must not emit an intersection");
    RequireNear(ranges[30], 1.F,
                "Invalid boundary scale must retain empty rays");
    // Preserve angle-sensitive mobile elimination and angle-free stationary
    // restricted zone elimination, including historical side/type gates.
    auto const victimId = registry.create();
    auto& victim = registry.emplace<sEntityState>(victimId);
    victim.m_entityId = victimId;
    victim.m_side = eEntitySide::Hostile;
    victim.m_type = eEntityType::Mobile;
    victim.m_x = 5.F;
    victim.m_y = 0.F;
    ownState.m_x = 0.F;
    ownState.m_y = 0.F;
    ownState.m_rotation = 0.F;
    ownState.m_type = eEntityType::Mobile;
    auto& ownSense = registry.emplace<sSNAState>(ownId);
    ownSense.m_noescapezone = 10.F;
    ownSense.m_escapezone = 20.F;
    ownSense.m_maxShootAngle = 1.F;
    BaseEntity victimHandle{&registry, victimId};
    RequireNear(SNASystem::EvaluateDestruction(own, victimHandle), 1.F,
                "Mobile frontal elimination changed");
    victim.m_x = -5.F;
    RequireNear(SNASystem::EvaluateDestruction(own, victimHandle), 0.F,
                "Mobile rear elimination must remain blocked by angle");
    victim.m_side = eEntitySide::Friend;
    RequireNear(SNASystem::EvaluateDestruction(own, victimHandle), 0.F,
                "Same-team elimination must remain blocked");
    victim.m_side = eEntitySide::Hostile;
    zoneState.m_side = eEntitySide::Friend;

    zoneSense.m_noescapezone = 10.F;
    zoneSense.m_escapezone = 20.F;
    zoneSense.m_maxShootAngle = 1.F;
    zoneState.m_type = eEntityType::RestrictedZone;
    zoneState.m_x = 0.F;
    zoneState.m_y = 0.F;
    zoneState.m_rotation = 3.F;
    BaseEntity zoneHandle{&registry, zoneId};
    RequireNear(SNASystem::EvaluateDestruction(zoneHandle, victimHandle), 1.F,
                "Stationary restricted zone must remain independent of angle");
    victim.m_type = eEntityType::RestrictedZone;
    RequireNear(SNASystem::EvaluateDestruction(zoneHandle, victimHandle), 0.F,
                "Restricted zones must not eliminate other restricted zones");


    // Dynamic sensing detects nearest teammates without changing physical
    // collisions or the stationary-zone sensor contract.
    zoneState.m_destroyed = true;
    auto const nearId = registry.create();
    auto& nearBlue = registry.emplace<sEntityState>(nearId);
    nearBlue.m_entityId = nearId;
    nearBlue.m_side = eEntitySide::Friend;
    nearBlue.m_type = eEntityType::Mobile;
    nearBlue.m_x = 100.F;
    auto const farId = registry.create();
    auto& farBlue = registry.emplace<sEntityState>(farId);
    farBlue.m_entityId = farId;
    farBlue.m_side = eEntitySide::Friend;
    farBlue.m_type = eEntityType::Mobile;
    farBlue.m_x = 160.F;
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend, 20.F);
    RequireNear(published.m_nearestCollisionDistances[30], 80.F,
                "Nearest friendly envelope must win");
    RequireNear(published.m_nearestCollisionRanges[30], 0.8F,
                "Friendly envelope normalization changed");

    nearBlue.m_destroyed = true;
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend, 20.F);
    RequireNear(published.m_nearestCollisionDistances[30], 140.F,
                "Destroyed friendlies must be excluded");

    farBlue.m_side = eEntitySide::Hostile;
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend, 20.F);
    RequireNear(published.m_nearestCollisionDistances[30], 200.F,
                "Hostiles must not count as friendly safety obstacles");
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend, 20.F, 2.F);
    RequireNear(published.m_nearestCollisionDistances[30], 400.F,
                "Blue ECS LiDAR must publish outer-boundary range");

    farBlue.m_side = eEntitySide::Friend;
    farBlue.m_x = 0.F;
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend, 20.F);
    RequireNear(published.m_nearestCollisionDistances[30], 20.F,
                "Overlap must retain a finite directional exit range");
    RequireNear(published.m_nearestCollisionDistances[0], 20.F,
                "Rear obstacles must be sensed");

    farBlue.m_destroyed = true;
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend);
    RequireNear(published.m_nearestCollisionDistances[30], 200.F,
                "Disabled dynamic sensing must retain world-boundary hits");

    // A danger-zone penetration must stay saturated when both the world
    // boundary and a teammate safety envelope overlap the same ray.
    zoneState.m_side = eEntitySide::Hostile;
    zoneState.m_destroyed = false;
    zoneState.m_x = ownState.m_x;
    zoneState.m_y = ownState.m_y;
    zoneSense.m_noescapezone = 10.F;
    farBlue.m_destroyed = false;
    farBlue.m_x = ownState.m_x + 30.F;
    farBlue.m_y = ownState.m_y;
    UpdateRestrictedZonePseudoLidars(registry, eEntitySide::Friend, 20.F);
    for (std::size_t ray = 0U; ray < kRestrictedZoneLidarBins; ++ray)
    {
        RequireNear(published.m_nearestCollisionRanges[ray], 0.F,
                    "A teammate hid a penetrated restricted-zone alert");
        RequireNear(published.m_nearestCollisionDistances[ray], 0.F,
                    "A boundary changed a saturated restricted-zone distance");
    }
    return 0;
}
