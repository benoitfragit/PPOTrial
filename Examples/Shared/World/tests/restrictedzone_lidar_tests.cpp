#include "restrictedzonelidar.hpp"
#include "snasystem.hpp"
#include "entitytypes.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace {
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

    return 0;
}
