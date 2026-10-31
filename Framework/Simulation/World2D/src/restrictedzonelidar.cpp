#include "restrictedzonelidar.hpp"
#include "restrictedzonegeometry.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ppo::model {
namespace {

// Preserve the exact historical RLMission pseudo-LiDAR normalization,
// sampling angles, danger expansion, intersection thresholds, and update order.
constexpr std::size_t kPseudoLidarBinCount = kRestrictedZoneLidarBins;
constexpr float kPseudoLidarAngleSpan = static_cast<float>(ppo::model::TWOPI);
constexpr float kPseudoLidarHalfAngleSpan = 0.5F * kPseudoLidarAngleSpan;
constexpr float kPseudoLidarStepAngle =
    kPseudoLidarAngleSpan / static_cast<float>(kPseudoLidarBinCount - 1U);
constexpr float kRestrictedZoneDangerMultiplier =
    kRestrictedZoneWarningRadiusScale;
constexpr float kNoEscapeEpsilon = 1.0e-4F;
constexpr float kRayEpsilon = 1.0e-5F;

float NormalizeRange(float distance, float rangeReference) noexcept
{
    return distance / (distance + std::max(rangeReference, 1.F));
}

/**
* @brief Updates one pseudo-lidar bin with a closer collision candidate.
*/
void
UpdatePseudoLidarHit(float hitDistance,
                     float dangerRadius,
                     float& bestRange,
                     float& bestDistance) noexcept
{
    // A saturated restricted-zone ray must stay saturated, including its
    // physical distance, when subsequent zones or boundaries are inspected.
    if (hitDistance < 0.F || bestRange <= 0.F)
        return;

    float const normalizedDistance = NormalizeRange(hitDistance, dangerRadius);
    if (bestDistance <= 0.F || hitDistance < bestDistance)
        bestDistance = hitDistance;

    bestRange = std::min(bestRange, normalizedDistance);
}

/**
* @brief Intersects one ray with the evaluation area boundary.
*/
float
ComputeBoundaryIntersectionDistance(ppo::model::sEntityState const& ownState,
                                    float worldAngle,
                                    float northLimit,
                                    float eastLimit) noexcept
{
    float const dirNorth = std::cos(worldAngle);
    float const dirEast = std::sin(worldAngle);
    float bestDistance = std::numeric_limits<float>::infinity();

    auto const tryUpdate = [&](float distance, float north, float east) noexcept
    {
        if (distance < 0.F)
            return;

        if (std::fabs(north) <= northLimit + 1.0e-3F
        &&  std::fabs(east) <= eastLimit + 1.0e-3F)
        {
            bestDistance = std::min(bestDistance, distance);
        }
    };

    if (std::fabs(dirNorth) > kRayEpsilon)
    {
        float const dNorthPos = (northLimit - ownState.m_x) / dirNorth;
        tryUpdate(dNorthPos,
                  northLimit,
                  ownState.m_y + dNorthPos * dirEast);

        float const dNorthNeg = (-northLimit - ownState.m_x) / dirNorth;
        tryUpdate(dNorthNeg,
                  -northLimit,
                  ownState.m_y + dNorthNeg * dirEast);
    }

    if (std::fabs(dirEast) > kRayEpsilon)
    {
        float const dEastPos = (eastLimit - ownState.m_y) / dirEast;
        tryUpdate(dEastPos,
                  ownState.m_x + dEastPos * dirNorth,
                  eastLimit);

        float const dEastNeg = (-eastLimit - ownState.m_y) / dirEast;
        tryUpdate(dEastNeg,
                  ownState.m_x + dEastNeg * dirNorth,
                  -eastLimit);
    }

    return std::isfinite(bestDistance) ? bestDistance : -1.F;
}


} // namespace

/**
* @brief Builds pseudo-lidar hits against hostile restricted zone danger areas.
*/
void
BuildRestrictedZonePseudoLidar(entt::registry& registry,
                    ppo::model::BaseEntity& ownEntity,
                    ppo::model::sEntityState const& ownState,
                    std::array<float, kPseudoLidarBinCount>& nearestCollisionRanges,
                    std::array<float, kPseudoLidarBinCount>& nearestCollisionDistances) noexcept
{
    nearestCollisionRanges.fill(1.F);
    nearestCollisionDistances.fill(0.F);

    auto view = registry.view<ppo::model::sEntityState, ppo::model::sSNAState>();
    for (auto entity : view)
    {
        auto const [otherState, otherSnaState] = view.get(entity);

        if (entity == ownEntity.uniqid()
        ||  otherState.m_destroyed
        ||  otherState.m_type != ppo::model::eEntityType::RestrictedZone
        ||  otherState.m_side == ownState.m_side)
        {
            continue;
        }

        float const dangerRadius = kRestrictedZoneDangerMultiplier
            * otherSnaState.m_noescapezone;
        float const mx = ownState.m_x - otherState.m_x;
        float const my = ownState.m_y - otherState.m_y;
        float const distanceToCenter = std::sqrt(mx * mx + my * my);

        for (std::size_t iang = 0; iang < nearestCollisionRanges.size(); ++iang)
        {
            float const localAngle = -kPseudoLidarHalfAngleSpan + static_cast<float>(iang) * kPseudoLidarStepAngle;
            float const worldAngle = ownState.m_rotation + localAngle;

            if (distanceToCenter <= dangerRadius + kNoEscapeEpsilon)
            {
                nearestCollisionRanges[iang] = 0.F;
                nearestCollisionDistances[iang] = 0.F;
                continue;
            }

            float const b = mx * std::cos(worldAngle) + my * std::sin(worldAngle);
            float const c = mx * mx + my * my - dangerRadius * dangerRadius;
            float const delta = b * b - c;
            if (delta < 0.F)
                continue;

            float const sqrtDelta = std::sqrt(delta);
            float const d1 = -b - sqrtDelta;
            float const d2 = -b + sqrtDelta;

            UpdatePseudoLidarHit(
                d1,
                dangerRadius,
                nearestCollisionRanges[iang],
                nearestCollisionDistances[iang]);
            UpdatePseudoLidarHit(
                d2,
                dangerRadius,
                nearestCollisionRanges[iang],
                nearestCollisionDistances[iang]);
        }
    }
}

/**
 * @brief Senses other living same-side mobiles as nonlethal moving circles.
 *
 * Inside a circle, the positive exit ray retains a usable escape direction,
 * unlike the deliberately saturated legacy restricted-zone reading.
 */
void BuildFriendlyMobilePseudoLidar(
    entt::registry& registry,
    BaseEntity& ownEntity,
    sEntityState const& ownState,
    float const safetyRadius,
    std::array<float, kPseudoLidarBinCount>& nearestCollisionRanges,
    std::array<float, kPseudoLidarBinCount>& nearestCollisionDistances) noexcept
{
    if (!std::isfinite(safetyRadius) || safetyRadius <= 0.F)
        return;
    auto view = registry.view<ppo::model::sEntityState>();
    for (auto entity : view)
    {
        auto const& other = view.get<ppo::model::sEntityState>(entity);
        if (entity == ownEntity.uniqid()
            || other.m_destroyed
            || other.m_type != ppo::model::eEntityType::Mobile
            || other.m_side != ownState.m_side)
        {
            continue;
        }
        float const mx = ownState.m_x - other.m_x;
        float const my = ownState.m_y - other.m_y;
        float const c = mx * mx + my * my - safetyRadius * safetyRadius;
        for (std::size_t ray = 0U; ray < nearestCollisionRanges.size(); ++ray)
        {
            float const angle = ownState.m_rotation - kPseudoLidarHalfAngleSpan
                + static_cast<float>(ray) * kPseudoLidarStepAngle;
            float const b = mx * std::cos(angle) + my * std::sin(angle);
            float const delta = b * b - c;
            if (delta < 0.F)
                continue;
            float const root = std::sqrt(delta);
            // Never replace a saturated restricted-zone hit with a
            // nonlethal teammate reading, even after boundary processing.
            // Otherwise choose by real metres, not by different per-obstacle
            // normalization references: a nearer blue must not be hidden by
            // the normalized distance to a much farther world boundary.
            if (nearestCollisionRanges[ray] == 0.F)
            {
                continue;
            }
            float const entry = -b - root;
            float const exit = -b + root;
            float const distance = c < 0.F ? exit
                : entry >= 0.F ? entry : exit;
            if (!std::isfinite(distance) || distance < 0.F)
                continue;
            if (nearestCollisionDistances[ray] == 0.F
                || distance < nearestCollisionDistances[ray])
            {
                nearestCollisionDistances[ray] = distance;
                nearestCollisionRanges[ray] =
                    NormalizeRange(distance, safetyRadius);
            }
        }
    }
}

/**
* @brief Extends pseudo-lidar hits with map-boundary intersections.
*/
void
BuildZoneBoundaryPseudoLidar(ppo::model::sEntityState const& ownState,
                             ppo::model::sEnvironmentState const& envState,
                             std::array<float, kPseudoLidarBinCount>& nearestCollisionRanges,
                             std::array<float, kPseudoLidarBinCount>& nearestCollisionDistances,
                             float const boundaryScale) noexcept
{
    if (!std::isfinite(boundaryScale) || boundaryScale <= 0.F)
        return;
    float const northLimit = 0.5F * envState.m_maxNorth * boundaryScale;
    float const eastLimit = 0.5F * envState.m_maxEast * boundaryScale;
    float const zoneRangeReference =
        std::max(std::sqrt(northLimit * northLimit + eastLimit * eastLimit), 1.F);

    for (std::size_t iang = 0; iang < nearestCollisionRanges.size(); ++iang)
    {
        float const localAngle =
            -kPseudoLidarHalfAngleSpan + static_cast<float>(iang) * kPseudoLidarStepAngle;
        float const worldAngle = ownState.m_rotation + localAngle;
        float const hitDistance = ComputeBoundaryIntersectionDistance(
            ownState,
            worldAngle,
            northLimit,
            eastLimit);

        UpdatePseudoLidarHit(
            hitDistance,
            zoneRangeReference,
            nearestCollisionRanges[iang],
            nearestCollisionDistances[iang]);
    }
}

void UpdateRestrictedZonePseudoLidars(entt::registry& registry,
                                      eEntitySide mobileSide,
                                      float const friendlyMobileSafetyRadius,
                                      float const boundaryScale) noexcept
{
    if (!registry.ctx().contains<ppo::model::sEnvironmentState>())
        return;
    auto const& environment =
        registry.ctx().get<ppo::model::sEnvironmentState>();
    auto view = registry.view<ppo::model::sEntityState,
                              ppo::model::sPseudoLidarState>();
    for (auto entity : view)
    {
        auto const& state = view.get<ppo::model::sEntityState>(entity);
        if (state.m_type != ppo::model::eEntityType::Mobile
            || state.m_side != mobileSide
            || state.m_destroyed)
            continue;
        auto& sensor = view.get<ppo::model::sPseudoLidarState>(entity);
        ppo::model::BaseEntity own{&registry, entity};
        BuildRestrictedZonePseudoLidar(
            registry, own, state, sensor.m_nearestCollisionRanges,
            sensor.m_nearestCollisionDistances);
        BuildZoneBoundaryPseudoLidar(
            state, environment, sensor.m_nearestCollisionRanges,
            sensor.m_nearestCollisionDistances, boundaryScale);
        BuildFriendlyMobilePseudoLidar(
            registry, own, state, friendlyMobileSafetyRadius,
            sensor.m_nearestCollisionRanges,
            sensor.m_nearestCollisionDistances);
    }
}


} // namespace ppo::model
