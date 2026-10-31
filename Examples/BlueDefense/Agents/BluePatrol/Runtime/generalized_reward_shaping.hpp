#pragma once

#include "generalized_runtime_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Weighted control effort for one active blue trajectory.
 *
 * Actions are already normalized; the configured shaping coefficient is
 * applied exactly once here, before the cooperative reward channel weight.
 */
inline float GeneralizedControlEffortPenalty(
    Parameters const& parameters, Action const& action) noexcept
{
    return parameters.controlEffortWeight * 0.5F
        * (std::abs(Clamp(action[0], -1.F, 1.F))
           + std::abs(Clamp(action[1], -1.F, 1.F)));
}

/**
 * @brief Constant-velocity prediction of near-term friendly separation.
 *
 * This reward-only assessment never modifies actions, the flight model or
 * physical collision resolution. Dead peers cannot produce false penalties.
 */
inline float GeneralizedBlueSeparationRisk(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    std::size_t const physicalIndex) noexcept
{
    if (physicalIndex >= state.counts.blueMobiles
        || !state.blues[physicalIndex].available)
        return 0.F;
    constexpr float horizonSeconds = 5.F;
    Mobile const& own = state.blues[physicalIndex];
    float const ownNorthSpeed = own.speed * std::cos(own.heading);
    float const ownEastSpeed = own.speed * std::sin(own.heading);
    float risk = 0.F;
    for (std::size_t other = 0U; other < state.counts.blueMobiles; ++other)
    {
        if (other == physicalIndex || !state.blues[other].available)
            continue;
        Mobile const& peer = state.blues[other];
        float const north = peer.north - own.north;
        float const east = peer.east - own.east;
        float const velocityNorth =
            peer.speed * std::cos(peer.heading) - ownNorthSpeed;
        float const velocityEast =
            peer.speed * std::sin(peer.heading) - ownEastSpeed;
        float const relativeSpeedSquared =
            velocityNorth * velocityNorth + velocityEast * velocityEast;
        float const time = relativeSpeedSquared > 1.e-4F
            ? Clamp(-(north * velocityNorth + east * velocityEast)
                / relativeSpeedSquared, 0.F, horizonSeconds) : 0.F;
        float const closestDistance = std::hypot(
            north + velocityNorth * time,
            east + velocityEast * time);
        float const proximity = Clamp(
            (parameters.cautionDistance - closestDistance)
                / parameters.cautionDistance, 0.F, 1.F);
        risk = std::max(risk, proximity / (1.F + time / horizonSeconds));
    }
    return risk;
}

/**
 * @brief Blue-only danger/caution gradient matching the red zone distances.
 *
 * dangerRadius=1.10*NEZ, cautionRadius=1.75*NEZ. A facing approach is
 * slightly more costly; an actual penetration always saturates to one.
 */
inline float GeneralizedRestrictedZoneProximityRisk(
    float const distance, float const noEscapeZone,
    float const forwardAlignment) noexcept
{
    if (!std::isfinite(distance) || !std::isfinite(noEscapeZone)
        || !std::isfinite(forwardAlignment) || noEscapeZone <= 0.F)
        return 0.F;
    float const dangerRadius = 1.10F * noEscapeZone;
    float const cautionRadius = 1.75F * noEscapeZone;
    if (distance <= dangerRadius)
        return 1.F;
    float const proximity = Clamp((cautionRadius - distance)
        / std::max(cautionRadius - dangerRadius, 1.F), 0.F, 1.F);
    return proximity * (0.5F + 0.5F * Clamp(forwardAlignment, 0.F, 1.F));
}

/**
 * @brief Bounded graduated exit risk: caution inside, increasing cost outside.
 *
 * The gradient only saturates at the hard failure rectangle, so the dense
 * BoundaryClearance channel stays secondary to sparse combat outcomes.
 */
inline float GeneralizedBoundaryRisk(
    Parameters const& parameters, Mobile const& mobile) noexcept
{
    if (!mobile.available)
        return 0.F;
    constexpr float horizonSeconds = 5.F;
    float const caution = std::max(
        parameters.cautionDistance, parameters.maxSpeed * horizonSeconds);
    float const projectedNorth = mobile.north
        + mobile.speed * std::cos(mobile.heading) * horizonSeconds;
    float const projectedEast = mobile.east
        + mobile.speed * std::sin(mobile.heading) * horizonSeconds;
    auto const axisRisk = [caution](float const position,
                                    float const limit) noexcept {
        return Clamp((std::abs(position) - limit + caution)
            / caution, 0.F, 1.F);
    };
    float const approaching = std::max({
        axisRisk(mobile.north, parameters.halfNorth),
        axisRisk(mobile.east, parameters.halfEast),
        axisRisk(projectedNorth, parameters.halfNorth),
        axisRisk(projectedEast, parameters.halfEast)
    });
    auto const exitFraction = [scale = parameters.hardBoundaryScale](
                                  float const position, float const limit) noexcept {
        return Clamp((std::abs(position) - limit)
            / std::max((scale - 1.F) * limit, 1.F), 0.F, 1.F);
    };
    float const outside = std::max({
        exitFraction(mobile.north, parameters.halfNorth),
        exitFraction(mobile.east, parameters.halfEast),
        exitFraction(projectedNorth, parameters.halfNorth),
        exitFraction(projectedEast, parameters.halfEast)
    });
    return Clamp(0.35F * approaching + 0.65F * outside, 0.F, 1.F);
}

/**
 * @brief Bounded dense safety penalty with an extra worsening-risk term.
 *
 * A holding cost discourages prolonged exposure; risk growth is penalized
 * immediately. Terminal collision/failure rewards remain independent.
 */
inline float GeneralizedSafetyPenalty(
    float const riskBefore, float const riskAfter) noexcept
{
    return Clamp(0.15F * std::max(riskBefore, riskAfter)
        + 0.35F * std::max(riskAfter - riskBefore, 0.F), 0.F, 1.F);
}

/** @brief Measures a teammate in the physical leader's heading frame. */
inline float GeneralizedFormationQuality(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    BlueTeamRoles const& roles, std::size_t const physicalIndex) noexcept
{
    auto const assignment = roles.Assignment(physicalIndex);
    if (!assignment || assignment->role != BlueRole::Teammate)
        return 0.F;
    Mobile const& leader = state.blues[roles.LeaderPhysicalIndex()];
    Mobile const& teammate = state.blues[physicalIndex];
    if (!leader.available || !teammate.available)
        return -1.F;

    FormationOffset const slot = EvaluateFormationOffset(
        assignment->formationSlot, parameters.geometry.separationDistance,
        parameters.geometry.separationAngle);
    float const cosine = std::cos(leader.heading);
    float const sine = std::sin(leader.heading);
    float const inward = static_cast<float>(parameters.geometry.direction)
        * slot.inwardMeters;
    float const expectedNorth = leader.north
        + slot.longitudinalMeters * cosine - inward * sine;
    float const expectedEast = leader.east
        + slot.longitudinalMeters * sine + inward * cosine;
    float const error = std::hypot(
        teammate.north - expectedNorth, teammate.east - expectedEast);
    return Clamp(1.F - error
        / std::max(parameters.geometry.separationDistance, 1.F), -1.F, 1.F);
}

/**
 * @brief Bounded one-step improvement of a living teammate's formation error.
 *
 * Static poor geometry does not incur an unbounded per-step debt that makes
 * self-destruction preferable. Destroyed participants or leaders cannot
 * contribute synthetic progress. The loss event has its own sparse penalty.
 */
inline float GeneralizedFormationProgressReward(
    Parameters const& parameters, GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after, BlueTeamRoles const& roles,
    std::size_t const physicalIndex) noexcept
{
    std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
    if (!roles.IsValid() || leaderIndex >= before.counts.blueMobiles
        || leaderIndex >= after.counts.blueMobiles
        || physicalIndex >= before.counts.blueMobiles
        || physicalIndex >= after.counts.blueMobiles
        || !before.blues[physicalIndex].available
        || !after.blues[physicalIndex].available
        || !before.blues[leaderIndex].available
        || !after.blues[leaderIndex].available)
        return 0.F;

    float const previousError = Clamp(
        1.F - GeneralizedFormationQuality(
            parameters, before, roles, physicalIndex)
            - parameters.formationTolerance, 0.F, 2.F);
    float const currentError = Clamp(
        1.F - GeneralizedFormationQuality(
            parameters, after, roles, physicalIndex)
            - parameters.formationTolerance, 0.F, 2.F);
    return Clamp(parameters.formationErrorWeight
        * (previousError - currentError), -1.F, 1.F);
}

/** @brief Risk gradient before a blue reaches critical low speed. */
inline float GeneralizedLowSpeedRisk(
    Parameters const& parameters, Mobile const& mobile) noexcept
{
    if (!mobile.available)
        return 0.F;
    // Aircraft start near cruise; warn well before the physics minimum speed.
    float const cautionSpeed = std::max(0.35F * parameters.cruiseSpeed, 1.F);
    return Clamp((cautionSpeed - mobile.speed) / cautionSpeed, 0.F, 1.F);
}

/** @brief Preserve the low-speed endpoint on an alive-to-dead transition. */
inline float GeneralizedLowSpeedTransitionPenalty(
    Parameters const& parameters, Mobile const& before,
    Mobile const& after) noexcept
{
    if (!before.available)
        return 0.F;
    Mobile endpoint = after;
    endpoint.available = true;
    return GeneralizedSafetyPenalty(GeneralizedLowSpeedRisk(parameters, before),
                                   GeneralizedLowSpeedRisk(parameters, endpoint));
}

/**
 * @brief Rewards physical progress along the racetrack, penalizing lateral and
 * heading errors. Unlike a pure arclength term, leaving the track costs reward.
 */
inline float GeneralizedLeaderTrackingReward(
    Parameters const& parameters, GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after, BlueTeamRoles const& roles) noexcept
{
    std::size_t const index = roles.LeaderPhysicalIndex();
    Mobile const& previous = before.blues[index];
    Mobile const& current = after.blues[index];
    if (!previous.available || !current.available)
        return -1.F;
    auto const from = ProjectOnPatrolCurve(
        parameters.geometry, previous.north, previous.east);
    auto const to = ProjectOnPatrolCurve(
        parameters.geometry, current.north, current.east);
    float const progress = Clamp(
        std::remainder(to.pose.arclength - from.pose.arclength,
                       CircuitLength(parameters))
            / std::max(parameters.cruiseSpeed * parameters.dt, 1.F),
        -1.F, 1.F);
    float const crossTrack = Clamp(
        std::abs(to.signedCrossTrackError)
            / std::max(parameters.patrolCrossTrackScale, 1.F),
        0.F, 2.F);
    float const headingError = 0.5F * (
        1.F - std::cos(current.heading - to.pose.heading));
    return Clamp(
        parameters.patrolProgressWeight * progress
            - parameters.patrolCrossTrackWeight * crossTrack
            - 0.25F * headingError,
        -1.F, 1.F);
}

/**
 * @brief Moving-blue-only progress towards the hostile set. Threat positions
 * are held at their pre-step values so red autopilot motion alone earns nothing.
 * Alive blues compete for nearest coverage; no target index is assigned.
 */
inline float GeneralizedPursuitProgress(
    Parameters const& parameters, GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after) noexcept
{
    DefenseTrack const objective{before.objectiveNorth, before.objectiveEast,
                                 0.F, 0.F, before.objectiveAvailable};
    float weightedProgress = 0.F;
    float totalWeight = 0.F;
    for (std::size_t redIndex = 0U; redIndex < before.counts.redMobiles;
         ++redIndex)
    {
        Mobile const& red = before.reds[redIndex];
        if (!red.available || !after.reds[redIndex].available)
            continue;
        float distanceBefore = std::numeric_limits<float>::infinity();
        float distanceAfter = std::numeric_limits<float>::infinity();
        for (std::size_t blueIndex = 0U; blueIndex < before.counts.blueMobiles;
             ++blueIndex)
        {
            Mobile const& blueBefore = before.blues[blueIndex];
            Mobile const& blueAfter = after.blues[blueIndex];
            if (!blueBefore.available || !blueAfter.available)
                continue;
            distanceBefore = std::min(distanceBefore,
                std::hypot(blueBefore.north - red.north,
                           blueBefore.east - red.east));
            distanceAfter = std::min(distanceAfter,
                std::hypot(blueAfter.north - red.north,
                           blueAfter.east - red.east));
        }
        if (!std::isfinite(distanceBefore) || !std::isfinite(distanceAfter))
            continue;
        DefenseTrack const hostile{red.north, red.east, red.heading,
                                   red.speed, red.available};
        float const weight = 0.25F + ObjectiveThreatRisk(
            parameters.defense, hostile, objective);
        weightedProgress += weight * Clamp(
            (distanceBefore - distanceAfter)
                / std::max(parameters.maxSpeed * parameters.dt, 1.F),
            -1.F, 1.F);
        totalWeight += weight;
    }
    return totalWeight > 0.F
        ? Clamp(weightedProgress / totalWeight, -1.F, 1.F) : 0.F;
}

/**
 * @brief Engagement pressure from nearby reds or danger to the objective.
 * No target is selected; distant harmless tracks do not reward pursuit.
 */
inline float GeneralizedEngagementWeight(
    Parameters const& parameters,
    GeneralizedRuntimeState const& state) noexcept
{
    float nearest = std::numeric_limits<float>::infinity();
    bool threatPresent = false;
    for (std::size_t redIndex = 0U; redIndex < state.counts.redMobiles;
         ++redIndex)
    {
        Mobile const& red = state.reds[redIndex];
        if (!red.available)
            continue;
        threatPresent = true;
        for (std::size_t blueIndex = 0U; blueIndex < state.counts.blueMobiles;
             ++blueIndex)
        {
            Mobile const& blue = state.blues[blueIndex];
            if (!blue.available)
                continue;
            nearest = std::min(nearest,
                std::hypot(blue.north - red.north,
                           blue.east - red.east));
        }
    }
    if (!threatPresent)
        return 0.F;
    if (!std::isfinite(nearest))
        return 0.F;
    float const engagementRange =
        parameters.defense.teammateThreatDistanceMeters;
    float const close = Clamp(
        (2.F * engagementRange - nearest) / engagementRange, 0.F, 1.F);
    float const pressure = generalized_runtime_detail::ThreatPressure(
        state, parameters);
    return std::max(close, pressure);
}

/** @brief Retain patrol outside engagement and relax it under pressure. */
inline float GeneralizedCooperationWeight(
    Parameters const& parameters, GeneralizedRuntimeState const& state) noexcept
{
    return 1.F - 0.75F * GeneralizedEngagementWeight(parameters, state);
}

} // namespace ppo::examples::bluedefense::defenders
