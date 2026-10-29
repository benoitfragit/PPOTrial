#pragma once

#include "generalized_runtime_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ppo::examples::bluedefense::defenders {

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
 * @brief Continuously blends autonomous distant pursuit with nearby teamwork.
 * Without hostile mobiles, full patrol/formation is required. Imminent threats
 * can relax formation to prioritize defense without removing the safety LiDAR.
 */
inline float GeneralizedCooperationWeight(
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
        return 1.F;
    if (!std::isfinite(nearest))
        return 0.F;
    float const engagementRange =
        parameters.defense.teammateThreatDistanceMeters;
    float const close = Clamp(
        (2.F * engagementRange - nearest) / engagementRange, 0.F, 1.F);
    float const pressure = generalized_runtime_detail::ThreatPressure(
        state, parameters);
    return close * (1.F - 0.75F * pressure);
}

} // namespace ppo::examples::bluedefense::defenders
