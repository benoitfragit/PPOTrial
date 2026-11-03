/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_reward_shaping.hpp
 * @brief Façonnage dense de patrouille, formation, proximité, frontière et basse vitesse.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "generalized_runtime_state.hpp"
#include "patrol_training_settings.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ppo::examples::bluedefense::defenders {

/** @brief Conservative dense boundary weight versus sparse neutralizations. */
inline constexpr float kGeneralizedBoundaryClearanceWeight = 0.01F;

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
    // The departure cost uses the actual position; anticipation stays in
    // the approaching term, so saturation cannot precede the hard limit.
    float const outside = std::max(
        exitFraction(mobile.north, parameters.halfNorth),
        exitFraction(mobile.east, parameters.halfEast));
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


/** @brief Number of physically visible Reds (delayed spawns are masked). */
inline std::size_t GeneralizedLiveRedCount(
    GeneralizedRuntimeState const& state) noexcept
{
    std::size_t count = 0U;
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
        if (state.reds[index].available)
            ++count;
    return count;
}

/**
 * @brief The existing patrol reward applies only while no hostile aircraft
 * can be engaged. This selects a training objective, never the actor's action.
 * Both snapshots must agree: first appearances and kills cannot earn an
 * unrelated racetrack reward on the same transition.
 */
inline bool GeneralizedPatrolRewardRelevant(
    GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after) noexcept
{
    return GeneralizedLiveRedCount(before) == 0U
        && GeneralizedLiveRedCount(after) == 0U;
}

/**
 * @brief Relative progress towards a fixed beginning-of-step waypoint.
 * Uses the exact same normalization in single-Red pursuit and patrol.
 */
inline float GeneralizedFrozenWaypointProgress(
    Parameters const& parameters, Mobile const& before, Mobile const& after,
    float const targetNorth, float const targetEast) noexcept
{
    float const oldDistance = std::hypot(
        before.north - targetNorth, before.east - targetEast);
    float const newDistance = std::hypot(
        after.north - targetNorth, after.east - targetEast);
    return Clamp((oldDistance - newDistance)
        / std::max(parameters.maxSpeed * parameters.dt, 1.F), -1.F, 1.F);
}

/**
 * @brief Progress of the physical leader towards the sole visible Red.
 * The Red is held at its before-step location, as in team pursuit shaping.
 * Combat scale remains maxSpeed*dt, matching the existing group pursuit.
 */
inline float GeneralizedSingleRedLeaderProgress(
    Parameters const& parameters, GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after, BlueTeamRoles const& roles) noexcept
{
    if (!roles.IsValid() || before.counts.blueMobiles != after.counts.blueMobiles
        || before.counts.redMobiles != after.counts.redMobiles
        || roles.ActiveBlueCount() != before.counts.blueMobiles
        || GeneralizedLiveRedCount(before) != 1U
        || GeneralizedLiveRedCount(after) != 1U)
        return 0.F;
    std::size_t const leader = roles.LeaderPhysicalIndex();
    if (!before.blues[leader].available || !after.blues[leader].available)
        return 0.F;
    for (std::size_t red = 0U; red < before.counts.redMobiles; ++red)
    {
        if (!before.reds[red].available || !after.reds[red].available)
            continue;
        Mobile const& hostile = before.reds[red];
        return GeneralizedFrozenWaypointProgress(parameters,
            before.blues[leader], after.blues[leader],
            hostile.north, hostile.east);
    }
    return 0.F;
}

/** @brief Individually normalized position, speed and heading errors. */
struct GeneralizedPatrolError final
{
    float position{};
    float speed{};
    float heading{};

    bool InTolerance() const noexcept
    {
        return position <= 1.F && speed <= 1.F && heading <= 1.F;
    }

    float BoundedCost() const noexcept
    {
        // A flat clip at one tolerance would remove all guidance for a
        // distant leader or follower. Rational saturation stays bounded
        // while preserving a gradient until recovery.
        auto const saturate = [](float const error) noexcept {
            return 1.F - 1.F / (1.F + std::max(error, 0.F));
        };
        return 0.5F * saturate(position)
            + 0.3F * saturate(speed)
            + 0.2F * saturate(heading);
    }
};

/** @brief Errors refer only to the appropriate role's projected target. */
inline GeneralizedPatrolError GeneralizedNoThreatError(
    GeneralizedRuntimeState const& state,
    GeneralizedPatrolTarget const& target, std::size_t const physicalIndex,
    PatrolNoThreatSettings const& settings) noexcept
{
    if (!target.valid || physicalIndex >= state.counts.blueMobiles
        || !state.blues[physicalIndex].available)
        return {2.F, 2.F, 2.F};
    Mobile const& blue = state.blues[physicalIndex];
    return {
        std::hypot(blue.north - target.pose.north,
                   blue.east - target.pose.east)
            / settings.positionToleranceMeters,
        std::abs(blue.speed - target.speed)
            / std::max(settings.speedToleranceFraction * target.speed, 1.F),
        std::abs(std::remainder(
            blue.heading - target.pose.heading, kPatrolTwoPi))
            / settings.headingToleranceRadians
    };
}

/** @brief Convenience overload; the runtime passes its already-computed target. */
inline GeneralizedPatrolError GeneralizedNoThreatError(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    BlueTeamRoles const& roles, std::size_t const physicalIndex,
    PatrolNoThreatSettings const& settings) noexcept
{
    return GeneralizedNoThreatError(state,
        GeneralizedNoThreatTarget(parameters, state, roles, physicalIndex),
        physicalIndex, settings);
}

/** @brief Bounded physical advance; projection jumps or station keeping earn zero. */
inline float GeneralizedPatrolAdvance(
    Parameters const& parameters, GeneralizedPatrolTarget const& before,
    GeneralizedPatrolTarget const& after) noexcept
{
    if (!before.valid || !after.valid)
        return 0.F;
    float const forward = std::remainder(
        after.pose.arclength - before.pose.arclength,
        CircuitLength(parameters.geometry));
    if (forward <= 0.F || forward > 1.5F * parameters.maxSpeed * parameters.dt)
        return 0.F;
    return Clamp(forward / std::max(parameters.cruiseSpeed * parameters.dt, 1.F),
        0.F, 1.F);
}

/**
 * @brief Bounded tracking improvement plus correctly flown forward motion.
 * Targets belong to their respective before/after snapshots. The quality term
 * is multiplied by physical progress, so parking on the curve earns no bonus.
 * The runtime compensates the formation-channel weight to keep effective
 * patrol and formation magnitudes comparable to stepRewardScale.
 */
inline float GeneralizedPatrolTransitionShaping(
    Parameters const& parameters, GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after,
    GeneralizedPatrolTargets const& beforeTargets,
    GeneralizedPatrolTargets const& afterTargets, BlueTeamRoles const& roles,
    std::size_t const index, PatrolNoThreatSettings const& settings) noexcept
{
    if (!settings.enabled || !roles.IsValid()
        || !Valid(before.counts) || !Valid(after.counts)
        || roles.ActiveBlueCount() != before.counts.blueMobiles
        || roles.ActiveBlueCount() != after.counts.blueMobiles
        || index >= before.counts.blueMobiles
        || index >= after.counts.blueMobiles
        || !beforeTargets[index].valid || !afterTargets[index].valid)
        return 0.F;
    auto const previous = GeneralizedNoThreatError(
        before, beforeTargets[index], index, settings);
    auto const current = GeneralizedNoThreatError(
        after, afterTargets[index], index, settings);
    float const cost = current.BoundedCost();
    float const improvement = Clamp(
        (previous.BoundedCost() - cost) * settings.positionToleranceMeters
            / std::max(parameters.cruiseSpeed * parameters.dt, 1.F), -1.F, 1.F);
    float const tracking = Clamp(1.F - std::max(
        current.position, std::max(current.speed, current.heading)), 0.F, 1.F);
    std::size_t const leader = roles.LeaderPhysicalIndex();
    float const forward = GeneralizedPatrolAdvance(
        parameters, beforeTargets[leader], afterTargets[leader]);
    return settings.stepRewardScale * Clamp(
        0.5F * improvement + forward * (0.25F * (1.F - cost) + 0.75F * tracking)
            - 0.25F * cost, -1.F, 1.F);
}


/**
 * @brief Pvs0 convergence at the same reward scale as single-Red pursuit.
 *
 * The current waypoint is frozen for the whole physics step. Far off-track,
 * use the closest projected reference instead of paying for parallel flight
 * towards a perpetual lookahead. The reward strongly prefers cruise speed
 * over a saturated 750 m/s chase, without making recovery gradients vanish.
 */
inline float GeneralizedPatrolConvergenceShaping(
    Parameters const& parameters, GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after,
    GeneralizedPatrolTargets const& beforeTargets,
    GeneralizedPatrolTargets const& afterTargets, BlueTeamRoles const& roles,
    std::size_t const index, PatrolNoThreatSettings const& settings) noexcept
{
    if (!settings.enabled || !roles.IsValid()
        || !Valid(before.counts) || !Valid(after.counts)
        || roles.ActiveBlueCount() != before.counts.blueMobiles
        || roles.ActiveBlueCount() != after.counts.blueMobiles
        || index >= before.counts.blueMobiles
        || index >= after.counts.blueMobiles
        || !before.blues[index].available || !after.blues[index].available
        || !beforeTargets[index].valid || !afterTargets[index].valid)
        return 0.F;

    Mobile const& previous = before.blues[index];
    Mobile const& current = after.blues[index];
    PatrolCurvePose const& track = beforeTargets[index].pose;
    PatrolCurvePose const& waypoint = beforeTargets[index].guidancePose;
    GeneralizedPatrolError const error = GeneralizedNoThreatError(
        after, afterTargets[index], index, settings);
    float const trackGate = Clamp(1.F - error.position, 0.F, 1.F);
    float const waypointClosing = GeneralizedFrozenWaypointProgress(
        parameters, previous, current, waypoint.north, waypoint.east);
    float const trackClosing = GeneralizedFrozenWaypointProgress(
        parameters, previous, current, track.north, track.east);
    float const closing = trackGate * waypointClosing
        + (1.F - trackGate) * trackClosing;
    // At 300 m/s, 60 m of correct flight earns approximately +0.8, the
    // same as a 60-m step towards one Red (weight 2, maxSpeed*dt = 150 m).
    // Beyond cruise speed, this bounded penalty defeats the incentive to
    // accelerate solely to maximize lookahead-distance reduction.
    float const speedPenalty = 0.4F * error.speed / (1.F + error.speed);
    float const alignment = Clamp(
        1.F - std::max(error.speed, error.heading), 0.F, 1.F);
    std::size_t const leader = roles.LeaderPhysicalIndex();
    float const forward = GeneralizedPatrolAdvance(
        parameters, beforeTargets[leader], afterTargets[leader]);
    // Saturate the progress bonus BEFORE applying the speed penalty:
    // otherwise 2*closing can exceed one and the final clamp conceals
    // the overspeed penalty (750 m/s would still earn the maximum).
    float const progressReward = Clamp(2.F * closing, -1.F, 1.F);
    return Clamp(
        progressReward - speedPenalty
            + settings.stepRewardScale * (
                forward * trackGate * alignment
                - 0.15F * error.BoundedCost()),
        -1.F, 1.F);
}

/**
 * @brief Physical Pvs0 wedge qualification, distinct from numeric tracking.
 * Unreachable references retain observations and their speed/position errors.
 */
inline bool GeneralizedPvs0WedgeHeld(
    GeneralizedRuntimeState const& state,
    GeneralizedPatrolTarget const& target, std::size_t const index,
    PatrolNoThreatSettings const& settings) noexcept
{
    return target.valid && target.physicallyFeasible
        && GeneralizedNoThreatError(state, target, index, settings).InTolerance();
}

/**
 * @brief Unified patrol/formation convergence from frozen references.
 * Applies equally when red slots are configured but currently inactive.
 */
inline float GeneralizedUnifiedPatrolConvergenceShaping(
    Parameters const& parameters, GeneralizedRuntimeState const& before,
    GeneralizedRuntimeState const& after,
    GeneralizedPatrolTargets const& beforeTargets,
    GeneralizedPatrolTargets const& afterTargets, BlueTeamRoles const& roles,
    std::size_t const index, PatrolNoThreatSettings const& settings) noexcept
{
    if (!settings.enabled || !roles.IsValid()
        || !Valid(before.counts) || !Valid(after.counts)
        || roles.ActiveBlueCount() != before.counts.blueMobiles
        || roles.ActiveBlueCount() != after.counts.blueMobiles
        || index >= before.counts.blueMobiles
        || index >= after.counts.blueMobiles
        || !before.blues[index].available || !after.blues[index].available
        || !beforeTargets[index].valid || !afterTargets[index].valid)
        return 0.F;

    Mobile const& previous = before.blues[index];
    Mobile const& current = after.blues[index];
    PatrolCurvePose const& track = beforeTargets[index].pose;
    PatrolCurvePose const& waypoint = beforeTargets[index].guidancePose;
    GeneralizedPatrolError const error = GeneralizedNoThreatError(
        after, afterTargets[index], index, settings);
    float const trackGate = Clamp(1.F - error.position, 0.F, 1.F);
    float const waypointClosing = GeneralizedFrozenWaypointProgress(
        parameters, previous, current, waypoint.north, waypoint.east);
    float const trackClosing = GeneralizedFrozenWaypointProgress(
        parameters, previous, current, track.north, track.east);
    float const closing = trackGate * waypointClosing
        + (1.F - trackGate) * trackClosing;

    float const speedReference = afterTargets[index].speed;
    float const speedRatio = speedReference > 0.F
        ? std::min(1.F, speedReference / std::max(current.speed, speedReference))
        : 0.F;
    float const speedGate = speedRatio * speedRatio;
    float const rawProgress = Clamp(2.F * closing, -1.F, 1.F);
    // Never gate a negative progress: retreat under overspeed stays costly.
    float const gatedProgress = rawProgress > 0.F
        ? rawProgress * speedGate : rawProgress;
    float const speedPenalty = 0.4F * error.speed / (1.F + error.speed);
    float const alignment = Clamp(
        1.F - std::max(error.speed, error.heading), 0.F, 1.F);
    std::size_t const leader = roles.LeaderPhysicalIndex();
    // Only physical leader arclength can earn leader forward-lap credit.
    float const forward = index == leader
        ? GeneralizedPatrolAdvance(
            parameters, beforeTargets[leader], afterTargets[leader])
        : 0.F;
    return Clamp(
        gatedProgress - speedPenalty
            + settings.stepRewardScale * (
                forward * trackGate * alignment * speedGate
                - 0.15F * error.BoundedCost()),
        -1.F, 1.F);
}


/**
 * @brief Accumulates one full physically attainable correctly flown lap.
 *
 * Only the leader qualifies the circuit; wingmen retain their own formation
 * and collision rewards. Leader tracking errors consume the unqualified-distance
 * budget. Reversing or implausible projection jumps reset progress. Station
 * keeping or a projection wrap-around cannot yield mission success.
 */
class GeneralizedPatrolLapProgress final
{
public:
    /** @brief Clears episode diagnostics and accumulated physical distance. */
    void Reset() noexcept { *this = {}; }
    bool Completed() const noexcept { return completed_; }
    float DistanceMeters() const noexcept { return distanceMeters_; }
    float QualifiedDistanceMeters() const noexcept { return qualifiedMeters_; }
    std::size_t ResetCount() const noexcept { return resets_; }
    std::size_t TrackingViolationSteps() const noexcept { return trackingViolations_; }
    char const* LastResetCause() const noexcept { return resetCause_; }

    /** @brief Reuses the exact snapshot references used by the policy and reward. */
    bool Observe(
        Parameters const& parameters, GeneralizedRuntimeState const& before,
        GeneralizedRuntimeState const& after, BlueTeamRoles const& roles,
        PatrolNoThreatSettings const& settings,
        GeneralizedPatrolTargets const& beforeTargets,
        GeneralizedPatrolTargets const& afterTargets) noexcept
    {
        if (!Valid(before.counts) || !Valid(after.counts) || !roles.IsValid()
            || roles.ActiveBlueCount() != after.counts.blueMobiles
            || before.counts.blueMobiles != after.counts.blueMobiles)
            return Invalidate("invalid_contract");
        if (!settings.enabled)
            return Invalidate("disabled");
        std::size_t const leader = roles.LeaderPhysicalIndex();
        if (leader >= before.counts.blueMobiles
            || !before.blues[leader].available || !after.blues[leader].available)
            return Invalidate("leader_lost");
        bool const qualified = afterTargets[leader].valid
            && GeneralizedNoThreatError(
                after, afterTargets[leader], leader, settings).InTolerance();
        if (!qualified)
        {
            ++trackingViolations_;
            if (settings.minimumQualifiedFraction >= 1.F)
                return Invalidate("tracking");
        }
        if (!beforeTargets[leader].valid || !afterTargets[leader].valid)
            return Invalidate("invalid_reference");
        float const circuit = CircuitLength(parameters.geometry);
        float const forward = std::remainder(
            afterTargets[leader].pose.arclength - beforeTargets[leader].pose.arclength,
            circuit);
        // The displacement check also rejects physically impossible test/runtime jumps.
        float const displacement = std::hypot(
            after.blues[leader].north - before.blues[leader].north,
            after.blues[leader].east - before.blues[leader].east);
        float const maximum = 1.5F * parameters.maxSpeed * parameters.dt;
        if (forward <= 0.F || forward > maximum || displacement > maximum)
            return Invalidate("reverse_stationary_or_jump");
        distanceMeters_ += forward;
        if (qualified)
            qualifiedMeters_ += forward;
        if (distanceMeters_ >= circuit - 0.25F)
        {
            completed_ = qualifiedMeters_ + 0.25F
                >= settings.minimumQualifiedFraction * distanceMeters_;
            if (!completed_)
                return Invalidate("insufficient_qualified_distance");
        }
        return completed_;
    }

    /** @brief Convenience overload for domain tests without a snapshot cache. */
    bool Observe(
        Parameters const& parameters, GeneralizedRuntimeState const& before,
        GeneralizedRuntimeState const& after, BlueTeamRoles const& roles,
        PatrolNoThreatSettings const& settings) noexcept
    {
        return Observe(parameters, before, after, roles, settings,
            ComputePatrolTargets(parameters, before, roles, settings.lookaheadSeconds),
            ComputePatrolTargets(parameters, after, roles, settings.lookaheadSeconds));
    }

private:
    bool Invalidate(char const* const cause) noexcept
    {
        distanceMeters_ = 0.F;
        qualifiedMeters_ = 0.F;
        completed_ = false;
        ++resets_;
        resetCause_ = cause;
        return false;
    }
    float distanceMeters_{};
    float qualifiedMeters_{};
    std::size_t resets_{};
    std::size_t trackingViolations_{};
    char const* resetCause_{"none"};
    bool completed_{};
};

/** @brief Retain patrol outside engagement and relax it under pressure. */
inline float GeneralizedCooperationWeight(
    Parameters const& parameters, GeneralizedRuntimeState const& state) noexcept
{
    return 1.F - 0.75F * GeneralizedEngagementWeight(parameters, state);
}

} // namespace ppo::examples::bluedefense::defenders
