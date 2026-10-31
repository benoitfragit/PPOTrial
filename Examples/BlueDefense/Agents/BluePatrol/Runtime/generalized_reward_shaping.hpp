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

/**
 * @brief Whether any actually active hostile requires combat attention.
 *
 * Delayed/inactive tracks do not cancel patrol shaping. This only inspects
 * the runtime availability masks; no red controller or target is changed.
 */
inline bool GeneralizedHasActiveRed(
    GeneralizedRuntimeState const& state) noexcept
{
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
        if (state.reds[index].available)
            return true;
    return false;
}

/** @brief A Blue-specific theoretical reference, never an autopilot command. */
struct GeneralizedPatrolTarget final
{
    PatrolCurvePose pose{};
    float speed{};
};

/**
 * @brief Leader: racetrack projection. Teammate: projected leader wedge slot.
 *
 * Differentiate the follower's offset curve to obtain its own tangent and
 * turn-speed ratio. This is necessary in the turns; copying the leader's
 * heading and speed would reward a geometrically incorrect wedge.
 */
inline GeneralizedPatrolTarget GeneralizedNoThreatTarget(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    BlueTeamRoles const& roles, std::size_t const physicalIndex) noexcept
{
    Mobile const& leader = state.blues[roles.LeaderPhysicalIndex()];
    PatrolCurvePose const projected = ProjectOnPatrolCurve(
        parameters.geometry, leader.north, leader.east).pose;
    BlueRoleAssignment const assignment = *roles.Assignment(physicalIndex);
    if (assignment.role == BlueRole::Leader)
        return {projected, parameters.cruiseSpeed};

    float constexpr derivativeHalfWidth = 10.F;
    PatrolCurvePose pose = EvaluateFormationPose(
        parameters.geometry, projected, assignment.formationSlot);
    PatrolCurvePose const previous = EvaluateFormationPose(
        parameters.geometry,
        EvaluatePatrolCurve(parameters.geometry,
            projected.arclength - derivativeHalfWidth),
        assignment.formationSlot);
    PatrolCurvePose const next = EvaluateFormationPose(
        parameters.geometry,
        EvaluatePatrolCurve(parameters.geometry,
            projected.arclength + derivativeHalfWidth),
        assignment.formationSlot);
    float const north = next.north - previous.north;
    float const east = next.east - previous.east;
    float const length = std::hypot(north, east);
    if (length > 1.e-3F)
        pose.heading = std::atan2(east, north);
    float const nominalSpeed = parameters.cruiseSpeed
        * length / (2.F * derivativeHalfWidth);
    return {pose, Clamp(nominalSpeed,
        0.5F * parameters.cruiseSpeed, parameters.maxSpeed)};
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
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    BlueTeamRoles const& roles, std::size_t const physicalIndex,
    PatrolNoThreatSettings const& settings) noexcept
{
    GeneralizedPatrolTarget const target = GeneralizedNoThreatTarget(
        parameters, state, roles, physicalIndex);
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

/**
 * @brief Weak bounded no-threat reward; positive only inside all tolerances.
 * A dead mobile never receives a synthetic position-based reward.
 */
inline float GeneralizedNoThreatShaping(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    BlueTeamRoles const& roles, std::size_t const physicalIndex,
    PatrolNoThreatSettings const& settings) noexcept
{
    if (!settings.enabled || !state.blues[physicalIndex].available)
        return 0.F;
    GeneralizedPatrolError const error = GeneralizedNoThreatError(
        parameters, state, roles, physicalIndex, settings);
    return error.InTolerance() ? 0.25F * settings.stepRewardScale
        : -settings.stepRewardScale * error.BoundedCost();
}

/** @brief All living participants must remain outside each other's sphere. */
inline bool GeneralizedPatrolSpacingSafe(
    GeneralizedRuntimeState const& state,
    float const minimumSeparationMeters) noexcept
{
    for (std::size_t i = 0U; i < state.counts.blueMobiles; ++i)
    {
        if (!state.blues[i].available)
            return false;
        for (std::size_t j = i + 1U; j < state.counts.blueMobiles; ++j)
        {
            Mobile const& first = state.blues[i];
            Mobile const& second = state.blues[j];
            if (!second.available ||
                std::hypot(first.north - second.north,
                           first.east - second.east) < minimumSeparationMeters)
                return false;
        }
    }
    return true;
}

/**
 * @brief Accumulates one full physically attainable correctly flown lap.
 *
 * Bad position/velocity/heading, unsafe separation, reversing direction or
 * an implausible arc jump reset the qualifying distance. No fixed-timestep
 * stationary hold or nearest-point wrap-around can yield mission success.
 */
class GeneralizedPatrolLapProgress final
{
public:
    void Reset() noexcept
    {
        distanceMeters_ = 0.F;
        completed_ = false;
    }

    bool Completed() const noexcept { return completed_; }

    bool Observe(
        Parameters const& parameters,
        GeneralizedRuntimeState const& before,
        GeneralizedRuntimeState const& after,
        BlueTeamRoles const& roles,
        PatrolNoThreatSettings const& settings) noexcept
    {
        if (!settings.enabled || !GeneralizedPatrolSpacingSafe(
                after, std::max(settings.minimumSeparationMeters,
                                parameters.collisionDistance)))
        {
            Reset();
            return false;
        }
        for (std::size_t i = 0U; i < after.counts.blueMobiles; ++i)
        {
            if (!GeneralizedNoThreatError(
                    parameters, after, roles, i, settings).InTolerance())
            {
                Reset();
                return false;
            }
        }
        std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
        Mobile const& leaderBefore = before.blues[leaderIndex];
        Mobile const& leaderAfter = after.blues[leaderIndex];
        float const circuit = CircuitLength(parameters.geometry);
        float const oldArc = ProjectOnPatrolCurve(
            parameters.geometry, leaderBefore.north, leaderBefore.east)
            .pose.arclength;
        float const newArc = ProjectOnPatrolCurve(
            parameters.geometry, leaderAfter.north, leaderAfter.east)
            .pose.arclength;
        float const forward = std::remainder(newArc - oldArc, circuit);
        float const maximumPhysicalProgress =
            1.5F * parameters.maxSpeed * parameters.dt;
        if (forward <= 0.F || forward > maximumPhysicalProgress)
        {
            Reset();
            return false;
        }
        distanceMeters_ += forward;
        // Float integration over thousands of steps can be slightly below
        // the exact analytical circumference at the finish line.
        completed_ = distanceMeters_ >= circuit - 0.25F;
        return completed_;
    }

private:
    float distanceMeters_{};
    bool completed_{};
};

/** @brief Retain patrol outside engagement and relax it under pressure. */
inline float GeneralizedCooperationWeight(
    Parameters const& parameters, GeneralizedRuntimeState const& state) noexcept
{
    return 1.F - 0.75F * GeneralizedEngagementWeight(parameters, state);
}

} // namespace ppo::examples::bluedefense::defenders
