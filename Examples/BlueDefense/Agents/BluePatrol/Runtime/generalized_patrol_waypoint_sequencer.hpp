/**
 * @file generalized_patrol_waypoint_sequencer.hpp
 * @brief Allocation-free RALLY/FOLLOW sequencer for one ten-waypoint patrol lap.
 */
#pragma once

#include "generalized_runtime_state.hpp"
#include "patrol_training_settings.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ppo::examples::bluedefense::defenders {

/** @brief No-threat patrol guidance phase, with recovery after invalidation. */
enum class GeneralizedPatrolMode : std::uint8_t
{
    Rally,
    Follow
};

/** @brief Result of observing one resolved physics step. */
enum class GeneralizedWaypointUpdate : std::uint8_t
{
    None,
    EnteredFollow,
    Advanced,
    Completed,
    Invalidated
};

/** @brief One immutable waypoint generated at the RALLY/FOLLOW transition. */
struct GeneralizedPatrolWaypoint final
{
    /** @brief Curve pose including tangent heading and signed curvature. */
    PatrolCurvePose pose{};
    /** @brief Theoretical leader speed at this waypoint. */
    float speed{};
    /** @brief Required forward distance from the end-of-rally projection. */
    float requiredDistanceMeters{};
};

/**
 * @brief Owns the leader's ten immutable no-threat patrol waypoints.
 *
 * Track measurement always comes from the instantaneous analytical projection;
 * only observation guidance is fixed to the active waypoint. The active index
 * is monotonic and advances at most once per physics step.
 */
class GeneralizedPatrolWaypointSequencer final
{
public:
    /** @brief Exact number of passage zones spanning one complete lap. */
    static constexpr std::size_t WaypointCount = 10U;

    /** @brief Clears all episode state and returns to RALLY. */
    void Reset() noexcept { *this = {}; }

    /** @brief Returns the current irreversible guidance mode. */
    GeneralizedPatrolMode Mode() const noexcept { return mode_; }

    /** @brief Returns true after the rally corridor has been acquired. */
    bool Following() const noexcept
    {
        return mode_ == GeneralizedPatrolMode::Follow;
    }

    /** @brief Returns true once WPT10 has been physically validated. */
    bool Completed() const noexcept { return completed_; }

    /** @brief Returns true while no shortcut/reverse invalidated this lap. */
    bool CircuitValid() const noexcept { return circuitValid_; }

    /** @brief Zero-based active waypoint; ten means the sequence is complete. */
    std::size_t ActiveWaypointIndex() const noexcept
    {
        return activeWaypointIndex_;
    }

    /** @brief Number of waypoints validated in strict WPT1..WPT10 order. */
    std::size_t ValidatedWaypointCount() const noexcept
    {
        return validatedWaypointCount_;
    }

    /** @brief Physically plausible projected forward distance since RALLY. */
    float DistanceMeters() const noexcept { return distanceMeters_; }

    /** @brief Fixed sequence used by policy guidance and visualization. */
    std::array<GeneralizedPatrolWaypoint, WaypointCount> const&
    Waypoints() const noexcept
    {
        return waypoints_;
    }

    /**
     * @brief Returns the active waypoint, or null before FOLLOW/after WPT10.
     * @return Pointer into fixed episode storage; never dynamically allocated.
     */
    GeneralizedPatrolWaypoint const* ActiveWaypoint() const noexcept
    {
        return Following() && activeWaypointIndex_ < WaypointCount
            ? &waypoints_[activeWaypointIndex_] : nullptr;
    }

    /**
     * @brief Builds targets for the current mode without mutating the sequence.
     * @param parameters Current patrol and vehicle parameters.
     * @param state Current fixed-capacity physical snapshot.
     * @param roles Immutable episode role assignment.
     * @param settings Valid no-threat settings.
     * @return Legacy projection-preview targets in RALLY/combat, otherwise the
     * fixed active leader waypoint and its derived wedge poses.
     */
    GeneralizedPatrolTargets Targets(
        Parameters const& parameters, GeneralizedRuntimeState const& state,
        BlueTeamRoles const& roles,
        PatrolNoThreatSettings const& settings) const noexcept
    {
        GeneralizedPatrolWaypoint const* const active = ActiveWaypoint();
        if (state.counts.redMobiles != 0U || active == nullptr)
        {
            return ComputePatrolTargets(
                parameters, state, roles, settings.lookaheadSeconds);
        }
        return ComputePatrolWaypointTargets(
            parameters, state, roles, active->pose);
    }

    /**
     * @brief Observes one resolved step and advances at most one waypoint.
     * @param parameters Current patrol and vehicle parameters.
     * @param before State before physics.
     * @param after State after physics.
     * @param roles Immutable leader/wedge role assignment.
     * @param settings RALLY corridor and tracking tolerances.
     * @return State-machine event produced by this step.
     */
    GeneralizedWaypointUpdate Observe(
        Parameters const& parameters, GeneralizedRuntimeState const& before,
        GeneralizedRuntimeState const& after, BlueTeamRoles const& roles,
        PatrolNoThreatSettings const& settings) noexcept
    {
        if (!Valid(parameters) || !Valid(settings) || !Valid(before)
            || !Valid(after) || !roles.IsValid()
            || roles.ActiveBlueCount() != before.counts.blueMobiles
            || roles.ActiveBlueCount() != after.counts.blueMobiles
            || before.counts.blueMobiles != after.counts.blueMobiles
            || before.counts.redMobiles != 0U
            || after.counts.redMobiles != 0U)
        {
            return Invalidate();
        }

        std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
        Mobile const& previousLeader = before.blues[leaderIndex];
        Mobile const& currentLeader = after.blues[leaderIndex];
        if (!previousLeader.available || !currentLeader.available)
            return Invalidate();

        PatrolProjection const previousProjection = ProjectOnPatrolCurve(
            parameters.geometry, previousLeader.north, previousLeader.east);
        PatrolProjection const currentProjection = ProjectOnPatrolCurve(
            parameters.geometry, currentLeader.north, currentLeader.east);

        if (!Following())
        {
            bool const previouslyAcquired = rallyCorridorLatched_;
            UpdateRallyCorridorLatch(
                currentProjection.distance, settings);
            // Confirm acquisition on the following sample, allowing the
            // deadband to retain it and the exit threshold to cancel it.
            if (!previouslyAcquired || !rallyCorridorLatched_)
                return GeneralizedWaypointUpdate::None;
            Generate(parameters, currentProjection.pose);
            return GeneralizedWaypointUpdate::EnteredFollow;
        }

        float const maximumStep = 1.5F * parameters.maxSpeed
            * parameters.dt + 1.F;
        float const displacement = std::hypot(
            currentLeader.north - previousLeader.north,
            currentLeader.east - previousLeader.east);
        float const forward = std::remainder(
            currentProjection.pose.arclength
                - previousProjection.pose.arclength,
            CircuitLength(parameters.geometry));
        float const reverseTolerance = std::max(
            1.F, 0.05F * currentLeader.speed * parameters.dt);
        if (forward < -reverseTolerance || forward > maximumStep
            || displacement > maximumStep)
        {
            return Invalidate();
        }
        if (completed_ || !circuitValid_)
            return GeneralizedWaypointUpdate::None;
        distanceMeters_ = std::max(0.F, distanceMeters_ + forward);
        if (forward <= 0.F)
            return GeneralizedWaypointUpdate::None;
        GeneralizedPatrolWaypoint const& active =
            waypoints_[activeWaypointIndex_];
        float const spacing = CircuitLength(parameters.geometry)
            / static_cast<float>(WaypointCount);
        float const acceptanceRadius = std::min(
            0.25F * spacing,
            std::max(0.4F * settings.positionToleranceMeters,
                2.F * currentLeader.speed * parameters.dt));
        bool const finalWaypoint =
            activeWaypointIndex_ + 1U == WaypointCount;
        float const requiredProgress = finalWaypoint
            ? CircuitLength(parameters.geometry) - 0.25F
            : active.requiredDistanceMeters - acceptanceRadius;
        if (distanceMeters_ < requiredProgress)
            return GeneralizedWaypointUpdate::None;
        if (!PassesWaypointZone(
                previousLeader, currentLeader, active.pose, acceptanceRadius))
        {
            // A missed passage cannot hold guidance on a stale waypoint.
            // Reacquire a fresh lap rather than crediting a skipped gate.
            return distanceMeters_ > active.requiredDistanceMeters + acceptanceRadius
                ? Invalidate() : GeneralizedWaypointUpdate::None;
        }

        ++validatedWaypointCount_;
        ++activeWaypointIndex_; // Exactly one advancement per resolved step.
        if (activeWaypointIndex_ == WaypointCount)
        {
            completed_ = true;
            return GeneralizedWaypointUpdate::Completed;
        }
        return GeneralizedWaypointUpdate::Advanced;
    }

    /**
     * @brief Euclidean leader distance to the active guidance point.
     * @param state Current runtime state.
     * @param roles Immutable role assignment.
     * @return Metres, or zero when no active waypoint is available.
     */
    float DistanceToActiveWaypoint(
        GeneralizedRuntimeState const& state,
        BlueTeamRoles const& roles) const noexcept
    {
        GeneralizedPatrolWaypoint const* const active = ActiveWaypoint();
        if (active == nullptr || !roles.IsValid()
            || roles.LeaderPhysicalIndex() >= state.counts.blueMobiles)
        {
            return 0.F;
        }
        Mobile const& leader = state.blues[roles.LeaderPhysicalIndex()];
        return std::hypot(
            leader.north - active->pose.north,
            leader.east - active->pose.east);
    }

private:
    /**
     * @brief Arms RALLY acquisition inside enter and cancels it outside exit.
     * @param distance Current analytical distance to the patrol curve.
     * @param settings Inner acquisition and outer release thresholds.
     */
    void UpdateRallyCorridorLatch(
        float const distance,
        PatrolNoThreatSettings const& settings) noexcept
    {
        if (distance > settings.rallyExitDistanceMeters)
            rallyCorridorLatched_ = false;
        else if (distance <= settings.rallyEnterDistanceMeters)
            rallyCorridorLatched_ = true;
    }

    /** @brief Generates ten equal-arclength waypoints for one complete lap. */
    void Generate(Parameters const& parameters,
                  PatrolCurvePose const& start) noexcept
    {
        float const circuit = CircuitLength(parameters.geometry);
        float const spacing = circuit / static_cast<float>(WaypointCount);
        for (std::size_t index = 0U; index < WaypointCount; ++index)
        {
            float const distance = spacing * static_cast<float>(index + 1U);
            waypoints_[index] = {
                EvaluatePatrolCurve(
                    parameters.geometry, start.arclength + distance),
                parameters.cruiseSpeed,
                distance};
        }
        mode_ = GeneralizedPatrolMode::Follow;
        activeWaypointIndex_ = 0U;
        validatedWaypointCount_ = 0U;
        distanceMeters_ = 0.F;
        circuitValid_ = true;
        completed_ = false;
    }

    /** @brief Returns the minimum distance from a step segment to a point. */
    static float SegmentDistance(
        Mobile const& before, Mobile const& after,
        PatrolCurvePose const& point) noexcept
    {
        float const stepNorth = after.north - before.north;
        float const stepEast = after.east - before.east;
        float const lengthSquared = stepNorth * stepNorth
            + stepEast * stepEast;
        float const fraction = lengthSquared > 1.e-6F
            ? Clamp(((point.north - before.north) * stepNorth
                    + (point.east - before.east) * stepEast)
                    / lengthSquared, 0.F, 1.F)
            : 0.F;
        return std::hypot(
            before.north + fraction * stepNorth - point.north,
            before.east + fraction * stepEast - point.east);
    }

    /** @brief Checks the directed tangent passage zone of one waypoint. */
    static bool PassesWaypointZone(
        Mobile const& before, Mobile const& after,
        PatrolCurvePose const& waypoint,
        float const acceptanceRadius) noexcept
    {
        float const tangentNorth = std::cos(waypoint.heading);
        float const tangentEast = std::sin(waypoint.heading);
        float const beforeAlong = (before.north - waypoint.north)
                * tangentNorth
            + (before.east - waypoint.east) * tangentEast;
        float const afterAlong = (after.north - waypoint.north)
                * tangentNorth
            + (after.east - waypoint.east) * tangentEast;
        return beforeAlong <= acceptanceRadius
            && afterAlong >= -acceptanceRadius
            && SegmentDistance(before, after, waypoint) <= acceptanceRadius;
    }

    /** @brief Invalidates the current lap and restores rally guidance. */
    GeneralizedWaypointUpdate Invalidate() noexcept
    {
        if (Following())
        {
            circuitValid_ = false;
            completed_ = false;
            mode_ = GeneralizedPatrolMode::Rally;
            rallyCorridorLatched_ = false;
        }
        return GeneralizedWaypointUpdate::Invalidated;
    }

    std::array<GeneralizedPatrolWaypoint, WaypointCount> waypoints_{};
    GeneralizedPatrolMode mode_{GeneralizedPatrolMode::Rally};
    std::size_t activeWaypointIndex_{};
    std::size_t validatedWaypointCount_{};
    float distanceMeters_{};
    bool rallyCorridorLatched_{};
    bool circuitValid_{true};
    bool completed_{};
};

static_assert(std::is_trivially_copyable_v<GeneralizedPatrolWaypointSequencer>,
    "The patrol sequencer must remain allocation-free fixed-capacity state");

} // namespace ppo::examples::bluedefense::defenders
