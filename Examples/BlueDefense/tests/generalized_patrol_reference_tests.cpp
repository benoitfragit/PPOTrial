/** @file
 * @brief Deterministic v10 references, physical laps and V3 reward regressions.
 */
#include "generalized_reward_shaping.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace blue = ppo::examples::bluedefense::defenders;
namespace {
bool Near(float a, float b, float epsilon = 0.02F)
{
    return std::abs(a - b) <= epsilon;
}

blue::Parameters Parameters()
{
    blue::Parameters parameters{};
    parameters.cruiseSpeed = 300.F;
    parameters.maxSpeed = 750.F;
    parameters.geometry.length = 40000.F;
    parameters.geometry.width = 16000.F;
    parameters.geometry.separationDistance = 2800.F;
    parameters.geometry.separationAngle = 0.55F;
    return parameters;
}

blue::GeneralizedRuntimeState OnTargets(
    blue::Parameters const& parameters, blue::BlueTeamRoles const& roles,
    float const arc)
{
    blue::GeneralizedRuntimeState state{};
    state.counts = {roles.ActiveBlueCount(), 0U};
    auto const pose = blue::EvaluatePatrolCurve(parameters.geometry, arc);
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
        state.blues[index] = {pose.north, pose.east, pose.heading,
            parameters.cruiseSpeed, 0.F, true};
    auto const targets = blue::ComputePatrolTargets(parameters, state, roles);
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        auto const& target = targets[index];
        state.blues[index] = {target.pose.north, target.pose.east,
            target.pose.heading, target.speed, 0.F, true};
    }
    return state;
}

bool References()
{
    auto parameters = Parameters();
    for (int direction : {-1, 1})
    {
        parameters.geometry.direction = direction;
        for (std::size_t count : {1U, 2U, 3U, 9U})
        {
            auto const roles = *blue::BlueTeamRoles::Create(count, count - 1U);
            for (float phase : {0.05F, 0.27F, 0.51F, 0.79F})
            {
                float const arc = phase * blue::CircuitLength(parameters.geometry);
                auto state = OnTargets(parameters, roles, arc);
                auto const targets = blue::ComputePatrolTargets(parameters, state, roles);
                std::array<blue::GeneralizedObservation, 9> observations{};
                if (!blue::BuildGeneralizedObservations(
                        parameters, state, roles, observations, &targets))
                    return false;
                for (std::size_t index = 0U; index < count; ++index)
                {
                    auto const& target = targets[index];
                    if (!target.valid || !std::isfinite(target.speed)
                        || !blue::GeneralizedNoThreatError(
                            state, target, index, {}).InTolerance())
                        return false;
                    auto const offset = blue::GeneralizedObservationLayout::PatrolOffset;
                    if (!Near(observations[index].values[offset + 7U], 1.F)
                        || !Near(observations[index].values[offset],
                            (target.guidancePose.north - state.blues[index].north)
                                / (2.F * parameters.halfNorth)))
                        return false;
                    if (index != roles.LeaderPhysicalIndex())
                    {
                        auto const slot = roles.Assignment(index)->formationSlot;
                        auto const from = blue::EvaluateFormationPose(parameters.geometry,
                            blue::EvaluatePatrolCurve(parameters.geometry, arc - 2.F), slot);
                        auto const to = blue::EvaluateFormationPose(parameters.geometry,
                            blue::EvaluatePatrolCurve(parameters.geometry, arc + 2.F), slot);
                        float const northVelocity = (to.north - from.north)
                            * parameters.cruiseSpeed / 4.F;
                        float const eastVelocity = (to.east - from.east)
                            * parameters.cruiseSpeed / 4.F;
                        if (!Near(target.speed * std::cos(target.pose.heading), northVelocity, 0.7F)
                            || !Near(target.speed * std::sin(target.pose.heading), eastVelocity, 0.7F))
                            return false;
                    }
                }
                for (std::size_t index = count; index < 9U; ++index)
                    if (targets[index].valid)
                        return false;
                // Availability and priority are independent: a distant active
                // red leaves the reference intact; delayed reds remain masked.
                state.counts.redMobiles = 1U;
                state.reds[0] = {250000.F, 250000.F, 0.F, 300.F, 0.F, true};
                if (!Near(blue::GeneralizedEngagementWeight(parameters, state), 0.F))
                    return false;
                auto const distant = blue::ComputePatrolTargets(parameters, state, roles);
                if (!Near(distant[0].pose.north, targets[0].pose.north))
                    return false;
                state.reds[0].north = state.blues[0].north;
                state.reds[0].east = state.blues[0].east;
                if (blue::GeneralizedEngagementWeight(parameters, state) < 0.99F)
                    return false;
                state.reds[0].available = false;
                state.blues[roles.LeaderPhysicalIndex()].available = false;
                auto const lost = blue::ComputePatrolTargets(parameters, state, roles);
                if (!blue::BuildGeneralizedObservations(
                        parameters, state, roles, observations, &lost))
                    return false;
                for (std::size_t index = 0U; index < count; ++index)
                {
                    if (lost[index].valid)
                        return false;
                    for (std::size_t feature = 0U; feature < 8U; ++feature)
                        if (observations[index].values[
                                blue::GeneralizedObservationLayout::PatrolOffset + feature] != 0.F)
                            return false;
                }
            }
        }
    }
    auto state = OnTargets(parameters, *blue::BlueTeamRoles::Create(1U, 0U), 0.F);
    state.blues[0].north = std::numeric_limits<float>::quiet_NaN();
    return !blue::ComputePatrolTargets(
        parameters, state, *blue::BlueTeamRoles::Create(1U, 0U))[0].valid;
}

bool Feasibility()
{
    auto parameters = Parameters();
    blue::PatrolGeometrySettings settings{};
    blue::PatrolSamplingConstraints constraints{parameters.halfNorth,
        parameters.halfEast, 300.F, 0.8F, 1000.F, 2500.F, 750.F};
    for (std::size_t count : {1U, 2U, 3U, 9U})
    {
        if (!blue::GeneralizedFormationCircuitFeasible(
                parameters.geometry, settings, constraints, count))
            return false;
        for (std::size_t sample = 0U; sample < 1024U; ++sample)
            if (!blue::GeneralizedFormationFitsBounds(parameters.geometry,
                    settings, constraints, count,
                    blue::CircuitLength(parameters.geometry) * sample / 1024.F))
                return false;
    }
    for (std::size_t count : {1U, 2U, 3U, 9U})
        for (unsigned seed = 0U; seed < 64U; ++seed)
        {
            std::mt19937 rng{seed};
            if (!blue::SampleGeneralizedPatrolStart(settings, constraints, count, rng))
                return false;
        }
    auto impossible = parameters.geometry;
    impossible.separationDistance = 6000.F;
    impossible.separationAngle = 11.F * blue::kPatrolPi / 180.F;
    if (blue::GeneralizedFormationCircuitFeasible(impossible, settings, constraints, 3U))
        return false;
    constraints.maximumSpeed = 310.F;
    return !blue::GeneralizedFormationCircuitFeasible(
        parameters.geometry, settings, constraints, 9U);
}

bool ShapingAndLaps()
{
    auto parameters = Parameters();
    auto const roles = *blue::BlueTeamRoles::Create(3U, 1U);
    blue::PatrolNoThreatSettings settings{};
    settings.enabled = true;
    settings.stepRewardScale = 0.004F;
    settings.minimumQualifiedFraction = 0.95F;
    auto before = OnTargets(parameters, roles, 2000.F);
    auto after = OnTargets(parameters, roles, 2060.F);
    auto const oldTargets = blue::ComputePatrolTargets(parameters, before, roles);
    auto const newTargets = blue::ComputePatrolTargets(parameters, after, roles);
    for (std::size_t index = 0U; index < 3U; ++index)
    {
        float const forward = blue::GeneralizedPatrolTransitionShaping(
            parameters, before, after, oldTargets, newTargets, roles, index, settings);
        float const parked = blue::GeneralizedPatrolTransitionShaping(
            parameters, before, before, oldTargets, oldTargets, roles, index, settings);
        if (forward < 0.0035F || parked != 0.F || forward > 0.0041F)
            return false;
    }
    auto poor = after;
    poor.blues[0].north += 4000.F;
    auto improving = poor;
    improving.blues[0].north -= 50.F;
    auto const poorTargets = blue::ComputePatrolTargets(parameters, poor, roles);
    auto const improvingTargets = blue::ComputePatrolTargets(parameters, improving, roles);
    if (blue::GeneralizedPatrolTransitionShaping(parameters, poor, improving,
            poorTargets, improvingTargets, roles, 0U, settings)
        <= blue::GeneralizedPatrolTransitionShaping(parameters, improving, poor,
            improvingTargets, poorTargets, roles, 0U, settings))
        return false;
    blue::GeneralizedPatrolLapProgress lap{};
    constexpr std::size_t segments = 2000U;
    float const circuit = blue::CircuitLength(parameters.geometry);
    // A brief speed transient must not erase the whole lap. A consistently
    // wrong speed must never qualify. Test both turn directions and all slots.
    for (int direction : {-1, 1})
    {
        parameters.geometry.direction = direction;
        for (bool transient : {false, true})
        {
            lap.Reset();
            bool completed = false;
            for (std::size_t step = 0U; step < segments; ++step)
            {
                before = OnTargets(parameters, roles, circuit * step / segments);
                after = OnTargets(parameters, roles, circuit * (step + 1U) / segments);
                if (transient && step < 20U)
                    after.blues[roles.LeaderPhysicalIndex()].speed = 0.F;
                completed = lap.Observe(parameters, before, after, roles, settings);
                if (completed && step + 1U != segments)
                    return false;
            }
            if (!completed || lap.QualifiedDistanceMeters() < 0.95F * circuit)
                return false;
        }
    }
    lap.Reset();
    for (std::size_t step = 0U; step < segments; ++step)
    {
        before = OnTargets(parameters, roles, circuit * step / segments);
        after = OnTargets(parameters, roles, circuit * (step + 1U) / segments);
        after.blues[roles.LeaderPhysicalIndex()].speed = 0.F;
        if (lap.Observe(parameters, before, after, roles, settings))
            return false;
    }
    lap.Reset();
    before = OnTargets(parameters, roles, 1000.F);
    after = before;
    if (lap.Observe(parameters, before, after, roles, settings))
        return false;
    after.blues[0] = after.blues[1]; // unsafe separation
    if (lap.Observe(parameters, before, after, roles, settings)
        || lap.DistanceMeters() != 0.F)
        return false;
    after.blues[1].available = false;
    if (!blue::GeneralizedNoThreatTeamLost(after, roles.LeaderPhysicalIndex())
        || lap.Observe(parameters, before, after, roles, settings))
        return false;
    after.counts.redMobiles = 1U;
    return !blue::GeneralizedNoThreatTeamLost(after, roles.LeaderPhysicalIndex());
}

/** @brief Convergence uses flight-step scales, not global-world distances. */
bool TestConvergenceAndSingleRed()
{
    auto parameters = Parameters();
    blue::PatrolNoThreatSettings settings{};
    settings.stepRewardScale = 0.004F;
    for (int direction : {-1, 1})
    {
        parameters.geometry.direction = direction;
        for (std::size_t count : {1U, 3U})
        {
            auto const roles = *blue::BlueTeamRoles::Create(count, count - 1U);
            auto const before = OnTargets(parameters, roles, 2000.F);
            auto const after = OnTargets(parameters, roles, 2060.F);
            auto const beforeTargets = blue::ComputePatrolTargets(parameters, before, roles);
            auto const afterTargets = blue::ComputePatrolTargets(parameters, after, roles);
            std::size_t const leader = roles.LeaderPhysicalIndex();
            float const perfect = blue::GeneralizedPatrolConvergenceShaping(
                parameters, before, after, beforeTargets, afterTargets, roles, leader, settings);
            // 60 m at cruise speed must have the same effective scale as
            // 2 * the single-Red progress channel (60 / 150 = 0.4).
            if (perfect < 0.75F || perfect > 0.85F)
                return false;
            auto overspeed = OnTargets(parameters, roles, 2150.F);
            overspeed.blues[leader].speed = 750.F;
            float const fastReward = blue::GeneralizedPatrolConvergenceShaping(
                parameters, before, overspeed, beforeTargets,
                blue::ComputePatrolTargets(parameters, overspeed, roles),
                roles, leader, settings);
            if (!(fastReward < perfect && fastReward < 0.75F))
                return false;
            auto offset = before;
            float const heading = offset.blues[leader].heading;
            float const nNorth = -std::sin(heading);
            float const nEast = std::cos(heading);
            offset.blues[leader].north += 2500.F * nNorth;
            offset.blues[leader].east += 2500.F * nEast;
            auto const offsetTargets = blue::ComputePatrolTargets(parameters, offset, roles);
            auto parallel = offset;
            parallel.blues[leader].north += 60.F * std::cos(heading);
            parallel.blues[leader].east += 60.F * std::sin(heading);
            auto closer = offset;
            closer.blues[leader].north -= 50.F * nNorth;
            closer.blues[leader].east -= 50.F * nEast;
            auto farther = offset;
            farther.blues[leader].north += 50.F * nNorth;
            farther.blues[leader].east += 50.F * nEast;
            auto reward = [&](blue::GeneralizedRuntimeState const& state) {
                return blue::GeneralizedPatrolConvergenceShaping(
                    parameters, offset, state, offsetTargets,
                    blue::ComputePatrolTargets(parameters, state, roles),
                    roles, leader, settings);
            };
            // A 2.5-km lateral offset must NOT earn a positive reward for
            // travelling parallel to the curve while never converging.
            if (!(reward(parallel) <= 0.F && reward(closer) > 0.1F
                    && reward(farther) < -0.1F))
                return false;

            auto combat = before;
            combat.counts.redMobiles = 1U;
            combat.reds[0] = {combat.blues[leader].north + 10000.F,
                combat.blues[leader].east, 0.F, 220.F, 0.F, true};
            auto towards = combat;
            towards.blues[leader].north += 60.F;
            auto away = combat;
            away.blues[leader].north -= 60.F;
            auto redMotionOnly = combat;
            redMotionOnly.reds[0].north -= 100.F;
            if (blue::GeneralizedSingleRedLeaderProgress(
                    parameters, combat, towards, roles) < 0.39F
                || blue::GeneralizedSingleRedLeaderProgress(
                    parameters, combat, away, roles) > -0.39F
                || blue::GeneralizedSingleRedLeaderProgress(
                    parameters, combat, redMotionOnly, roles) != 0.F)
                return false;
            combat.counts.redMobiles = 2U;
            towards.counts.redMobiles = 2U;
            combat.reds[1] = {12000.F, -10000.F, 0.F, 220.F, 0.F, true};
            towards.reds[1] = combat.reds[1];
            if (blue::GeneralizedSingleRedLeaderProgress(
                    parameters, combat, towards, roles) != 0.F)
                return false;
        }
    }
    return true;
}

/** @brief Moves only the leader along its local track normal, in metres. */
void OffsetLeader(
    blue::GeneralizedRuntimeState& state, blue::BlueTeamRoles const& roles,
    float const distanceMeters) noexcept
{
    auto& leader = state.blues[roles.LeaderPhysicalIndex()];
    leader.north -= distanceMeters * std::sin(leader.heading);
    leader.east += distanceMeters * std::cos(leader.heading);
}

/** @brief Fixed guidance must preserve every legacy track measurement. */
bool SamePatrolMeasurements(
    blue::GeneralizedRuntimeState const& state,
    blue::GeneralizedPatrolTargets const& legacy,
    blue::GeneralizedPatrolTargets const& fixed,
    blue::PatrolNoThreatSettings const& settings) noexcept
{
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        auto const& from = legacy[index];
        auto const& to = fixed[index];
        auto const previous = blue::GeneralizedNoThreatError(
            state, from, index, settings);
        auto const current = blue::GeneralizedNoThreatError(
            state, to, index, settings);
        if (!from.valid || !to.valid
            || !Near(from.pose.north, to.pose.north, 0.0001F)
            || !Near(from.pose.east, to.pose.east, 0.0001F)
            || !Near(from.pose.heading, to.pose.heading, 0.0001F)
            || !Near(from.pose.arclength, to.pose.arclength, 0.0001F)
            || !Near(from.pose.phase, to.pose.phase, 0.0001F)
            || !Near(from.pose.curvature, to.pose.curvature, 0.0001F)
            || !Near(from.speed, to.speed, 0.0001F)
            || !Near(previous.position, current.position, 0.0001F)
            || !Near(previous.speed, current.speed, 0.0001F)
            || !Near(previous.heading, current.heading, 0.0001F))
            return false;
    }
    return true;
}

/**
 * @brief Checks numerical rewards and isolates a waypoint change after physics.
 * @param startArc Arclength at which the ten-waypoint FOLLOW lap began.
 * @param expectedConvergence Independently calculated dense convergence reward.
 * @param expectedTransition Independently calculated tracking transition reward.
 */
bool CheckFrozenWaypointRewards(
    blue::Parameters const& parameters, blue::BlueTeamRoles const& roles,
    blue::GeneralizedRuntimeState const& before,
    blue::GeneralizedRuntimeState const& after, float const startArc,
    float const expectedConvergence, float const expectedTransition)
{
    blue::PatrolNoThreatSettings settings{};
    settings.stepRewardScale = 0.004F;
    float const spacing = blue::CircuitLength(parameters.geometry) / 10.F;
    auto const waypoint = blue::EvaluatePatrolCurve(
        parameters.geometry, startArc + spacing);
    auto const nextWaypoint = blue::EvaluatePatrolCurve(
        parameters.geometry, startArc + 2.F * spacing);
    auto const beforeTargets = blue::ComputePatrolWaypointTargets(
        parameters, before, roles, waypoint);
    auto const afterTargets = blue::ComputePatrolWaypointTargets(
        parameters, after, roles, waypoint);
    auto const advancedTargets = blue::ComputePatrolWaypointTargets(
        parameters, after, roles, nextWaypoint);
    if (!SamePatrolMeasurements(before,
            blue::ComputePatrolTargets(parameters, before, roles), beforeTargets, settings)
        || !SamePatrolMeasurements(after,
            blue::ComputePatrolTargets(parameters, after, roles), afterTargets, settings)
        || !SamePatrolMeasurements(after, afterTargets, advancedTargets, settings))
        return false;

    std::size_t const leader = roles.LeaderPhysicalIndex();
    float const convergence = blue::GeneralizedPatrolConvergenceShaping(
        parameters, before, after, beforeTargets, afterTargets, roles, leader, settings);
    float const transition = blue::GeneralizedPatrolTransitionShaping(
        parameters, before, after, beforeTargets, afterTargets, roles, leader, settings);
    if (!Near(convergence, expectedConvergence, 0.00015F)
        || !Near(transition, expectedTransition, 0.000002F))
    {
        std::cerr << "frozen rewards: " << convergence << "/" << transition
            << " expected " << expectedConvergence << "/" << expectedTransition << '\n';
        return false;
    }
    // A newly selected after-step waypoint changes the next observation, not
    // the reward for movement made towards the beginning-of-step waypoint.
    for (std::size_t index = 0U; index < before.counts.blueMobiles; ++index)
    {
        float const originalConvergence = blue::GeneralizedPatrolConvergenceShaping(
            parameters, before, after, beforeTargets, afterTargets, roles, index, settings);
        float const originalTransition = blue::GeneralizedPatrolTransitionShaping(
            parameters, before, after, beforeTargets, afterTargets, roles, index, settings);
        if (!Near(originalConvergence, blue::GeneralizedPatrolConvergenceShaping(
                parameters, before, after, beforeTargets, advancedTargets,
                roles, index, settings), 0.000001F)
            || !Near(originalTransition, blue::GeneralizedPatrolTransitionShaping(
                parameters, before, after, beforeTargets, advancedTargets,
                roles, index, settings), 0.000001F)
            || std::hypot(afterTargets[index].guidancePose.north
                    - advancedTargets[index].guidancePose.north,
                afterTargets[index].guidancePose.east
                    - advancedTargets[index].guidancePose.east) < 1000.F)
            return false;
    }
    return true;
}

/** @brief Locks fixed-waypoint reward values on and off a straight in both directions. */
bool TestFrozenWaypointRewardRegression()
{
    auto parameters = Parameters();
    parameters.dt = 0.2F;
    parameters.geometry.heading = 0.F;
    for (int direction : {-1, 1})
    {
        parameters.geometry.direction = direction;
        // Both starts and WPT1 lie on a straight; no turn approximation is
        // involved in the independent 60-m / 150-m reward calculations.
        float const start = direction > 0 ? 6000.F
            : blue::CircuitLength(parameters.geometry) - 18000.F;
        for (std::size_t count : {1U, 3U})
        {
            auto const roles = *blue::BlueTeamRoles::Create(count, count - 1U);
            auto const onTrack = OnTargets(parameters, roles, start);
            auto const forward = OnTargets(parameters, roles, start + 60.F);
            if (!CheckFrozenWaypointRewards(parameters, roles, onTrack, forward,
                    start, 0.804F, 0.004F))
                return false;

            auto nearBefore = onTrack;
            auto nearAfter = forward;
            OffsetLeader(nearBefore, roles, 1250.F);
            OffsetLeader(nearAfter, roles, 1250.F);
            if (!CheckFrozenWaypointRewards(parameters, roles, nearBefore, nearAfter,
                    start, 0.3890886F, 0.002166667F))
                return false;

            auto farBefore = onTrack;
            auto farParallel = forward;
            OffsetLeader(farBefore, roles, 3000.F);
            OffsetLeader(farParallel, roles, 3000.F);
            if (!CheckFrozenWaypointRewards(parameters, roles, farBefore, farParallel,
                    start, -0.008162837F, 0.0004545455F))
                return false;
            auto recovering = onTrack;
            OffsetLeader(recovering, roles, 2940.F);
            if (!CheckFrozenWaypointRewards(parameters, roles, farBefore, recovering,
                    start, 0.7998379F, -0.00006133021F))
                return false;
        }
    }
    return true;
}
}

int main()
{
    if (!References()) { std::cerr << "references\n"; return 1; }
    if (!Feasibility()) { std::cerr << "feasibility\n"; return 2; }
    if (!ShapingAndLaps()) { std::cerr << "shaping/laps\n"; return 3; }
    if (!TestConvergenceAndSingleRed()) {
        std::cerr << "convergence/single-red\n"; return 4;
    }
    if (!TestFrozenWaypointRewardRegression()) {
        std::cerr << "frozen-waypoint rewards\n"; return 5;
    }
    return 0;
}
