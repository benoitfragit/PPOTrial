#include "patrol_core.hpp"
#include "patrol_reference_controller.hpp"
#include "ppo/core/team_episode.hpp"

#include <array>
#include <cmath>
#include <iostream>

namespace blue = ppo::examples::twovstwo::defenders;

int main()
{
    static_assert(blue::ObservationLayout::LocalSize == 124U);
    static_assert(blue::ObservationLayout::Size == 156U);
    static_assert(blue::ObservationLayout::ActionSize == 2U);
    blue::Parameters const p{};
    if (!blue::Valid(p))
        return 1;
    float const length = blue::CircuitLength(p);
    if (blue::StraightHalfLength(p) <= 0.F
        || p.geometry.length <= p.geometry.width)
        return 29;
    // Entire orbit stays within world, starts/ends on the same tangent.
    for (int side : {-1, 1})
    {
        auto const first = blue::PatrolReference(p, side, 0.F);
        auto const last = blue::PatrolReference(p, side, length);
        if (std::abs(first.north - last.north) > 0.01F
            || std::abs(first.east - last.east) > 0.01F
            || std::abs(std::sin(first.heading - last.heading)) > 0.001F)
            return 2;
        for (unsigned i = 0; i < 2000U; ++i)
        {
            auto const at = blue::PatrolReference(
                p, side, length * static_cast<float>(i) / 2000.F);
            auto const wingman = blue::WingmanReference(p, at);
            if (std::abs(at.north) >= p.halfNorth
                || std::abs(at.east) >= p.halfEast
                || std::abs(wingman.north) >= p.halfNorth
                || std::abs(wingman.east) >= p.halfEast)
                return 3;
        }
    }
    auto state = blue::InitialState(p, 1, 0U, 0.F);
    auto const r0 = blue::PatrolReference(p, 1, 0.F);
    auto const r1 = blue::WingmanReference(p, r0);
    if (std::hypot(r0.north - r1.north, r0.east - r1.east)
            <= p.collisionDistance
        || blue::FormationGap(p) <= p.collisionDistance)
        return 4; // Rear/inward wedge remains safely separated.
    auto compressedFormation = state;
    compressedFormation.guards[0] = compressedFormation.guards[1];
    if (!(blue::FormationReward(p, state, true)
          > blue::FormationReward(p, compressedFormation, true))
        || blue::FormationReward(p, state, false)
            != blue::SeparationPenalty(p, state))
        return 27;
    for (std::size_t i = 0; i < 61; ++i)
    {
        auto const obs = blue::Observe(p, state);
        if (obs[blue::ObservationLayout::Own + i] < 0.F
            || obs[blue::ObservationLayout::Own + i] > 1.F)
            return 5;
    }
    static_assert(blue::ObservationLayout::Lidar == 61U,
                  "the v7 Actor requires exactly 61 pseudo-LiDAR values");
    auto sensedState = state;
    sensedState.lidarRanges[0].fill(0.25F);
    sensedState.lidarAvailable[0] = true;
    auto const sensedObservation = blue::Observe(p, sensedState);
    for (std::size_t i = 0U; i < blue::ObservationLayout::Lidar; ++i)
        if (sensedObservation[blue::ObservationLayout::Own + i] != 0.25F)
            return 39; // Runtime observations must consume the ECS sensor.
    auto const observation = blue::Observe(p, state);
    auto clockShifted = state;
    clockShifted.arclength += 0.25F * length;
    auto const clockShiftedObservation = blue::Observe(p, clockShifted);
    for (std::size_t i = blue::ObservationLayout::PatrolOffset;
         i < blue::ObservationLayout::GlobalOffset; ++i)
        if (observation[i] != clockShiftedObservation[i])
            return 30; // No observation may follow an artificial clock.
    if (std::abs(observation[blue::ObservationLayout::PatrolOffset] - 1.F)
            > 1.e-4F
        || std::abs(observation[blue::ObservationLayout::PatrolOffset + 1U])
            > 1.e-4F
        || std::abs(observation[blue::ObservationLayout::PatrolOffset + 2U])
            > 1.e-4F
        || observation[blue::ObservationLayout::PatrolOffset + 8U] != -1.F
        || observation[blue::ObservationLayout::PatrolOffset + 9U] != 0.F)
        return 31;
    auto followerState = state;
    followerState.controlledRole = 1U;
    auto const followerObservation = blue::Observe(p, followerState);
    if (followerObservation[blue::ObservationLayout::PatrolOffset + 8U]
            != 1.F)
        return 32;
    for (std::size_t i = 0U; i < 6U; ++i)
        if (followerObservation[blue::ObservationLayout::PatrolOffset + i]
            != 0.F)
            return 36; // The wingman receives no racetrack-relative inputs.
    if (observation[blue::ObservationLayout::PatrolOffset + 6U] != 0.F
        || observation[blue::ObservationLayout::PatrolOffset + 7U] != 0.F)
        return 37; // Wedge errors are reserved for the wingman.
    for (std::size_t i = blue::ObservationLayout::RedOffset;
         i < blue::ObservationLayout::FriendOffset; ++i)
        if (observation[i] != 0.F)
            return 6;
    auto threatState = state;
    threatState.attackers[0] = blue::Mobile{1000.F, -2000.F, 0.2F, 220.F,
                                            0.F, true};
    auto const threatObservation = blue::Observe(p, threatState);
    if (threatObservation[blue::ObservationLayout::RedOffset
            + blue::ObservationLayout::RedFeatures - 1U] != 1.F
        || threatObservation[blue::ObservationLayout::RedOffset
            + 2U * blue::ObservationLayout::RedFeatures - 1U] != 0.F)
        return 18;
    if (observation[blue::ObservationLayout::FriendOffset + 8] != 1.F
        || observation[blue::ObservationLayout::FriendOffset + 17] != 1.F)
        return 7;
    auto lostObjective = state;
    lostObjective.objectiveAvailable = false;
    auto const lostObjectiveObservation = blue::Observe(p, lostObjective);
    if (lostObjectiveObservation[blue::ObservationLayout::FriendOffset + 17] != 0.F)
        return 19;
    auto const next = blue::Advance(p, state, blue::Action{0.F, 0.F});
    if (next.step != state.step + 1
        || next.guards[0].east <= state.guards[0].east
        || !std::isfinite(blue::TeamReward(p, state, {0.F, 0.F}, next)))
        return 8;
    auto const nominalRoleRewards = blue::TrainingRoleRewardBreakdowns(
        p, state, std::array<blue::Action, 2>{}, next);
    if (nominalRoleRewards[0].values[static_cast<std::size_t>(
            blue::RewardChannel::PatrolTracking)] == 0.F
        || nominalRoleRewards[1].values[static_cast<std::size_t>(
            blue::RewardChannel::PatrolTracking)] != 0.F)
        return 38; // Racetrack credit belongs only to the semantic leader.

    auto displacedWingman = state;
    displacedWingman.guards[1].north += 3000.F;
    auto const displacedRoleRewards = blue::TrainingRoleRewardBreakdowns(
        p, displacedWingman, std::array<blue::Action, 2>{},
        displacedWingman);
    if (displacedRoleRewards[0].values[static_cast<std::size_t>(
            blue::RewardChannel::FormationSafety)] != 0.F
        || displacedRoleRewards[1].values[static_cast<std::size_t>(
            blue::RewardChannel::FormationSafety)] >= 0.F)
        return 39; // Wedge error belongs only to the semantic wingman.
    // A stopped agent must not score the same progression as the reference.
    auto stopped = state;
    stopped.guards[0].speed = 0.F;
    if (!(blue::TrackingReward(p, state, stopped)
        < blue::TrackingReward(p, state, next)))
        return 9;
    auto projectionJump = next;
    projectionJump.patrolProgress += 0.5F * length;
    if (blue::TrackingReward(p, state, projectionJump) > 0.F)
        return 33;
    auto backward = state;
    backward.guards[0] = blue::OnReference(
        p, blue::PatrolReference(p, 1, length - 60.F));
    blue::UpdatePatrolProgress(p, backward);
    if (blue::TrackingReward(p, state, backward) >= 0.F)
        return 34;
    auto outside = state;
    outside.guards[0].north = 1.10F * p.halfNorth;
    auto const allowedOutside = blue::Advance(p, outside, {0.F, 0.F});
    if (allowedOutside.failed || !(blue::SoftBoundaryPenalty(p, allowedOutside) < 0.F))
        return 10;
    auto remoteThreat = allowedOutside;
    remoteThreat.attackers[0] = blue::Mobile{
        0.F, -static_cast<float>(remoteThreat.side) * p.halfEast,
        0.F, 200.F, 0.F, true};
    if (!(std::abs(blue::SoftBoundaryPenalty(p, remoteThreat))
          < std::abs(blue::SoftBoundaryPenalty(p, allowedOutside))))
        return 20;
    auto hardOutside = state;
    hardOutside.guards[0].north = p.hardBoundaryScale * p.halfNorth + 1.F;
    auto const failed = blue::Advance(p, hardOutside, {0.F, 0.F});
    if (!failed.failed
        || blue::TeamReward(p, hardOutside, {0.F, 0.F}, failed) != -100.F)
        return 21;
    // A collision must be terminal and have the same severe negative reward
    // even when the teammate is the deterministic, non-PPO reference driver.
    auto colliding = state;
    colliding.guards[0] = colliding.guards[1];
    auto const crashed = blue::Advance(p, colliding, {0.F, 0.F});
    if (!crashed.collided || !crashed.failed
        || blue::TeamReward(p, colliding, {0.F, 0.F}, crashed) != -100.F)
        return 15;

    // Swept collision between two observations, without endpoint overlap.
    auto crossing = state;
    crossing.guards[0].north = -800.F;
    crossing.guards[0].east = 0.F;
    crossing.guards[1].north = 800.F;
    crossing.guards[1].east = 0.F;
    auto crossed = crossing;
    crossed.guards[0].north = 800.F;
    crossed.guards[1].north = -800.F;
    if (blue::GuardSeparation(crossed) <= p.collisionDistance
        || !blue::GuardsCollided(p, crossing, crossed))
        return 16;

    // The shaping penalty applies before contact, without ending the episode.
    auto far = state, close = state;
    far.guards[0].north = 0.F; far.guards[0].east = 0.F;
    far.guards[1].north = 2500.F; far.guards[1].east = 0.F;
    close.guards[0].north = 0.F; close.guards[0].east = 0.F;
    close.guards[1].north = 1500.F; close.guards[1].east = 0.F;
    if (!(blue::SeparationPenalty(p, far) == 0.F
          && blue::SeparationPenalty(p, close) < 0.F
          && blue::SeparationPenalty(p, close) > -0.5F))
        return 17;

    // A patrol-only horizon is a normal finite-horizon completion and must not
    // receive the unresolved-threat timeout penalty.
    auto timeout = next;
    timeout.step = p.episodeSteps;
    auto const patrolTimeout = blue::TeamRewardBreakdown(
        p, state, std::array<blue::Action, 2>{}, timeout);
    if (timeout.failed
        || patrolTimeout.values[static_cast<std::size_t>(
            blue::RewardChannel::Timeout)] != 0.F)
        return 11;

    // With one threat assigned to the leader, racetrack shaping is suspended;
    // the wingman nevertheless keeps its wedge unless it is the interceptor.
    auto oneThreat = state;
    oneThreat.guards[0].north = 0.F;
    oneThreat.guards[0].east = 0.F;
    oneThreat.guards[1].north = 10000.F;
    oneThreat.guards[1].east = 0.F;
    oneThreat.attackers[0] = blue::Mobile{1000.F, 0.F, 0.F, 220.F, 0.F, true};
    if (blue::ActiveThreatCount(oneThreat) != 1U
        || blue::MissionPatrolReward(p, oneThreat, oneThreat) != 0.F
        || blue::MutualSupportWeight(p, oneThreat) != 1.F)
        return 22;
    auto dangerousDefense = oneThreat;
    dangerousDefense.guards[1] = dangerousDefense.guards[0];
    dangerousDefense.guards[1].north += 1500.F;
    auto const dangerousDefenseReward = blue::TeamRewardBreakdown(
        p, dangerousDefense, std::array<blue::Action, 2>{},
        dangerousDefense);
    if (dangerousDefenseReward.values[static_cast<std::size_t>(
            blue::RewardChannel::FormationSafety)] >= 0.F)
        return 35;

    // Joint interception progress must beat opening the range.
    auto closing = oneThreat;
    auto retreating = oneThreat;
    closing.guards[0].north = 100.F;
    retreating.guards[0].north = -100.F;
    if (!(blue::EngagementReward(p, oneThreat, closing)
          > blue::EngagementReward(p, oneThreat, retreating)))
        return 23;

    // Joint coverage scores both reds without assigning an Actor target.
    auto twoThreats = oneThreat;
    twoThreats.attackers[1] = blue::Mobile{
        9000.F, 0.F, 0.F, 220.F, 0.F, true};
    if (blue::ActiveThreatCount(twoThreats) != 2U
        || blue::MissionPatrolReward(p, twoThreats, twoThreats) != 0.F
        || blue::MutualSupportWeight(p, twoThreats) != 0.F)
        return 24;
    auto duplicateCoverage = twoThreats;
    duplicateCoverage.guards[1] = duplicateCoverage.guards[0];
    if (!(blue::EngagementReward(p, twoThreats, twoThreats)
          > blue::EngagementReward(p, duplicateCoverage, duplicateCoverage)))
        return 28;

    // A remaining threat at the finite horizon is an explicit mission failure
    // signal, while destroying the last threat earns both sparse bonuses.
    auto threatTimeout = oneThreat;
    threatTimeout.step = p.episodeSteps;
    auto const threatTimeoutReward = blue::TeamRewardBreakdown(
        p, oneThreat, std::array<blue::Action, 2>{}, threatTimeout);
    if (threatTimeoutReward.values[static_cast<std::size_t>(
            blue::RewardChannel::Timeout)] != -15.F)
        return 25;
    auto defeated = oneThreat;
    defeated.attackers[0].available = false;
    auto const victoryReward = blue::TeamRewardBreakdown(
        p, oneThreat, std::array<blue::Action, 2>{}, defeated);
    auto const stagedVictoryReward = blue::TeamRewardBreakdown(
        p, oneThreat, std::array<blue::Action, 2>{}, defeated, true);
    if (victoryReward.values[static_cast<std::size_t>(
            blue::RewardChannel::Interception)] != 10.F
        || victoryReward.values[static_cast<std::size_t>(
            blue::RewardChannel::MissionOutcome)] != 25.F
        || stagedVictoryReward.values[static_cast<std::size_t>(
            blue::RewardChannel::MissionOutcome)] != 0.F)
        return 26;
    // Reference controllers use the same normalized SNA actions/physics as
    // the production ECS; both guards must move and turn without PPO weights.
    for (int side : {-1, 1})
    {
        auto patrol = blue::InitialState(p, side, 0U, 0.F);
        float maxError = 0.F;
        float accumulatedTurn = 0.F;
        for (std::size_t i = 0; i < 3000U; ++i)
        {
            std::array<blue::Action, 2> commands{};
            for (std::size_t role = 0; role < commands.size(); ++role)
                commands[role] = blue::PatrolReferenceAction(
                    p, patrol.guards[role], patrol.guards[0], side, role,
                    patrol.arclength);
            auto const previous = patrol;
            patrol = blue::AdvanceBoth(p, patrol, commands);
            if (patrol.failed || patrol.step != i + 1U)
                return 12;
            auto const projection = blue::ProjectOnPatrolCurve(
                p.geometry, patrol.guards[0].north, patrol.guards[0].east);
            auto const formation = blue::MeasureFormationErrors(p, patrol);
            maxError = std::max(maxError, projection.distance);
            maxError = std::max(maxError,
                std::hypot(formation.longitudinal, formation.inward)
                    * p.geometry.separationDistance);
            for (std::size_t role = 0; role < patrol.guards.size(); ++role)
                accumulatedTurn += std::abs(
                    patrol.guards[role].turnRate * p.dt);
            if (i == 0U
                && (std::abs(patrol.guards[0].north - previous.guards[0].north)
                    + std::abs(patrol.guards[0].east - previous.guards[0].east)) < 1.F)
                return 13;
        }
        // The slight speed advantage increases pure-pursuit lag; the reference
        // remains bounded to half the requested wedge separation.
        if (maxError > 0.5F * p.geometry.separationDistance
            || accumulatedTurn < blue::kTwoPi)
        {
            std::cerr << "reference max error=" << maxError
                      << " accumulated turn=" << accumulatedTurn << '\n';
            return 14;
        }
    }
    return 0;
}
