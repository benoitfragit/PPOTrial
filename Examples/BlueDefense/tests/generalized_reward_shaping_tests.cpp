/**
 * @file Examples/BlueDefense/tests/generalized_reward_shaping_tests.cpp
 * @brief Tests et assertions déterministes pour generalized reward shaping tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "generalized_reward_shaping.hpp"
#include "generalized_cooperative_reward.hpp"

#include <cmath>
#include <cstddef>

namespace patrol = ppo::examples::bluedefense::defenders;

namespace {
/**
 * @brief Réalise l'opération Close; voir les préconditions et effets dans son corps C++.
 */
bool Close(float const a, float const b, float const tolerance = 1.e-4F)
{
    return std::abs(a - b) <= tolerance;
}

patrol::GeneralizedRuntimeState MakeState(
    std::size_t const blues, std::size_t const reds)
{
    patrol::GeneralizedRuntimeState state{};
    state.counts = {blues, reds};
    for (std::size_t index = 0; index < blues; ++index)
        state.blues[index] = {0.F, 0.F, 0.F, 0.F, 0.F, true};
    for (std::size_t index = 0; index < reds; ++index)
        state.reds[index] = {90000.F, 0.F, 3.14159265F, 400.F, 0.F, true};
    state.objectiveNorth = 0.F;
    state.objectiveEast = 0.F;
    return state;
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool TestControlEffortPenalty()
{
    patrol::Parameters parameters{};
    parameters.controlEffortWeight = 0.002F;
    if (!Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{0.F, 0.F}), 0.F, 1.e-7F)
        || !Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{1.F, -1.F}), 0.002F, 1.e-7F)
        || !Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{0.5F, -0.5F}), 0.001F, 1.e-7F)
        || !Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{2.F, -3.F}), 0.002F, 1.e-7F))
        return false;
    parameters.controlEffortWeight = 0.F;
    return Close(patrol::GeneralizedControlEffortPenalty(
        parameters, patrol::Action{1.F, 1.F}), 0.F, 1.e-7F);
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool TestTracking()
{
    patrol::Parameters parameters{};
    auto const roles = patrol::BlueTeamRoles::Create(1U, 0U);
    if (!roles)
        return false;
    auto before = MakeState(1U, 0U);
    auto const reference = patrol::EvaluatePatrolCurve(
        parameters.geometry, 3000.F);
    before.blues[0] = {reference.north, reference.east,
        reference.heading, parameters.cruiseSpeed, 0.F, true};
    auto onTrack = before;
    auto const advanced = patrol::EvaluatePatrolCurve(
        parameters.geometry, 3000.F + parameters.cruiseSpeed * parameters.dt);
    onTrack.blues[0].north = advanced.north;
    onTrack.blues[0].east = advanced.east;
    onTrack.blues[0].heading = advanced.heading;
    auto offTrack = onTrack;
    offTrack.blues[0].north += 5000.F;
    offTrack.blues[0].east += 5000.F;
    auto wrongHeading = onTrack;
    wrongHeading.blues[0].heading += 3.14159265F;
    float const good = patrol::GeneralizedLeaderTrackingReward(
        parameters, before, onTrack, *roles);
    return good > patrol::GeneralizedLeaderTrackingReward(
            parameters, before, offTrack, *roles)
        && good > patrol::GeneralizedLeaderTrackingReward(
            parameters, before, wrongHeading, *roles);
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool TestWedgeReference()
{
    patrol::Parameters parameters{};
    auto const roles = patrol::BlueTeamRoles::Create(2U, 0U);
    if (!roles)
        return false;
    auto state = MakeState(2U, 0U);
    state.blues[0] = {5000.F, -2000.F, 1.25F, 450.F, 0.F, true};
    auto const offset = patrol::EvaluateFormationOffset(
        0U, parameters.geometry.separationDistance,
        parameters.geometry.separationAngle);
    float const cosHeading = std::cos(state.blues[0].heading);
    float const sinHeading = std::sin(state.blues[0].heading);
    float const inside = static_cast<float>(parameters.geometry.direction)
        * offset.inwardMeters;
    state.blues[1].north = state.blues[0].north
        + offset.longitudinalMeters * cosHeading - inside * sinHeading;
    state.blues[1].east = state.blues[0].east
        + offset.longitudinalMeters * sinHeading + inside * cosHeading;
    std::array<patrol::GeneralizedObservation,
               patrol::TeamCapacity::MaximumBlueMobiles> observations{};
    if (!patrol::BuildGeneralizedObservations(
            parameters, state, *roles, observations))
        return false;
    std::size_t const offsetInObservation =
        patrol::GeneralizedObservationLayout::PatrolOffset;
    return Close(patrol::GeneralizedFormationQuality(
            parameters, state, *roles, 1U), 1.F)
        && Close(observations[1].values[offsetInObservation + 6U], 0.F)
        && Close(observations[1].values[offsetInObservation + 7U], 0.F);
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool TestPursuit()
{
    patrol::Parameters parameters{};
    auto before = MakeState(3U, 2U);
    before.reds[0].north = 75000.F;
    before.reds[1].north = 120000.F;
    auto approaching = before;
    approaching.blues[0].north += 80.F;
    approaching.blues[1].north += 80.F;
    approaching.blues[2].north += 80.F;
    auto withdrawing = before;
    withdrawing.blues[0].north -= 80.F;
    withdrawing.blues[1].north -= 80.F;
    withdrawing.blues[2].north -= 80.F;
    auto redOnly = before;
    redOnly.reds[0].north -= 80.F;
    redOnly.reds[1].north -= 80.F;
    return patrol::GeneralizedPursuitProgress(
            parameters, before, approaching) > 0.F
        && patrol::GeneralizedPursuitProgress(
            parameters, before, withdrawing) < 0.F
        && Close(patrol::GeneralizedPursuitProgress(
            parameters, before, redOnly), 0.F)
        && Close(patrol::GeneralizedPursuitProgress(
            parameters, MakeState(1U, 0U),
            MakeState(1U, 0U)), 0.F);
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool TestCooperation()
{
    patrol::Parameters parameters{};
    auto noRed = MakeState(2U, 0U);
    auto farRed = MakeState(2U, 1U);
    farRed.reds[0].north = 250000.F;
    auto objectiveThreat = farRed;
    objectiveThreat.reds[0].north = 90000.F;
    auto closeRed = farRed;
    closeRed.reds[0].north = 20000.F;
    return Close(patrol::GeneralizedCooperationWeight(
            parameters, noRed), 1.F)
        && Close(patrol::GeneralizedCooperationWeight(
            parameters, farRed), 1.F)
        && patrol::GeneralizedCooperationWeight(
            parameters, closeRed) > 0.F
        && patrol::GeneralizedCooperationWeight(parameters, closeRed) < 1.F
        && Close(patrol::GeneralizedEngagementWeight(parameters, noRed), 0.F)
        && Close(patrol::GeneralizedEngagementWeight(parameters, farRed), 0.F)
        && patrol::GeneralizedEngagementWeight(parameters, closeRed) > 0.F
        && patrol::GeneralizedEngagementWeight(parameters, objectiveThreat) > 0.F;
}
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool TestFormationAndLowSpeed()
{
    patrol::Parameters parameters{};
    auto const roles = patrol::BlueTeamRoles::Create(2U, 0U);
    if (!roles)
        return false;
    auto before = MakeState(2U, 0U);
    before.blues[0] = {0.F, 0.F, 0.F, parameters.cruiseSpeed, 0.F, true};
    patrol::FormationOffset const slot = patrol::EvaluateFormationOffset(
        0U, parameters.geometry.separationDistance,
        parameters.geometry.separationAngle);
    before.blues[1] = {
        slot.longitudinalMeters + 2000.F,
        static_cast<float>(parameters.geometry.direction) * slot.inwardMeters,
        0.F, parameters.cruiseSpeed, 0.F, true};
    auto closer = before;
    closer.blues[1].north -= 1000.F;
    if (!Close(patrol::GeneralizedFormationProgressReward(
            parameters, before, before, *roles, 1U), 0.F)
        || !(patrol::GeneralizedFormationProgressReward(
            parameters, before, closer, *roles, 1U) > 0.F)
        || !(patrol::GeneralizedFormationProgressReward(
            parameters, closer, before, *roles, 1U) < 0.F))
        return false;
    auto leaderLost = closer;
    leaderLost.blues[0].available = false;
    if (!Close(patrol::GeneralizedFormationProgressReward(
            parameters, before, leaderLost, *roles, 1U), 0.F))
        return false;
    auto teammateLost = closer;
    teammateLost.blues[1].available = false;
    if (!Close(patrol::GeneralizedFormationProgressReward(
            parameters, before, teammateLost, *roles, 1U), 0.F))
        return false;

    patrol::Mobile mobile{0.F, 0.F, 0.F,
                          parameters.cruiseSpeed, 0.F, true};
    float const cruiseRisk = patrol::GeneralizedLowSpeedRisk(
        parameters, mobile);
    mobile.speed = 0.2F * parameters.cruiseSpeed;
    float const cautionRisk = patrol::GeneralizedLowSpeedRisk(
        parameters, mobile);
    mobile.speed = 0.F;
    float const zeroRisk = patrol::GeneralizedLowSpeedRisk(
        parameters, mobile);
    mobile.available = false;
    auto beforeLoss = mobile;
    beforeLoss.available = true;
    beforeLoss.speed = 0.1F * parameters.cruiseSpeed;
    float const fatalPenalty = patrol::GeneralizedLowSpeedTransitionPenalty(
        parameters, beforeLoss, mobile);
    float const maskedPenalty = patrol::GeneralizedLowSpeedTransitionPenalty(
        parameters, mobile, mobile);
    return fatalPenalty > 0.F && Close(maskedPenalty, 0.F)
        && Close(cruiseRisk, 0.F)
        && cautionRisk > 0.F && cautionRisk < 1.F
        && Close(zeroRisk, 1.F)
        && Close(patrol::GeneralizedLowSpeedRisk(parameters, mobile), 0.F);
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool TestSafetyRisks()
{
    patrol::Parameters parameters{};
    auto state = MakeState(2U, 0U);
    state.blues[0] = {0.F, 0.F, 0.F, 500.F, 0.F, true};
    state.blues[1] = {5000.F, 0.F, patrol::kPi, 500.F, 0.F, true};
    float const approaching = patrol::GeneralizedBlueSeparationRisk(
        parameters, state, 0U);
    state.blues[1].heading = 0.F;
    if (!(approaching > 0.F)
        || !Close(patrol::GeneralizedBlueSeparationRisk(
            parameters, state, 0U), 0.F))
        return false;

    // P=1/N=0 and dead peers produce no friendly-separation penalty.
    state.blues[1].available = false;
    if (!Close(patrol::GeneralizedBlueSeparationRisk(
            parameters, state, 0U), 0.F)
        || !Close(patrol::GeneralizedBlueSeparationRisk(
            parameters, state, 1U), 0.F))
        return false;
    auto alone = MakeState(1U, 0U);
    if (!Close(patrol::GeneralizedBlueSeparationRisk(
            parameters, alone, 0U), 0.F))
        return false;

    // Anticipate the soft world edge, not only the terminal hard boundary.
    alone.blues[0] = {0.F, 0.F, 0.F, 500.F, 0.F, true};
    float const inside = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    alone.blues[0].north = parameters.halfNorth - 3500.F;
    float const approach = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    alone.blues[0].north = parameters.halfNorth + 100.F;
    float const justOutside = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    alone.blues[0].north = parameters.halfNorth
        * (1.F + 0.5F * (parameters.hardBoundaryScale - 1.F));
    float const midwayOutside = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    // Approaching the hard boundary must never saturate a full step early
    // because projected position is already beyond the terminal limit.
    alone.blues[0].north = parameters.halfNorth
        * parameters.hardBoundaryScale - 1000.F;
    float const beforeTerminal = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    alone.blues[0].north = parameters.halfNorth
        * parameters.hardBoundaryScale;
    float const terminal = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    alone.blues[0].north = parameters.halfNorth
        * (parameters.hardBoundaryScale + 0.5F);
    float const beyond = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    alone.blues[0].available = false;
    float const dead = patrol::GeneralizedBoundaryRisk(
        parameters, alone.blues[0]);
    if (!Close(inside, 0.F) || !(approach > inside)
        || !(justOutside > approach && justOutside < midwayOutside)
        || !(midwayOutside < beforeTerminal)
        || !(beforeTerminal < terminal) || !Close(terminal, 1.F)
        || !Close(beyond, 1.F) || !Close(dead, 0.F)
        || !(patrol::GeneralizedSafetyPenalty(
                justOutside, midwayOutside) < 0.5F)
        // A sustained worst-case risk must remain below one red
        // neutralization event even over 200 steps of dense shaping.
        || !(200.F * patrol::kGeneralizedBoundaryClearanceWeight
                * patrol::GeneralizedSafetyPenalty(1.F, 1.F)
             < 0.25F * 2.F))
        return false;

    // Red-zone band: 1.75 NEZ caution / 1.10 NEZ physical danger.
    float const nearFacing = patrol::GeneralizedRestrictedZoneProximityRisk(
        1400.F, 1000.F, 1.F);
    float const nearReceding = patrol::GeneralizedRestrictedZoneProximityRisk(
        1400.F, 1000.F, -1.F);
    if (!Close(patrol::GeneralizedRestrictedZoneProximityRisk(
            2000.F, 1000.F, 1.F), 0.F)
        || !Close(patrol::GeneralizedRestrictedZoneProximityRisk(
            1000.F, 1000.F, -1.F), 1.F)
        || !(nearFacing > nearReceding && nearReceding > 0.F)
        || !Close(patrol::GeneralizedRestrictedZoneProximityRisk(
            1500.F, 0.F, 1.F), 0.F))
        return false;

    return Close(patrol::GeneralizedSafetyPenalty(0.F, 0.F), 0.F)
        && patrol::GeneralizedSafetyPenalty(0.1F, 0.9F)
            > patrol::GeneralizedSafetyPenalty(0.9F, 0.1F)
        && patrol::GeneralizedSafetyPenalty(0.9F, 0.1F) > 0.F;
}

/**
 * @brief Pvs0 leader laps require progress, position, speed and heading.
 */
bool TestNoThreatPatrolLap()
{
    patrol::Parameters parameters{};
    parameters.cruiseSpeed = 300.F;
    parameters.maxSpeed = 750.F;
    auto roles = patrol::BlueTeamRoles::Create(1U, 0U);
    if (!roles)
        return false;
    patrol::PatrolNoThreatSettings settings{};
    settings.enabled = true;
    settings.stepRewardScale = 0.0005F;
    auto previous = MakeState(1U, 0U);
    auto next = previous;
    patrol::GeneralizedPatrolLapProgress lap{};
    float const circuit = patrol::CircuitLength(parameters.geometry);
    constexpr std::size_t segments = 2000U;
    float const step = circuit / static_cast<float>(segments);
    bool completed = false;
    for (std::size_t index = 0U; index < segments; ++index)
    {
        auto const from = patrol::EvaluatePatrolCurve(
            parameters.geometry, 1000.F + step * static_cast<float>(index));
        auto const to = patrol::EvaluatePatrolCurve(
            parameters.geometry, 1000.F + step * static_cast<float>(index + 1U));
        previous.blues[0] = {from.north, from.east, from.heading,
                             parameters.cruiseSpeed, 0.F, true};
        next.blues[0] = {to.north, to.east, to.heading,
                         parameters.cruiseSpeed, 0.F, true};
        completed = lap.Observe(parameters, previous, next, *roles, settings);
        if (completed != (index + 1U == segments))
            return false;
    }
    if (!completed || !lap.Completed()
        || patrol::GeneralizedNoThreatShaping(
               parameters, next, *roles, 0U, settings) <= 0.F)
        return false;

    // A stationary leader cannot earn another lap, even on the curve.
    lap.Reset();
    if (lap.Observe(parameters, next, next, *roles, settings)
        || lap.Completed())
        return false;
    // The shaping must retain a recovery gradient far beyond tolerance.
    patrol::GeneralizedPatrolError const moderatelyOff{2.F, 0.F, 0.F};
    patrol::GeneralizedPatrolError const severelyOff{6.F, 0.F, 0.F};
    if (!(moderatelyOff.BoundedCost() < severelyOff.BoundedCost())
        || !(severelyOff.BoundedCost() < 1.F))
        return false;
    auto stopped = next;
    stopped.blues[0].speed = 0.F;
    if (patrol::GeneralizedNoThreatError(
            parameters, stopped, *roles, 0U, settings).InTolerance()
        || patrol::GeneralizedNoThreatShaping(
            parameters, stopped, *roles, 0U, settings) >= 0.F)
        return false;
    return true;
}

/** @brief Follower slots use projected racetrack, never a drifting leader. */
bool TestProjectedWedgeAndInactiveRedMask()
{
    patrol::Parameters parameters{};
    parameters.cruiseSpeed = 300.F;
    parameters.maxSpeed = 750.F;
    patrol::PatrolNoThreatSettings settings{};
    settings.enabled = true;
    auto roles = patrol::BlueTeamRoles::Create(2U, 0U);
    if (!roles)
        return false;
    auto state = MakeState(2U, 1U);
    state.reds[0].available = false; // Delayed activation.
    if (patrol::GeneralizedHasActiveRed(state))
        return false;

    auto const leader = patrol::EvaluatePatrolCurve(
        parameters.geometry, 2000.F);
    state.blues[0] = {leader.north + 4000.F, leader.east,
                      leader.heading, parameters.cruiseSpeed, 0.F, true};
    auto const target = patrol::GeneralizedNoThreatTarget(
        parameters, state, *roles, 1U);
    state.blues[1] = {target.pose.north, target.pose.east,
        target.pose.heading, target.speed, 0.F, true};

    std::array<patrol::GeneralizedObservation,
               patrol::TeamCapacity::MaximumBlueMobiles> projected{};
    std::array<patrol::GeneralizedObservation,
               patrol::TeamCapacity::MaximumBlueMobiles> historical{};
    if (!patrol::BuildGeneralizedObservations(
            parameters, state, *roles, projected, true)
        || !patrol::BuildGeneralizedObservations(
            parameters, state, *roles, historical))
        return false;
    std::size_t const index =
        patrol::GeneralizedObservationLayout::PatrolOffset + 6U;
    if (!Close(projected[1].values[index], 0.F, 0.02F)
        || !Close(projected[1].values[index + 1U], 0.F, 0.02F)
        || (Close(historical[1].values[index], 0.F, 0.02F)
            && Close(historical[1].values[index + 1U], 0.F, 0.02F))
        || !patrol::GeneralizedNoThreatError(
            parameters, state, *roles, 1U, settings).InTolerance()
        || patrol::GeneralizedNoThreatError(
            parameters, state, *roles, 0U, settings).InTolerance())
        return false;

    // On a turn, a wedge slot has its own tangent and speed.
    auto const turn = patrol::EvaluatePatrolCurve(
        parameters.geometry,
        parameters.geometry.length - parameters.geometry.width
            + 0.25F * patrol::kPatrolPi * parameters.geometry.width);
    state.blues[0] = {turn.north, turn.east, turn.heading,
                      parameters.cruiseSpeed, 0.F, true};
    auto const turningTarget = patrol::GeneralizedNoThreatTarget(
        parameters, state, *roles, 1U);
    if (Close(turningTarget.speed, parameters.cruiseSpeed, 1.F))
        return false;

    state.reds[0].available = true;
    return patrol::GeneralizedHasActiveRed(state);
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    if (!TestTracking()) return 1;
    if (!TestWedgeReference()) return 2;
    if (!TestPursuit()) return 3;
    if (!TestCooperation()) return 4;
    if (!TestControlEffortPenalty()) return 5;
    if (!TestSafetyRisks()) return 6;
    if (!TestFormationAndLowSpeed()) return 7;
    if (!TestNoThreatPatrolLap()) return 8;
    if (!TestProjectedWedgeAndInactiveRedMask()) return 9;
    return 0;
}
