/** @file
 * @brief Integrated waypoint/lap, terminal reward and PPO timeout regressions.
 */
#include "generalized_episode_outcome.hpp"
#include "generalized_reward_shaping.hpp"
#include "generalized_cooperative_reward.hpp"
#include "generalized_advantage.hpp"
#include "generalized_leader_succession.hpp"

#include <cmath>
#include <cstring>
#include <iostream>

namespace blue = ppo::examples::bluedefense::defenders;
namespace episode = blue::generalized_episode_detail;
namespace {

/** @brief Fixed feasible geometry with sixty metres of cruise flight per step. */
blue::Parameters TestParameters(int const direction)
{
    blue::Parameters parameters{};
    parameters.dt = 0.2F;
    parameters.cruiseSpeed = 300.F;
    parameters.maxSpeed = 750.F;
    parameters.geometry.length = 40000.F;
    parameters.geometry.width = 16000.F;
    parameters.geometry.separationDistance = 2800.F;
    parameters.geometry.separationAngle = 0.55F;
    parameters.geometry.direction = direction;
    return parameters;
}

/** @brief Places all roles on the actual curve and its physical wedge. */
blue::GeneralizedRuntimeState StateAt(blue::Parameters const& parameters,
    blue::BlueTeamRoles const& roles, float const arclength)
{
    blue::GeneralizedRuntimeState state{};
    state.counts = {roles.ActiveBlueCount(), 0U};
    auto const leader = blue::EvaluatePatrolCurve(parameters.geometry, arclength);
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        auto const assignment = *roles.Assignment(index);
        auto const reference = assignment.role == blue::BlueRole::Leader
            ? blue::PatrolFormationReference{leader, parameters.cruiseSpeed}
            : blue::EvaluateFormationReference(parameters.geometry, leader,
                assignment.formationSlot, parameters.cruiseSpeed);
        state.blues[index] = {reference.pose.north, reference.pose.east,
            reference.pose.heading, reference.speed, 0.F, true};
    }
    return state;
}

/** @brief Converts the resolved snapshot into the existing reward event masks. */
blue::GeneralizedCombatStateMask CombatMask(
    blue::GeneralizedRuntimeState const& state)
{
    blue::GeneralizedCombatStateMask mask{};
    mask.objectiveAvailable = state.objectiveAvailable;
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        mask.blueActive[index] = true;
        mask.blueAlive[index] = state.blues[index].available;
    }
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
    {
        mask.redActive[index] = true;
        mask.redAlive[index] = state.reds[index].available;
    }
    return mask;
}

/** @brief Checks terminal channels for the team and every participating Blue. */
bool TerminalRewards(blue::BlueTeamRoles const& roles,
    blue::GeneralizedRuntimeState const& before,
    blue::GeneralizedRuntimeState const& after, episode::EpisodeOutcome const& outcome,
    float const expectedOutcome, float const expectedTimeout,
    bool const pendingThreats = false)
{
    blue::GeneralizedRewardTransition transition{};
    transition.before = CombatMask(before);
    transition.after = CombatMask(after);
    transition.missionFailed = outcome.missionFailed;
    transition.patrolCompleted = outcome.patrolCompleted;
    transition.timeout = outcome.truncated;
    transition.pendingThreats = pendingThreats;
    blue::GeneralizedCooperativeRewards rewards{};
    if (blue::AggregateGeneralizedCooperativeRewards(roles, transition, {}, rewards)
        != blue::GeneralizedRewardStatus::Applied)
        return false;
    std::size_t const mission = static_cast<std::size_t>(
        blue::GeneralizedRewardChannel::MissionOutcome);
    std::size_t const timeout = static_cast<std::size_t>(
        blue::GeneralizedRewardChannel::Timeout);
    if (rewards.team.values[mission] != expectedOutcome
        || rewards.team.values[timeout] != expectedTimeout)
        return false;
    for (std::size_t index = 0U; index < roles.ActiveBlueCount(); ++index)
    {
        float const individualOutcome = outcome.patrolCompleted
            && index != roles.LeaderPhysicalIndex() ? 0.F : expectedOutcome;
        if (rewards.perBlue[index].values[mission] != individualOutcome
            || rewards.perBlue[index].values[timeout] != expectedTimeout)
            return false;
    }
    return true;
}

/** @brief Exercises the same frozen references and terminal resolver as the agent. */
bool CompleteLap(int const direction, std::size_t const blueCount,
                 bool const wingmanOffFormation = false,
                 bool const wingmanLost = false)
{
    auto const parameters = TestParameters(direction);
    auto const roles = *blue::BlueTeamRoles::Create(blueCount, blueCount - 1U);
    blue::PatrolNoThreatSettings settings{};
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    blue::GeneralizedPatrolLapProgress lap{};
    auto before = StateAt(parameters, roles, 0.F);
    sequencer.Observe(parameters, before, before, roles, settings);
    if (sequencer.Observe(parameters, before, before, roles, settings)
        != blue::GeneralizedWaypointUpdate::EnteredFollow)
        return false;
    float const circuit = blue::CircuitLength(parameters.geometry);
    auto const steps = static_cast<std::size_t>(std::ceil(circuit / 60.F));
    for (std::size_t step = 1U; step <= steps; ++step)
    {
        auto after = StateAt(parameters, roles,
            circuit * static_cast<float>(step) / static_cast<float>(steps));
        if (blueCount > 1U && wingmanOffFormation)
        {
            after.blues[0].heading += 1.F;
            after.blues[0].speed += 100.F;
        }
        if (blueCount > 1U && wingmanLost)
            after.blues[0].available = false;
        auto const targets = sequencer.Targets(parameters, before, roles, settings);
        auto const frozen = blue::ComputePatrolWaypointTargets(parameters, after,
            roles, targets[roles.LeaderPhysicalIndex()].guidancePose);
        sequencer.Observe(parameters, before, after, roles, settings);
        bool const lapCompleted = lap.Observe(
            parameters, before, after, roles, settings, targets, frozen);
        auto const outcome = episode::EvaluateOutcome(after, false, sequencer,
            roles.LeaderPhysicalIndex(),
            lapCompleted, settings, parameters.dt, step + 2U, 4096U);
        bool const final = step == steps;
        if (outcome.patrolCompleted != final || outcome.terminated != final
            || outcome.truncated || outcome.missionFailed
            || !TerminalRewards(roles, before, after, outcome, final ? 1.F : 0.F, 0.F))
            return false;
        before = after;
    }
    auto const success = episode::EvaluateOutcome(before, false, sequencer,
            roles.LeaderPhysicalIndex(),
        lap.Completed(), settings, parameters.dt, 2400U, 4096U);
    if (!success.terminated || success.truncated
        || std::strcmp(success.DiagnosticCause(), "success") != 0
        || episode::EvaluateOutcome(before, false, sequencer,
            roles.LeaderPhysicalIndex(), false,
            settings, parameters.dt, steps, 4096U).patrolCompleted
        || episode::EvaluateOutcome(before, true, sequencer,
            roles.LeaderPhysicalIndex(), true,
            settings, parameters.dt, steps, 4096U).patrolCompleted)
        return false;
    blue::GeneralizedPatrolWaypointSequencer incomplete{};
    return !episode::EvaluateOutcome(before, false, incomplete,
            roles.LeaderPhysicalIndex(), true,
            settings, parameters.dt, steps, 4096U).patrolCompleted
        && blue::GeneralizedMissionAdvantage(2.F, 3.F, 4.F, 100.F,
            0.9F, 0.8F, success.terminated, success.truncated) == -1.F;
}

/** @brief Times out at exactly 480 simulated seconds without a success bonus. */
bool TimeoutAndFailure()
{
    auto const parameters = TestParameters(1);
    auto const roles = *blue::BlueTeamRoles::Create(1U, 0U);
    auto state = StateAt(parameters, roles, 0.F);
    blue::PatrolNoThreatSettings settings{};
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    blue::GeneralizedPatrolLapProgress lap{};
    for (std::size_t step = 1U; step <= 2400U; ++step)
    {
        auto const targets = sequencer.Targets(parameters, state, roles, settings);
        auto const update = sequencer.Observe(parameters, state, state, roles, settings);
        bool completed = false;
        if (update == blue::GeneralizedWaypointUpdate::EnteredFollow)
            lap.Reset();
        else if (sequencer.Following())
            completed = lap.Observe(parameters, state, state, roles, settings,
                targets, targets);
        auto const outcome = episode::EvaluateOutcome(state, false, sequencer,
            roles.LeaderPhysicalIndex(),
            completed, settings, parameters.dt, step, 4096U);
        if (outcome.terminated || outcome.missionFailed || outcome.patrolCompleted
            || outcome.truncated != (step == 2400U)
            || !TerminalRewards(roles, state, state, outcome, 0.F,
                step == 2400U ? -1.F : 0.F))
            return false;
        if (step == 2400U
            && (std::strcmp(outcome.DiagnosticCause(), "timeout") != 0
                || std::abs(blue::GeneralizedMissionAdvantage(2.F, 3.F, 4.F, 100.F,
                    0.9F, 0.8F, outcome.terminated, outcome.truncated) - 2.6F) > 1.e-5F))
            return false;
    }
    auto const before = state;
    state.blues[0].available = false;
    auto const failure = episode::EvaluateOutcome(state, false, sequencer,
            roles.LeaderPhysicalIndex(),
        false, settings, parameters.dt, 2400U, 4096U);
    return failure.missionFailed && failure.terminated && !failure.truncated
        && std::strcmp(failure.DiagnosticCause(), "failure") == 0
        && TerminalRewards(roles, before, state, failure, -1.F, 0.F);
}

/** @brief Retains combat and delayed-activation horizons and success labels. */
bool CombatAndPendingThreats()
{
    auto const parameters = TestParameters(1);
    auto const roles = *blue::BlueTeamRoles::Create(3U, 1U);
    auto state = StateAt(parameters, roles, 0.F);
    blue::PatrolNoThreatSettings settings{};
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    for (std::size_t redCount : {1U, 2U, 10U})
    {
        state.counts.redMobiles = redCount;
        for (std::size_t red = 0U; red < redCount; ++red)
            state.reds[red].available = false;
        auto const pending = episode::EvaluateOutcome(state, true, sequencer,
            roles.LeaderPhysicalIndex(),
            false, settings, parameters.dt, 2400U, 4096U);
        if (pending.terminated || pending.truncated || pending.noThreatRemains
            || !TerminalRewards(roles, state, state, pending, 0.F, 0.F, true))
            return false;
        auto const pendingTimeout = episode::EvaluateOutcome(state, true, sequencer,
            roles.LeaderPhysicalIndex(),
            false, settings, parameters.dt, 4096U, 4096U);
        if (pendingTimeout.terminated || !pendingTimeout.truncated
            || !TerminalRewards(roles, state, state, pendingTimeout, 0.F, -1.F, true))
            return false;
        state.reds[0].available = true;
        auto const combat = episode::EvaluateOutcome(state, false, sequencer,
            roles.LeaderPhysicalIndex(),
            false, settings, parameters.dt, 2400U, 4096U);
        if (combat.terminated || combat.truncated)
            return false;
        auto const before = state;
        state.reds[0].available = false;
        state.failed = true; // Existing victory precedence at a boundary exit.
        auto const victory = episode::EvaluateOutcome(state, false, sequencer,
            roles.LeaderPhysicalIndex(),
            false, settings, parameters.dt, 4096U, 4096U);
        if (!victory.noThreatRemains || !victory.terminated || victory.truncated
            || victory.missionFailed || victory.patrolCompleted
            || std::strcmp(victory.DiagnosticCause(), "success") != 0
            || !TerminalRewards(roles, before, state, victory, 1.F, 0.F))
            return false;
        state.objectiveAvailable = false;
        if (!episode::EvaluateOutcome(state, false, sequencer,
            roles.LeaderPhysicalIndex(), false,
                settings, parameters.dt, 4096U, 4096U).missionFailed)
            return false;
        state.failed = false;
        state.objectiveAvailable = true;
    }
    return true;
}

/** @brief Repeated relief preserves physical losses, masks and mission credit. */
bool LeaderSuccession(std::size_t const count, bool const activeRed)
{
    auto const parameters = TestParameters(1);
    auto roles = *blue::BlueTeamRoles::Create(count, count - 1U);
    auto state = StateAt(parameters, roles, 0.F);
    blue::PatrolNoThreatSettings settings{};
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    blue::GeneralizedPatrolLapProgress lap{};
    // Seed actual progress so relief must discard more than an empty state.
    sequencer.Observe(parameters, state, state, roles, settings);
    sequencer.Observe(parameters, state, state, roles, settings);
    auto advanced = StateAt(parameters, roles, 60.F);
    lap.Observe(parameters, state, advanced, roles, settings);
    if (!sequencer.Following() || lap.DistanceMeters() <= 0.F)
        return false;
    state = advanced;
    state.counts.redMobiles = 1U;
    state.reds[0] = {20000.F, 10000.F, 0.F, 250.F, 0.F, activeRed};
    if (blue::RelieveGeneralizedLeader(state, roles, sequencer, lap)
        || !sequencer.Following() || lap.DistanceMeters() <= 0.F)
        return false; // A live leader is never replaced.

    for (std::size_t lost = 0U; lost < count; ++lost)
    {
        auto const oldRoles = roles;
        auto const before = state;
        auto const beforeTargets = sequencer.PhysicalTargets(
            parameters, before, oldRoles, settings);
        std::size_t const oldLeader = oldRoles.LeaderPhysicalIndex();
        state.blues[oldLeader].available = false;
        auto const rewardTargets = sequencer.PhysicalTargets(
            parameters, state, oldRoles, settings);
        bool const lastLost = lost + 1U == count;
        if (blue::RelieveGeneralizedLeader(state, roles, sequencer, lap) == lastLost)
            return false;
        if (!lastLost && (roles.LeaderPhysicalIndex() != lost
                || !roles.IsValid() || roles.ActiveBlueCount() != count
                || sequencer.Following() || sequencer.Completed()
                || sequencer.ActiveWaypointIndex() != 0U
                || sequencer.ValidatedWaypointCount() != 0U
                || lap.DistanceMeters() != 0.F || lap.Completed()))
            return false;
        state.failed = blue::GeneralizedBlueTeamFailed(
            state, parameters, roles.LeaderPhysicalIndex());
        auto const outcome = episode::EvaluateOutcome(state, !activeRed,
            sequencer, roles.LeaderPhysicalIndex(), false,
            settings, parameters.dt, lost + 1U, 4096U);
        if (outcome.missionFailed != lastLost || outcome.terminated != lastLost
            || outcome.patrolCompleted || outcome.truncated
            || !TerminalRewards(oldRoles, before, state, outcome,
                lastLost ? -1.F : 0.F, 0.F, !activeRed))
            return false;
        blue::GeneralizedRewardTransition transition{};
        transition.before = CombatMask(before);
        transition.after = CombatMask(state);
        transition.pendingThreats = !activeRed;
        transition.missionFailed = lastLost;
        blue::GeneralizedCooperativeRewards rewards{};
        if (blue::AggregateGeneralizedCooperativeRewards(
                oldRoles, transition, {}, rewards)
                != blue::GeneralizedRewardStatus::Applied
            || rewards.perBlue[oldLeader].values[static_cast<std::size_t>(
                    blue::GeneralizedRewardChannel::OwnLoss)] != -1.F)
            return false;
        for (std::size_t index = 0U; index < count; ++index)
            if (blue::GeneralizedUnifiedPatrolConvergenceShaping(parameters,
                    before, state, beforeTargets, rewardTargets, oldRoles,
                    index, settings) != 0.F)
                return false; // No synthetic progress across leader loss.
        auto const targets = sequencer.PhysicalTargets(
            parameters, state, roles, settings);
        std::array<blue::GeneralizedObservation,
            blue::TeamCapacity::MaximumBlueMobiles> observations{};
        if (!blue::BuildGeneralizedObservations(
                parameters, state, roles, observations, &targets))
            return false;
        for (std::size_t index = 0U; index < count; ++index)
        {
            auto const& values = observations[index].values;
            if (!blue::Valid(observations[index])
                || targets[index].valid != state.blues[index].available
                || values[blue::GeneralizedObservationLayout::PatrolOffset + 7U]
                    != (state.blues[index].available ? 1.F : 0.F)
                || values[blue::GeneralizedObservationLayout::RoleOffset]
                    != (index == roles.LeaderPhysicalIndex() ? -1.F : 1.F)
                || values[blue::GeneralizedTrackMaskIndex(
                        blue::GeneralizedObservationLayout::RedOffset, 0U,
                        blue::GeneralizedObservationLayout::RedFeatures)]
                    != (activeRed ? 1.F : 0.F))
                return false;
        }
        float const advantage = blue::GeneralizedMissionAdvantage(
            2.F, 3.F, 4.F, 5.F, 0.9F, 0.8F,
            outcome.terminated, outcome.truncated);
        if (std::abs(advantage - (lastLost ? -1.F : 6.2F)) > 1.e-5F)
            return false; // Relief keeps bootstrap and trace; last loss stops both.
    }
    return true;
}

/** @brief No relief in Pvs0, even while a teammate survives the leader. */
bool NoThreatLeaderLoss()
{
    auto roles = *blue::BlueTeamRoles::Create(2U, 1U);
    auto state = StateAt(TestParameters(1), roles, 0.F);
    state.blues[1].available = false;
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    blue::GeneralizedPatrolLapProgress lap{};
    auto const outcome = episode::EvaluateOutcome(state, false, sequencer,
        1U, false, {}, 0.2F, 1U, 4096U);
    return !blue::RelieveGeneralizedLeader(state, roles, sequencer, lap)
        && roles.LeaderPhysicalIndex() == 1U
        && outcome.missionFailed && outcome.terminated && !outcome.patrolCompleted;
}
} // namespace

int main()
{
    for (int direction : {-1, 1})
        for (std::size_t count : {1U, 3U, 9U})
            if (!CompleteLap(direction, count))
            {
                std::cerr << "lap/reward direction=" << direction << " P=" << count << '\n';
                return 1;
            }
    for (int direction : {-1, 1})
        for (std::size_t count : {3U, 9U})
        {
            if (!CompleteLap(direction, count, true, false)
                || !CompleteLap(direction, count, false, true))
            {
                std::cerr << "leader independent lap direction=" << direction
                          << " P=" << count << '\n';
                return 4;
            }
        }
    if (!TimeoutAndFailure()) { std::cerr << "timeout/failure\n"; return 2; }
    if (!CombatAndPendingThreats()) { std::cerr << "combat/pending\n"; return 3; }
    for (std::size_t count : {1U, 2U, 3U, 4U, 9U})
        for (bool const activeRed : {false, true})
            if (!LeaderSuccession(count, activeRed))
            {
                std::cerr << "leader succession P=" << count
                          << " activeRed=" << activeRed << '\n';
                return 5;
            }
    if (!NoThreatLeaderLoss()) { std::cerr << "Pvs0 leader loss\n"; return 6; }
    return 0;
}
