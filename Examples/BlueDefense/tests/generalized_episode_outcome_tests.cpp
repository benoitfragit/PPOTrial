/** @file
 * @brief Integrated waypoint/lap, terminal reward and PPO timeout regressions.
 */
#include "generalized_episode_outcome.hpp"
#include "generalized_reward_shaping.hpp"
#include "generalized_cooperative_reward.hpp"
#include "generalized_advantage.hpp"

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
        if (rewards.perBlue[index].values[mission] != expectedOutcome
            || rewards.perBlue[index].values[timeout] != expectedTimeout)
            return false;
    return true;
}

/** @brief Exercises the same frozen references and terminal resolver as the agent. */
bool CompleteLap(int const direction, std::size_t const blueCount)
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
        auto const after = StateAt(parameters, roles,
            circuit * static_cast<float>(step) / static_cast<float>(steps));
        auto const targets = sequencer.Targets(parameters, before, roles, settings);
        auto const frozen = blue::ComputePatrolWaypointTargets(parameters, after,
            roles, targets[roles.LeaderPhysicalIndex()].guidancePose);
        sequencer.Observe(parameters, before, after, roles, settings);
        bool const lapCompleted = lap.Observe(
            parameters, before, after, roles, settings, targets, frozen);
        auto const outcome = episode::EvaluateOutcome(after, false, sequencer,
            lapCompleted, settings, parameters.dt, step + 2U, 4096U);
        bool const final = step == steps;
        if (outcome.patrolCompleted != final || outcome.terminated != final
            || outcome.truncated || outcome.missionFailed
            || !TerminalRewards(roles, before, after, outcome, final ? 1.F : 0.F, 0.F))
            return false;
        before = after;
    }
    auto const success = episode::EvaluateOutcome(before, false, sequencer,
        lap.Completed(), settings, parameters.dt, 2400U, 4096U);
    if (!success.terminated || success.truncated
        || std::strcmp(success.DiagnosticCause(), "success") != 0
        || episode::EvaluateOutcome(before, false, sequencer, false,
            settings, parameters.dt, steps, 4096U).patrolCompleted
        || episode::EvaluateOutcome(before, true, sequencer, true,
            settings, parameters.dt, steps, 4096U).patrolCompleted)
        return false;
    blue::GeneralizedPatrolWaypointSequencer incomplete{};
    return !episode::EvaluateOutcome(before, false, incomplete, true,
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
            false, settings, parameters.dt, 2400U, 4096U);
        if (pending.terminated || pending.truncated || pending.noThreatRemains
            || !TerminalRewards(roles, state, state, pending, 0.F, 0.F, true))
            return false;
        auto const pendingTimeout = episode::EvaluateOutcome(state, true, sequencer,
            false, settings, parameters.dt, 4096U, 4096U);
        if (pendingTimeout.terminated || !pendingTimeout.truncated
            || !TerminalRewards(roles, state, state, pendingTimeout, 0.F, -1.F, true))
            return false;
        state.reds[0].available = true;
        auto const combat = episode::EvaluateOutcome(state, false, sequencer,
            false, settings, parameters.dt, 2400U, 4096U);
        if (combat.terminated || combat.truncated)
            return false;
        auto const before = state;
        state.reds[0].available = false;
        state.failed = true; // Existing victory precedence at a boundary exit.
        auto const victory = episode::EvaluateOutcome(state, false, sequencer,
            false, settings, parameters.dt, 4096U, 4096U);
        if (!victory.noThreatRemains || !victory.terminated || victory.truncated
            || victory.missionFailed || victory.patrolCompleted
            || std::strcmp(victory.DiagnosticCause(), "success") != 0
            || !TerminalRewards(roles, before, state, victory, 1.F, 0.F))
            return false;
        state.objectiveAvailable = false;
        if (!episode::EvaluateOutcome(state, false, sequencer, false,
                settings, parameters.dt, 4096U, 4096U).missionFailed)
            return false;
        state.failed = false;
        state.objectiveAvailable = true;
    }
    return true;
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
    if (!TimeoutAndFailure()) { std::cerr << "timeout/failure\n"; return 2; }
    if (!CombatAndPendingThreats()) { std::cerr << "combat/pending\n"; return 3; }
    return 0;
}
