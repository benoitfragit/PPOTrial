#include "generalized_evaluation_campaign.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

using namespace ppo::examples::bluedefense::defenders;

namespace {

constexpr float kTolerance = 1.0e-6F;

bool Near(float const first, float const second) noexcept
{
    return std::abs(first - second) <= kTolerance;
}

GeneralizedCombatStateMask MakeState(std::size_t const blueCount,
                                     std::size_t const redCount) noexcept
{
    GeneralizedCombatStateMask state{};
    for (std::size_t index = 0U; index < blueCount; ++index)
    {
        state.blueActive[index] = true;
        state.blueAlive[index] = true;
    }
    for (std::size_t index = 0U; index < redCount; ++index)
    {
        state.redActive[index] = true;
        state.redAlive[index] = true;
    }
    state.objectiveAvailable = true;
    return state;
}

float Channel(GeneralizedRewardBreakdown const& breakdown,
              GeneralizedRewardChannel const channel) noexcept
{
    return breakdown.values[static_cast<std::size_t>(channel)];
}


bool TestObjectiveThreatPriority()
{
    DefenseSettings const settings{};
    DefenseTrack const objective{0.F, 0.F, 0.F, 0.F, true};
    DefenseTrack const approachingClose{15000.F, 0.F, 3.14159265F, 400.F, true};
    DefenseTrack const approachingFar{120000.F, 0.F, 3.14159265F, 400.F, true};
    float const closeRisk = ObjectiveThreatRisk(settings, approachingClose, objective);
    float const farRisk = ObjectiveThreatRisk(settings, approachingFar, objective);
    if (!(closeRisk > farRisk) || closeRisk <= 0.F)
        return false;
    auto const roles = BlueTeamRoles::Create(3U, 0U);
    if (!roles)
        return false;
    GeneralizedRewardTransition transition{};
    transition.before = MakeState(3U, 2U);
    transition.after = transition.before;
    transition.responsibilities.threatContainment = (closeRisk - farRisk) / 2.F;
    GeneralizedCooperativeRewards rewards{};
    if (AggregateGeneralizedCooperativeRewards(*roles, transition, {}, rewards)
        != GeneralizedRewardStatus::Applied)
        return false;
    float const expected = (closeRisk - farRisk) / 2.F;
    for (std::size_t i = 0U; i < 3U; ++i)
        if (!Near(Channel(rewards.perBlue[i],
                GeneralizedRewardChannel::ThreatContainment), expected))
            return false;
    transition.responsibilities.threatContainment = -expected;
    return AggregateGeneralizedCooperativeRewards(*roles, transition, {}, rewards)
        == GeneralizedRewardStatus::Applied
        && Near(Channel(rewards.team,
            GeneralizedRewardChannel::ThreatContainment), -expected);
}

bool TestOneVersusZero()
{
    auto const roles = BlueTeamRoles::Create(1U, 0U);
    if (!roles)
        return false;
    GeneralizedRewardTransition transition{};
    transition.before = MakeState(1U, 0U);
    transition.after = transition.before;
    transition.responsibilities.leaderPatrol = 0.8F;
    transition.responsibilities.controlEffort[0] = 0.2F;

    GeneralizedCooperativeRewards rewards{};
    if (AggregateGeneralizedCooperativeRewards(
            *roles, transition, {}, rewards)
        != GeneralizedRewardStatus::Applied)
    {
        return false;
    }
    if (!rewards.active[0]
        || rewards.active[1]
        || !Near(Channel(rewards.perBlue[0],
            GeneralizedRewardChannel::PatrolTracking), 0.8F)
        || !Near(Channel(rewards.perBlue[0],
            GeneralizedRewardChannel::ControlEffort), -0.2F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::RedNeutralization), 0.F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::ObjectiveSurvival), 0.F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::MissionOutcome), 0.F))
    {
        return false;
    }

    GeneralizedEvaluationSample sample{};
    sample.finalState = transition.after;
    sample.leaderPatrolQuality = 0.8F;
    GeneralizedEvaluationAccumulator evaluation{};
    if (evaluation.Record(*roles, sample)
        != GeneralizedEvaluationStatus::Recorded)
    {
        return false;
    }
    GeneralizedEvaluationMetrics const metrics = evaluation.Metrics();
    return metrics.episodes == 1U
        && Near(metrics.blueSurvivalRate, 1.F)
        && Near(metrics.redNeutralizationRate, 1.F)
        && Near(metrics.objectiveSurvivalRate, 1.F)
        && Near(metrics.leaderPatrolQuality, 0.8F)
        && Near(metrics.formationQuality, 1.F)
        && Near(metrics.missionSuccessRate, 1.F)
        && Near(metrics.score, 97.F);
}

bool TestIndependentRoleAndPursuitWeights()
{
    auto const roles = BlueTeamRoles::Create(2U, 0U);
    if (!roles)
        return false;
    GeneralizedRewardTransition transition{};
    transition.before = MakeState(2U, 1U);
    transition.after = transition.before;
    transition.responsibilities.leaderPatrol = 0.5F;
    transition.responsibilities.formationSlots[0] = -0.25F;
    transition.responsibilities.pursuitProgress = 0.75F;

    GeneralizedRewardSettings settings{};
    settings.weights.fill(0.F);
    settings.weights[static_cast<std::size_t>(
        GeneralizedRewardChannel::PatrolTracking)] = 1.F;
    settings.weights[static_cast<std::size_t>(
        GeneralizedRewardChannel::FormationKeeping)] = 4.F;
    settings.weights[static_cast<std::size_t>(
        GeneralizedRewardChannel::PursuitProgress)] = 2.F;
    GeneralizedCooperativeRewards result{};
    if (AggregateGeneralizedCooperativeRewards(
            *roles, transition, settings, result)
        != GeneralizedRewardStatus::Applied)
        return false;

    if (!Near(result.perBlue[0].Total(settings), 2.F)
        || !Near(result.perBlue[1].Total(settings), 0.5F)
        || !Near(Channel(result.team,
            GeneralizedRewardChannel::PatrolTracking), 0.25F)
        || !Near(Channel(result.team,
            GeneralizedRewardChannel::FormationKeeping), -0.125F)
        || !Near(Channel(result.team,
            GeneralizedRewardChannel::PursuitProgress), 0.75F))
        return false;
    transition.responsibilities.pursuitProgress =
        std::numeric_limits<float>::quiet_NaN();
    return AggregateGeneralizedCooperativeRewards(
        *roles, transition, settings, result)
        == GeneralizedRewardStatus::InvalidSignals;
}

bool TestTwoVersusTwo()
{
    auto const roles = BlueTeamRoles::Create(2U, 1U);
    if (!roles)
        return false;
    GeneralizedRewardTransition transition{};
    transition.before = MakeState(2U, 2U);
    transition.after = transition.before;
    transition.after.blueAlive[0] = false;
    transition.after.redAlive[0] = false;
    transition.responsibilities.leaderPatrol = 0.75F;
    transition.responsibilities.formationSlots[0] = -0.25F;
    transition.responsibilities.controlEffort[0] = 0.2F;
    transition.responsibilities.controlEffort[1] = 0.1F;
    transition.timeout = true;

    GeneralizedCooperativeRewards rewards{};
    if (AggregateGeneralizedCooperativeRewards(
            *roles, transition, {}, rewards)
        != GeneralizedRewardStatus::Applied)
    {
        return false;
    }
    if (!Near(Channel(rewards.perBlue[0],
            GeneralizedRewardChannel::FormationKeeping), -0.25F)
        || !Near(Channel(rewards.perBlue[1],
            GeneralizedRewardChannel::PatrolTracking), 0.75F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::PatrolTracking), 0.375F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::FormationKeeping), -0.125F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::BlueSurvival), -0.5F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::RedNeutralization), 0.5F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::Timeout), -1.F))
    {
        return false;
    }

    GeneralizedRewardTransition objectiveLost = transition;
    objectiveLost.after.objectiveAvailable = false;
    if (AggregateGeneralizedCooperativeRewards(
            *roles, objectiveLost, {}, rewards)
            != GeneralizedRewardStatus::Applied
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::ObjectiveSurvival), -1.F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::MissionOutcome), -1.F))
    {
        return false;
    }

    // A scheduled red remains an episode participant while unavailable, then
    // may become alive at its activation step. This is not a resurrection and
    // must neither invalidate the transition nor count as a neutralization.
    GeneralizedRewardTransition delayedActivation{};
    delayedActivation.before = MakeState(2U, 2U);
    delayedActivation.before.redAlive[1] = false;
    delayedActivation.after = delayedActivation.before;
    delayedActivation.after.redAlive[1] = true;
    if (AggregateGeneralizedCooperativeRewards(
            *roles, delayedActivation, {}, rewards)
            != GeneralizedRewardStatus::Applied
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::RedNeutralization), 0.F))
    {
        return false;
    }

    GeneralizedEvaluationSample sample{};
    sample.finalState = transition.after;
    sample.leaderPatrolQuality = 0.75F;
    sample.formationSlotQuality[0] = 1.F;
    sample.timeout = true;
    GeneralizedEvaluationAccumulator evaluation{};
    if (evaluation.Record(*roles, sample)
        != GeneralizedEvaluationStatus::Recorded)
    {
        return false;
    }
    GeneralizedEvaluationMetrics const metrics = evaluation.Metrics();
    return Near(metrics.blueSurvivalRate, 0.5F)
        && Near(metrics.redNeutralizationRate, 0.5F)
        && Near(metrics.leaderPatrolQuality, 0.75F)
        && Near(metrics.formationQuality, 0.F)
        && Near(metrics.missionSuccessRate, 0.F)
        && Near(metrics.timeoutRate, 1.F);
}

bool TestNineVersusTenAndMaskValidation()
{
    auto const roles = BlueTeamRoles::Create(9U, 4U);
    if (!roles)
        return false;
    GeneralizedRewardTransition transition{};
    transition.before = MakeState(9U, 10U);
    transition.after = transition.before;
    transition.after.blueAlive[2] = false;
    transition.after.blueAlive[7] = false;
    for (std::size_t index = 0U; index < 5U; ++index)
        transition.after.redAlive[index] = false;
    transition.responsibilities.leaderPatrol = 0.5F;
    transition.responsibilities.formationSlots.fill(0.25F);

    GeneralizedCooperativeRewards rewards{};
    if (AggregateGeneralizedCooperativeRewards(
            *roles, transition, {}, rewards)
        != GeneralizedRewardStatus::Applied
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::BlueSurvival), -2.F / 9.F)
        || !Near(Channel(rewards.team,
            GeneralizedRewardChannel::RedNeutralization), 0.5F))
    {
        return false;
    }

    GeneralizedEvaluationSample sample{};
    sample.finalState = transition.after;
    sample.leaderPatrolQuality = 0.5F;
    sample.formationSlotQuality.fill(1.F);
    GeneralizedEvaluationAccumulator evaluation{};
    if (evaluation.Record(*roles, sample)
        != GeneralizedEvaluationStatus::Recorded)
    {
        return false;
    }
    GeneralizedEvaluationMetrics const metrics = evaluation.Metrics();
    if (!Near(metrics.blueSurvivalRate, 7.F / 9.F)
        || !Near(metrics.redNeutralizationRate, 0.5F)
        || !Near(metrics.formationQuality, 0.75F))
    {
        return false;
    }

    // A blue destroyed before this step remains part of P but receives no new
    // trajectory reward; it is not confused with unused capacity either.
    GeneralizedRewardTransition masked = transition;
    masked.before.blueAlive[0] = false;
    masked.after.blueAlive[0] = false;
    if (AggregateGeneralizedCooperativeRewards(
            *roles, masked, {}, rewards)
            != GeneralizedRewardStatus::Applied
        || rewards.active[0]
        || !Near(rewards.perBlue[0].Total({}), 0.F))
    {
        return false;
    }

    // Unused capacity cannot masquerade as a destroyed participant.
    transition.after.blueAlive[8] = false;
    transition.after.blueAlive[0] = true;
    transition.after.blueActive[8] = false;
    transition.after.blueActive[0] = true;
    return AggregateGeneralizedCooperativeRewards(
        *roles, transition, {}, rewards)
        == GeneralizedRewardStatus::InvalidMasks;
}

bool TestPermutationInvarianceAndInvalidQuality()
{
    auto const firstRoles = BlueTeamRoles::Create(3U, 0U);
    auto const permutedRoles = BlueTeamRoles::Create(3U, 2U);
    if (!firstRoles || !permutedRoles)
        return false;

    GeneralizedRewardTransition first{};
    first.before = MakeState(3U, 2U);
    first.after = first.before;
    first.after.blueAlive[1] = false; // Slot zero.
    first.after.redAlive[0] = false;
    first.responsibilities.leaderPatrol = 0.8F;
    first.responsibilities.formationSlots[0] = 0.2F;
    first.responsibilities.formationSlots[1] = 0.6F;
    first.responsibilities.controlEffort[0] = 0.1F; // Leader.
    first.responsibilities.controlEffort[1] = 0.2F; // Slot zero.
    first.responsibilities.controlEffort[2] = 0.3F; // Slot one.

    GeneralizedRewardTransition permuted = first;
    permuted.after.blueAlive = {false, true, true};
    permuted.responsibilities.controlEffort[0] = 0.2F; // Slot zero.
    permuted.responsibilities.controlEffort[1] = 0.3F; // Slot one.
    permuted.responsibilities.controlEffort[2] = 0.1F; // Leader.

    GeneralizedCooperativeRewards firstRewards{};
    GeneralizedCooperativeRewards permutedRewards{};
    if (AggregateGeneralizedCooperativeRewards(
            *firstRoles, first, {}, firstRewards)
            != GeneralizedRewardStatus::Applied
        || AggregateGeneralizedCooperativeRewards(
            *permutedRoles, permuted, {}, permutedRewards)
            != GeneralizedRewardStatus::Applied)
    {
        return false;
    }
    for (std::size_t channel = 0U;
         channel < kGeneralizedRewardChannelCount; ++channel)
    {
        if (!Near(firstRewards.team.values[channel],
                  permutedRewards.team.values[channel]))
        {
            return false;
        }
    }

    GeneralizedEvaluationSample firstSample{};
    firstSample.finalState = first.after;
    firstSample.leaderPatrolQuality = 0.8F;
    firstSample.formationSlotQuality[0] = 0.2F;
    firstSample.formationSlotQuality[1] = 0.6F;
    GeneralizedEvaluationSample permutedSample = firstSample;
    permutedSample.finalState = permuted.after;

    GeneralizedEvaluationAccumulator firstEvaluation{};
    GeneralizedEvaluationAccumulator permutedEvaluation{};
    if (firstEvaluation.Record(*firstRoles, firstSample)
            != GeneralizedEvaluationStatus::Recorded
        || permutedEvaluation.Record(*permutedRoles, permutedSample)
            != GeneralizedEvaluationStatus::Recorded)
    {
        return false;
    }
    GeneralizedEvaluationMetrics const firstMetrics = firstEvaluation.Metrics();
    GeneralizedEvaluationMetrics const permutedMetrics =
        permutedEvaluation.Metrics();
    if (!Near(firstMetrics.score, permutedMetrics.score)
        || !Near(firstMetrics.blueSurvivalRate,
                 permutedMetrics.blueSurvivalRate)
        || !Near(firstMetrics.formationQuality,
                 permutedMetrics.formationQuality))
    {
        return false;
    }

    firstSample.leaderPatrolQuality =
        std::numeric_limits<float>::quiet_NaN();
    return firstEvaluation.Record(*firstRoles, firstSample)
        == GeneralizedEvaluationStatus::InvalidQuality
        && firstEvaluation.Metrics().episodes == 1U;
}

bool TestSparseDynamicsAdapter()
{
    GeneralizedBlueTeamState blue{};
    blue.active[4] = true;
    blue.active[8] = true;
    blue.activeCount = 2U;
    blue.mobiles[4].available = true;
    blue.mobiles[8].available = false;
    std::array<bool, TeamCapacity::MaximumRedMobiles> redActive{};
    std::array<bool, TeamCapacity::MaximumRedMobiles> redAlive{};
    redActive[7] = true;
    redAlive[7] = true;

    GeneralizedCombatStateMask const mask = MakeGeneralizedCombatStateMask(
        blue, redActive, redAlive, true);
    return mask.blueActive[0] && mask.blueActive[1]
        && !mask.blueActive[2] && mask.blueAlive[0] && !mask.blueAlive[1]
        && mask.redActive[7] && mask.redAlive[7]
        && mask.objectiveAvailable;
}

bool TestDeterministicCampaignBoundary()
{
    PatrolEvaluationSettings settings{};
    settings.episodes = 2U;
    settings.maxSteps = 4U;
    settings.seed = 700U;
    GeneralizedEvaluationCampaign first{settings};
    GeneralizedEvaluationCampaign second{settings};
    Parameters parameters{};
    auto const roles = BlueTeamRoles::Create(1U, 0U);
    if (!roles)
        return false;

    first.Start();
    second.Start();
    for (std::size_t episode = 0U; episode < settings.episodes; ++episode)
    {
        GeneralizedEvaluationCase const firstCase =
            first.CurrentCase(parameters);
        GeneralizedEvaluationCase const secondCase =
            second.CurrentCase(parameters);
        if (firstCase.seed != 700U + episode
            || firstCase.seed != secondCase.seed
            || firstCase.side != secondCase.side
            || !Near(firstCase.arclength, secondCase.arclength))
        {
            return false;
        }
        first.ObserveStep();
        second.ObserveStep();
        GeneralizedEvaluationSample sample{};
        sample.finalState = MakeState(1U, 0U);
        sample.leaderPatrolQuality = 0.8F;
        bool const firstHasMore = first.FinishEpisode(*roles, sample);
        bool const secondHasMore = second.FinishEpisode(*roles, sample);
        if (firstHasMore != (episode + 1U < settings.episodes)
            || firstHasMore != secondHasMore)
        {
            return false;
        }
    }

    GeneralizedEvaluationMetrics const metrics = first.Metrics();
    return !first.InProgress() && first.HasLatest() && first.Improved()
        && first.CompletedCampaigns() == 1U
        && first.TestsFinished() == settings.episodes
        && first.StepsExecuted() == settings.episodes
        && Near(first.Progress(), 1.F)
        && Near(first.BestScore(), 97.F)
        && Near(metrics.score, second.Metrics().score)
        && Near(metrics.blueSurvivalRate, 1.F)
        && Near(metrics.redNeutralizationRate, 1.F);
}

} // namespace

int main()
{
    if (!TestObjectiveThreatPriority())
        return 21;
    if (!TestOneVersusZero())
        return 1;
    if (!TestTwoVersusTwo())
        return 2;
    if (!TestIndependentRoleAndPursuitWeights())
        return 7;
    if (!TestNineVersusTenAndMaskValidation())
        return 3;
    if (!TestPermutationInvarianceAndInvalidQuality())
        return 4;
    if (!TestSparseDynamicsAdapter())
        return 5;
    if (!TestDeterministicCampaignBoundary())
        return 6;
    return 0;
}
