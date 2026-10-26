#include "patrol_evaluator.hpp"
#include "patrol_evaluation_campaign.hpp"
#include <stdexcept>
#include <limits>

namespace patrol = ppo::examples::twovstwo::defenders;

int main()
{
    patrol::Parameters params{};
    auto const straight = [](patrol::Observation const&) {
        return patrol::Action{0.F, 0.F};
    };

    patrol::PatrolEvaluationSettings settings{};
    settings.episodes = 2U;
    settings.maxSteps = 1U;
    settings.minimumCircuits = 1.F;
    auto report = patrol::PatrolEvaluator{params, settings}.Evaluate(straight);
    if (report.truncations != 2U || report.failures != 0U)
        return 1;

    settings.minimumCircuits = 0.F;
    settings.minimumTrackingFraction = 0.F;
    report = patrol::PatrolEvaluator{params, settings}.Evaluate(straight);
    if (!report.Passed() || report.successes != 2U)
        return 2;

    settings.collisionDistance = 1000000.F;
    report = patrol::PatrolEvaluator{params, settings}.Evaluate(straight);
    if (report.failures != 2U
        || report.episodes[0].failure != patrol::PatrolFailure::Collision)
        return 3;

    settings.collisionDistance = 50.F;
    settings.restrictedRadius = 1000000.F;
    report = patrol::PatrolEvaluator{params, settings}.Evaluate(straight);
    if (report.failures != 2U
        || report.episodes[0].failure != patrol::PatrolFailure::RestrictedZone)
        return 4;

    settings.restrictedRadius = 0.F;
    settings.minimumTrackingFraction = 1.F;
    settings.trackingTolerance = 1.e-9F;
    auto const bad = [](patrol::Observation const&) { return patrol::Action{1.F, 1.F}; };
    report = patrol::PatrolEvaluator{params, settings}.Evaluate(bad);
    if (report.failures != 2U
        || report.episodes[0].failure != patrol::PatrolFailure::TrackingQuality)
        return 5;

    // Non-finite actions are rejected, not silently treated as zero.
    auto const nanPolicy = [](patrol::Observation const&) {
        return patrol::Action{std::numeric_limits<float>::quiet_NaN(), 0.F};
    };
    report = patrol::PatrolEvaluator{params, settings}.Evaluate(nanPolicy);
    if (report.failures != 2U
        || report.episodes[0].failure != patrol::PatrolFailure::InvalidState)
        return 6;

    // The live ECS campaign alternates mirrored 2v2/2v1 cases and publishes
    // only a completed deterministic score.
    patrol::PatrolEvaluationSettings campaignSettings{};
    campaignSettings.episodes = 2U;
    campaignSettings.maxSteps = 1U;
    campaignSettings.minimumCircuits = 0.F;
    campaignSettings.minimumTrackingFraction = 0.F;
    patrol::PatrolEvaluationCampaign campaign{campaignSettings};
    campaign.Start(true);
    for (std::size_t episode = 0; episode < campaignSettings.episodes; ++episode)
    {
        auto const test = campaign.CurrentCase(params);
        if (test.side != (episode == 0U ? 1 : -1)
            || test.secondAttackerDestroyed)
            return 7;
        auto state = patrol::InitialState(params, test.side, 0U, test.arclength);
        state.attackers[0] = patrol::Mobile{0.F, 0.F, 0.F, 100.F, 0.F, true};
        state.attackers[1] = patrol::Mobile{0.F, 0.F, 0.F, 100.F, 0.F, true};
        campaign.BeginEpisode(state);
        state.attackers[0].available = false;
        state.attackers[1].available = false;
        campaign.ObserveStep(params, state);
        bool const hasMore = campaign.FinishEpisode(params, state, false);
        if (hasMore != (episode + 1U < campaignSettings.episodes))
            return 8;
    }
    if (campaign.InProgress() || !campaign.HasLatest() || !campaign.Improved()
        || campaign.CompletedCampaigns() != 1U
        || campaign.SuccessRate() != 1.F || campaign.SurvivalRate() != 1.F
        || std::abs(campaign.LatestScore() - 100.F) > 1.e-4F)
        return 9;
    return 0;
}
