#include "generalized_patrol_learner.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace blue = ppo::examples::twovstwo::defenders;
namespace rlt = rl_tools;

namespace {

blue::GeneralizedPatrolLearner::Observations MakeObservations(
    blue::ActiveTeamCounts const counts)
{
    blue::GeneralizedPatrolLearner::Observations observations{};
    auto const roles = blue::BlueTeamRoles::Create(counts.blueMobiles, 0U);
    if (!roles)
        return observations;

    for (std::size_t agent = 0U; agent < counts.blueMobiles; ++agent)
    {
        auto& observation = observations[agent];
        observation.metadata = {counts.blueMobiles, counts.redMobiles};
        observation.values[0] = 0.01F * static_cast<float>(agent + 1U);
        observation.values[blue::GeneralizedObservationLayout::LidarOffset]
            = 1.F;
        blue::EncodeGeneralizedRole(observation.values,
                                    *roles->Assignment(agent));
        blue::EncodeGeneralizedTeamCounts(observation.values, counts);
    }
    return observations;
}

bool SamePooledContext(blue::GeneralizedPatrolLearner::CentralCriticObservation const& lhs,
          blue::GeneralizedPatrolLearner::CentralCriticObservation const& rhs)
{
    for (std::size_t feature = blue::GeneralizedCriticLayout::MeanOffset;
         feature < blue::GeneralizedCriticLayout::LivingFractionOffset;
         ++feature)
        if (std::abs(lhs[feature] - rhs[feature]) > 1.0e-6F)
            return false;
    return true;
}

template<typename Callable>
bool ThrowsRuntimeError(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (std::runtime_error const&)
    {
        return true;
    }
    return false;
}

std::string MakeV4TrainingFixture()
{
    blue::Device device{};
    using LegacyActor = blue::TrainingConfig::NN::ACTOR_TYPE;
    using Critic = blue::TrainingConfig::NN::CRITIC_TYPE;
    LegacyActor actor{};
    Critic critic{};
    rlt::malloc(device, actor);
    rlt::malloc(device, critic);
    auto rng = rlt::random::default_engine(
        typename blue::Device::SPEC::RANDOM{}, 4455U);
    rlt::init_weights(device, actor, rng);
    rlt::init_weights(device, critic, rng);
    typename blue::TrainingConfig::NN::ACTOR_OPTIMIZER actorOptimizer{};
    typename blue::TrainingConfig::NN::CRITIC_OPTIMIZER criticOptimizer{};
    rlt::reset_optimizer_state(device, actorOptimizer, actor);
    rlt::reset_optimizer_state(device, criticOptimizer, critic);
    rlt::set(actor.red.encoderW.gradient_first_order_moment, 0, 0,
             0.375F);
    std::stringstream stream;
    blue::WritePatrolTrainingState(
        stream, actor, critic, actorOptimizer, criticOptimizer, rng, 77U);
    std::string fixture = stream.str();
    rlt::free(device, critic);
    rlt::free(device, actor);
    return fixture;
}

void WriteText(std::filesystem::path const& path, std::string const& text)
{
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    stream << text;
    if (!stream)
        throw std::runtime_error("Cannot write generalized learner fixture");
}

bool VerifyTeamSize(blue::ActiveTeamCounts const counts)
{
    blue::GeneralizedPatrolLearner learner{counts, 731U, 2U};
    auto observations = MakeObservations(counts);
    blue::GeneralizedPatrolLearner::AvailabilityMask availability{};
    std::fill_n(availability.begin(), counts.blueMobiles, true);
    if (counts.blueMobiles > 1U)
        availability[counts.blueMobiles - 1U] = false;

    void const* actorAddress{};
    float initialLogStd{};
    learner.WithActor([&](auto const& actor) {
        actorAddress = &actor;
        initialLogStd = rlt::get(actor.log_std.parameters, 0, 0);
    });
    auto const actions = learner.ChooseActions(observations, availability);
    if (actorAddress == nullptr || learner.ActorEvaluationCount() != 1U
        || learner.LastTeamActionCount() != counts.blueMobiles
        || learner.TotalActionsSampled() != counts.blueMobiles
        || !learner.HasPendingAction())
    {
        return false;
    }
    for (std::size_t agent = 0U; agent < counts.blueMobiles; ++agent)
    {
        if (!std::isfinite(actions[agent][0])
            || !std::isfinite(actions[agent][1]))
        {
            return false;
        }
    }

    blue::GeneralizedPatrolLearner::Rewards rewards{};
    std::fill_n(rewards.begin(), counts.blueMobiles, 0.25F);
    if (learner.Observe(observations, rewards, availability, false, false)
        || learner.CollectedTeamSteps() != 1U
        || learner.LastTeamTransitionCount() != counts.blueMobiles
        || learner.TotalTransitionsRecorded() != counts.blueMobiles
        || learner.CurrentMasks() != availability)
    {
        return false;
    }

    (void)learner.ChooseActions(observations, availability);
    if (!learner.Observe(observations, rewards, availability, true, false)
        || learner.CollectedTeamSteps() != 0U || learner.StepCount() != 1U
        || learner.TotalActionsSampled() != 2U * counts.blueMobiles
        || learner.TotalTransitionsRecorded() != 2U * counts.blueMobiles)
    {
        return false;
    }
    void const* actorAfterUpdate{};
    float updatedLogStd{};
    learner.WithActor(
        [&](auto const& actor) {
            actorAfterUpdate = &actor;
            updatedLogStd = rlt::get(actor.log_std.parameters, 0, 0);
        });
    auto const diagnostics = learner.Diagnostics();
    std::size_t const liveAgents = counts.blueMobiles > 1U
        ? counts.blueMobiles - 1U
        : 1U;
    return actorAddress == actorAfterUpdate
        && diagnostics.learnableTransitions == 2U * liveAgents
        && diagnostics.generalizedAdvantageAbsMean > 0.F
        && diagnostics.actorDerivativeL1 > 0.F
        && diagnostics.criticDerivativeL1 > 0.F
        && diagnostics.logStdGradientL1 > 0.F
        && std::isfinite(updatedLogStd) && updatedLogStd != initialLogStd;
}

} // namespace

int main()
{
    if (!VerifyTeamSize({1U, 0U}))
        return 1;
    if (!VerifyTeamSize({3U, 1U}))
        return 14;
    if (!VerifyTeamSize({4U, 4U}))
        return 15;
    if (!VerifyTeamSize({2U, 2U}))
        return 2;
    if (!VerifyTeamSize({9U, 10U}))
        return 3;

    blue::ActiveTeamCounts const counts{2U, 2U};
    auto observations = MakeObservations(counts);
    blue::GeneralizedPatrolLearner::AvailabilityMask availability{};
    availability[0] = true;
    availability[1] = true;
    blue::GeneralizedPatrolLearner first{counts, 91U, 1U};
    blue::GeneralizedPatrolLearner permuted{counts, 91U, 1U};
    (void)first.ChooseActions(observations, availability);
    std::swap(observations[0], observations[1]);
    (void)permuted.ChooseActions(observations, availability);
    if (!SamePooledContext(first.LastCentralCriticObservation(),
              permuted.LastCentralCriticObservation()))
    {
        return 4;
    }

    auto const& ownFirst = first.LastAgentCriticObservation(0U);
    auto const& ownSecond = first.LastAgentCriticObservation(1U);
    if (std::abs(ownFirst[0] - ownSecond[0]) < 1.0e-5F)
        return 13;
    blue::GeneralizedPatrolLearner::Rewards rewards{};
    rewards[0] = 1.F;
    rewards[1] = 1.F;
    if (!first.Observe(observations, rewards, availability, true, false))
        return 5;
    auto const firstEvaluation = first.EvaluateActions(
        observations, availability);
    auto const repeatedEvaluation = first.EvaluateActions(
        observations, availability);
    if (firstEvaluation != repeatedEvaluation || first.HasPendingAction()
        || first.CollectedTeamSteps() != 0U
        || first.TotalActionsSampled() != counts.blueMobiles)
    {
        return 12;
    }

    std::filesystem::path const directory =
        std::filesystem::temp_directory_path();
    std::filesystem::path const policy =
        directory / "ppo_generalized_learner_v8_test.txt";
    std::filesystem::path const training =
        directory / "ppo_generalized_learner_v6_test.txt";
    std::error_code ignored;
    std::filesystem::remove(policy, ignored);
    std::filesystem::remove(training, ignored);
    first.SaveActor(policy);
    first.SaveTrainingState(training);

    blue::GeneralizedPatrolLearner resumed{counts, 123U, 1U};
    resumed.LoadActor(policy);
    resumed.LoadTrainingState(training);
    if (resumed.StepCount() != first.StepCount()
        || resumed.ActiveCounts().blueMobiles != counts.blueMobiles
        || resumed.ActiveCounts().redMobiles != counts.redMobiles)
    {
        return 6;
    }
    // New learner shape is independent of the checkpoint's old team count.
    blue::GeneralizedPatrolLearner resumedThree{{3U, 1U}, 123U, 1U};
    resumedThree.LoadTrainingState(training);
    if (resumedThree.StepCount() != first.StepCount()
        || resumedThree.ActiveCounts().blueMobiles != 3U
        || resumedThree.ActiveCounts().redMobiles != 1U)
        return 16;
    blue::GeneralizedPatrolLearner resumedNine{{9U, 10U}, 123U, 1U};
    resumedNine.LoadTrainingState(training);
    if (resumedNine.StepCount() != first.StepCount())
        return 17;

    auto const uninterruptedActions =
        first.ChooseActions(observations, availability);
    auto const resumedActions =
        resumed.ChooseActions(observations, availability);
    if (uninterruptedActions != resumedActions)
        return 7;

    std::string const v4Fixture = MakeV4TrainingFixture();
    std::filesystem::path const v4 =
        directory / "ppo_generalized_learner_v4_migration_test.txt";
    std::filesystem::path const badV4 =
        directory / "ppo_generalized_learner_v4_bad_test.txt";
    WriteText(v4, v4Fixture);
    blue::GeneralizedPatrolLearner migrated{counts, 999U, 1U};
    auto const migration = migrated.MigrateV4TrainingState(v4, 765U);
    float migratedMoment{};
    migrated.WithActor([&](auto const& actor) {
        migratedMoment = rlt::get(
            actor.red.encoderW.gradient_first_order_moment, 0, 0);
    });
    if (!migration.actorParametersMigrated
        || !migration.compatibleActorMomentsMigrated
        || !migration.criticResetRequired
        || !migration.optimizerScalarsResetRequired
        || !migration.rolloutAndRngResetRequired
        || migratedMoment != 0.375F || migrated.StepCount() != 0U)
    {
        return 8;
    }

    WriteText(badV4, v4Fixture.substr(0U, v4Fixture.size() / 2U));
    if (!ThrowsRuntimeError(
            [&] { migrated.MigrateV4TrainingState(badV4, 765U); }))
    {
        return 9;
    }
    std::string wrongVersion = v4Fixture;
    auto const version = wrongVersion.find("BLUE_PATROL_TRAINING_V4 4");
    if (version == std::string::npos)
        return 10;
    wrongVersion[version + std::string{"BLUE_PATROL_TRAINING_V4 "}.size()]
        = '3';
    WriteText(badV4, wrongVersion);
    if (!ThrowsRuntimeError(
            [&] { migrated.MigrateV4TrainingState(badV4, 765U); }))
    {
        return 11;
    }

    std::filesystem::remove(policy, ignored);
    std::filesystem::remove(training, ignored);
    std::filesystem::remove(v4, ignored);
    std::filesystem::remove(badV4, ignored);
    return 0;
}
