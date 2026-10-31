/**
 * @file Examples/BlueDefense/tests/generalized_learner_tests.cpp
 * @brief Tests et assertions déterministes pour generalized learner tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "generalized_patrol_learner.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace blue = ppo::examples::bluedefense::defenders;
namespace rlt = rl_tools;

namespace {

/**
 * @brief Construit, transforme ou échantillonne les données produites par MakeObservations.
 */
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

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
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

/**
 * @brief Construit, transforme ou échantillonne les données produites par MakeV4TrainingFixture.
 * @details Utilise des fichiers ou chemins. Peut signaler un état invalide par exception.
 */
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

/**
 * @brief Sérialise ou écrit les données associées à WriteText selon les garanties implémentées.
 * @details Utilise des fichiers ou chemins. Peut signaler un état invalide par exception.
 */
void WriteText(std::filesystem::path const& path, std::string const& text)
{
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    stream << text;
    if (!stream)
        throw std::runtime_error("Cannot write generalized learner fixture");
}

/**
 * @brief Réalise l'opération VerifyOptimizationFailureGuard; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
bool VerifyOptimizationFailureGuard()
{
    blue::GeneralizedOptimizationGuard guard{};
    // Exhausted storage must be rejected before any transition indexing.
    try
    {
        guard.RequireWritable(2U, 2U, __FILE__, __LINE__);
        return false;
    }
    catch (std::out_of_range const& error)
    {
        std::string const message = error.what();
        if (message.find("generalized_learner_tests.cpp") == std::string::npos
            || message.find("line=") == std::string::npos
            || message.find("collected=2 capacity=2") == std::string::npos)
            return false;
    }
    if (guard.Failed())
        return false;

    try
    {
        guard.Run([] { throw std::runtime_error("simulated Adam failure"); },
                  __FILE__, __LINE__);
        return false;
    }
    catch (std::runtime_error const& error)
    {
        std::string const message = error.what();
        if (message.find("simulated Adam failure") == std::string::npos
            || message.find("file=") == std::string::npos
            || message.find("line=") == std::string::npos)
            return false;
    }
    if (!guard.Failed())
        return false;
    try
    {
        guard.RequireWritable(0U, 2U, __FILE__, __LINE__);
        return false;
    }
    catch (std::logic_error const& error)
    {
        if (std::string{error.what()}.find("restart from a valid checkpoint")
            == std::string::npos)
            return false;
    }
    bool invoked = false;
    try
    {
        guard.Run([&invoked] { invoked = true; }, __FILE__, __LINE__);
        return false;
    }
    catch (std::logic_error const&)
    {
        return !invoked;
    }
}

/**
 * @brief Réalise l'opération VerifyAbortPendingStep; voir les préconditions et effets dans son corps C++.
 */
bool VerifyAbortPendingStep()
{
    blue::ActiveTeamCounts const counts{1U, 0U};
    blue::GeneralizedPatrolLearner learner{counts, 412U, 2U};
    auto const observations = MakeObservations(counts);
    blue::GeneralizedPatrolLearner::AvailabilityMask availability{};
    availability[0] = true;
    blue::GeneralizedPatrolLearner::Rewards rewards{};
    rewards[0] = 0.25F;

    (void)learner.ChooseActions(observations, availability);
    learner.AbortPendingStep();
    if (learner.HasPendingAction() || learner.CollectedTeamSteps() != 0U
        || learner.OptimizationFailed())
        return false;
    (void)learner.ChooseActions(observations, availability);
    if (learner.Observe(observations, rewards, availability, false, false))
        return false;
    (void)learner.ChooseActions(observations, availability);
    return learner.Observe(observations, rewards, availability, true, false)
        && learner.StepCount() == 1U && !learner.OptimizationFailed();
}

/**
 * @brief Réalise l'opération VerifyDeadActorMask; voir les préconditions et effets dans son corps C++.
 */
bool VerifyDeadActorMask()
{
    blue::ActiveTeamCounts const counts{1U, 0U};
    blue::GeneralizedPatrolLearner learner{counts, 812U, 2U};
    auto const observations = MakeObservations(counts);
    blue::GeneralizedPatrolLearner::AvailabilityMask alive{};
    alive[0] = true;
    blue::GeneralizedPatrolLearner::AvailabilityMask dead{};
    auto const means = learner.EvaluateActions(observations, alive);
    blue::GeneralizedPatrolLearner::Rewards rewards{};
    rewards[0] = -1.F;
    (void)learner.ChooseActions(observations, dead);
    if (learner.Observe(observations, rewards, dead, false, false))
        return false;
    (void)learner.ChooseActions(observations, dead);
    if (!learner.Observe(observations, rewards, dead, true, false))
        return false;
    auto const diagnostics = learner.Diagnostics();
    return diagnostics.learnableTransitions == 0U
        && diagnostics.criticTransitions == 2U
        && diagnostics.actorDerivativeL1 == 0.F
        && diagnostics.logStdGradientL1 == 0.F
        && diagnostics.criticDerivativeL1 > 0.F
        && means == learner.EvaluateActions(observations, alive);
}

/**
 * @brief Réalise l'opération VerifyPostLossMissionCredit; voir les préconditions et effets dans son corps C++.
 */
bool VerifyPostLossMissionCredit()
{
    blue::ActiveTeamCounts const counts{2U, 0U};
    auto const observations = MakeObservations(counts);
    float absoluteAdvantages[2]{};
    for (std::size_t run = 0U; run < 2U; ++run)
    {
        blue::GeneralizedPatrolLearner learner{counts, 192U, 3U};
        blue::GeneralizedPatrolLearner::AvailabilityMask availability{};
        availability[0] = availability[1] = true;
        blue::GeneralizedPatrolLearner::Rewards rewards{};
        rewards[0] = -24.F;
        (void)learner.ChooseActions(observations, availability);
        availability[0] = false;
        if (learner.Observe(observations, rewards, availability, false, false))
            return false;
        (void)learner.ChooseActions(observations, availability);
        auto const& dead = learner.LastAgentCriticObservation(0U);
        auto const& living = learner.LastAgentCriticObservation(1U);
        for (std::size_t feature = 0U; feature < blue::GeneralizedCriticLayout::Compact; ++feature)
            if (dead[feature] != 0.F)
                return false;
        if (!SamePooledContext(dead, living)
            || dead[blue::GeneralizedCriticLayout::LivingFractionOffset] <= 0.F)
            return false;
        rewards.fill(0.F);
        if (learner.Observe(observations, rewards, availability, false, false))
            return false;
        (void)learner.ChooseActions(observations, availability);
        rewards[0] = run == 0U ? 0.F : -10.F;
        if (!learner.Observe(observations, rewards, availability, true, false))
            return false;
        auto const diagnostics = learner.Diagnostics();
        if (diagnostics.learnableTransitions != 4U || diagnostics.criticTransitions != 6U)
            return false;
        absoluteAdvantages[run] = diagnostics.generalizedAdvantageAbsMean;
    }
    // Only the dead slot's final reward differs: it must affect earlier
    // learnable actions rather than being discarded by availability masking.
    return absoluteAdvantages[1] > absoluteAdvantages[0] + 0.1F;
}

/**
 * @brief Réalise l'opération VerifyTeamSize; voir les préconditions et effets dans son corps C++.
 */
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
        && diagnostics.criticTransitions == 2U * counts.blueMobiles
        && diagnostics.generalizedAdvantageAbsMean > 0.F
        && diagnostics.actorDerivativeL1 > 0.F
        && diagnostics.criticDerivativeL1 > 0.F
        && diagnostics.logStdGradientL1 > 0.F
        && std::isfinite(updatedLogStd) && updatedLogStd != initialLogStd;
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    if (!VerifyOptimizationFailureGuard())
        return 18;
    if (!VerifyAbortPendingStep())
        return 19;
    if (!VerifyDeadActorMask())
        return 20;
    if (!VerifyPostLossMissionCredit())
        return 21;
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
    // Evaluation must be read-only: no new PPO update, rollout or sampling.
    if (firstEvaluation != repeatedEvaluation || first.HasPendingAction()
        || first.CollectedTeamSteps() != 0U
        || first.StepCount() != 1U
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
