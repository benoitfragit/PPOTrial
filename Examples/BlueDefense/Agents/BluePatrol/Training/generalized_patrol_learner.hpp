/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_patrol_learner.hpp
 * @brief Collecte de transitions multi-agents, optimisation PPO et sérialisation des états d'entraînement.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "generalized_artifact.hpp"
#include "generalized_training_checkpoint.hpp"
#include "generalized_critic_network.hpp"
#include "ppokit/runtime/hosted_trainer.hpp"

#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "ppokit/backend/rltools/mlp_probes.hpp"
#include "ppokit/graph/probes.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppo::examples::bluedefense::defenders {

/** @brief Finite diagnostics from the latest generalized PPO update. */
struct GeneralizedTrainingDiagnostics final
{
    /** @brief Living actor rows used for policy gradients. */
    std::size_t learnableTransitions{};
    /** @brief Participating rows, including dead slots, used by the critic. */
    std::size_t criticTransitions{};
    /** @brief Mean absolute GAE value before normalization. */
    float generalizedAdvantageAbsMean{};
    /** @brief Sum of absolute actor-output loss derivatives. */
    float actorDerivativeL1{};
    /** @brief Sum of absolute centralized-value loss derivatives. */
    float criticDerivativeL1{};
    /** @brief Sum of absolute log-standard-deviation gradients. */
    float logStdGradientL1{};
    /** @brief Mean negative clipped PPO surrogate over all actor samples. */
    float policyLoss{};
    /** @brief Mean squared centralized-value error over all critic samples. */
    float valueLoss{};
    /** @brief Mean diagonal-Gaussian policy entropy. */
    float entropy{};
    /** @brief Mean non-negative approximate KL estimator. */
    float approximateKl{};
    /** @brief Fraction of actor samples outside the PPO clipping interval. */
    float clipFraction{};
    /** @brief Fraction of return variance explained by pre-update values. */
    float explainedVariance{};
    /** @brief Mean action standard deviation after the completed update. */
    float actionStdMean{};
};

/**
 * @brief Enforces fail-stop behavior if PPO optimization throws.
 *
 * A partially updated actor, critic or Adam state must never be used for the
 * next rollout. The diagnostic location is the detecting/catching call site;
 * an exception from RLTools may originate deeper in the library.
 */
class GeneralizedOptimizationGuard final
{
public:
/**
 * @brief Réalise l'opération Diagnostic; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
    static std::string Diagnostic(char const* const file, int const line,
                                  char const* const phase,
                                  std::string const& cause)
    {
        return std::string{"[MAPPO_ERROR] file="} + file
            + " line=" + std::to_string(line)
            + " phase=" + phase + " cause=" + cause;
    }

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
    void RequireWritable(std::size_t const collected,
                         std::size_t const capacity,
                         char const* const file, int const line) const
    {
        if (failed_)
            throw std::logic_error(Diagnostic(
                file, line, "rollout",
                "PPO optimizer previously failed; restart from a valid checkpoint"));
        if (collected >= capacity)
            throw std::out_of_range(Diagnostic(
                file, line, "rollout",
                "training buffer full: collected=" + std::to_string(collected)
                    + " capacity=" + std::to_string(capacity)));
    }

    template<typename Optimizer>
    void Run(Optimizer&& optimizer,
             char const* const file, int const line)
    {
        if (failed_)
            throw std::logic_error(Diagnostic(
                file, line, "OptimizeRollout",
                "PPO optimizer previously failed; restart required"));
        try
        {
            std::forward<Optimizer>(optimizer)();
        }
        catch (std::exception const& error)
        {
            failed_ = true;
            throw std::runtime_error(Diagnostic(
                file, line, "OptimizeRollout (caught here; origin may be external)",
                error.what()));
        }
        catch (...)
        {
            failed_ = true;
            throw std::runtime_error(Diagnostic(
                file, line, "OptimizeRollout (caught here; origin may be external)",
                "unknown non-std exception"));
        }
    }

/**
 * @brief Réalise l'opération Failed; voir les préconditions et effets dans son corps C++.
 */
    bool Failed() const noexcept { return failed_; }

private:
    bool failed_{};
};

/**
 * @brief One shared-actor MAPPO rollout contract for a runtime P/N blue team.
 *
 * A call to ChooseActions evaluates one actor batch containing exactly the P
 * episode mobiles. The matching Observe call then appends exactly P rows with
 * one common team-step index. Destroyed mobiles keep their stable row but have
 * a zero actor mask. Their collective returns and critic rows remain valid.
 * Storage and neural-network workspaces are allocated by
 * the constructor, never by the per-team-step success path.
 *
 * The historical two-agent PatrolLearner remains unchanged. This class owns the
 * v10 actor, agent-conditioned centralized critic and V7 training state.
 */
/** @brief What the actor graph blocks and the critic layers output during a probed update. */
struct GeneralizedLayerProbes final
{
    std::vector<ppokit::runtime::BlockProbe> actor;
    std::vector<ppokit::runtime::BlockProbe> critic;
};

class GeneralizedPatrolLearner final
{
public:
    using Observations = std::array<GeneralizedObservation,
                                    TeamCapacity::MaximumBlueMobiles>;
    using Actions = std::array<Action, TeamCapacity::MaximumBlueMobiles>;
    using Rewards = std::array<float, TeamCapacity::MaximumBlueMobiles>;
    using AvailabilityMask = std::array<bool,
                                        TeamCapacity::MaximumBlueMobiles>;
    using CentralCriticObservation = GeneralizedCriticObservation;

    /**
     * @brief Creates a generalized learner for immutable episode P/N counts.
     * @param counts Active mobile counts, with P in [1,9] and N in [0,10].
     * @param seed Shared actor initialization and Gaussian sampling seed.
     * @param rolloutTeamSteps Synchronized ECS steps collected per update.
     */
    GeneralizedPatrolLearner(ActiveTeamCounts const counts,
                             std::uint32_t const seed,
                             std::size_t const rolloutTeamSteps = 2048U)
        : counts_(counts), rolloutTeamSteps_(rolloutTeamSteps), seed_(seed),
          rng_(MakePatrolEngine(seed)),
          trainer_(models_, random_, CheckedTeamSize(counts, rolloutTeamSteps),
                   rolloutTeamSteps, TeamSettings())
    {
        if (!Valid(counts_))
            throw std::invalid_argument("Invalid generalized P/N team counts");

        trainer_.SetHooks(ProgressHooks());

        try
        {
            Allocate();
            rlt::init_weights(device_, actor_, rng_);
            rlt::init_weights(device_, critic_, rng_);
            rlt::reset_optimizer_state(device_, actorOptimizer_, actor_);
            rlt::reset_optimizer_state(device_, criticOptimizer_, critic_);
            actorOptimizer_.parameters.alpha = 3.e-4F;
            criticOptimizer_.parameters.alpha = 3.e-4F;
        }
        catch (...)
        {
            Release();
            throw;
        }
    }

    ~GeneralizedPatrolLearner() { Release(); }
    GeneralizedPatrolLearner(GeneralizedPatrolLearner const&) = delete;
    GeneralizedPatrolLearner& operator=(GeneralizedPatrolLearner const&) = delete;
    GeneralizedPatrolLearner(GeneralizedPatrolLearner&&) = delete;
    GeneralizedPatrolLearner& operator=(GeneralizedPatrolLearner&&) = delete;

    /**
     * @brief Samples one decentralized action for each of the P stable slots.
     * @param observations First P entries are valid same-instant observations.
     * @param availability True for currently alive slots; false for destroyed.
     * @return Fixed-capacity result whose first P actions are meaningful.
     */
    Actions ChooseActions(Observations const& observations,
                          AvailabilityMask const& availability)
    {
        optimizationGuard_.RequireWritable(
            trainer_.CollectedTeamSteps(), rolloutTeamSteps_, __FILE__, __LINE__);
        if (trainer_.HasPendingAction())
            throw std::logic_error(GeneralizedOptimizationGuard::Diagnostic(
                __FILE__, __LINE__, "ChooseActions",
                "synchronized feedback is missing"));
        ValidateBatch(observations, availability);

        TransformObservations(observations);
        ++actorEvaluationCount_;

        BuildAgentCriticObservations(availability, pendingAgentCritic_);

        // The shared rollout and the PPO update belong to PPOKit (HostedTrainer):
        // this pass already gave the means, so they are handed over as they are.
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            teamObservations_[agent] = observations[agent].values.data();
            teamCritics_[agent] = pendingAgentCritic_[agent].data();
            teamPresence_[agent] = availability[agent] ? ppokit::runtime::AgentPresence::Active
                                                       : ppokit::runtime::AgentPresence::Destroyed;
            for (std::size_t action = 0U;
                 action < GeneralizedObservationLayout::ActionSize; ++action)
            {
                teamMeans_[agent * GeneralizedObservationLayout::ActionSize + action] =
                    rlt::get(inferenceOutput_, static_cast<Index>(agent),
                             static_cast<Index>(action));
            }
        }
        Actions actions{};
        try
        {
            trainer_.ChooseActionsFromMeans(teamObservations_.data(), teamMeans_.data(),
                                            teamCritics_.data(), teamPresence_.data(),
                                            teamActions_.data());
        }
        catch (std::runtime_error const& error)
        {
            throw std::runtime_error(GeneralizedOptimizationGuard::Diagnostic(
                __FILE__, __LINE__, "ChooseActions", error.what()));
        }

        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            for (std::size_t action = 0U;
                 action < GeneralizedObservationLayout::ActionSize; ++action)
            {
                actions[agent][action] =
                    teamActions_[agent * GeneralizedObservationLayout::ActionSize + action];
            }
        }
        pendingAvailability_ = availability;
        lastTeamActionCount_ = counts_.blueMobiles;
        totalActionsSampled_ += counts_.blueMobiles;
        return actions;
    }

    /**
     * @brief Evaluates deterministic actor means without collecting rollout.
     * @param observations First P entries are valid same-instant observations.
     * @param availability True for currently alive slots; false for destroyed.
     * @return Fixed-capacity actor means; unavailable slots remain zero.
     *
     * This update-boundary path is reserved for deterministic live-ECS
     * evaluation. It neither advances the learner RNG nor creates a pending
     * transition, so evaluation cannot contaminate PPO collection or resume.
     */
    Actions EvaluateActions(Observations const& observations,
                            AvailabilityMask const& availability)
    {
        if (optimizationGuard_.Failed())
            throw std::logic_error(GeneralizedOptimizationGuard::Diagnostic(
                __FILE__, __LINE__, "EvaluateActions",
                "PPO optimizer previously failed; evaluation forbidden"));
        if (!trainer_.AtUpdateBoundary())
            throw std::logic_error(
                "Generalized MAPPO evaluation requires an update boundary");
        ValidateBatch(observations, availability);
        TransformObservations(observations);
        ++actorEvaluationCount_;

        Actions actions{};
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            if (!availability[agent])
                continue;
            for (std::size_t action = 0U;
                 action < GeneralizedObservationLayout::ActionSize; ++action)
            {
                float const mean = rlt::get(
                    inferenceOutput_, static_cast<Index>(agent),
                    static_cast<Index>(action));
                if (!std::isfinite(mean))
                    throw std::runtime_error(
                        "Nonfinite generalized MAPPO evaluation action");
                actions[agent][action] = mean;
            }
        }
        return actions;
    }

    /**
     * @brief Records the P simultaneous transitions and updates when full.
     * @param nextObservations First P post-resolution observations.
     * @param rewards Per-slot cooperative/credit-assigned finite rewards.
     * @param nextAvailability Alive mask after world resolution.
     * @param terminated True for a terminal mission outcome.
     * @param truncated True for a finite-horizon boundary.
     * @return True when one rollout update completed.
     */
    bool Observe(Observations const& nextObservations,
                 Rewards const& rewards,
                 AvailabilityMask const& nextAvailability,
                 bool const terminated, bool const truncated)
    {
        optimizationGuard_.RequireWritable(
            trainer_.CollectedTeamSteps(), rolloutTeamSteps_, __FILE__, __LINE__);
        if (!trainer_.HasPendingAction())
            throw std::logic_error(GeneralizedOptimizationGuard::Diagnostic(
                __FILE__, __LINE__, "Observe",
                "actions are required before feedback"));
        ValidateBatch(nextObservations, nextAvailability);

        TransformObservations(nextObservations);
        std::array<CentralCriticObservation,
                   TeamCapacity::MaximumBlueMobiles> nextCritics{};
        BuildAgentCriticObservations(nextAvailability, nextCritics);
        std::array<float const*, TeamCapacity::MaximumBlueMobiles> nextCriticRows{};
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            nextCriticRows[agent] = nextCritics[agent].data();
        }

        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            if (!std::isfinite(rewards[agent]))
                throw std::invalid_argument(GeneralizedOptimizationGuard::Diagnostic(
                    __FILE__, __LINE__, "Observe",
                    "nonfinite reward for agent=" + std::to_string(agent)));
        }
        // The whole team ends together; a terminated step never bootstraps.
        ppokit::runtime::StepEnd const end = terminated
            ? ppokit::runtime::StepEnd::Terminated
            : (truncated ? ppokit::runtime::StepEnd::Truncated
                         : ppokit::runtime::StepEnd::Running);
        std::array<ppokit::runtime::StepEnd, TeamCapacity::MaximumBlueMobiles> ends{};
        ends.fill(end);

        lastMasks_ = pendingAvailability_;
        lastTeamTransitionCount_ = counts_.blueMobiles;
        totalTransitionsRecorded_ += counts_.blueMobiles;
        bool const fillsRollout = trainer_.CollectedTeamSteps() + 1U == rolloutTeamSteps_;
        if (!fillsRollout)
            return trainer_.Observe(nextCriticRows.data(), rewards.data(), ends.data());

        // Never advance/reset this rollout if optimizer or Adam updates fail.
        // Fail-stop prevents indexing a full buffer on the next team step.
        if (probing_)
            StartProbes();
        optimizationGuard_.Run(
            [&] { trainer_.Observe(nextCriticRows.data(), rewards.data(), ends.data()); },
            __FILE__, __LINE__);
        if (probeActive_)
            FinishProbes();
        ++completedUpdates_;
        return true;
    }

    /** @brief Returns immutable episode P/N counts. */
    ActiveTeamCounts ActiveCounts() const noexcept { return counts_; }
    /** @brief Returns synchronized team steps currently buffered. */
    std::size_t CollectedTeamSteps() const noexcept
    {
        return trainer_.CollectedTeamSteps();
    }
    /** @brief Returns whether Observe still owes feedback for one actor batch. */
    bool HasPendingAction() const noexcept { return trainer_.HasPendingAction(); }
    /** @brief True after a possibly partial/invalid PPO optimizer update. */
    bool OptimizationFailed() const noexcept
    {
        return optimizationGuard_.Failed();
    }

    /**
     * @brief Discards an action batch that cannot receive valid feedback.
     *
     * Collected transitions and optimizer state are preserved. This recovery
     * path is reserved for a rejected ECS step. It never clears a poisoned
     * optimizer or resets a full rollout after an optimization failure.
     */
    void AbortPendingStep() noexcept
    {
        trainer_.AbortPendingStep();
        pendingAvailability_.fill(false);
    }

    /** @brief Returns completed generalized policy updates. */
    std::uint64_t StepCount() const noexcept { return completedUpdates_; }

    /**
     * @brief Probes the actor blocks and critic layers of the next updates (outputs, dead and
     *        saturated units, gradient sizes). Read-only: the same weights as without probes.
     */
    void ProbeLayers(bool probing) noexcept { probing_ = probing; }
    /** @brief Probes of the last update run while probing (empty before). */
    GeneralizedLayerProbes const& LayerProbes() const noexcept { return lastProbes_; }
    /** @brief Statistics of the latest completed update (zeros before the first). */
    ppokit::runtime::TeamUpdateStatistics const& LastUpdate() const noexcept
    {
        return trainer_.LastUpdate();
    }
    /** @brief The PPO hyper-parameters this learner trains with. */
    static ppokit::runtime::TeamPpoSettings PpoSettings() noexcept { return TeamSettings(); }
    /** @brief Returns sampled decentralized actions in the latest team step. */
    std::size_t LastTeamActionCount() const noexcept
    {
        return lastTeamActionCount_;
    }
    /** @brief Returns recorded synchronized rows in the latest team step. */
    std::size_t LastTeamTransitionCount() const noexcept
    {
        return lastTeamTransitionCount_;
    }
    /** @brief Returns total action rows sampled since construction/load. */
    std::uint64_t TotalActionsSampled() const noexcept
    {
        return totalActionsSampled_;
    }
    /** @brief Returns total transition rows recorded since construction/load. */
    std::uint64_t TotalTransitionsRecorded() const noexcept
    {
        return totalTransitionsRecorded_;
    }
    /** @brief Returns the current/last per-slot rollout learning masks. */
    AvailabilityMask CurrentMasks() const noexcept
    {
        return trainer_.HasPendingAction() ? pendingAvailability_ : lastMasks_;
    }
    /** @brief Returns one agent-conditioned critic input. */
    CentralCriticObservation const& LastAgentCriticObservation(
        std::size_t const agent) const
    {
        if (agent >= counts_.blueMobiles)
            throw std::out_of_range("Invalid generalized critic agent slot");
        return pendingAgentCritic_[agent];
    }

    /** @brief Compatibility accessor: first agent's critic input. */
    CentralCriticObservation const& LastCentralCriticObservation() const noexcept
    {
        return pendingAgentCritic_[0];
    }
    /** @brief Returns the number of shared actor batch evaluations. */
    std::uint64_t ActorEvaluationCount() const noexcept
    {
        return actorEvaluationCount_;
    }
    /** @brief Returns finite metrics from the latest completed PPO update. */
    GeneralizedTrainingDiagnostics Diagnostics() const noexcept
    {
        return diagnostics_;
    }

    /**
     * @brief Invokes a consumer once with the sole shared v10 actor instance.
     * @tparam Consumer Callable accepting a const actor reference.
     * @param consumer Actor consumer.
     */
    template<typename Consumer>
    void WithActor(Consumer&& consumer) const
    {
        std::forward<Consumer>(consumer)(actor_);
    }

    /**
     * @brief Hands the whole training state to `consumer(actor, critic, actorOptimizer,
     *        criticOptimizer, generator, updates, seed)`, read-only (checkpoint conversion).
     * @throws std::logic_error between two update boundaries (a rollout is being collected).
     */
    template<typename Consumer>
    void WithTrainingState(Consumer&& consumer) const
    {
        if (!trainer_.AtUpdateBoundary())
            throw std::logic_error("Generalized MAPPO training state is read at an update boundary only");
        std::forward<Consumer>(consumer)(actor_, critic_, actorOptimizer_, criticOptimizer_, rng_,
                                         completedUpdates_, seed_);
    }

    /** @brief Saves the inference actor atomically as a PPOKit policy artifact. */
    void SaveActor(std::filesystem::path const& checkpoint) const
    {
        SaveGeneralizedPolicyArtifact<Critic>(checkpoint, actor_, Metadata(), completedUpdates_, seed_);
    }

    /**
     * @brief Loads an actor and resets its Adam moments.
     * @param checkpoint A PPOKit policy or training artifact of these networks, or a former
     *        V10 actor file (whose own header is checked strictly).
     */
    void LoadActor(std::filesystem::path const& checkpoint)
    {
        RequireUpdateBoundary("load a generalized actor");
        if (IsPpokitArtifact(checkpoint))
        {
            (void)LoadGeneralizedPolicyArtifact<Critic>(checkpoint, actor_);
        }
        else
        {
            LoadV10Actor(checkpoint);
        }
        rlt::reset_optimizer_state(device_, actorOptimizer_, actor_);
    }

    /** @brief Saves the V10 actor of the former format (conversion checks only). */
    void SaveV10Actor(std::filesystem::path const& checkpoint) const
    {
        SaveGeneralizedPolicy(checkpoint, actor_, Metadata());
    }

    /**
     * @brief Strictly migrates only the compatible actor state from v4.
     *
     * Actor parameters and their matching Adam first/second moments are
     * copied. The fixed-2v2 critic, both optimizer scalar states, source RNG,
     * source update count and rollout are deliberately discarded. The new
     * critic and RNG are reproducibly restarted from @p restartSeed.
     *
     * @param checkpoint Historical BLUE_PATROL_TRAINING_V4 file.
     * @param restartSeed Seed for the fresh generalized critic and RNG.
     * @return Explicit migration/reset report for logs and user interfaces.
     */
    GeneralizedTrainingMigrationReport MigrateV4TrainingState(
        std::filesystem::path const& checkpoint,
        std::uint32_t const restartSeed)
    {
        RequireUpdateBoundary("migrate v4 training state");
        LegacyActor legacyActor{};
        LegacyCritic legacyCritic{};
        ActorOptimizer legacyActorOptimizer{};
        CriticOptimizer legacyCriticOptimizer{};
        auto legacyRng = MakePatrolEngine(0U);
        bool legacyActorAllocated = false;
        bool legacyCriticAllocated = false;
        GeneralizedTrainingMigrationReport report{};
        try
        {
            rlt::malloc(device_, legacyActor);
            legacyActorAllocated = true;
            rlt::malloc(device_, legacyCritic);
            legacyCriticAllocated = true;
            std::ifstream stream{checkpoint};
            if (!stream)
                throw std::runtime_error(
                    "Cannot open legacy blue training checkpoint: "
                    + checkpoint.string());
            (void)ReadPatrolTrainingState(
                stream, legacyActor, legacyCritic, legacyActorOptimizer,
                legacyCriticOptimizer, legacyRng);

            report = MigrateV7TrainingActor(legacyActor, actor_);
            actorOptimizer_ = ActorOptimizer{};
            actorOptimizer_.parameters.alpha = 3.e-4F;
            criticOptimizer_ = CriticOptimizer{};
            criticOptimizer_.parameters.alpha = 3.e-4F;
            rng_ = MakePatrolEngine(restartSeed);
            rlt::init_weights(device_, critic_, rng_);
            rlt::reset_optimizer_state(device_, criticOptimizer_, critic_);
            completedUpdates_ = 0U;
            totalActionsSampled_ = 0U;
            totalTransitionsRecorded_ = 0U;
            actorEvaluationCount_ = 0U;
            lastTeamActionCount_ = 0U;
            lastTeamTransitionCount_ = 0U;
            pendingAvailability_ = {};
            lastMasks_ = {};
            pendingAgentCritic_ = {};
            diagnostics_ = {};
        }
        catch (...)
        {
            if (legacyCriticAllocated)
                rlt::free(device_, legacyCritic);
            if (legacyActorAllocated)
                rlt::free(device_, legacyActor);
            throw;
        }
        rlt::free(device_, legacyCritic);
        rlt::free(device_, legacyActor);
        return report;
    }

    /** @brief Saves the complete update-boundary training state as a PPOKit training artifact. */
    void SaveTrainingState(std::filesystem::path const& checkpoint) const
    {
        RequireUpdateBoundary("save generalized training state");
        SaveGeneralizedTrainingArtifact(checkpoint, actor_, critic_, actorOptimizer_, criticOptimizer_, rng_,
            completedUpdates_, Metadata(), seed_);
    }

    /** @brief Saves the V7 training state of the former format (conversion checks only). */
    void SaveV7TrainingState(std::filesystem::path const& checkpoint) const
    {
        RequireUpdateBoundary("save generalized training state");
        SaveGeneralizedTrainingState(
            checkpoint, actor_, critic_, actorOptimizer_, criticOptimizer_,
            rng_, completedUpdates_, Metadata());
    }

    /**
     * @brief Restores an update-boundary model across valid P/N counts.
     * @param checkpoint A PPOKit training artifact of these networks, or a former V7
     *        training file (whose own header is checked strictly).
     */
    void LoadTrainingState(std::filesystem::path const& checkpoint)
    {
        RequireUpdateBoundary("load generalized training state");
        GeneralizedCheckpointMetadata metadata{};
        std::uint64_t const updates = IsPpokitArtifact(checkpoint)
            ? LoadGeneralizedTrainingArtifact(checkpoint, actor_, critic_, actorOptimizer_, criticOptimizer_, rng_,
                  metadata, &seed_)
            : LoadV7TrainingState(checkpoint, metadata);
        // P/N are properties of the saved episode, not model dimensions.
        // A new rollout is constructed with the current P/N and fresh masks.
        // Shared actor/critic weights and Adam moments remain compatible.
        completedUpdates_ = updates;
        totalActionsSampled_ = 0U;
        totalTransitionsRecorded_ = 0U;
        actorEvaluationCount_ = 0U;
        lastTeamActionCount_ = 0U;
        lastTeamTransitionCount_ = 0U;
        lastMasks_ = {};
        pendingAgentCritic_ = {};
        diagnostics_ = {};
    }

private:
    /** @brief Reads a former V10 actor file (strict header with guidance semantics). */
    void LoadV10Actor(std::filesystem::path const& checkpoint)
    {
        std::ifstream stream{checkpoint};
        if (!stream)
            throw std::runtime_error(
                "Cannot open generalized blue actor checkpoint: "
                + checkpoint.string());
        (void)ReadGeneralizedPolicy(stream, actor_, true);
    }

    /**
     * @brief Reads a former V7 training file.
     * @param metadata Receives the P/N it was saved with.
     * @return Its completed PPO updates.
     */
    std::uint64_t LoadV7TrainingState(std::filesystem::path const& checkpoint,
        GeneralizedCheckpointMetadata& metadata)
    {
        std::ifstream stream{checkpoint};
        if (!stream)
            throw std::runtime_error(
                "Cannot open generalized blue training checkpoint: "
                + checkpoint.string());
        return ReadGeneralizedTrainingState(
            stream, actor_, critic_, actorOptimizer_, criticOptimizer_, rng_,
            metadata, true);
    }

    static constexpr Index kTrainingBatch = TrainingParameters::BATCH_SIZE;
    using ActorSpecification = TrainingConfig::NN::ACTOR_SPEC;
    static_assert(ActorSpecification::BATCH_SIZE >= kTrainingBatch,
                  "RLtools actor batch capacity is too small");
    using Actor = GeneralizedAttentionActor<ActorSpecification>;
    using LegacyActor = TrainingConfig::NN::ACTOR_TYPE;
    using Critic = GeneralizedCriticConfig::Network;
    using LegacyCritic = TrainingConfig::NN::CRITIC_TYPE;
    using ActorOptimizer = TrainingConfig::NN::ACTOR_OPTIMIZER;
    using CriticOptimizer = TrainingConfig::NN::CRITIC_OPTIMIZER;

    using InferenceRawSpecification = rlt::matrix::Specification<
        float, Index, static_cast<Index>(TeamCapacity::MaximumBlueMobiles),
        static_cast<Index>(GeneralizedObservationLayout::Size)>;
    using InferenceCompactSpecification = rlt::matrix::Specification<
        float, Index, static_cast<Index>(TeamCapacity::MaximumBlueMobiles),
        static_cast<Index>(ObservationLayout::Size)>;
    using InferenceOutputSpecification = rlt::matrix::Specification<
        float, Index, static_cast<Index>(TeamCapacity::MaximumBlueMobiles),
        static_cast<Index>(GeneralizedObservationLayout::ActionSize)>;
    using TrainingRawSpecification = rlt::matrix::Specification<
        float, Index, kTrainingBatch,
        static_cast<Index>(GeneralizedObservationLayout::Size)>;
    using TrainingCriticSpecification = rlt::matrix::Specification<
        float, Index, kTrainingBatch,
        static_cast<Index>(GeneralizedCriticLayout::Size)>;
    using TrainingActionSpecification = rlt::matrix::Specification<
        float, Index, kTrainingBatch,
        static_cast<Index>(GeneralizedObservationLayout::ActionSize)>;
    using TrainingValueSpecification = rlt::matrix::Specification<
        float, Index, kTrainingBatch, 1>;

/**
 * @brief Team size P of the rollout, after checking the rollout capacity.
 * @throws std::invalid_argument for invalid counts or an unbounded rollout.
 */
    static std::size_t CheckedTeamSize(
        ActiveTeamCounts const counts, std::size_t const rolloutTeamSteps)
    {
        if (!Valid(counts) || rolloutTeamSteps == 0U
            || rolloutTeamSteps > 4096U
            || rolloutTeamSteps
                   > std::numeric_limits<std::size_t>::max()
                         / counts.blueMobiles)
        {
            throw std::invalid_argument(
                "Invalid generalized MAPPO rollout capacity");
        }
        return counts.blueMobiles;
    }

    /** @brief The Blue PPO hyper-parameters, for the PPOKit trainer. */
    static ppokit::runtime::TeamPpoSettings TeamSettings() noexcept
    {
        ppokit::runtime::TeamPpoSettings settings;
        settings.gamma = PPOParameters::GAMMA;
        settings.lambda = PPOParameters::LAMBDA;
        settings.clip = PPOParameters::EPSILON_CLIP;
        settings.entropyCoefficient = PPOParameters::ACTION_ENTROPY_COEFFICIENT;
        settings.epochs = static_cast<std::size_t>(PPOParameters::N_EPOCHS);
        settings.minibatch = static_cast<std::size_t>(kTrainingBatch);
        return settings;
    }

    /**
     * @brief The rl_tools actor, critic and optimizers of this learner, as PPOKit Models.
     * @details Rows are copied into the fixed rl_tools training batch (zeros for padding),
     * so a minibatch runs exactly the historical rl_tools calls in the historical order.
     */
    class TeamModels final
    {
    public:
        explicit TeamModels(GeneralizedPatrolLearner& learner) noexcept : learner_(learner) {}

        std::size_t ObservationSize() const noexcept { return ObservationWidth; }
        std::size_t CriticSize() const noexcept { return CriticWidth; }
        std::size_t ActionSize() const noexcept { return ActionWidth; }

        float LogStd(std::size_t action) const
        {
            return rlt::get(learner_.actor_.log_std.parameters, 0, static_cast<Index>(action));
        }

        /** @brief Not used: Blue hands its own means to ChooseActionsFromMeans. */
        void ActorMeans(float const* const*, std::size_t, float*)
        {
            throw std::logic_error("Blue MAPPO computes the team means in TransformObservations");
        }

        void CriticValues(float const* const* inputs, std::size_t rows, float* values)
        {
            FillCritic(inputs, rows);
            rlt::evaluate(learner_.device_, learner_.critic_, learner_.trainingCritic_,
                          learner_.trainingCriticOutput_, learner_.trainingCriticBuffer_);
            ReadValues(rows, values);
        }

        void ActorForward(float const* const* observations, std::size_t rows, float* means)
        {
            CopyRows(observations, rows, ObservationWidth, learner_.trainingRaw_);
            rlt::zero_gradient(learner_.device_, learner_.actor_);
            rlt::forward(learner_.device_, learner_.actor_, learner_.trainingRaw_, learner_.trainingActorOutput_);
            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t action = 0; action < ActionWidth; ++action)
                {
                    means[row * ActionWidth + action] = rlt::get(
                        learner_.trainingActorOutput_, static_cast<Index>(row), static_cast<Index>(action));
                }
            }
            actorRows_ = rows;
        }

        void ActorBackwardAndStep(float const* meanGradients, float const* logStdGradients)
        {
            auto& derivative = learner_.trainingActorDerivative_;
            rlt::set_all(learner_.device_, derivative, 0.F);
            for (std::size_t row = 0; row < actorRows_; ++row)
            {
                for (std::size_t action = 0; action < ActionWidth; ++action)
                {
                    rlt::set(derivative, static_cast<Index>(row), static_cast<Index>(action),
                             meanGradients[row * ActionWidth + action]);
                }
            }
            for (std::size_t action = 0; action < ActionWidth; ++action)
            {
                rlt::increment(learner_.actor_.log_std.gradient, 0, static_cast<Index>(action),
                               logStdGradients[action]);
            }
            if (learner_.probeActive_)
                learner_.actor_.graph->SetProbes(learner_.actorProbes_.get(), actorRows_);
            rlt::backward(learner_.device_, learner_.actor_, learner_.trainingRaw_, derivative,
                          learner_.trainingActorBuffer_);
            learner_.actor_.graph->SetProbes(nullptr, 0);
            rlt::step(learner_.device_, learner_.actorOptimizer_, learner_.actor_);
        }

        void CriticForward(float const* const* inputs, std::size_t rows, float* values)
        {
            FillCritic(inputs, rows);
            rlt::zero_gradient(learner_.device_, learner_.critic_);
            rlt::forward(learner_.device_, learner_.critic_, learner_.trainingCritic_,
                         learner_.trainingCriticOutput_);
            if (learner_.probeActive_)
                learner_.criticProbes_->ObserveForward(learner_.critic_, rows);
            ReadValues(rows, values);
            criticRows_ = rows;
        }

        void CriticBackwardAndStep(float const* valueGradients)
        {
            auto& derivative = learner_.trainingCriticDerivative_;
            rlt::set_all(learner_.device_, derivative, 0.F);
            for (std::size_t row = 0; row < criticRows_; ++row)
            {
                rlt::set(derivative, static_cast<Index>(row), 0, valueGradients[row]);
            }
            rlt::backward(learner_.device_, learner_.critic_, learner_.trainingCritic_, derivative,
                          learner_.trainingCriticBuffer_);
            if (learner_.probeActive_)
                learner_.criticProbes_->ObserveGradients(learner_.critic_);
            rlt::step(learner_.device_, learner_.criticOptimizer_, learner_.critic_);
        }

    private:
        static constexpr std::size_t ObservationWidth = GeneralizedObservationLayout::Size;
        static constexpr std::size_t CriticWidth = GeneralizedCriticLayout::Size;
        static constexpr std::size_t ActionWidth = GeneralizedObservationLayout::ActionSize;

        void FillCritic(float const* const* inputs, std::size_t rows)
        {
            CopyRows(inputs, rows, CriticWidth, learner_.trainingCritic_);
        }

        /** @brief Zeroes the batch matrix, then copies every non-null row into it. */
        template<typename Matrix>
        void CopyRows(float const* const* sources, std::size_t rows, std::size_t width, Matrix& matrix)
        {
            rlt::set_all(learner_.device_, matrix, 0.F);
            for (std::size_t row = 0; row < rows; ++row)
            {
                float const* const source = sources[row];
                if (source == nullptr)
                {
                    continue; // padding row
                }
                for (std::size_t feature = 0; feature < width; ++feature)
                {
                    rlt::set(matrix, static_cast<Index>(row), static_cast<Index>(feature), source[feature]);
                }
            }
        }

        void ReadValues(std::size_t rows, float* values) const
        {
            for (std::size_t row = 0; row < rows; ++row)
            {
                values[row] = rlt::get(learner_.trainingCriticOutput_, static_cast<Index>(row), 0);
            }
        }

        GeneralizedPatrolLearner& learner_;
        std::size_t actorRows_ = 0;
        std::size_t criticRows_ = 0;
    };

    /** @brief The rl_tools generator of this learner (saved in the training state), for PPOKit. */
    class TeamRandom final
    {
    public:
        explicit TeamRandom(GeneralizedPatrolLearner& learner) noexcept : learner_(learner) {}

        void Shuffle(std::vector<std::size_t>& order) { std::shuffle(order.begin(), order.end(), learner_.rng_); }

        float Normal(float mean, float deviation)
        {
            return rlt::random::normal_distribution::sample(
                typename Device::SPEC::RANDOM{}, mean, deviation, learner_.rng_);
        }

    private:
        GeneralizedPatrolLearner& learner_;
    };

    using Trainer = ppokit::runtime::HostedTrainer<TeamModels, TeamRandom>;
    using Clock = std::chrono::steady_clock;

    /** @brief Clears the probes and lets the models feed them during the update that starts. */
    void StartProbes()
    {
        if (!actorProbes_)
        {
            actorProbes_ = std::make_unique<ppokit::graph::GraphProbes>(actor_.graph->Network());
            criticProbes_ = std::make_unique<CriticProbes>();
        }
        actorProbes_->Reset();
        criticProbes_->Reset();
        probeActive_ = true;
    }

    /** @brief Keeps what the probes saw during the update that just ended. */
    void FinishProbes()
    {
        probeActive_ = false;
        lastProbes_ = {actorProbes_->Summary(), criticProbes_->Summary()};
    }

    static long long Milliseconds(Clock::duration duration) noexcept
    {
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    }

    /**
     * @brief Historical [MAPPO_TIMING] and [MAPPO_UPDATE] logs, and the diagnostics.
     * @details The first line is emitted before the synchronous update so that a stalled
     * first update cannot look like an idle evaluator in CI logs.
     */
    ppokit::runtime::HostedUpdateHooks ProgressHooks()
    {
        ppokit::runtime::HostedUpdateHooks hooks;
        hooks.began = [this](std::uint64_t, std::size_t rows) {
            updateStarted_ = Clock::now();
            diagnostics_ = {};
            std::clog << "[MAPPO_TIMING] optimizer=begin update=" << completedUpdates_ + 1U
                      << " transitions=" << rows << std::endl;
        };
        hooks.baselinesDone = [this]() {
            baselinesDone_ = Clock::now();
            std::clog << "[MAPPO_TIMING] phase=critic_baselines_done update=" << completedUpdates_ + 1U
                      << " elapsed_ms=" << Milliseconds(baselinesDone_ - updateStarted_) << std::endl;
        };
        hooks.advantagesDone = [this]() { advantagesDone_ = Clock::now(); };
        hooks.epochDone = [this](std::size_t epoch) {
            std::clog << "[MAPPO_TIMING] phase=epoch_done update=" << completedUpdates_ + 1U << " epoch=" << epoch
                      << '/' << PPOParameters::N_EPOCHS
                      << " elapsed_ms=" << Milliseconds(Clock::now() - updateStarted_) << std::endl;
        };
        hooks.finished = [this](ppokit::runtime::TeamUpdateStatistics const& statistics) {
            RecordDiagnostics(statistics);
            LogUpdate(Clock::now());
        };
        return hooks;
    }

    void RecordDiagnostics(ppokit::runtime::TeamUpdateStatistics const& statistics) noexcept
    {
        diagnostics_.learnableTransitions = statistics.learnableRows;
        diagnostics_.criticTransitions = statistics.criticRows;
        diagnostics_.generalizedAdvantageAbsMean = statistics.advantageAbsMean;
        diagnostics_.actorDerivativeL1 = statistics.actorDerivativeL1;
        diagnostics_.criticDerivativeL1 = statistics.criticDerivativeL1;
        diagnostics_.logStdGradientL1 = statistics.logStdGradientL1;
        diagnostics_.policyLoss = statistics.policyLoss;
        diagnostics_.valueLoss = statistics.valueLoss;
        diagnostics_.entropy = statistics.entropy;
        diagnostics_.approximateKl = statistics.approximateKl;
        diagnostics_.clipFraction = statistics.clipFraction;
        diagnostics_.explainedVariance = statistics.explainedVariance;
        diagnostics_.actionStdMean = statistics.actionStdMean;
    }

    void LogUpdate(Clock::time_point const ended) const
    {
        std::clog << "[MAPPO_TIMING] optimizer=done update=" << completedUpdates_ + 1U
                  << " learnable=" << diagnostics_.learnableTransitions
                  << " critic_rows=" << diagnostics_.criticTransitions
                  << " critic_baselines_ms=" << Milliseconds(baselinesDone_ - updateStarted_)
                  << " advantages_ms=" << Milliseconds(advantagesDone_ - baselinesDone_)
                  << " ppo_ms=" << Milliseconds(ended - advantagesDone_)
                  << " total_ms=" << Milliseconds(ended - updateStarted_) << std::endl;
        std::ostringstream metrics;
        metrics << std::setprecision(9)
                << "[MAPPO_UPDATE] update=" << completedUpdates_ + 1U
                << " policy_loss=" << diagnostics_.policyLoss
                << " value_loss=" << diagnostics_.valueLoss
                << " entropy=" << diagnostics_.entropy
                << " approx_kl=" << diagnostics_.approximateKl
                << " clip_fraction=" << diagnostics_.clipFraction
                << " explained_variance=" << diagnostics_.explainedVariance
                << " advantage_abs_mean=" << diagnostics_.generalizedAdvantageAbsMean
                << " actor_derivative_l1=" << diagnostics_.actorDerivativeL1
                << " critic_derivative_l1=" << diagnostics_.criticDerivativeL1
                << " log_std_gradient_l1=" << diagnostics_.logStdGradientL1
                << " action_std_mean=" << diagnostics_.actionStdMean
                << " learnable=" << diagnostics_.learnableTransitions
                << " critic_rows=" << diagnostics_.criticTransitions
                << " total_ms=" << Milliseconds(ended - updateStarted_);
        std::clog << metrics.str() << std::endl;
    }

    GeneralizedCheckpointMetadata Metadata() const noexcept
    {
        return {counts_.blueMobiles, counts_.redMobiles};
    }

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
    void ValidateBatch(Observations const& observations,
                       AvailabilityMask const& availability) const
    {
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            auto const& metadata = observations[agent].metadata;
            if (metadata.activeBlueMobiles != counts_.blueMobiles
                || metadata.activeRedMobiles != counts_.redMobiles
                || !Valid(observations[agent]))
            {
                throw std::invalid_argument(GeneralizedOptimizationGuard::Diagnostic(
                    __FILE__, __LINE__, "ValidateBatch",
                    "invalid observation for agent=" + std::to_string(agent)));
            }
        }
        for (std::size_t agent = counts_.blueMobiles;
             agent < TeamCapacity::MaximumBlueMobiles; ++agent)
        {
            if (availability[agent])
                throw std::invalid_argument(
                    "Absent generalized blue slot cannot be available");
        }
    }

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
    void RequireUpdateBoundary(char const* const operation) const
    {
        if (!trainer_.AtUpdateBoundary())
            throw std::logic_error(std::string{"Cannot "} + operation
                                   + " during a rollout");
    }

/**
 * @brief Runs the actor on the team: action means in inferenceOutput_, and the compact v7
 *        rows in inferenceCompact_, from which the centralized critic input is built.
 */
    void TransformObservations(Observations const& observations)
    {
        rlt::set_all(device_, inferenceRaw_, 0.F);
        for (Index agent = 0;
             agent < static_cast<Index>(counts_.blueMobiles); ++agent)
        {
            for (Index feature = 0;
                 feature < static_cast<Index>(
                               GeneralizedObservationLayout::Size);
                 ++feature)
            {
                rlt::set(inferenceRaw_, agent, feature,
                         observations[agent].values[feature]);
            }
        }
        rlt::blue_generalized_attention_detail::evaluate_with_compact(
            device_, actor_, inferenceRaw_, inferenceOutput_, inferenceCompact_);
    }

/**
 * @brief Construit, transforme ou échantillonne les données produites par BuildAgentCriticObservations.
 * @details Applique des bornes numériques.
 */
    void BuildAgentCriticObservations(
        AvailabilityMask const& availability,
        std::array<CentralCriticObservation,
                   TeamCapacity::MaximumBlueMobiles>& outputs) const
    {
        outputs = {};
        CentralCriticObservation mean{};
        CentralCriticObservation maximum{};
        maximum.fill(-std::numeric_limits<float>::infinity());
        std::size_t living = 0U;
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            if (!availability[agent])
                continue;
            ++living;
            for (std::size_t feature = 0U;
                 feature < GeneralizedCriticLayout::Compact; ++feature)
            {
                float const value = rlt::get(
                    inferenceCompact_, static_cast<Index>(agent),
                    static_cast<Index>(feature));
                mean[feature] += value;
                maximum[feature] = std::max(maximum[feature], value);
            }
        }
        float const inverseLiving = living == 0U ? 0.F
            : 1.F / static_cast<float>(living);
        float const livingFraction = static_cast<float>(living)
            / static_cast<float>(TeamCapacity::MaximumBlueMobiles);
        float const teamFraction = static_cast<float>(counts_.blueMobiles)
            / static_cast<float>(TeamCapacity::MaximumBlueMobiles);
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            auto& result = outputs[agent];
            for (std::size_t feature = 0U;
                 feature < GeneralizedCriticLayout::Compact; ++feature)
            {
                // Zero own block represents death; pooled living context is
                // still required to predict future cooperative outcomes.
                if (availability[agent])
                    result[GeneralizedCriticLayout::OwnOffset + feature] = rlt::get(
                        inferenceCompact_, static_cast<Index>(agent),
                        static_cast<Index>(feature));
                result[GeneralizedCriticLayout::MeanOffset + feature] =
                    mean[feature] * inverseLiving;
                result[GeneralizedCriticLayout::MaxOffset + feature] =
                    living == 0U ? 0.F : maximum[feature];
            }
            result[GeneralizedCriticLayout::LivingFractionOffset] =
                livingFraction;
            result[GeneralizedCriticLayout::TeamFractionOffset] =
                teamFraction;
        }
    }
/**
 * @brief Réalise l'opération Allocate; voir les préconditions et effets dans son corps C++.
 */
    void Allocate()
    {
        rlt::malloc(device_, actor_);
        actorAllocated_ = true;
        rlt::malloc(device_, critic_);
        criticAllocated_ = true;
        rlt::malloc(device_, inferenceRaw_);
        inferenceRawAllocated_ = true;
        rlt::malloc(device_, inferenceCompact_);
        inferenceCompactAllocated_ = true;
        rlt::malloc(device_, inferenceOutput_);
        inferenceOutputAllocated_ = true;
        rlt::malloc(device_, trainingRaw_);
        trainingRawAllocated_ = true;
        rlt::malloc(device_, trainingCritic_);
        trainingCriticAllocated_ = true;
        rlt::malloc(device_, trainingActorOutput_);
        trainingActorOutputAllocated_ = true;
        rlt::malloc(device_, trainingActorDerivative_);
        trainingActorDerivativeAllocated_ = true;
        rlt::malloc(device_, trainingCriticOutput_);
        trainingCriticOutputAllocated_ = true;
        rlt::malloc(device_, trainingCriticDerivative_);
        trainingCriticDerivativeAllocated_ = true;
        rlt::malloc(device_, trainingActorBuffer_);
        trainingActorBufferAllocated_ = true;
        rlt::malloc(device_, trainingCriticBuffer_);
        trainingCriticBufferAllocated_ = true;
    }

    void Release() noexcept
    {
        if (trainingCriticBufferAllocated_)
            rlt::free(device_, trainingCriticBuffer_);
        if (trainingActorBufferAllocated_)
            rlt::free(device_, trainingActorBuffer_);
        if (trainingCriticDerivativeAllocated_)
            rlt::free(device_, trainingCriticDerivative_);
        if (trainingCriticOutputAllocated_)
            rlt::free(device_, trainingCriticOutput_);
        if (trainingActorDerivativeAllocated_)
            rlt::free(device_, trainingActorDerivative_);
        if (trainingActorOutputAllocated_)
            rlt::free(device_, trainingActorOutput_);
        if (trainingCriticAllocated_)
            rlt::free(device_, trainingCritic_);
        if (trainingRawAllocated_)
            rlt::free(device_, trainingRaw_);
        if (inferenceOutputAllocated_)
            rlt::free(device_, inferenceOutput_);
        if (inferenceCompactAllocated_)
            rlt::free(device_, inferenceCompact_);
        if (inferenceRawAllocated_)
            rlt::free(device_, inferenceRaw_);
        if (criticAllocated_)
            rlt::free(device_, critic_);
        if (actorAllocated_)
            rlt::free(device_, actor_);
    }

    ActiveTeamCounts counts_{};
    std::size_t rolloutTeamSteps_{};
    /** @brief Construction seed, recorded in the checkpoint headers. */
    std::uint32_t seed_{};
    GeneralizedOptimizationGuard optimizationGuard_{};
    Device device_{};
    Engine rng_{};
    Actor actor_{};
    Critic critic_{};
    ActorOptimizer actorOptimizer_{};
    CriticOptimizer criticOptimizer_{};

    rlt::MatrixDynamic<InferenceRawSpecification> inferenceRaw_{};
    rlt::MatrixDynamic<InferenceCompactSpecification> inferenceCompact_{};
    rlt::MatrixDynamic<InferenceOutputSpecification> inferenceOutput_{};
    rlt::MatrixDynamic<TrainingRawSpecification> trainingRaw_{};
    rlt::MatrixDynamic<TrainingCriticSpecification> trainingCritic_{};
    rlt::MatrixDynamic<TrainingActionSpecification> trainingActorOutput_{};
    rlt::MatrixDynamic<TrainingActionSpecification>
        trainingActorDerivative_{};
    rlt::MatrixDynamic<TrainingValueSpecification> trainingCriticOutput_{};
    rlt::MatrixDynamic<TrainingValueSpecification>
        trainingCriticDerivative_{};
    typename Actor::template Buffer<kTrainingBatch> trainingActorBuffer_{};
    typename Critic::template Buffer<kTrainingBatch> trainingCriticBuffer_{};

    AvailabilityMask pendingAvailability_{};
    AvailabilityMask lastMasks_{};
    std::array<CentralCriticObservation,
               TeamCapacity::MaximumBlueMobiles> pendingAgentCritic_{};
    GeneralizedTrainingDiagnostics diagnostics_{};
    /** @brief Team rows handed to the trainer and its sampled actions (no allocation per team step). */
    std::array<float const*, TeamCapacity::MaximumBlueMobiles> teamObservations_{};
    std::array<float const*, TeamCapacity::MaximumBlueMobiles> teamCritics_{};
    std::array<ppokit::runtime::AgentPresence, TeamCapacity::MaximumBlueMobiles> teamPresence_{};
    std::array<float, TeamCapacity::MaximumBlueMobiles * GeneralizedObservationLayout::ActionSize> teamMeans_{};
    std::array<float, TeamCapacity::MaximumBlueMobiles * GeneralizedObservationLayout::ActionSize> teamActions_{};
    Clock::time_point updateStarted_{};
    Clock::time_point baselinesDone_{};
    Clock::time_point advantagesDone_{};
    std::size_t lastTeamActionCount_{};
    std::size_t lastTeamTransitionCount_{};
    std::uint64_t completedUpdates_{};
    std::uint64_t totalActionsSampled_{};
    std::uint64_t totalTransitionsRecorded_{};
    std::uint64_t actorEvaluationCount_{};

    bool actorAllocated_{};
    bool criticAllocated_{};
    bool inferenceRawAllocated_{};
    bool inferenceCompactAllocated_{};
    bool inferenceOutputAllocated_{};
    bool trainingRawAllocated_{};
    bool trainingCriticAllocated_{};
    bool trainingActorOutputAllocated_{};
    bool trainingActorDerivativeAllocated_{};
    bool trainingCriticOutputAllocated_{};
    bool trainingCriticDerivativeAllocated_{};
    bool trainingActorBufferAllocated_{};
    bool trainingCriticBufferAllocated_{};

    /** @brief Layer probes, created at the first probed update. */
    using CriticProbes = ppokit::backend::rltools::MlpProbes<Critic>;
    std::unique_ptr<ppokit::graph::GraphProbes> actorProbes_;
    std::unique_ptr<CriticProbes> criticProbes_;
    GeneralizedLayerProbes lastProbes_{};
    bool probing_{};
    /** @brief True during a probed update: the models feed the probes. */
    bool probeActive_{};

    // Declared last: the trainer refers to the models and the generator, which refer
    // to the networks and buffers above.
    TeamModels models_{*this};
    TeamRandom random_{*this};
    Trainer trainer_;
};

} // namespace ppo::examples::bluedefense::defenders
