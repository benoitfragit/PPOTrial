#pragma once

#include "generalized_training_checkpoint.hpp"
#include "generalized_critic_network.hpp"

#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppo::examples::twovstwo::defenders {

/** @brief Finite diagnostics from the latest generalized PPO update. */
struct GeneralizedTrainingDiagnostics final
{
    /** @brief Number of non-destroyed transition rows used for learning. */
    std::size_t learnableTransitions{};
    /** @brief Mean absolute GAE value before normalization. */
    float generalizedAdvantageAbsMean{};
    /** @brief Sum of absolute actor-output loss derivatives. */
    float actorDerivativeL1{};
    /** @brief Sum of absolute centralized-value loss derivatives. */
    float criticDerivativeL1{};
    /** @brief Sum of absolute log-standard-deviation gradients. */
    float logStdGradientL1{};
};

/**
 * @brief One shared-actor MAPPO rollout contract for a runtime P/N blue team.
 *
 * A call to ChooseActions evaluates one actor batch containing exactly the P
 * episode mobiles. The matching Observe call then appends exactly P rows with
 * one common team-step index. Destroyed mobiles keep their stable row but have
 * a zero learning mask. Storage and neural-network workspaces are allocated by
 * the constructor, never by the per-team-step success path.
 *
 * The historical two-agent PatrolLearner remains unchanged. This class owns a
 * distinct v8 actor, agent-conditioned centralized critic and v6 training state.
 */
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
        : counts_(counts), rolloutTeamSteps_(rolloutTeamSteps),
          transitions_(CheckedTransitionCapacity(counts, rolloutTeamSteps)),
          trainingOrder_(transitions_.size()),
          rng_(rlt::random::default_engine(
              typename Device::SPEC::RANDOM{}, seed))
    {
        if (!Valid(counts_))
            throw std::invalid_argument("Invalid generalized P/N team counts");

        std::iota(trainingOrder_.begin(), trainingOrder_.end(), 0U);

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
        if (pending_)
            throw std::logic_error(
                "Generalized MAPPO: synchronized feedback is missing");
        ValidateBatch(observations, availability);

        TransformObservations(observations);
        using ActorMlp = rlt::nn_models::mlp::NeuralNetwork<ActorSpecification>;
        rlt::evaluate(device_, static_cast<ActorMlp const&>(actor_),
                      inferenceCompact_, inferenceOutput_,
                      inferenceActorBuffer_);
        ++actorEvaluationCount_;

        BuildAgentCriticObservations(availability, pendingAgentCritic_);

        Actions actions{};
        constexpr float kTwoPi = 6.28318530717958647692F;
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            auto& transition = transitions_[TransitionIndex(
                collectedTeamSteps_, agent)];
            transition.observation = observations[agent].values;
            transition.centralCritic = pendingAgentCritic_[agent];
            transition.available = availability[agent];
            transition.teamStep = collectedTeamSteps_;
            transition.oldLogProbability = 0.F;
            for (std::size_t action = 0U;
                 action < GeneralizedObservationLayout::ActionSize; ++action)
            {
                float const mean = rlt::get(
                    inferenceOutput_, static_cast<Index>(agent),
                    static_cast<Index>(action));
                float const logStd = rlt::get(
                    actor_.log_std.parameters, 0,
                    static_cast<Index>(action));
                float const deviation = std::exp(logStd);
                if (!std::isfinite(mean) || !std::isfinite(deviation)
                    || deviation <= 0.F)
                {
                    throw std::runtime_error(
                        "Invalid generalized MAPPO Gaussian distribution");
                }
                float const sampled =
                    rlt::random::normal_distribution::sample(
                        typename Device::SPEC::RANDOM{}, mean, deviation,
                        rng_);
                if (!std::isfinite(sampled))
                    throw std::runtime_error(
                        "Nonfinite generalized MAPPO action");
                float const normalized = (sampled - mean) / deviation;
                transition.oldLogProbability +=
                    -0.5F * normalized * normalized - logStd
                    - 0.5F * std::log(kTwoPi);
                actions[agent][action] = sampled;
                transition.action[action] = sampled;
            }
        }

        pendingAvailability_ = availability;
        pending_ = true;
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
        if (pending_ || collectedTeamSteps_ != 0U)
            throw std::logic_error(
                "Generalized MAPPO evaluation requires an update boundary");
        ValidateBatch(observations, availability);
        TransformObservations(observations);
        using ActorMlp = rlt::nn_models::mlp::NeuralNetwork<ActorSpecification>;
        rlt::evaluate(device_, static_cast<ActorMlp const&>(actor_),
                      inferenceCompact_, inferenceOutput_,
                      inferenceActorBuffer_);
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
        if (!pending_)
            throw std::logic_error(
                "Generalized MAPPO: actions are required before feedback");
        ValidateBatch(nextObservations, nextAvailability);

        TransformObservations(nextObservations);
        std::array<CentralCriticObservation,
                   TeamCapacity::MaximumBlueMobiles> nextCritics{};
        BuildAgentCriticObservations(nextAvailability, nextCritics);

        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            if (!std::isfinite(rewards[agent]))
                throw std::invalid_argument(
                    "Nonfinite generalized MAPPO reward");
            auto& transition = transitions_[TransitionIndex(
                collectedTeamSteps_, agent)];
            transition.nextCentralCritic = nextCritics[agent];
            transition.reward = rewards[agent];
            transition.nextAvailable = nextAvailability[agent];
            transition.terminated = terminated;
            transition.truncated = truncated;
        }

        pending_ = false;
        lastMasks_ = pendingAvailability_;
        lastTeamTransitionCount_ = counts_.blueMobiles;
        totalTransitionsRecorded_ += counts_.blueMobiles;
        ++collectedTeamSteps_;
        if (collectedTeamSteps_ != rolloutTeamSteps_)
            return false;

        OptimizeRollout();
        collectedTeamSteps_ = 0U;
        ++completedUpdates_;
        return true;
    }

    /** @brief Returns immutable episode P/N counts. */
    ActiveTeamCounts ActiveCounts() const noexcept { return counts_; }
    /** @brief Returns synchronized team steps currently buffered. */
    std::size_t CollectedTeamSteps() const noexcept
    {
        return collectedTeamSteps_;
    }
    /** @brief Returns synchronized team steps per update. */
    std::size_t RolloutSize() const noexcept { return rolloutTeamSteps_; }
    /** @brief Returns whether Observe still owes feedback for one actor batch. */
    bool HasPendingAction() const noexcept { return pending_; }

    /**
     * @brief Discards an action batch that cannot receive valid feedback.
     *
     * Collected transitions and optimizer state are preserved. This recovery
     * path is reserved for a rejected ECS step so the next episode can resume
     * the partial rollout instead of leaving the learner permanently pending.
     */
    void AbortPendingStep() noexcept
    {
        pending_ = false;
        pendingAvailability_.fill(false);
    }

    /** @brief Returns completed generalized policy updates. */
    std::uint64_t StepCount() const noexcept { return completedUpdates_; }
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
        return pending_ ? pendingAvailability_ : lastMasks_;
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
     * @brief Invokes a consumer once with the sole shared v8 actor instance.
     * @tparam Consumer Callable accepting a const actor reference.
     * @param consumer Actor consumer.
     */
    template<typename Consumer>
    void WithActor(Consumer&& consumer) const
    {
        std::forward<Consumer>(consumer)(actor_);
    }

    /** @brief Saves the inference-only v8 actor atomically. */
    void SaveActor(std::filesystem::path const& checkpoint) const
    {
        SaveGeneralizedPolicy(checkpoint, actor_, Metadata());
    }

    /** @brief Loads a strict v8 actor and resets its Adam moments. */
    void LoadActor(std::filesystem::path const& checkpoint)
    {
        RequireUpdateBoundary("load a generalized actor");
        std::ifstream stream{checkpoint};
        if (!stream)
            throw std::runtime_error(
                "Cannot open generalized blue actor checkpoint: "
                + checkpoint.string());
        (void)ReadGeneralizedPolicy(stream, actor_);
        rlt::reset_optimizer_state(device_, actorOptimizer_, actor_);
    }

    /**
     * @brief Migrates a historical v7 inference actor into the v8 actor.
     * @param checkpoint Strict historical v7 actor checkpoint.
     */
    void LoadV7Actor(std::filesystem::path const& checkpoint)
    {
        RequireUpdateBoundary("migrate a v7 actor");
        LegacyActor legacy{};
        rlt::malloc(device_, legacy);
        try
        {
            rlt::reset_optimizer_state(device_, actorOptimizer_, legacy);
            std::ifstream stream{checkpoint};
            if (!stream)
                throw std::runtime_error(
                    "Cannot open legacy blue actor checkpoint: "
                    + checkpoint.string());
            ReadPatrolPolicy(stream, legacy);
            MigrateV7Actor(legacy, actor_);
            rlt::reset_optimizer_state(device_, actorOptimizer_, actor_);
        }
        catch (...)
        {
            rlt::free(device_, legacy);
            throw;
        }
        rlt::free(device_, legacy);
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
        auto legacyRng = rlt::random::default_engine(
            typename Device::SPEC::RANDOM{}, 0U);
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
            rng_ = rlt::random::default_engine(
                typename Device::SPEC::RANDOM{}, restartSeed);
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

    /**
     * @brief Migrates only the shared v8 actor from a historical v5 trainer.
     *        The incompatible old critic/Adam state is deliberately reset.
     */
    void MigrateV5TrainingActor(std::filesystem::path const& checkpoint)
    {
        RequireUpdateBoundary("migrate v5 actor");
        std::ifstream stream{checkpoint};
        if (!stream)
            throw std::runtime_error("Cannot open old generalized checkpoint");
        std::string magic;
        unsigned version{}, observationVersion{};
        std::size_t raw{}, compact{}, output{}, actorHidden{}, actorLayers{};
        std::size_t criticInput{}, criticHidden{}, criticLayers{};
        if (!(stream >> magic >> version >> observationVersion
              >> raw >> compact >> output >> actorHidden >> actorLayers
              >> criticInput >> criticHidden >> criticLayers)
            || magic != "BLUE_TEAM_TRAINING_V5" || version != 5U
            || observationVersion != GeneralizedObservationLayout::Version
            || raw != GeneralizedObservationLayout::Size
            || compact != ObservationLayout::Size
            || output != GeneralizedObservationLayout::ActionSize
            || actorHidden != TrainingParameters::ACTOR_HIDDEN_DIM
            || actorLayers != TrainingParameters::ACTOR_NUM_LAYERS
            || criticInput != ObservationLayout::Size
            || criticHidden != TrainingParameters::CRITIC_HIDDEN_DIM
            || criticLayers != TrainingParameters::CRITIC_NUM_LAYERS)
        {
            throw std::runtime_error("Incompatible v5 generalized checkpoint");
        }
        auto const metadata = generalized_checkpoint_detail::ReadMetadata(stream);
        training_checkpoint_detail::RequirePatrolCheckpointToken(
            stream, "ACTOR");
        auto const actorMetadata = ReadGeneralizedPolicyParameters(stream, actor_);
        if (actorMetadata.activeBlueMobiles != metadata.activeBlueMobiles
            || actorMetadata.activeRedMobiles != metadata.activeRedMobiles)
        {
            throw std::runtime_error("Inconsistent V5 actor metadata");
        }
        rlt::reset_optimizer_state(device_, actorOptimizer_, actor_);
        rlt::reset_optimizer_state(device_, criticOptimizer_, critic_);
        completedUpdates_ = 0U;
        totalActionsSampled_ = 0U;
        totalTransitionsRecorded_ = 0U;
        pendingAgentCritic_ = {};
        diagnostics_ = {};
    }

    /** @brief Saves the complete update-boundary v6 training state. */
    void SaveTrainingState(std::filesystem::path const& checkpoint) const
    {
        RequireUpdateBoundary("save generalized training state");
        SaveGeneralizedTrainingState(
            checkpoint, actor_, critic_, actorOptimizer_, criticOptimizer_,
            rng_, completedUpdates_, Metadata());
    }

    /** @brief Restores a v6 update-boundary model across valid P/N counts. */
    void LoadTrainingState(std::filesystem::path const& checkpoint)
    {
        RequireUpdateBoundary("load generalized training state");
        std::ifstream stream{checkpoint};
        if (!stream)
            throw std::runtime_error(
                "Cannot open generalized blue training checkpoint: "
                + checkpoint.string());
        GeneralizedCheckpointMetadata metadata{};
        std::uint64_t const updates = ReadGeneralizedTrainingState(
            stream, actor_, critic_, actorOptimizer_, criticOptimizer_, rng_,
            metadata);
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
    static constexpr Index kTrainingBatch = 128;
    using ActorSpecification = TrainingConfig::NN::ACTOR_SPEC;
    using Actor = GeneralizedAttentionActor<ActorSpecification>;
    using LegacyActor = TrainingConfig::NN::ACTOR_TYPE;
    using Critic = GeneralizedCriticConfig::Network;
    using LegacyCritic = TrainingConfig::NN::CRITIC_TYPE;
    using ActorOptimizer = TrainingConfig::NN::ACTOR_OPTIMIZER;
    using CriticOptimizer = TrainingConfig::NN::CRITIC_OPTIMIZER;

    struct Transition final
    {
        GeneralizedRawObservation observation{};
        CentralCriticObservation centralCritic{};
        CentralCriticObservation nextCentralCritic{};
        Action action{};
        float oldLogProbability{};
        float reward{};
        float value{};
        float nextValue{};
        float returnValue{};
        float advantage{};
        std::size_t teamStep{};
        bool available{};
        bool nextAvailable{};
        bool terminated{};
        bool truncated{};
    };

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

    static std::size_t CheckedTransitionCapacity(
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
        return rolloutTeamSteps * counts.blueMobiles;
    }

    std::size_t TransitionIndex(std::size_t const teamStep,
                                std::size_t const agent) const noexcept
    {
        return teamStep * counts_.blueMobiles + agent;
    }

    GeneralizedCheckpointMetadata Metadata() const noexcept
    {
        return {counts_.blueMobiles, counts_.redMobiles};
    }

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
                throw std::invalid_argument(
                    "Invalid generalized observation batch");
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

    void RequireUpdateBoundary(char const* const operation) const
    {
        if (pending_ || collectedTeamSteps_ != 0U)
            throw std::logic_error(std::string{"Cannot "} + operation
                                   + " during a rollout");
    }

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
        rlt::blue_generalized_attention_detail::transform(
            device_, actor_, inferenceRaw_, inferenceCompact_);
    }

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
        if (living == 0U)
            return;
        float const inverseLiving = 1.F / static_cast<float>(living);
        float const livingFraction = static_cast<float>(living)
            / static_cast<float>(TeamCapacity::MaximumBlueMobiles);
        float const teamFraction = static_cast<float>(counts_.blueMobiles)
            / static_cast<float>(TeamCapacity::MaximumBlueMobiles);
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            if (!availability[agent])
                continue;
            auto& result = outputs[agent];
            for (std::size_t feature = 0U;
                 feature < GeneralizedCriticLayout::Compact; ++feature)
            {
                result[GeneralizedCriticLayout::OwnOffset + feature] = rlt::get(
                    inferenceCompact_, static_cast<Index>(agent),
                    static_cast<Index>(feature));
                result[GeneralizedCriticLayout::MeanOffset + feature] =
                    mean[feature] * inverseLiving;
                result[GeneralizedCriticLayout::MaxOffset + feature] =
                    maximum[feature];
            }
            result[GeneralizedCriticLayout::LivingFractionOffset] =
                livingFraction;
            result[GeneralizedCriticLayout::TeamFractionOffset] =
                teamFraction;
        }
    }

    void OptimizeRollout()
    {
        diagnostics_ = {};
        EvaluateCriticBaselines();
        std::size_t learnable = EstimateGeneralizedAdvantages();
        diagnostics_.learnableTransitions = learnable;
        if (learnable == 0U)
            return;
        NormalizeAdvantages(learnable);

        for (Index epoch = 0; epoch < PPOParameters::N_EPOCHS; ++epoch)
        {
            std::shuffle(trainingOrder_.begin(), trainingOrder_.end(), rng_);
            for (std::size_t offset = 0U; offset < transitions_.size();
                 offset += static_cast<std::size_t>(kTrainingBatch))
            {
                FillTrainingBatch(offset, false);
                std::size_t const batchLearnable =
                    CountLearnableBatch(offset);
                if (batchLearnable == 0U)
                    continue;
                TrainActorBatch(offset, batchLearnable);
                TrainCriticBatch(offset, batchLearnable);
            }
        }
    }

    Transition* BatchTransition(std::size_t const offset,
                                Index const row) noexcept
    {
        std::size_t const orderIndex =
            offset + static_cast<std::size_t>(row);
        if (orderIndex >= trainingOrder_.size())
            return nullptr;
        return &transitions_[trainingOrder_[orderIndex]];
    }

    void FillTrainingBatch(std::size_t const offset,
                           bool const useNextCritic)
    {
        rlt::set_all(device_, trainingRaw_, 0.F);
        rlt::set_all(device_, trainingCritic_, 0.F);
        for (Index row = 0; row < kTrainingBatch; ++row)
        {
            Transition const* const transition = BatchTransition(offset, row);
            if (transition == nullptr)
                continue;
            for (Index feature = 0;
                 feature < static_cast<Index>(
                               GeneralizedObservationLayout::Size);
                 ++feature)
            {
                rlt::set(trainingRaw_, row, feature,
                         transition->observation[feature]);
            }
            for (Index feature = 0;
                 feature < static_cast<Index>(GeneralizedCriticLayout::Size);
                 ++feature)
            {
                rlt::set(trainingCritic_, row, feature,
                         useNextCritic
                             ? transition->nextCentralCritic[feature]
                             : transition->centralCritic[feature]);
            }
        }
    }

    void EvaluateCriticBaselines()
    {
        std::iota(trainingOrder_.begin(), trainingOrder_.end(), 0U);
        for (std::size_t offset = 0U; offset < transitions_.size();
             offset += static_cast<std::size_t>(kTrainingBatch))
        {
            FillTrainingBatch(offset, false);
            rlt::evaluate(device_, critic_, trainingCritic_,
                          trainingCriticOutput_, trainingCriticBuffer_);
            for (Index row = 0; row < kTrainingBatch; ++row)
            {
                Transition* const transition = BatchTransition(offset, row);
                if (transition != nullptr)
                    transition->value = rlt::get(
                        trainingCriticOutput_, row, 0);
            }

            FillTrainingBatch(offset, true);
            rlt::evaluate(device_, critic_, trainingCritic_,
                          trainingCriticOutput_, trainingCriticBuffer_);
            for (Index row = 0; row < kTrainingBatch; ++row)
            {
                Transition* const transition = BatchTransition(offset, row);
                if (transition != nullptr)
                    transition->nextValue = rlt::get(
                        trainingCriticOutput_, row, 0);
            }
        }
    }

    std::size_t EstimateGeneralizedAdvantages() noexcept
    {
        std::size_t learnable = 0U;
        float absoluteAdvantage = 0.F;
        for (std::size_t agent = 0U; agent < counts_.blueMobiles; ++agent)
        {
            float followingAdvantage = 0.F;
            for (std::size_t reverse = rolloutTeamSteps_; reverse > 0U;
                 --reverse)
            {
                auto& transition = transitions_[TransitionIndex(
                    reverse - 1U, agent)];
                if (!transition.available)
                {
                    transition.returnValue = 0.F;
                    transition.advantage = 0.F;
                    followingAdvantage = 0.F;
                    continue;
                }
                bool const terminal = transition.terminated
                    || !transition.nextAvailable;
                bool const traceBoundary = terminal || transition.truncated;
                // A time-limit truncation ends the GAE trace but is not a
                // terminal physical state: retain V(s') in the TD target.
                float const bootstrap = terminal ? 0.F
                                                 : transition.nextValue;
                float const delta = transition.reward
                    + PPOParameters::GAMMA * bootstrap - transition.value;
                transition.advantage = delta
                    + (traceBoundary ? 0.F
                                     : PPOParameters::GAMMA
                                           * PPOParameters::LAMBDA
                                           * followingAdvantage);
                transition.returnValue =
                    transition.value + transition.advantage;
                followingAdvantage = transition.advantage;
                absoluteAdvantage += std::abs(transition.advantage);
                ++learnable;
            }
        }
        diagnostics_.generalizedAdvantageAbsMean = learnable == 0U
            ? 0.F
            : absoluteAdvantage / static_cast<float>(learnable);
        return learnable;
    }

    void NormalizeAdvantages(std::size_t const learnable) noexcept
    {
        float mean = 0.F;
        for (auto const& transition : transitions_)
            if (transition.available)
                mean += transition.advantage;
        mean /= static_cast<float>(learnable);
        float variance = 0.F;
        for (auto const& transition : transitions_)
        {
            if (!transition.available)
                continue;
            float const centered = transition.advantage - mean;
            variance += centered * centered;
        }
        float const scale = 1.F / std::sqrt(
            variance / static_cast<float>(learnable)
            + PPOParameters::ADVANTAGE_EPSILON);
        for (auto& transition : transitions_)
            if (transition.available)
                transition.advantage =
                    (transition.advantage - mean) * scale;
    }

    std::size_t CountLearnableBatch(std::size_t const offset) noexcept
    {
        std::size_t count = 0U;
        for (Index row = 0; row < kTrainingBatch; ++row)
        {
            Transition const* const transition = BatchTransition(offset, row);
            if (transition != nullptr && transition->available)
                ++count;
        }
        return count;
    }

    void TrainActorBatch(std::size_t const offset,
                         std::size_t const batchLearnable)
    {
        rlt::zero_gradient(device_, actor_);
        rlt::forward(device_, actor_, trainingRaw_, trainingActorOutput_);
        rlt::set_all(device_, trainingActorDerivative_, 0.F);
        for (Index row = 0; row < kTrainingBatch; ++row)
        {
            Transition const* const transition = BatchTransition(offset, row);
            if (transition == nullptr || !transition->available)
                continue;

            float currentLogProbability = 0.F;
            constexpr float kTwoPi = 6.28318530717958647692F;
            for (Index action = 0;
                 action < static_cast<Index>(
                              GeneralizedObservationLayout::ActionSize);
                 ++action)
            {
                float const mean = rlt::get(
                    trainingActorOutput_, row, action);
                float const logStd = rlt::get(
                    actor_.log_std.parameters, 0, action);
                float const deviation = std::exp(logStd);
                float const normalized =
                    (transition->action[action] - mean) / deviation;
                currentLogProbability +=
                    -0.5F * normalized * normalized - logStd
                    - 0.5F * std::log(kTwoPi);
            }
            float const ratio = std::exp(std::clamp(
                currentLogProbability - transition->oldLogProbability,
                -20.F, 20.F));
            float const clipped = std::clamp(
                ratio, 1.F - PPOParameters::EPSILON_CLIP,
                1.F + PPOParameters::EPSILON_CLIP);
            float const surrogate = ratio * transition->advantage;
            float const clippedSurrogate = clipped * transition->advantage;
            float const logProbabilityDerivative =
                surrogate <= clippedSurrogate
                    ? -transition->advantage * ratio
                          / static_cast<float>(batchLearnable)
                    : 0.F;
            for (Index action = 0;
                 action < static_cast<Index>(
                              GeneralizedObservationLayout::ActionSize);
                 ++action)
            {
                float const mean = rlt::get(
                    trainingActorOutput_, row, action);
                float const logStd = rlt::get(
                    actor_.log_std.parameters, 0, action);
                float const variance = std::exp(2.F * logStd);
                float const difference = transition->action[action] - mean;
                float const derivative =
                    logProbabilityDerivative * difference / variance;
                rlt::set(
                    trainingActorDerivative_, row, action,
                    derivative);
                diagnostics_.actorDerivativeL1 += std::abs(derivative);
                float const normalizedSquared =
                    difference * difference / variance;
                float const logStdGradient =
                    logProbabilityDerivative * (normalizedSquared - 1.F)
                    - PPOParameters::ACTION_ENTROPY_COEFFICIENT
                          / static_cast<float>(batchLearnable);
                rlt::increment(actor_.log_std.gradient, 0, action,
                               logStdGradient);
                diagnostics_.logStdGradientL1 +=
                    std::abs(logStdGradient);
            }
        }
        rlt::backward(device_, actor_, trainingRaw_,
                      trainingActorDerivative_, trainingActorBuffer_);
        rlt::step(device_, actorOptimizer_, actor_);
    }

    void TrainCriticBatch(std::size_t const offset,
                          std::size_t const batchLearnable)
    {
        rlt::zero_gradient(device_, critic_);
        rlt::forward(device_, critic_, trainingCritic_,
                     trainingCriticOutput_);
        rlt::set_all(device_, trainingCriticDerivative_, 0.F);
        for (Index row = 0; row < kTrainingBatch; ++row)
        {
            Transition const* const transition = BatchTransition(offset, row);
            if (transition == nullptr || !transition->available)
                continue;
            float const value = rlt::get(trainingCriticOutput_, row, 0);
            float const derivative =
                2.F * (value - transition->returnValue)
                / static_cast<float>(batchLearnable);
            rlt::set(trainingCriticDerivative_, row, 0, derivative);
            diagnostics_.criticDerivativeL1 += std::abs(derivative);
        }
        rlt::backward(device_, critic_, trainingCritic_,
                      trainingCriticDerivative_, trainingCriticBuffer_);
        rlt::step(device_, criticOptimizer_, critic_);
    }

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
        rlt::malloc(device_, inferenceActorBuffer_);
        inferenceActorBufferAllocated_ = true;
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
        if (inferenceActorBufferAllocated_)
            rlt::free(device_, inferenceActorBuffer_);
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
    std::vector<Transition> transitions_;
    std::vector<std::size_t> trainingOrder_;
    Device device_{};
    Engine rng_{};
    Actor actor_{};
    Critic critic_{};
    ActorOptimizer actorOptimizer_{};
    CriticOptimizer criticOptimizer_{};

    rlt::MatrixDynamic<InferenceRawSpecification> inferenceRaw_{};
    rlt::MatrixDynamic<InferenceCompactSpecification> inferenceCompact_{};
    rlt::MatrixDynamic<InferenceOutputSpecification> inferenceOutput_{};
    typename Actor::template Buffer<
        static_cast<Index>(TeamCapacity::MaximumBlueMobiles)>
        inferenceActorBuffer_{};
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
    std::size_t collectedTeamSteps_{};
    std::size_t lastTeamActionCount_{};
    std::size_t lastTeamTransitionCount_{};
    std::uint64_t completedUpdates_{};
    std::uint64_t totalActionsSampled_{};
    std::uint64_t totalTransitionsRecorded_{};
    std::uint64_t actorEvaluationCount_{};
    bool pending_{};

    bool actorAllocated_{};
    bool criticAllocated_{};
    bool inferenceRawAllocated_{};
    bool inferenceCompactAllocated_{};
    bool inferenceOutputAllocated_{};
    bool inferenceActorBufferAllocated_{};
    bool trainingRawAllocated_{};
    bool trainingCriticAllocated_{};
    bool trainingActorOutputAllocated_{};
    bool trainingActorDerivativeAllocated_{};
    bool trainingCriticOutputAllocated_{};
    bool trainingCriticDerivativeAllocated_{};
    bool trainingActorBufferAllocated_{};
    bool trainingCriticBufferAllocated_{};
};

} // namespace ppo::examples::twovstwo::defenders
