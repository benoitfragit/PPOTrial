#pragma once

// Native RLtools MAPPO backend. One optimizer owns the shared actor and the
// centralized critic; the ECS supplies the two simultaneous team transitions.
#include "patrol_rltools.hpp"
#include "patrol_checkpoint.hpp"
#include "patrol_training_checkpoint.hpp"
#include "patrol_training_settings.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ppo::examples::twovstwo::defenders {

/**
 * @brief Last finite PPO mini-batch diagnostics published to the desktop UI.
 */
struct PatrolTrainingDiagnostics final
{
    float totalLoss{};
    float actorLoss{};
    float valueLoss{};
    float entropyLoss{};
};

/**
 * @brief Owns one RLtools MAPPO optimizer with a configurable rollout preset.
 *
 * Both defenders share Actor parameters. Each Actor row receives its local
 * observation and masks the privileged suffix; the Critic consumes that
 * suffix during centralized training.
 */
class PatrolLearner final
{
public:
    using Observations = std::array<Observation, 2>;
    using Actions = std::array<Action, 2>;

    /**
     * @brief Creates the MAPPO learner.
     * @param parameters Valid environment and vehicle parameters.
     * @param seed RLtools random seed.
     * @param rolloutSteps ECS team steps collected before one update.
     */
    PatrolLearner(Parameters const& parameters, std::uint32_t seed,
                  std::size_t rolloutSteps = 2048U)
    {
        if (!Valid(parameters))
            throw std::invalid_argument("Invalid patrol training parameters");
        switch (rolloutSteps)
        {
        case 128U:  learner_.emplace<0>(std::make_unique<Implementation<128>>(parameters, seed)); break;
        case 256U:  learner_.emplace<1>(std::make_unique<Implementation<256>>(parameters, seed)); break;
        case 512U:  learner_.emplace<2>(std::make_unique<Implementation<512>>(parameters, seed)); break;
        case 1024U: learner_.emplace<3>(std::make_unique<Implementation<1024>>(parameters, seed)); break;
        case 2048U: learner_.emplace<4>(std::make_unique<Implementation<2048>>(parameters, seed)); break;
        case 4096U: learner_.emplace<5>(std::make_unique<Implementation<4096>>(parameters, seed)); break;
        default:
            throw std::invalid_argument("Unsupported patrol rollout size");
        }
    }

    PatrolLearner(PatrolLearner const&) = delete;
    PatrolLearner& operator=(PatrolLearner const&) = delete;
    PatrolLearner(PatrolLearner&&) = delete;
    PatrolLearner& operator=(PatrolLearner&&) = delete;

    /**
     * @brief Samples both decentralized actions from the frozen shared Actor.
     * @param observations Role-specific observations for the same ECS state.
     * @return Raw Gaussian actions; the ECS clamps only the applied commands.
     */
    Actions ChooseActions(Observations const& observations)
    {
        return std::visit([&](auto& learner) {
            return learner->ChooseActions(observations);
        }, learner_);
    }

    /**
     * @brief Records joint feedback and runs one RLtools update when full.
     * @param nextObservations Role-specific observations after ECS resolution.
     * @param teamReward Shared cooperative reward supplied to both trajectories.
     * @param terminated True for a terminal mission outcome.
     * @param truncated True for a finite-horizon time boundary.
     * @return True when an optimization update completed.
     */
    bool Observe(Observations const& nextObservations, float teamReward,
                 bool terminated, bool truncated)
    {
        return std::visit([&](auto& learner) {
            return learner->Observe(nextObservations, teamReward,
                                    terminated, truncated);
        }, learner_);
    }

    /** @brief Returns collected team steps in the active rollout. */
    std::size_t CollectedSteps() const noexcept
    {
        return std::visit([](auto const& learner) { return learner->CollectedSteps(); },
                          learner_);
    }

    /** @brief Returns the configured number of team steps per update. */
    std::size_t RolloutSize() const noexcept
    {
        return std::visit([](auto const& learner) { return learner->RolloutSize(); },
                          learner_);
    }

    /** @brief Returns whether joint ECS feedback is still required. */
    bool HasPendingAction() const noexcept
    {
        return std::visit([](auto const& learner) { return learner->HasPendingAction(); },
                          learner_);
    }

    /**
     * @brief Runs one optional headless RLtools collection/update.
     * @return RLtools loop continuation state.
     */
    bool Step()
    {
        return std::visit([](auto& learner) { return learner->Step(); }, learner_);
    }

    /** @brief Returns the number of completed MAPPO updates. */
    std::uint64_t StepCount() const noexcept
    {
        return std::visit([](auto const& learner) { return learner->StepCount(); },
                          learner_);
    }

    /** @brief Returns diagnostics from the last completed PPO update. */
    PatrolTrainingDiagnostics Diagnostics() const noexcept
    {
        return std::visit([](auto const& learner) {
            return learner->Diagnostics();
        }, learner_);
    }

    /**
     * @brief Replaces the trainable shared Actor with checkpoint parameters.
     * @param checkpoint Compatible v6 Actor checkpoint.
     */
    void LoadActor(std::filesystem::path const& checkpoint)
    {
        std::ifstream stream{checkpoint};
        if (!stream)
            throw std::runtime_error("Cannot open blue MAPPO checkpoint: "
                                     + checkpoint.string());
        std::visit([&](auto& learner) {
            ReadPatrolPolicy(stream, learner->MutableActor());
        }, learner_);
    }

    /**
     * @brief Saves Actor, Critic, Adam states, update count and RLTools RNG.
     *
     * The current rollout buffer is intentionally not persisted.
     * @param checkpoint Destination training-state checkpoint.
     */
    void SaveTrainingState(std::filesystem::path const& checkpoint) const
    {
        std::visit([&](auto const& learner) {
            learner->SaveTrainingState(checkpoint);
        }, learner_);
    }

    /**
     * @brief Restores a complete update-boundary training state.
     * @param checkpoint Compatible training-state checkpoint.
     */
    void LoadTrainingState(std::filesystem::path const& checkpoint)
    {
        std::visit([&](auto& learner) {
            learner->LoadTrainingState(checkpoint);
        }, learner_);
    }

    /**
     * @brief Invokes a consumer with the concrete shared RLtools Actor.
     * @tparam Consumer Callable accepting a const Actor reference.
     * @param consumer Actor consumer.
     */
    template<typename Consumer>
    void WithActor(Consumer&& consumer) const
    {
        std::visit([&](auto const& learner) {
            std::forward<Consumer>(consumer)(learner->Actor());
        }, learner_);
    }

private:
    template<Index RolloutSteps>
    class Implementation final
    {
    public:
        using Config = TrainingConfigFor<RolloutSteps>;
        using StateType = typename Config::template State<Config>;

        Implementation(Parameters const& parameters, std::uint32_t seed)
            : state_(std::make_unique<StateType>())
        {
            for (auto& environment : state_->envs)
                environment.parameters = parameters;
            rlt::malloc(device_, *state_);
            allocated_ = true;
            try
            {
                rlt::malloc(device_, input_);
                rlt::malloc(device_, output_);
                rlt::init(device_, *state_, seed);
                state_->actor_optimizer.parameters.alpha = 3.e-4F;
                state_->critic_optimizer.parameters.alpha = 3.e-4F;
            }
            catch (...)
            {
                Release();
                throw;
            }
        }

        ~Implementation() { Release(); }
        Implementation(Implementation const&) = delete;
        Implementation& operator=(Implementation const&) = delete;

        Actions ChooseActions(Observations const& observations)
        {
            if (pending_)
                throw std::logic_error("Patrol MAPPO: joint feedback missing");

            auto& dataset = state_->on_policy_runner_dataset;
            for (Index role = 0; role < 2; ++role)
            {
                Index const row = static_cast<Index>(samples_ * 2U) + role;
                for (Index i = 0; i < ObservationLayout::Size; ++i)
                {
                    float const value = observations[role][i];
                    if (!std::isfinite(value))
                        throw std::invalid_argument("Nonfinite patrol observation");
                    rlt::set(input_, role, i, value);
                    rlt::set(dataset.observations, row, i, value);
                    rlt::set(dataset.observations_normalized, row, i, value);
                }
            }

            rlt::evaluate(device_, state_->ppo.actor, input_, output_,
                          state_->actor_eval_buffers);
            Actions actions{};
            for (Index role = 0; role < 2; ++role)
            {
                Index const row = static_cast<Index>(samples_ * 2U) + role;
                float logProbability = 0.F;
                for (Index i = 0; i < ObservationLayout::ActionSize; ++i)
                {
                    float const mean = rlt::get(output_, role, i);
                    float const logStd = rlt::get(
                        state_->ppo.actor.log_std.parameters, 0, i);
                    float const stddev = std::exp(logStd);
                    if (!std::isfinite(mean) || !std::isfinite(stddev)
                        || stddev <= 0.F)
                        throw std::runtime_error("Invalid MAPPO Gaussian distribution");
                    float const raw = rlt::random::normal_distribution::sample(
                        typename Device::SPEC::RANDOM{}, mean, stddev, state_->rng);
                    if (!std::isfinite(raw))
                        throw std::runtime_error("Nonfinite MAPPO sampled action");
                    float const normalized = (raw - mean) / stddev;
                    logProbability += -0.5F * normalized * normalized
                                    - logStd - 0.5F * std::log(kTwoPi);
                    actions[role][i] = raw;
                    rlt::set(dataset.actions_mean, row, i, mean);
                    rlt::set(dataset.actions, row, i, raw);
                }
                rlt::set(dataset.action_log_probs, row, 0, logProbability);
            }
            pending_ = true;
            return actions;
        }

        bool Observe(Observations const& nextObservations, float teamReward,
                     bool terminated, bool truncated)
        {
            if (!pending_)
                throw std::logic_error("Patrol MAPPO: actions required before feedback");
            if (!std::isfinite(teamReward))
                throw std::invalid_argument("Nonfinite patrol team reward");

            auto& dataset = state_->on_policy_runner_dataset;
            for (Index role = 0; role < 2; ++role)
            {
                Index const row = static_cast<Index>((samples_ + 1U) * 2U) + role;
                for (Index i = 0; i < ObservationLayout::Size; ++i)
                {
                    float const value = nextObservations[role][i];
                    if (!std::isfinite(value))
                        throw std::invalid_argument("Nonfinite next patrol observation");
                    rlt::set(dataset.all_observations, row, i, value);
                    rlt::set(dataset.all_observations_normalized, row, i, value);
                }
                Index const transition = static_cast<Index>(samples_ * 2U) + role;
                rlt::set(dataset.rewards, transition, 0, teamReward);
                // RLtools v1 discards the TD error at a non-terminal
                // truncation. This task has a genuine finite horizon, so its
                // final reward (notably an unresolved-threat penalty) must be
                // retained and the value function must not bootstrap across
                // the following ECS reset.
                bool const terminalForLearning = terminated || truncated;
                rlt::set(dataset.terminated, transition, 0,
                         terminalForLearning ? 1.F : 0.F);
                rlt::set(dataset.truncated, transition, 0,
                         terminalForLearning ? 1.F : 0.F);
            }

            pending_ = false;
            ++samples_;
            if (samples_ != static_cast<std::size_t>(RolloutSteps))
                return false;

            // Native RLtools centralized Critic evaluation, GAE and clipped
            // PPO update over both interleaved defender trajectories.
            rlt::evaluate(device_, state_->ppo.critic, dataset.all_observations,
                          dataset.all_values, state_->critic_buffers_gae);
            rlt::estimate_generalized_advantages(device_, dataset, PPOParameters{});
            rlt::train(device_, state_->ppo, dataset,
                       state_->actor_optimizer, state_->critic_optimizer,
                       state_->ppo_buffers, state_->actor_buffers,
                       state_->critic_buffers, state_->rng);
            UpdateDiagnostics(dataset);
            ++state_->step;
            samples_ = 0U;
            return true;
        }

        std::size_t CollectedSteps() const noexcept { return samples_; }
        std::size_t RolloutSize() const noexcept
        {
            return static_cast<std::size_t>(RolloutSteps);
        }
        bool HasPendingAction() const noexcept { return pending_; }

        bool Step()
        {
            if (pending_ || samples_ != 0U)
                throw std::logic_error("Cannot mix ECS and headless MAPPO collection");
            return rlt::step(device_, *state_);
        }

        std::uint64_t StepCount() const noexcept
        {
            return static_cast<std::uint64_t>(state_->step);
        }

        PatrolTrainingDiagnostics Diagnostics() const noexcept
        {
            return diagnostics_;
        }

        auto const& Actor() const noexcept { return state_->ppo.actor; }
        auto& MutableActor() noexcept { return state_->ppo.actor; }

        void SaveTrainingState(std::filesystem::path const& checkpoint) const
        {
            SavePatrolAtomically(checkpoint, [&](std::ostream& stream) {
                WritePatrolTrainingState(
                    stream, state_->ppo.actor, state_->ppo.critic,
                    state_->actor_optimizer, state_->critic_optimizer,
                    state_->rng, static_cast<std::uint64_t>(state_->step));
            });
        }

        void LoadTrainingState(std::filesystem::path const& checkpoint)
        {
            if (pending_ || samples_ != 0U)
                throw std::logic_error(
                    "Cannot restore blue MAPPO during a rollout");
            std::ifstream stream{checkpoint};
            if (!stream)
                throw std::runtime_error(
                    "Cannot open blue MAPPO training checkpoint: "
                    + checkpoint.string());
            std::uint64_t const updates = ReadPatrolTrainingState(
                stream, state_->ppo.actor, state_->ppo.critic,
                state_->actor_optimizer, state_->critic_optimizer,
                state_->rng);
            if (updates > static_cast<std::uint64_t>(
                    std::numeric_limits<Index>::max()))
                throw std::runtime_error(
                    "Blue MAPPO update count exceeds the runtime index type");
            state_->step = static_cast<Index>(updates);
            diagnostics_ = {};
        }

    private:
        template<typename Dataset>
        void UpdateDiagnostics(Dataset const& dataset) noexcept
        {
            constexpr Index batchSize = TrainingParametersFor<
                RolloutSteps>::BATCH_SIZE;
            constexpr Index total = static_cast<Index>(RolloutSteps * 2U);
            constexpr Index offset = total - batchSize;
            float advantageMean = 0.F;
            float advantageSquaredMean = 0.F;
            for (Index row = 0; row < batchSize; ++row)
            {
                float const advantage = rlt::get(
                    dataset.advantages, offset + row, 0);
                advantageMean += advantage;
                advantageSquaredMean += advantage * advantage;
            }
            advantageMean /= static_cast<float>(batchSize);
            advantageSquaredMean /= static_cast<float>(batchSize);
            float const advantageStd = std::sqrt(std::max(
                advantageSquaredMean - advantageMean * advantageMean, 0.F));

            float actorLoss = 0.F;
            float valueLoss = 0.F;
            auto const& currentMeans = state_->ppo_buffers.current_batch_actions;
            auto const& criticOutput = rlt::output(state_->ppo.critic);
            for (Index row = 0; row < batchSize; ++row)
            {
                float currentLogProbability = 0.F;
                for (Index action = 0;
                     action < ObservationLayout::ActionSize; ++action)
                {
                    float const mean = rlt::get(currentMeans, row, action);
                    float const sampled = rlt::get(
                        dataset.actions, offset + row, action);
                    float const logStd = rlt::get(
                        state_->ppo.actor.log_std.parameters, 0, action);
                    float const stddev = std::exp(logStd);
                    float const normalized = (mean - sampled) / stddev;
                    currentLogProbability += -0.5F * normalized * normalized
                        - logStd - 0.5F * std::log(kTwoPi);
                }
                float advantage = rlt::get(
                    dataset.advantages, offset + row, 0);
                if constexpr (PPOParameters::NORMALIZE_ADVANTAGE)
                    advantage = (advantage - advantageMean)
                        / (advantageStd + PPOParameters::ADVANTAGE_EPSILON);
                float const oldLogProbability = rlt::get(
                    dataset.action_log_probs, offset + row, 0);
                float const ratio = std::exp(Clamp(
                    currentLogProbability - oldLogProbability, -20.F, 20.F));
                float const clippedRatio = Clamp(
                    ratio, 1.F - PPOParameters::EPSILON_CLIP,
                    1.F + PPOParameters::EPSILON_CLIP);
                actorLoss -= std::min(
                    ratio * advantage, clippedRatio * advantage);

                float const value = rlt::get(criticOutput, row, 0);
                float const target = rlt::get(
                    dataset.target_values, offset + row, 0);
                float const difference = value - target;
                valueLoss += difference * difference;
            }
            actorLoss /= static_cast<float>(batchSize);
            valueLoss /= static_cast<float>(batchSize);

            float entropyLoss = 0.F;
            for (Index action = 0;
                 action < ObservationLayout::ActionSize; ++action)
            {
                float const logStd = rlt::get(
                    state_->ppo.actor.log_std.parameters, 0, action);
                entropyLoss -= PPOParameters::ACTION_ENTROPY_COEFFICIENT
                    * (logStd + 0.5F * std::log(kTwoPi) + 0.5F);
            }
            PatrolTrainingDiagnostics const candidate{
                actorLoss + valueLoss + entropyLoss,
                actorLoss, valueLoss, entropyLoss};
            if (std::isfinite(candidate.totalLoss)
                && std::isfinite(candidate.actorLoss)
                && std::isfinite(candidate.valueLoss)
                && std::isfinite(candidate.entropyLoss))
                diagnostics_ = candidate;
        }

        void Release()
        {
            if (!allocated_)
                return;
            rlt::free(device_, input_);
            rlt::free(device_, output_);
            // RLtools v1.0.0 has no usable free(State) overload.
            rlt::free(device_, state_->ppo);
            rlt::free(device_, state_->ppo_buffers);
            rlt::free(device_, state_->on_policy_runner_dataset);
            rlt::free(device_, state_->on_policy_runner);
            rlt::free(device_, state_->actor_eval_buffers);
            rlt::free(device_, state_->actor_deterministic_evaluation_buffers);
            rlt::free(device_, state_->actor_buffers);
            rlt::free(device_, state_->critic_buffers);
            rlt::free(device_, state_->critic_buffers_gae);
            rlt::free(device_, state_->observation_normalizer);
            rlt::free(device_, state_->observations_mean);
            rlt::free(device_, state_->observations_std);
            allocated_ = false;
        }

        using Input = rlt::MatrixDynamic<rlt::matrix::Specification<
            float, Index, 2, ObservationLayout::Size>>;
        using Output = rlt::MatrixDynamic<rlt::matrix::Specification<
            float, Index, 2, ObservationLayout::ActionSize>>;

        Device device_{};
        std::unique_ptr<StateType> state_;
        Input input_{};
        Output output_{};
        std::size_t samples_ = 0U;
        PatrolTrainingDiagnostics diagnostics_{};
        bool pending_ = false;
        bool allocated_ = false;
    };

    using Learner = std::variant<
        std::unique_ptr<Implementation<128>>,
        std::unique_ptr<Implementation<256>>,
        std::unique_ptr<Implementation<512>>,
        std::unique_ptr<Implementation<1024>>,
        std::unique_ptr<Implementation<2048>>,
        std::unique_ptr<Implementation<4096>>>;

    Learner learner_{};
};

} // namespace ppo::examples::twovstwo::defenders
