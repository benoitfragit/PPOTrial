#pragma once

// RLtools PPO backend. The ECS agent feeds on-policy transitions; this class
// owns only the optimizer and its dataset, never the environment or entities.
#include "patrol_rltools.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ppo::examples::twovstwo::defenders {

class PatrolLearner final
{
public:
    static constexpr std::size_t RolloutSteps =
        static_cast<std::size_t>(TrainingParameters::ON_POLICY_RUNNER_STEPS_PER_ENV);

    PatrolLearner(Parameters const& parameters, std::uint32_t seed)
        : state_(std::make_unique<TrainingState>())
    {
        if (!Valid(parameters))
            throw std::invalid_argument("Invalid patrol training parameters");
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

    ~PatrolLearner() { Release(); }
    PatrolLearner(PatrolLearner const&) = delete;
    PatrolLearner& operator=(PatrolLearner const&) = delete;
    PatrolLearner(PatrolLearner&&) = delete;
    PatrolLearner& operator=(PatrolLearner&&) = delete;

    // Called once before the ECS advances. Sample from the same RLtools actor
    // whose frozen log probability is recorded for the later clipped PPO loss.
    // The raw Gaussian action is stored; the SNA actuator clamps to [-1, 1].
    Action ChooseAction(Observation const& observation)
    {
        if (pending_)
            throw std::logic_error("Patrol learner: transition feedback missing");
        auto& dataset = state_->on_policy_runner_dataset;
        for (Index i = 0; i < ObservationLayout::Size; ++i)
        {
            if (!std::isfinite(observation[i]))
                throw std::invalid_argument("Nonfinite patrol observation");
            rlt::set(input_, 0, i, observation[i]);
            rlt::set(dataset.observations, samples_, i, observation[i]);
            rlt::set(dataset.observations_normalized, samples_, i, observation[i]);
        }
        rlt::evaluate(device_, state_->ppo.actor, input_, output_,
                      state_->actor_deterministic_evaluation_buffers);
        float logProbability = 0.F;
        Action action{};
        for (Index i = 0; i < ObservationLayout::ActionSize; ++i)
        {
            float const mean = rlt::get(output_, 0, i);
            float const logStd = rlt::get(state_->ppo.actor.log_std.parameters, 0, i);
            float const stddev = std::exp(logStd);
            if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0.F)
                throw std::runtime_error("Invalid PPO Gaussian distribution");
            float const raw = rlt::random::normal_distribution::sample(
                typename Device::SPEC::RANDOM{}, mean, stddev, state_->rng);
            if (!std::isfinite(raw))
                throw std::runtime_error("Nonfinite PPO sampled action");
            float const normalized = (raw - mean) / stddev;
            logProbability += -0.5F * normalized * normalized
                            - logStd - 0.5F * std::log(kTwoPi);
            action[i] = raw;
            rlt::set(dataset.actions_mean, samples_, i, mean);
            rlt::set(dataset.actions, samples_, i, raw);
        }
        rlt::set(dataset.action_log_probs, samples_, 0, logProbability);
        pending_ = true;
        return action;
    }

    // Feedback belongs to the ECS step just executed, not to a second world.
    // A reset is allowed only after feedback has been received. Episode
    // boundaries may fall inside an on-policy rollout.
    bool Observe(Observation const& nextObservation, float reward,
                 bool terminated, bool truncated)
    {
        if (!pending_)
            throw std::logic_error("Patrol learner: action required before feedback");
        if (!std::isfinite(reward))
            throw std::invalid_argument("Nonfinite patrol reward");
        auto& dataset = state_->on_policy_runner_dataset;
        for (Index i = 0; i < ObservationLayout::Size; ++i)
        {
            if (!std::isfinite(nextObservation[i]))
                throw std::invalid_argument("Nonfinite patrol next observation");
            rlt::set(dataset.all_observations, samples_ + 1U, i, nextObservation[i]);
            rlt::set(dataset.all_observations_normalized, samples_ + 1U, i,
                     nextObservation[i]);
        }
        rlt::set(dataset.rewards, samples_, 0, reward);
        rlt::set(dataset.terminated, samples_, 0, terminated ? 1.F : 0.F);
        rlt::set(dataset.truncated, samples_, 0,
                 (terminated || truncated) ? 1.F : 0.F);
        pending_ = false;
        ++samples_;
        if (samples_ != RolloutSteps)
            return false;

        // Same native RLtools GAE and clipped PPO implementation as the CLI
        // learner, but with externally collected ECS experiences.
        rlt::evaluate(device_, state_->ppo.critic, dataset.all_observations,
                      dataset.all_values, state_->critic_buffers_gae);
        rlt::estimate_generalized_advantages(device_, dataset, PPOParameters{});
        rlt::train(device_, state_->ppo, dataset,
                   state_->actor_optimizer, state_->critic_optimizer,
                   state_->ppo_buffers, state_->actor_buffers,
                   state_->critic_buffers, state_->rng);
        ++state_->step;
        samples_ = 0U;
        return true;
    }

    std::size_t CollectedSteps() const noexcept { return samples_; }
    bool HasPendingAction() const noexcept { return pending_; }

    // Optional headless command-line training remains supported. Never call
    // this method from the ECS IAgent: it collects a separate simulated world.
    bool Step()
    {
        if (pending_ || samples_ != 0U)
            throw std::logic_error("Cannot mix ECS rollout and headless PPO step");
        return rlt::step(device_, *state_);
    }

    std::uint64_t StepCount() const noexcept
    {
        return static_cast<std::uint64_t>(state_->step);
    }

    template<typename Consumer>
    void WithActor(Consumer&& consumer) const
    {
        std::forward<Consumer>(consumer)(state_->ppo.actor);
    }

private:
    void Release()
    {
        if (!allocated_)
            return;
        rlt::free(device_, input_);
        rlt::free(device_, output_);
        // The pinned RLtools version has a broken free(State) overload.
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
        float, Index, 1, ObservationLayout::Size>>;
    using Output = rlt::MatrixDynamic<rlt::matrix::Specification<
        float, Index, 1, ObservationLayout::ActionSize>>;
    Device device_{};
    std::unique_ptr<TrainingState> state_;
    Input input_{};
    Output output_{};
    std::size_t samples_ = 0U;
    bool pending_ = false;
    bool allocated_ = false;
};

} // namespace ppo::examples::twovstwo::defenders
