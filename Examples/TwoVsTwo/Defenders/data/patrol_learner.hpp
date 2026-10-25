#pragma once

// CPU PPO optimizer backend reused by both the GUI IAgent and optional CLI.
// The surrounding IAgent, not this engine, is scheduled by Environment.
#include "patrol_rltools.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ppo::examples::twovstwo::defenders {

class PatrolLearner final
{
public:
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

    bool Step() { return rlt::step(device_, *state_); }
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
        // The pinned RLtools version has a broken free(State) overload.
        // Free the allocations made by malloc(State), in reverse ownership order.
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

    Device device_{};
    std::unique_ptr<TrainingState> state_;
    bool allocated_ = false;
};

} // namespace ppo::examples::twovstwo::defenders
