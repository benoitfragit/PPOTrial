/**
 * @file Examples/Airspace/Agents/Bindings/data/external_training.hpp
 * @brief A training agent driven from outside the world (an environment DLL for ppokit-run).
 * @details The agent has no learner: the action comes from the caller, and the agent
 * publishes what the learner reads (the next observation, the step's clamped reward and its
 * weighted channels, the end of the episode).
 */
#pragma once

#include "ppo/core/environment.hpp"

#include <cstddef>
#include <vector>

namespace ppo::model {

/** @brief What one step of an external training agent exchanges with its caller. */
struct ExternalStep final
{
    /** @brief Observation the next action is chosen on (after a reset or a step). */
    std::vector<float> observation;
    /** @brief Action of the step, written by the caller before the world steps. */
    std::vector<float> action;
    /** @brief The step's reward: weighted channels summed, then clamped (rewardClamp). */
    double reward = 0.0;
    /** @brief The step's weighted reward channels, in the model's REWARDS order. */
    std::vector<double> channels;
    /** @brief The reward function ended the episode on a terminal state (SetDone). */
    bool terminated = false;
    /** @brief The reward function ended the episode (terminal or not). */
    bool ended = false;
};

/** @brief A training agent the environment DLL drives (TrainingAgent). */
class IExternalTrainingAgent
{
public:
    virtual ~IExternalTrainingAgent() = default;

    /** @brief The exchange of the current step. */
    virtual ExternalStep& External() noexcept = 0;
    /** @brief Steps of one rollout of the agent's settings: the episode's step limit. */
    virtual std::size_t RolloutSteps() const noexcept = 0;
    /** @brief Sets the weights of the reward channels (Reward.Scaling), in REWARDS order. */
    virtual void SetRewardWeights(double const* weights) noexcept = 0;
    /** @brief The weights of the reward channels, in REWARDS order. */
    virtual void RewardWeights(double* weights) const noexcept = 0;
};

/**
 * @brief A policy whose action the caller sets before the agent chooses: a production agent
 *        playing ppokit-run's policy in an evaluation episode of the environment DLL.
 */
class ExternalPolicy final : public ppo::core::IPolicy
{
public:
    /** @param actions Size of an action. */
    explicit ExternalPolicy(std::size_t actions) : action_(actions, 0.F) {}

    /** @brief The action of the next step, written by the caller. */
    ppo::core::Action& Action() noexcept { return action_; }

    ppo::core::Action ChooseAction(ppo::core::Observation const& /*observation*/) override { return action_; }

private:
    ppo::core::Action action_;
};

} // namespace ppo::model
