/**
* @file Runtime/include/ppo/runtime/iagent.hpp
* @brief Declarations for iagent.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "ppo/core/evaluator_state.hpp"
#include "ppo/runtime/ppoparameters.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <atomic>
#include <vector>

namespace ppo::model {

/**
* @brief Returns a unique identifier for agent-like runtime objects.
* @return Monotonically increasing identifier.
*/
inline unsigned long long
GetNewUniqId(void)
{
    static std::atomic<unsigned long long> counter{ 0 };
    ++counter;
    return counter.load(std::memory_order_relaxed);
}

/**
* @brief Runtime training state exposed by one agent.
*/
struct sCommonAgentState :
    public PPOParamters
{
    /**
    * @brief Current step inside the rollout.
    */
    long long m_step{ 0 };
    /**
    * @brief Maximum number of steps in one rollout.
    */
    long long m_maxStep{ 0 };
    /**
    * @brief Current clamped reward value.
    */
    float m_reward{ 0.F };
    /**
    * @brief Reward clamping threshold.
    */
    float m_rewardClamp{ 3.F };
    /**
    * @brief Total PPO loss.
    */
    float m_loss{ 0.F };
    /**
    * @brief Actor loss term.
    */
    float m_actorloss{ 0.F };
    /**
    * @brief Entropy loss term.
    */
    float m_entropyloss{ 0.F };
    /**
    * @brief Value loss term.
    */
    float m_valueloss{ 0.F };
    /**
    * @brief Evaluation checkpoint period in rollouts.
    */
    unsigned int m_checkpoint{ 0 };
    /**
    * @brief Current model version.
    */
    unsigned int m_version{ 0 };
    /**
    * @brief Latest evaluator score.
    */
    float m_notation{ 0.F };
    float m_latestEvaluationNotation{ 0.F };
    std::uint64_t m_completedEvaluations{ 0 };
    bool m_hasLatestEvaluation{ false };
    float m_latestEvaluationSuccessRate{ 0.F };
    float m_latestEvaluationSurvivalRate{ 0.F };
    unsigned int m_latestEvaluationTestCount{ 0U };
    unsigned int m_evaluationTestsFinished{ 0U };
    unsigned int m_evaluationTimeoutTests{ 0U };
    float m_latestEvaluationTimeoutRate{ 0.F };
    unsigned long long m_evaluationStepsExecuted{ 0ULL };
    float m_evaluationStepsPerSecond{ 0.F };
    float m_evaluationElapsedSeconds{ 0.F };
    /**
    * @brief Latest evaluator progress ratio.
    */
    float m_evaluationprogress{ 0.F };
    /**
    * @brief True while an evaluator run is active.
    */
    bool m_evaluationinprogress{ false };
    /**
    * @brief Episode termination flag.
    */
    bool m_done{ false };
    /**
    * @brief True while PPO optimization is active.
    */
    bool m_training{ false };
    /** Completed MAPPO optimizer updates (zero for other agent families). */
    std::uint64_t m_completedTrainingUpdates{ 0U };
};

/**
* @brief Serializable view of one agent state for the UI.
*/
struct sAgentStateSnaphot
{
    /**
    * @brief Common scalar agent state.
    */
    sCommonAgentState m_common{};
    /**
    * @brief Reward scaling factors.
    */
    std::vector<float> m_scalings{};
    /**
    * @brief Accumulated rewards per channel.
    */
    std::vector<float> m_rewards{};
    /**
    * @brief Reward channel names.
    */
    std::vector<std::string> m_rewardnames{};
    /**
    * @brief Logical agent name.
    */
    std::string m_name{ "" };
    /**
    * @brief Unique runtime identifier.
    */
    unsigned long long m_id{ 0 };
};

/**
* @brief PPO loss outputs reported after one optimization step.
*/
struct PPOOutputs
{
    /**
    * @brief Total PPO loss.
    */
    float m_loss{ 0.F };
    /**
    * @brief Actor loss term.
    */
    float m_actorloss{ 0.F };
    /**
    * @brief Critic loss term.
    */
    float m_valueloss{ 0.F };
    /**
    * @brief Entropy regularization term.
    */
    float m_entropyloss{ 0.F };
};

/**
* @brief Real-time command targeting one specific agent instance.
*/
struct sAgentRealTimeCommand
{
    /**
    * @brief Supported real-time command types.
    */
    enum class eTypeCommand
    {
        None,
        RolloutBuffer,
        RewardClamp,
        RewardScaling
    };

    /**
    * @brief Command type.
    */
    eTypeCommand m_type{ eTypeCommand::None };
    /**
    * @brief Target agent identifier.
    */
    unsigned long long m_id{ 0 };
    /**
    * @brief New rollout buffer size.
    */
    long long m_rolloutBuffer{ 0 };
    /**
    * @brief New reward clamp value.
    */
    float m_rewardClamp{ 0.F };
    /**
    * @brief New reward scaling factor.
    */
    float m_rewardScaling{ 0.F };
    /**
    * @brief Index of the reward channel to scale.
    */
    unsigned char m_rewardScalingType{ 0 };
};

/**
* @brief Common interface implemented by all runtime agents.
*/
class IAgent
{
public:
    /**
    * @brief Virtual destructor for interface-safe cleanup.
    */
    virtual ~IAgent(void) = default;

    /**
    * @brief Initializes the agent before a new episode.
    */
    virtual void Init(void) = 0;
    /**
    * @brief Chooses and applies one action.
    */
    virtual void ChooseAction(void) = 0;
    /**
    * @brief Pushes one new state transition into the agent.
    * @return True when a rollout or episode milestone is reached.
    */
    virtual bool SetNewState(void) = 0;
    /**
    * @brief Processes one real-time command.
    * @param command Command payload.
    */
    virtual void ProcessCommand(sAgentRealTimeCommand const& command) noexcept = 0;
    /**
    * @brief Builds a UI-friendly state snapshot.
    * @return Immutable snapshot copy.
    */
    virtual sAgentStateSnaphot GetSnapshot(void) const noexcept = 0;
};

/**
* @brief Interface for asynchronous evaluator backends.
*/
class IEvaluator
{
public:
    /**
    * @brief Virtual destructor for interface-safe cleanup.
    */
    virtual ~IEvaluator(void) = default;
    /**
    * @brief Starts one asynchronous evaluation campaign.
    * @param name Agent family name.
    * @param modelpath Root model folder.
    */
    virtual void Start(std::string const& name, std::filesystem::path const& modelpath) = 0;
    /**
    * @brief Stops the current evaluation campaign.
    */
    virtual void Stop(void) = 0;
    /**
    * @brief Returns the latest evaluator snapshot.
    * @return Immutable evaluator state.
    */
    virtual sEvaluatorState GetSnapshot(void) const noexcept = 0;
};

} /* ppo::model */
