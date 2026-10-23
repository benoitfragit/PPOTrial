/**
* @file PPO/data/trainingagent.hpp
* @brief Declarations for trainingagent.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include <array>
#include <vector>
#include "torchlearner.hpp"
#include "ppo/core/buffers.hpp"
#include <functional>
#include <string>
#include "serializer.hpp"

namespace ppo::model {

/**
* @brief Generic training-time agent implementation.
* @tparam Policy Policy network type used by RLAgent.
* @tparam Model Domain model exposing state, action, and reward definitions.
*/
template<typename Policy, typename Model>
class TrainingAgent :
    public IAgent
{
public:
    /**
    * @brief Alias to the model-defined state vector type.
    */
    using StateVector = typename Model::STATES;
    /**
    * @brief Alias to the model-defined action vector type.
    */
    using ActionVector = typename Model::ACTIONS;

    /**
    * @brief Creates a training agent bound to one model and one PPO workspace.
    * @param model Controlled domain model.
    * @param name Logical agent name.
    * @param modelpath Root model directory.
    * @param evaluator Optional evaluator triggered at checkpoints.
    */
    TrainingAgent(Model* const model,
                  std::string const& name,
                  std::string const& modelpath,
                  IEvaluator* const evaluator = nullptr);

    /**
    * @brief Initializes model bindings and internal RL components.
    */
    void Init(void) override;
    /**
    * @brief Handles real-time control commands sent by the environment.
    * @param command Real-time command payload.
    */
    void ProcessCommand(sAgentRealTimeCommand const& command) noexcept override;
    /**
    * @brief Computes reward transition and forwards it to the RL backend.
    * @return True when one rollout has just completed.
    */
    bool SetNewState(void) override;
    /**
    * @brief Requests a new action from the RL backend and applies it to the model.
    */
    void ChooseAction(void) override;

    /**
    * @brief Domain-specific reward computation hook.
    * @return True when the current episode should stop.
    */
    virtual bool RewardFunction(void) noexcept = 0;

    /**
    * @brief Sets the terminal state flag for the current episode.
    * @param done Terminal flag to accumulate.
    */
    void SetDone(bool done) noexcept;
    /**
    * @brief Adds one reward contribution in a reward channel.
    * @param type Reward channel index.
    * @param reward Reward contribution.
    */
    void AddReward(unsigned char type, float reward) noexcept;

    /**
    * @brief Returns true when the configured training budget was reached.
    * @return Training completion flag.
    */
    bool IsTrained(void) const noexcept;
    /**
    * @brief Returns true while optimization is active.
    * @return Training activity flag.
    */
    bool IsTraining(void) const noexcept;
    /**
    * @brief Returns true on the final step of the current rollout window.
    * @return Final-step flag.
    */
    bool IsLastStep(void) const noexcept;
    /**
    * @brief Produces a serializable snapshot of the training state.
    * @return Immutable agent snapshot.
    */
    sAgentStateSnaphot GetSnapshot(void) const noexcept override;

protected:
    /**
    * @brief Domain model controlled by this agent.
    */
    Model* p_model{ nullptr };
    /**
    * @brief Overrides the evaluator pointer after construction.
    * @param evaluator Evaluator instance to use.
    */
    void SetEvaluator(IEvaluator* evaluator) noexcept { p_evaluator = evaluator; }

private:
    /**
    * @brief Loads agent configuration from JSON.
    * @param ar JSON archive.
    */
    void LoadFromFile(ppo::serialization::json& ar);
    /**
    * @brief Saves agent configuration to JSON.
    * @param ar JSON archive.
    */
    void SaveToFile(ppo::serialization::json& ar);

    /**
    * @brief Shared training policy network.
    */
    std::shared_ptr<Policy> p_policy{ nullptr };
    /**
    * @brief Current state buffer.
    */
    StateVector m_stateVector{};
    /**
    * @brief Current action buffer.
    */
    ActionVector m_actionVector{};
    /**
    * @brief Reinforcement-learning backend.
    */
    std::unique_ptr<TorchLearner> p_rlagent{ nullptr };
    /**
    * @brief Shared internal training state.
    */
    sCommonAgentState m_internal{};
    /**
    * @brief Reward vector storing per-channel accumulated rewards.
    */
    using RewardVector = std::array<float, Model::REWARDS.size()>;
    /**
    * @brief Reward scaling vector storing per-channel weights.
    */
    using RewardScaling = std::array<float, Model::REWARDS.size()>;

    /**
    * @brief Per-channel accumulated reward values.
    */
    RewardVector m_rewardVector{};
    /**
    * @brief Per-channel reward scaling values.
    */
    RewardScaling m_rewardScalings{};
    
    /**
    * @brief Logical agent name.
    */
    std::string m_name{ "" };
    /**
    * @brief Root model directory used for serialization.
    */
    std::string m_modelpath{ "" };
    /**
    * @brief Optional evaluator used at checkpoints.
    */
    IEvaluator* p_evaluator{ nullptr };

    /**
    * @brief Unique runtime identifier.
    */
    unsigned long long m_id{ GetNewUniqId() };
};

} /* ppo::model */

#include "trainingagent_def.hpp"
