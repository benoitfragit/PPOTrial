/**
* @file RL/torch/include/rlagent.hpp
* @brief Declarations for rlagent.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <memory>
#include <cstdint>
#include <filesystem>
#include "trainingnet.hpp"

namespace ppo::model {

/**
* @brief High-level reinforcement-learning agent wrapper.
*
* This class coordinates policy inference, rollout buffering, training steps,
* and model persistence through the PPO backend.
*/
class RLAgent
{
public:
    /**
    * @brief Creates a PPO-backed RL agent wrapper.
    * @param policy Shared training network.
    * @param modelpath Root model directory.
    * @param state_dim Flattened state size.
    * @param action_dim Flattened action size.
    * @param buffer_dim Rollout buffer length.
    * @param parameters PPO hyper-parameters.
    * @param stateVector Pointer to the live state buffer.
    * @param actionVector Pointer to the live action buffer.
    * @param loadExistingModel True to restore an existing model if present.
    */
    RLAgent(std::shared_ptr<ITrainingNet> policy,
            std::string const& modelpath,
            long long state_dim,
            long long action_dim,
            long long buffer_dim,
            PPOParamters const& parameters,
            float* const stateVector,
            float* const actionVector,
            bool loadExistingModel = true);

    /**
    * @brief Destroys the RL backend and owned resources.
    */
    ~RLAgent(void);

    RLAgent(RLAgent const&) = delete;
    RLAgent& operator=(RLAgent const&) = delete;
    RLAgent(RLAgent&&) = delete;
    RLAgent& operator=(RLAgent&&) = delete;

    /**
    * @brief Initializes internal training and inference state.
    */
    void Init(void);
    /**
    * @brief Computes and stores the next action from the current state.
    */
    void ChooseAction(void);
    /**
    * @brief Pushes transition feedback for the previously chosen action.
    * @param reward Reward observed after the action.
    * @param done Episode termination flag.
    */
    void SetNewState(float reward, bool done);
    /**
    * @brief Indicates whether the agent is currently in training mode.
    * @return Training activity flag.
    */
    bool IsTraining(void) const noexcept;
    /**
    * @brief Returns the current training step index.
    * @return Current rollout step.
    */
    long long GetCurrentStep(void) const noexcept;
    /**
    * @brief Returns the configured maximum number of steps.
    * @return Rollout length.
    */
    long long GetMaxStep(void) const noexcept;
    /**
    * @brief Saves model weights and metadata to disk.
    * @param modelpath Destination path.
    */
    void Save(std::filesystem::path const& modelpath);
    /**
    * @brief Returns last optimization metrics.
    * @return Immutable PPO output structure.
    */
    PPOOutputs const& GetOutputs(void) const noexcept;

private:
    /**
    * @brief Hidden implementation storing PPO runtime state.
    */
    struct Impl;
    /**
    * @brief Pointer to the RL-agent implementation.
    */
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::model */
