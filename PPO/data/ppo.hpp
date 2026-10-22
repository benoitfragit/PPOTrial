/**
* @file PPO/data/ppo.hpp
* @brief Declarations for ppo.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include <torch/torch.h>
#include <memory>
#include <tuple>
#include <filesystem>

#include "trainingnet.hpp"
#include "ppoparameters.hpp"
#include "iagent.hpp"

namespace ppo::model {

/**
* @brief Proximal Policy Optimization training engine.
*/
class PPO
{
public:
    /**
    * @brief Builds the PPO optimizer around one policy network.
    * @param policy Shared training network.
    * @param modelpath Root model directory.
    * @param state_dim Flattened state size.
    * @param action_dim Flattened action size.
    * @param parameters PPO hyper-parameters.
    * @param loadExistingModel True to restore existing weights if present.
    */
    PPO(std::shared_ptr<ITrainingNet> policy,
        std::string const& modelpath,
        int64_t state_dim,
        int64_t action_dim,
        PPOParamters const& parameters,
        bool loadExistingModel = true);

    /**
    * @brief Destroys optimizer state and owned resources.
    */
    ~PPO(void);

    PPO(PPO const&) = delete;
    PPO& operator=(PPO const&) = delete;
    PPO(PPO&&) = delete;
    PPO& operator=(PPO&&) = delete;

    /**
    * @brief Performs one PPO optimization pass on collected rollout tensors.
    * @param states Rollout state tensor.
    * @param actions Rollout action tensor.
    * @param old_logp Previous action log probabilities.
    * @param values Previous value estimates.
    * @param rewards Rollout rewards.
    * @param dones Episode termination flags.
    * @param last_values Bootstrap values for the rollout tail.
    */
    void UpdatePolicy(
        const torch::Tensor& states,
        const torch::Tensor& actions,
        const torch::Tensor& old_logp,
        const torch::Tensor& values,
        const torch::Tensor& rewards,
        const torch::Tensor& dones,
        const torch::Tensor& last_values);

    /**
    * @brief Samples an action from the policy for one state tensor.
    * @param state Input observation tensor.
    * @return Tuple containing action, log-probability, and value estimate.
    */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    SelectAction(const torch::Tensor& state);

    /**
    * @brief Saves policy and optimizer state to disk.
    * @param modelpath Destination path.
    */
    void Save(std::filesystem::path const& modelpath);

    /**
    * @brief Returns latest loss metrics produced by UpdatePolicy().
    * @return Immutable PPO output structure.
    */
    PPOOutputs const& GetOutputs(void) const noexcept;

private:
    /**
    * @brief Hidden implementation storing optimizer runtime state.
    */
    struct Impl;
    /**
    * @brief Pointer to the PPO implementation.
    */
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::model */
