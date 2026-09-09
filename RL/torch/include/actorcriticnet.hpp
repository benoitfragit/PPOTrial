/**
* @file Agents/Generic/data/actorcriticnet.hpp
* @brief Declarations for actorcriticnet.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "trainingnet.hpp"

#include <memory>
#include <filesystem>

#include <torch/torch.h>

namespace rl {

/**
* @brief Training network used by the continuous-action PPO agent.
*/
class ActorCriticNet :
    public ppo::model::ITrainingNet
{
public:
    /**
    * @brief Builds the continuous-action network.
    * @param state_dim Flattened state size.
    * @param action_dim Flattened action size.
    * @param hidden_dim Shared hidden width.
    */
    ActorCriticNet(int64_t state_dim, int64_t action_dim, int64_t hidden_dim = 64);

    /**
    * @brief Runs the full actor-critic forward pass.
    * @param x Input state tensor.
    * @return Action parameters and value tensor.
    */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) final;
    /**
    * @brief Evaluates buffered actions against a batch of states.
    * @param states Input state batch.
    * @param actions Buffered action batch.
    * @return Log probabilities, values, and entropies.
    */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(
        const torch::Tensor& states,
        const torch::Tensor& actions) final;
    /**
    * @brief Samples one action for one state batch.
    * @param state Input state tensor.
    * @return Action sample, log probability, and value.
    */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> selectaction(
        const torch::Tensor& state) final;

    /**
    * @brief Runs only the actor mean head for production inference.
    * @param x Input state tensor.
    * @return Mean action tensor.
    */
    torch::Tensor forward_mu(torch::Tensor x);

private:
    /**
    * @brief First hidden layer.
    */
    torch::nn::Linear fc1{ nullptr };
    /**
    * @brief Second hidden layer.
    */
    torch::nn::Linear fc2{ nullptr };
    /**
    * @brief Mean action head.
    */
    torch::nn::Linear mu_head{ nullptr };
    /**
    * @brief Log-standard-deviation head.
    */
    torch::nn::Linear logstd_head{ nullptr };
    /**
    * @brief Value head.
    */
    torch::nn::Linear critic{ nullptr };
};

/**
* @brief Production wrapper around the continuous-action actor-critic network.
*/
class ActorCriticNetProd
{
public:
    /**
    * @brief Builds the production wrapper.
    * @param state_dim Flattened state size.
    * @param action_dim Flattened action size.
    * @param hidden_dim Shared hidden width.
    */
    ActorCriticNetProd(int64_t state_dim, int64_t action_dim, int64_t hidden_dim = 64);

    /**
    * @brief Loads serialized weights from disk.
    * @param modelpath Path to the serialized model file.
    */
    void load(std::filesystem::path const& modelpath);

    /**
    * @brief Runs inference on one state tensor.
    * @param state Input state tensor.
    * @return Output action tensor.
    */
    torch::Tensor forward(torch::Tensor state);

private:
    /**
    * @brief Shared production policy instance.
    */
    std::shared_ptr<ActorCriticNet> p_policy{ nullptr };
};

} /* rl */
