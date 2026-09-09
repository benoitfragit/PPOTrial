/**
* @file PPO/data/trainingnet.hpp
* @brief Declarations for trainingnet.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "ppo_parameters.hpp"

#include <torch/torch.h>
#include <tuple>
#include <string>

namespace ppo::model {

/**
* @brief Abstract neural-network interface consumed by PPO training code.
*/
class ITrainingNet :
    public torch::nn::Module
{
public:
    /**
    * @brief Virtual destructor for interface-safe cleanup.
    */
    virtual ~ITrainingNet(void) = default;

    /**
    * @brief Runs a forward pass returning policy and value outputs.
    * @param x Input state tensor.
    * @return Policy parameters and value estimate.
    */
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) = 0;
    /**
    * @brief Evaluates stored actions against a batch of states.
    * @param states Batch of states.
    * @param actions Batch of actions.
    * @return Log probabilities, values, and entropies.
    */
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(
        const torch::Tensor& states,
        const torch::Tensor& actions) = 0;
    /**
    * @brief Samples or selects one action for a state tensor.
    * @param state Input state tensor.
    * @return Action, log probability, and value estimate.
    */
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> selectaction(
        const torch::Tensor& state) = 0;
};

} /* ppo::model */
