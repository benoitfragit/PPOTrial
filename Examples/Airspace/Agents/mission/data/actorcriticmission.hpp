/**
* @file Agents/mission/data/actorcriticmission.hpp
* @brief Declarations for actorcriticmission.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "missionlayout.hpp"
#include "trainingnet.hpp"

#include <memory>
#include <filesystem>
#include <torch/torch.h>
#include <tuple>

namespace rl {

/**
* @brief Parsed view of the flattened mission observation tensor.
*/
struct ParsedInput
{
    /**
    * @brief Per-track features shaped as [B, N, M].
    */
    torch::Tensor tracks;
    /**
    * @brief Ammunition-related scalar features.
    */
    torch::Tensor ammo;
    /**
    * @brief Remaining global impact features.
    */
    torch::Tensor impacts;
};

/**
* @brief Mission actor-critic network with track attention.
*/
class ActorCriticTrackAttention :
    public ppo::model::ITrainingNet
{
public:
    /**
    * @brief Builds the mission network from state and action dimensions.
    * @param state_dim Flattened state size.
    * @param action_dim Flattened action size.
    */
    ActorCriticTrackAttention(int64_t state_dim, int64_t action_dim);

    /**
    * @brief Samples one action for one state batch.
    * @param state Input state tensor.
    * @return Action sample, log probability, and value.
    */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    selectaction(const torch::Tensor& state) final;

    /**
    * @brief Evaluates buffered actions against one state batch.
    * @param state Input state tensor.
    * @param actions Buffered action tensor.
    * @return Log probabilities, values, and entropies.
    */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    evaluate(const torch::Tensor& state,
             const torch::Tensor& actions) final;

    /**
    * @brief Runs the full actor-critic forward pass.
    * @param state Input state tensor.
    * @return Action parameters and value tensor.
    */
    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor
    >
    forward(torch::Tensor state) final;

    /**
    * @brief Runs only the actor mean head for production inference.
    * @param state Input state tensor.
    * @return Mean action tensor.
    */
    torch::Tensor forward_mu(torch::Tensor state);

private:
    /**
    * @brief Splits one flattened state tensor into semantic groups.
    * @param input Flattened input tensor.
    * @return Parsed tensor view.
    */
    ParsedInput parse_input(const torch::Tensor& input) const;

    /**
    * @brief Computes the attention context over valid tracks only.
    * @param feat Encoded track features.
    * @param query Attention query.
    * @param mask Track availability mask.
    * @return Context vector.
    */
    torch::Tensor masked_attention_context(
        const torch::Tensor& feat,
        const torch::Tensor& query,
        const torch::Tensor& mask);

private:
    /**
    * @brief Shared hidden width used by the attention encoder.
    */
    int64_t d_model{ 64 };

    /**
    * @brief First track encoder layer.
    */
    torch::nn::Linear enc1{ nullptr };
    /**
    * @brief Second track encoder layer.
    */
    torch::nn::Linear enc2{ nullptr };

    /**
    * @brief Attention query projection.
    */
    torch::nn::Linear attn_q{ nullptr };
    /**
    * @brief Attention key projection.
    */
    torch::nn::Linear attn_k{ nullptr };

    /**
    * @brief Continuous action mean head.
    */
    torch::nn::Linear mu_head{ nullptr };
    /**
    * @brief Continuous action log-standard-deviation head.
    */
    torch::nn::Linear logstd_head{ nullptr };
    /**
    * @brief Binary fire decision head.
    */
    torch::nn::Linear fire_head{ nullptr };

    /**
    * @brief First critic layer.
    */
    torch::nn::Linear critic1{ nullptr };
    /**
    * @brief Final critic output layer.
    */
    torch::nn::Linear critic2{ nullptr };
};

/**
* @brief Production wrapper around the mission actor-critic network.
*/
class ActorCriticTrackAttentionProd
{
public:
    /**
    * @brief Builds the production wrapper from state and action dimensions.
    * @param state_dim Flattened state size.
    * @param action_dim Flattened action size.
    */
    ActorCriticTrackAttentionProd(int64_t state_dim, int64_t action_dim);

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
    std::shared_ptr<ActorCriticTrackAttention> p_policy{ nullptr };
};

} /* rl */
