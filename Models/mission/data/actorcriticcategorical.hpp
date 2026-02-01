#pragma once

#include "trainingnet.hpp"

#include <memory>
#include <filesystem>
#include <torch/torch.h>

namespace ppo::mission {

class ActorCriticCategoricalNet :
    public ppo::model::ITrainingNet
{
public:
    ActorCriticCategoricalNet(int64_t M_,
                              int64_t actor_dim = 64,
                              int64_t critic_dim = 64,
                              int64_t d_model_ = 32);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor x) final;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    evaluate(const torch::Tensor& states,
             const torch::Tensor& actions) final;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    selectaction(const torch::Tensor& state) final;

private:
    torch::Tensor masked_attention_context(const torch::Tensor& kv,
                                           const torch::Tensor& query,
                                           const torch::Tensor& mask);
    int64_t M;

    int64_t d_model;        // per-track latent dim
    int64_t pooled_dim;     // = 2 * d_model
    int64_t actor_in_dim;   // = 4 * d_model
    int64_t critic_in_dim;  // = 3 * d_model

    int64_t actor_hidden;
    int64_t critic_hidden;

    // Encoder per-track
    torch::nn::Linear enc1{ nullptr };
    torch::nn::Linear enc2{ nullptr };

    // Attention
    torch::nn::Linear attn_k{ nullptr };
    torch::nn::Linear attn_q{ nullptr };
    torch::nn::Linear attn_v{ nullptr };
    torch::nn::Linear attn_s{ nullptr };

    // Actor
    torch::nn::Linear actor1{ nullptr };
    torch::nn::Linear actor2{ nullptr };

    // Critic
    torch::nn::Linear critic1{ nullptr };
    torch::nn::Linear critic2{ nullptr };
};

class ActorCriticCategoricalNetProd
{
public:
    ActorCriticCategoricalNetProd(int64_t M_,
                                  int64_t actor_dim = 64,
                                  int64_t critic_dim = 64,
                                  int64_t d_model = 32);

    void load(std::filesystem::path const& modelpath);
    torch::Tensor forward(torch::Tensor state);

private:
    std::shared_ptr<ActorCriticCategoricalNet> p_policy{ nullptr };
};

} // namespace ppo::mission
