#pragma once
        
#include "trainingnet.hpp"

#include <memory>
#include <filesystem>

#include <torch/torch.h>

namespace ppo::mission {

/// <summary>
/// This network is going to be use by the training agent for the training
/// </summary>
class ActorCriticCategoricalNet : 
    public ppo::model::ITrainingNet
{
public:
    ActorCriticCategoricalNet(int64_t M_, int64_t actor_dim=64, int64_t critic_dim = 64);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) final;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(const torch::Tensor& states, const torch::Tensor& actions) final;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> selectaction(const torch::Tensor& state) final;

private:
	torch::Tensor masked_attention_context(const torch::Tensor& kv,
										   const torch::Tensor& query,
										   const torch::Tensor& mask);


    int64_t M; // input dim per track
    int64_t actor_hidden;
    int64_t critic_hidden;
    // Encoder per-track : M -> 64 -> 32
    torch::nn::Linear enc1{nullptr};
    torch::nn::Linear enc2{nullptr};

    // Attention simple (1 tête)
    torch::nn::Linear attn_k{nullptr};
    torch::nn::Linear attn_q{nullptr};
    torch::nn::Linear attn_v{nullptr};
    torch::nn::Linear attn_s{nullptr};

    // Actor: (track32 + pooled64 + ctx32) = 128 -> hidden -> 1
    torch::nn::Linear actor1{nullptr};
    torch::nn::Linear actor2{nullptr};

    // Critic: pooled64 + ctx32 = 96 -> hidden -> 1
    torch::nn::Linear critic1{nullptr};
    torch::nn::Linear critic2{nullptr};
	
	int64_t d_model = 32;
};

/// <summary>
/// This is the production ready network that is going to be exported
/// </summary>
class ActorCriticCategoricalNetProd
{
public:
    ActorCriticCategoricalNetProd(int64_t M_, int64_t actor_dim=64, int64_t critic_dim = 64);

    void load(std::filesystem::path const& modelpath);

    torch::Tensor forward(torch::Tensor state);

private:
    std::shared_ptr<ActorCriticCategoricalNet> p_policy{ nullptr };
};

} /* ppo::mission */