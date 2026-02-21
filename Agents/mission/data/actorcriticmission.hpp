#pragma once

#include "trainingnet.hpp"

#include <memory>
#include <filesystem>
#include <torch/torch.h>
#include <tuple>

namespace rl {

struct ParsedInput
{
    torch::Tensor tracks;
    torch::Tensor ammo;
    torch::Tensor impacts;
};

class ActorCriticTrackAttention :
    public ppo::model::ITrainingNet
{
public:
    ActorCriticTrackAttention(int64_t state_dim, int64_t action_dim);

    // =====================================================
    // PPO API
    // =====================================================

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    selectaction(const torch::Tensor& state) final;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    evaluate(const torch::Tensor& state,
             const torch::Tensor& actions) final;// [B]

    std::tuple<
        torch::Tensor, // mu [B,2+1]
        torch::Tensor, // log_std [B,2]
        torch::Tensor  // value [B,1]
    >
   forward(torch::Tensor state) final;

   torch::Tensor forward_mu(torch::Tensor state);
private:
    ParsedInput parse_input(const torch::Tensor& input) const;

    torch::Tensor masked_attention_context(
        const torch::Tensor& feat,   // [B,N,d]
        const torch::Tensor& query,  // [B,d]
        const torch::Tensor& mask);  // [B,N]

private:
    // =====================================================
    // Constantes
    // =====================================================
    int64_t M{ 9 };
    int64_t N{ 10 };
    int64_t P{ 62 }; 
    int64_t d_model{ 64 };

    // =====================================================
    // Encoder par track
    // =====================================================
    torch::nn::Linear enc1{nullptr};
    torch::nn::Linear enc2{nullptr};

    // =====================================================
    // Attention
    // =====================================================
    torch::nn::Linear attn_q{nullptr};
    torch::nn::Linear attn_k{nullptr};

    // =====================================================
    // Actor heads
    // =====================================================
    torch::nn::Linear mu_head{nullptr};
    torch::nn::Linear logstd_head{nullptr};
    torch::nn::Linear fire_head{nullptr};

    // =====================================================
    // Critic
    // =====================================================
    torch::nn::Linear critic1{nullptr};
    torch::nn::Linear critic2{nullptr};
};

class ActorCriticTrackAttentionProd
{
public:
    ActorCriticTrackAttentionProd(int64_t state_dim, int64_t action_dim);

    void load(std::filesystem::path const& modelpath);
    torch::Tensor forward(torch::Tensor state);

private:
    std::shared_ptr<ActorCriticTrackAttention> p_policy{ nullptr };
};

} // namespace ppo::mission
