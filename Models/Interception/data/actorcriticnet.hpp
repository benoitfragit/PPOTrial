#pragma once
        
#include "trainingnet.hpp"

#include <memory>
#include <filesystem>

#include <torch/torch.h>

namespace ppo::interception {

/// <summary>
/// This network is going to be use by the training agent for the training
/// </summary>
class ActorCriticNet : 
    public ppo::model::ITrainingNet
{
public:
    ActorCriticNet(int64_t state_dim, int64_t action_dim, int64_t hidden_dim = 64);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        forward(torch::Tensor x);

    torch::Tensor forward_mu(torch::Tensor x);

private:
    torch::nn::Linear fc1           { nullptr };
    torch::nn::Linear fc2           { nullptr };
    torch::nn::Linear mu_head       { nullptr };
    torch::nn::Linear logstd_head   { nullptr };
    torch::nn::Linear critic        { nullptr };
};

/// <summary>
/// This is the production ready network that is going to be exported
/// </summary>
class ActorCriticNetProd
{
public:
    ActorCriticNetProd(int64_t state_dim, int64_t action_dim, int64_t hidden_dim = 64);

    void load(std::filesystem::path const& modelpath);

    torch::Tensor forward(torch::Tensor state);

private:
    std::shared_ptr<ActorCriticNet> p_policy{ nullptr };
};

} /* ppo::interception */