#pragma once

#include <torch/torch.h>
#include <tuple>

namespace ppo::model {

struct PPOParamters
{
    float           m_gamma{ 0.99F };
    float           m_lambda_gae{ 0.95F };
    float           m_clip_epsilon{ 0.2F };
    float           m_value_coef{ 0.5F };
    float           m_entropy_coef{ 0.02F };
    float           m_learningReate{ 3e-4F };
    int             m_K_epochs{ 4 };
    int             m_batch_size{ 64 };
    float           m_target_kl{ 0.01F };
};

struct sCommonAgentState  :
    public PPOParamters
{
    long long m_step{ 0 };
    long long m_maxStep{ 0 };
    float     m_reward{ 0.F };
    float     m_rewardClamp{ 3.F };
    float     m_loss{ 0.F };
    float     m_actorloss{ 0.F };
    float     m_entropyloss{ 0.F };
    float     m_valueloss{ 0.F };
    unsigned int m_checkpoint{ 0 };
    unsigned int m_version{ 0 };
    float     m_notation{ 0.F };
    float     m_evaluationprogress{ 0.F };
    bool      m_evaluationinprogress{ false };
    bool      m_done{ false };
    bool      m_training{ false };
};

struct PPOOutputs {
    float m_loss{ 0.F };
    float m_actorloss{ 0.F };
    float m_valueloss{ 0.F };
    float m_entropyloss{ 0.F };
};

class ITrainingNet : 
    public torch::nn::Module 
{
public:
    virtual ~ITrainingNet(void) = default;

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) = 0;
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(const torch::Tensor& states, const torch::Tensor& actions) = 0;
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> selectaction(const torch::Tensor& state) = 0;
};

}