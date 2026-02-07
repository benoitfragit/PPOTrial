#pragma once

#include "iagent.hpp"

#include <torch/torch.h>
#include <tuple>
#include <string>

namespace ppo::model {

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