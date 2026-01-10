#pragma once
#include <torch/torch.h>
#include <memory>
#include <tuple>
#include <filesystem>

#include "trainingnet.hpp"

namespace ppo::model {

class PPO {
public:
                    PPO(std::shared_ptr<ITrainingNet> policy, std::string const& modelpath, int64_t state_dim, int64_t action_dim, PPOParamters const& parameters);

                    ~PPO(void);

                    PPO(PPO const&) = delete;
    PPO&            operator=(PPO const&) = delete;
                    PPO(PPO&&) = delete;
    PPO&            operator=(PPO&&) = delete;

    void            UpdatePolicy(
                        const torch::Tensor& states,
                        const torch::Tensor& actions,
                        const torch::Tensor& old_logp,
                        const torch::Tensor& values,
                        const torch::Tensor& rewards,
                        const torch::Tensor& dones,
                        const torch::Tensor& last_values);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                    SelectAction(const torch::Tensor& state);

    void              Save(std::filesystem::path const& modelpath);

    PPOOutputs const& GetOutputs(void) const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::model */