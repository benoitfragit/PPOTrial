#pragma once
#include <torch/torch.h>
#include <memory>
#include <tuple>
#include <filesystem>

#include "trainingnet.hpp"

namespace ppo::model {

/**
 * @brief Proximal Policy Optimization training engine.
 */
class PPO {
public:
                    PPO(std::shared_ptr<ITrainingNet> policy, std::string const& modelpath, int64_t state_dim, int64_t action_dim, PPOParamters const& parameters);

                    ~PPO(void);

                    PPO(PPO const&) = delete;
    PPO&            operator=(PPO const&) = delete;
                    PPO(PPO&&) = delete;
    PPO&            operator=(PPO&&) = delete;

    /**
     * @brief Performs one PPO optimization pass on collected rollout tensors.
     */
    void            UpdatePolicy(
                        const torch::Tensor& states,
                        const torch::Tensor& actions,
                        const torch::Tensor& old_logp,
                        const torch::Tensor& values,
                        const torch::Tensor& rewards,
                        const torch::Tensor& dones,
                        const torch::Tensor& last_values);

    /**
     * @brief Samples an action from the policy for one state tensor.
     * @param state Input observation tensor.
     * @return Tuple containing action, log-probability and value estimate.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                    SelectAction(const torch::Tensor& state);

    /**
     * @brief Saves policy and optimizer state to disk.
     * @param modelpath Destination path.
     */
    void              Save(std::filesystem::path const& modelpath);

    /** @brief Returns latest loss metrics produced by UpdatePolicy(). */
    PPOOutputs const& GetOutputs(void) const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::model */