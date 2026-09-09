#pragma once
#include "ppo/rl/runtime.hpp"
#include "ppo_parameters.hpp"
#include <memory>

namespace ppo::rl {
/// CPU float32 continuous-action backend reusing the extracted PPO/RLAgent/network.
/// No implicit checkpoint restore. Caller must isolate checkpoints by environment/schema.
class TorchLearner final : public ILearner
{
public:
    TorchLearner(std::size_t observation_size, std::size_t action_size, std::size_t rollout_size,
                 const std::filesystem::path& checkpoint_root, const model::PPOParamters& parameters = {});
    ~TorchLearner() override;
    Value act(const Value& observation) override;
    void observe(const StepResult& transition) override;
    /// Saves an eval/model.pt candidate using the existing archive layout.
    void save_candidate(const std::filesystem::path& path);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
/// Independently loaded network, deterministic mean actions, no optimizer or PPO updates.
class FrozenTorchPolicy final : public IPolicy
{
public:
    FrozenTorchPolicy(std::size_t observation_size, std::size_t action_size,
                      const std::filesystem::path& checkpoint);
    ~FrozenTorchPolicy() override;
    Value act(const Value& observation) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ppo::rl
