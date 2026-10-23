#pragma once

#include "ppo/core/environment.hpp"
#include "rlagent.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ppo::model {

// Generic policy adapter: any ITrainingNet works with any ppo::core::IEnvironment
// exposing the matching flattened float observation and action dimensions.
// The adapter owns fixed-size buffers for RLAgent's torch::from_blob view.
// It is deliberately non-movable to preserve tensor aliases.
class TorchLearner final : public ppo::core::ILearner
{
public:
    TorchLearner(std::shared_ptr<ITrainingNet> policy,
                 std::string const& modelpath,
                 std::size_t observation_size,
                 std::size_t action_size,
                 std::int64_t rollout_size,
                 PPOParamters const& parameters,
                 bool load_existing_model = true);

    TorchLearner(TorchLearner const&) = delete;
    TorchLearner& operator=(TorchLearner const&) = delete;
    TorchLearner(TorchLearner&&) = delete;
    TorchLearner& operator=(TorchLearner&&) = delete;

    ppo::core::Action ChooseAction(ppo::core::Observation const& observation) override;
    void Observe(ppo::core::Transition const& transition) override;

    // Reset only the rollout collector, not the environment or policy weights.
    // Call at the same boundaries as the legacy RLAgent::Init().
    void ResetRollout();
    bool RolloutComplete() const noexcept;
    long long RolloutStep() const noexcept;
    long long RolloutSize() const noexcept { return agent_.GetMaxStep(); }
    PPOOutputs const& Outputs() const noexcept;
    void Save(std::filesystem::path const& path);

private:
    std::vector<float> observation_;
    std::vector<float> action_;
    RLAgent agent_;
    bool pending_action_ = false;
};

} // namespace ppo::model
