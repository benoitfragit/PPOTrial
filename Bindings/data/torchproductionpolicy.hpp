#pragma once

#include "ppo/core/environment.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>

namespace ppo::model {

// Production-only LibTorch adapter. Domain agents see only IPolicy and
// owned float observations/actions. Existing Policy::load/forward semantics
// and checkpoint structure are not modified.
template<typename Policy>
class TorchProductionPolicy final : public ppo::core::IPolicy
{
public:
    TorchProductionPolicy(std::size_t observationSize, std::size_t actionSize)
        : observation_(CheckedSize(observationSize), 0.F),
          actionSize_(CheckedSize(actionSize)),
          policy_(std::make_shared<Policy>(
              static_cast<std::int64_t>(observationSize),
              static_cast<std::int64_t>(actionSize)))
    {
        // from_blob refers to observation_'s stable, owned storage.
        state_ = torch::from_blob(
            observation_.data(),
            {1, static_cast<std::int64_t>(observation_.size())}, torch::kFloat);
    }

    TorchProductionPolicy(TorchProductionPolicy const&) = delete;
    TorchProductionPolicy& operator=(TorchProductionPolicy const&) = delete;
    TorchProductionPolicy(TorchProductionPolicy&&) = delete;
    TorchProductionPolicy& operator=(TorchProductionPolicy&&) = delete;

    void Load(std::filesystem::path const& path) { policy_->load(path); }

    ppo::core::Action ChooseAction(ppo::core::Observation const& observation) override
    {
        if (observation.size() != observation_.size())
            throw std::invalid_argument("TorchProductionPolicy: observation dimension mismatch");

        std::copy(observation.begin(), observation.end(), observation_.begin());
        auto const output = policy_->forward(state_);
        if (!output.defined() || output.dim() != 2 || output.size(0) != 1
            || output.size(1) != static_cast<std::int64_t>(actionSize_)
            || output.scalar_type() != torch::kFloat || !output.is_contiguous())
            throw std::runtime_error("TorchProductionPolicy: invalid action shape or dtype");

        auto const* values = output.data_ptr<float>();
        return ppo::core::Action(values, values + actionSize_);
    }

private:
    static std::size_t CheckedSize(std::size_t size)
    {
        if (size == 0 || size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::invalid_argument("TorchProductionPolicy: invalid dimensions");
        return size;
    }

    ppo::core::Observation observation_;
    std::size_t actionSize_;
    std::shared_ptr<Policy> policy_;
    torch::Tensor state_;
};

} // namespace ppo::model
