#pragma once

// Inference-only native RLtools actor. Reads the exact weight order emitted
// by PPOBlueTrain. No optimizer, LibTorch, Python or GPU at inference time.
#include "patrol_checkpoint.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <cmath>

namespace ppo::examples::twovstwo::defenders {

namespace rlt = rl_tools;

using InferenceNetwork=AttentionActor<typename TrainingConfig::NN::ACTOR_SPEC>;
using InferenceInput = rlt::MatrixDynamic<
    rlt::matrix::Specification<float, Index, 1, ObservationLayout::Size>>;
using InferenceOutput = rlt::MatrixDynamic<
    rlt::matrix::Specification<float, Index, 1, ObservationLayout::ActionSize>>;

class PatrolPolicy
{
public:
    explicit PatrolPolicy(std::filesystem::path const& modelPath)
    {
        rlt::malloc(device_, *network_);
        rlt::malloc(device_, input_);
        rlt::malloc(device_, output_);
        rlt::malloc(device_, buffer_);
        try
        {
            Load(modelPath);
        }
        catch (...)
        {
            Free();
            throw;
        }
        ready_ = true;
        trained_ = true;
    }

    // PPO actor structure and initial weights, but no optimization has occurred.
    // Use only for live visual inspection until a real checkpoint is trained.
    explicit PatrolPolicy(std::uint32_t seed)
    {
        rlt::malloc(device_, *network_);
        rlt::malloc(device_, input_);
        rlt::malloc(device_, output_);
        rlt::malloc(device_, buffer_);
        try
        {
            auto rng = rlt::random::default_engine(
                typename Device::SPEC::RANDOM{}, seed);
            rlt::init_weights(device_, *network_, rng);
        }
        catch (...)
        {
            Free();
            throw;
        }
        ready_ = true;
    }

    bool IsTrained() const noexcept { return trained_; }

    PatrolPolicy(PatrolPolicy const&) = delete;
    PatrolPolicy& operator=(PatrolPolicy const&) = delete;
    ~PatrolPolicy() { if (ready_) Free(); }

    // The environment thread alone owns inference and safely replaces its
    // weights from an immutable serialized snapshot of the learner.
    void ReloadFromText(std::string const& serialized)
    {
        std::istringstream in{serialized};
        Load(in);
        trained_ = true;
    }

    Action Predict(Observation const& observation)
    {
        for (Index i = 0; i < ObservationLayout::Size; ++i)
        {
            if (!std::isfinite(observation[i]))
                throw std::invalid_argument("Nonfinite blue MAPPO observation");
            rlt::set(input_, 0, i, observation[i]);
        }
        rlt::evaluate(device_, *network_, input_, output_, buffer_);
        Action action{};
        for (Index i = 0; i < ObservationLayout::ActionSize; ++i)
        {
            float const value = rlt::get(output_, 0, i);
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite blue MAPPO action");
            action[i] = Clamp(value, -1.F, 1.F);
        }
        return action;
    }

private:
    void Load(std::filesystem::path const& path)
    {
        std::ifstream stream{path};
        if (!stream)
            throw std::runtime_error("Cannot open blue MAPPO checkpoint: " + path.string());
        Load(stream);
    }

    void Load(std::istream& stream)
    {
        ReadPatrolPolicy(stream, *network_);
    }

    void Free()
    {
        rlt::free(device_, *network_);
        rlt::free(device_, input_);
        rlt::free(device_, output_);
        rlt::free(device_, buffer_);
        ready_ = false;
    }

    Device device_{};
    std::unique_ptr<InferenceNetwork> network_{new InferenceNetwork};
    InferenceInput input_{};
    InferenceOutput output_{};
    InferenceNetwork::Buffer<1> buffer_{};
    bool ready_ = false;
    bool trained_ = false;
};
} // namespace ppo::examples::twovstwo::defenders
