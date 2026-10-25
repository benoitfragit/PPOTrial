#pragma once

// Inference-only native RLtools actor. Reads the exact weight order emitted
// by PPOBlueTrain. No optimizer, LibTorch, Python or GPU at inference time.
#include "patrol_rltools.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
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

    Action Predict(Observation const& observation)
    {
        for (Index i = 0; i < ObservationLayout::Size; ++i)
        {
            if (!std::isfinite(observation[i]))
                throw std::invalid_argument("Nonfinite Blue PPO observation");
            rlt::set(input_, 0, i, observation[i]);
        }
        rlt::evaluate(device_, *network_, input_, output_, buffer_);
        Action action{};
        for (Index i = 0; i < ObservationLayout::ActionSize; ++i)
        {
            float const value = rlt::get(output_, 0, i);
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite Blue PPO action");
            action[i] = Clamp(value, -1.F, 1.F);
        }
        return action;
    }

private:
    template<typename MATRIX>
    static void ReadMatrix(std::istream& stream, MATRIX& matrix)
    {
        for (typename MATRIX::SPEC::TI i = 0; i < MATRIX::SPEC::ROWS; ++i)
            for (typename MATRIX::SPEC::TI j = 0; j < MATRIX::SPEC::COLS; ++j)
            {
                float value = 0.F;
                if (!(stream >> value) || !std::isfinite(value))
                    throw std::runtime_error("Invalid Blue PPO weight matrix");
                rlt::set(matrix, i, j, value);
            }
    }

    template<typename LAYER>
    static void ReadLayer(std::istream& stream, LAYER& layer)
    {
        ReadMatrix(stream, layer.weights.parameters);
        ReadMatrix(stream, layer.biases.parameters);
    }

    template<typename Block>
    static void ReadAttention(std::istream& in,Block& b)
    {
        ReadMatrix(in,b.encoderW.parameters);
        ReadMatrix(in,b.encoderB.parameters);
        ReadMatrix(in,b.queryW.parameters);
        ReadMatrix(in,b.queryB.parameters);
    }

    void Load(std::filesystem::path const& path)
    {
        std::ifstream stream{path};
        if (!stream)
            throw std::runtime_error("Cannot open Blue PPO checkpoint: " + path.string());
        std::string magic;
        unsigned version = 0U;
        unsigned input = 0U, output = 0U, hidden = 0U, layers = 0U;
        if (!(stream >> magic >> version >> input >> output >> hidden >> layers)
            || magic != "BLUE_PATROL_PPO_V2"
            || version != ObservationLayout::Version
            || input != ObservationLayout::Size || output != ObservationLayout::ActionSize
            || hidden != TrainingParameters::ACTOR_HIDDEN_DIM
            || layers != TrainingParameters::ACTOR_NUM_LAYERS)
            throw std::runtime_error("Incompatible Blue PPO checkpoint contract");

        ReadAttention(stream,network_->red);
        ReadAttention(stream,network_->friendlies);
        ReadLayer(stream, network_->input_layer);
        for (auto& layer : network_->hidden_layers)
            ReadLayer(stream, layer);
        ReadLayer(stream, network_->output_layer);
        // PPO exports stochastic exploration std too, but inference is
        // deliberately deterministic: only action means are evaluated.
        for (std::size_t i = 0; i < ObservationLayout::ActionSize; ++i)
        {
            float ignored = 0.F;
            if (!(stream >> ignored) || !std::isfinite(ignored))
                throw std::runtime_error("Invalid PPO exploration parameters");
        }
        std::string unexpected;
        if (stream >> unexpected)
            throw std::runtime_error("Trailing data in Blue PPO checkpoint");
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
