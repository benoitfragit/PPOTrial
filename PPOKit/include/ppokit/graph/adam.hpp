/**
 * @file PPOKit/include/ppokit/graph/adam.hpp
 * @brief Adam optimiser over the parameter tensors of a PPOKit graph.
 * @details Moments are allocated once for the tensors given at construction; Step never
 * allocates. The tensor list must stay the same between steps (Graph::Parameters returns
 * it in a stable order).
 */
#pragma once

#include "ppokit/core/ppo_settings.hpp"
#include "ppokit/graph/blocks.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppokit::graph {

/** @brief Adam hyper-parameters (Kingma and Ba, 2015). */
struct AdamSettings final
{
    float learningRate = 3e-4F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float epsilon = 1e-8F;
    /** @brief Order of the floating-point operations of a step (see core::AdamArithmetic). */
    core::AdamArithmetic arithmetic = core::AdamArithmetic::Reference;
};

/** @brief Adam of a learner: `learningRate`, and the arithmetic and epsilon of its settings. */
inline AdamSettings AdamFor(core::PpoSettings const& settings, double learningRate) noexcept
{
    AdamSettings adam;
    adam.learningRate = static_cast<float>(learningRate);
    adam.epsilon = static_cast<float>(settings.adamEpsilon);
    adam.arithmetic = settings.adamArithmetic;
    return adam;
}

/** @brief Adam with bias correction; descends the accumulated gradients. */
class Adam final
{
public:
    /** @brief Added to the norm before dividing, as torch's clip_grad_norm_. */
    static constexpr double ClipNormEpsilon = 1e-6;

    Adam(std::vector<Parameter*> parameters, AdamSettings settings)
        : parameters_(std::move(parameters)), settings_(settings)
    {
        for (Parameter const* parameter : parameters_)
        {
            firstMoments_.emplace_back(parameter->value.size(), 0.F);
            secondMoments_.emplace_back(parameter->value.size(), 0.F);
            next_.emplace_back(parameter->value.size());
        }
    }

    std::size_t StepCount() const noexcept { return steps_; }

    /** @brief Moments and step count, everything a resumed training needs. */
    struct State final
    {
        std::size_t steps = 0;
        std::vector<std::vector<float>> first;
        std::vector<std::vector<float>> second;
    };
    State Save() const { return {steps_, firstMoments_, secondMoments_}; }

    /** @brief Reusable in-memory copy of one optimizer and its weights at the start of a PPO update. */
    class UpdateBackup final
    {
    public:
        /** @brief Capture before the first minibatch; subsequent updates reuse the same buffers. */
        void Capture(Adam& optimizer)
        {
            weights_.resize(optimizer.parameters_.size());
            for (std::size_t index = 0; index < weights_.size(); ++index)
            {
                weights_[index] = optimizer.parameters_[index]->value;
            }
            state_.first = optimizer.firstMoments_;
            state_.second = optimizer.secondMoments_;
            state_.steps = optimizer.steps_;
            optimizer_ = &optimizer;
        }

        /** @brief Restore the captured optimizer once, without allocation or checkpoint I/O. */
        void Restore() noexcept
        {
            if (optimizer_ == nullptr)
            {
                return;
            }
            for (std::size_t index = 0; index < weights_.size(); ++index)
            {
                auto& values = optimizer_->parameters_[index]->value;
                std::copy(weights_[index].begin(), weights_[index].end(), values.begin());
            }
            optimizer_->firstMoments_.swap(state_.first);
            optimizer_->secondMoments_.swap(state_.second);
            optimizer_->steps_ = state_.steps;
            optimizer_ = nullptr;
        }

    private:
        Adam* optimizer_ = nullptr;
        State state_;
        std::vector<std::vector<float>> weights_;
    };

    /**
     * @brief Restores a saved state.
     * @throws std::invalid_argument when its tensors do not match this optimizer's.
     */
    void ValidateState(State const& state) const
    {
        if (state.steps == std::numeric_limits<std::size_t>::max() || state.first.size() != parameters_.size()
            || state.second.size() != parameters_.size())
        {
            throw std::invalid_argument("PPOKit Adam: invalid step count or tensor count");
        }
        for (std::size_t tensor = 0; tensor < parameters_.size(); ++tensor)
        {
            auto const& first = state.first[tensor];
            auto const& second = state.second[tensor];
            if (first.size() != firstMoments_[tensor].size() || second.size() != secondMoments_[tensor].size()
                || !std::all_of(first.begin(), first.end(), [](float m) { return std::isfinite(m); })
                || !std::all_of(second.begin(), second.end(), [](float v) { return std::isfinite(v) && v >= 0.F; }))
            {
                throw std::invalid_argument("PPOKit Adam: invalid moment tensor " + std::to_string(tensor));
            }
        }
    }

    /** @brief Installs both sets of moments only once the entire state passes validation. */
    void Restore(State state)
    {
        ValidateState(state);
        steps_ = state.steps;
        firstMoments_ = std::move(state.first);
        secondMoments_ = std::move(state.second);
    }

    /**
     * @brief Scales the gradients of this optimizer's tensors so that their global L2 norm
     *        is at most `maximumNorm`, as torch::nn::utils::clip_grad_norm_ (factor
     *        maximumNorm / (norm + 1e-6), applied only below 1).
     * @return The norm before clipping.
     */
    double ClipGradientNorm(double maximumNorm)
    {
        double squares = 0.0;
        for (Parameter const* parameter : parameters_)
        {
            for (float const gradient : parameter->gradient)
            {
                squares += static_cast<double>(gradient) * static_cast<double>(gradient);
            }
        }
        double const norm = std::sqrt(squares);
        if (!std::isfinite(norm) || !std::isfinite(maximumNorm) || maximumNorm < 0.0)
        {
            throw std::runtime_error("PPOKit Adam: invalid gradient norm");
        }
        double const factor = maximumNorm / (norm + ClipNormEpsilon);
        if (factor >= 1.0)
        {
            return norm;
        }
        float const scale = static_cast<float>(factor);
        for (Parameter* parameter : parameters_)
        {
            for (float& gradient : parameter->gradient)
            {
                gradient *= scale;
            }
        }
        return norm;
    }

    /** @brief One update of every parameter from its gradient; gradients are left untouched. */
    void Step()
    {
        if (steps_ == std::numeric_limits<std::size_t>::max())
        {
            throw std::overflow_error("PPOKit Adam: exhausted step counter");
        }
        float const age = static_cast<float>(steps_ + 1);
        float const firstCorrection = 1.F - std::pow(settings_.beta1, age);
        float const secondCorrection = 1.F - std::pow(settings_.beta2, age);
        Corrections const corrections{firstCorrection, secondCorrection, 1.F / firstCorrection,
            1.F / secondCorrection};
        // Stage the whole optimizer step. A late overflow cannot publish an earlier tensor.
        for (std::size_t tensor = 0; tensor < parameters_.size(); ++tensor)
        {
            Parameter const& parameter = *parameters_[tensor];
            if (parameter.gradient.size() != next_[tensor].size() || parameter.value.size() != next_[tensor].size())
            {
                throw std::runtime_error("PPOKit Adam: parameter or gradient changed shape");
            }
            for (std::size_t index = 0; index < parameter.value.size(); ++index)
            {
                float const gradient = parameter.gradient[index];
                Candidate& next = next_[tensor][index];
                next.first = settings_.beta1 * firstMoments_[tensor][index] + (1.F - settings_.beta1) * gradient;
                next.second = settings_.beta2 * secondMoments_[tensor][index]
                    + (1.F - settings_.beta2) * gradient * gradient;
                float const update = Update(next.first, next.second, corrections);
                next.value = parameter.value[index] - update;
                if (!std::isfinite(next.first) || !std::isfinite(next.second) || next.second < 0.F
                    || !std::isfinite(next.first / corrections.first)
                    || !std::isfinite(next.second / corrections.second) || !std::isfinite(update)
                    || !std::isfinite(next.value))
                {
                    throw std::runtime_error("PPOKit Adam: non-finite optimizer step");
                }
            }
        }
        for (std::size_t tensor = 0; tensor < parameters_.size(); ++tensor)
        {
            for (std::size_t index = 0; index < next_[tensor].size(); ++index)
            {
                Candidate const& next = next_[tensor][index];
                firstMoments_[tensor][index] = next.first;
                secondMoments_[tensor][index] = next.second;
                parameters_[tensor]->value[index] = next.value;
            }
        }
        ++steps_;
    }

private:
    /** @brief 1 - beta^t of both moments, and their reciprocals. */
    struct Corrections final
    {
        float first;
        float second;
        float firstScale;
        float secondScale;
    };

    /**
     * @brief Size of the descent of one parameter from its new moments.
     * @details Reference: lr * (m / c1) / (sqrt(v / c2) + eps). RlTools, as
     * rl_tools::nn::optimizers::Adam: (lr * (1 / c1)) * m / (sqrt(v * (1 / c2)) + eps).
     */
    float Update(float first, float second, Corrections const& corrections) const noexcept
    {
        if (settings_.arithmetic == core::AdamArithmetic::RlTools)
        {
            return settings_.learningRate * corrections.firstScale * first
                / (std::sqrt(second * corrections.secondScale) + settings_.epsilon);
        }
        float const firstUnbiased = first / corrections.first;
        float const secondUnbiased = second / corrections.second;
        return settings_.learningRate * firstUnbiased / (std::sqrt(secondUnbiased) + settings_.epsilon);
    }

    struct Candidate final
    {
        float value;
        float first;
        float second;
    };
    std::vector<std::vector<Candidate>> next_;
    std::vector<Parameter*> parameters_;
    AdamSettings settings_;
    std::vector<std::vector<float>> firstMoments_;
    std::vector<std::vector<float>> secondMoments_;
    std::size_t steps_ = 0;
};

} // namespace ppokit::graph
