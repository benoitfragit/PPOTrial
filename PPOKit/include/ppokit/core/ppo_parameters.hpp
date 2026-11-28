/**
 * @file PPOKit/include/ppokit/core/ppo_parameters.hpp
 * @brief The PPO hyper-parameters a user can set, with their ranges and defaults.
 * @details One catalog read by `train.json` (section "ppo"), the studio editor and the
 * learners, so that a name, a range or a default is written once. The values are kept
 * as numbers in catalog order; Settings() turns them into the PpoSettings a learner takes.
 *
 * @code{.json}
 * { "schema": "ppokit.train/1", "ppo": { "gamma": 0.995, "actor_learning_rate": 1e-4 } }
 * @endcode
 *
 * A parameter the file does not set takes its default. The Adam arithmetic and epsilon come
 * from the "optimizer" section of the training plan (SetAdam).
 */
#pragma once

#include "ppokit/core/ppo_settings.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ppokit::core {

/** @brief A value for every parameter, in catalog order. */
class PpoParameterValues final
{
public:
    /** @brief Every parameter at its default. */
    PpoParameterValues() noexcept
    {
        for (std::size_t index = 0; index < PpoParameterCount; ++index)
        {
            values_[index] = PpoParameterSpecs[index].defaultValue;
        }
    }

    double operator[](PpoParameter parameter) const noexcept { return values_[static_cast<std::size_t>(parameter)]; }
    /** @brief Sets a value, constrained to the parameter's range. */
    void Set(PpoParameter parameter, double value) noexcept
    {
        values_[static_cast<std::size_t>(parameter)] = Constrain(parameter, value);
    }
    bool IsDefault(PpoParameter parameter) const noexcept
    {
        return (*this)[parameter] == SpecOf(parameter).defaultValue;
    }

    /**
     * @brief Sets the Adam arithmetic and epsilon of train.json's "optimizer" section, which
     *        are not catalog numbers. Unchecked: ValidatePpoSettings checks the epsilon.
     */
    void SetAdam(AdamArithmetic arithmetic, double epsilon) noexcept
    {
        adamArithmetic_ = arithmetic;
        adamEpsilon_ = epsilon;
    }

    /** @brief Environment steps collected per update. */
    std::size_t RolloutSteps() const noexcept { return Whole(PpoParameter::RolloutSteps); }

    /** @brief The learner settings; values not in the catalog keep their PpoSettings default. */
    PpoSettings Settings() const noexcept
    {
        PpoSettings settings;
        settings.gamma = (*this)[PpoParameter::Gamma];
        settings.lambda = (*this)[PpoParameter::Lambda];
        settings.clip = (*this)[PpoParameter::Clip];
        settings.targetKl = (*this)[PpoParameter::TargetKl];
        settings.actorLearningRate = (*this)[PpoParameter::ActorLearningRate];
        settings.criticLearningRate = (*this)[PpoParameter::CriticLearningRate];
        settings.entropyCoefficient = (*this)[PpoParameter::EntropyCoefficient];
        settings.valueCoefficient = (*this)[PpoParameter::ValueCoefficient];
        settings.maxGradientNorm = (*this)[PpoParameter::MaxGradientNorm];
        settings.epochs = Whole(PpoParameter::Epochs);
        settings.minibatch = Whole(PpoParameter::Minibatch);
        settings.initialLogStd = (*this)[PpoParameter::InitialLogStd];
        settings.minimumLogStd = (*this)[PpoParameter::MinimumLogStd];
        settings.maximumLogStd = (*this)[PpoParameter::MaximumLogStd];
        settings.seed = static_cast<std::uint32_t>((*this)[PpoParameter::Seed]);
        settings.adamArithmetic = adamArithmetic_;
        settings.adamEpsilon = adamEpsilon_;
        return settings;
    }

private:
    std::size_t Whole(PpoParameter parameter) const noexcept
    {
        return static_cast<std::size_t>((*this)[parameter]);
    }

    std::array<double, PpoParameterCount> values_{};
    AdamArithmetic adamArithmetic_ = AdamArithmetic::Reference;
    double adamEpsilon_ = PpoSettings{}.adamEpsilon;
};

} // namespace ppokit::core
