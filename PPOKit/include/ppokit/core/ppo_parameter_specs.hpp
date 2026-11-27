/**
 * @file ppo_parameter_specs.hpp
 * @brief Numerical domain shared by the editor, JSON parser and native learners.
 */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string_view>

namespace ppokit::core {

/** @brief The parameters, in catalog (and display) order. */
enum class PpoParameter : std::size_t
{
    Gamma,
    Lambda,
    RolloutSteps,
    Minibatch,
    Epochs,
    ActorLearningRate,
    CriticLearningRate,
    Clip,
    TargetKl,
    EntropyCoefficient,
    ValueCoefficient,
    MaxGradientNorm,
    InitialLogStd,
    MinimumLogStd,
    MaximumLogStd,
    Seed,
    Count
};

inline constexpr std::size_t PpoParameterCount = static_cast<std::size_t>(PpoParameter::Count);

/** @brief Name, group, range and default of one parameter. */
struct PpoParameterSpec final
{
    /** @brief Key in train.json and in the run journal. */
    char const* name;
    /** @brief Section of the editor that shows it. */
    char const* group;
    double minimum;
    double maximum;
    double defaultValue;
    /** @brief Whole numbers only. */
    bool integral;
    /** @brief Edited on a logarithmic scale (learning rates). */
    bool logarithmic;
};

/** @brief The catalog, in PpoParameter order; defaults are those of PpoSettings. */
inline constexpr std::array<PpoParameterSpec, PpoParameterCount> PpoParameterSpecs{{
    {"gamma", "Returns", 0.0, 1.0, 0.99, false, false},
    {"lambda", "Returns", 0.0, 1.0, 0.95, false, false},
    {"rollout_steps", "Data per update", 1.0, 1048576.0, 2048.0, true, false},
    {"minibatch", "Data per update", 1.0, 65536.0, 64.0, true, false},
    {"epochs", "Data per update", 1.0, 100.0, 4.0, true, false},
    {"actor_learning_rate", "Step size", 1e-7, 1e-1, 3e-4, false, true},
    {"critic_learning_rate", "Step size", 1e-7, 1e-1, 1e-3, false, true},
    {"clip", "Step size", 0.01, 1.0, 0.2, false, false},
    {"target_kl", "Step size", 0.0, 1.0, 0.0, false, false},
    {"entropy_coefficient", "Exploration", 0.0, 0.5, 0.0, false, false},
    {"value_coefficient", "Step size", 0.01, 10.0, 0.5, false, false},
    {"max_gradient_norm", "Step size", 0.0, 100.0, 0.0, false, false},
    {"initial_log_std", "Exploration", -5.0, 2.0, -0.5, false, false},
    {"minimum_log_std", "Exploration", -10.0, 2.0, -5.0, false, false},
    {"maximum_log_std", "Exploration", -10.0, 5.0, 2.0, false, false},
    {"seed", "Reproducibility", 0.0, 4294967295.0, 1.0, true, false},
}};

/** @brief Spec of a parameter. */
constexpr PpoParameterSpec const& SpecOf(PpoParameter parameter) noexcept
{
    return PpoParameterSpecs[static_cast<std::size_t>(parameter)];
}

/** @brief The parameter named `name` in train.json, if any. */
inline std::optional<PpoParameter> FindPpoParameter(std::string_view name) noexcept
{
    for (std::size_t index = 0; index < PpoParameterCount; ++index)
    {
        if (name == PpoParameterSpecs[index].name)
        {
            return static_cast<PpoParameter>(index);
        }
    }
    return std::nullopt;
}

/** @brief Whether `value` is accepted for `parameter` (finite, in range, whole when integral). */
inline bool AcceptsValue(PpoParameter parameter, double value) noexcept
{
    PpoParameterSpec const& spec = SpecOf(parameter);
    return std::isfinite(value) && value >= spec.minimum && value <= spec.maximum
        && (!spec.integral || std::floor(value) == value);
}

/** @brief `value` brought into the range of `parameter`, rounded when integral. */
inline double Constrain(PpoParameter parameter, double value) noexcept
{
    PpoParameterSpec const& spec = SpecOf(parameter);
    if (!std::isfinite(value))
    {
        return spec.defaultValue;
    }
    double const rounded = spec.integral ? std::round(value) : value;
    return rounded < spec.minimum ? spec.minimum : (rounded > spec.maximum ? spec.maximum : rounded);
}

} // namespace ppokit::core
