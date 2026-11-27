/**
 * @file PPOKit/include/ppokit/backend/rltools/mlp_probes.hpp
 * @brief Probes on the dense layers of an rl_tools MLP: what each layer outputs, and how
 *        much gradient reaches it, during a PPO update.
 * @details The same statistics as the graph probes (ppokit/graph/probes.hpp), for the
 * learners that train rl_tools networks (FlightControl3D, Blue MAPPO). Read-only: the
 * probes read the layer outputs kept by a training forward and the parameter gradients
 * before an optimizer step; they change no value and use no random number, so a probed
 * update is bit-identical to an unprobed one. Nothing is allocated while observing.
 *
 * @code
 * MlpProbes actorProbes;                      // once
 * actorProbes.Reset();                        // before a probed update
 * rlt::forward(device, actor, input, output);
 * actorProbes.ObserveForward(actor, rows);    // after each training forward
 * actorProbes.ObserveGradients(actor);        // before each optimizer step
 * journal.WriteProbes({update, "actor", actorProbes.Summary()});
 * @endcode
 *
 * Include the rl_tools headers before this one.
 */
#pragma once

#ifndef RL_TOOLS_RL_TOOLS_H
#error "Include the rl_tools operation headers before mlp_probes.hpp"
#endif

#include "ppokit/runtime/probe_moments.hpp"
#include "ppokit/runtime/probe_record.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace ppokit::backend::rltools {

namespace rlt_probes = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

/** @brief Readable block type of a dense layer, e.g. "dense tanh". */
template<typename Layer>
constexpr char const* DenseLayerType() noexcept
{
    using Activation = rlt_probes::nn::activation_functions::ActivationFunction;
    constexpr Activation activation = Layer::SPEC::ACTIVATION_FUNCTION;
    if constexpr (activation == Activation::RELU)
    {
        return "dense relu";
    }
    else if constexpr (activation == Activation::TANH || activation == Activation::FAST_TANH)
    {
        return "dense tanh";
    }
    else if constexpr (activation == Activation::SIGMOID)
    {
        return "dense sigmoid";
    }
    else if constexpr (activation == Activation::GELU)
    {
        return "dense gelu";
    }
    else
    {
        return "dense linear";
    }
}

/**
 * @brief Probes of the layers of one rl_tools MLP (input layer, hidden layers, output layer).
 * @tparam Mlp An rl_tools `nn_models::mlp` network (or a type derived from one) whose layers
 *             keep their outputs (backward-gradient capability: Adam training networks do).
 */
template<typename Mlp>
class MlpProbes final
{
public:
    static constexpr std::size_t HiddenLayers = Mlp::NUM_HIDDEN_LAYERS;
    /** @brief Input layer, hidden layers, output layer. */
    static constexpr std::size_t Layers = HiddenLayers + 2;

    MlpProbes() { Reset(); }

    /** @brief Forgets what was observed, before a new update. */
    void Reset() noexcept
    {
        for (Layer& layer : layers_)
        {
            layer.values = runtime::ProbeMoments{};
            layer.gradientNormSum = 0.0;
            layer.steps = 0;
        }
    }

    /** @brief Reads the outputs of every layer for the first `rows` rows of the last training forward. */
    void ObserveForward(Mlp const& mlp, std::size_t rows) noexcept
    {
        ObserveOutputs(mlp.input_layer, rows, layers_[0]);
        for (std::size_t hidden = 0; hidden < HiddenLayers; ++hidden)
        {
            ObserveOutputs(mlp.hidden_layers[hidden], rows, layers_[1 + hidden]);
        }
        ObserveOutputs(mlp.output_layer, rows, layers_[Layers - 1]);
    }

    /** @brief Reads the weight and bias gradients of every layer, once per optimizer step (before it). */
    void ObserveGradients(Mlp const& mlp) noexcept
    {
        ObserveGradient(mlp.input_layer, layers_[0]);
        for (std::size_t hidden = 0; hidden < HiddenLayers; ++hidden)
        {
            ObserveGradient(mlp.hidden_layers[hidden], layers_[1 + hidden]);
        }
        ObserveGradient(mlp.output_layer, layers_[Layers - 1]);
    }

    /** @brief What was observed since Reset: "layer 1" ... "layer N", then "output". */
    std::vector<runtime::BlockProbe> Summary() const
    {
        std::vector<runtime::BlockProbe> summary;
        summary.reserve(Layers);
        for (std::size_t index = 0; index < Layers; ++index)
        {
            runtime::BlockProbe probe;
            probe.block = index + 1 == Layers ? std::string("output") : "layer " + std::to_string(index + 1);
            probe.type = LayerType(index);
            layers_[index].values.Fill(probe);
            probe.hasGradient = true;
            if (layers_[index].steps > 0)
            {
                probe.gradientNorm = layers_[index].gradientNormSum / static_cast<double>(layers_[index].steps);
            }
            summary.push_back(std::move(probe));
        }
        return summary;
    }

private:
    struct Layer final
    {
        runtime::ProbeMoments values;
        double gradientNormSum = 0.0;
        std::size_t steps = 0;
    };

    template<typename DenseLayer>
    static void ObserveOutputs(DenseLayer const& layer, std::size_t rows, Layer& probe) noexcept
    {
        using Output = std::decay_t<decltype(layer.output)>;
        std::size_t const kept = rows < static_cast<std::size_t>(Output::ROWS) ? rows
                                                                              : static_cast<std::size_t>(Output::ROWS);
        for (std::size_t row = 0; row < kept; ++row)
        {
            for (std::size_t column = 0; column < static_cast<std::size_t>(Output::COLS); ++column)
            {
                probe.values.Add(static_cast<double>(rlt_probes::get(layer.output,
                    static_cast<typename Output::TI>(row), static_cast<typename Output::TI>(column))));
            }
        }
    }

    template<typename Matrix>
    static double SquaredNorm(Matrix const& matrix) noexcept
    {
        using Type = std::decay_t<Matrix>;
        double squares = 0.0;
        for (std::size_t row = 0; row < static_cast<std::size_t>(Type::ROWS); ++row)
        {
            for (std::size_t column = 0; column < static_cast<std::size_t>(Type::COLS); ++column)
            {
                double const value = rlt_probes::get(matrix, static_cast<typename Type::TI>(row),
                    static_cast<typename Type::TI>(column));
                squares += value * value;
            }
        }
        return squares;
    }

    template<typename DenseLayer>
    static void ObserveGradient(DenseLayer const& layer, Layer& probe) noexcept
    {
        double const squares = SquaredNorm(layer.weights.gradient) + SquaredNorm(layer.biases.gradient);
        probe.gradientNormSum += std::sqrt(squares);
        ++probe.steps;
    }

    static char const* LayerType(std::size_t index) noexcept
    {
        if (index == 0)
        {
            return DenseLayerType<std::decay_t<decltype(std::declval<Mlp const&>().input_layer)>>();
        }
        if (index + 1 == Layers)
        {
            return DenseLayerType<std::decay_t<decltype(std::declval<Mlp const&>().output_layer)>>();
        }
        return DenseLayerType<std::decay_t<decltype(std::declval<Mlp const&>().hidden_layers[0])>>();
    }

    std::array<Layer, Layers> layers_{};
};

} // namespace ppokit::backend::rltools
