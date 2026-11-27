/**
 * @file PPOKit/include/ppokit/component/component_node.hpp
 * @brief A DLL component as a block of a PPOKit graph.
 * @details The node exposes the float "in" ports of the manifest as graph ports; a bool
 * mask port is bound to the presence mask of the set it masks, so it is never wired by
 * hand. Symbolic dimensions ("N") take their value from the wired shapes, once, at
 * construction. Version 1 of the graph binding takes one float output (vector [n], or a
 * set [N, n] following an input mask) and refuses stateful components: their outputs
 * would need to be stored in the rollout to keep the PPO ratio valid.
 *
 * Every buffer is allocated here at construction; Forward and Backward allocate nothing.
 * A trainable component's parameters are mirrored in a graph Parameter, so PPOKit's
 * optimizer and checkpoints treat them like any other block's.
 */
#pragma once

#include "ppokit/component/library.hpp"
#include "ppokit/graph/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::component {

class ComponentNode final : public graph::Node
{
public:
    /**
     * @param inputs Shapes wired to the float input ports, in manifest order.
     * @param seed Seed of the instance (its initial parameters).
     * @throws std::invalid_argument naming the port when the wiring does not fit the manifest.
     */
    ComponentNode(ComponentRef component, std::string configJson, std::vector<graph::Shape> const& inputs,
        std::uint64_t seed)
        : component_(std::move(component)), config_(std::move(configJson))
    {
        Manifest const& manifest = *component_.manifest;
        if (manifest.capabilities.stateful)
        {
            throw std::invalid_argument("component " + component_.Id()
                + " is stateful: graphs do not take stateful components yet (their outputs would have to be stored "
                  "in the rollout); use it outside the network");
        }
        SplitPorts(manifest);
        BindSymbols(inputs);
        ResolveOutput();
        instance_ = std::make_unique<Instance>(component_, config_, SymbolsJson(), seed);
        AllocateBuffers();
        if (manifest.capabilities.trainable)
        {
            std::size_t count = 0;
            float const* const values = instance_->Parameters(count);
            parameters_ = graph::Parameter(count);
            std::copy(values, values + count, parameters_.value.begin());
        }
    }

    char const* Type() const noexcept override { return "component"; }
    std::vector<std::string> InputPorts() const override
    {
        std::vector<std::string> names;
        for (PortSpec const* port : dataInputs_)
        {
            names.push_back(port->name);
        }
        return names;
    }
    std::vector<graph::Shape> InputShapes() const override { return inputShapes_; }
    graph::Shape OutputShape() const override { return outputShape_; }

    graph::Signal Forward(graph::Signal const* inputs) override
    {
        BindInputs(inputs);
        if (component_.manifest->capabilities.trainable)
        {
            // The optimizer updates the mirror: hand the current values to the component.
            std::size_t count = 0;
            float* const values = instance_->Parameters(count);
            std::copy(parameters_.value.begin(), parameters_.value.end(), values);
        }
        instance_->Forward(1, inputBuffers_.data(), &outputBuffer_, cache_.empty() ? nullptr : cache_.data());
        std::uint8_t const* mask = outputMaskInput_ < dataInputs_.size() ? inputs[outputMaskInput_].mask : nullptr;
        return {output_.data(), mask};
    }

    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        // A component without gradient sits before every trainable block (checked when
        // the graph is read): nothing upstream needs a gradient from it.
        if (!component_.manifest->capabilities.differentiable)
        {
            return;
        }
        for (auto& gradient : inputGradients_)
        {
            std::fill(gradient.begin(), gradient.end(), 0.F);
        }
        PpokitBuffer outputGradientBuffer = outputBuffer_;
        outputGradientBuffer.data = const_cast<float*>(outputGradient);
        float* const parameterGradients = parameters_.gradient.empty() ? nullptr : parameters_.gradient.data();
        instance_->Backward(1, &outputGradientBuffer, cache_.empty() ? nullptr : cache_.data(),
            gradientBuffers_.data(), parameterGradients);
        for (std::size_t port = 0; port < dataInputs_.size(); ++port)
        {
            std::vector<float> const& gradient = inputGradients_[port];
            for (std::size_t index = 0; index < gradient.size(); ++index)
            {
                inputGradients[port][index] += gradient[index];
            }
        }
    }

    void CollectParameters(std::vector<graph::Parameter*>& parameters) override
    {
        if (!parameters_.value.empty())
        {
            parameters.push_back(&parameters_);
        }
    }

    ComponentRef const& Component() const noexcept { return component_; }
    std::string const& Config() const noexcept { return config_; }

private:
    /** @brief Float inputs (graph ports) and bool masks (bound from the sets), in manifest order. */
    void SplitPorts(Manifest const& manifest)
    {
        std::vector<PortSpec const*> outputs;
        for (PortSpec const& port : manifest.ports)
        {
            if (port.output)
            {
                outputs.push_back(&port);
                continue;
            }
            allInputs_.push_back(&port);
            if (port.dtype == Dtype::F32)
            {
                dataInputs_.push_back(&port);
            }
            else if (port.dtype != Dtype::Bool)
            {
                throw std::invalid_argument("component " + component_.Id() + ", port '" + port.name
                    + "': graphs pass float values and bool masks; other types are not supported yet");
            }
        }
        if (outputs.size() != 1 || outputs[0]->dtype != Dtype::F32)
        {
            throw std::invalid_argument("component " + component_.Id()
                + ": graphs take components with exactly one float output for now");
        }
        outputPort_ = outputs[0];
        if (dataInputs_.empty())
        {
            throw std::invalid_argument("component " + component_.Id() + " has no float input to wire");
        }
        // A mask is bound from the set it masks: one nobody names could never be filled.
        for (PortSpec const* port : allInputs_)
        {
            bool const named = std::any_of(dataInputs_.begin(), dataInputs_.end(),
                [port](PortSpec const* data) { return data->mask == port->name; });
            if (port->dtype == Dtype::Bool && !named)
            {
                throw std::invalid_argument("component " + component_.Id() + ", mask port '" + port->name
                    + "': no set input names it as its \"mask\"");
            }
        }
    }

    /** @brief Fixes every symbol from the wired shapes; a conflict names both ports. */
    void BindSymbols(std::vector<graph::Shape> const& inputs)
    {
        if (inputs.size() != dataInputs_.size())
        {
            throw std::invalid_argument("component " + component_.Id() + " has "
                + std::to_string(dataInputs_.size()) + " input port(s)");
        }
        std::map<std::string, std::string> boundBy;
        for (std::size_t port = 0; port < inputs.size(); ++port)
        {
            PortSpec const& spec = *dataInputs_[port];
            graph::Shape const& shape = inputs[port];
            bool const wantsSet = spec.shape.size() == 2;
            if (wantsSet != (shape.kind == graph::SignalKind::Set))
            {
                throw std::invalid_argument("port '" + spec.name + "' expects a " + (wantsSet ? "set" : "vector")
                    + ", the wiring gives " + graph::Describe(shape));
            }
            std::vector<std::size_t> const sizes = wantsSet ? std::vector<std::size_t>{shape.capacity, shape.width}
                                                           : std::vector<std::size_t>{shape.width};
            for (std::size_t axis = 0; axis < sizes.size(); ++axis)
            {
                DimensionSpec const& dimension = spec.shape[axis];
                if (dimension.symbol.empty())
                {
                    if (dimension.size != sizes[axis])
                    {
                        throw std::invalid_argument("port '" + spec.name + "' expects " + std::to_string(dimension.size)
                            + " on axis " + std::to_string(axis) + ", the wiring gives " + graph::Describe(shape));
                    }
                    continue;
                }
                auto const known = symbols_.find(dimension.symbol);
                if (known != symbols_.end() && known->second != sizes[axis])
                {
                    throw std::invalid_argument("port '" + spec.name + "': " + dimension.symbol + " is "
                        + std::to_string(sizes[axis]) + " here but " + std::to_string(known->second) + " on port '"
                        + boundBy[dimension.symbol] + "'");
                }
                symbols_[dimension.symbol] = sizes[axis];
                boundBy[dimension.symbol] = spec.name;
            }
            inputShapes_.push_back(shape);
        }
    }

    std::size_t SizeOf(DimensionSpec const& dimension) const
    {
        return dimension.symbol.empty() ? dimension.size : symbols_.at(dimension.symbol);
    }

    void ResolveOutput()
    {
        PortSpec const& port = *outputPort_;
        if (port.shape.size() == 1)
        {
            outputShape_ = graph::Shape::Vector(SizeOf(port.shape[0]));
            return;
        }
        // A set output keeps the entities of an input set: it follows that set's mask.
        for (std::size_t input = 0; input < dataInputs_.size(); ++input)
        {
            if (!port.mask.empty() && dataInputs_[input]->mask == port.mask)
            {
                outputMaskInput_ = input;
            }
        }
        if (outputMaskInput_ >= dataInputs_.size())
        {
            throw std::invalid_argument("component " + component_.Id() + ", output '" + port.name
                + "': a set output must follow the mask of a set input (\"mask\")");
        }
        outputShape_ = graph::Shape::Set(SizeOf(port.shape[0]), SizeOf(port.shape[1]));
    }

    std::string SymbolsJson() const
    {
        std::string json = "{";
        for (auto const& [symbol, size] : symbols_)
        {
            json += (json.size() == 1 ? "\"" : ",\"") + symbol + "\":" + std::to_string(size);
        }
        return json + "}";
    }

    /** @brief The shape arrays and buffers of every port, once. */
    void AllocateBuffers()
    {
        shapes_.resize(allInputs_.size() + 1);
        inputBuffers_.resize(allInputs_.size());
        gradientBuffers_.resize(allInputs_.size());
        inputGradients_.resize(dataInputs_.size());
        for (std::size_t port = 0; port < allInputs_.size(); ++port)
        {
            PortSpec const& spec = *allInputs_[port];
            for (DimensionSpec const& dimension : spec.shape)
            {
                shapes_[port].push_back(static_cast<std::uint32_t>(SizeOf(dimension)));
            }
            inputBuffers_[port].dtype = spec.dtype == Dtype::Bool ? PPOKIT_DTYPE_BOOL : PPOKIT_DTYPE_F32;
            inputBuffers_[port].rank = static_cast<std::uint32_t>(shapes_[port].size());
            inputBuffers_[port].shape = shapes_[port].data();
            gradientBuffers_[port] = inputBuffers_[port];
            gradientBuffers_[port].data = nullptr;
        }
        for (std::size_t port = 0; port < dataInputs_.size(); ++port)
        {
            inputGradients_[port].assign(inputShapes_[port].Size(), 0.F);
            gradientBuffers_[IndexOf(dataInputs_[port])].data = inputGradients_[port].data();
        }
        for (DimensionSpec const& dimension : outputPort_->shape)
        {
            shapes_.back().push_back(static_cast<std::uint32_t>(SizeOf(dimension)));
        }
        output_.assign(outputShape_.Size(), 0.F);
        outputBuffer_ = {output_.data(), PPOKIT_DTYPE_F32, static_cast<std::uint32_t>(shapes_.back().size()),
            shapes_.back().data()};
        cache_.assign(instance_->CacheSize(1), 0);
    }

    std::size_t IndexOf(PortSpec const* port) const
    {
        return static_cast<std::size_t>(std::find(allInputs_.begin(), allInputs_.end(), port) - allInputs_.begin());
    }

    /** @brief Points the input buffers at this step's signals; masks come from the sets they mask. */
    void BindInputs(graph::Signal const* inputs)
    {
        for (std::size_t port = 0; port < dataInputs_.size(); ++port)
        {
            inputBuffers_[IndexOf(dataInputs_[port])].data = const_cast<float*>(inputs[port].values);
            if (!dataInputs_[port]->mask.empty())
            {
                PortSpec const* const mask = FindInput(dataInputs_[port]->mask);
                inputBuffers_[IndexOf(mask)].data = const_cast<std::uint8_t*>(inputs[port].mask);
            }
        }
    }

    PortSpec const* FindInput(std::string const& name) const
    {
        for (PortSpec const* port : allInputs_)
        {
            if (port->name == name)
            {
                return port;
            }
        }
        return nullptr;
    }

    ComponentRef component_;
    std::string config_;
    std::vector<PortSpec const*> allInputs_;
    std::vector<PortSpec const*> dataInputs_;
    PortSpec const* outputPort_ = nullptr;
    std::map<std::string, std::size_t> symbols_;
    std::vector<graph::Shape> inputShapes_;
    graph::Shape outputShape_;
    /** @brief Data input whose mask a set output follows (past the end for a vector output). */
    std::size_t outputMaskInput_ = static_cast<std::size_t>(-1);
    std::unique_ptr<Instance> instance_;
    std::vector<std::vector<std::uint32_t>> shapes_;
    std::vector<PpokitBuffer> inputBuffers_;
    std::vector<PpokitBuffer> gradientBuffers_;
    std::vector<std::vector<float>> inputGradients_;
    std::vector<float> output_;
    PpokitBuffer outputBuffer_{};
    std::vector<unsigned char> cache_;
    graph::Parameter parameters_;
};

} // namespace ppokit::component
