/**
 * @file PPOKit/include/ppokit/graph/graph.hpp
 * @brief Graph engine of PPOKit: nodes wired port to port, forward and backward passes.
 * @details A Graph is a list of nodes in topological order. Every node reads the outputs of
 * earlier nodes (its sources) and owns its own output buffer, so a forward pass is a single
 * sweep and a backward pass the reverse sweep. Gradients reaching a node from several
 * consumers are summed. Every buffer is allocated while the graph is built: Forward and
 * Backward never allocate.
 *
 * Signals are either vectors `[width]` or sets `[capacity x width]` with a presence mask.
 * The mask of a set follows the values: an entity encoder keeps the mask of its input set.
 *
 * Inputs bound with Forward must stay valid until Backward returns, as for the blocks.
 * Design: docs/architecture/ppokit-graph-blocks.md.
 */
#pragma once

#include "ppokit/graph/blocks.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppokit::graph {

/** @brief Kind of signal carried by a port. */
enum class SignalKind : std::uint8_t
{
    Vector,
    Set,
};

/** @brief Shape of a signal: `width` values, or `capacity` entities of `width` values. */
struct Shape final
{
    SignalKind kind = SignalKind::Vector;
    std::size_t capacity = 1;
    std::size_t width = 0;

    static Shape Vector(std::size_t width) { return {SignalKind::Vector, 1, width}; }
    static Shape Set(std::size_t capacity, std::size_t width) { return {SignalKind::Set, capacity, width}; }

    std::size_t Size() const
    {
        if (width && capacity > std::numeric_limits<std::size_t>::max() / width)
            throw std::length_error("PPOKit graph: signal size overflow");
        return capacity * width;
    }
    bool operator==(Shape const& other) const noexcept
    {
        return kind == other.kind && capacity == other.capacity && width == other.width;
    }
    bool operator!=(Shape const& other) const noexcept { return !(*this == other); }
};

/** @brief Readable form of a shape, for error messages: `vector[8]` or `set[10 x 13]`. */
inline std::string Describe(Shape const& shape)
{
    if (shape.kind == SignalKind::Vector)
    {
        return "vector[" + std::to_string(shape.width) + "]";
    }
    return "set[" + std::to_string(shape.capacity) + " x " + std::to_string(shape.width) + "]";
}

/** @brief Non-owning view of a signal; `mask` is null for vectors. */
struct Signal final
{
    float const* values = nullptr;
    std::uint8_t const* mask = nullptr;
};

/**
 * @brief One block of a graph.
 * @details A node declares its input ports and their shapes when it is constructed, then
 * computes its output from the signals of its sources. Backward ADDS the gradient of each
 * input to the given buffers (several nodes may feed from the same source).
 */
class Node
{
public:
    Node() = default;
    Node(Node const&) = delete;
    Node& operator=(Node const&) = delete;
    virtual ~Node() = default;

    /** @brief Block type, as written in `graph.json` (`dense`, `masked_attention`...). */
    virtual char const* Type() const noexcept = 0;
    /** @brief Names of the input ports, in the order of Forward's `inputs`. */
    virtual std::vector<std::string> InputPorts() const = 0;
    /** @brief Shape expected on each input port. */
    virtual std::vector<Shape> InputShapes() const = 0;
    virtual Shape OutputShape() const = 0;

    virtual Signal Forward(Signal const* inputs) = 0;
    virtual void Backward(float const* outputGradient, float* const* inputGradients) = 0;

    /** @brief Random initial parameters; nodes without parameters do nothing. */
    virtual void Initialize(std::mt19937& /*rng*/) {}
    /** @brief Appends the parameter tensors of this node, in a stable order. */
    virtual void CollectParameters(std::vector<Parameter*>& /*parameters*/) {}
};

namespace detail {

inline void Accumulate(float* target, std::vector<float> const& gradient) noexcept
{
    for (std::size_t index = 0; index < gradient.size(); ++index)
    {
        target[index] += gradient[index];
    }
}

} // namespace detail

/** @brief Graph input: a group of observation or critic fields, bound at each Forward. */
class InputNode final : public Node
{
public:
    explicit InputNode(Shape shape) : shape_(shape)
    {
        if (shape.Size() == 0)
        {
            throw std::invalid_argument("PPOKit input: the size must be positive");
        }
    }

    char const* Type() const noexcept override { return "input"; }
    std::vector<std::string> InputPorts() const override { return {}; }
    std::vector<Shape> InputShapes() const override { return {}; }
    Shape OutputShape() const override { return shape_; }

    void Bind(Signal signal) noexcept { bound_ = signal; }
    Signal Forward(Signal const* /*inputs*/) override { return bound_; }
    void Backward(float const* /*outputGradient*/, float* const* /*inputGradients*/) override {}

private:
    Shape shape_;
    Signal bound_;
};

/** @brief Dense layer on a vector. */
class DenseNode final : public Node
{
public:
    DenseNode(std::size_t inputs, std::size_t width, Activation activation) : dense_(inputs, width, activation) {}

    char const* Type() const noexcept override { return "dense"; }
    std::vector<std::string> InputPorts() const override { return {"in"}; }
    std::vector<Shape> InputShapes() const override { return {Shape::Vector(dense_.Inputs())}; }
    Shape OutputShape() const override { return Shape::Vector(dense_.Outputs()); }

    Signal Forward(Signal const* inputs) override { return {dense_.Forward(inputs[0].values).data(), nullptr}; }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        detail::Accumulate(inputGradients[0], dense_.Backward(outputGradient));
    }
    void Initialize(std::mt19937& rng) override { dense_.Initialize(rng); }
    void CollectParameters(std::vector<Parameter*>& parameters) override
    {
        parameters.push_back(&dense_.Weights());
        parameters.push_back(&dense_.Biases());
    }

    Dense& Layer() noexcept { return dense_; }

private:
    Dense dense_;
};

/** @brief Stack of dense layers of the same width and activation. */
class MlpNode final : public Node
{
public:
    MlpNode(std::size_t inputs, std::size_t layers, std::size_t width, Activation activation)
    {
        if (layers == 0)
        {
            throw std::invalid_argument("PPOKit mlp: at least one layer is required");
        }
        layers_.reserve(layers);
        for (std::size_t layer = 0; layer < layers; ++layer)
        {
            layers_.emplace_back(layer == 0 ? inputs : width, width, activation);
        }
    }

    char const* Type() const noexcept override { return "mlp"; }
    std::vector<std::string> InputPorts() const override { return {"in"}; }
    std::vector<Shape> InputShapes() const override { return {Shape::Vector(layers_.front().Inputs())}; }
    Shape OutputShape() const override { return Shape::Vector(layers_.back().Outputs()); }

    Signal Forward(Signal const* inputs) override
    {
        float const* values = inputs[0].values;
        for (Dense& layer : layers_)
        {
            values = layer.Forward(values).data();
        }
        return {values, nullptr};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        float const* gradient = outputGradient;
        for (auto layer = layers_.rbegin(); layer != layers_.rend(); ++layer)
        {
            gradient = layer->Backward(gradient).data();
        }
        for (std::size_t index = 0; index < layers_.front().Inputs(); ++index)
        {
            inputGradients[0][index] += gradient[index];
        }
    }
    void Initialize(std::mt19937& rng) override
    {
        for (Dense& layer : layers_)
        {
            layer.Initialize(rng);
        }
    }
    void CollectParameters(std::vector<Parameter*>& parameters) override
    {
        for (Dense& layer : layers_)
        {
            parameters.push_back(&layer.Weights());
            parameters.push_back(&layer.Biases());
        }
    }

    std::vector<Dense>& Layers() noexcept { return layers_; }

private:
    std::vector<Dense> layers_;
};

/** @brief Concatenation of vectors, in port order `in1`, `in2`... */
class ConcatNode final : public Node
{
public:
    explicit ConcatNode(std::vector<std::size_t> widths) : widths_(std::move(widths))
    {
        if (widths_.empty())
        {
            throw std::invalid_argument("PPOKit concat: at least one input is required");
        }
        std::size_t total = 0;
        for (std::size_t const width : widths_)
        {
            if (width == 0)
            {
                throw std::invalid_argument("PPOKit concat: input widths must be positive");
            }
            total += width;
        }
        output_.resize(total);
    }

    char const* Type() const noexcept override { return "concat"; }
    std::vector<std::string> InputPorts() const override
    {
        std::vector<std::string> ports;
        for (std::size_t index = 0; index < widths_.size(); ++index)
        {
            ports.push_back("in" + std::to_string(index + 1));
        }
        return ports;
    }
    std::vector<Shape> InputShapes() const override
    {
        std::vector<Shape> shapes;
        for (std::size_t const width : widths_)
        {
            shapes.push_back(Shape::Vector(width));
        }
        return shapes;
    }
    Shape OutputShape() const override { return Shape::Vector(output_.size()); }

    Signal Forward(Signal const* inputs) override
    {
        float* cursor = output_.data();
        for (std::size_t index = 0; index < widths_.size(); ++index)
        {
            cursor = std::copy(inputs[index].values, inputs[index].values + widths_[index], cursor);
        }
        return {output_.data(), nullptr};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        for (std::size_t index = 0; index < widths_.size(); ++index)
        {
            for (std::size_t value = 0; value < widths_[index]; ++value)
            {
                inputGradients[index][value] += outputGradient[value];
            }
            outputGradient += widths_[index];
        }
    }

private:
    std::vector<std::size_t> widths_;
    std::vector<float> output_;
};

/**
 * @brief Picks features of a vector by index, e.g. one scalar of a group.
 * @details An index may appear more than once; its gradients are then summed.
 */
class SelectNode final : public Node
{
public:
    SelectNode(std::size_t inputs, std::vector<std::size_t> indices)
        : inputs_(inputs), indices_(std::move(indices)), output_(indices_.size())
    {
        if (indices_.empty())
        {
            throw std::invalid_argument("PPOKit select: at least one index is required");
        }
        for (std::size_t const index : indices_)
        {
            if (index >= inputs_)
            {
                throw std::invalid_argument("PPOKit select: index " + std::to_string(index)
                    + " is outside an input of width " + std::to_string(inputs_));
            }
        }
    }

    char const* Type() const noexcept override { return "select"; }
    std::vector<std::string> InputPorts() const override { return {"in"}; }
    std::vector<Shape> InputShapes() const override { return {Shape::Vector(inputs_)}; }
    Shape OutputShape() const override { return Shape::Vector(indices_.size()); }

    Signal Forward(Signal const* inputs) override
    {
        for (std::size_t position = 0; position < indices_.size(); ++position)
        {
            output_[position] = inputs[0].values[indices_[position]];
        }
        return {output_.data(), nullptr};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        for (std::size_t position = 0; position < indices_.size(); ++position)
        {
            inputGradients[0][indices_[position]] += outputGradient[position];
        }
    }

private:
    std::size_t inputs_;
    std::vector<std::size_t> indices_;
    std::vector<float> output_;
};

/**
 * @brief Stacks sets of the same width along the entity axis, masks included.
 * @details E.g. friendly tracks followed by the objective, attended as one set.
 */
class ConcatSetsNode final : public Node
{
public:
    ConcatSetsNode(std::vector<std::size_t> capacities, std::size_t width)
        : capacities_(std::move(capacities)), width_(width)
    {
        if (capacities_.empty() || width_ == 0)
        {
            throw std::invalid_argument("PPOKit concat_sets: at least one set of positive width is required");
        }
        std::size_t total = 0;
        for (std::size_t const capacity : capacities_)
        {
            if (capacity == 0)
            {
                throw std::invalid_argument("PPOKit concat_sets: capacities must be positive");
            }
            total += capacity;
        }
        values_.resize(total * width_);
        mask_.resize(total);
    }

    char const* Type() const noexcept override { return "concat_sets"; }
    std::vector<std::string> InputPorts() const override
    {
        std::vector<std::string> ports;
        for (std::size_t index = 0; index < capacities_.size(); ++index)
        {
            ports.push_back("in" + std::to_string(index + 1));
        }
        return ports;
    }
    std::vector<Shape> InputShapes() const override
    {
        std::vector<Shape> shapes;
        for (std::size_t const capacity : capacities_)
        {
            shapes.push_back(Shape::Set(capacity, width_));
        }
        return shapes;
    }
    Shape OutputShape() const override { return Shape::Set(mask_.size(), width_); }

    Signal Forward(Signal const* inputs) override
    {
        float* values = values_.data();
        std::uint8_t* mask = mask_.data();
        for (std::size_t index = 0; index < capacities_.size(); ++index)
        {
            values = std::copy(inputs[index].values, inputs[index].values + capacities_[index] * width_, values);
            mask = std::copy(inputs[index].mask, inputs[index].mask + capacities_[index], mask);
        }
        return {values_.data(), mask_.data()};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        for (std::size_t index = 0; index < capacities_.size(); ++index)
        {
            std::size_t const size = capacities_[index] * width_;
            for (std::size_t value = 0; value < size; ++value)
            {
                inputGradients[index][value] += outputGradient[value];
            }
            outputGradient += size;
        }
    }

private:
    std::vector<std::size_t> capacities_;
    std::size_t width_;
    std::vector<float> values_;
    std::vector<std::uint8_t> mask_;
};

/** @brief Same dense layer applied to every present entity of a set. */
class EntityEncoderNode final : public Node
{
public:
    /** @param encodeAbsent Encode absent entities too (see EntityEncoder). */
    EntityEncoderNode(std::size_t capacity, std::size_t features, std::size_t width, Activation activation,
        bool encodeAbsent = false)
        : capacity_(capacity), features_(features), encoder_(capacity, features, width, activation, encodeAbsent)
    {
    }

    char const* Type() const noexcept override { return "entity_encoder"; }
    std::vector<std::string> InputPorts() const override { return {"in"}; }
    std::vector<Shape> InputShapes() const override { return {Shape::Set(capacity_, features_)}; }
    Shape OutputShape() const override { return Shape::Set(capacity_, encoder_.Width()); }

    Signal Forward(Signal const* inputs) override
    {
        return {encoder_.Forward(inputs[0].values, inputs[0].mask).data(), inputs[0].mask};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        detail::Accumulate(inputGradients[0], encoder_.Backward(outputGradient));
    }
    void Initialize(std::mt19937& rng) override { encoder_.Initialize(rng); }
    void CollectParameters(std::vector<Parameter*>& parameters) override
    {
        parameters.push_back(&encoder_.Layer().Weights());
        parameters.push_back(&encoder_.Layer().Biases());
    }

    EntityEncoder& Encoder() noexcept { return encoder_; }

private:
    std::size_t capacity_;
    std::size_t features_;
    EntityEncoder encoder_;
};

/**
 * @brief Scaled dot-product attention of one query over the present entities of a set.
 * @details With separate keys (port "keys", a set of the same shape, e.g. a projection of
 * the entities), the scores read the keys and the output averages the set; without, the
 * entities are their own keys. The mask is the set's.
 */
class MaskedAttentionNode final : public Node
{
public:
    /** @param fill Score added to absent entities, the softmax over every entity (see MaskedAttention). */
    MaskedAttentionNode(std::size_t capacity, std::size_t width, bool separateKeys = false,
        std::optional<float> fill = std::nullopt)
        : capacity_(capacity), width_(width), separateKeys_(separateKeys), attention_(capacity, width, fill)
    {
    }

    char const* Type() const noexcept override { return "masked_attention"; }
    std::vector<std::string> InputPorts() const override
    {
        if (separateKeys_)
        {
            return {"query", "set", "keys"};
        }
        return {"query", "set"};
    }
    std::vector<Shape> InputShapes() const override
    {
        if (separateKeys_)
        {
            return {Shape::Vector(width_), Shape::Set(capacity_, width_), Shape::Set(capacity_, width_)};
        }
        return {Shape::Vector(width_), Shape::Set(capacity_, width_)};
    }
    Shape OutputShape() const override { return Shape::Vector(width_); }

    Signal Forward(Signal const* inputs) override
    {
        float const* keys = separateKeys_ ? inputs[2].values : inputs[1].values;
        return {attention_.Forward(inputs[0].values, keys, inputs[1].values, inputs[1].mask).data(), nullptr};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        attention_.Backward(outputGradient);
        detail::Accumulate(inputGradients[0], attention_.QueryGradient());
        detail::Accumulate(inputGradients[1], attention_.SetGradient());
        if (separateKeys_)
        {
            detail::Accumulate(inputGradients[2], attention_.KeyGradient());
        }
    }

    /** @brief Attention weights of the last Forward (zero for absent entities), for probes. */
    std::vector<float> const& Weights() const noexcept { return attention_.Weights(); }

private:
    std::size_t capacity_;
    std::size_t width_;
    bool separateKeys_;
    MaskedAttention attention_;
};

/** @brief Mean or maximum over the present entities of a set. */
class MaskedPoolNode final : public Node
{
public:
    /** @param fill Value of absent entities in a maximum (see MaskedPool). */
    MaskedPoolNode(std::size_t capacity, std::size_t width, PoolMode mode, std::optional<float> fill = std::nullopt)
        : capacity_(capacity), width_(width), pool_(capacity, width, mode, fill)
    {
    }

    char const* Type() const noexcept override { return "masked_pool"; }
    std::vector<std::string> InputPorts() const override { return {"in"}; }
    std::vector<Shape> InputShapes() const override { return {Shape::Set(capacity_, width_)}; }
    Shape OutputShape() const override { return Shape::Vector(width_); }

    Signal Forward(Signal const* inputs) override
    {
        return {pool_.Forward(inputs[0].values, inputs[0].mask).data(), nullptr};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        detail::Accumulate(inputGradients[0], pool_.Backward(outputGradient));
    }

private:
    std::size_t capacity_;
    std::size_t width_;
    MaskedPool pool_;
};

/** @brief Kind of distribution produced by a head node. */
enum class HeadKind : std::uint8_t
{
    Gaussian,
    Bernoulli,
    Value,
};

/** @brief Where the log standard deviations of a Gaussian head come from. */
struct LogStdSettings final
{
    /** @brief False: learned values, independent of the state; true: a dense layer of the head's input. */
    bool stateDependent = false;
    /** @brief Bounds of a state-dependent log-std (its gradient is zero outside them). */
    float minimum = -20.F;
    float maximum = 2.F;
};

/**
 * @brief Linear output layer: Gaussian means, Bernoulli logits or the state value.
 * @details A Gaussian head also owns the log standard deviations of its actions. By
 * default they are parameters independent of the state: not used by Forward (which
 * returns the means) but part of the graph's parameters, so the optimiser updates them
 * with the rest. With LogStdSettings::stateDependent, a second dense layer computes them
 * from the head's input, clamped to [minimum, maximum]: Forward computes them with the
 * means (CurrentLogStd), and the learner gives their gradient (LogStdGradient) before
 * Backward.
 */
class HeadNode final : public Node
{
public:
    HeadNode(HeadKind kind, std::size_t inputs, std::size_t outputs, float initialLogStd = 0.F,
        LogStdSettings logStd = {})
        : kind_(kind), dense_(inputs, outputs, Activation::Identity),
          logStd_(kind == HeadKind::Gaussian && !logStd.stateDependent ? outputs : 0), logStdSettings_(logStd)
    {
        if (kind == HeadKind::Value && outputs != 1)
        {
            throw std::invalid_argument("PPOKit value head: exactly one output is required");
        }
        if (logStd.stateDependent && kind != HeadKind::Gaussian)
        {
            throw std::invalid_argument("PPOKit head: only a Gaussian head has a log-std");
        }
        if (logStd.stateDependent)
        {
            if (!(logStd.minimum < logStd.maximum))
            {
                throw std::invalid_argument("PPOKit Gaussian head: the log-std bounds must be increasing");
            }
            logStdDense_ = std::make_unique<Dense>(inputs, outputs, Activation::Identity);
            currentLogStd_.assign(outputs, 0.F);
            logStdGradient_.assign(outputs, 0.F);
        }
        std::fill(logStd_.value.begin(), logStd_.value.end(), initialLogStd);
    }

    char const* Type() const noexcept override
    {
        switch (kind_)
        {
        case HeadKind::Gaussian:
            return "gaussian_head";
        case HeadKind::Bernoulli:
            return "bernoulli_head";
        case HeadKind::Value:
            break;
        }
        return "value";
    }
    std::vector<std::string> InputPorts() const override { return {"in"}; }
    std::vector<Shape> InputShapes() const override { return {Shape::Vector(dense_.Inputs())}; }
    Shape OutputShape() const override { return Shape::Vector(dense_.Outputs()); }

    Signal Forward(Signal const* inputs) override
    {
        if (logStdDense_ != nullptr)
        {
            std::vector<float> const& raw = logStdDense_->Forward(inputs[0].values);
            for (std::size_t output = 0; output < raw.size(); ++output)
            {
                currentLogStd_[output] = std::clamp(raw[output], logStdSettings_.minimum, logStdSettings_.maximum);
            }
            std::fill(logStdGradient_.begin(), logStdGradient_.end(), 0.F);
        }
        return {dense_.Forward(inputs[0].values).data(), nullptr};
    }
    void Backward(float const* outputGradient, float* const* inputGradients) override
    {
        detail::Accumulate(inputGradients[0], dense_.Backward(outputGradient));
        if (logStdDense_ == nullptr)
        {
            return;
        }
        // The clamp passes the gradient inside its bounds only (as torch.clamp).
        std::vector<float> const& raw = logStdDense_->Output();
        for (std::size_t output = 0; output < raw.size(); ++output)
        {
            bool const inside = raw[output] >= logStdSettings_.minimum && raw[output] <= logStdSettings_.maximum;
            if (!inside)
            {
                logStdGradient_[output] = 0.F;
            }
        }
        detail::Accumulate(inputGradients[0], logStdDense_->Backward(logStdGradient_.data()));
    }
    void Initialize(std::mt19937& rng) override
    {
        dense_.Initialize(rng);
        if (logStdDense_ != nullptr)
        {
            logStdDense_->Initialize(rng);
        }
    }
    void CollectParameters(std::vector<Parameter*>& parameters) override
    {
        parameters.push_back(&dense_.Weights());
        parameters.push_back(&dense_.Biases());
        if (logStdDense_ != nullptr)
        {
            parameters.push_back(&logStdDense_->Weights());
            parameters.push_back(&logStdDense_->Biases());
        }
        else if (kind_ == HeadKind::Gaussian)
        {
            parameters.push_back(&logStd_);
        }
    }

    HeadKind Kind() const noexcept { return kind_; }
    Dense& Layer() noexcept { return dense_; }
    /** @brief True when the log-std is computed from the state (LogStdSettings). */
    bool StateDependentLogStd() const noexcept { return logStdDense_ != nullptr; }
    /** @brief Learned log standard deviations of a state-independent Gaussian head (else empty). */
    Parameter& LogStd() noexcept { return logStd_; }
    /** @brief Dense layer of a state-dependent log-std (null otherwise). */
    Dense* LogStdLayer() noexcept { return logStdDense_.get(); }
    /** @brief Log-std of a component for the last Forward: computed or learned. */
    float CurrentLogStd(std::size_t component) const
    {
        if (logStdDense_ != nullptr)
        {
            return currentLogStd_.at(component);
        }
        return logStd_.value.at(component);
    }
    /**
     * @brief Gradient of the loss with respect to the state-dependent log-std of the last
     *        Forward, set by the learner before Backward (reset by Forward).
     */
    std::vector<float>& LogStdGradient() noexcept { return logStdGradient_; }

private:
    HeadKind kind_;
    Dense dense_;
    Parameter logStd_;
    LogStdSettings logStdSettings_;
    std::unique_ptr<Dense> logStdDense_;
    std::vector<float> currentLogStd_;
    std::vector<float> logStdGradient_;
};

/** @brief Identifier of a node inside its graph. */
using NodeId = std::size_t;

/**
 * @brief Nodes in topological order with their connections.
 * @details Build it with AddInput and AddNode: a node may only read nodes added before it,
 * so the graph is acyclic by construction (the JSON loader sorts the nodes first). Shapes
 * are checked on every connection, with the node and port named in the error.
 */
class Graph final
{
public:
    Graph() = default;
    Graph(Graph const&) = delete;
    Graph& operator=(Graph const&) = delete;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    /** @brief Adds an input bound at each Forward, in the order of the AddInput calls. */
    NodeId AddInput(std::string name, Shape shape)
    {
        auto input = std::make_unique<InputNode>(shape);
        inputs_.push_back(input.get());
        return Append(std::move(name), std::move(input), {});
    }

    /** @brief Adds a node reading `sources`, one per input port in port order. */
    NodeId AddNode(std::string name, std::unique_ptr<Node> node, std::vector<NodeId> const& sources)
    {
        if (!node)
        {
            throw std::invalid_argument("PPOKit graph: node '" + name + "' is empty");
        }
        auto const ports = node->InputPorts();
        auto const shapes = node->InputShapes();
        if (sources.size() != ports.size())
        {
            throw std::invalid_argument("PPOKit graph: node '" + name + "' (" + node->Type() + ") has "
                + std::to_string(ports.size()) + " input port(s), " + std::to_string(sources.size())
                + " source(s) given");
        }
        for (std::size_t port = 0; port < ports.size(); ++port)
        {
            if (sources[port] >= nodes_.size())
            {
                throw std::invalid_argument("PPOKit graph: port '" + name + "." + ports[port]
                    + "' reads an unknown node (nodes must be added after their sources)");
            }
            Shape const given = nodes_[sources[port]].node->OutputShape();
            if (given != shapes[port])
            {
                throw std::invalid_argument("PPOKit graph: port '" + name + "." + ports[port] + "' expects "
                    + Describe(shapes[port]) + ", node '" + nodes_[sources[port]].name + "' gives " + Describe(given));
            }
        }
        return Append(std::move(name), std::move(node), sources);
    }

    /**
     * @brief Declares a graph output, such as a head; outputs are numbered in call order.
     * @details An actor with Gaussian and Bernoulli actions has two outputs.
     */
    void AddOutput(NodeId node)
    {
        if (node >= nodes_.size())
        {
            throw std::invalid_argument("PPOKit graph: unknown output node");
        }
        if (std::find(outputs_.begin(), outputs_.end(), node) != outputs_.end())
        {
            throw std::invalid_argument("PPOKit graph: node '" + nodes_[node].name + "' is already an output");
        }
        outputs_.push_back(node);
    }

    std::size_t NodeCount() const noexcept { return nodes_.size(); }
    std::size_t InputCount() const noexcept { return inputs_.size(); }
    /** @brief Shape of input `input`, in AddInput order. */
    Shape InputShape(std::size_t input) const { return inputs_.at(input)->OutputShape(); }
    Node& At(NodeId node) { return *nodes_.at(node).node; }
    std::string const& NameOf(NodeId node) const { return nodes_.at(node).name; }
    /** @brief Identifier of the node called `name`; throws if there is none. */
    NodeId Find(std::string const& name) const
    {
        for (NodeId node = 0; node < nodes_.size(); ++node)
        {
            if (nodes_[node].name == name)
            {
                return node;
            }
        }
        throw std::invalid_argument("PPOKit graph: no node named '" + name + "'");
    }
    std::size_t OutputCount() const noexcept { return outputs_.size(); }
    /** @brief Node of output `output`, e.g. to reach the log-std of a Gaussian head. */
    NodeId OutputId(std::size_t output) const { return OutputNode(output); }
    Shape OutputShape(std::size_t output = 0) const { return nodes_.at(OutputNode(output)).node->OutputShape(); }

    /** @brief Initializes every parameter from one generator, in node order. */
    void Initialize(std::mt19937& rng)
    {
        for (auto& entry : nodes_)
        {
            entry.node->Initialize(rng);
        }
    }

    /** @brief Parameter tensors of the whole graph, in node order (stable for checkpoints). */
    std::vector<Parameter*> Parameters()
    {
        std::vector<Parameter*> parameters;
        for (auto& entry : nodes_)
        {
            entry.node->CollectParameters(parameters);
        }
        return parameters;
    }

    void ZeroGradients()
    {
        for (Parameter* parameter : Parameters())
        {
            parameter->ZeroGradient();
        }
    }

    /**
     * @brief Forward pass; `inputs` follows the order of AddInput.
     * @return The first output, valid until the next Forward (see Output for the others).
     */
    Signal Forward(std::vector<Signal> const& inputs)
    {
        if (inputs.size() != inputs_.size())
        {
            throw std::invalid_argument("PPOKit graph: " + std::to_string(inputs_.size()) + " input(s) expected, "
                + std::to_string(inputs.size()) + " given");
        }
        for (std::size_t index = 0; index < inputs.size(); ++index)
        {
            bool const isSet = inputs_[index]->OutputShape().kind == SignalKind::Set;
            if (inputs[index].values == nullptr || (isSet && inputs[index].mask == nullptr))
            {
                throw std::invalid_argument("PPOKit graph: input " + std::to_string(index + 1)
                    + (isSet ? " needs values and a presence mask" : " needs values"));
            }
            inputs_[index]->Bind(inputs[index]);
        }
        for (auto& entry : nodes_)
        {
            for (std::size_t port = 0; port < entry.sources.size(); ++port)
            {
                entry.inputSignals[port] = nodes_[entry.sources[port]].output;
            }
            entry.output = entry.node->Forward(entry.inputSignals.data());
        }
        return nodes_[OutputNode(0)].output;
    }

    /** @brief Output `output` of the last Forward. */
    Signal Output(std::size_t output) const { return nodes_[OutputNode(output)].output; }

    /** @brief Backward pass of a graph with a single output; see the general overload. */
    void Backward(float const* outputGradient)
    {
        if (outputs_.size() != 1)
        {
            throw std::invalid_argument("PPOKit graph: one gradient per output is required");
        }
        Backward(&outputGradient, 1);
    }

    /**
     * @brief Backward pass of the last Forward from the gradient of every output.
     * @param outputGradients One gradient per output, in AddOutput order; null for an output
     *        that does not contribute to the loss.
     * @param count Number of gradients, equal to OutputCount().
     * @details Accumulates parameter gradients (call ZeroGradients between updates) and
     * leaves the gradient of every node output available through OutputGradient.
     */
    void Backward(float const* const* outputGradients, std::size_t count)
    {
        if (count != outputs_.size() || outputs_.empty())
        {
            throw std::invalid_argument("PPOKit graph: " + std::to_string(outputs_.size()) + " output gradient(s) expected, "
                + std::to_string(count) + " given");
        }
        for (auto& entry : nodes_)
        {
            std::fill(entry.outputGradient.begin(), entry.outputGradient.end(), 0.F);
        }
        for (std::size_t output = 0; output < count; ++output)
        {
            if (outputGradients[output] != nullptr)
            {
                SeedGradient(outputs_[output], outputGradients[output]);
            }
        }
        for (auto entry = nodes_.rbegin(); entry != nodes_.rend(); ++entry)
        {
            entry->node->Backward(entry->outputGradient.data(), entry->inputGradients.data());
        }
    }

    /** @brief Output signal of a node during the last Forward (probes). */
    Signal OutputOf(NodeId node) const { return nodes_.at(node).output; }
    /** @brief Gradient of the loss with respect to a node output after the last Backward. */
    std::vector<float> const& OutputGradient(NodeId node) const { return nodes_.at(node).outputGradient; }

private:
    struct Entry final
    {
        std::string name;
        std::unique_ptr<Node> node;
        std::vector<NodeId> sources;
        std::vector<Signal> inputSignals;
        std::vector<float*> inputGradients;
        std::vector<float> outputGradient;
        Signal output;
    };

    NodeId Append(std::string name, std::unique_ptr<Node> node, std::vector<NodeId> const& sources)
    {
        for (auto const& entry : nodes_)
        {
            if (entry.name == name)
            {
                throw std::invalid_argument("PPOKit graph: node '" + name + "' is defined twice");
            }
        }
        Entry entry;
        entry.name = std::move(name);
        entry.outputGradient.assign(node->OutputShape().Size(), 0.F);
        entry.node = std::move(node);
        entry.sources = sources;
        entry.inputSignals.resize(sources.size());
        nodes_.push_back(std::move(entry));
        // Moving entries keeps the heap buffers of outputGradient, so pointers taken
        // before this push_back stay valid; only the new entry needs its own.
        auto& added = nodes_.back();
        for (NodeId const source : added.sources)
        {
            added.inputGradients.push_back(nodes_[source].outputGradient.data());
        }
        return nodes_.size() - 1;
    }

    NodeId OutputNode(std::size_t output) const
    {
        if (output >= outputs_.size())
        {
            throw std::logic_error("PPOKit graph: output " + std::to_string(output) + " is not declared");
        }
        return outputs_[output];
    }

    /** @brief Adds the gradient of a loss to an output (an output may also feed other nodes). */
    void SeedGradient(NodeId node, float const* gradient) noexcept
    {
        std::vector<float>& target = nodes_[node].outputGradient;
        for (std::size_t index = 0; index < target.size(); ++index)
        {
            target[index] += gradient[index];
        }
    }

    std::vector<Entry> nodes_;
    std::vector<InputNode*> inputs_;
    std::vector<NodeId> outputs_;
};

} // namespace ppokit::graph
