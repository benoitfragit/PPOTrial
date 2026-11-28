/**
 * @file PPOKit/include/ppokit/graph/blocks.hpp
 * @brief Differentiable blocks of PPOKit graphs: dense, entity encoder, masked attention,
 *        masked pooling and concatenation.
 * @details Each block owns its parameters and the buffers of one forward pass, all
 * allocated in its constructor: Forward and Backward never allocate. Backward computes
 * the exact gradient of the outputs it receives, accumulates parameter gradients and
 * returns the gradient of its inputs. Sets are row-major `[capacity x features]` with a
 * presence mask: absent entities never influence an output nor receive a gradient.
 *
 * Masking by fill value (LibTorch's `masked_fill` networks, e.g. the red RLMission agents):
 * an entity encoder that encodes absent entities too, a maximum that counts them as the fill
 * value, an attention that adds the fill value to their scores and softmaxes over every
 * entity. With no present entity these networks output what LibTorch computed (the fill
 * value propagates), where the default masking outputs zeros.
 * Design: docs/architecture/ppokit-graph-blocks.md.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace ppokit::graph {

/** @brief Activation of a dense layer. */
enum class Activation : std::uint8_t
{
    Identity,
    Relu,
    Tanh,
};

/** @brief Values of one parameter tensor and their accumulated gradient. */
struct Parameter final
{
    std::vector<float> value;
    std::vector<float> gradient;

    explicit Parameter(std::size_t size = 0) : value(size, 0.F), gradient(size, 0.F) {}
    void ZeroGradient() noexcept { std::fill(gradient.begin(), gradient.end(), 0.F); }
};

/** @brief Applies an activation to a pre-activation value. */
inline float Activate(Activation activation, float x) noexcept
{
    switch (activation)
    {
    case Activation::Relu:
        return x > 0.F ? x : 0.F;
    case Activation::Tanh:
        return std::tanh(x);
    case Activation::Identity:
        break;
    }
    return x;
}

/** @brief Derivative of the activation, from its pre-activation and output values. */
inline float ActivationSlope(Activation activation, float preActivation, float output) noexcept
{
    switch (activation)
    {
    case Activation::Relu:
        return preActivation > 0.F ? 1.F : 0.F;
    case Activation::Tanh:
        return 1.F - output * output;
    case Activation::Identity:
        break;
    }
    return 1.F;
}

/**
 * @brief Fully connected layer y = activation(W x + b), W of shape [outputs x inputs].
 * @details Also the per-entity transform of EntityEncoder, which calls the row-level
 * helpers on each present entity with shared weights.
 */
class Dense final
{
public:
    Dense(std::size_t inputs, std::size_t outputs, Activation activation)
        : inputs_(inputs), outputs_(outputs), activation_(activation), weights_(inputs * outputs),
          biases_(outputs), preActivation_(outputs), output_(outputs), inputGradient_(inputs)
    {
        if (inputs == 0 || outputs == 0)
        {
            throw std::invalid_argument("PPOKit dense: sizes must be positive");
        }
    }

    /**
     * @brief Weights and biases uniform in +/- 1 / sqrt(inputs): the default of rl_tools
     *        (init_kaiming with a = sqrt(5), as PyTorch), that of the FlightControl3D
     *        learner. Every dense layer, heads included.
     */
    template<typename Engine>
    void Initialize(Engine& rng)
    {
        float const bound = 1.F / std::sqrt(static_cast<float>(inputs_));
        std::uniform_real_distribution<float> uniform(-bound, bound);
        for (float& weight : weights_.value)
        {
            weight = uniform(rng);
        }
        for (float& bias : biases_.value)
        {
            bias = uniform(rng);
        }
    }

    std::size_t Inputs() const noexcept { return inputs_; }
    std::size_t Outputs() const noexcept { return outputs_; }

    /** @brief Forward pass on one input row; `input` must stay valid until Backward. */
    std::vector<float> const& Forward(float const* input)
    {
        lastInput_ = input;
        ForwardRow(input, preActivation_.data(), output_.data());
        return output_;
    }

    /** @brief Backward pass of the last Forward: accumulates parameter gradients, returns d input. */
    std::vector<float> const& Backward(float const* outputGradient)
    {
        BackwardRow(lastInput_, preActivation_.data(), output_.data(), outputGradient, inputGradient_.data());
        return inputGradient_;
    }

    /** @brief y = activation(W x + b) for one row, writing the pre-activation and the output. */
    void ForwardRow(float const* input, float* preActivation, float* output) const noexcept
    {
        for (std::size_t out = 0; out < outputs_; ++out)
        {
            float sum = biases_.value[out];
            float const* row = &weights_.value[out * inputs_];
            for (std::size_t in = 0; in < inputs_; ++in)
            {
                sum += row[in] * input[in];
            }
            preActivation[out] = sum;
            output[out] = Activate(activation_, sum);
        }
    }

    /** @brief Gradient of one row: accumulates dW and db, writes d input (overwritten). */
    void BackwardRow(float const* input, float const* preActivation, float const* output, float const* outputGradient,
        float* inputGradient) noexcept
    {
        std::fill(inputGradient, inputGradient + inputs_, 0.F);
        for (std::size_t out = 0; out < outputs_; ++out)
        {
            float const delta = outputGradient[out] * ActivationSlope(activation_, preActivation[out], output[out]);
            biases_.gradient[out] += delta;
            float* gradientRow = &weights_.gradient[out * inputs_];
            float const* row = &weights_.value[out * inputs_];
            for (std::size_t in = 0; in < inputs_; ++in)
            {
                gradientRow[in] += delta * input[in];
                inputGradient[in] += delta * row[in];
            }
        }
    }

    /** @brief Output of the last Forward. */
    std::vector<float> const& Output() const noexcept { return output_; }
    Parameter& Weights() noexcept { return weights_; }
    Parameter& Biases() noexcept { return biases_; }

private:
    std::size_t inputs_;
    std::size_t outputs_;
    Activation activation_;
    Parameter weights_;
    Parameter biases_;
    std::vector<float> preActivation_;
    std::vector<float> output_;
    std::vector<float> inputGradient_;
    float const* lastInput_ = nullptr;
};

/**
 * @brief Encodes every present entity of a set with the same dense layer.
 * @details Output set [capacity x width]; absent entities give zero rows and no gradient,
 * unless `encodeAbsent`: then every entity is encoded and receives its gradient, as a
 * LibTorch network without masking did (the mask still travels with the set).
 */
class EntityEncoder final
{
public:
    EntityEncoder(std::size_t capacity, std::size_t features, std::size_t width, Activation activation,
        bool encodeAbsent = false)
        : capacity_(capacity), encodeAbsent_(encodeAbsent), dense_(features, width, activation),
          preActivation_(capacity * width), output_(capacity * width), inputGradient_(capacity * features)
    {
        if (capacity == 0)
        {
            throw std::invalid_argument("PPOKit entity encoder: the capacity must be positive");
        }
    }

    template<typename Engine>
    void Initialize(Engine& rng)
    {
        dense_.Initialize(rng);
    }

    std::size_t Width() const noexcept { return dense_.Outputs(); }

    /** @param input Set [capacity x features]. @param mask One flag per entity, true when present. */
    std::vector<float> const& Forward(float const* input, std::uint8_t const* mask)
    {
        lastInput_ = input;
        lastMask_ = mask;
        std::size_t const features = dense_.Inputs();
        std::size_t const width = dense_.Outputs();
        std::fill(output_.begin(), output_.end(), 0.F);
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (Encodes(mask, entity))
            {
                dense_.ForwardRow(input + entity * features, &preActivation_[entity * width], &output_[entity * width]);
            }
        }
        return output_;
    }

    std::vector<float> const& Backward(float const* outputGradient)
    {
        std::size_t const features = dense_.Inputs();
        std::size_t const width = dense_.Outputs();
        std::fill(inputGradient_.begin(), inputGradient_.end(), 0.F);
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (Encodes(lastMask_, entity))
            {
                dense_.BackwardRow(lastInput_ + entity * features, &preActivation_[entity * width],
                    &output_[entity * width], outputGradient + entity * width, &inputGradient_[entity * features]);
            }
        }
        return inputGradient_;
    }

    Dense& Layer() noexcept { return dense_; }

private:
    bool Encodes(std::uint8_t const* mask, std::size_t entity) const noexcept
    {
        return encodeAbsent_ || mask[entity] != 0;
    }

    std::size_t capacity_;
    bool encodeAbsent_;
    Dense dense_;
    std::vector<float> preActivation_;
    std::vector<float> output_;
    std::vector<float> inputGradient_;
    float const* lastInput_ = nullptr;
    std::uint8_t const* lastMask_ = nullptr;
};

/**
 * @brief Single-head scaled dot-product attention of a query over the present entities.
 * @details p_i = softmax_i(q . k_i / sqrt(d)) over present entities, output = sum_i p_i v_i.
 * An empty set outputs zeros. The values v_i are the entity embeddings; the keys k_i are
 * the same embeddings, or a separate set of the same shape (e.g. a projection of them).
 *
 * With a fill value f (LibTorch's masked_fill): s_i = q . k_i / sqrt(d) + (absent ? f : 0),
 * in float, and the softmax runs over every entity. Absent entities then weigh nothing
 * beside a present one (exp underflows), and with no present entity the output is what
 * LibTorch computed rather than zeros.
 */
class MaskedAttention final
{
public:
    MaskedAttention(std::size_t capacity, std::size_t width, std::optional<float> fill = std::nullopt)
        : capacity_(capacity), width_(width), fill_(fill), everyEntity_(capacity, std::uint8_t{1}),
          probabilities_(capacity), output_(width), queryGradient_(width), setGradient_(capacity * width),
          keyGradient_(capacity * width)
    {
        if (capacity == 0 || width == 0)
        {
            throw std::invalid_argument("PPOKit attention: sizes must be positive");
        }
    }

    /** @brief Attention whose keys are the values themselves. */
    std::vector<float> const& Forward(float const* query, float const* set, std::uint8_t const* mask)
    {
        return Forward(query, set, set, mask);
    }

    /**
     * @brief Attention with keys apart from the values; `keys` may be `set`.
     * @param keys Capacity x width keys, read for the scores.
     * @param set Capacity x width values, averaged by the attention weights.
     */
    std::vector<float> const& Forward(float const* query, float const* keys, float const* set, std::uint8_t const* mask)
    {
        lastQuery_ = query;
        lastKeys_ = keys;
        lastSet_ = set;
        if (fill_)
        {
            return ForwardFilled(query, keys, set, mask);
        }
        lastMask_ = mask;
        float const scale = 1.F / std::sqrt(static_cast<float>(width_));
        float largest = -std::numeric_limits<float>::infinity();
        present_ = 0;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            probabilities_[entity] = 0.F;
            if (mask[entity] == 0)
            {
                continue;
            }
            ++present_;
            probabilities_[entity] = Dot(query, keys + entity * width_) * scale;
            largest = std::max(largest, probabilities_[entity]);
        }
        std::fill(output_.begin(), output_.end(), 0.F);
        if (present_ == 0)
        {
            return output_;
        }
        // Softmax shifted by the largest score for numerical stability.
        float denominator = 0.F;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (mask[entity] != 0)
            {
                probabilities_[entity] = std::exp(probabilities_[entity] - largest);
                denominator += probabilities_[entity];
            }
        }
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (mask[entity] == 0)
            {
                continue;
            }
            probabilities_[entity] /= denominator;
            for (std::size_t d = 0; d < width_; ++d)
            {
                output_[d] += probabilities_[entity] * set[entity * width_ + d];
            }
        }
        return output_;
    }

    /**
     * @brief Writes d query, d values and d keys; read them with QueryGradient, SetGradient
     *        and KeyGradient. Without separate keys, the set gradient holds both parts.
     */
    void Backward(float const* outputGradient)
    {
        std::fill(queryGradient_.begin(), queryGradient_.end(), 0.F);
        std::fill(setGradient_.begin(), setGradient_.end(), 0.F);
        std::fill(keyGradient_.begin(), keyGradient_.end(), 0.F);
        if (present_ == 0)
        {
            return;
        }
        float const scale = 1.F / std::sqrt(static_cast<float>(width_));
        // d out / d p_i = v_i; through the softmax: d s_i = p_i (g . v_i - sum_j p_j g . v_j).
        float expected = 0.F;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (lastMask_[entity] != 0)
            {
                expected += probabilities_[entity] * Dot(outputGradient, lastSet_ + entity * width_);
            }
        }
        bool const sameKeys = lastKeys_ == lastSet_;
        std::vector<float>& keyGradient = sameKeys ? setGradient_ : keyGradient_;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (lastMask_[entity] == 0)
            {
                continue;
            }
            float const* value = lastSet_ + entity * width_;
            float const* key = lastKeys_ + entity * width_;
            float const scoreGradient = probabilities_[entity] * (Dot(outputGradient, value) - expected) * scale;
            for (std::size_t d = 0; d < width_; ++d)
            {
                setGradient_[entity * width_ + d] += probabilities_[entity] * outputGradient[d];
                keyGradient[entity * width_ + d] += scoreGradient * lastQuery_[d];
                queryGradient_[d] += scoreGradient * key[d];
            }
        }
    }

    std::vector<float> const& QueryGradient() const noexcept { return queryGradient_; }
    std::vector<float> const& SetGradient() const noexcept { return setGradient_; }
    /** @brief Gradient of the separate keys (zeros when the keys are the values). */
    std::vector<float> const& KeyGradient() const noexcept { return keyGradient_; }
    std::vector<float> const& Weights() const noexcept { return probabilities_; }

private:
    float Dot(float const* left, float const* right) const noexcept
    {
        float sum = 0.F;
        for (std::size_t d = 0; d < width_; ++d)
        {
            sum += left[d] * right[d];
        }
        return sum;
    }

    /**
     * @brief Forward with a fill value: every entity takes part, an absent one with the fill
     *        added to its score. Backward then treats every entity as present.
     */
    std::vector<float> const& ForwardFilled(float const* query, float const* keys, float const* set,
        std::uint8_t const* mask)
    {
        lastMask_ = everyEntity_.data();
        present_ = capacity_;
        // q . k / sqrt(d) as LibTorch divides, then the fill in float.
        float const root = std::sqrt(static_cast<float>(width_));
        float largest = -std::numeric_limits<float>::infinity();
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            float score = Dot(query, keys + entity * width_) / root;
            if (mask[entity] == 0)
            {
                score += *fill_;
            }
            probabilities_[entity] = score;
            largest = std::max(largest, score);
        }
        float denominator = 0.F;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            probabilities_[entity] = std::exp(probabilities_[entity] - largest);
            denominator += probabilities_[entity];
        }
        std::fill(output_.begin(), output_.end(), 0.F);
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            probabilities_[entity] /= denominator;
            for (std::size_t d = 0; d < width_; ++d)
            {
                output_[d] += probabilities_[entity] * set[entity * width_ + d];
            }
        }
        return output_;
    }

    std::size_t capacity_;
    std::size_t width_;
    std::optional<float> fill_;
    /** @brief The mask of a filled attention's backward: every entity took part. */
    std::vector<std::uint8_t> everyEntity_;
    std::vector<float> probabilities_;
    std::vector<float> output_;
    std::vector<float> queryGradient_;
    std::vector<float> setGradient_;
    std::vector<float> keyGradient_;
    float const* lastQuery_ = nullptr;
    float const* lastKeys_ = nullptr;
    float const* lastSet_ = nullptr;
    std::uint8_t const* lastMask_ = nullptr;
    std::size_t present_ = 0;
};

/** @brief Reduction of a masked pool. */
enum class PoolMode : std::uint8_t
{
    Mean,
    Max,
};

/**
 * @brief Mean or maximum over the present entities, per feature; zeros for an empty set.
 * @details A maximum with a fill value counts absent entities as that value (LibTorch's
 * masked_fill before the max): the maximum of the present ones, or the fill for an empty set.
 */
class MaskedPool final
{
public:
    MaskedPool(std::size_t capacity, std::size_t width, PoolMode mode, std::optional<float> fill = std::nullopt)
        : capacity_(capacity), width_(width), mode_(mode), fill_(fill), output_(width), inputGradient_(capacity * width)
    {
        if (capacity == 0 || width == 0)
        {
            throw std::invalid_argument("PPOKit pooling: sizes must be positive");
        }
        if (fill && mode != PoolMode::Max)
        {
            throw std::invalid_argument("PPOKit pooling: a fill value applies to a maximum only");
        }
    }

    std::vector<float> const& Forward(float const* set, std::uint8_t const* mask)
    {
        lastSet_ = set;
        lastMask_ = mask;
        present_ = 0;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            present_ += mask[entity] != 0 ? 1 : 0;
        }
        std::fill(output_.begin(), output_.end(), fill_.value_or(0.F));
        if (present_ == 0)
        {
            return output_;
        }
        for (std::size_t d = 0; d < width_; ++d)
        {
            output_[d] = mode_ == PoolMode::Mean ? MeanOf(set, d) : MaxOf(set, d);
        }
        return output_;
    }

    std::vector<float> const& Backward(float const* outputGradient)
    {
        std::fill(inputGradient_.begin(), inputGradient_.end(), 0.F);
        if (present_ == 0)
        {
            return inputGradient_;
        }
        for (std::size_t d = 0; d < width_; ++d)
        {
            if (mode_ == PoolMode::Max)
            {
                DistributeMaxGradient(d, outputGradient[d]);
                continue;
            }
            float const share = outputGradient[d] / static_cast<float>(present_);
            for (std::size_t entity = 0; entity < capacity_; ++entity)
            {
                if (lastMask_[entity] != 0)
                {
                    inputGradient_[entity * width_ + d] = share;
                }
            }
        }
        return inputGradient_;
    }

private:
    float MeanOf(float const* set, std::size_t d) const noexcept
    {
        float sum = 0.F;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (lastMask_[entity] != 0)
            {
                sum += set[entity * width_ + d];
            }
        }
        return sum / static_cast<float>(present_);
    }

    float MaxOf(float const* set, std::size_t d) const noexcept
    {
        float best = fill_.value_or(-std::numeric_limits<float>::infinity());
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (lastMask_[entity] != 0)
            {
                best = std::max(best, set[entity * width_ + d]);
            }
        }
        return best;
    }

    /**
     * @brief Shares the gradient of a maximum equally between the entities that reach it.
     * @details Ties are exact (several entities saturating tanh, or duplicates); sharing
     * gives the mean subgradient, as the hand-written Blue attention does.
     */
    void DistributeMaxGradient(std::size_t d, float gradient) noexcept
    {
        float const best = output_[d];
        std::size_t ties = 0;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            ties += lastMask_[entity] != 0 && lastSet_[entity * width_ + d] == best ? 1 : 0;
        }
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            if (lastMask_[entity] != 0 && lastSet_[entity * width_ + d] == best)
            {
                inputGradient_[entity * width_ + d] = gradient / static_cast<float>(ties);
            }
        }
    }

    std::size_t capacity_;
    std::size_t width_;
    PoolMode mode_;
    std::optional<float> fill_;
    std::vector<float> output_;
    std::vector<float> inputGradient_;
    float const* lastSet_ = nullptr;
    std::uint8_t const* lastMask_ = nullptr;
    std::size_t present_ = 0;
};

} // namespace ppokit::graph
