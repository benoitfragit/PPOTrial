/**
 * @file PPOKit/tests/graph_blocks_tests.cpp
 * @brief Graph blocks: exact gradients against finite differences, masks and empty sets.
 *
 * Each check uses the scalar loss L = sum_k c_k y_k with fixed random c_k, so that
 * dL/dy = c. Inputs and parameters are perturbed one at a time.
 */
#include "ppokit/graph/blocks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ppokit::graph;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::vector<float> RandomVector(std::size_t size, std::mt19937& rng, float scale = 1.F)
{
    std::uniform_real_distribution<float> uniform(-scale, scale);
    std::vector<float> values(size);
    for (float& value : values)
    {
        value = uniform(rng);
    }
    return values;
}

float Weighted(std::vector<float> const& output, std::vector<float> const& weights)
{
    double sum = 0;
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        sum += static_cast<double>(output[index]) * weights[index];
    }
    return static_cast<float>(sum);
}

/** @brief Compares an analytic derivative with a centred finite difference of `loss` in `value`. */
bool MatchesFiniteDifference(float& value, float analytic, std::function<float()> const& loss)
{
    constexpr float Step = 1e-3F;
    float const original = value;
    value = original + Step;
    float const plus = loss();
    value = original - Step;
    float const minus = loss();
    value = original;
    float const numerical = (plus - minus) / (2 * Step);
    return std::abs(numerical - analytic) <= 2e-3F + 2e-2F * std::abs(numerical);
}

void DenseGradients()
{
    std::mt19937 rng(1);
    for (Activation const activation : {Activation::Tanh, Activation::Relu, Activation::Identity})
    {
        Dense dense(4, 3, activation);
        dense.Initialize(rng);
        auto input = RandomVector(4, rng);
        auto const weights = RandomVector(3, rng);
        auto const loss = [&]() { return Weighted(dense.Forward(input.data()), weights); };
        loss();
        dense.Weights().ZeroGradient();
        dense.Biases().ZeroGradient();
        auto const inputGradient = dense.Backward(weights.data());
        bool ok = true;
        for (std::size_t index = 0; index < input.size(); ++index)
        {
            ok = ok && MatchesFiniteDifference(input[index], inputGradient[index], loss);
        }
        auto const weightGradient = dense.Weights().gradient;
        for (std::size_t index = 0; index < weightGradient.size(); ++index)
        {
            ok = ok && MatchesFiniteDifference(dense.Weights().value[index], weightGradient[index], loss);
        }
        auto const biasGradient = dense.Biases().gradient;
        for (std::size_t index = 0; index < biasGradient.size(); ++index)
        {
            ok = ok && MatchesFiniteDifference(dense.Biases().value[index], biasGradient[index], loss);
        }
        Check(ok, "dense: input, weight and bias gradients");
    }
}

void EntityEncoderGradients()
{
    std::mt19937 rng(2);
    EntityEncoder encoder(3, 4, 5, Activation::Tanh);
    encoder.Initialize(rng);
    auto set = RandomVector(3 * 4, rng);
    std::vector<std::uint8_t> const mask{1, 0, 1};
    auto const weights = RandomVector(3 * 5, rng);
    auto const loss = [&]() { return Weighted(encoder.Forward(set.data(), mask.data()), weights); };
    auto const output = encoder.Forward(set.data(), mask.data());
    Check(output[5] == 0.F && output[9] == 0.F, "an absent entity encodes to zeros");
    encoder.Layer().Weights().ZeroGradient();
    auto const inputGradient = encoder.Backward(weights.data());
    bool ok = true;
    for (std::size_t index = 0; index < set.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(set[index], inputGradient[index], loss);
    }
    Check(ok, "entity encoder: input gradients");
    Check(inputGradient[4] == 0.F && inputGradient[7] == 0.F, "an absent entity receives no gradient");
    auto const weightGradient = encoder.Layer().Weights().gradient;
    ok = true;
    for (std::size_t index = 0; index < weightGradient.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(encoder.Layer().Weights().value[index], weightGradient[index], loss);
    }
    Check(ok, "entity encoder: shared weights accumulate the gradient of every present entity");
}

void AttentionGradients()
{
    std::mt19937 rng(3);
    MaskedAttention attention(4, 3);
    auto query = RandomVector(3, rng, 2.F);
    auto set = RandomVector(4 * 3, rng, 2.F);
    std::vector<std::uint8_t> const mask{1, 1, 0, 1};
    auto const weights = RandomVector(3, rng);
    auto const loss = [&]() { return Weighted(attention.Forward(query.data(), set.data(), mask.data()), weights); };
    loss();
    double probabilitySum = 0;
    for (float const probability : attention.Weights())
    {
        probabilitySum += probability;
    }
    Check(std::abs(probabilitySum - 1.0) < 1e-6 && attention.Weights()[2] == 0.F,
        "attention weights sum to one over present entities only");
    attention.Backward(weights.data());
    auto const queryGradient = attention.QueryGradient();
    auto const setGradient = attention.SetGradient();
    bool ok = true;
    for (std::size_t index = 0; index < query.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(query[index], queryGradient[index], loss);
    }
    for (std::size_t index = 0; index < set.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(set[index], setGradient[index], loss);
    }
    Check(ok, "attention: query and entity gradients");
    std::vector<std::uint8_t> const empty(4, 0);
    auto const nothing = attention.Forward(query.data(), set.data(), empty.data());
    Check(nothing[0] == 0.F && nothing[1] == 0.F && nothing[2] == 0.F, "an empty set attends to zeros");
}

void AttentionWithKeysGradients()
{
    std::mt19937 rng(5);
    MaskedAttention attention(4, 3);
    auto query = RandomVector(3, rng, 2.F);
    auto keys = RandomVector(4 * 3, rng, 2.F);
    auto set = RandomVector(4 * 3, rng, 2.F);
    std::vector<std::uint8_t> const mask{1, 0, 1, 1};
    auto const weights = RandomVector(3, rng);
    auto const loss = [&]() {
        return Weighted(attention.Forward(query.data(), keys.data(), set.data(), mask.data()), weights);
    };
    loss();
    attention.Backward(weights.data());
    auto const queryGradient = attention.QueryGradient();
    auto const keyGradient = attention.KeyGradient();
    auto const setGradient = attention.SetGradient();
    bool ok = true;
    for (std::size_t index = 0; index < query.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(query[index], queryGradient[index], loss);
    }
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(keys[index], keyGradient[index], loss);
        ok = ok && MatchesFiniteDifference(set[index], setGradient[index], loss);
    }
    Check(ok, "attention with separate keys: query, key and value gradients");
}

void PoolGradients()
{
    std::mt19937 rng(4);
    for (PoolMode const mode : {PoolMode::Mean, PoolMode::Max})
    {
        MaskedPool pool(4, 3, mode);
        auto set = RandomVector(4 * 3, rng);
        std::vector<std::uint8_t> const mask{0, 1, 1, 1};
        set[0] = 100.F;  // an absent entity must not win the maximum
        auto const weights = RandomVector(3, rng);
        auto const loss = [&]() { return Weighted(pool.Forward(set.data(), mask.data()), weights); };
        auto const output = pool.Forward(set.data(), mask.data());
        Check(output[0] < 100.F, "pooling ignores absent entities");
        auto const inputGradient = pool.Backward(weights.data());
        bool ok = true;
        for (std::size_t index = 3; index < set.size(); ++index)
        {
            ok = ok && MatchesFiniteDifference(set[index], inputGradient[index], loss);
        }
        Check(ok, mode == PoolMode::Mean ? "mean pooling gradients" : "max pooling gradients");
        std::vector<std::uint8_t> const empty(4, 0);
        Check(pool.Forward(set.data(), empty.data())[1] == 0.F, "pooling an empty set gives zeros");
    }

    // Two entities saturate at the same maximum: each receives half of the gradient.
    MaskedPool pool(3, 1, PoolMode::Max);
    std::vector<float> const tied{1.F, 0.2F, 1.F};
    std::vector<std::uint8_t> const all{1, 1, 1};
    pool.Forward(tied.data(), all.data());
    float const gradient = 2.F;
    auto const shared = pool.Backward(&gradient);
    Check(shared[0] == 1.F && shared[1] == 0.F && shared[2] == 1.F, "max pooling shares the gradient between ties");
}

/**
 * @brief LibTorch masking by fill value: encoders that encode absent entities, a maximum
 *        that counts them as the fill, an attention that adds the fill to their scores.
 */
void MaskFillBlocks()
{
    std::mt19937 rng(6);
    std::vector<std::uint8_t> const someAbsent{1, 0, 1, 0};
    std::vector<std::uint8_t> const empty(4, 0);

    EntityEncoder encoder(4, 3, 2, Activation::Relu, true);
    encoder.Initialize(rng);
    auto features = RandomVector(4 * 3, rng);
    std::vector<float> row(2);
    std::vector<float> preActivation(2);
    encoder.Layer().ForwardRow(features.data() + 3, preActivation.data(), row.data());
    auto const encoded = encoder.Forward(features.data(), someAbsent.data());
    Check(encoded[2] == row[0] && encoded[3] == row[1], "encode_absent: an absent entity is encoded too");
    auto const encoderWeights = RandomVector(4 * 2, rng);
    auto const encoderLoss = [&]() { return Weighted(encoder.Forward(features.data(), someAbsent.data()), encoderWeights); };
    encoderLoss();
    auto const featureGradient = encoder.Backward(encoderWeights.data());
    bool ok = true;
    for (std::size_t index = 0; index < features.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(features[index], featureGradient[index], encoderLoss);
    }
    Check(ok, "encode_absent: every entity receives its gradient");

    MaskedPool filled(4, 3, PoolMode::Max, -1e9F);
    MaskedPool plain(4, 3, PoolMode::Max);
    auto set = RandomVector(4 * 3, rng);
    Check(filled.Forward(set.data(), someAbsent.data()) == plain.Forward(set.data(), someAbsent.data()),
        "a filled maximum over present entities is their maximum");
    Check(filled.Forward(set.data(), empty.data())[0] == -1e9F, "a filled maximum of an empty set is the fill");
    auto const poolWeights = RandomVector(3, rng);
    auto const poolLoss = [&]() { return Weighted(filled.Forward(set.data(), someAbsent.data()), poolWeights); };
    poolLoss();
    auto const poolGradient = filled.Backward(poolWeights.data());
    ok = true;
    for (std::size_t index = 0; index < set.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(set[index], poolGradient[index], poolLoss);
    }
    Check(ok, "filled maximum: gradients");
    bool refused = false;
    try
    {
        MaskedPool mean(4, 3, PoolMode::Mean, -1e9F);
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "a fill value applies to a maximum only");

    auto query = RandomVector(3, rng, 2.F);
    auto keys = RandomVector(4 * 3, rng, 2.F);
    MaskedAttention torchLike(4, 3, -1e9F);
    MaskedAttention masked(4, 3);
    auto const withFill = torchLike.Forward(query.data(), keys.data(), set.data(), someAbsent.data());
    auto const withMask = masked.Forward(query.data(), keys.data(), set.data(), someAbsent.data());
    ok = true;
    for (std::size_t d = 0; d < 3; ++d)
    {
        ok = ok && std::abs(withFill[d] - withMask[d]) <= 1e-6F;
    }
    Check(ok && torchLike.Weights()[1] == 0.F, "beside a present entity, a filled absent one weighs nothing");

    // With no present entity: the softmax of q.k/sqrt(d) + fill over every entity.
    auto const none = torchLike.Forward(query.data(), keys.data(), set.data(), empty.data());
    std::vector<float> expected(3, 0.F);
    std::vector<float> scores(4);
    float largest = -1e30F;
    for (std::size_t entity = 0; entity < 4; ++entity)
    {
        float dot = 0.F;
        for (std::size_t d = 0; d < 3; ++d)
        {
            dot += query[d] * keys[entity * 3 + d];
        }
        scores[entity] = dot / std::sqrt(3.F) + -1e9F;
        largest = std::max(largest, scores[entity]);
    }
    float total = 0.F;
    for (float& score : scores)
    {
        score = std::exp(score - largest);
        total += score;
    }
    for (std::size_t entity = 0; entity < 4; ++entity)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            expected[d] += scores[entity] / total * set[entity * 3 + d];
        }
    }
    Check(none == expected, "a filled attention over an empty set computes LibTorch's softmax in float");

    // Gradients through the filled softmax, with a fill small enough to keep absent weights.
    MaskedAttention soft(4, 3, -1.5F);
    auto const softWeights = RandomVector(3, rng);
    auto const softLoss = [&]() {
        return Weighted(soft.Forward(query.data(), keys.data(), set.data(), someAbsent.data()), softWeights);
    };
    softLoss();
    soft.Backward(softWeights.data());
    auto const queryGradient = soft.QueryGradient();
    auto const keyGradient = soft.KeyGradient();
    auto const setGradient = soft.SetGradient();
    ok = soft.Weights()[1] > 0.F;
    for (std::size_t index = 0; index < query.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(query[index], queryGradient[index], softLoss);
    }
    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(keys[index], keyGradient[index], softLoss);
        ok = ok && MatchesFiniteDifference(set[index], setGradient[index], softLoss);
    }
    Check(ok, "filled attention: gradients of every entity, absent ones included");
}

} // namespace

int main()
{
    DenseGradients();
    EntityEncoderGradients();
    AttentionGradients();
    AttentionWithKeysGradients();
    PoolGradients();
    MaskFillBlocks();
    if (failures != 0)
    {
        std::cerr << failures << " graph block check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit graph block tests passed\n";
    return EXIT_SUCCESS;
}
