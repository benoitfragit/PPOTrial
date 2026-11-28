/**
 * @file PPOKit/tests/graph_engine_tests.cpp
 * @brief Graph engine: end-to-end gradients of a Blue-like attention actor against finite
 *        differences, fan-out accumulation, masks, shape errors and the Adam optimiser.
 *
 * The loss is L = sum_k c_k y_k with fixed random c_k, so that dL/dy = c.
 */
#include "ppokit/graph/adam.hpp"
#include "ppokit/graph/graph.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
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

std::vector<float> RandomVector(std::size_t size, std::mt19937& rng)
{
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    std::vector<float> values(size);
    for (float& value : values)
    {
        value = uniform(rng);
    }
    return values;
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

constexpr std::size_t OwnWidth = 5;
constexpr std::size_t RedCapacity = 4;
constexpr std::size_t RedFeatures = 6;
constexpr std::size_t EmbedWidth = 3;
constexpr std::size_t Actions = 2;

/**
 * @brief Small copy of the Blue actor: own features, a red set encoded per entity, an
 *        attention whose query comes from own, mean and max pooling, a trunk and a head.
 * @details The encoded red set feeds three consumers, so its gradient is a sum.
 */
Graph BuildAttentionActor()
{
    Graph graph;
    NodeId const own = graph.AddInput("own", Shape::Vector(OwnWidth));
    NodeId const red = graph.AddInput("red", Shape::Set(RedCapacity, RedFeatures));
    NodeId const embed = graph.AddNode(
        "red_embed", std::make_unique<EntityEncoderNode>(RedCapacity, RedFeatures, EmbedWidth, Activation::Tanh), {red});
    NodeId const query =
        graph.AddNode("red_query", std::make_unique<DenseNode>(OwnWidth, EmbedWidth, Activation::Tanh), {own});
    NodeId const attend =
        graph.AddNode("red_attend", std::make_unique<MaskedAttentionNode>(RedCapacity, EmbedWidth), {query, embed});
    NodeId const mean =
        graph.AddNode("red_mean", std::make_unique<MaskedPoolNode>(RedCapacity, EmbedWidth, PoolMode::Mean), {embed});
    NodeId const max =
        graph.AddNode("red_max", std::make_unique<MaskedPoolNode>(RedCapacity, EmbedWidth, PoolMode::Max), {embed});
    std::vector<std::size_t> const widths{OwnWidth, EmbedWidth, EmbedWidth, EmbedWidth};
    NodeId const features =
        graph.AddNode("features", std::make_unique<ConcatNode>(widths), {own, attend, mean, max});
    NodeId const trunk = graph.AddNode(
        "trunk", std::make_unique<MlpNode>(OwnWidth + 3 * EmbedWidth, 2, 7, Activation::Tanh), {features});
    NodeId const policy =
        graph.AddNode("policy", std::make_unique<HeadNode>(HeadKind::Gaussian, 7, Actions, -0.5F), {trunk});
    graph.AddOutput(policy);
    return graph;
}

float WeightedOutput(Graph& graph, std::vector<Signal> const& inputs, std::vector<float> const& weights)
{
    Signal const output = graph.Forward(inputs);
    double sum = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index)
    {
        sum += static_cast<double>(output.values[index]) * weights[index];
    }
    return static_cast<float>(sum);
}

/** @brief Gradients of every parameter and input of the actor match finite differences. */
void EndToEndGradients(std::vector<std::uint8_t> const& mask, char const* description)
{
    std::mt19937 rng(7);
    Graph graph = BuildAttentionActor();
    graph.Initialize(rng);
    auto own = RandomVector(OwnWidth, rng);
    auto red = RandomVector(RedCapacity * RedFeatures, rng);
    auto const weights = RandomVector(Actions, rng);
    std::vector<Signal> const inputs{{own.data(), nullptr}, {red.data(), mask.data()}};
    auto const loss = [&]() { return WeightedOutput(graph, inputs, weights); };

    loss();
    graph.ZeroGradients();
    graph.Backward(weights.data());
    auto const ownGradient = graph.OutputGradient(graph.Find("own"));
    auto const redGradient = graph.OutputGradient(graph.Find("red"));

    bool ok = true;
    for (Parameter* parameter : graph.Parameters())
    {
        auto const analytic = parameter->gradient;
        for (std::size_t index = 0; index < analytic.size(); ++index)
        {
            ok = ok && MatchesFiniteDifference(parameter->value[index], analytic[index], loss);
        }
    }
    Check(ok, description);

    ok = true;
    for (std::size_t index = 0; index < own.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(own[index], ownGradient[index], loss);
    }
    for (std::size_t index = 0; index < red.size(); ++index)
    {
        bool const absent = mask[index / RedFeatures] == 0;
        ok = ok && (!absent || redGradient[index] == 0.F);
        ok = ok && MatchesFiniteDifference(red[index], redGradient[index], loss);
    }
    Check(ok, "graph: input gradients match, absent entities receive none");
}

/** @brief The log-std of a Gaussian head is a parameter but takes no part in Forward. */
void GaussianHeadParameters()
{
    Graph graph = BuildAttentionActor();
    auto const parameters = graph.Parameters();
    // encoder 2 + query 2 + trunk 4 + head 3 (weights, biases, log-std).
    Check(parameters.size() == 11, "graph: parameter tensors in node order");
    Check(parameters.back()->value.size() == Actions && parameters.back()->value[0] == -0.5F,
        "graph: the Gaussian head owns its initial log-std");
}

/**
 * @brief A Gaussian head whose log-std is computed from its input: L = c . mean + d . log-std,
 *        gradients of the input and of both layers against finite differences; the log-std
 *        is clamped and passes no gradient outside its bounds.
 */
void StateDependentLogStd()
{
    std::mt19937 rng(11);
    constexpr std::size_t Inputs = 4;
    LogStdSettings bounds;
    bounds.stateDependent = true;
    bounds.minimum = -1.F;
    bounds.maximum = 0.5F;
    HeadNode head(HeadKind::Gaussian, Inputs, Actions, 0.F, bounds);
    head.Initialize(rng);
    auto input = RandomVector(Inputs, rng);
    auto const meanWeights = RandomVector(Actions, rng);
    auto const logStdWeights = RandomVector(Actions, rng);
    Signal const signal{input.data(), nullptr};
    auto const loss = [&]() {
        head.Forward(&signal);
        double sum = 0;
        for (std::size_t action = 0; action < Actions; ++action)
        {
            sum += static_cast<double>(head.Layer().Output()[action]) * meanWeights[action]
                + static_cast<double>(head.CurrentLogStd(action)) * logStdWeights[action];
        }
        return static_cast<float>(sum);
    };
    std::vector<Parameter*> parameters;
    head.CollectParameters(parameters);
    Check(parameters.size() == 4 && head.StateDependentLogStd() && head.LogStd().value.empty(),
        "state log-std head: means and log-std layers, no learned log-std");
    loss();
    for (Parameter* parameter : parameters)
    {
        parameter->ZeroGradient();
    }
    std::copy(logStdWeights.begin(), logStdWeights.end(), head.LogStdGradient().begin());
    std::vector<float> inputGradient(Inputs, 0.F);
    float* gradients[] = {inputGradient.data()};
    head.Backward(meanWeights.data(), gradients);
    bool ok = true;
    for (std::size_t index = 0; index < Inputs; ++index)
    {
        ok = ok && MatchesFiniteDifference(input[index], inputGradient[index], loss);
    }
    for (Parameter* parameter : parameters)
    {
        for (std::size_t index = 0; index < parameter->value.size(); ++index)
        {
            ok = ok && MatchesFiniteDifference(parameter->value[index], parameter->gradient[index], loss);
        }
    }
    Check(ok, "state log-std head: input and parameter gradients match finite differences");

    // Far outside its bounds the log-std is clamped and its layer receives nothing.
    for (float& bias : parameters[3]->value)
    {
        bias = 50.F;
    }
    head.Forward(&signal);
    Check(head.CurrentLogStd(0) == 0.5F && head.CurrentLogStd(1) == 0.5F, "state log-std: clamped to its maximum");
    parameters[2]->ZeroGradient();
    std::fill(head.LogStdGradient().begin(), head.LogStdGradient().end(), 1.F);
    head.Backward(meanWeights.data(), gradients);
    bool silent = true;
    for (float const gradient : parameters[2]->gradient)
    {
        silent = silent && gradient == 0.F;
    }
    Check(silent, "state log-std: no gradient through the clamp outside its bounds");
}

/** @brief Gradients accumulate across Backward calls until ZeroGradients. */
void GradientsAccumulate()
{
    std::mt19937 rng(3);
    Graph graph = BuildAttentionActor();
    graph.Initialize(rng);
    auto const own = RandomVector(OwnWidth, rng);
    auto const red = RandomVector(RedCapacity * RedFeatures, rng);
    std::vector<std::uint8_t> const mask{1, 1, 0, 1};
    auto const weights = RandomVector(Actions, rng);
    std::vector<Signal> const inputs{{own.data(), nullptr}, {red.data(), mask.data()}};

    graph.ZeroGradients();
    graph.Forward(inputs);
    graph.Backward(weights.data());
    auto const once = graph.Parameters().front()->gradient;
    graph.Forward(inputs);
    graph.Backward(weights.data());
    auto const twice = graph.Parameters().front()->gradient;
    bool ok = true;
    for (std::size_t index = 0; index < once.size(); ++index)
    {
        ok = ok && std::abs(twice[index] - 2.F * once[index]) <= 1e-6F + 1e-5F * std::abs(once[index]);
    }
    Check(ok, "graph: parameter gradients accumulate until ZeroGradients");
}

bool Throws(std::function<void()> const& action, std::string const& expected)
{
    try
    {
        action();
    }
    catch (std::exception const& error)
    {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

/** @brief Wrong shapes, unknown sources and duplicate names are refused with names. */
void ConstructionErrors()
{
    Check(Throws(
              []() {
                  Graph graph;
                  NodeId const own = graph.AddInput("own", Shape::Vector(4));
                  graph.AddNode("pool", std::make_unique<MaskedPoolNode>(3, 4, PoolMode::Mean), {own});
              },
              "port 'pool.in' expects set[3 x 4], node 'own' gives vector[4]"),
        "graph: a vector on a set port names the node and the port");
    Check(Throws(
              []() {
                  Graph graph;
                  NodeId const own = graph.AddInput("own", Shape::Vector(4));
                  graph.AddNode("query", std::make_unique<DenseNode>(5, 2, Activation::Tanh), {own});
              },
              "expects vector[5], node 'own' gives vector[4]"),
        "graph: a width mismatch is reported");
    Check(Throws(
              []() {
                  Graph graph;
                  graph.AddInput("own", Shape::Vector(4));
                  graph.AddInput("own", Shape::Vector(4));
              },
              "node 'own' is defined twice"),
        "graph: duplicate node names are refused");
    Check(Throws(
              []() {
                  Graph graph;
                  graph.AddNode("dense", std::make_unique<DenseNode>(4, 2, Activation::Tanh), {3});
              },
              "reads an unknown node"),
        "graph: a source must be added before its consumer");
    Check(Throws(
              []() {
                  Graph graph;
                  NodeId const own = graph.AddInput("own", Shape::Vector(4));
                  graph.AddNode("attend", std::make_unique<MaskedAttentionNode>(3, 4), {own});
              },
              "2 input port(s), 1 source(s) given"),
        "graph: every port must be wired");
    Check(Throws(
              []() {
                  Graph graph;
                  NodeId const red = graph.AddInput("red", Shape::Set(2, 3));
                  graph.AddOutput(red);
                  std::vector<float> const values(6, 0.F);
                  graph.Forward({{values.data(), nullptr}});
              },
              "input 1 needs values and a presence mask"),
        "graph: a set input requires its mask");
    Check(Throws([]() { HeadNode head(HeadKind::Value, 4, 2); }, "exactly one output"),
        "graph: a value head has one output");
}

/** @brief Two heads on a shared trunk (Gaussian controls and a Bernoulli gear, as in Flight). */
void TwoHeadsShareTheirTrunk()
{
    std::mt19937 rng(11);
    Graph graph;
    NodeId const in = graph.AddInput("in", Shape::Vector(4));
    NodeId const trunk = graph.AddNode("trunk", std::make_unique<MlpNode>(4, 1, 6, Activation::Tanh), {in});
    NodeId const controls = graph.AddNode("controls", std::make_unique<HeadNode>(HeadKind::Gaussian, 6, 3), {trunk});
    NodeId const gear = graph.AddNode("gear", std::make_unique<HeadNode>(HeadKind::Bernoulli, 6, 1), {trunk});
    graph.AddOutput(controls);
    graph.AddOutput(gear);
    graph.Initialize(rng);
    auto input = RandomVector(4, rng);
    auto const controlWeights = RandomVector(3, rng);
    auto const gearWeights = RandomVector(1, rng);
    std::vector<Signal> const inputs{{input.data(), nullptr}};
    auto const loss = [&]() {
        graph.Forward(inputs);
        double sum = static_cast<double>(graph.Output(1).values[0]) * gearWeights[0];
        for (std::size_t index = 0; index < 3; ++index)
        {
            sum += static_cast<double>(graph.Output(0).values[index]) * controlWeights[index];
        }
        return static_cast<float>(sum);
    };
    loss();
    graph.ZeroGradients();
    float const* const gradients[] = {controlWeights.data(), gearWeights.data()};
    graph.Backward(gradients, 2);
    bool ok = graph.OutputCount() == 2;
    for (Parameter* parameter : graph.Parameters())
    {
        auto const analytic = parameter->gradient;
        for (std::size_t index = 0; index < analytic.size(); ++index)
        {
            ok = ok && MatchesFiniteDifference(parameter->value[index], analytic[index], loss);
        }
    }
    Check(ok, "graph: two heads, the trunk receives the sum of both gradients");

    graph.ZeroGradients();
    graph.Forward(inputs);
    float const* const gearOnly[] = {nullptr, gearWeights.data()};
    graph.Backward(gearOnly, 2);
    bool untouched = true;
    for (float const value : dynamic_cast<HeadNode&>(graph.At(controls)).Layer().Weights().gradient)
    {
        untouched = untouched && value == 0.F;
    }
    Check(untouched, "graph: an output without gradient leaves its head untouched");
    Check(Throws([&]() { graph.Backward(gearWeights.data()); }, "one gradient per output"),
        "graph: the single-output Backward refuses a two-head graph");
}

/** @brief select (with a repeated index) and concat_sets (masks carried over) differentiate correctly. */
void SelectAndConcatSets()
{
    std::mt19937 rng(23);
    Graph graph;
    NodeId const own = graph.AddInput("own", Shape::Vector(4));
    NodeId const tracks = graph.AddInput("tracks", Shape::Set(2, 3));
    NodeId const objective = graph.AddInput("objective", Shape::Set(1, 3));
    NodeId const picked =
        graph.AddNode("picked", std::make_unique<SelectNode>(4, std::vector<std::size_t>{2, 0, 2}), {own});
    NodeId const friends = graph.AddNode(
        "friends", std::make_unique<ConcatSetsNode>(std::vector<std::size_t>{2, 1}, 3), {tracks, objective});
    NodeId const pooled = graph.AddNode("pooled", std::make_unique<MaskedPoolNode>(3, 3, PoolMode::Mean), {friends});
    NodeId const joined =
        graph.AddNode("joined", std::make_unique<ConcatNode>(std::vector<std::size_t>{3, 3}), {picked, pooled});
    NodeId const value = graph.AddNode("value", std::make_unique<HeadNode>(HeadKind::Value, 6, 1), {joined});
    graph.AddOutput(value);
    graph.Initialize(rng);

    auto ownValues = RandomVector(4, rng);
    auto trackValues = RandomVector(6, rng);
    auto objectiveValues = RandomVector(3, rng);
    std::vector<std::uint8_t> const trackMask{1, 0};
    std::vector<std::uint8_t> const objectiveMask{1};
    std::vector<Signal> const inputs{{ownValues.data(), nullptr}, {trackValues.data(), trackMask.data()},
        {objectiveValues.data(), objectiveMask.data()}};
    std::vector<float> const weight{1.F};
    auto const loss = [&]() { return WeightedOutput(graph, inputs, weight); };
    loss();
    Signal const merged = graph.OutputOf(friends);
    Check(merged.mask[0] == 1 && merged.mask[1] == 0 && merged.mask[2] == 1, "concat_sets: masks follow the values");
    graph.ZeroGradients();
    graph.Backward(weight.data());
    auto const ownGradient = graph.OutputGradient(own);
    auto const trackGradient = graph.OutputGradient(tracks);
    auto const objectiveGradient = graph.OutputGradient(objective);
    bool ok = ownGradient[1] == 0.F && ownGradient[3] == 0.F;
    for (std::size_t index = 0; index < ownValues.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(ownValues[index], ownGradient[index], loss);
    }
    for (std::size_t index = 0; index < trackValues.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(trackValues[index], trackGradient[index], loss);
    }
    for (std::size_t index = 0; index < objectiveValues.size(); ++index)
    {
        ok = ok && MatchesFiniteDifference(objectiveValues[index], objectiveGradient[index], loss);
    }
    Check(ok, "select and concat_sets: input gradients match finite differences");
    Check(Throws([]() { SelectNode node(3, {0, 3}); }, "index 3 is outside an input of width 3"),
        "select: an index outside the input is refused");
}

/** @brief Adam drives a dense layer to a target output. */
void AdamFitsTarget()
{
    std::mt19937 rng(5);
    Graph graph;
    NodeId const in = graph.AddInput("in", Shape::Vector(3));
    NodeId const head = graph.AddNode("head", std::make_unique<HeadNode>(HeadKind::Value, 3, 1), {in});
    graph.AddOutput(head);
    graph.Initialize(rng);
    std::vector<float> const input{0.5F, -1.F, 2.F};
    std::vector<Signal> const inputs{{input.data(), nullptr}};
    constexpr float Target = 1.5F;
    AdamSettings settings;
    settings.learningRate = 0.05F;
    Adam adam(graph.Parameters(), settings);
    float error = 0.F;
    for (int step = 0; step < 300; ++step)
    {
        error = graph.Forward(inputs).values[0] - Target;
        graph.ZeroGradients();
        std::vector<float> const gradient{2.F * error};
        graph.Backward(gradient.data());
        adam.Step();
    }
    Check(std::abs(error) < 1e-3F && adam.StepCount() == 300, "adam: a value head reaches its target");
}

/** @brief A bad last component leaves every parameter, moment and counter unchanged. */
void AdamRejectsNonFiniteSteps()
{
    Parameter first(1);
    Parameter last(1);
    Adam optimizer({&first, &last}, AdamSettings{});
    first.gradient[0] = 1.F;
    last.gradient[0] = -2.F;
    optimizer.Step();
    auto const before = optimizer.Save();
    float const firstValue = first.value[0];
    float const lastValue = last.value[0];
    for (float gradient : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::max()})
    {
        last.gradient[0] = gradient;
        Check(Throws([&] { optimizer.Step(); }, "non-finite"), "reject non-finite and overflowing Adam steps");
        if (!std::isfinite(gradient))
        {
            Check(Throws([&] { optimizer.ClipGradientNorm(1.0); }, "gradient norm"), "refuse invalid norm");
        }
        auto const after = optimizer.Save();
        Check(after.steps == before.steps && after.first == before.first && after.second == before.second
                && first.value[0] == firstValue && last.value[0] == lastValue,
            "an invalid step cannot commit partial changes");
    }
}

/** @brief Global gradient norm clipping, as torch's clip_grad_norm_: scaled once over every tensor. */
void AdamClipsGradientNorm()
{
    Parameter first;
    Parameter second;
    first.value = {0.F, 0.F};
    first.gradient = {3.F, 0.F};
    second.value = {0.F};
    second.gradient = {4.F};
    Adam adam({&first, &second}, AdamSettings{});
    double const norm = adam.ClipGradientNorm(1.0);
    float const factor = static_cast<float>(1.0 / (5.0 + Adam::ClipNormEpsilon));
    Check(std::abs(norm - 5.0) < 1e-12 && first.gradient[0] == 3.F * factor && second.gradient[0] == 4.F * factor
            && first.gradient[1] == 0.F,
        "adam: a gradient above the limit is scaled to it, over all tensors together");
    std::vector<float> const kept{first.gradient[0], second.gradient[0]};
    adam.ClipGradientNorm(10.0);
    Check(first.gradient[0] == kept[0] && second.gradient[0] == kept[1], "adam: a gradient below the limit is kept");
}

} // namespace

int main()
{
    EndToEndGradients({1, 0, 1, 1}, "graph: attention actor parameter gradients match finite differences");
    EndToEndGradients({0, 0, 0, 0}, "graph: empty set, parameter gradients still match");
    GaussianHeadParameters();
    StateDependentLogStd();
    GradientsAccumulate();
    ConstructionErrors();
    TwoHeadsShareTheirTrunk();
    SelectAndConcatSets();
    AdamFitsTarget();
    AdamRejectsNonFiniteSteps();
    AdamClipsGradientNorm();
    if (failures != 0)
    {
        std::cerr << failures << " graph engine check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit graph engine tests passed\n";
    return 0;
}
