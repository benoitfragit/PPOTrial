/**
 * @file Examples/BlueDefense/tests/graph_actor_parity_tests.cpp
 * @brief The production Blue actor (computed by the PPOKit block graph of
 *        graph_blocks.json) matches the former hand-written actor, kept as a test oracle.
 *
 * Both actors share the same rl_tools storage and are initialized from the same seed, so
 * they hold identical weights. A full training batch (0 to 10 Reds, 0 to 9 friendly
 * entities, rotated masks) then goes through the public rl_tools API of each: inference,
 * the compact v7 rows read by the centralized critic, training forward, and backward with
 * random output derivatives. Every gradient tensor is compared.
 *
 * Tolerances are relative to the largest magnitude of each tensor: the two implementations
 * sum in different orders in float, so they agree to rounding, not bit for bit.
 */
#include "reference/hand_written_attention_actor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>

namespace blue = ppo::examples::bluedefense::defenders;
namespace rlt = rl_tools;

namespace {

using RawLayout = blue::GeneralizedObservationLayout;
using LegacyLayout = blue::ObservationLayout;
using ActorSpecification = blue::TrainingConfig::NN::ACTOR_SPEC;
using ProductionActor = blue::GeneralizedAttentionActor<ActorSpecification>;
using ReferenceActor = blue::reference::GeneralizedAttentionActor<ActorSpecification>;
constexpr blue::Index Batch = ActorSpecification::BATCH_SIZE;
constexpr blue::Index Actions = LegacyLayout::ActionSize;
constexpr blue::Index Team = blue::TeamCapacity::MaximumBlueMobiles;
/** @brief Relative tolerance on outputs (float, different summation orders). */
constexpr double OutputTolerance = 1e-5;
/** @brief Relative tolerance on gradients summed over the batch. */
constexpr double GradientTolerance = 1e-4;
constexpr unsigned SharedSeed = 4242U;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief Largest difference between two rl_tools matrices, relative to the largest reference value. */
template<typename Expected, typename Actual>
double RelativeError(Expected const& expected, Actual const& actual)
{
    double largest = 1e-12;
    double error = 0;
    for (blue::Index row = 0; row < Expected::SPEC::ROWS; ++row)
    {
        for (blue::Index column = 0; column < Expected::SPEC::COLS; ++column)
        {
            double const reference = rlt::get(expected, row, column);
            largest = std::max(largest, std::abs(reference));
            error = std::max(error, std::abs(reference - rlt::get(actual, row, column)));
        }
    }
    return error / largest;
}

/** @brief Sets the presence flag of `present` of the `tracks` entities of a group, the rest absent. */
template<typename Matrix>
void SetMasks(Matrix& input, blue::Index row, std::size_t offset, std::size_t tracks, std::size_t features,
    std::size_t present, std::size_t rotation)
{
    for (std::size_t track = 0; track < tracks; ++track)
    {
        bool const isPresent = (track + rotation) % tracks < present;
        rlt::set(input, row, static_cast<blue::Index>(blue::GeneralizedTrackMaskIndex(offset, track, features)),
            isPresent ? 1.F : 0.F);
    }
}

template<typename Matrix>
void FillObservations(Matrix& input, std::mt19937& rng)
{
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    constexpr std::size_t FriendlyTracks = RawLayout::BlueTracks + RawLayout::ObjectiveTracks;
    for (blue::Index row = 0; row < Matrix::SPEC::ROWS; ++row)
    {
        for (blue::Index column = 0; column < Matrix::SPEC::COLS; ++column)
        {
            rlt::set(input, row, column, uniform(rng));
        }
        SetMasks(input, row, RawLayout::RedOffset, RawLayout::RedTracks, RawLayout::RedFeatures,
            row % (RawLayout::RedTracks + 1), row);
        SetMasks(input, row, RawLayout::BlueOffset, FriendlyTracks, RawLayout::FriendlyFeatures,
            (row / 3) % (FriendlyTracks + 1), row / 2);
    }
}

/** @brief The two actors in rl_tools storage, initialized identically. */
struct Actors final
{
    blue::Device device{};
    std::unique_ptr<ProductionActor> production = std::make_unique<ProductionActor>();
    std::unique_ptr<ReferenceActor> reference = std::make_unique<ReferenceActor>();

    Actors()
    {
        rlt::malloc(device, *production);
        rlt::malloc(device, *reference);
        auto productionEngine = rlt::random::default_engine(typename blue::Device::SPEC::RANDOM{}, SharedSeed);
        auto referenceEngine = rlt::random::default_engine(typename blue::Device::SPEC::RANDOM{}, SharedSeed);
        rlt::init_weights(device, *production, productionEngine);
        rlt::init_weights(device, *reference, referenceEngine);
    }
    ~Actors()
    {
        rlt::free(device, *reference);
        rlt::free(device, *production);
    }
    Actors(Actors const&) = delete;
    Actors& operator=(Actors const&) = delete;
};

/** @brief Means of a team with the formation-slot weight of the first trunk unit shifted by `delta`. */
template<typename Input, typename Output, typename Buffer>
void EvaluateWithShiftedWeight(Actors& actors, blue::Index column, float delta, Input const& input, Output& output,
    Buffer& buffer)
{
    auto& weights = actors.production->input_layer.weights.parameters;
    float const original = rlt::get(weights, 0, column);
    rlt::set(weights, 0, column, original + delta);
    rlt::evaluate(actors.device, *actors.production, input, output, buffer);
    rlt::set(weights, 0, column, original);
}

/** @brief Inference on a team, and the compact rows from which the critic input is built. */
void InferenceAndCompact(Actors& actors)
{
    using RawMatrix = rlt::MatrixDynamic<rlt::matrix::Specification<float, blue::Index, Team, RawLayout::Size>>;
    using MeanMatrix = rlt::MatrixDynamic<rlt::matrix::Specification<float, blue::Index, Team, Actions>>;
    using CompactMatrix =
        rlt::MatrixDynamic<rlt::matrix::Specification<float, blue::Index, Team, LegacyLayout::Size>>;
    RawMatrix input{};
    MeanMatrix production{};
    MeanMatrix reference{};
    MeanMatrix withCompact{};
    CompactMatrix productionCompact{};
    CompactMatrix referenceCompact{};
    rlt::malloc(actors.device, input);
    rlt::malloc(actors.device, production);
    rlt::malloc(actors.device, reference);
    rlt::malloc(actors.device, withCompact);
    rlt::malloc(actors.device, productionCompact);
    rlt::malloc(actors.device, referenceCompact);
    std::mt19937 rng(7);
    FillObservations(input, rng);

    typename ProductionActor::template Buffer<Team> productionBuffer{};
    typename ReferenceActor::template Buffer<Team> referenceBuffer{};
    rlt::malloc(actors.device, productionBuffer);
    rlt::malloc(actors.device, referenceBuffer);
    rlt::evaluate(actors.device, *actors.production, input, production, productionBuffer);
    rlt::evaluate(actors.device, *actors.reference, input, reference, referenceBuffer);
    rlt::blue_generalized_attention_detail::evaluate_with_compact(
        actors.device, *actors.production, input, withCompact, productionCompact);
    rlt::blue_reference_attention_detail::transform(actors.device, *actors.reference, input, referenceCompact);

    double const meanError = RelativeError(reference, production);
    double const compactError = RelativeError(referenceCompact, productionCompact);
    std::cout << "inference over " << Team << " agents: means " << meanError << ", compact rows " << compactError
              << " (relative error)\n";
    Check(meanError <= OutputTolerance, "inference means match the hand-written actor");
    Check(RelativeError(production, withCompact) == 0.0, "evaluate_with_compact gives the evaluate means");
    Check(compactError <= OutputTolerance, "the compact rows read by the critic match");

    // Sensitivity: the formation slot (role[1]) feeds the trunk, a central column never does.
    EvaluateWithShiftedWeight(actors, static_cast<blue::Index>(blue::GeneralizedCompactFeatures::FormationSlot), 0.5F,
        input, production, productionBuffer);
    Check(RelativeError(reference, production) > 1e-3, "the test detects a change of the formation slot weight");
    EvaluateWithShiftedWeight(actors, static_cast<blue::Index>(LegacyLayout::CentralOffset), 0.5F, input,
        production, productionBuffer);
    Check(RelativeError(reference, production) <= OutputTolerance,
        "central compact columns stay unused, as in the hand-written actor");

    rlt::free(actors.device, referenceBuffer);
    rlt::free(actors.device, productionBuffer);
    rlt::free(actors.device, referenceCompact);
    rlt::free(actors.device, productionCompact);
    rlt::free(actors.device, withCompact);
    rlt::free(actors.device, reference);
    rlt::free(actors.device, production);
    rlt::free(actors.device, input);
}

/** @brief Training forward and backward over a full batch: outputs and every gradient. */
void TrainingGradients(Actors& actors)
{
    using RawMatrix = rlt::MatrixDynamic<rlt::matrix::Specification<float, blue::Index, Batch, RawLayout::Size>>;
    using OutputMatrix = rlt::MatrixDynamic<rlt::matrix::Specification<float, blue::Index, Batch, Actions>>;
    RawMatrix input{};
    OutputMatrix production{};
    OutputMatrix reference{};
    OutputMatrix derivative{};
    rlt::malloc(actors.device, input);
    rlt::malloc(actors.device, production);
    rlt::malloc(actors.device, reference);
    rlt::malloc(actors.device, derivative);
    std::mt19937 rng(99);
    FillObservations(input, rng);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    for (blue::Index row = 0; row < Batch; ++row)
    {
        for (blue::Index action = 0; action < Actions; ++action)
        {
            rlt::set(derivative, row, action, uniform(rng));
        }
    }

    typename ProductionActor::template Buffer<Batch> productionBuffer{};
    typename ReferenceActor::template Buffer<Batch> referenceBuffer{};
    rlt::malloc(actors.device, productionBuffer);
    rlt::malloc(actors.device, referenceBuffer);
    rlt::zero_gradient(actors.device, *actors.production);
    rlt::zero_gradient(actors.device, *actors.reference);
    rlt::forward(actors.device, *actors.production, input, production);
    rlt::forward(actors.device, *actors.reference, input, reference);
    rlt::backward(actors.device, *actors.production, input, derivative, productionBuffer);
    rlt::backward(actors.device, *actors.reference, input, derivative, referenceBuffer);

    double const outputError = RelativeError(reference, production);
    std::cout << "training batch of " << Batch << ": outputs " << outputError << '\n';
    Check(outputError <= OutputTolerance, "training forward matches the hand-written actor");

    ProductionActor const& p = *actors.production;
    ReferenceActor const& r = *actors.reference;
    struct GradientCheck final
    {
        char const* name;
        double error;
    };
    GradientCheck const gradients[] = {
        {"red encoder weights", RelativeError(r.red.encoderW.gradient, p.red.encoderW.gradient)},
        {"red encoder biases", RelativeError(r.red.encoderB.gradient, p.red.encoderB.gradient)},
        {"red query weights", RelativeError(r.red.queryW.gradient, p.red.queryW.gradient)},
        {"red query biases", RelativeError(r.red.queryB.gradient, p.red.queryB.gradient)},
        {"friendly encoder weights", RelativeError(r.friendlies.encoderW.gradient, p.friendlies.encoderW.gradient)},
        {"friendly encoder biases", RelativeError(r.friendlies.encoderB.gradient, p.friendlies.encoderB.gradient)},
        {"friendly query weights", RelativeError(r.friendlies.queryW.gradient, p.friendlies.queryW.gradient)},
        {"friendly query biases", RelativeError(r.friendlies.queryB.gradient, p.friendlies.queryB.gradient)},
        {"trunk input layer weights", RelativeError(r.input_layer.weights.gradient, p.input_layer.weights.gradient)},
        {"trunk input layer biases", RelativeError(r.input_layer.biases.gradient, p.input_layer.biases.gradient)},
        {"trunk hidden layer weights",
            RelativeError(r.hidden_layers[0].weights.gradient, p.hidden_layers[0].weights.gradient)},
        {"output layer weights", RelativeError(r.output_layer.weights.gradient, p.output_layer.weights.gradient)},
        {"output layer biases", RelativeError(r.output_layer.biases.gradient, p.output_layer.biases.gradient)},
    };
    for (GradientCheck const& gradient : gradients)
    {
        std::cout << "  " << gradient.name << ": relative error " << gradient.error << '\n';
        Check(gradient.error <= GradientTolerance, std::string("same gradient: ") + gradient.name);
    }

    rlt::free(actors.device, referenceBuffer);
    rlt::free(actors.device, productionBuffer);
    rlt::free(actors.device, derivative);
    rlt::free(actors.device, reference);
    rlt::free(actors.device, production);
    rlt::free(actors.device, input);
}

} // namespace

int main()
{
    try
    {
        Actors actors;
        bool const sameWeights = rlt::get(actors.production->red.encoderW.parameters, 0, 0)
                == rlt::get(actors.reference->red.encoderW.parameters, 0, 0)
            && rlt::get(actors.production->input_layer.weights.parameters, 3, 5)
                == rlt::get(actors.reference->input_layer.weights.parameters, 3, 5);
        Check(sameWeights, "one seed gives both actors the same weights");
        InferenceAndCompact(actors);
        TrainingGradients(actors);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0)
    {
        std::cerr << failures << " Blue graph parity check(s) failed\n";
        return 1;
    }
    std::cout << "Blue graph actor parity tests passed\n";
    return 0;
}
