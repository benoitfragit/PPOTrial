/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_attention_actor.hpp
 * @brief Shared Blue actor: rl_tools parameter storage, computed by the PPOKit block graph.
 * @details The parameters (and their Adam moments) keep the historical rl_tools layout,
 * so checkpoints, migrations and the production model are unchanged. The computation is
 * the block graph of graph_blocks.json (GeneralizedActorGraph): masked attention, mean
 * and max pooling over red and friendly tracks, then the compact v7 trunk. Before every
 * pass the weights are copied into the graph; after a backward pass the graph gradients
 * are added to the rl_tools gradients, which rl_tools Adam then applies.
 *
 * graph_actor_parity_tests checks on every build that this actor computes the means and
 * gradients of the former hand-written implementation (tests/reference).
 */
#pragma once

#include "generalized_actor_graph.hpp"
#include "generalized_observation.hpp"

// Establish CPU and model operations before defining RLtools adapters.
#include "patrol_rltools.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ppo::examples::bluedefense::defenders {

namespace rlt = rl_tools;

/**
 * @brief Parameters of one attention block (encoder and query), in rl_tools storage.
 * @tparam T Scalar type.
 * @tparam TI Index type.
 * @tparam F Features per raw track; the final feature is the mask.
 * @tparam N Raw track capacity.
 * @tparam D Learned embedding width.
 * @details The block itself is computed by the graph (entity_encoder, dense query,
 * masked_attention, masked_pool); these matrices are what checkpoints store.
 */
template<typename T, typename TI, TI F, TI N, TI D>
struct GeneralizedTrackAttention final
{
    static constexpr TI Features = F;
    static constexpr TI Tracks = N;
    static constexpr TI Width = D;

    template<TI Rows, TI Cols, typename Kind>
    using Parameter = typename rlt::nn::parameters::Adam::template instance<
        typename rlt::nn::parameters::Plain::template spec<
            rlt::MatrixDynamic<rlt::matrix::Specification<
                T, TI, Rows, Cols>>,
            rlt::nn::parameters::groups::Normal, Kind>>;
    using Weights = rlt::nn::parameters::categories::Weights;
    using Biases = rlt::nn::parameters::categories::Biases;

    Parameter<D, F, Weights> encoderW{};
    Parameter<1, D, Biases> encoderB{};
    Parameter<D, ObservationLayout::Own, Weights> queryW{};
    Parameter<1, D, Biases> queryB{};
};

/**
 * @brief Evaluation buffer of the actor API; the graph keeps its own state, so it is empty.
 * @details Kept so that callers allocate and pass a buffer as for any rl_tools model.
 */
template<typename Specification, typename Specification::TI Batch,
         typename ContainerTag = rlt::MatrixDynamicTag>
struct GeneralizedAttentionActorBuffer final
{
};

/**
 * @brief P/N-capable shared actor with raw-set pooling and a compact v7 trunk.
 *
 * The inherited rl_tools MLP stores the trunk on the historical 156 compact inputs (the
 * graph uses the 124 that are ever non-zero) and the state-independent log-std. Raw track
 * capacities therefore do not enlarge the dense layers, and v7 checkpoints migrate.
 */
template<typename Specification>
struct GeneralizedAttentionActor
    : rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<
          Specification>
{
    using Base =
        rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<
            Specification>;
    using T = typename Specification::T;
    using TI = typename Specification::TI;
    static_assert(Specification::INPUT_DIM == ObservationLayout::Size,
                  "The generalized actor trunk must retain the v7 width");
    static_assert(std::is_same_v<T, float>, "The PPOKit graph computes in float");

    template<TI Batch, typename Tag = rlt::MatrixDynamicTag>
    using Buffer = GeneralizedAttentionActorBuffer<Specification, Batch, Tag>;

    GeneralizedTrackAttention<T, TI,
                              GeneralizedObservationLayout::RedFeatures,
                              GeneralizedObservationLayout::RedTracks,
                              GeneralizedActorGraph::RedWidth> red;
    GeneralizedTrackAttention<T, TI,
                              GeneralizedObservationLayout::FriendlyFeatures,
                              GeneralizedObservationLayout::BlueTracks
                                  + GeneralizedObservationLayout::ObjectiveTracks,
                              GeneralizedActorGraph::FriendlyWidth> friendlies;

    /**
     * @brief Computation of the actor, created by malloc.
     * @details Shared between copies of the storage: it only holds per-pass state, and the
     * weights are copied in before every pass.
     */
    std::shared_ptr<GeneralizedActorGraph> graph;
};

} // namespace ppo::examples::bluedefense::defenders

namespace rl_tools {
namespace blue_generalized_attention_detail {


template<typename Device, typename Block>
/**
 * @brief Réalise l'opération malloc; voir les préconditions et effets dans son corps C++.
 */
void malloc(Device& device, Block& block)
{
    rl_tools::malloc(device, block.encoderW);
    rl_tools::malloc(device, block.encoderB);
    rl_tools::malloc(device, block.queryW);
    rl_tools::malloc(device, block.queryB);
}

template<typename Device, typename Block>
void free(Device& device, Block& block)
{
    rl_tools::free(device, block.encoderW);
    rl_tools::free(device, block.encoderB);
    rl_tools::free(device, block.queryW);
    rl_tools::free(device, block.queryB);
}

template<typename Device, typename Block>
/**
 * @brief Réalise l'opération zero_gradient; voir les préconditions et effets dans son corps C++.
 */
void zero_gradient(Device& device, Block& block)
{
    rl_tools::set_all(device, block.encoderW.gradient, 0);
    rl_tools::set_all(device, block.encoderB.gradient, 0);
    rl_tools::set_all(device, block.queryW.gradient, 0);
    rl_tools::set_all(device, block.queryB.gradient, 0);
}

template<typename Device, typename Block, typename Optimizer>
void reset_optimizer(Device& device, Block& block, Optimizer& optimizer)
{
    rl_tools::_reset_optimizer_state(device, block.encoderW, optimizer);
    rl_tools::_reset_optimizer_state(device, block.encoderB, optimizer);
    rl_tools::_reset_optimizer_state(device, block.queryW, optimizer);
    rl_tools::_reset_optimizer_state(device, block.queryB, optimizer);
}

template<typename Device, typename Block, typename Optimizer>
/**
 * @brief Réalise l'opération update; voir les préconditions et effets dans son corps C++.
 */
void update(Device& device, Block& block, Optimizer& optimizer)
{
    rl_tools::update(device, block.encoderW, optimizer);
    rl_tools::update(device, block.encoderB, optimizer);
    rl_tools::update(device, block.queryW, optimizer);
    rl_tools::update(device, block.queryB, optimizer);
}

template<typename Device, typename Block, typename RandomEngine>
void initialize(Device&, Block& block, RandomEngine& randomEngine)
{
    auto const fill = [&](auto& matrix, bool const weight) {
        using Matrix = std::remove_reference_t<decltype(matrix)>;
        for (typename Device::index_t row = 0; row < Matrix::SPEC::ROWS; ++row)
        {
            for (typename Device::index_t column = 0;
                 column < Matrix::SPEC::COLS; ++column)
            {
                rl_tools::set(
                    matrix, row, column,
                    weight
                        ? rl_tools::random::uniform_real_distribution(
                              typename Device::SPEC::RANDOM{}, -0.05F, 0.05F,
                              randomEngine)
                        : 0.F);
            }
        }
    };
    fill(block.encoderW.parameters, true);
    fill(block.encoderB.parameters, false);
    fill(block.queryW.parameters, true);
    fill(block.queryB.parameters, false);
}

/** @brief Copies an rl_tools matrix into a graph tensor, reading `columns` of each row. */
template<typename Matrix>
void CopyIntoGraph(Matrix const& source, ppokit::graph::Parameter& target, std::vector<std::size_t> const& columns)
{
    std::size_t const inputs = columns.size();
    if (target.value.size() != static_cast<std::size_t>(Matrix::SPEC::ROWS) * inputs)
    {
        throw std::logic_error("Blue actor graph: a tensor no longer matches its rl_tools storage");
    }
    for (std::size_t row = 0; row * inputs < target.value.size(); ++row)
    {
        for (std::size_t column = 0; column < inputs; ++column)
        {
            target.value[row * inputs + column] = rl_tools::get(source,
                static_cast<typename Matrix::SPEC::TI>(row), static_cast<typename Matrix::SPEC::TI>(columns[column]));
        }
    }
}

/** @brief Adds a graph gradient to an rl_tools gradient matrix, writing `columns` of each row. */
template<typename Matrix>
void AddFromGraph(ppokit::graph::Parameter const& source, Matrix& target, std::vector<std::size_t> const& columns)
{
    std::size_t const inputs = columns.size();
    for (std::size_t row = 0; row * inputs < source.gradient.size(); ++row)
    {
        for (std::size_t column = 0; column < inputs; ++column)
        {
            rl_tools::increment(target, static_cast<typename Matrix::SPEC::TI>(row),
                static_cast<typename Matrix::SPEC::TI>(columns[column]), source.gradient[row * inputs + column]);
        }
    }
}

/**
 * @brief Visits every (rl_tools parameter, graph dense layer, input columns) pair.
 * @details The trunk input layer reads the compact columns of the features concat; every
 * other layer reads all its columns. The log-std is not part of the graph.
 */
template<typename Actor, typename Visitor>
void VisitLayers(Actor& actor, Visitor&& visit)
{
    using Storage = std::remove_const_t<Actor>;
    auto& graph = *actor.graph;
    auto& trunk = graph.Trunk();
    if (trunk.size() != Storage::NUM_HIDDEN_LAYERS + 1)
    {
        throw std::logic_error("Blue actor graph: the trunk depth no longer matches its rl_tools storage");
    }
    visit(actor.red.encoderW, actor.red.encoderB, graph.RedEncoder(), graph.Columns(graph.RedEncoder().Inputs()));
    visit(actor.red.queryW, actor.red.queryB, graph.RedQuery(), graph.Columns(graph.RedQuery().Inputs()));
    visit(actor.friendlies.encoderW, actor.friendlies.encoderB, graph.FriendlyEncoder(),
        graph.Columns(graph.FriendlyEncoder().Inputs()));
    visit(actor.friendlies.queryW, actor.friendlies.queryB, graph.FriendlyQuery(),
        graph.Columns(graph.FriendlyQuery().Inputs()));
    visit(actor.input_layer.weights, actor.input_layer.biases, trunk[0], graph.TrunkColumns());
    for (std::size_t layer = 0; layer < Storage::NUM_HIDDEN_LAYERS; ++layer)
    {
        visit(actor.hidden_layers[layer].weights, actor.hidden_layers[layer].biases, trunk[layer + 1],
            graph.Columns(trunk[layer + 1].Inputs()));
    }
    visit(actor.output_layer.weights, actor.output_layer.biases, graph.Head(), graph.Columns(graph.Head().Inputs()));
}

/** @brief Copies every weight of the storage into the graph. */
struct CopyWeightsIntoGraph final
{
    ppo::examples::bluedefense::defenders::GeneralizedActorGraph const& graph;

    template<typename Weights, typename Biases>
    void operator()(Weights const& weights, Biases const& biases, ppokit::graph::Dense& layer,
        std::vector<std::size_t> const& columns) const
    {
        CopyIntoGraph(weights.parameters, layer.Weights(), columns);
        CopyIntoGraph(biases.parameters, layer.Biases(), graph.Columns(layer.Outputs()));
    }
};

/** @brief Adds every gradient of the graph to the storage gradients. */
struct AddGradientsFromGraph final
{
    ppo::examples::bluedefense::defenders::GeneralizedActorGraph const& graph;

    template<typename Weights, typename Biases>
    void operator()(Weights& weights, Biases& biases, ppokit::graph::Dense& layer,
        std::vector<std::size_t> const& columns) const
    {
        AddFromGraph(layer.Weights(), weights.gradient, columns);
        AddFromGraph(layer.Biases(), biases.gradient, graph.Columns(layer.Outputs()));
    }
};

/** @brief Graph of an allocated actor, with the storage weights copied in. */
template<typename Actor>
ppo::examples::bluedefense::defenders::GeneralizedActorGraph& SynchronizedGraph(Actor const& actor)
{
    if (!actor.graph)
    {
        throw std::logic_error("Blue actor: malloc must be called before using the actor");
    }
    // The storage is the source of truth; the graph is only computation.
    VisitLayers(actor, CopyWeightsIntoGraph{*actor.graph});
    return *actor.graph;
}

/** @brief Copies one row of an rl_tools matrix into `row` (already of the right size). */
template<typename Matrix>
void CopyRow(Matrix const& matrix, typename Matrix::SPEC::TI index, std::vector<float>& row)
{
    for (typename Matrix::SPEC::TI column = 0; column < Matrix::SPEC::COLS; ++column)
    {
        row[column] = rl_tools::get(matrix, index, column);
    }
}

/**
 * @brief Runs the actor on every row: action means, and optionally the compact v7 vector
 *        (the trunk input, which the centralized critic also reads).
 */
template<typename Actor, typename Input, typename Output>
void RunRows(Actor const& actor, Input const& input, Output& output, float* compactRows, std::size_t compactStride)
{
    namespace blue = ppo::examples::bluedefense::defenders;
    static_assert(Input::SPEC::COLS == blue::GeneralizedObservationLayout::Size);
    static_assert(Output::SPEC::COLS == blue::GeneralizedObservationLayout::ActionSize);
    auto& graph = SynchronizedGraph(actor);
    std::vector<float>& row = graph.RowScratch();
    for (typename Input::SPEC::TI index = 0; index < Input::SPEC::ROWS; ++index)
    {
        CopyRow(input, index, row);
        float const* const means = graph.Forward(row.data());
        for (typename Output::SPEC::TI action = 0; action < Output::SPEC::COLS; ++action)
        {
            rl_tools::set(output, index, action, means[action]);
        }
        if (compactRows == nullptr)
        {
            continue;
        }
        float* const compact = compactRows + static_cast<std::size_t>(index) * compactStride;
        std::fill(compact, compact + compactStride, 0.F);
        float const* const features = graph.Features();
        std::vector<std::size_t> const& columns = graph.TrunkColumns();
        for (std::size_t column = 0; column < columns.size(); ++column)
        {
            compact[columns[column]] = features[column];
        }
    }
}

/**
 * @brief Action means and compact v7 rows of a batch of raw observations, in one pass.
 * @details The learner needs both: the means to act, the compact rows to build the
 * centralized critic input (mean and max over the team).
 */
template<typename Device, typename Actor, typename Input, typename Output, typename Compact>
void evaluate_with_compact(Device& device, Actor const& actor, Input const& input, Output& output, Compact& compact)
{
    namespace blue = ppo::examples::bluedefense::defenders;
    static_assert(Compact::SPEC::COLS == blue::ObservationLayout::Size);
    static_assert(Compact::SPEC::ROWS >= Input::SPEC::ROWS);
    std::vector<float>& rows =
        actor.graph->CompactScratch(static_cast<std::size_t>(Input::SPEC::ROWS) * blue::ObservationLayout::Size);
    RunRows(actor, input, output, rows.data(), blue::ObservationLayout::Size);
    for (typename Input::SPEC::TI index = 0; index < Input::SPEC::ROWS; ++index)
    {
        for (typename Compact::SPEC::TI column = 0; column < Compact::SPEC::COLS; ++column)
        {
            rl_tools::set(compact, index, column,
                rows[static_cast<std::size_t>(index) * blue::ObservationLayout::Size + column]);
        }
    }
    (void)device;
}

} // namespace blue_generalized_attention_detail

template<typename Device, typename Specification>
void malloc(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    rl_tools::malloc(device, static_cast<typename Actor::Base&>(actor));
    blue_generalized_attention_detail::malloc(device, actor.red);
    blue_generalized_attention_detail::malloc(device, actor.friendlies);
    actor.graph = std::make_shared<ppo::examples::bluedefense::defenders::GeneralizedActorGraph>();
}

template<typename Device, typename Specification>
void free(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    actor.graph.reset();
    blue_generalized_attention_detail::free(device, actor.red);
    blue_generalized_attention_detail::free(device, actor.friendlies);
    rl_tools::free(device, static_cast<typename Actor::Base&>(actor));
}

template<typename Device, typename Specification,
         typename Specification::TI Batch, typename ContainerTag>
void malloc(
    Device&,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActorBuffer<
        Specification, Batch, ContainerTag>&)
{
}

template<typename Device, typename Specification,
         typename Specification::TI Batch, typename ContainerTag>
void free(
    Device&,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActorBuffer<
        Specification, Batch, ContainerTag>&)
{
}

/** @brief Historical initialization: rl_tools MLP, then attention weights in [-0.05, 0.05]. */
template<typename Device, typename Specification, typename RandomEngine>
void init_weights(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor,
    RandomEngine& randomEngine)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    rl_tools::init_weights(device, static_cast<typename Actor::Base&>(actor),
                           randomEngine);
    blue_generalized_attention_detail::initialize(device, actor.red,
                                                   randomEngine);
    blue_generalized_attention_detail::initialize(device, actor.friendlies,
                                                   randomEngine);
}

template<typename Device, typename Specification>
void zero_gradient(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    rl_tools::zero_gradient(device, static_cast<typename Actor::Base&>(actor));
    blue_generalized_attention_detail::zero_gradient(device, actor.red);
    blue_generalized_attention_detail::zero_gradient(device,
                                                      actor.friendlies);
}

/** @brief Action means of a batch of raw observations (inference). */
template<typename Device, typename Specification, typename InputSpecification,
         typename OutputSpecification, typename Buffer>
void evaluate(
    Device&,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification> const& actor,
    Matrix<InputSpecification> const& input,
    Matrix<OutputSpecification>& output, Buffer&)
{
    blue_generalized_attention_detail::RunRows(actor, input, output, nullptr, 0);
}

/** @brief Action means of a training batch; backward recomputes each row before differentiating it. */
template<typename Device, typename Specification, typename InputSpecification,
         typename OutputSpecification>
void forward(
    Device&,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor,
    Matrix<InputSpecification> const& input,
    Matrix<OutputSpecification>& output)
{
    static_assert(Specification::BATCH_SIZE >= InputSpecification::ROWS);
    blue_generalized_attention_detail::RunRows(actor, input, output, nullptr, 0);
}

/**
 * @brief Adds d loss / d parameters of the batch to the storage gradients.
 * @param outputDerivative d loss / d mean of each row.
 * @details Each row is run forward again, then backward: the graph keeps the state of one
 * observation. Gradients are summed over the rows, as rl_tools does.
 */
template<typename Device, typename Specification, typename InputSpecification,
         typename OutputSpecification, typename Buffer>
void backward(
    Device&,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor,
    Matrix<InputSpecification> const& input,
    Matrix<OutputSpecification>& outputDerivative, Buffer&)
{
    namespace detail = blue_generalized_attention_detail;
    auto& graph = detail::SynchronizedGraph(actor);
    graph.ZeroGradients();
    std::vector<float>& row = graph.RowScratch();
    std::array<float, OutputSpecification::COLS> derivative{};
    for (typename InputSpecification::TI index = 0; index < InputSpecification::ROWS; ++index)
    {
        detail::CopyRow(input, index, row);
        graph.Forward(row.data());
        graph.ObserveRow(static_cast<std::size_t>(index));
        for (typename OutputSpecification::TI action = 0; action < OutputSpecification::COLS; ++action)
        {
            derivative[action] = rl_tools::get(outputDerivative, index, action);
        }
        graph.Backward(derivative.data());
    }
    graph.ObserveGradients();
    detail::VisitLayers(actor, detail::AddGradientsFromGraph{graph});
}

} // namespace rl_tools

namespace ppo::examples::bluedefense::defenders {

template<typename Device, typename Specification, typename Optimizer>
/**
 * @brief Réalise l'opération update; voir les préconditions et effets dans son corps C++.
 */
void update(Device& device, GeneralizedAttentionActor<Specification>& actor,
            Optimizer& optimizer)
{
    rl_tools::update(device,
                     static_cast<typename GeneralizedAttentionActor<
                         Specification>::Base&>(actor),
                     optimizer);
    rl_tools::blue_generalized_attention_detail::update(device, actor.red,
                                                        optimizer);
    rl_tools::blue_generalized_attention_detail::update(device,
                                                        actor.friendlies,
                                                        optimizer);
}

template<typename Device, typename Specification, typename Optimizer>
void _reset_optimizer_state(
    Device& device, GeneralizedAttentionActor<Specification>& actor,
    Optimizer& optimizer)
{
    rl_tools::_reset_optimizer_state(
        device,
        static_cast<typename GeneralizedAttentionActor<Specification>::Base&>(
            actor),
        optimizer);
    rl_tools::blue_generalized_attention_detail::reset_optimizer(
        device, actor.red, optimizer);
    rl_tools::blue_generalized_attention_detail::reset_optimizer(
        device, actor.friendlies, optimizer);
}

} // namespace ppo::examples::bluedefense::defenders
