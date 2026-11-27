/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/shared_backend_conversion.hpp
 * @brief Writes the training state of the Blue MAPPO learner as a training artifact of the
 *        shared graph backend (ppokit_graph_network, TeamGraphLearner).
 * @details Nothing is re-learned or approximated: retained tensors and optimizer values are copied.
 * - Actor: the parameters of its block graph, taken from their rl_tools storage through the
 *   same column maps as the learner uses (VisitLayers), the learned log-std, and the Adam
 *   moments of each, element for element.
 * - Critic: the graph can retain all 156 columns of each historical group, including the
 *   zero-valued central features and their optimizer state, or use the compact 124 columns.
 * - Adam: step count = rl_tools age - 1, moments copied; the generator's state; the update
 *   count and the seed.
 *
 * The target networks are those of `task` and `graphs` (a team task whose network.json has
 * the Blue actor and a compatible critic MLP of two hidden layers); a mismatch throws.
 */
#pragma once

#include "generalized_patrol_learner.hpp"

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/graph/adam.hpp"
#include "ppokit/graph/checkpoint_io.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppo::examples::bluedefense::defenders {

namespace shared_backend_detail {

namespace attention = rl_tools::blue_generalized_attention_detail;

/** @brief Values, first and second moments of every tensor of one network, in Graph order. */
struct ConvertedNetwork final
{
    ppokit::graph::checkpoint::Tensors values;
    ppokit::graph::checkpoint::Tensors first;
    ppokit::graph::checkpoint::Tensors second;
};

/** @brief Index of each parameter of `network` in Graph::Parameters() order. */
inline std::map<ppokit::graph::Parameter const*, std::size_t> ParameterIndex(ppokit::graph::Graph& network)
{
    std::map<ppokit::graph::Parameter const*, std::size_t> index;
    std::vector<ppokit::graph::Parameter*> const parameters = network.Parameters();
    for (std::size_t position = 0; position < parameters.size(); ++position)
    {
        index[parameters[position]] = position;
    }
    return index;
}

/** @brief An rl_tools matrix read through `columns` of each row, in graph tensor order. */
template<typename Matrix>
std::vector<float> Read(Matrix const& matrix, std::size_t size, std::vector<std::size_t> const& columns)
{
    ppokit::graph::Parameter staging(size);
    attention::CopyIntoGraph(matrix, staging, columns);
    return staging.value;
}

/** @brief Fills the slot of `layer`'s weights and biases from one rl_tools parameter pair. */
struct CopyActorLayer final
{
    std::map<ppokit::graph::Parameter const*, std::size_t> const& index;
    GeneralizedActorGraph const& graph;
    ConvertedNetwork& out;
    std::vector<bool>& filled;

    template<typename Weights, typename Biases>
    void operator()(Weights const& weights, Biases const& biases, ppokit::graph::Dense& layer,
        std::vector<std::size_t> const& columns)
    {
        Copy(weights, layer.Weights(), columns);
        Copy(biases, layer.Biases(), graph.Columns(layer.Outputs()));
    }

    template<typename Tensor>
    void Copy(Tensor const& tensor, ppokit::graph::Parameter const& parameter, std::vector<std::size_t> const& columns)
    {
        auto const found = index.find(&parameter);
        if (found == index.end() || filled[found->second])
        {
            throw std::logic_error("Blue conversion: a graph tensor is unknown or mapped twice");
        }
        std::size_t const slot = found->second;
        std::size_t const size = parameter.value.size();
        out.values[slot] = Read(tensor.parameters, size, columns);
        out.first[slot] = Read(tensor.gradient_first_order_moment, size, columns);
        out.second[slot] = Read(tensor.gradient_second_order_moment, size, columns);
        filled[slot] = true;
    }
};

/** @brief Every actor tensor: the attention graph's layers and the learned log-std. */
template<typename Actor>
ConvertedNetwork ConvertActor(Actor const& actor)
{
    GeneralizedActorGraph& graph = attention::SynchronizedGraph(actor);
    ppokit::graph::Graph& network = graph.Network();
    auto const index = ParameterIndex(network);
    std::size_t const count = index.size();
    ConvertedNetwork out{ppokit::graph::checkpoint::Tensors(count), ppokit::graph::checkpoint::Tensors(count),
        ppokit::graph::checkpoint::Tensors(count)};
    std::vector<bool> filled(count, false);
    attention::VisitLayers(actor, CopyActorLayer{index, graph, out, filled});
    ppokit::graph::Parameter const& logStd = graph.PolicyHead().LogStd();
    CopyActorLayer{index, graph, out, filled}.Copy(actor.log_std, logStd, graph.Columns(logStd.value.size()));
    for (bool const done : filled)
    {
        if (!done)
        {
            throw std::logic_error("Blue conversion: an actor graph tensor has no rl_tools storage");
        }
    }
    return out;
}

/**
 * @brief rl_tools critic column of each column of the shared critic's first layer: own,
 *        mean and max of the requested historical group width, then the two fractions.
 */
inline std::vector<std::size_t> CriticInputColumns(std::size_t features)
{
    std::vector<std::size_t> columns;
    for (std::size_t const offset :
        {GeneralizedCriticLayout::OwnOffset, GeneralizedCriticLayout::MeanOffset, GeneralizedCriticLayout::MaxOffset})
    {
        for (std::size_t feature = 0; feature < features; ++feature)
        {
            columns.push_back(offset + feature);
        }
    }
    columns.push_back(GeneralizedCriticLayout::LivingFractionOffset);
    columns.push_back(GeneralizedCriticLayout::TeamFractionOffset);
    return columns;
}

/** @brief Identity columns 0 .. size - 1. */
inline std::vector<std::size_t> AllColumns(std::size_t size)
{
    std::vector<std::size_t> columns(size);
    for (std::size_t column = 0; column < size; ++column)
    {
        columns[column] = column;
    }
    return columns;
}

/** @brief Appends one rl_tools tensor (rows x columns.size() values) with both moments. */
template<typename Tensor>
void AppendTensor(Tensor const& tensor, std::size_t rows, std::vector<std::size_t> const& columns,
    ConvertedNetwork& out)
{
    std::size_t const size = rows * columns.size();
    out.values.push_back(Read(tensor.parameters, size, columns));
    out.first.push_back(Read(tensor.gradient_first_order_moment, size, columns));
    out.second.push_back(Read(tensor.gradient_second_order_moment, size, columns));
}

/** @brief Appends one rl_tools dense layer, weights read through `columns`, then biases. */
template<typename Layer>
void AppendCriticLayer(Layer const& layer, std::vector<std::size_t> const& columns, std::size_t outputs,
    ConvertedNetwork& out)
{
    AppendTensor(layer.weights, outputs, columns, out);
    AppendTensor(layer.biases, 1, AllColumns(outputs), out);
}

/** @brief The critic MLP as the graph MLP: first layer through CriticInputColumns. */
template<typename Critic>
ConvertedNetwork ConvertCritic(Critic const& critic, std::size_t features)
{
    static_assert(Critic::NUM_HIDDEN_LAYERS == 1, "the Blue critic has two hidden layers");
    constexpr std::size_t Hidden = static_cast<std::size_t>(Critic::SPEC::HIDDEN_DIM);
    ConvertedNetwork out;
    AppendCriticLayer(critic.input_layer, CriticInputColumns(features), Hidden, out);
    AppendCriticLayer(critic.hidden_layers[0], AllColumns(Hidden), Hidden, out);
    AppendCriticLayer(critic.output_layer, AllColumns(Hidden), 1, out);
    return out;
}

/** @brief Checks that converted tensors have the sizes of the target network's. */
inline void RequireSizes(ConvertedNetwork const& converted, std::vector<ppokit::graph::Parameter*> const& target,
    char const* network)
{
    bool same = converted.values.size() == target.size();
    for (std::size_t tensor = 0; same && tensor < target.size(); ++tensor)
    {
        same = converted.values[tensor].size() == target[tensor]->value.size();
    }
    if (!same)
    {
        throw std::invalid_argument(std::string("Blue conversion: the ") + network
            + " of network.json does not have the Blue learner's tensors");
    }
}

/** @brief Writes one section of tensors in the graph checkpoint format. */
inline void WriteValues(std::ostream& out, char const* label, ppokit::graph::checkpoint::Tensors const& tensors)
{
    std::vector<ppokit::graph::Parameter> staging;
    staging.reserve(tensors.size());
    std::vector<ppokit::graph::Parameter*> pointers;
    for (auto const& values : tensors)
    {
        staging.emplace_back(values.size());
        staging.back().value = values;
        pointers.push_back(&staging.back());
    }
    ppokit::graph::checkpoint::WriteTensors(out, label, pointers);
}

/** @brief Header of a shared-backend artifact of these networks. */
inline ppokit::artifacts::ArtifactHeader HeaderOf(ppokit::artifacts::ArtifactKind kind,
    ppokit::schema::TaskSchema const& task, ppokit::schema::BlockGraphSchema const& graphs, std::uint64_t updates,
    std::uint32_t seed)
{
    ppokit::artifacts::ArtifactHeader header;
    header.kind = kind;
    header.identity = ppokit::schema::ArtifactIdentityOf(task, graphs);
    header.updates = updates;
    header.seed = seed;
    header.created = ppokit::artifacts::UtcNow();
    return header;
}

/** @brief The team critic section that the conversion needs, or a clear refusal. */
inline ppokit::schema::TeamCriticSpec const& RequireTeamCritic(ppokit::schema::BlockGraphSchema const& graphs)
{
    if (!graphs.teamCritic)
    {
        throw std::invalid_argument("Blue conversion: network.json has no \"team_critic\" section");
    }
    return *graphs.teamCritic;
}

} // namespace shared_backend_detail

/**
 * @brief Writes `learner`'s training state as a training artifact of the shared backend.
 * @param task, graphs The team task and network.json of the shared-backend project.
 * @throws std::logic_error outside an update boundary; std::invalid_argument when the
 *         networks of `graphs` do not have the learner's tensors.
 */
inline void WriteSharedBackendTraining(GeneralizedPatrolLearner const& learner, ppokit::schema::TaskSchema const& task,
    ppokit::schema::BlockGraphSchema const& graphs, std::filesystem::path const& path)
{
    namespace detail = shared_backend_detail;
    auto const& teamCritic = detail::RequireTeamCritic(graphs);
    constexpr std::size_t activeFeatures = ObservationLayout::Size - ObservationLayout::CentralSize;
    if (!task.critic || teamCritic.width != activeFeatures)
        throw std::invalid_argument("Blue conversion: team critic features do not match the Blue actor");
    auto const& groups = task.critic->groups;
    std::size_t const features = groups.at(teamCritic.own).Size();
    if ((features != activeFeatures && features != GeneralizedCriticLayout::Compact)
        || groups.at(teamCritic.mean).Size() != features || groups.at(teamCritic.max).Size() != features)
        throw std::invalid_argument("Blue conversion: critic groups must have the same supported width");
    ppokit::graph::Graph actorTarget = ppokit::schema::BuildGraph(graphs.actor);
    ppokit::graph::Graph criticTarget = ppokit::schema::BuildGraph(graphs.critic);
    learner.WithTrainingState([&](auto const& actor, auto const& critic, auto const& actorOptimizer,
                                  auto const& criticOptimizer, auto const& generator, std::uint64_t updates,
                                  std::uint32_t seed) {
        detail::ConvertedNetwork const actorTensors = detail::ConvertActor(actor);
        detail::ConvertedNetwork const criticTensors = detail::ConvertCritic(critic, features);
        detail::RequireSizes(actorTensors, actorTarget.Parameters(), "actor");
        detail::RequireSizes(criticTensors, criticTarget.Parameters(), "critic");
        ppokit::artifacts::ArtifactHeader const header =
            detail::HeaderOf(ppokit::artifacts::ArtifactKind::Training, task, graphs, updates, seed);
        ppokit::graph::Adam::State const actorState{
            static_cast<std::size_t>(actorOptimizer.age - 1), actorTensors.first, actorTensors.second};
        ppokit::graph::Adam::State const criticState{
            static_cast<std::size_t>(criticOptimizer.age - 1), criticTensors.first, criticTensors.second};
        ppokit::artifacts::WriteFileAtomically(
            path,
            [&](std::ostream& out) {
                out << ppokit::artifacts::HeaderLine(header);
                detail::WriteValues(out, "actor", actorTensors.values);
                detail::WriteValues(out, "critic", criticTensors.values);
                ppokit::graph::checkpoint::WriteAdam(out, "actor_adam", actorState);
                ppokit::graph::checkpoint::WriteAdam(out, "critic_adam", criticState);
                ppokit::graph::checkpoint::WriteRandomState(out, generator);
            },
            ppokit::artifacts::AtomicWriteOptions{}, "Blue shared-backend training artifact");
    });
}

/**
 * @brief Writes `learner`'s actor as a policy artifact of the shared backend.
 * @throws std::invalid_argument when the actor of `graphs` does not have the learner's tensors.
 */
inline void WriteSharedBackendPolicy(GeneralizedPatrolLearner const& learner, ppokit::schema::TaskSchema const& task,
    ppokit::schema::BlockGraphSchema const& graphs, std::filesystem::path const& path)
{
    namespace detail = shared_backend_detail;
    (void)detail::RequireTeamCritic(graphs);
    ppokit::graph::Graph actorTarget = ppokit::schema::BuildGraph(graphs.actor);
    learner.WithTrainingState([&](auto const& actor, auto const&, auto const&, auto const&, auto const&,
                                  std::uint64_t updates, std::uint32_t seed) {
        detail::ConvertedNetwork const actorTensors = detail::ConvertActor(actor);
        detail::RequireSizes(actorTensors, actorTarget.Parameters(), "actor");
        ppokit::artifacts::ArtifactHeader const header =
            detail::HeaderOf(ppokit::artifacts::ArtifactKind::Policy, task, graphs, updates, seed);
        ppokit::artifacts::WriteFileAtomically(
            path,
            [&](std::ostream& out) {
                out << ppokit::artifacts::HeaderLine(header);
                detail::WriteValues(out, "actor", actorTensors.values);
            },
            ppokit::artifacts::AtomicWriteOptions{}, "Blue shared-backend policy artifact");
    });
}

} // namespace ppo::examples::bluedefense::defenders
