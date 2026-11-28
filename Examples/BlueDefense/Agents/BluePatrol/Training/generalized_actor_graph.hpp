/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_actor_graph.hpp
 * @brief Computation of the Blue actor: the PPOKit block graph of graph_blocks.json.
 * @details The actor's parameters stay in their rl_tools storage (GeneralizedAttentionActor:
 * Adam moments, checkpoints, migrations). This class runs the network itself: it builds
 * the graph from task.json and graph_blocks.json, embedded at compile time so that the
 * actor never depends on the working directory, and checks that the graph still has the
 * structure the storage and the checkpoints rely on.
 *
 * The trunk reads the legacy v7 "compact" layout. The graph's "features" concat lists the
 * same values in another order; TrunkColumns() maps each concat column to its compact
 * column (role[0] in patrol[8], role[1] in global[1], team counts in the spare red slots).
 */
#pragma once

#include "generalized_observation.hpp"

#include "ppokit/graph/graph.hpp"
#include "ppokit/graph/policy_layout.hpp"
#include "ppokit/graph/probes.hpp"
#include "ppokit/schema/block_graph_schema.hpp"

#include <blue_actor_graph_files.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppo::examples::bluedefense::defenders {

/** @brief Compact columns introduced by the generalized raw-to-MLP adapter. */
struct GeneralizedCompactFeatures final
{
    static constexpr std::size_t FormationSlot =
        ObservationLayout::GlobalOffset + 1U;
    static constexpr std::size_t ActiveBlueCount =
        ObservationLayout::RedOffset + 3U * 8U;
    static constexpr std::size_t ActiveRedCount =
        ActiveBlueCount + 1U;
};

/**
 * @brief The Blue actor graph, ready to run one observation at a time.
 * @details Not thread-safe: forward and backward state lives in the graph.
 */
class GeneralizedActorGraph final
{
public:
    /** @brief Red and friendly embedding widths fixed by the checkpoint format. */
    static constexpr std::size_t RedWidth = 8;
    static constexpr std::size_t FriendlyWidth = 6;

    /** @throws std::runtime_error when graph_blocks.json no longer fits the actor storage. */
    GeneralizedActorGraph()
    {
        try
        {
            auto const task = ppokit::schema::ParseTaskSchema(embedded::BlueTaskJson, "task.json");
            auto const graphs = ppokit::schema::ParseBlockGraph(embedded::BlueGraphBlocksJson, "graph_blocks.json", task);
            RequireCompactStructure(graphs.actor);
            graph_ = ppokit::schema::BuildGraph(graphs.actor);
            binder_ = std::make_unique<ppokit::graph::InputBinder>(
                ppokit::schema::MakeInputSlices(graphs.actor, "actor", task), GeneralizedObservationLayout::Size);
        }
        catch (ppokit::schema::SchemaError const& error)
        {
            throw std::runtime_error(std::string("Blue actor graph: ") + error.what());
        }
        features_ = graph_.Find("features");
        RequireShape("own", ppokit::graph::Shape::Vector(ObservationLayout::Own));
        RequireShape("red_embed", ppokit::graph::Shape::Set(GeneralizedObservationLayout::RedTracks, RedWidth));
        RequireShape("friends_embed",
            ppokit::graph::Shape::Set(
                GeneralizedObservationLayout::BlueTracks + GeneralizedObservationLayout::ObjectiveTracks,
                FriendlyWidth));
        trunkColumns_ = MakeTrunkColumns();
        if (trunkColumns_.size() != graph_.At(features_).OutputShape().width)
        {
            throw std::runtime_error("Blue actor graph: the features concat no longer has the compact v7 width");
        }
        PrepareIdentityColumns();
        row_.resize(GeneralizedObservationLayout::Size);
    }

    GeneralizedActorGraph(GeneralizedActorGraph const&) = delete;
    GeneralizedActorGraph& operator=(GeneralizedActorGraph const&) = delete;

    /** @brief Forward pass of one raw observation; returns the action means. */
    float const* Forward(float const* observation)
    {
        return graph_.Forward(binder_->Bind(observation)).values;
    }

    /** @brief Backward pass of the last Forward; accumulates parameter gradients. */
    void Backward(float const* meanGradient) { graph_.Backward(meanGradient); }

    /** @brief Trunk input of the last Forward, in concat order (see TrunkColumns). */
    float const* Features() const { return graph_.OutputOf(features_).values; }

    /** @brief Compact v7 column of each column of the features concat. */
    std::vector<std::size_t> const& TrunkColumns() const noexcept { return trunkColumns_; }

    ppokit::graph::Dense& RedEncoder() { return Node<ppokit::graph::EntityEncoderNode>("red_embed").Encoder().Layer(); }
    ppokit::graph::Dense& RedQuery() { return Node<ppokit::graph::DenseNode>("red_query").Layer(); }
    ppokit::graph::Dense& FriendlyEncoder()
    {
        return Node<ppokit::graph::EntityEncoderNode>("friends_embed").Encoder().Layer();
    }
    ppokit::graph::Dense& FriendlyQuery() { return Node<ppokit::graph::DenseNode>("friends_query").Layer(); }
    std::vector<ppokit::graph::Dense>& Trunk() { return Node<ppokit::graph::MlpNode>("trunk").Layers(); }
    ppokit::graph::Dense& Head() { return Node<ppokit::graph::HeadNode>("policy").Layer(); }
    /** @brief The Gaussian head itself (its log-std tensor is stored by rl_tools, not trained here). */
    ppokit::graph::HeadNode& PolicyHead() { return Node<ppokit::graph::HeadNode>("policy"); }

    void ZeroGradients() { graph_.ZeroGradients(); }

    /** @brief The block graph itself, for probes. */
    ppokit::graph::Graph& Network() noexcept { return graph_; }

    /**
     * @brief Probes fed by the next backward passes, with the first `rows` rows of the batch
     *        being real training rows (the others are padding). Null: off. Read-only.
     */
    void SetProbes(ppokit::graph::GraphProbes* probes, std::size_t rows) noexcept
    {
        probes_ = probes;
        probedRows_ = rows;
    }

    /** @brief Hands the last Forward to the probes when it ran training row `row`. */
    void ObserveRow(std::size_t row) noexcept
    {
        if (probes_ != nullptr && row < probedRows_)
        {
            probes_->ObserveForward();
        }
    }

    /** @brief Hands the batch gradients to the probes, before they reach the optimizer. */
    void ObserveGradients() noexcept
    {
        if (probes_ != nullptr)
        {
            probes_->ObserveGradients();
        }
    }

    /**
     * @brief Columns 0 .. size - 1, to read a whole matrix row.
     * @throws std::logic_error for a size that no layer of the graph has.
     */
    std::vector<std::size_t> const& Columns(std::size_t size) const
    {
        auto const columns = identityColumns_.find(size);
        if (columns == identityColumns_.end())
        {
            throw std::logic_error("Blue actor graph: no layer of width " + std::to_string(size));
        }
        return columns->second;
    }

    /** @brief One raw observation, reused by every pass (no allocation per step). */
    std::vector<float>& RowScratch() noexcept { return row_; }

    /** @brief Compact rows of a batch; grows on the first, largest batch only. */
    std::vector<float>& CompactScratch(std::size_t size)
    {
        if (compact_.size() < size)
        {
            compact_.resize(size);
        }
        return compact_;
    }

private:
    template<typename NodeType>
    NodeType& Node(char const* name)
    {
        auto* const node = dynamic_cast<NodeType*>(&graph_.At(graph_.Find(name)));
        if (node == nullptr)
        {
            throw std::runtime_error(std::string("Blue actor graph: node '") + name + "' has another block type");
        }
        return *node;
    }

    void RequireShape(char const* name, ppokit::graph::Shape const& shape)
    {
        if (graph_.At(graph_.Find(name)).OutputShape() != shape)
        {
            throw std::runtime_error(std::string("Blue actor graph: node '") + name + "' must give "
                + ppokit::graph::Describe(shape) + " (checkpoint format)");
        }
    }

    static ppokit::schema::BlockSpec const* FindBlock(ppokit::schema::BlockNetworkSpec const& network, char const* name)
    {
        for (auto const& block : network.blocks)
        {
            if (block.name == name)
            {
                return &block;
            }
        }
        return nullptr;
    }

    /** @brief The concat order and the kept features that TrunkColumns assumes. */
    static void RequireCompactStructure(ppokit::schema::BlockNetworkSpec const& network)
    {
        std::vector<std::string> const expected{"own", "lidar", "red_attend", "red_mean", "red_max", "team_counts",
            "friends_attend", "friends_mean", "friends_max", "patrol_kept", "role", "global_kept"};
        auto const* const features = FindBlock(network, "features");
        auto const* const patrol = FindBlock(network, "patrol_kept");
        auto const* const global = FindBlock(network, "global_kept");
        bool const matches = features != nullptr && features->sources == expected && patrol != nullptr
            && patrol->indices == std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 9} && global != nullptr
            && global->indices == std::vector<std::size_t>{0};
        if (!matches)
        {
            throw std::runtime_error("Blue actor graph: the 'features' concat must keep the compact v7 order "
                                     "(checkpoints store the trunk in that layout)");
        }
    }

    /** @brief Mirrors the historical compact transform of the hand-written actor. */
    static std::vector<std::size_t> MakeTrunkColumns()
    {
        std::vector<std::size_t> columns;
        auto const append = [&columns](std::size_t first, std::size_t count) {
            for (std::size_t column = 0; column < count; ++column)
            {
                columns.push_back(first + column);
            }
        };
        append(0, ObservationLayout::Own);
        append(ObservationLayout::Own, ObservationLayout::Lidar);
        append(ObservationLayout::RedOffset, 3 * RedWidth);
        columns.push_back(GeneralizedCompactFeatures::ActiveBlueCount);
        columns.push_back(GeneralizedCompactFeatures::ActiveRedCount);
        append(ObservationLayout::FriendOffset, 3 * FriendlyWidth);
        append(ObservationLayout::PatrolOffset, 8);
        columns.push_back(ObservationLayout::PatrolOffset + 9);
        columns.push_back(ObservationLayout::PatrolOffset + 8);
        columns.push_back(GeneralizedCompactFeatures::FormationSlot);
        columns.push_back(ObservationLayout::GlobalOffset);
        return columns;
    }

    /** @brief Identity column lists for every input and output width of the dense layers. */
    void PrepareIdentityColumns()
    {
        std::vector<ppokit::graph::Dense*> layers{&RedEncoder(), &RedQuery(), &FriendlyEncoder(), &FriendlyQuery(),
            &Head()};
        for (ppokit::graph::Dense& layer : Trunk())
        {
            layers.push_back(&layer);
        }
        for (ppokit::graph::Dense const* layer : layers)
        {
            for (std::size_t const size : {layer->Inputs(), layer->Outputs()})
            {
                std::vector<std::size_t>& columns = identityColumns_[size];
                columns.resize(size);
                for (std::size_t column = 0; column < size; ++column)
                {
                    columns[column] = column;
                }
            }
        }
    }

    ppokit::graph::Graph graph_;
    std::unique_ptr<ppokit::graph::InputBinder> binder_;
    ppokit::graph::NodeId features_ = 0;
    std::vector<std::size_t> trunkColumns_;
    std::map<std::size_t, std::vector<std::size_t>> identityColumns_;
    std::vector<float> row_;
    std::vector<float> compact_;
    ppokit::graph::GraphProbes* probes_ = nullptr;
    std::size_t probedRows_ = 0;
};

} // namespace ppo::examples::bluedefense::defenders
