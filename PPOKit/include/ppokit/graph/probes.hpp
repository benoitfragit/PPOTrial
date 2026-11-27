/**
 * @file PPOKit/include/ppokit/graph/probes.hpp
 * @brief Probes on the blocks of a graph: what each block outputs, and how much gradient
 *        reaches it, during an update.
 * @details For every block (inputs excluded) the probes keep the mean, spread, minimum
 * and maximum of its outputs, the share of outputs exactly zero (dead ReLU units), the
 * share at |x| >= 0.99 (saturated tanh or sigmoid units) and the mean size (L2 norm) of
 * the gradient of its parameters per optimizer step. Set outputs count present entities
 * only.
 *
 * Zero cost when off: a learner without probes attached only tests a null pointer; with
 * probes, the accumulators are allocated once, at construction, and nothing is allocated
 * while observing.
 *
 * @code
 * ppokit::graph::GraphProbes actorProbes(learner.Actor());
 * ppokit::graph::GraphProbes criticProbes(learner.Critic());
 * learner.SetProbes(&actorProbes, &criticProbes);   // this update is probed
 * learner.Update(buffer);
 * learner.SetProbes(nullptr, nullptr);
 * journal.WriteProbes(update, "actor", actorProbes.Summary());
 * @endcode
 */
#pragma once

#include "ppokit/graph/graph.hpp"
#include "ppokit/runtime/probe_moments.hpp"
#include "ppokit/runtime/probe_record.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace ppokit::graph {

class GraphProbes final
{
public:
    /** @brief Threshold of a saturated output, for tanh and sigmoid blocks. */
    static constexpr double SaturationLevel = runtime::ProbeSaturationLevel;

    /** @brief Probes every block of `graph` that is not an input. */
    explicit GraphProbes(Graph& graph) : graph_(graph)
    {
        for (NodeId node = 0; node < graph.NodeCount(); ++node)
        {
            if (std::string(graph.At(node).Type()) == "input")
            {
                continue;
            }
            Block block;
            block.node = node;
            block.shape = graph.At(node).OutputShape();
            graph.At(node).CollectParameters(block.parameters);
            blocks_.push_back(std::move(block));
        }
        Reset();
    }

    /** @brief Forgets what was observed, before a new update. */
    void Reset() noexcept
    {
        for (Block& block : blocks_)
        {
            block.values = runtime::ProbeMoments{};
            block.gradientNormSum = 0.0;
            block.steps = 0;
        }
    }

    /** @brief Reads every block output of the last Forward of the graph. */
    void ObserveForward() noexcept
    {
        for (Block& block : blocks_)
        {
            Signal const output = graph_.OutputOf(block.node);
            if (block.shape.kind == SignalKind::Vector)
            {
                block.values.Add(output.values, block.shape.width);
                continue;
            }
            // A set: only the rows of present entities carry meaning.
            for (std::size_t entity = 0; entity < block.shape.capacity; ++entity)
            {
                if (output.mask != nullptr && output.mask[entity] != 0)
                {
                    block.values.Add(output.values + entity * block.shape.width, block.shape.width);
                }
            }
        }
    }

    /** @brief Reads the parameter gradients of every block, once per optimizer step (before it). */
    void ObserveGradients() noexcept
    {
        for (Block& block : blocks_)
        {
            if (block.parameters.empty())
            {
                continue;
            }
            double squares = 0.0;
            for (Parameter const* parameter : block.parameters)
            {
                for (float const gradient : parameter->gradient)
                {
                    squares += static_cast<double>(gradient) * static_cast<double>(gradient);
                }
            }
            block.gradientNormSum += std::sqrt(squares);
            ++block.steps;
        }
    }

    /** @brief What was observed since Reset, one entry per block in graph order. */
    std::vector<runtime::BlockProbe> Summary() const
    {
        std::vector<runtime::BlockProbe> summary;
        summary.reserve(blocks_.size());
        for (Block const& block : blocks_)
        {
            runtime::BlockProbe probe;
            probe.block = graph_.NameOf(block.node);
            probe.type = graph_.At(block.node).Type();
            block.values.Fill(probe);
            probe.hasGradient = !block.parameters.empty();
            if (block.steps > 0)
            {
                probe.gradientNorm = block.gradientNormSum / static_cast<double>(block.steps);
            }
            summary.push_back(std::move(probe));
        }
        return summary;
    }

private:
    struct Block final
    {
        NodeId node = 0;
        Shape shape;
        std::vector<Parameter*> parameters;
        runtime::ProbeMoments values;
        double gradientNormSum = 0.0;
        std::size_t steps = 0;
    };

    Graph& graph_;
    std::vector<Block> blocks_;
};

} // namespace ppokit::graph
