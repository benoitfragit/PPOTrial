/**
 * @file PPOKit/include/ppokit/runtime/observers.hpp
 * @brief The observers of a graph (Visualiseur blocks), called after each environment step.
 * @details An observer only watches: it receives, for every agent of the step, the signals
 * graph.json wires to it (observation and world groups, actions, reward channels) and
 * returns nothing. What it does with them (send them in UDP to a viewer, log them...) is
 * hidden in its DLL. The runner calls Notify after each environment step of collection,
 * evaluation or inference, never during the PPO optimization passes.
 *
 * Every buffer is allocated at construction for the maximum number of agents: Notify
 * copies and calls, it allocates nothing. An observer that fails is switched off and keeps
 * its error; it never stops the run.
 */
#pragma once

#include "ppokit/component/library.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ppokit::runtime {

/**
 * @brief One environment step of a team, as the observers see it.
 * @details Each signal holds one row per agent, rows of the task's sizes: observation
 * (observation.size), world (world size), actions (actionSize), rewards (one
 * value per channel). `world` may be null when the task has no world section.
 */
struct ObservedStep final
{
    std::size_t agents = 0;
    float const* observation = nullptr;
    float const* world = nullptr;
    float const* actions = nullptr;
    float const* rewards = nullptr;
};

/** @brief The observers of a graph, ready to be called each step. */
class ObserverSet final
{
public:
    /**
     * @param task Task the signals follow.
     * @param observers Validated observers of graph.json (BlockGraphSchema::observers).
     * @param maximumAgents Largest number of agents of a step.
     * @param seed Seed of the run; each observer gets one derived from it and its name.
     * @throws std::runtime_error when an observer cannot be created.
     */
    ObserverSet(schema::TaskSchema const& task, std::vector<schema::ObserverSpec> const& observers,
        std::size_t maximumAgents, std::uint64_t seed);
    ~ObserverSet();
    ObserverSet(ObserverSet const&) = delete;
    ObserverSet& operator=(ObserverSet const&) = delete;

    /**
     * @brief Hands one step to every active observer. Never throws, never allocates on success.
     * @param step Signals of the step; more agents than maximumAgents are ignored.
     */
    void Notify(ObservedStep const& step) noexcept;

    std::size_t Size() const noexcept { return observers_.size(); }
    /** @brief Name of observer `index`, as in graph.json. */
    std::string const& Name(std::size_t index) const noexcept;
    /** @brief False once the observer has failed; it is not called again. */
    bool Active(std::size_t index) const noexcept;
    /** @brief Why the observer stopped; empty while it is active. */
    std::string const& Error(std::size_t index) const noexcept;

private:
    struct Input;
    struct Observer;

    /** @brief Copies the rows of one input from the step into its buffer. */
    void Gather(Input& input, ObservedStep const& step, std::size_t agents) const noexcept;

    std::size_t observationStride_ = 0;
    std::size_t worldStride_ = 0;
    std::size_t actionStride_ = 0;
    std::size_t rewardStride_ = 0;
    std::size_t maximumAgents_ = 0;
    std::vector<std::unique_ptr<Observer>> observers_;
};

} // namespace ppokit::runtime
