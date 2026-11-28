/**
 * @file Examples/Airspace/Agents/Bindings/data/graph_network.hpp
 * @brief PPOKit graph networks of the domain agents: task.json + graph.json -> learner.
 */
#pragma once

#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"

#include "ppo/core/environment.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace ppo::model {

/** @brief The two files that describe an agent's networks. */
struct GraphNetworkFiles final
{
    /** @brief task.json: observation groups, action heads, reward channels. */
    std::filesystem::path task;
    /** @brief graph.json: actor and critic block graphs. */
    std::filesystem::path graph;
};

/**
 * @brief A PPOKit graph learner built from an agent's files, with the identity its checkpoints carry.
 * @tparam ObservationSize Flat observation size of the agent.
 * @tparam ActionSize Flat action size of the agent.
 */
template<std::size_t ObservationSize, std::size_t ActionSize>
struct GraphNetwork final
{
    using Learner = ppokit::graph::GraphLearner<ObservationSize, ActionSize>;

    ppokit::artifacts::ArtifactIdentity identity;
    std::unique_ptr<Learner> learner;
};

/**
 * @brief Reads task.json and graph.json and builds their learner.
 * @param files Files of the agent.
 * @param settings PPO settings (the seed decides the initial weights).
 * @return The learner and the checkpoint identity.
 * @throw std::runtime_error when the files do not describe these sizes.
 */
template<std::size_t ObservationSize, std::size_t ActionSize>
GraphNetwork<ObservationSize, ActionSize> LoadGraphNetwork(GraphNetworkFiles const& files,
    ppokit::core::PpoSettings const& settings)
{
    ppokit::schema::TaskSchema const task = ppokit::schema::LoadTaskSchema(files.task);
    if (task.observation.size != ObservationSize || task.actionSize != ActionSize)
    {
        throw std::runtime_error(files.task.string() + " declares " + std::to_string(task.observation.size)
            + " observation and " + std::to_string(task.actionSize) + " action values, the agent "
            + std::to_string(ObservationSize) + " and " + std::to_string(ActionSize));
    }
    ppokit::schema::BlockGraphSchema const graphs = ppokit::schema::LoadBlockGraph(files.graph, task);
    GraphNetwork<ObservationSize, ActionSize> network;
    network.identity = ppokit::schema::ArtifactIdentityOf(task, graphs);
    network.learner = std::make_unique<typename GraphNetwork<ObservationSize, ActionSize>::Learner>(
        ppokit::schema::BuildGraph(graphs.actor), ppokit::schema::BuildGraph(graphs.critic),
        ppokit::schema::MakePolicyLayout(graphs, task), settings);
    return network;
}

/**
 * @brief Deterministic production policy of a PPOKit graph: Gaussian means, Bernoulli logit > 0.
 * @details Loads a policy (or training) checkpoint written by the agent's training.
 */
template<std::size_t ObservationSize, std::size_t ActionSize>
class GraphPolicy final : public ppo::core::IPolicy
{
public:
    /**
     * @param files Files of the agent.
     * @param checkpoint policy.ppokit (or a training checkpoint) of these files.
     * @throw std::runtime_error when the checkpoint belongs to another task or network.
     */
    GraphPolicy(GraphNetworkFiles const& files, std::filesystem::path const& checkpoint)
        : network_(LoadGraphNetwork<ObservationSize, ActionSize>(
              files, ppokit::core::InferenceSettings(ppokit::core::PpoSettings{})))
    {
        network_.learner->LoadPolicy(checkpoint, network_.identity);
    }

    ppo::core::Action ChooseAction(ppo::core::Observation const& observation) override
    {
        if (observation.size() != ObservationSize)
        {
            throw std::invalid_argument("GraphPolicy: observation dimension mismatch");
        }
        std::copy(observation.begin(), observation.end(), observation_.begin());
        auto const action = network_.learner->Mean(observation_);
        return ppo::core::Action(action.begin(), action.end());
    }

private:
    GraphNetwork<ObservationSize, ActionSize> network_;
    typename GraphNetwork<ObservationSize, ActionSize>::Learner::Observation observation_{};
};

} // namespace ppo::model
