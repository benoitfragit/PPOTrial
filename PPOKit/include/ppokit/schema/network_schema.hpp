/**
 * @file PPOKit/include/ppokit/schema/network_schema.hpp
 * @brief Actor and critic networks of a task, read from `graph.json`.
 * @details Version 1 describes multilayer perceptrons, the networks of the current
 * examples, and lets an example keep a hand-written network ("model": "external"),
 * such as the BlueDefense attention actor. Input and output sizes are not written
 * in the file: they come from the task (`task.json`), so both files cannot disagree.
 *
 * @code{.json}
 * {
 *   "schema": "ppokit.graph/1",
 *   "actor":  { "model": "mlp", "hidden": { "layers": 2, "width": 128 }, "activation": "tanh",
 *               "log_std": "state_independent" },
 *   "critic": { "model": "mlp", "input": "observation", "hidden": { "layers": 2, "width": 128 },
 *               "activation": "tanh" }
 * }
 * @endcode
 */
#pragma once

#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace ppokit::schema {

/** @brief Activation of the hidden layers of an MLP. */
enum class Activation : std::uint8_t
{
    Identity,
    Relu,
    Tanh,
};

/** @brief How a network is provided. */
enum class NetworkModel : std::uint8_t
{
    /** @brief Generated multilayer perceptron. */
    Mlp,
    /** @brief Written by the example; PPOKit generates nothing for it. */
    External,
};

/** @brief Which vector the critic reads. */
enum class CriticInput : std::uint8_t
{
    /** @brief The actor observation. */
    Observation,
    /** @brief The dedicated critic input of the task (centralized critic). */
    Critic,
};

/** @brief Multilayer perceptron with `hiddenLayers` layers of `width` units. */
struct MlpSpec final
{
    std::size_t hiddenLayers = 2;
    std::size_t width = 64;
    Activation activation = Activation::Tanh;
};

/** @brief Actor: observation -> Gaussian means, plus one learned log-std per action component. */
struct ActorSpec final
{
    NetworkModel model = NetworkModel::Mlp;
    MlpSpec mlp;
    std::size_t inputs = 0;
    std::size_t outputs = 0;
};

/** @brief Critic: observation or critic input -> one value. */
struct CriticSpec final
{
    NetworkModel model = NetworkModel::Mlp;
    MlpSpec mlp;
    CriticInput input = CriticInput::Observation;
    std::size_t inputs = 0;
};

/** @brief Validated network description of a task. */
struct NetworkSchema final
{
    /** @brief C++ namespace of the generated networks header; defaults to "<task namespace>::networks". */
    std::string cppNamespace;
    ActorSpec actor;
    CriticSpec critic;
};

/**
 * @brief Parses and validates a network description against its task.
 * @throws SchemaError naming the faulty key, e.g. "graph.json: critic.input: the task has no critic input".
 */
NetworkSchema ParseNetworkSchema(std::string_view text, std::string_view sourceName, TaskSchema const& task);

/** @brief Reads and parses a network file. @throws SchemaError, including when the file is missing. */
NetworkSchema LoadNetworkSchema(std::filesystem::path const& path, TaskSchema const& task);

/**
 * @brief Generates the rl_tools type definitions of the generated networks.
 * @details The header must be included after the rl_tools operation headers of the
 * consumer, so that it uses the same rl_tools namespace (RL_TOOLS_NAMESPACE_WRAPPER).
 */
std::string GenerateNetworksHeader(NetworkSchema const& networks, TaskSchema const& task,
    std::string_view sourceName);

} // namespace ppokit::schema
