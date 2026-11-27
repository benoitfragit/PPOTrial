/**
 * @file PPOKit/include/ppokit/schema/block_graph_schema.hpp
 * @brief Actor and critic block graphs of a task, read from `graph.json` (`ppokit.graph/2`).
 * @details Version 2 describes each network as named blocks (nodes) wired port to port
 * (edges), the format of the network editor. Shapes are never written: inputs take theirs
 * from the observation and critic groups of `task.json`, heads from its actions, and every
 * other block infers its output from its inputs and parameters. Every error names the
 * network, the node and the port, e.g.
 * "graph.json: actor.nodes.red_attend.query: expects vector[8], node 'own' gives vector[7]".
 *
 * @code{.json}
 * {
 *   "schema": "ppokit.graph/2",
 *   "actor": {
 *     "nodes": {
 *       "own":    { "type": "input", "from": "observation.own" },
 *       "red":    { "type": "input", "from": "observation.red_tracks" },
 *       "embed":  { "type": "entity_encoder", "width": 8, "activation": "tanh" },
 *       "query":  { "type": "dense", "width": 8, "activation": "tanh" },
 *       "attend": { "type": "masked_attention" },
 *       "pooled": { "type": "masked_pool", "mode": "max" },
 *       "joined": { "type": "concat" },
 *       (also "select" with "indices": [0, 3] to pick features, and "concat_sets" to stack sets)
 *       "trunk":  { "type": "mlp", "layers": 2, "width": 128, "activation": "tanh" },
 *       "policy": { "type": "gaussian_head" }
 *     },
 *     "edges": [ ["red", "embed"], ["own", "query"], ["query", "attend.query"], ["embed", "attend.set"],
 *                ["embed", "pooled"], ["own", "joined.in1"], ["attend", "joined.in2"],
 *                ["pooled", "joined.in3"], ["joined", "trunk"], ["trunk", "policy"] ]
 *   },
 *   "critic": { "nodes": { ... "value": { "type": "value" } }, "edges": [ ... ] }
 * }
 * @endcode
 *
 * Black-box blocks come from component DLLs (ppokit/component/abi.h) listed in
 * "components", found beside graph.json:
 *
 * @code{.json}
 *   "components": ["trackfilter"],
 *   "actor": { "nodes": { "filter": { "type": "component:trackfilter@1", "config": { "alpha": 0.2 } }, ... },
 *              "edges": [ ["red", "filter.tracks"], ... ] }
 * @endcode
 *
 * Their ports are those of their manifest; a component without gradient must come before
 * every trainable block, since nothing could train what lies upstream of it.
 *
 * Observers (Visualiseur blocks) are components that only watch. They live in their own
 * section, outside the networks, in the same form: input nodes read the signals of a step
 * (an observation or world group of task.json, the actions or one action group, the
 * reward channels of the step) and are wired to the observer's ports:
 *
 * @code{.json}
 *   "observers": {
 *     "nodes": { "pose":    { "type": "input", "from": "world.pose" },
 *                "runway":  { "type": "input", "from": "observation.runway" },
 *                "actions": { "type": "input", "from": "action" },
 *                "rewards": { "type": "input", "from": "reward" },
 *                "view":    { "type": "component:flightview@1", "config": { "port": 47000 } } },
 *     "edges": [ ["pose", "view.pose"], ["runway", "view.runway"],
 *                ["actions", "view.controls"], ["rewards", "view.rewards"] ]
 *   }
 * @endcode
 *
 * They change neither the networks, nor the actions, nor the artifact identity.
 */
#pragma once

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/component/library.hpp"
#include "ppokit/graph/graph.hpp"
#include "ppokit/graph/policy_layout.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ppokit::schema {

/** @brief Block types of `ppokit.graph/2`, as written in the "type" key. */
enum class BlockType : std::uint8_t
{
    Input,
    Dense,
    Mlp,
    Concat,
    Select,
    ConcatSets,
    EntityEncoder,
    MaskedAttention,
    MaskedPool,
    GaussianHead,
    BernoulliHead,
    Value,
    /** @brief A black-box block of a component DLL ("component:<name>@<version>"). */
    Component,
};

/** @brief Name of a block type in `graph.json`, e.g. "masked_attention". */
char const* BlockTypeName(BlockType type) noexcept;

/** @brief Block type written `name` in `graph.json`, if there is one. */
std::optional<BlockType> BlockTypeFromName(std::string_view name) noexcept;

/** @brief True for blocks whose ports are numbered in1, in2... (concat, concat_sets). */
bool HasNumberedPorts(BlockType type) noexcept;

/**
 * @brief Input ports of a block, in order.
 * @param numberedPorts Number of ports of a concatenation (ignored for other blocks).
 */
std::vector<std::string> BlockInputPorts(BlockType type, std::size_t numberedPorts);
/**
 * @brief Input ports of a built-in block with its parameters: an attention with
 *        "keys": true has a third port, "keys".
 */
std::vector<std::string> BlockInputPorts(BlockType type, std::size_t numberedPorts, bool separateKeys);

/** @brief Input group of the task read by an input block. */
struct InputSource final
{
    /** @brief True for a group of the critic input, false for the observation. */
    bool critic = false;
    /** @brief Index of the group in its InputLayout. */
    std::size_t group = 0;
};

/** @brief One validated block with its wiring and inferred output shape. */
struct BlockSpec final
{
    std::string name;
    BlockType type = BlockType::Input;
    /** @brief Group read by an input block. */
    InputSource source;
    /** @brief Output width of dense, mlp and entity_encoder blocks. */
    std::size_t width = 0;
    /** @brief Number of layers of an mlp block. */
    std::size_t layers = 0;
    /** @brief Features picked by a select block, in output order. */
    std::vector<std::size_t> indices;
    graph::Activation activation = graph::Activation::Tanh;
    graph::PoolMode pool = graph::PoolMode::Mean;
    /** @brief Attention whose scores read a separate set of keys (port "keys"). */
    bool keys = false;
    /**
     * @brief LibTorch masking by fill value: "mask_fill" of a max pool or an attention (the
     *        value of absent entities), "encode_absent" of an entity encoder.
     */
    std::optional<float> maskFill;
    bool encodeAbsent = false;
    /** @brief Gaussian head: learned log-std, or computed from the state within bounds. */
    graph::LogStdSettings logStd;
    /** @brief Critic block that uses the parameters of this actor block (empty: its own). */
    std::string share;
    /** @brief Source block of each input port, in port order (in1, in2... for concat). */
    std::vector<std::string> sources;
    graph::Shape output;
    /** @brief Component blocks: the component, its complete configuration, and the hash of both. */
    component::ComponentRef component;
    std::string componentConfig;
    std::uint64_t componentHash = 0;
};

/** @brief A network as blocks in topological order. */
struct BlockNetworkSpec final
{
    std::vector<BlockSpec> blocks;
    /** @brief Input blocks in the order Graph::Forward binds them. */
    std::vector<std::string> inputs;
    /**
     * @brief Head blocks in the order of Graph outputs: the Gaussian head, then the
     *        Bernoulli head for an actor; the value head for a critic.
     */
    std::vector<std::string> outputs;
};

/** @brief Signal of an environment step that an observer port reads. */
enum class ObservedSignal : std::uint8_t
{
    /** @brief A group of the observation. */
    Observation,
    /** @brief A group of the world state (task.json "world"). */
    World,
    /** @brief The applied actions, or one action group. */
    Action,
    /** @brief The value of every reward channel of the step, as the run records it, in task.json order. */
    Reward,
};

/** @brief Where one input port of an observer reads, within one agent's signal. */
struct ObservedInput final
{
    /** @brief Port of the observer, as in its manifest. */
    std::string port;
    /** @brief Signal read, as written in the input node's "from", e.g. "world.pose". */
    std::string source;
    ObservedSignal signal = ObservedSignal::Observation;
    /** @brief First value and number of values in the agent's signal vector. */
    std::size_t offset = 0;
    std::size_t size = 0;
    /** @brief Shape of the port for one agent: {features} or {capacity, features} for a set. */
    std::vector<std::size_t> shape;
};

/** @brief One validated observer block. */
struct ObserverSpec final
{
    std::string name;
    component::ComponentRef component;
    /** @brief Complete configuration handed to the DLL. */
    std::string config;
    /** @brief Symbolic dimensions fixed by the wiring, as JSON, e.g. {"N":10}. */
    std::string shapes;
    /** @brief One entry per input port, in manifest order. */
    std::vector<ObservedInput> inputs;
};

/**
 * @brief Centralized critic input of a team, computed by the learner ("team_critic" section).
 * @details Each agent's critic input is built from the output of one actor block for every
 * member of the team: its own row (zero once destroyed), the mean and the maximum over the
 * living members (0 when none lives), the living members and the members as fractions of
 * the task's agent capacity. Each of these goes to one group of task.json "critic"; the
 * groups take them in their layout order. An optional "pad_features": true keeps any
 * trailing positions in the own, mean and max groups at zero; smaller groups are rejected.
 *
 * @code{.json}
 * "team_critic": { "features": "critic_features", "own": "critic.own", "mean": "critic.team_mean",
 *                  "max": "critic.team_max", "living_fraction": "critic.living_fraction",
 *                  "team_fraction": "critic.team_fraction" }
 * @endcode
 */
struct TeamCriticSpec final
{
    /** @brief Actor block whose output (a vector) describes one agent. */
    std::string features;
    /** @brief Width of that output. */
    std::size_t width = 0;
    /** @brief Allow larger critic groups, leaving trailing entries zero. */
    bool padFeatures = false;
    /** @brief Group of the critic input of each part, an index of task.critic->groups. */
    std::size_t own = 0;
    std::size_t mean = 0;
    std::size_t max = 0;
    std::size_t livingFraction = 0;
    std::size_t teamFraction = 0;
};

/** @brief Validated `ppokit.graph/2` file. */
struct BlockGraphSchema final
{
    BlockNetworkSpec actor;
    BlockNetworkSpec critic;
    /** @brief Observers ("observers" section); not part of the artifact identity. */
    std::vector<ObserverSpec> observers;
    /** @brief How a team learner builds the critic input; absent for a single agent. */
    std::optional<TeamCriticSpec> teamCritic;
};

/**
 * @brief Parses and validates a block graph against its task.
 * @details Checks types, parameters, ports (unknown, unwired or wired twice), set and
 * vector kinds, widths, cycles, blocks whose output nobody reads, and that the actor
 * heads cover the task actions (one Gaussian head for the Gaussian components, one
 * Bernoulli head for the Bernoulli ones) and the critic ends in one value head.
 * @throws SchemaError naming the faulty network, node, port or edge.
 */
BlockGraphSchema ParseBlockGraph(std::string_view text, std::string_view sourceName, TaskSchema const& task);

/**
 * @brief Same, with component libraries looked up in `directory` (the folder of graph.json).
 */
BlockGraphSchema ParseBlockGraph(std::string_view text, std::string_view sourceName, TaskSchema const& task,
    std::filesystem::path const& directory);

/** @brief Input ports of a component block: the float "in" ports of its manifest. */
std::vector<std::string> ComponentInputPorts(component::Manifest const& manifest);

/**
 * @brief Reads and parses a block graph file; its component libraries are found beside it.
 * @throws SchemaError, including when the file is missing.
 */
BlockGraphSchema LoadBlockGraph(std::filesystem::path const& path, TaskSchema const& task);

/**
 * @brief The identity checkpoints of these networks carry (ppokit/artifacts/artifact.hpp).
 * @details The graph hash covers every block's type, parameters, wiring and shape, and the
 * hash of every component: anything that changes what the saved numbers mean.
 * @param backend Engine of the networks, e.g. "graph".
 */
artifacts::ArtifactIdentity ArtifactIdentityOf(TaskSchema const& task, BlockGraphSchema const& graphs,
    std::string backend = "graph");

/**
 * @brief Builds the executable graph of a validated network, parameters not initialized.
 * @details Graph inputs follow `network.inputs`, graph outputs `network.outputs`.
 */
graph::Graph BuildGraph(BlockNetworkSpec const& network);

/**
 * @brief Observation slices bound to the inputs of one network, in graph input order.
 * @param networkName "actor" or "critic", for the error message.
 * @throws SchemaError when the network reads a critic group (not part of the observation).
 */
std::vector<graph::InputSlice> MakeInputSlices(BlockNetworkSpec const& network, std::string_view networkName,
    TaskSchema const& task);

/**
 * @brief Slices of the critic input (critic layout of task.json) bound to a hosted critic.
 * @throws SchemaError when the task has no critic layout or the network reads an
 *         observation group.
 */
std::vector<graph::InputSlice> MakeCriticInputSlices(BlockNetworkSpec const& network, TaskSchema const& task);

/**
 * @brief Where the graphs read the observation and write the actions, for a graph learner.
 * @details Each input block becomes a slice of the observation (sets with their mask
 * feature); each head output becomes an action index, in task.json order.
 * @throws SchemaError when a network reads a critic group: a learner that stores one
 *         observation per step cannot feed it (a centralized critic needs the hosted mode).
 */
graph::PolicyLayout MakePolicyLayout(BlockGraphSchema const& graphs, TaskSchema const& task);

} // namespace ppokit::schema
