/**
 * @file PPOKit/src/schema/block_graph_schema.cpp
 * @brief Parsing of `ppokit.graph/2` block graphs and construction of executable graphs.
 * @details Parsing runs in four passes per network: blocks and their parameters, edges
 * resolved to ports, a topological order, then shape inference. Shapes are inferred by
 * constructing the very nodes BuildGraph uses, so validation and execution cannot drift.
 */
#include "ppokit/schema/block_graph_schema.hpp"

#include "json_validation.hpp"

#include "ppokit/component/component_node.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ppokit::schema {
namespace {

using detail::ChildPath;
using detail::ErrorReporter;
using detail::IsIdentifier;
using detail::Json;
using detail::RequireKnownKeys;
using detail::RequireObject;
using detail::RequirePositiveInteger;

/** @brief Format identifier accepted in the "schema" key. */
constexpr char const* BlockGraphSchemaVersion = "ppokit.graph/2";

/** @brief Error on one input port while a node is built, reported with its path by the parser. */
struct PortError final
{
    std::size_t port = 0;
    std::string message;
};

struct BlockTypeEntry final
{
    char const* name;
    BlockType type;
};

constexpr BlockTypeEntry BlockTypes[] = {
    {"input", BlockType::Input},
    {"dense", BlockType::Dense},
    {"mlp", BlockType::Mlp},
    {"concat", BlockType::Concat},
    {"select", BlockType::Select},
    {"concat_sets", BlockType::ConcatSets},
    {"entity_encoder", BlockType::EntityEncoder},
    {"masked_attention", BlockType::MaskedAttention},
    {"masked_pool", BlockType::MaskedPool},
    {"gaussian_head", BlockType::GaussianHead},
    {"bernoulli_head", BlockType::BernoulliHead},
    {"value", BlockType::Value},
};

/** @brief Prefix of component block types: "component:trackfilter@1". */
constexpr char const* ComponentPrefix = "component:";

/** @brief Component libraries of one graph.json, opened from its "components" list. */
using Libraries = std::vector<std::shared_ptr<component::Library const>>;

/** @brief Seed of a component instance: a hash of its node name, so that a rebuild gives the same parameters. */
std::uint64_t ComponentSeed(std::string const& name) noexcept
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (char const character : name)
    {
        hash = (hash ^ static_cast<unsigned char>(character)) * 1099511628211ULL;
    }
    return hash;
}

bool IsHead(BlockType type) noexcept
{
    return type == BlockType::GaussianHead || type == BlockType::BernoulliHead || type == BlockType::Value;
}

/**
 * @brief Input ports of a built-in block; concatenations have one port per wired input
 *        (in1, in2...), an attention with separate keys a "keys" port.
 */
std::vector<std::string> PortsOf(BlockType type, std::size_t numberedPorts, bool separateKeys = false)
{
    switch (type)
    {
    case BlockType::Input:
        return {};
    case BlockType::MaskedAttention:
        if (separateKeys)
        {
            return {"query", "set", "keys"};
        }
        return {"query", "set"};
    case BlockType::Concat:
    case BlockType::ConcatSets:
    {
        std::vector<std::string> ports;
        for (std::size_t index = 1; index <= numberedPorts; ++index)
        {
            ports.push_back("in" + std::to_string(index));
        }
        return ports;
    }
    default:
        break;
    }
    return {"in"};
}

/** @brief Input ports of any block, components included. */
std::vector<std::string> PortsOf(BlockSpec const& block, std::size_t numberedPorts)
{
    if (block.type == BlockType::Component)
    {
        return ComponentInputPorts(*block.component.manifest);
    }
    return PortsOf(block.type, numberedPorts, block.keys);
}

/** @brief Kind each input port accepts: sets for encoders, pools and the attention set. */
graph::SignalKind PortKind(BlockType type, std::size_t port) noexcept
{
    bool const readsSet = type == BlockType::EntityEncoder || type == BlockType::MaskedPool
        || type == BlockType::ConcatSets || (type == BlockType::MaskedAttention && port >= 1);
    return readsSet ? graph::SignalKind::Set : graph::SignalKind::Vector;
}

/** @brief Constructs the graph node of a block, its parameters read from `block`. */
std::unique_ptr<graph::Node> ConstructNode(BlockSpec const& block, std::vector<graph::Shape> const& inputs)
{
    switch (block.type)
    {
    case BlockType::Input:
        return std::make_unique<graph::InputNode>(block.output);
    case BlockType::Dense:
        return std::make_unique<graph::DenseNode>(inputs[0].width, block.width, block.activation);
    case BlockType::Mlp:
        return std::make_unique<graph::MlpNode>(inputs[0].width, block.layers, block.width, block.activation);
    case BlockType::Concat:
    {
        std::vector<std::size_t> widths;
        for (graph::Shape const& shape : inputs)
        {
            widths.push_back(shape.width);
        }
        return std::make_unique<graph::ConcatNode>(widths);
    }
    case BlockType::Select:
        return std::make_unique<graph::SelectNode>(inputs[0].width, block.indices);
    case BlockType::ConcatSets:
    {
        // The first set fixes the width; another width is reported on its port.
        std::vector<std::size_t> capacities;
        for (graph::Shape const& shape : inputs)
        {
            capacities.push_back(shape.capacity);
        }
        return std::make_unique<graph::ConcatSetsNode>(capacities, inputs[0].width);
    }
    case BlockType::EntityEncoder:
        return std::make_unique<graph::EntityEncoderNode>(
            inputs[0].capacity, inputs[0].width, block.width, block.activation, block.encodeAbsent);
    case BlockType::MaskedAttention:
        // The set fixes the width; a query of another width is reported on its port.
        return std::make_unique<graph::MaskedAttentionNode>(
            inputs[1].capacity, inputs[1].width, block.keys, block.maskFill);
    case BlockType::MaskedPool:
        return std::make_unique<graph::MaskedPoolNode>(inputs[0].capacity, inputs[0].width, block.pool, block.maskFill);
    case BlockType::GaussianHead:
        return std::make_unique<graph::HeadNode>(
            graph::HeadKind::Gaussian, inputs[0].width, block.width, 0.F, block.logStd);
    case BlockType::BernoulliHead:
        return std::make_unique<graph::HeadNode>(graph::HeadKind::Bernoulli, inputs[0].width, block.width);
    case BlockType::Component:
        return std::make_unique<component::ComponentNode>(
            block.component, block.componentConfig, inputs, ComponentSeed(block.name));
    case BlockType::Value:
        break;
    }
    return std::make_unique<graph::HeadNode>(graph::HeadKind::Value, inputs[0].width, 1);
}

/**
 * @brief Builds the node of a block after checking its input kinds and widths.
 * @throws PortError on the first port whose signal does not fit.
 */
std::unique_ptr<graph::Node> MakeNode(BlockSpec const& block, std::vector<graph::Shape> const& inputs,
    std::vector<std::string> const& sourceNames)
{
    // A component checks its ports itself, against its manifest.
    for (std::size_t port = 0; port < inputs.size() && block.type != BlockType::Component; ++port)
    {
        if (inputs[port].kind != PortKind(block.type, port))
        {
            bool const wantsSet = PortKind(block.type, port) == graph::SignalKind::Set;
            throw PortError{port, std::string("expects a ") + (wantsSet ? "set" : "vector") + ", node '"
                    + sourceNames[port] + "' gives " + graph::Describe(inputs[port])};
        }
    }
    auto node = ConstructNode(block, inputs);
    auto const expected = node->InputShapes();
    for (std::size_t port = 0; port < inputs.size(); ++port)
    {
        if (expected[port] != inputs[port])
        {
            throw PortError{port, "expects " + graph::Describe(expected[port]) + ", node '" + sourceNames[port]
                    + "' gives " + graph::Describe(inputs[port])};
        }
    }
    return node;
}

/** @brief An optional boolean parameter (false when absent). */
bool ReadFlag(Json const& node, char const* key, std::string const& path, ErrorReporter const& errors)
{
    if (!node.contains(key))
    {
        return false;
    }
    if (!node.at(key).is_boolean())
    {
        errors.Fail(ChildPath(path, key), "expected true or false");
    }
    return node.at(key).get<bool>();
}

/** @brief "mask_fill": the finite value of absent entities (nothing when absent). */
std::optional<float> ReadMaskFill(Json const& node, std::string const& path, ErrorReporter const& errors)
{
    if (!node.contains("mask_fill"))
    {
        return std::nullopt;
    }
    Json const& fill = node.at("mask_fill");
    if (!fill.is_number() || !std::isfinite(fill.get<double>()))
    {
        errors.Fail(ChildPath(path, "mask_fill"), "expected a finite number, e.g. -1e9");
    }
    return fill.get<float>();
}

/**
 * @brief "log_std" of a Gaussian head: "parameter" (default, learned and state independent)
 *        or "state" (a dense layer of the head's input, clamped to log_std_min..log_std_max).
 */
graph::LogStdSettings ReadLogStd(Json const& node, std::string const& path, ErrorReporter const& errors)
{
    graph::LogStdSettings settings;
    std::string const mode = node.value("log_std", std::string("parameter"));
    bool const hasBounds = node.contains("log_std_min") || node.contains("log_std_max");
    if (mode == "parameter")
    {
        if (hasBounds)
        {
            errors.Fail(path, "log_std_min and log_std_max bound a log-std computed from the state (\"log_std\": "
                              "\"state\"); a learned one is bounded by training.json");
        }
        return settings;
    }
    if (mode != "state")
    {
        errors.Fail(ChildPath(path, "log_std"), "expected \"parameter\" or \"state\"");
    }
    for (char const* key : {"log_std_min", "log_std_max"})
    {
        if (!node.contains(key) || !node.at(key).is_number())
        {
            errors.Fail(ChildPath(path, key), "a log-std computed from the state needs its bounds");
        }
    }
    settings.stateDependent = true;
    settings.minimum = node.at("log_std_min").get<float>();
    settings.maximum = node.at("log_std_max").get<float>();
    if (!(settings.minimum < settings.maximum))
    {
        errors.Fail(ChildPath(path, "log_std_max"), "must be above log_std_min");
    }
    return settings;
}

graph::Activation ReadActivation(Json const& node, std::string const& path, ErrorReporter const& errors)
{
    if (!node.contains("activation"))
    {
        return graph::Activation::Tanh;
    }
    Json const& activation = node.at("activation");
    if (activation == "tanh")
    {
        return graph::Activation::Tanh;
    }
    if (activation == "relu")
    {
        return graph::Activation::Relu;
    }
    if (activation == "identity")
    {
        return graph::Activation::Identity;
    }
    errors.Fail(ChildPath(path, "activation"), "expected \"tanh\", \"relu\" or \"identity\"");
}

std::size_t RequireKey(Json const& node, char const* key, std::string const& path, ErrorReporter const& errors)
{
    if (!node.contains(key))
    {
        errors.Fail(ChildPath(path, key), "missing (a positive integer)");
    }
    return RequirePositiveInteger(node.at(key), ChildPath(path, key), errors);
}

/** @brief Resolves "observation.<group>" or "critic.<group>" against the task. */
InputSource ReadInputSource(Json const& node, std::string const& path, TaskSchema const& task, bool actor,
    ErrorReporter const& errors)
{
    std::string const fromPath = ChildPath(path, "from");
    if (!node.contains("from") || !node.at("from").is_string())
    {
        errors.Fail(fromPath, "missing, e.g. \"observation.<group>\"");
    }
    std::string const from = node.at("from").get<std::string>();
    std::size_t const dot = from.find('.');
    std::string const layoutName = from.substr(0, dot);
    std::string const groupName = dot == std::string::npos ? std::string() : from.substr(dot + 1);
    InputSource source;
    source.critic = layoutName == "critic";
    if (layoutName != "observation" && !source.critic)
    {
        errors.Fail(fromPath, "expected \"observation.<group>\" or \"critic.<group>\"");
    }
    if (source.critic && actor)
    {
        errors.Fail(fromPath, "the actor reads only observation groups (the critic input is centralized)");
    }
    if (source.critic && !task.critic)
    {
        errors.Fail(fromPath, "the task has no \"critic\" input in task.json");
    }
    InputLayout const& layout = source.critic ? *task.critic : task.observation;
    for (std::size_t group = 0; group < layout.groups.size(); ++group)
    {
        if (layout.groups[group].name == groupName)
        {
            source.group = group;
            return source;
        }
    }
    std::string known;
    for (ObservationGroup const& group : layout.groups)
    {
        known += (known.empty() ? "" : ", ") + group.name;
    }
    errors.Fail(fromPath, "no group '" + groupName + "' in " + layoutName + " (groups: " + known + ")");
}

graph::Shape InputShape(InputSource const& source, TaskSchema const& task)
{
    ObservationGroup const& group =
        source.critic ? task.critic->groups[source.group] : task.observation.groups[source.group];
    if (group.isSet)
    {
        return graph::Shape::Set(group.capacity, group.features);
    }
    return graph::Shape::Vector(group.Size());
}

std::size_t ActionComponents(TaskSchema const& task, ActionHead head) noexcept
{
    std::size_t components = 0;
    for (ActionGroup const& group : task.actions)
    {
        if (group.head == head)
        {
            components += group.Size();
        }
    }
    return components;
}

/** @brief Feature indices of a select block; their range is checked against its input later. */
std::vector<std::size_t> ReadIndices(Json const& node, std::string const& path, ErrorReporter const& errors)
{
    std::string const indicesPath = ChildPath(path, "indices");
    if (!node.contains("indices") || !node.at("indices").is_array() || node.at("indices").empty())
    {
        errors.Fail(indicesPath, "missing, e.g. [0, 3] (features picked from the input vector)");
    }
    std::vector<std::size_t> indices;
    std::size_t position = 0;
    for (Json const& index : node.at("indices"))
    {
        std::string const indexPath = indicesPath + "[" + std::to_string(position++) + "]";
        if (!index.is_number_integer() || index.get<long long>() < 0)
        {
            errors.Fail(indexPath, "expected a non-negative integer");
        }
        indices.push_back(index.get<std::size_t>());
    }
    return indices;
}

/** @brief Reads the type and parameters of one block (wiring and shapes come later). */
/**
 * @brief A "component:<name>@<version>" block: the component found in the libraries, its
 *        configuration validated against its manifest.
 */
BlockSpec ReadComponent(std::string const& name, Json const& node, std::string const& path, Libraries const& libraries,
    ErrorReporter const& errors)
{
    RequireKnownKeys(node, {"type", "config"}, path, errors);
    std::string const typePath = ChildPath(path, "type");
    std::string const reference = node.at("type").get<std::string>().substr(std::string(ComponentPrefix).size());
    std::size_t const at = reference.rfind('@');
    if (at == std::string::npos || at == 0 || at + 1 == reference.size())
    {
        errors.Fail(typePath, "expected \"component:<name>@<version>\", e.g. \"component:trackfilter@1\"");
    }
    std::string const componentName = reference.substr(0, at);
    std::string const versionText = reference.substr(at + 1);
    if (versionText.find_first_not_of("0123456789") != std::string::npos)
    {
        errors.Fail(typePath, "the version after '@' is a whole number");
    }
    auto const version = static_cast<std::uint32_t>(std::stoul(versionText));
    BlockSpec block;
    block.name = name;
    block.type = BlockType::Component;
    std::string available;
    for (auto const& library : libraries)
    {
        for (std::size_t index = 0; index < library->Count(); ++index)
        {
            component::ComponentRef const candidate = library->At(index);
            if (candidate.manifest->name == componentName && candidate.manifest->version == version)
            {
                block.component = candidate;
            }
            available += (available.empty() ? "" : ", ") + candidate.Id();
        }
    }
    if (block.component.api == nullptr)
    {
        errors.Fail(typePath, "no component " + reference + " in the libraries of \"components\" (found: "
                + (available.empty() ? std::string("none") : available) + ")");
    }
    try
    {
        std::string const config = node.contains("config") ? node.at("config").dump() : std::string("{}");
        block.componentConfig = component::ValidateConfig(*block.component.manifest, config);
    }
    catch (std::runtime_error const& error)
    {
        errors.Fail(ChildPath(path, "config"), error.what());
    }
    block.componentHash = component::ComponentHash(block.component, block.componentConfig);
    return block;
}

BlockSpec ReadBlock(std::string const& name, Json const& node, std::string const& path, TaskSchema const& task,
    bool actor, Libraries const& libraries, ErrorReporter const& errors)
{
    RequireObject(node, path, errors);
    std::string const typePath = ChildPath(path, "type");
    if (!node.contains("type") || !node.at("type").is_string())
    {
        errors.Fail(typePath, "missing, e.g. \"dense\"");
    }
    std::string const typeName = node.at("type").get<std::string>();
    if (typeName.rfind(ComponentPrefix, 0) == 0)
    {
        return ReadComponent(name, node, path, libraries, errors);
    }
    auto const entry = std::find_if(std::begin(BlockTypes), std::end(BlockTypes),
        [&typeName](BlockTypeEntry const& candidate) { return typeName == candidate.name; });
    if (entry == std::end(BlockTypes))
    {
        std::string known;
        for (BlockTypeEntry const& candidate : BlockTypes)
        {
            known += (known.empty() ? "" : ", ") + std::string(candidate.name);
        }
        errors.Fail(typePath, "unknown block type '" + typeName + "' (types: " + known
                + ", or component:<name>@<version>)");
    }
    BlockSpec block;
    block.name = name;
    block.type = entry->type;
    switch (block.type)
    {
    case BlockType::Input:
        RequireKnownKeys(node, {"type", "from"}, path, errors);
        block.source = ReadInputSource(node, path, task, actor, errors);
        block.output = InputShape(block.source, task);
        break;
    case BlockType::Dense:
        RequireKnownKeys(node, {"type", "width", "activation", "share"}, path, errors);
        block.width = RequireKey(node, "width", path, errors);
        block.activation = ReadActivation(node, path, errors);
        break;
    case BlockType::EntityEncoder:
        RequireKnownKeys(node, {"type", "width", "activation", "share", "encode_absent"}, path, errors);
        block.width = RequireKey(node, "width", path, errors);
        block.activation = ReadActivation(node, path, errors);
        block.encodeAbsent = ReadFlag(node, "encode_absent", path, errors);
        break;
    case BlockType::Mlp:
        RequireKnownKeys(node, {"type", "layers", "width", "activation", "share"}, path, errors);
        block.layers = RequireKey(node, "layers", path, errors);
        block.width = RequireKey(node, "width", path, errors);
        block.activation = ReadActivation(node, path, errors);
        break;
    case BlockType::MaskedPool:
    {
        RequireKnownKeys(node, {"type", "mode", "mask_fill"}, path, errors);
        Json const mode = node.contains("mode") ? node.at("mode") : Json();
        if (mode != "mean" && mode != "max")
        {
            errors.Fail(ChildPath(path, "mode"), "expected \"mean\" or \"max\"");
        }
        block.pool = mode == "max" ? graph::PoolMode::Max : graph::PoolMode::Mean;
        block.maskFill = ReadMaskFill(node, path, errors);
        if (block.maskFill && block.pool != graph::PoolMode::Max)
        {
            errors.Fail(ChildPath(path, "mask_fill"), "a fill value applies to a \"max\" pool only");
        }
        break;
    }
    case BlockType::GaussianHead:
        // The initial log-std of a learned one is a training setting (PpoSettings), not part of the network.
        RequireKnownKeys(node, {"type", "log_std", "log_std_min", "log_std_max", "share"}, path, errors);
        block.width = ActionComponents(task, ActionHead::Gaussian);
        block.logStd = ReadLogStd(node, path, errors);
        break;
    case BlockType::BernoulliHead:
        RequireKnownKeys(node, {"type", "share"}, path, errors);
        block.width = ActionComponents(task, ActionHead::Bernoulli);
        break;
    case BlockType::Select:
        RequireKnownKeys(node, {"type", "indices"}, path, errors);
        block.indices = ReadIndices(node, path, errors);
        break;
    case BlockType::MaskedAttention:
        RequireKnownKeys(node, {"type", "keys", "mask_fill"}, path, errors);
        block.keys = ReadFlag(node, "keys", path, errors);
        block.maskFill = ReadMaskFill(node, path, errors);
        break;
    case BlockType::Concat:
    case BlockType::ConcatSets:
        RequireKnownKeys(node, {"type"}, path, errors);
        break;
    case BlockType::Value:
        RequireKnownKeys(node, {"type", "share"}, path, errors);
        break;
    case BlockType::Component:
        break; // read by ReadComponent
    }
    if (node.contains("share"))
    {
        Json const& share = node.at("share");
        if (!share.is_string() || share.get<std::string>().empty())
        {
            errors.Fail(ChildPath(path, "share"), "expected the name of an actor block");
        }
        block.share = share.get<std::string>();
    }
    return block;
}

/** @brief Edges grouped by destination block: port name -> (source block, edge path). */
using PortWiring = std::map<std::string, std::map<std::string, std::pair<std::string, std::string>>>;

PortWiring ReadEdges(Json const& network, std::string const& networkPath, std::map<std::string, std::size_t> const& index,
    std::vector<BlockSpec> const& blocks, ErrorReporter const& errors)
{
    std::string const edgesPath = ChildPath(networkPath, "edges");
    if (!network.contains("edges") || !network.at("edges").is_array())
    {
        errors.Fail(edgesPath, "missing, e.g. [[\"source\", \"target.port\"], ...]");
    }
    PortWiring wiring;
    std::size_t position = 0;
    for (Json const& edge : network.at("edges"))
    {
        std::string const edgePath = edgesPath + "[" + std::to_string(position++) + "]";
        if (!edge.is_array() || edge.size() != 2 || !edge[0].is_string() || !edge[1].is_string())
        {
            errors.Fail(edgePath, "expected [\"source\", \"target\"] or [\"source\", \"target.port\"]");
        }
        std::string const source = edge[0].get<std::string>();
        std::string const target = edge[1].get<std::string>();
        std::size_t const dot = target.find('.');
        std::string const targetName = target.substr(0, dot);
        std::string port = dot == std::string::npos ? std::string() : target.substr(dot + 1);
        if (index.count(source) == 0)
        {
            errors.Fail(edgePath, "unknown source node '" + source + "' (a source has no port)");
        }
        if (index.count(targetName) == 0)
        {
            errors.Fail(edgePath, "unknown target node '" + targetName + "'");
        }
        BlockSpec const& block = blocks[index.at(targetName)];
        if (block.type == BlockType::Input)
        {
            errors.Fail(edgePath, "node '" + targetName + "' is an input and has no input port");
        }
        if (port.empty())
        {
            if (block.type == BlockType::Component)
            {
                // A component with one input takes an unnamed edge on it.
                std::vector<std::string> const ports = PortsOf(block, 0);
                if (ports.size() != 1)
                {
                    std::string names;
                    for (std::string const& name : ports)
                    {
                        names += (names.empty() ? "" : ", ") + name;
                    }
                    errors.Fail(edgePath, "node '" + targetName + "' has several ports, name one (" + names + ")");
                }
                port = ports[0];
            }
            else if (block.type == BlockType::MaskedAttention || HasNumberedPorts(block.type))
            {
                std::string const attentionPorts = block.keys ? "query, set or keys" : "query or set";
                errors.Fail(edgePath, "node '" + targetName + "' has several ports, name one ("
                        + (HasNumberedPorts(block.type) ? "in1, in2..." : attentionPorts) + ")");
            }
            else
            {
                port = "in";
            }
        }
        auto& ports = wiring[targetName];
        if (ports.count(port) != 0)
        {
            errors.Fail(edgePath, "port '" + targetName + "." + port + "' is already wired from '"
                    + ports.at(port).first + "' (" + ports.at(port).second + ")");
        }
        ports[port] = {source, edgePath};
    }
    return wiring;
}

/** @brief Fills `sources` of every block from the wiring, checking port names and coverage. */
void ResolvePorts(std::vector<BlockSpec>& blocks, PortWiring const& wiring, std::string const& networkPath,
    ErrorReporter const& errors)
{
    for (BlockSpec& block : blocks)
    {
        std::string const blockPath = ChildPath(ChildPath(networkPath, "nodes"), block.name);
        auto const wired = wiring.find(block.name);
        std::size_t const wiredCount = wired == wiring.end() ? 0 : wired->second.size();
        std::vector<std::string> const ports = PortsOf(block, wiredCount);
        if (wired != wiring.end())
        {
            for (auto const& [port, origin] : wired->second)
            {
                if (std::find(ports.begin(), ports.end(), port) == ports.end())
                {
                    std::string portList;
                    for (std::string const& name : ports)
                    {
                        portList += (portList.empty() ? "" : ", ") + name;
                    }
                    std::string const hint = HasNumberedPorts(block.type)
                        ? std::string(BlockTypeName(block.type)) + " ports are in1..inN without gaps"
                        : "ports of " + std::string(BlockTypeName(block.type)) + ": " + portList;
                    errors.Fail(origin.second, "node '" + block.name + "' has no port '" + port + "' (" + hint + ")");
                }
            }
        }
        if (ports.empty() && HasNumberedPorts(block.type))
        {
            errors.Fail(blockPath, std::string(BlockTypeName(block.type)) + " has no input (wire in1, in2...)");
        }
        for (std::string const& port : ports)
        {
            if (wired == wiring.end() || wired->second.count(port) == 0)
            {
                errors.Fail(ChildPath(blockPath, port), "input port is not wired");
            }
            block.sources.push_back(wired->second.at(port).first);
        }
    }
}

/**
 * @brief Orders blocks so that each comes after its sources (Kahn's algorithm).
 * @details Among ready blocks the first in file order goes first, so inputs keep the
 * order the author wrote them in. A cycle is reported with the blocks it involves.
 */
std::vector<BlockSpec> TopologicalOrder(std::vector<BlockSpec> blocks, std::string const& networkPath,
    ErrorReporter const& errors)
{
    std::vector<BlockSpec> ordered;
    std::set<std::string> placed;
    while (!blocks.empty())
    {
        auto const ready = std::find_if(blocks.begin(), blocks.end(), [&placed](BlockSpec const& block) {
            return std::all_of(block.sources.begin(), block.sources.end(),
                [&placed](std::string const& source) { return placed.count(source) != 0; });
        });
        if (ready == blocks.end())
        {
            std::string cycle;
            for (BlockSpec const& block : blocks)
            {
                cycle += (cycle.empty() ? "" : ", ") + block.name;
            }
            errors.Fail(ChildPath(networkPath, "edges"), "cycle between nodes " + cycle);
        }
        placed.insert(ready->name);
        ordered.push_back(std::move(*ready));
        blocks.erase(ready);
    }
    return ordered;
}

/** @brief Infers every output shape in topological order by building the nodes. */
void InferShapes(std::vector<BlockSpec>& blocks, std::string const& networkPath, ErrorReporter const& errors)
{
    std::map<std::string, graph::Shape> shapes;
    for (BlockSpec& block : blocks)
    {
        std::string const blockPath = ChildPath(ChildPath(networkPath, "nodes"), block.name);
        if (block.type == BlockType::Input)
        {
            shapes[block.name] = block.output;
            continue;
        }
        if ((block.type == BlockType::GaussianHead || block.type == BlockType::BernoulliHead) && block.width == 0)
        {
            errors.Fail(blockPath, std::string("the task has no ")
                    + (block.type == BlockType::GaussianHead ? "Gaussian" : "Bernoulli") + " action");
        }
        std::vector<graph::Shape> inputs;
        for (std::string const& source : block.sources)
        {
            inputs.push_back(shapes.at(source));
        }
        try
        {
            block.output = MakeNode(block, inputs, block.sources)->OutputShape();
        }
        catch (PortError const& error)
        {
            auto const ports = PortsOf(block, block.sources.size());
            errors.Fail(ChildPath(blockPath, ports[error.port]), error.message);
        }
        catch (std::invalid_argument const& error)
        {
            // Parameters the node rejects for its input, e.g. a select index past the width.
            errors.Fail(blockPath, error.what());
        }
        shapes[block.name] = block.output;
    }
}

/**
 * @brief Whether a block learns: it has parameters the optimizer updates. Attention and
 *        pooling have none; what feeds them is followed through their sources.
 */
bool IsTrainable(BlockSpec const& block) noexcept
{
    switch (block.type)
    {
    case BlockType::Dense:
    case BlockType::Mlp:
    case BlockType::EntityEncoder:
    case BlockType::GaussianHead:
    case BlockType::BernoulliHead:
    case BlockType::Value:
        return true;
    case BlockType::Component:
        return block.component.manifest->capabilities.trainable;
    default:
        return false;
    }
}

/**
 * @brief A component without gradient must come before every trainable block: nothing
 *        upstream of it could learn (rule 12.4 of the design).
 */
void RequireGradientPaths(std::vector<BlockSpec> const& blocks, std::string const& networkPath,
    ErrorReporter const& errors)
{
    // Blocks are in topological order: the trainable ancestor of each block is known when it is reached.
    std::map<std::string, std::string> trainableAncestor;
    for (BlockSpec const& block : blocks)
    {
        std::string ancestor;
        for (std::string const& source : block.sources)
        {
            auto const found = trainableAncestor.find(source);
            if (found != trainableAncestor.end() && !found->second.empty())
            {
                ancestor = found->second;
            }
        }
        bool const blocksGradient =
            block.type == BlockType::Component && !block.component.manifest->capabilities.differentiable;
        if (blocksGradient && !ancestor.empty())
        {
            errors.Fail(ChildPath(ChildPath(networkPath, "nodes"), block.name),
                "component " + block.component.Id() + " has no gradient, so the trainable block '" + ancestor
                    + "' before it could not learn: place it on the inputs, before every trainable block");
        }
        trainableAncestor[block.name] = IsTrainable(block) ? block.name : ancestor;
    }
}

/**
 * @brief Rejects blocks whose output nobody reads, and heads that feed other blocks.
 * @param readOutside Blocks read from outside the network (the team critic's features).
 */
void RequireConsumed(std::vector<BlockSpec> const& blocks, std::set<std::string> const& readOutside,
    std::string const& networkPath, ErrorReporter const& errors)
{
    std::set<std::string> consumed = readOutside;
    for (BlockSpec const& block : blocks)
    {
        consumed.insert(block.sources.begin(), block.sources.end());
    }
    for (BlockSpec const& block : blocks)
    {
        std::string const blockPath = ChildPath(ChildPath(networkPath, "nodes"), block.name);
        if (IsHead(block.type) && consumed.count(block.name) != 0)
        {
            errors.Fail(blockPath, "a head is an output and cannot feed other nodes");
        }
        if (!IsHead(block.type) && consumed.count(block.name) == 0)
        {
            errors.Fail(blockPath, "output is not used by any node");
        }
    }
}

/** @brief Finds the single head of a type; fails if there are several, or none when required. */
std::string FindHead(std::vector<BlockSpec> const& blocks, BlockType type, bool required,
    std::string const& networkPath, ErrorReporter const& errors)
{
    std::string found;
    for (BlockSpec const& block : blocks)
    {
        if (block.type != type)
        {
            continue;
        }
        if (!found.empty())
        {
            errors.Fail(ChildPath(ChildPath(networkPath, "nodes"), block.name),
                std::string("a second ") + BlockTypeName(type) + " (one head per kind, '" + found + "' exists)");
        }
        found = block.name;
    }
    if (found.empty() && required)
    {
        errors.Fail(ChildPath(networkPath, "nodes"), std::string("missing a ") + BlockTypeName(type) + " node");
    }
    return found;
}

/** @brief Fails if the network holds a head of a type it must not have. */
void ForbidHead(std::vector<BlockSpec> const& blocks, BlockType type, char const* reason,
    std::string const& networkPath, ErrorReporter const& errors)
{
    std::string const head = FindHead(blocks, type, false, networkPath, errors);
    if (!head.empty())
    {
        errors.Fail(ChildPath(ChildPath(networkPath, "nodes"), head), reason);
    }
}

/** @brief Checks the heads of a network and records its outputs in graph order. */
void ResolveOutputs(BlockNetworkSpec& network, bool actor, TaskSchema const& task, std::string const& networkPath,
    ErrorReporter const& errors)
{
    if (!actor)
    {
        ForbidHead(network.blocks, BlockType::GaussianHead, "a critic has no action head", networkPath, errors);
        ForbidHead(network.blocks, BlockType::BernoulliHead, "a critic has no action head", networkPath, errors);
        network.outputs.push_back(FindHead(network.blocks, BlockType::Value, true, networkPath, errors));
        return;
    }
    ForbidHead(network.blocks, BlockType::Value, "the value head belongs to the critic", networkPath, errors);
    bool const gaussian = ActionComponents(task, ActionHead::Gaussian) != 0;
    bool const bernoulli = ActionComponents(task, ActionHead::Bernoulli) != 0;
    std::string const gaussianHead = FindHead(network.blocks, BlockType::GaussianHead, gaussian, networkPath, errors);
    std::string const bernoulliHead =
        FindHead(network.blocks, BlockType::BernoulliHead, bernoulli, networkPath, errors);
    if (!gaussianHead.empty())
    {
        network.outputs.push_back(gaussianHead);
    }
    if (!bernoulliHead.empty())
    {
        network.outputs.push_back(bernoulliHead);
    }
}

/** @brief Input shapes of a block, from the outputs of its sources. */
std::vector<graph::Shape> SourceShapes(BlockNetworkSpec const& network, BlockSpec const& block)
{
    std::vector<graph::Shape> shapes;
    for (std::string const& source : block.sources)
    {
        for (BlockSpec const& candidate : network.blocks)
        {
            if (candidate.name == source)
            {
                shapes.push_back(candidate.output);
            }
        }
    }
    return shapes;
}

/**
 * @brief A critic block that shares an actor block must be the same layer: type,
 *        parameters and input shapes; the actor never shares (it owns the parameters).
 */
void CheckSharedBlocks(BlockGraphSchema const& graphs, ErrorReporter const& errors)
{
    for (BlockSpec const& block : graphs.actor.blocks)
    {
        if (!block.share.empty())
        {
            errors.Fail(ChildPath(ChildPath("actor.nodes", block.name), "share"),
                "only a critic block shares the parameters of an actor block");
        }
    }
    for (BlockSpec const& block : graphs.critic.blocks)
    {
        if (block.share.empty())
        {
            continue;
        }
        std::string const path = ChildPath(ChildPath("critic.nodes", block.name), "share");
        auto const actor = std::find_if(graphs.actor.blocks.begin(), graphs.actor.blocks.end(),
            [&block](BlockSpec const& candidate) { return candidate.name == block.share; });
        if (actor == graphs.actor.blocks.end())
        {
            errors.Fail(path, "no actor block '" + block.share + "'");
        }
        bool const sameLayer = actor->type == block.type && actor->width == block.width
            && actor->layers == block.layers && actor->activation == block.activation && actor->output == block.output
            && !IsHead(block.type) && IsTrainable(block) && block.type != BlockType::Component
            && SourceShapes(graphs.actor, *actor) == SourceShapes(graphs.critic, block);
        if (!sameLayer)
        {
            errors.Fail(path, "actor block '" + block.share
                    + "' is not the same layer (a dense, mlp or entity_encoder block of the same type, sizes, "
                      "activation and inputs)");
        }
    }
}

/**
 * @brief Reads, orders and checks one network.
 * @param actor True for the actor (observation inputs, action heads), false for the critic.
 */
BlockNetworkSpec ReadNetwork(Json const& root, std::string const& networkPath, bool actor, TaskSchema const& task,
    Libraries const& libraries, std::set<std::string> const& readOutside, ErrorReporter const& errors)
{
    if (!root.contains(networkPath))
    {
        errors.Fail(networkPath, "missing (an object with \"nodes\" and \"edges\")");
    }
    Json const& network = root.at(networkPath);
    RequireObject(network, networkPath, errors);
    RequireKnownKeys(network, {"nodes", "edges"}, networkPath, errors);
    std::string const nodesPath = ChildPath(networkPath, "nodes");
    if (!network.contains("nodes"))
    {
        errors.Fail(nodesPath, "missing (an object of named nodes)");
    }
    RequireObject(network.at("nodes"), nodesPath, errors);

    std::vector<BlockSpec> blocks;
    std::map<std::string, std::size_t> index;
    for (auto const& [name, node] : network.at("nodes").items())
    {
        if (!IsIdentifier(name))
        {
            errors.Fail(ChildPath(nodesPath, name), "node names are identifiers (letters, digits, '_')");
        }
        index[name] = blocks.size();
        blocks.push_back(ReadBlock(name, node, ChildPath(nodesPath, name), task, actor, libraries, errors));
        BlockSpec const& block = blocks.back();
        if (block.type == BlockType::Component && block.component.manifest->capabilities.observer)
        {
            errors.Fail(ChildPath(nodesPath, name), "component " + block.component.Id()
                    + " is an observer: it goes in the \"observers\" section, not in a network");
        }
    }
    ResolvePorts(blocks, ReadEdges(network, networkPath, index, blocks, errors), networkPath, errors);

    BlockNetworkSpec spec;
    spec.blocks = TopologicalOrder(std::move(blocks), networkPath, errors);
    InferShapes(spec.blocks, networkPath, errors);
    RequireGradientPaths(spec.blocks, networkPath, errors);
    RequireConsumed(spec.blocks, readOutside, networkPath, errors);
    ResolveOutputs(spec, actor, task, networkPath, errors);
    for (BlockSpec const& block : spec.blocks)
    {
        if (block.type == BlockType::Input)
        {
            spec.inputs.push_back(block.name);
        }
    }
    return spec;
}

/** @brief The actor block named in "team_critic": "features", when the section names one. */
std::set<std::string> TeamCriticFeatures(Json const& root)
{
    if (!root.contains("team_critic") || !root.at("team_critic").is_object())
    {
        return {};
    }
    Json const& section = root.at("team_critic");
    if (!section.contains("features") || !section.at("features").is_string())
    {
        return {};
    }
    return {section.at("features").get<std::string>()};
}

/** @brief Index of the critic group "critic.<group>" read by part `key` of "team_critic". */
std::size_t ReadTeamCriticGroup(Json const& section, char const* key, std::size_t size, TaskSchema const& task,
    std::vector<bool>& used, ErrorReporter const& errors, bool allowPadding = false)
{
    std::string const path = ChildPath("team_critic", key);
    if (!section.contains(key) || !section.at(key).is_string())
    {
        errors.Fail(path, "missing, e.g. \"critic.<group>\"");
    }
    std::string const from = section.at(key).get<std::string>();
    std::string const prefix = "critic.";
    std::string const groupName = from.rfind(prefix, 0) == 0 ? from.substr(prefix.size()) : std::string();
    std::vector<ObservationGroup> const& groups = task.critic->groups;
    auto const group = std::find_if(groups.begin(), groups.end(),
        [&groupName](ObservationGroup const& candidate) { return candidate.name == groupName; });
    if (group == groups.end())
    {
        errors.Fail(path, "expected \"critic.<group>\", a group of task.json \"critic\"");
    }
    std::size_t const index = static_cast<std::size_t>(group - groups.begin());
    if (group->isSet || group->Size() < size || (!allowPadding && group->Size() != size))
    {
        errors.Fail(path, "critic." + groupName + " must be a vector of "
            + (allowPadding ? "at least " : "") + std::to_string(size) + " values");
    }
    if (used[index])
    {
        errors.Fail(path, "critic." + groupName + " is already filled by another part");
    }
    used[index] = true;
    return index;
}

/** @brief Reads the "team_critic" section against the task and the actor network. */
TeamCriticSpec ReadTeamCritic(Json const& section, TaskSchema const& task, BlockNetworkSpec const& actor,
    ErrorReporter const& errors)
{
    RequireObject(section, "team_critic", errors);
    RequireKnownKeys(section, {"features", "pad_features", "own", "mean", "max", "living_fraction", "team_fraction"}, "team_critic",
        errors);
    if (task.maximumAgents < 2 || !task.critic)
    {
        errors.Fail("team_critic", "only for a team task: agents.max above 1 and a \"critic\" input in task.json");
    }
    TeamCriticSpec spec;
    if (section.contains("pad_features"))
    {
        if (!section.at("pad_features").is_boolean())
            errors.Fail("team_critic.pad_features", "expected true or false");
        spec.padFeatures = section.at("pad_features").get<bool>();
    }
    if (!section.contains("features") || !section.at("features").is_string())
    {
        errors.Fail("team_critic.features", "missing: the actor block that describes one agent");
    }
    spec.features = section.at("features").get<std::string>();
    auto const block = std::find_if(actor.blocks.begin(), actor.blocks.end(),
        [&spec](BlockSpec const& candidate) { return candidate.name == spec.features; });
    if (block == actor.blocks.end() || IsHead(block->type) || block->output.kind != graph::SignalKind::Vector)
    {
        errors.Fail("team_critic.features", "'" + spec.features + "' must be a vector block of the actor (not a head)");
    }
    spec.width = block->output.width;
    std::vector<bool> used(task.critic->groups.size(), false);
    spec.own = ReadTeamCriticGroup(section, "own", spec.width, task, used, errors, spec.padFeatures);
    spec.mean = ReadTeamCriticGroup(section, "mean", spec.width, task, used, errors, spec.padFeatures);
    spec.max = ReadTeamCriticGroup(section, "max", spec.width, task, used, errors, spec.padFeatures);
    spec.livingFraction = ReadTeamCriticGroup(section, "living_fraction", 1, task, used, errors);
    spec.teamFraction = ReadTeamCriticGroup(section, "team_fraction", 1, task, used, errors);
    if (std::find(used.begin(), used.end(), false) != used.end())
    {
        errors.Fail("team_critic", "every group of task.json \"critic\" must be filled by one part");
    }
    return spec;
}

/** @brief Canonical text of the team critic, for the graph hash. */
std::string Canonical(TeamCriticSpec const& spec)
{
    return "team_critic " + spec.features + " " + std::to_string(spec.width) + " " + std::to_string(spec.own) + " "
        + std::to_string(spec.mean) + " " + std::to_string(spec.max) + " " + std::to_string(spec.livingFraction)
        + " " + std::to_string(spec.teamFraction) + (spec.padFeatures ? " pad_features\n" : "\n");
}

/** @brief Per-agent shape of a group: {features}, or {capacity, features} for a set. */
std::vector<std::size_t> GroupShape(ObservationGroup const& group)
{
    if (group.isSet)
    {
        return {group.capacity, group.features};
    }
    return {group.features};
}

/** @brief The group named `name` of a layout, or null. */
ObservationGroup const* FindGroup(InputLayout const& layout, std::string const& name) noexcept
{
    for (ObservationGroup const& group : layout.groups)
    {
        if (group.name == name)
        {
            return &group;
        }
    }
    return nullptr;
}

/** @brief Lists the group names of a layout, for error messages. */
std::string GroupNames(InputLayout const& layout)
{
    std::string names;
    for (ObservationGroup const& group : layout.groups)
    {
        names += (names.empty() ? "" : ", ") + group.name;
    }
    return names.empty() ? std::string("none") : names;
}

/** @brief The action group named `name`, or null. */
ActionGroup const* FindActionGroup(TaskSchema const& task, std::string const& name) noexcept
{
    for (ActionGroup const& group : task.actions)
    {
        if (group.name == name)
        {
            return &group;
        }
    }
    return nullptr;
}

/** @brief A shape as text, e.g. "[10, 7]". */
std::string DescribeShape(std::vector<std::size_t> const& shape)
{
    std::string text = "[";
    for (std::size_t index = 0; index < shape.size(); ++index)
    {
        text += (index == 0 ? "" : ", ") + std::to_string(shape[index]);
    }
    return text + "]";
}

/** @brief Index of the input named `port` of an observer, or its number of inputs. */
std::size_t FindObserverPort(ObserverSpec const& observer, std::string const& port) noexcept
{
    for (std::size_t index = 0; index < observer.inputs.size(); ++index)
    {
        if (observer.inputs[index].port == port)
        {
            return index;
        }
    }
    return observer.inputs.size();
}

/**
 * @brief Resolves the signal of an observer input node ("from"): "observation.<group>",
 *        "world.<group>", "action", "action.<group>" or "reward".
 */
ObservedInput ReadObservedSource(std::string const& source, std::string const& edgePath, TaskSchema const& task,
    ErrorReporter const& errors)
{
    std::size_t const dot = source.find('.');
    std::string const kind = source.substr(0, dot);
    std::string const group = dot == std::string::npos ? std::string() : source.substr(dot + 1);
    ObservedInput input;
    input.source = source;
    if (kind == "observation" || kind == "world")
    {
        bool const world = kind == "world";
        if (world && !task.world)
        {
            errors.Fail(edgePath, "'" + source + "': task.json has no \"world\" section");
        }
        InputLayout const& layout = world ? *task.world : task.observation;
        ObservationGroup const* const found = FindGroup(layout, group);
        if (found == nullptr)
        {
            errors.Fail(edgePath, "'" + source + "': no " + kind + " group '" + group + "' (groups: "
                    + GroupNames(layout) + ")");
        }
        input.signal = world ? ObservedSignal::World : ObservedSignal::Observation;
        input.offset = found->offset;
        input.size = found->Size();
        input.shape = GroupShape(*found);
        return input;
    }
    if (kind == "action")
    {
        input.signal = ObservedSignal::Action;
        input.size = task.actionSize;
        if (!group.empty())
        {
            ActionGroup const* const found = FindActionGroup(task, group);
            if (found == nullptr)
            {
                errors.Fail(edgePath, "'" + source + "': no action group '" + group + "'");
            }
            input.offset = found->offset;
            input.size = found->Size();
        }
        input.shape = {input.size};
        return input;
    }
    if (kind == "reward" && group.empty())
    {
        input.signal = ObservedSignal::Reward;
        input.size = task.rewards.size();
        input.shape = {input.size};
        return input;
    }
    errors.Fail(edgePath, "unknown source '" + source
            + "' (observation.<group>, world.<group>, action, action.<group> or reward)");
    return input;
}

/**
 * @brief Checks that a source shape fits an observer port and fixes its symbols.
 * @param symbols Symbols already fixed by other ports of the same observer.
 */
void BindObservedShape(component::PortSpec const& port, ObservedInput const& input, std::string const& edgePath,
    std::map<std::string, std::size_t>& symbols, ErrorReporter const& errors)
{
    if (port.shape.size() != input.shape.size())
    {
        errors.Fail(edgePath, "port '" + port.name + "' expects rank " + std::to_string(port.shape.size())
                + ", '" + input.source + "' gives " + DescribeShape(input.shape));
    }
    for (std::size_t axis = 0; axis < port.shape.size(); ++axis)
    {
        component::DimensionSpec const& dimension = port.shape[axis];
        std::size_t const size = input.shape[axis];
        if (dimension.symbol.empty())
        {
            if (dimension.size != size)
            {
                errors.Fail(edgePath, "port '" + port.name + "' expects " + std::to_string(dimension.size)
                        + " on axis " + std::to_string(axis) + ", '" + input.source + "' gives " + DescribeShape(input.shape));
            }
            continue;
        }
        auto const bound = symbols.find(dimension.symbol);
        if (bound != symbols.end() && bound->second != size)
        {
            errors.Fail(edgePath, "'" + dimension.symbol + "' is " + std::to_string(bound->second)
                    + " on another port but " + std::to_string(size) + " on '" + port.name + "'");
        }
        symbols[dimension.symbol] = size;
    }
}

/**
 * @brief Reads the "observers" section: input nodes reading the signals of a step, and
 *        observer components wired to them, in the same form as a network.
 */
std::vector<ObserverSpec> ReadObservers(Json const& section, TaskSchema const& task, Libraries const& libraries,
    ErrorReporter const& errors)
{
    std::string const sectionPath = "observers";
    RequireObject(section, sectionPath, errors);
    RequireKnownKeys(section, {"nodes", "edges"}, sectionPath, errors);
    std::string const nodesPath = ChildPath(sectionPath, "nodes");
    if (!section.contains("nodes"))
    {
        errors.Fail(nodesPath, "missing (an object of named inputs and observers)");
    }
    RequireObject(section.at("nodes"), nodesPath, errors);

    std::vector<ObserverSpec> observers;
    std::map<std::string, std::size_t> index;
    std::map<std::string, ObservedInput> sources;
    std::set<std::string> usedSources;
    for (auto const& [name, node] : section.at("nodes").items())
    {
        std::string const path = ChildPath(nodesPath, name);
        if (!IsIdentifier(name))
        {
            errors.Fail(path, "node names are identifiers (letters, digits, '_')");
        }
        RequireObject(node, path, errors);
        std::string const typePath = ChildPath(path, "type");
        if (!node.contains("type") || !node.at("type").is_string())
        {
            errors.Fail(typePath, "missing: \"input\" or \"component:<name>@<version>\"");
        }
        if (node.at("type") == "input")
        {
            RequireKnownKeys(node, {"type", "from"}, path, errors);
            std::string const fromPath = ChildPath(path, "from");
            if (!node.contains("from") || !node.at("from").is_string())
            {
                errors.Fail(fromPath, "missing, e.g. \"world.pose\"");
            }
            sources[name] = ReadObservedSource(node.at("from").get<std::string>(), fromPath, task, errors);
            continue;
        }
        if (node.at("type").get<std::string>().rfind(ComponentPrefix, 0) != 0)
        {
            errors.Fail(typePath, "an observers node is an \"input\" or an observer component "
                                  "(\"component:<name>@<version>\")");
        }
        BlockSpec const block = ReadComponent(name, node, path, libraries, errors);
        if (!block.component.manifest->capabilities.observer)
        {
            errors.Fail(ChildPath(path, "type"), "component " + block.component.Id()
                    + " is not an observer (its manifest does not declare \"observer\")");
        }
        ObserverSpec observer;
        observer.name = name;
        observer.component = block.component;
        observer.config = block.componentConfig;
        for (component::PortSpec const& port : block.component.manifest->ports)
        {
            ObservedInput input;
            input.port = port.name;
            observer.inputs.push_back(input);
        }
        index[name] = observers.size();
        observers.push_back(std::move(observer));
    }

    std::string const edgesPath = ChildPath(sectionPath, "edges");
    if (!section.contains("edges") || !section.at("edges").is_array())
    {
        errors.Fail(edgesPath, "missing, e.g. [[\"pose\", \"view.pose\"], ...]");
    }
    std::vector<std::map<std::string, std::size_t>> symbols(observers.size());
    std::vector<std::vector<bool>> wired(observers.size());
    for (std::size_t observer = 0; observer < observers.size(); ++observer)
    {
        wired[observer].assign(observers[observer].inputs.size(), false);
    }
    std::size_t position = 0;
    for (Json const& edge : section.at("edges"))
    {
        std::string const edgePath = edgesPath + "[" + std::to_string(position++) + "]";
        if (!edge.is_array() || edge.size() != 2 || !edge[0].is_string() || !edge[1].is_string())
        {
            errors.Fail(edgePath, "expected [\"source\", \"observer.port\"]");
        }
        std::string const target = edge[1].get<std::string>();
        std::size_t const dot = target.find('.');
        std::string const targetName = target.substr(0, dot);
        if (index.count(targetName) == 0)
        {
            errors.Fail(edgePath, "unknown observer '" + targetName + "'");
        }
        ObserverSpec& observer = observers[index.at(targetName)];
        std::string port = dot == std::string::npos ? std::string() : target.substr(dot + 1);
        if (port.empty() && observer.inputs.size() == 1)
        {
            port = observer.inputs.front().port;
        }
        std::size_t const portIndex = FindObserverPort(observer, port);
        if (portIndex == observer.inputs.size())
        {
            errors.Fail(edgePath, "observer '" + targetName + "' has no port '" + port + "'");
        }
        if (wired[index.at(targetName)][portIndex])
        {
            errors.Fail(edgePath, "port '" + target + "' is already wired");
        }
        std::string const source = edge[0].get<std::string>();
        if (sources.count(source) == 0)
        {
            errors.Fail(edgePath, "unknown source node '" + source + "' (an input node of \"observers\")");
        }
        usedSources.insert(source);
        ObservedInput input = sources.at(source);
        input.port = port;
        BindObservedShape(observer.component.manifest->ports[portIndex], input, edgePath,
            symbols[index.at(targetName)], errors);
        observer.inputs[portIndex] = std::move(input);
        wired[index.at(targetName)][portIndex] = true;
    }
    for (std::size_t observer = 0; observer < observers.size(); ++observer)
    {
        for (std::size_t port = 0; port < observers[observer].inputs.size(); ++port)
        {
            if (!wired[observer][port])
            {
                errors.Fail(ChildPath(nodesPath, observers[observer].name),
                    "port '" + observers[observer].inputs[port].port + "' is not wired");
            }
        }
        Json shapes = Json::object();
        for (auto const& [symbol, size] : symbols[observer])
        {
            shapes[symbol] = size;
        }
        observers[observer].shapes = shapes.dump();
    }
    for (auto const& [name, input] : sources)
    {
        if (usedSources.count(name) == 0)
        {
            errors.Fail(ChildPath(nodesPath, name), "input '" + name + "' (" + input.source + ") feeds no observer");
        }
    }
    return observers;
}

} // namespace

bool HasNumberedPorts(BlockType type) noexcept
{
    return type == BlockType::Concat || type == BlockType::ConcatSets;
}

std::vector<std::string> BlockInputPorts(BlockType type, std::size_t numberedPorts)
{
    return PortsOf(type, numberedPorts);
}

std::vector<std::string> BlockInputPorts(BlockType type, std::size_t numberedPorts, bool separateKeys)
{
    return PortsOf(type, numberedPorts, separateKeys);
}

std::optional<BlockType> BlockTypeFromName(std::string_view name) noexcept
{
    for (BlockTypeEntry const& entry : BlockTypes)
    {
        if (name == entry.name)
        {
            return entry.type;
        }
    }
    return std::nullopt;
}

char const* BlockTypeName(BlockType type) noexcept
{
    if (type == BlockType::Component)
    {
        return "component";
    }
    for (BlockTypeEntry const& entry : BlockTypes)
    {
        if (entry.type == type)
        {
            return entry.name;
        }
    }
    return "unknown";
}

namespace {

/** @brief Everything about a block that fixes what its parameters mean, as text. */
std::string Canonical(BlockSpec const& block)
{
    std::string text = block.name + "|" + BlockTypeName(block.type) + "|" + std::to_string(block.width) + "|"
        + std::to_string(block.layers) + "|" + std::to_string(static_cast<int>(block.activation)) + "|"
        + std::to_string(static_cast<int>(block.pool)) + "|" + graph::Describe(block.output) + "|";
    for (std::size_t const index : block.indices)
    {
        text += std::to_string(index) + ",";
    }
    text += "|";
    for (std::string const& source : block.sources)
    {
        text += source + ",";
    }
    if (block.type == BlockType::Input)
    {
        text += "|" + std::to_string(block.source.critic) + ":" + std::to_string(block.source.group);
    }
    text += "|" + std::to_string(block.componentHash);
    // Written only when used, so the networks without them keep their identity.
    if (block.keys)
    {
        text += "|keys";
    }
    if (block.logStd.stateDependent)
    {
        text += "|log_std:" + std::to_string(block.logStd.minimum) + ":" + std::to_string(block.logStd.maximum);
    }
    if (!block.share.empty())
    {
        text += "|share:" + block.share;
    }
    if (block.maskFill)
    {
        text += "|mask_fill:" + std::to_string(*block.maskFill);
    }
    if (block.encodeAbsent)
    {
        text += "|encode_absent";
    }
    return text + "\n";
}

std::uint64_t Fnv(std::uint64_t hash, std::string const& text) noexcept
{
    for (char const character : text)
    {
        hash = (hash ^ static_cast<unsigned char>(character)) * 1099511628211ULL;
    }
    return hash;
}

std::vector<std::string> BlockNames(BlockNetworkSpec const& network)
{
    std::vector<std::string> names;
    for (BlockSpec const& block : network.blocks)
    {
        names.push_back(block.name + ":" + BlockTypeName(block.type));
    }
    return names;
}

} // namespace

artifacts::ArtifactIdentity ArtifactIdentityOf(TaskSchema const& task, BlockGraphSchema const& graphs,
    std::string backend)
{
    artifacts::ArtifactIdentity identity;
    identity.task = task.name;
    identity.taskHash = ShapeHash(task);
    identity.semanticHash = SemanticHash(task);
    identity.contract = task.contract;
    identity.backend = std::move(backend);
    identity.observationSize = task.observation.size;
    identity.actionSize = task.actionSize;
    identity.actorBlocks = BlockNames(graphs.actor);
    identity.criticBlocks = BlockNames(graphs.critic);
    std::uint64_t hash = 14695981039346656037ULL;
    for (BlockNetworkSpec const* network : {&graphs.actor, &graphs.critic})
    {
        hash = Fnv(hash, network == &graphs.actor ? "actor\n" : "critic\n");
        for (BlockSpec const& block : network->blocks)
        {
            hash = Fnv(hash, Canonical(block));
            if (block.type == BlockType::Component)
            {
                identity.components.emplace_back(block.component.Id(), block.componentHash);
            }
        }
    }
    if (graphs.teamCritic)
    {
        hash = Fnv(hash, Canonical(*graphs.teamCritic));
    }
    identity.graphHash = hash;
    return identity;
}

std::vector<std::string> ComponentInputPorts(component::Manifest const& manifest)
{
    std::vector<std::string> ports;
    for (component::PortSpec const& port : manifest.ports)
    {
        if (!port.output && port.dtype == component::Dtype::F32)
        {
            ports.push_back(port.name);
        }
    }
    return ports;
}

BlockGraphSchema ParseBlockGraph(std::string_view text, std::string_view sourceName, TaskSchema const& task)
{
    return ParseBlockGraph(text, sourceName, task, std::filesystem::current_path());
}

BlockGraphSchema ParseBlockGraph(std::string_view text, std::string_view sourceName, TaskSchema const& task,
    std::filesystem::path const& directory)
{
    ErrorReporter const errors(sourceName);
    Json const root = detail::ParseRootObject(text, errors);
    RequireKnownKeys(root, {"schema", "components", "actor", "critic", "observers", "team_critic"}, "", errors);
    if (!root.contains("schema") || root.at("schema") != BlockGraphSchemaVersion)
    {
        errors.Fail("schema", std::string("expected \"") + BlockGraphSchemaVersion + "\"");
    }
    Libraries libraries;
    if (root.contains("components"))
    {
        Json const& list = root.at("components");
        if (!list.is_array())
        {
            errors.Fail("components", "expected a list of component libraries, e.g. [\"trackfilter\"]");
        }
        for (std::size_t index = 0; index < list.size(); ++index)
        {
            std::string const path = "components[" + std::to_string(index) + "]";
            if (!list[index].is_string())
            {
                errors.Fail(path, "expected a library name or path");
            }
            try
            {
                libraries.push_back(component::Library::Open(list[index].get<std::string>(), directory));
            }
            catch (std::runtime_error const& error)
            {
                errors.Fail(path, error.what());
            }
        }
    }
    BlockGraphSchema schema;
    schema.actor = ReadNetwork(root, "actor", true, task, libraries, TeamCriticFeatures(root), errors);
    schema.critic = ReadNetwork(root, "critic", false, task, libraries, {}, errors);
    CheckSharedBlocks(schema, errors);
    if (root.contains("team_critic"))
    {
        schema.teamCritic = ReadTeamCritic(root.at("team_critic"), task, schema.actor, errors);
    }
    if (root.contains("observers"))
    {
        schema.observers = ReadObservers(root.at("observers"), task, libraries, errors);
    }
    return schema;
}

BlockGraphSchema LoadBlockGraph(std::filesystem::path const& path, TaskSchema const& task)
{
    std::filesystem::path const directory = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    return ParseBlockGraph(detail::ReadTextFile(path), path.filename().string(), task, directory);
}

namespace {

std::vector<std::size_t> ActionAxes(TaskSchema const& task, ActionHead head)
{
    std::vector<std::size_t> axes;
    for (ActionGroup const& group : task.actions)
    {
        if (group.head != head)
        {
            continue;
        }
        for (std::size_t component = 0; component < group.Size(); ++component)
        {
            axes.push_back(group.offset + component);
        }
    }
    return axes;
}

} // namespace

namespace {

/**
 * @brief Slices of the inputs of a network, all read from one layout of the task.
 * @param fromCritic True to read the critic layout, false the observation.
 */
std::vector<graph::InputSlice> SlicesFrom(BlockNetworkSpec const& network, std::string_view networkName,
    TaskSchema const& task, bool fromCritic)
{
    std::vector<graph::InputSlice> slices;
    for (std::string const& name : network.inputs)
    {
        auto const block = std::find_if(network.blocks.begin(), network.blocks.end(),
            [&name](BlockSpec const& candidate) { return candidate.name == name; });
        if (block->source.critic != fromCritic)
        {
            throw SchemaError(std::string(networkName) + " input '" + name
                + (fromCritic ? "' reads an observation group: the hosted critic reads the critic layout of task.json"
                              : "' reads a critic group, which is not part of the observation "
                                "(a centralized critic needs the hosted mode)"));
        }
        InputLayout const& layout = fromCritic ? *task.critic : task.observation;
        ObservationGroup const& group = layout.groups[block->source.group];
        graph::InputSlice slice;
        slice.offset = group.offset;
        slice.shape = block->output;
        slice.maskFeature = group.maskFeature;
        slices.push_back(slice);
    }
    return slices;
}

} // namespace

std::vector<graph::InputSlice> MakeInputSlices(BlockNetworkSpec const& network, std::string_view networkName,
    TaskSchema const& task)
{
    return SlicesFrom(network, networkName, task, false);
}

std::vector<graph::InputSlice> MakeCriticInputSlices(BlockNetworkSpec const& network, TaskSchema const& task)
{
    if (!task.critic)
    {
        throw SchemaError("task.json", "critic", "missing: the hosted critic reads the critic layout");
    }
    return SlicesFrom(network, "critic", task, true);
}

graph::PolicyLayout MakePolicyLayout(BlockGraphSchema const& graphs, TaskSchema const& task)
{
    graph::PolicyLayout layout;
    layout.actorInputs = MakeInputSlices(graphs.actor, "actor", task);
    layout.criticInputs = MakeInputSlices(graphs.critic, "critic", task);
    layout.gaussianAxes = ActionAxes(task, ActionHead::Gaussian);
    layout.bernoulliAxes = ActionAxes(task, ActionHead::Bernoulli);
    for (ActionGroup const& group : task.actions)
    {
        if (group.onZero)
        {
            for (std::size_t axis = 0; axis < group.Size(); ++axis)
            {
                layout.oneOnTie.push_back(group.offset + axis);
            }
        }
    }
    for (BlockSpec const& block : graphs.critic.blocks)
    {
        if (!block.share.empty())
        {
            layout.sharedBlocks.push_back({block.name, block.share});
        }
    }
    for (ActionActivity const& activity : task.activities)
    {
        graph::ComponentActivity rule;
        rule.axis = activity.axis;
        rule.feature = activity.feature;
        rule.threshold = activity.threshold;
        rule.above = activity.test == ActivityTest::Above;
        layout.activities.push_back(rule);
    }
    return layout;
}

graph::Graph BuildGraph(BlockNetworkSpec const& network)
{
    graph::Graph result;
    std::map<std::string, graph::NodeId> ids;
    std::map<std::string, graph::Shape> shapes;
    for (BlockSpec const& block : network.blocks)
    {
        if (block.type == BlockType::Input)
        {
            ids[block.name] = result.AddInput(block.name, block.output);
            shapes[block.name] = block.output;
            continue;
        }
        std::vector<graph::NodeId> sources;
        std::vector<graph::Shape> inputs;
        for (std::string const& source : block.sources)
        {
            sources.push_back(ids.at(source));
            inputs.push_back(shapes.at(source));
        }
        std::unique_ptr<graph::Node> node;
        try
        {
            node = MakeNode(block, inputs, block.sources);
        }
        catch (PortError const& error)
        {
            throw std::invalid_argument("PPOKit graph: node '" + block.name + "': " + error.message);
        }
        shapes[block.name] = node->OutputShape();
        ids[block.name] = result.AddNode(block.name, std::move(node), sources);
    }
    for (std::string const& output : network.outputs)
    {
        result.AddOutput(ids.at(output));
    }
    return result;
}

} // namespace ppokit::schema
