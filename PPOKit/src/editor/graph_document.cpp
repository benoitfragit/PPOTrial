/**
 * @file PPOKit/src/editor/graph_document.cpp
 * @brief Editable block graph: lenient reading, operations with undo, live validation.
 */
#include "ppokit/editor/graph_document.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace ppokit::editor {
namespace {

using Json = nlohmann::ordered_json;

constexpr char const* GraphSchemaVersion = "ppokit.graph/2";
constexpr char const* LayoutSchemaVersion = "ppokit.graph-layout/1";
/** @brief Undo steps kept; older ones are dropped. */
constexpr std::size_t UndoDepth = 200;
/** @brief Spacing of automatically placed nodes, in canvas units. */
constexpr float ColumnSpacing = 240.F;
/** @brief Height of a node without ports (title, type and output), at 100 % zoom. */
constexpr float NodeBaseHeight = 90.F;
/** @brief Height of one input port row, at 100 % zoom. */
constexpr float PortHeight = 28.F;
/** @brief Free space between two nodes of one column. */
constexpr float RowGap = 36.F;

bool IsIdentifier(std::string const& text) noexcept
{
    if (text.empty() || std::isdigit(static_cast<unsigned char>(text.front())) != 0)
    {
        return false;
    }
    return std::all_of(text.begin(), text.end(),
        [](char const character) { return std::isalnum(static_cast<unsigned char>(character)) != 0 || character == '_'; });
}

bool HasNode(NetworkDocument const& network, std::string const& name)
{
    return std::any_of(network.nodes.begin(), network.nodes.end(),
        [&name](EditorNode const& node) { return node.name == name; });
}

/** @brief First unused name "<base>_<n>", n counting from 1. */
std::string FreeName(NetworkDocument const& network, std::string const& base)
{
    std::size_t index = 1;
    while (HasNode(network, base + "_" + std::to_string(index)))
    {
        ++index;
    }
    return base + "_" + std::to_string(index);
}

std::vector<BlockInfo> MakeCatalog()
{
    using schema::BlockType;
    return {
        {BlockType::Input, "Input", "Reads one group of task.json: a vector, or a masked set of entities.",
            Json{{"from", ""}}},
        {BlockType::Dense, "Dense", "Fully connected layer: width outputs, then the activation.",
            Json{{"width", 64}, {"activation", "tanh"}}},
        {BlockType::Mlp, "MLP", "Stack of dense layers of the same width and activation.",
            Json{{"layers", 2}, {"width", 64}, {"activation", "tanh"}}},
        {BlockType::Concat, "Concat", "Joins vectors end to end, in port order in1, in2...", Json::object()},
        {BlockType::Select, "Select", "Keeps some features of a vector, by index (e.g. one scalar of a group).",
            Json{{"indices", Json::array({0})}}},
        {BlockType::ConcatSets, "Concat sets",
            "Stacks sets of the same width into one set (e.g. friendly tracks then the objective). Masks follow.",
            Json::object()},
        {BlockType::EntityEncoder, "Entity encoder",
            "Same dense layer applied to every present entity of a set; absent entities stay zero.",
            Json{{"width", 8}, {"activation", "tanh"}}},
        {BlockType::MaskedAttention, "Masked attention",
            "One query attends over the present entities (score q.e/sqrt(d), softmax). Zero for an empty set. "
            "With keys, the scores use a separate keys set and the output averages the set.",
            Json::object()},
        {BlockType::MaskedPool, "Masked pool",
            "Mean or maximum over the present entities, so the output ignores their order and number.",
            Json{{"mode", "mean"}}},
        {BlockType::GaussianHead, "Gaussian head",
            "Means of the Gaussian actions of task.json; the log-std is a learned parameter, or a clamped "
            "dense layer of the same input (state).",
            Json::object()},
        {BlockType::BernoulliHead, "Bernoulli head", "Logits of the Bernoulli (on/off) actions of task.json.",
            Json::object()},
        {BlockType::Value, "Value", "State value estimated by the critic.", Json::object()},
    };
}

Json ReadJson(std::string_view text, char const* source)
{
    try
    {
        return Json::parse(text.begin(), text.end());
    }
    catch (nlohmann::json::exception const& error)
    {
        throw schema::SchemaError(source, "", std::string("invalid JSON: ") + error.what());
    }
}

/** @brief Reads a network as drawn; malformed entries are kept as far as possible. */
NetworkDocument ReadNetwork(Json const& root, char const* name)
{
    NetworkDocument network;
    if (!root.contains(name) || !root.at(name).is_object())
    {
        return network;
    }
    Json const& source = root.at(name);
    if (source.contains("nodes") && source.at("nodes").is_object())
    {
        for (auto const& [nodeName, node] : source.at("nodes").items())
        {
            EditorNode entry;
            entry.name = nodeName;
            if (node.is_object())
            {
                for (auto const& [key, value] : node.items())
                {
                    if (key == "type" && value.is_string())
                    {
                        entry.type = value.get<std::string>();
                    }
                    else if (key != "type")
                    {
                        entry.parameters[key] = value;
                    }
                }
            }
            network.nodes.push_back(std::move(entry));
        }
    }
    if (source.contains("edges") && source.at("edges").is_array())
    {
        for (Json const& edge : source.at("edges"))
        {
            if (!edge.is_array() || edge.size() != 2 || !edge[0].is_string() || !edge[1].is_string())
            {
                continue;
            }
            std::string const target = edge[1].get<std::string>();
            std::size_t const dot = target.find('.');
            network.edges.push_back({edge[0].get<std::string>(), target.substr(0, dot),
                dot == std::string::npos ? std::string() : target.substr(dot + 1)});
        }
    }
    return network;
}

Json WriteNetwork(NetworkDocument const& network)
{
    Json nodes = Json::object();
    for (EditorNode const& node : network.nodes)
    {
        Json entry = Json::object();
        entry["type"] = node.type;
        for (auto const& [key, value] : node.parameters.items())
        {
            entry[key] = value;
        }
        nodes[node.name] = std::move(entry);
    }
    Json edges = Json::array();
    for (EditorEdge const& edge : network.edges)
    {
        edges.push_back(Json::array({edge.source, edge.port.empty() ? edge.target : edge.target + "." + edge.port}));
    }
    return Json{{"nodes", std::move(nodes)}, {"edges", std::move(edges)}};
}

/** @brief Splits "actor.nodes.attend.query" or "actor.edges[3]" into its parts. */
void Locate(std::string const& path, Validation& validation)
{
    std::size_t const dot = path.find('.');
    std::string const head = path.substr(0, std::min(dot, path.find('[')));
    for (Network const network : AllNetworks)
    {
        if (head == NetworkName(network))
        {
            validation.network = network;
        }
    }
    if (!validation.network || dot == std::string::npos)
    {
        return;
    }
    std::string const rest = path.substr(dot + 1);
    if (rest.rfind("edges[", 0) == 0)
    {
        validation.edge = static_cast<std::size_t>(std::stoul(rest.substr(6)));
        return;
    }
    if (rest.rfind("nodes.", 0) != 0)
    {
        return;
    }
    std::string const node = rest.substr(6);
    std::size_t const fieldDot = node.find('.');
    validation.node = node.substr(0, fieldDot);
    if (fieldDot != std::string::npos)
    {
        validation.field = node.substr(fieldDot + 1);
        validation.field = validation.field.substr(0, validation.field.find_first_of(".["));
    }
}

std::map<std::string, graph::Shape> Shapes(schema::BlockNetworkSpec const& network)
{
    std::map<std::string, graph::Shape> shapes;
    for (schema::BlockSpec const& block : network.blocks)
    {
        shapes[block.name] = block.output;
    }
    return shapes;
}

/** @brief Longest chain of wires leading to each node (0 for nodes fed by nothing). */
std::map<std::string, std::size_t> Depths(NetworkDocument const& network)
{
    std::map<std::string, std::size_t> depth;
    for (EditorNode const& node : network.nodes)
    {
        depth[node.name] = 0;
    }
    // Bellman-Ford style relaxation, bounded by the node count so that a cycle cannot loop.
    for (std::size_t pass = 0; pass < network.nodes.size(); ++pass)
    {
        bool changed = false;
        for (EditorEdge const& edge : network.edges)
        {
            if (depth.count(edge.source) == 0 || depth.count(edge.target) == 0)
            {
                continue;
            }
            if (depth[edge.target] < depth[edge.source] + 1)
            {
                depth[edge.target] = depth[edge.source] + 1;
                changed = true;
            }
        }
        if (!changed)
        {
            break;
        }
    }
    return depth;
}

/** @brief True for the wire on `target.port`. */
bool IsWireTo(EditorEdge const& edge, std::string const& target, std::string const& port)
{
    return edge.target == target && edge.port == port;
}

/** @brief Removes the wire on `target.port`, if any. */
void DropWire(NetworkDocument& document, std::string const& target, std::string const& port)
{
    auto const end = std::remove_if(document.edges.begin(), document.edges.end(),
        [&](EditorEdge const& edge) { return IsWireTo(edge, target, port); });
    document.edges.erase(end, document.edges.end());
}

} // namespace

char const* NetworkName(Network network) noexcept
{
    switch (network)
    {
    case Network::Actor:
        return "actor";
    case Network::Critic:
        return "critic";
    case Network::Observers:
        return "observers";
    }
    return "actor";
}

std::vector<BlockInfo> const& BlockCatalog()
{
    static std::vector<BlockInfo> const catalog = MakeCatalog();
    return catalog;
}

BlockInfo const* FindBlock(std::string_view typeName)
{
    std::optional<schema::BlockType> const type = schema::BlockTypeFromName(typeName);
    if (!type)
    {
        return nullptr;
    }
    for (BlockInfo const& info : BlockCatalog())
    {
        if (info.type == *type)
        {
            return &info;
        }
    }
    return nullptr;
}

std::vector<std::string> InputChoices(schema::TaskSchema const& task, Network network)
{
    std::vector<std::string> choices;
    for (schema::ObservationGroup const& group : task.observation.groups)
    {
        choices.push_back("observation." + group.name);
    }
    if (network == Network::Critic && task.critic)
    {
        for (schema::ObservationGroup const& group : task.critic->groups)
        {
            choices.push_back("critic." + group.name);
        }
    }
    if (network == Network::Observers)
    {
        if (task.world)
        {
            for (schema::ObservationGroup const& group : task.world->groups)
            {
                choices.push_back("world." + group.name);
            }
        }
        choices.push_back("action");
        for (schema::ActionGroup const& group : task.actions)
        {
            choices.push_back("action." + group.name);
        }
        choices.push_back("reward");
    }
    return choices;
}

GraphDocument GraphDocument::Open(std::string_view graphText, std::string_view layoutText)
{
    Json const root = ReadJson(graphText, "graph.json");
    if (!root.is_object() || !root.contains("schema") || root.at("schema") != GraphSchemaVersion)
    {
        throw schema::SchemaError("graph.json", "schema",
            std::string("the editor opens \"") + GraphSchemaVersion + "\" files");
    }
    GraphDocument document;
    document.actor_ = ReadNetwork(root, "actor");
    document.critic_ = ReadNetwork(root, "critic");
    document.observers_ = ReadNetwork(root, "observers");
    if (root.contains("team_critic"))
    {
        document.teamCritic_ = root.at("team_critic");
    }
    if (root.contains("components") && root.at("components").is_array())
    {
        for (Json const& reference : root.at("components"))
        {
            if (reference.is_string())
            {
                document.components_.push_back(reference.get<std::string>());
            }
        }
    }

    Json const layout = layoutText.empty() ? Json::object() : ReadJson(layoutText, "graph layout");
    for (Network const network : AllNetworks)
    {
        NetworkDocument& target = document.Mutable(network);
        std::vector<bool> positioned(target.nodes.size(), false);
        char const* const name = NetworkName(network);
        for (std::size_t index = 0; index < target.nodes.size(); ++index)
        {
            if (!layout.contains(name) || !layout.at(name).contains(target.nodes[index].name))
            {
                continue;
            }
            Json const& position = layout.at(name).at(target.nodes[index].name);
            if (position.is_array() && position.size() == 2 && position[0].is_number() && position[1].is_number())
            {
                target.nodes[index].position = {position[0].get<float>(), position[1].get<float>()};
                positioned[index] = true;
            }
        }
        document.PlaceUnpositioned(network, positioned);
    }
    return document;
}

std::string GraphDocument::GraphJson() const
{
    Json root = Json::object();
    root["schema"] = GraphSchemaVersion;
    if (!components_.empty())
    {
        root["components"] = components_;
    }
    root["actor"] = WriteNetwork(actor_);
    root["critic"] = WriteNetwork(critic_);
    if (!observers_.nodes.empty())
    {
        root["observers"] = WriteNetwork(observers_);
    }
    if (!teamCritic_.is_null())
    {
        root["team_critic"] = teamCritic_;
    }
    return root.dump(2) + "\n";
}

std::string GraphDocument::LayoutJson() const
{
    Json root = Json::object();
    root["schema"] = LayoutSchemaVersion;
    for (Network const network : AllNetworks)
    {
        Json positions = Json::object();
        for (EditorNode const& node : At(network).nodes)
        {
            positions[node.name] = Json::array({node.position.x, node.position.y});
        }
        root[NetworkName(network)] = std::move(positions);
    }
    return root.dump(2) + "\n";
}

NetworkDocument const& GraphDocument::At(Network network) const noexcept
{
    switch (network)
    {
    case Network::Critic:
        return critic_;
    case Network::Observers:
        return observers_;
    case Network::Actor:
        break;
    }
    return actor_;
}

NetworkDocument& GraphDocument::Mutable(Network network) noexcept
{
    switch (network)
    {
    case Network::Critic:
        return critic_;
    case Network::Observers:
        return observers_;
    case Network::Actor:
        break;
    }
    return actor_;
}

EditorNode& GraphDocument::RequireNode(Network network, std::string const& name)
{
    for (EditorNode& node : Mutable(network).nodes)
    {
        if (node.name == name)
        {
            return node;
        }
    }
    throw std::invalid_argument(std::string("PPOKit editor: no node '") + name + "' in the " + NetworkName(network));
}

std::string GraphDocument::AddNode(Network network, schema::BlockType type, Position position)
{
    Record();
    NetworkDocument& target = Mutable(network);
    std::string const base = schema::BlockTypeName(type);
    std::string const name = FreeName(target, base);
    EditorNode node;
    node.name = name;
    node.type = base;
    BlockInfo const* const info = FindBlock(base);
    node.parameters = info != nullptr ? info->defaults : Json::object();
    node.position = position;
    target.nodes.push_back(std::move(node));
    return name;
}

void GraphDocument::RemoveNode(Network network, std::string const& name)
{
    RequireNode(network, name);
    Record();
    NetworkDocument& target = Mutable(network);
    target.nodes.erase(std::remove_if(target.nodes.begin(), target.nodes.end(),
                           [&name](EditorNode const& node) { return node.name == name; }),
        target.nodes.end());
    target.edges.erase(std::remove_if(target.edges.begin(), target.edges.end(),
                           [&name](EditorEdge const& edge) { return edge.source == name || edge.target == name; }),
        target.edges.end());
}

void GraphDocument::RenameNode(Network network, std::string const& name, std::string const& newName)
{
    RequireNode(network, name);
    if (!IsIdentifier(newName))
    {
        throw std::invalid_argument("PPOKit editor: '" + newName + "' is not an identifier (letters, digits, '_')");
    }
    if (newName != name && HasNode(At(network), newName))
    {
        throw std::invalid_argument("PPOKit editor: a node is already named '" + newName + "'");
    }
    Record();
    NetworkDocument& target = Mutable(network);
    RequireNode(network, name).name = newName;
    for (EditorEdge& edge : target.edges)
    {
        edge.source = edge.source == name ? newName : edge.source;
        edge.target = edge.target == name ? newName : edge.target;
    }
    // The team critic reads an actor block by name.
    bool const teamFeatures = network == Network::Actor && teamCritic_.is_object()
        && teamCritic_.contains("features") && teamCritic_.at("features") == name;
    if (teamFeatures)
    {
        teamCritic_["features"] = newName;
    }
}

void GraphDocument::SetParameter(Network network, std::string const& name, std::string const& key,
    nlohmann::ordered_json const& value)
{
    SetParameters(network, name, Json{{key, value}});
}

void GraphDocument::SetParameters(Network network, std::string const& name, nlohmann::ordered_json const& changes)
{
    EditorNode const& selected = RequireNode(network, name);
    if (!changes.is_object())
    {
        throw std::invalid_argument("PPOKit editor: parameter changes must be an object");
    }
    if (changes.contains("type"))
    {
        throw std::invalid_argument("PPOKit editor: the type of a node is fixed, add another node instead");
    }
    Network ownerNetwork = network;
    std::string ownerName = name;
    if (network == Network::Critic && !changes.contains("share"))
    {
        auto const share = selected.parameters.find("share");
        if (share != selected.parameters.end() && share->is_string())
        {
            ownerNetwork = Network::Actor;
            ownerName = share->get<std::string>();
            RequireNode(ownerNetwork, ownerName);
        }
    }
    Record();
    auto const apply = [&changes](EditorNode& node) {
        for (auto const& [key, value] : changes.items())
        {
            if (value.is_null()) node.parameters.erase(key);
            else node.parameters[key] = value;
        }
    };
    EditorNode& node = RequireNode(ownerNetwork, ownerName);
    apply(node);
    if (ownerNetwork == Network::Actor && !changes.contains("share"))
    {
        for (EditorNode& critic : critic_.nodes)
        {
            auto const share = critic.parameters.find("share");
            if (share != critic.parameters.end() && share->is_string() && share->get<std::string>() == ownerName)
                apply(critic);
        }
    }
    auto const keys = node.parameters.find("keys");
    bool const keysPort = keys != node.parameters.end() && keys->is_boolean() && keys->get<bool>();
    if (node.type == schema::BlockTypeName(schema::BlockType::MaskedAttention) && !keysPort)
    {
        DropWire(Mutable(ownerNetwork), ownerName, "keys");
    }
}

void GraphDocument::Connect(Network network, std::string const& source, std::string const& target,
    std::string const& port)
{
    RequireNode(network, source);
    RequireNode(network, target);
    Record();
    NetworkDocument& document = Mutable(network);
    DropWire(document, target, port);
    document.edges.push_back({source, target, port});
}

void GraphDocument::Disconnect(Network network, std::string const& target, std::string const& port)
{
    NetworkDocument const& current = At(network);
    bool const wired = std::any_of(current.edges.begin(), current.edges.end(),
        [&](EditorEdge const& edge) { return IsWireTo(edge, target, port); });
    if (!wired)
    {
        return;
    }
    Record();
    DropWire(Mutable(network), target, port);
}

std::string GraphDocument::NextNumberedPort(Network network, std::string const& target) const
{
    NetworkDocument const& document = At(network);
    for (std::size_t index = 1;; ++index)
    {
        std::string const port = "in" + std::to_string(index);
        bool const used = std::any_of(document.edges.begin(), document.edges.end(),
            [&](EditorEdge const& edge) { return IsWireTo(edge, target, port); });
        if (!used)
        {
            return port;
        }
    }
}

void GraphDocument::MoveNode(Network network, std::string const& name, Position position)
{
    RequireNode(network, name).position = position;
}

void GraphDocument::BeginMove()
{
    moveStart_ = Current();
}

void GraphDocument::EndMove()
{
    if (!moveStart_)
    {
        return;
    }
    Snapshot start = std::move(*moveStart_);
    moveStart_.reset();
    GraphDocument before;
    before.Restore(start);
    if (before.LayoutJson() != LayoutJson())
    {
        PushUndo(std::move(start));
    }
}

void GraphDocument::Undo()
{
    if (undo_.empty())
    {
        return;
    }
    redo_.push_back(Current());
    Restore(undo_.back());
    undo_.pop_back();
}

void GraphDocument::Redo()
{
    if (redo_.empty())
    {
        return;
    }
    undo_.push_back(Current());
    Restore(redo_.back());
    redo_.pop_back();
}

Validation GraphDocument::Validate(schema::TaskSchema const& task) const
{
    Validation validation;
    try
    {
        std::filesystem::path const directory = directory_.empty() ? std::filesystem::current_path() : directory_;
        schema::BlockGraphSchema const graphs = schema::ParseBlockGraph(GraphJson(), "graph.json", task, directory);
        validation.valid = true;
        validation.actorShapes = Shapes(graphs.actor);
        validation.criticShapes = Shapes(graphs.critic);
    }
    catch (schema::SchemaError const& error)
    {
        validation.message = error.what();
        validation.detail = error.Detail().empty() ? error.what() : error.Detail();
        Locate(error.Path(), validation);
    }
    return validation;
}

void GraphDocument::Record()
{
    PushUndo(Current());
}

void GraphDocument::PushUndo(Snapshot snapshot)
{
    undo_.push_back(std::move(snapshot));
    if (undo_.size() > UndoDepth)
    {
        undo_.erase(undo_.begin());
    }
    redo_.clear();
}

GraphDocument::Snapshot GraphDocument::Current() const
{
    return {actor_, critic_, observers_, components_, teamCritic_};
}

void GraphDocument::Restore(Snapshot const& snapshot)
{
    actor_ = snapshot.actor;
    critic_ = snapshot.critic;
    observers_ = snapshot.observers;
    components_ = snapshot.components;
    teamCritic_ = snapshot.teamCritic;
}

std::string ComponentType(component::ComponentRef const& component)
{
    return "component:" + component.Id();
}

std::vector<ComponentLibrary> GraphDocument::ComponentLibraries() const
{
    std::filesystem::path const directory = directory_.empty() ? std::filesystem::current_path() : directory_;
    std::vector<ComponentLibrary> libraries;
    for (std::string const& reference : components_)
    {
        ComponentLibrary entry;
        entry.reference = reference;
        try
        {
            auto const library = component::Library::Open(reference, directory);
            for (std::size_t index = 0; index < library->Count(); ++index)
            {
                entry.components.push_back(library->At(index));
            }
        }
        catch (std::exception const& error)
        {
            entry.error = error.what();
        }
        libraries.push_back(std::move(entry));
    }
    return libraries;
}

void GraphDocument::AddComponentLibrary(std::string const& reference)
{
    if (reference.empty() || std::find(components_.begin(), components_.end(), reference) != components_.end())
    {
        return;
    }
    Record();
    components_.push_back(reference);
}

void GraphDocument::RemoveComponentLibrary(std::string const& reference)
{
    auto const found = std::find(components_.begin(), components_.end(), reference);
    if (found == components_.end())
    {
        return;
    }
    Record();
    components_.erase(found);
}

std::string GraphDocument::AddComponentNode(Network network, component::ComponentRef const& component,
    Position position)
{
    Record();
    NetworkDocument& target = Mutable(network);
    EditorNode node;
    node.name = FreeName(target, component.manifest->name);
    node.type = ComponentType(component);
    node.parameters = Json::object();
    node.position = position;
    target.nodes.push_back(node);
    return node.name;
}

namespace
{

/**
 * @brief Estimates the drawn height of a node from its wired inputs.
 * @param network Network holding the node.
 * @param name Node name.
 * @return Height in canvas units at 100 % zoom, with one free input row as the editor draws it.
 */
float EstimatedNodeHeight(NetworkDocument const& network, std::string const& name)
{
    std::size_t inputs = 0;
    for (EditorEdge const& edge : network.edges)
    {
        if (edge.target == name)
        {
            ++inputs;
        }
    }
    return NodeBaseHeight + PortHeight * static_cast<float>(inputs + 1);
}

} // namespace

/** @brief Places nodes without a saved position: one column per depth, stacked in file order without overlap. */
void GraphDocument::PlaceUnpositioned(Network network, std::vector<bool> const& positioned)
{
    NetworkDocument& target = Mutable(network);
    std::map<std::string, std::size_t> const depth = Depths(target);
    std::map<std::size_t, float> columnBottom;
    for (std::size_t index = 0; index < target.nodes.size(); ++index)
    {
        if (positioned[index])
        {
            continue;
        }
        EditorNode& node = target.nodes[index];
        std::size_t const column = depth.at(node.name);
        float& bottom = columnBottom[column];
        node.position = {ColumnSpacing * static_cast<float>(column), bottom};
        bottom += EstimatedNodeHeight(target, node.name) + RowGap;
    }
}

} // namespace ppokit::editor
