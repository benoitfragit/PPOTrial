/**
 * @file PPOKit/tests/graph_document_tests.cpp
 * @brief Editor model: drawing a network from scratch until it validates, errors located on
 *        nodes, ports and wires, undo/redo, renaming, layout files, lossless round trips, and
 *        component libraries (palette, nodes, settings, validation beside graph.json).
 */
#include "ppokit/editor/graph_document.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace ppokit;
using editor::GraphDocument;
using editor::Network;
using schema::BlockType;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::string ReadFile(char const* path)
{
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

/** @brief A user draws the corridor actor and critic, fixing each reported error in turn. */
void DrawFromScratch(schema::TaskSchema const& task)
{
    GraphDocument document;
    editor::Validation validation = document.Validate(task);
    Check(!validation.valid && validation.network == Network::Actor && validation.message.find("missing a gaussian_head")
            != std::string::npos,
        "an empty document asks for the actor head first, got: " + validation.message);

    std::string const state = document.AddNode(Network::Actor, BlockType::Input, {0, 0});
    std::string const trunk = document.AddNode(Network::Actor, BlockType::Mlp, {240, 0});
    std::string const policy = document.AddNode(Network::Actor, BlockType::GaussianHead, {480, 0});
    Check(state == "input_1" && trunk == "mlp_1" && policy == "gaussian_head_1", "new nodes get free names");
    Check(document.At(Network::Actor).nodes[1].parameters["width"] == 64, "new nodes get default parameters");

    validation = document.Validate(task);
    Check(validation.node == "input_1" && validation.field == "from",
        "an input without group is reported on its 'from' field, got: " + validation.message);
    auto const choices = editor::InputChoices(task, Network::Actor);
    Check(choices.size() == 1 && choices[0] == "observation.state", "the actor offers the observation groups");
    document.SetParameter(Network::Actor, state, "from", choices[0]);

    validation = document.Validate(task);
    Check(validation.node == "mlp_1" && validation.field == "in", "an unwired port is located on the node and port");
    document.Connect(Network::Actor, state, trunk, "");
    document.Connect(Network::Actor, trunk, policy, "");

    // The critic, with a mistake: the value head reads the input directly but the trunk is unused.
    std::string const criticState = document.AddNode(Network::Critic, BlockType::Input, {0, 0});
    std::string const criticTrunk = document.AddNode(Network::Critic, BlockType::Mlp, {240, 0});
    std::string const value = document.AddNode(Network::Critic, BlockType::Value, {480, 0});
    document.SetParameter(Network::Critic, criticState, "from", "observation.state");
    document.Connect(Network::Critic, criticState, criticTrunk, "");
    document.Connect(Network::Critic, criticState, value, "");
    validation = document.Validate(task);
    Check(validation.network == Network::Critic && validation.node == "mlp_1"
            && validation.detail == "output is not used by any node",
        "an unused node is reported in the critic, got: " + validation.message);
    document.Connect(Network::Critic, criticTrunk, value, "");
    Check(document.At(Network::Critic).edges.size() == 2, "a new wire replaces the one on the same port");

    validation = document.Validate(task);
    Check(validation.valid, "the drawn graph validates, got: " + validation.message);
    Check(validation.actorShapes.at("gaussian_head_1") == graph::Shape::Vector(1)
            && validation.actorShapes.at("mlp_1") == graph::Shape::Vector(64),
        "valid graphs expose the inferred shapes for display");

    // What the editor saves is what training loads.
    auto const graphs = schema::ParseBlockGraph(document.GraphJson(), "graph.json", task);
    Check(graphs.actor.outputs.size() == 1, "the saved graph parses for training");
}

/** @brief Undo and redo walk through edits; a new edit clears the redo history. */
void UndoRedo()
{
    GraphDocument document;
    std::string const dense = document.AddNode(Network::Actor, BlockType::Dense, {0, 0});
    document.SetParameter(Network::Actor, dense, "width", 32);
    document.RenameNode(Network::Actor, dense, "hidden");
    Check(document.At(Network::Actor).nodes[0].name == "hidden", "rename applies");
    document.Undo();
    Check(document.At(Network::Actor).nodes[0].name == "dense_1", "undo reverts the rename");
    document.Undo();
    Check(document.At(Network::Actor).nodes[0].parameters["width"] == 64, "undo reverts the parameter");
    document.Redo();
    Check(document.At(Network::Actor).nodes[0].parameters["width"] == 32, "redo replays the parameter");
    document.AddNode(Network::Actor, BlockType::Dense, {0, 100});
    Check(!document.CanRedo(), "a new edit clears the redo history");

    document.BeginMove();
    document.MoveNode(Network::Actor, "dense_1", {20, 30});
    document.MoveNode(Network::Actor, "dense_1", {50, 60});
    document.EndMove();
    document.Undo();
    Check(document.At(Network::Actor).nodes[0].position.x == 0.F, "one undo puts a dragged node back");
    document.BeginMove();
    document.EndMove();
    Check(document.CanRedo(), "a click without move records nothing");
}

/** @brief Renaming and removing keep the wires consistent; bad names are refused. */
void RenameAndRemove()
{
    GraphDocument document;
    std::string const a = document.AddNode(Network::Actor, BlockType::Input, {0, 0});
    std::string const b = document.AddNode(Network::Actor, BlockType::Concat, {0, 0});
    document.Connect(Network::Actor, a, b, document.NextNumberedPort(Network::Actor, b));
    document.Connect(Network::Actor, a, b, document.NextNumberedPort(Network::Actor, b));
    Check(document.At(Network::Actor).edges[1].port == "in2", "the next numbered port is in2");
    document.RenameNode(Network::Actor, a, "own");
    Check(document.At(Network::Actor).edges[0].source == "own", "rename updates the wires");
    bool refused = false;
    try
    {
        document.RenameNode(Network::Actor, "own", "concat_1");
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "a name already used is refused");
    refused = false;
    try
    {
        document.RenameNode(Network::Actor, "own", "2own");
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "a name that is not an identifier is refused");
    document.Disconnect(Network::Actor, b, "in1");
    Check(document.At(Network::Actor).edges.size() == 1, "disconnect removes the wire on the port");
    document.RemoveNode(Network::Actor, "own");
    Check(document.At(Network::Actor).nodes.size() == 1 && document.At(Network::Actor).edges.empty(),
        "removing a node removes its wires");
}

/** @brief The Blue graph survives open/save unchanged; layout is kept apart and restored. */
void BlueRoundTrip(schema::TaskSchema const& task)
{
    std::string const text = ReadFile(PPOKIT_BLUE_GRAPH_BLOCKS);
    GraphDocument document = GraphDocument::Open(text);
    Check(document.Validate(task).valid, "the Blue graph validates in the editor");
    auto const original = schema::ParseBlockGraph(text, "graph.json", task);
    auto const saved = schema::ParseBlockGraph(document.GraphJson(), "graph.json", task);
    Check(original.actor.blocks.size() == saved.actor.blocks.size() && original.actor.inputs == saved.actor.inputs,
        "saving keeps every block and the input order");
    Check(document.GraphJson().find("position") == std::string::npos, "positions stay out of graph.json");

    // Nodes are laid out left to right along the wires.
    auto const& actor = document.At(Network::Actor);
    auto const positionOf = [&actor](std::string const& name) {
        for (auto const& node : actor.nodes)
        {
            if (node.name == name)
            {
                return node.position;
            }
        }
        return editor::Position{-1, -1};
    };
    Check(positionOf("red").x < positionOf("red_embed").x && positionOf("red_embed").x < positionOf("red_attend").x
            && positionOf("trunk").x < positionOf("policy").x,
        "automatic layout follows the wires");

    document.MoveNode(Network::Actor, "policy", {1234, 56});
    GraphDocument const reopened = GraphDocument::Open(document.GraphJson(), document.LayoutJson());
    bool restored = false;
    for (auto const& node : reopened.At(Network::Actor).nodes)
    {
        restored = restored || (node.name == "policy" && node.position.x == 1234 && node.position.y == 56);
    }
    Check(restored, "the layout file restores node positions");
}

/** @brief A broken file still opens: errors are shown, not thrown, and point at the wire. */
void BrokenFilesOpen(schema::TaskSchema const& task)
{
    GraphDocument const document = GraphDocument::Open(R"({
      "schema": "ppokit.graph/2",
      "actor": { "nodes": { "state": { "type": "input", "from": "observation.state" },
                            "odd": { "type": "transformer" } },
                 "edges": [ ["state", "nowhere"] ] }
    })");
    Check(document.At(Network::Actor).nodes[1].type == "transformer", "an unknown block type is kept");
    editor::Validation const validation = document.Validate(task);
    Check(!validation.valid && validation.node == "odd" && validation.field == "type",
        "the unknown type is located on its node, got: " + validation.message);
    Check(editor::FindBlock("transformer") == nullptr && editor::FindBlock("masked_attention") != nullptr,
        "the palette knows the block types");

    GraphDocument const wiring = GraphDocument::Open(R"({
      "schema": "ppokit.graph/2",
      "actor": { "nodes": { "state": { "type": "input", "from": "observation.state" } },
                 "edges": [ ["state", "nowhere"] ] }
    })");
    editor::Validation const edge = wiring.Validate(task);
    Check(edge.network == Network::Actor && edge.edge == std::size_t{0}, "a bad wire is located by its index");

    bool refused = false;
    try
    {
        GraphDocument::Open(R"({ "schema": "ppokit.graph/1" })");
    }
    catch (schema::SchemaError const& error)
    {
        refused = error.Path() == "schema";
    }
    Check(refused, "another schema version is refused with its path");
}

void Catalog()
{
    bool documented = true;
    for (editor::BlockInfo const& info : editor::BlockCatalog())
    {
        documented = documented && std::string(info.help).size() > 20 && std::string(info.title).size() > 2;
    }
    Check(editor::BlockCatalog().size() == 12 && documented, "every block type has a title and a tooltip");
}

} // namespace

/** @brief A user adds a component library, puts its trainable scale in the corridor actor, and saves. */
void Components(schema::TaskSchema const& task)
{
    std::filesystem::path const plugin(PPOKIT_TEST_COMPONENTS);
    GraphDocument document;
    document.SetDirectory(plugin.parent_path());
    document.AddComponentLibrary("ppokit_test_components");
    auto libraries = document.ComponentLibraries();
    Check(libraries.size() == 1 && libraries[0].error.empty() && libraries[0].components.size() == 7,
        "the library opens and lists its components");
    std::string const state = document.AddNode(Network::Actor, BlockType::Input, {0, 0});
    document.SetParameter(Network::Actor, state, "from", "observation.state");
    std::string const scaled = document.AddComponentNode(Network::Actor, libraries[0].components[0], {200, 0});
    std::string const trunk = document.AddNode(Network::Actor, BlockType::Mlp, {400, 0});
    std::string const policy = document.AddNode(Network::Actor, BlockType::GaussianHead, {600, 0});
    Check(scaled == "scale_1" && document.At(Network::Actor).nodes[1].type == "component:scale@1",
        "a component node is named after its component");
    document.Connect(Network::Actor, state, scaled, "x");
    document.Connect(Network::Actor, scaled, trunk, "");
    document.Connect(Network::Actor, trunk, policy, "");
    std::string const criticState = document.AddNode(Network::Critic, BlockType::Input, {0, 0});
    document.SetParameter(Network::Critic, criticState, "from", "observation.state");
    std::string const value = document.AddNode(Network::Critic, BlockType::Value, {200, 0});
    document.Connect(Network::Critic, criticState, value, "");
    document.SetParameter(Network::Actor, scaled, "config", nlohmann::ordered_json{{"initial", 50}});
    editor::Validation validation = document.Validate(task);
    Check(!validation.valid && validation.node == "scale_1" && validation.message.find("outside") != std::string::npos,
        "a setting out of range is reported on the component node, got: " + validation.message);
    document.SetParameter(Network::Actor, scaled, "config", nlohmann::ordered_json{{"initial", 2}});
    validation = document.Validate(task);
    Check(validation.valid, "the graph with a component validates, got: " + validation.message);

    std::string const saved = document.GraphJson();
    Check(saved.find("\"components\": [\n    \"ppokit_test_components\"") != std::string::npos,
        "the library list is saved in graph.json");
    auto reopened = GraphDocument::Open(saved);
    reopened.SetDirectory(plugin.parent_path());
    Check(reopened.ComponentReferences().size() == 1 && reopened.Validate(task).valid,
        "a reopened graph keeps its libraries and still validates");
    reopened.RemoveComponentLibrary("ppokit_test_components");
    Check(!reopened.Validate(task).valid, "without its library, the component node is reported");
    reopened.Undo();
    Check(reopened.Validate(task).valid, "undo brings the library back");

    GraphDocument broken;
    broken.AddComponentLibrary("no_such_library");
    auto const missing = broken.ComponentLibraries();
    Check(missing.size() == 1 && missing[0].error.find("looked for") != std::string::npos,
        "a library that cannot be opened is listed with the reason");
}

/** @brief A user wires a Visualiseur block in the observers canvas, saves and reopens. */
void Observers(schema::TaskSchema const& task)
{
    std::filesystem::path const plugin(PPOKIT_TEST_COMPONENTS);
    GraphDocument document;
    document.SetDirectory(plugin.parent_path());
    document.AddComponentLibrary("ppokit_test_components");
    std::string const own = document.AddNode(Network::Actor, BlockType::Input, {0, 0});
    document.SetParameter(Network::Actor, own, "from", "observation.own");
    std::string const policy = document.AddNode(Network::Actor, BlockType::GaussianHead, {200, 0});
    document.Connect(Network::Actor, own, policy, "");
    std::string const criticOwn = document.AddNode(Network::Critic, BlockType::Input, {0, 0});
    document.SetParameter(Network::Critic, criticOwn, "from", "observation.own");
    std::string const value = document.AddNode(Network::Critic, BlockType::Value, {200, 0});
    document.Connect(Network::Critic, criticOwn, value, "");

    std::vector<std::string> const choices = editor::InputChoices(task, Network::Observers);
    auto const offers = [&choices](std::string const& choice) {
        return std::find(choices.begin(), choices.end(), choice) != choices.end();
    };
    Check(offers("observation.own") && offers("world.pose") && offers("action") && offers("action.brake")
            && offers("reward") && !offers("critic.state"),
        "the observers read observations, the world, the actions and the rewards");

    component::ComponentRef const checksum = document.ComponentLibraries()[0].components[5];
    Check(checksum.manifest->capabilities.observer, "the test library provides an observer");
    std::string const watcher = document.AddComponentNode(Network::Observers, checksum, {300, 0});
    for (auto const& [port, from] : {std::pair<char const*, char const*>{"pose", "world.pose"},
             {"tracks", "observation.red_tracks"}, {"controls", "action"}, {"rewards", "reward"}})
    {
        std::string const input = document.AddNode(Network::Observers, BlockType::Input, {0, 0});
        document.SetParameter(Network::Observers, input, "from", from);
        document.Connect(Network::Observers, input, watcher, port);
    }
    editor::Validation const validation = document.Validate(task);
    Check(validation.valid, "a wired observer validates, got: " + validation.message);

    std::string const saved = document.GraphJson();
    Check(saved.find("\"observers\"") != std::string::npos, "the observers are saved in graph.json");
    auto reopened = GraphDocument::Open(saved, document.LayoutJson());
    reopened.SetDirectory(plugin.parent_path());
    Check(reopened.At(Network::Observers).nodes.size() == 5 && reopened.Validate(task).valid,
        "a reopened graph keeps its observers and still validates");
    reopened.Disconnect(Network::Observers, watcher, "tracks");
    editor::Validation const unwired = reopened.Validate(task);
    Check(!unwired.valid && unwired.network == Network::Observers && unwired.node == watcher,
        "an unwired observer port is located on the observer, got: " + unwired.message);
    reopened.Undo();
    Check(reopened.Validate(task).valid, "undo rewires it");
    Check(GraphDocument().GraphJson().find("observers") == std::string::npos,
        "a graph without observers keeps its usual form");
}

/**
 * @brief Extended parameters: several keys change as one undo step, and turning off the keys
 *        of a masked attention removes the wire on its keys port.
 */
void ExtendedParameters()
{
    GraphDocument document;
    std::string const set = document.AddNode(Network::Actor, BlockType::Input, {0, 0});
    std::string const attend = document.AddNode(Network::Actor, BlockType::MaskedAttention, {200, 0});
    document.SetParameter(Network::Actor, attend, "keys", true);
    document.Connect(Network::Actor, set, attend, "set");
    document.Connect(Network::Actor, set, attend, "keys");
    Check(document.At(Network::Actor).edges.size() == 2, "the keys port takes a wire");
    document.SetParameter(Network::Actor, attend, "keys", nlohmann::ordered_json());
    Check(document.At(Network::Actor).edges.size() == 1 && document.At(Network::Actor).edges[0].port == "set"
            && !document.At(Network::Actor).nodes[1].parameters.contains("keys"),
        "turning the keys off removes the keys parameter and its wire");
    document.Undo();
    Check(document.At(Network::Actor).edges.size() == 2, "one undo brings back the keys and the wire");

    std::string const head = document.AddNode(Network::Actor, BlockType::GaussianHead, {400, 0});
    document.SetParameters(Network::Actor, head, {{"log_std", "state"}, {"log_std_min", -20.0}, {"log_std_max", 2.0}});
    nlohmann::ordered_json const& parameters = document.At(Network::Actor).nodes[2].parameters;
    Check(parameters.value("log_std", "") == "state" && parameters.value("log_std_max", 0.0) == 2.0,
        "several parameters are set together");
    document.Undo();
    Check(!document.At(Network::Actor).nodes[2].parameters.contains("log_std")
            && !document.At(Network::Actor).nodes[2].parameters.contains("log_std_min"),
        "one undo reverts all of them");
    bool refused = false;
    try
    {
        document.SetParameters(Network::Actor, head, {{"type", "value"}});
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "the type cannot be changed through several parameters either");
}

/** @brief A team's "team_critic" section survives editing and follows a rename of its actor block. */
void TeamCritic()
{
    auto const task = schema::LoadTaskSchema(std::string(PPOKIT_TEAM_GRAPH_PROJECT) + "/task.json");
    GraphDocument document = GraphDocument::Open(ReadFile(PPOKIT_TEAM_GRAPH_PROJECT "/network.json"));
    Check(document.Validate(task).valid, "team critic: the team graph validates in the editor");
    document.RenameNode(Network::Actor, "embed", "agent_features");
    auto const renamed = schema::ParseBlockGraph(document.GraphJson(), "network.json", task);
    Check(renamed.teamCritic && renamed.teamCritic->features == "agent_features",
        "team critic: renaming its actor block renames the reference");
    document.Undo();
    auto const restored = schema::ParseBlockGraph(document.GraphJson(), "network.json", task);
    Check(restored.teamCritic && restored.teamCritic->features == "embed", "team critic: undo restores it");
    document.RenameNode(Network::Critic, "trunk", "embed");
    auto const critic = schema::ParseBlockGraph(document.GraphJson(), "network.json", task);
    Check(critic.teamCritic && critic.teamCritic->features == "embed",
        "team critic: a critic node of the same name is another block");
}

int main()
{
    try
    {
        auto const corridor = schema::LoadTaskSchema(PPOKIT_CORRIDOR_TASK);
        auto const blue = schema::LoadTaskSchema(PPOKIT_BLUE_TASK);
        DrawFromScratch(corridor);
        UndoRedo();
        RenameAndRemove();
        BlueRoundTrip(blue);
        BrokenFilesOpen(corridor);
        Catalog();
        ExtendedParameters();
        Components(corridor);
        Observers(schema::LoadTaskSchema(PPOKIT_OBSERVER_TASK));
        TeamCritic();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " graph document check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit graph document tests passed\n";
    return 0;
}
