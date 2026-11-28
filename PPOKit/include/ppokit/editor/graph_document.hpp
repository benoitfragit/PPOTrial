/**
 * @file PPOKit/include/ppokit/editor/graph_document.hpp
 * @brief Editable block graph behind the network editor: nodes, positions, wires, undo,
 *        and live validation with the same parser as training.
 * @details The document holds what the user draws, valid or not: an unknown block type, an
 * unwired port or a missing parameter are kept and reported, never refused. Validate runs
 * ppokit::schema::ParseBlockGraph on the document, so the editor accepts exactly the graphs
 * that train, and turns the first error into a node and a port to highlight.
 *
 * Node positions are not part of graph.json: they live in a sidecar layout file
 * (`ppokit.graph-layout/1`) so that a graph diff shows only network changes.
 *
 * The document does not depend on ImGui; the imnodes view draws it and calls its operations.
 */
#pragma once

#include "ppokit/graph/graph.hpp"
#include "ppokit/schema/block_graph_schema.hpp"

#include <cstddef>
#include <filesystem>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace ppokit::editor {

/** @brief The networks of a graph file, and its observers (Visualiseur blocks). */
enum class Network : std::uint8_t
{
    Actor,
    Critic,
    /** @brief Input nodes reading the signals of a step, wired to observer components. */
    Observers,
};

/** @brief Every section of a graph file, in display order. */
inline constexpr Network AllNetworks[] = {Network::Actor, Network::Critic, Network::Observers};

/** @brief "actor", "critic" or "observers". */
char const* NetworkName(Network network) noexcept;

/** @brief Position of a node on the canvas, in canvas units. */
struct Position final
{
    float x = 0.F;
    float y = 0.F;
};

/** @brief One block as drawn: its type name is kept even when unknown. */
struct EditorNode final
{
    std::string name;
    std::string type;
    /** @brief Block parameters, every key of the node except "type". */
    nlohmann::ordered_json parameters = nlohmann::ordered_json::object();
    Position position;
};

/** @brief One wire from the output of `source` to an input port of `target`. */
struct EditorEdge final
{
    std::string source;
    std::string target;
    /** @brief Input port of the target; empty for the single port "in". */
    std::string port;
};

/** @brief Nodes (in file order) and wires of one network. */
struct NetworkDocument final
{
    std::vector<EditorNode> nodes;
    std::vector<EditorEdge> edges;
};

/** @brief Result of a live validation, located for highlighting. */
struct Validation final
{
    bool valid = false;
    /** @brief Full message, e.g. "graph.json: actor.nodes.attend.query: expects vector[8]...". */
    std::string message;
    /** @brief Message without its location, for a tooltip on the faulty node. */
    std::string detail;
    std::optional<Network> network;
    /** @brief Faulty node, empty when the error is not on a node. */
    std::string node;
    /** @brief Faulty port or parameter of the node ("query", "width"...), possibly empty. */
    std::string field;
    /** @brief Faulty wire (index in NetworkDocument::edges), if the error is on a wire. */
    std::optional<std::size_t> edge;
    /** @brief Inferred output shape of every node, when the graph is valid. */
    std::map<std::string, graph::Shape> actorShapes;
    std::map<std::string, graph::Shape> criticShapes;
};

/** @brief Palette entry of the editor: a block type, its default parameters and its help. */
struct BlockInfo final
{
    schema::BlockType type;
    /** @brief Short title shown on the node, e.g. "Masked attention". */
    char const* title;
    /** @brief Tooltip: what the block computes and when to use it. */
    char const* help;
    /** @brief Parameters given to a new block. */
    nlohmann::ordered_json defaults;
};

/** @brief Every block type of ppokit.graph/2, in palette order. */
std::vector<BlockInfo> const& BlockCatalog();

/** @brief Palette entry of a type, or null for an unknown type name. */
BlockInfo const* FindBlock(std::string_view typeName);

/**
 * @brief Input groups a node of `network` may read, e.g. "observation.own".
 * @details The actor reads observation groups only; the critic also reads critic groups;
 * the observers also read world groups, the actions and the reward channels.
 */
std::vector<std::string> InputChoices(schema::TaskSchema const& task, Network network);

/** @brief A component library of the document: what it provides, or why it cannot be opened. */
struct ComponentLibrary final
{
    /** @brief As written in "components" (a name or a path, relative to graph.json). */
    std::string reference;
    std::vector<component::ComponentRef> components;
    /** @brief Empty when the library opened. */
    std::string error;
};

/** @brief "component:<name>@<version>", the type of a component node. */
std::string ComponentType(component::ComponentRef const& component);

/** @brief Editable actor and critic, with an undo history. */
class GraphDocument final
{
public:
    /** @brief Empty actor and critic. */
    GraphDocument() = default;

    /**
     * @brief Opens a graph file and, optionally, its layout.
     * @param graphText Content of graph.json; must be JSON with "schema": "ppokit.graph/2",
     *        everything else may be incomplete or invalid.
     * @param layoutText Content of the layout sidecar, or empty: nodes without a position
     *        are placed by depth, left to right.
     * @throws schema::SchemaError for unreadable JSON or another schema version.
     */
    static GraphDocument Open(std::string_view graphText, std::string_view layoutText = {});

    /** @brief graph.json text (ppokit.graph/2), nodes in document order. */
    std::string GraphJson() const;
    /** @brief Layout sidecar text (ppokit.graph-layout/1). */
    std::string LayoutJson() const;

    NetworkDocument const& At(Network network) const noexcept;

    /**
     * @brief Adds a block with its default parameters and a free name such as "dense_2".
     * @return The name of the new node.
     */
    std::string AddNode(Network network, schema::BlockType type, Position position);
    /** @brief Removes a node and every wire touching it. */
    void RemoveNode(Network network, std::string const& name);
    /**
     * @brief Renames a node and its wires.
     * @throws std::invalid_argument for a name that is not an identifier or already used.
     */
    void RenameNode(Network network, std::string const& name, std::string const& newName);
    /** @brief Sets one parameter (e.g. "width", "from"); a null value removes it. */
    void SetParameter(Network network, std::string const& name, std::string const& key,
        nlohmann::ordered_json const& value);
    /**
     * @brief Sets several parameters as one undo step; a null value removes its key.
     *
     * Turning off the "keys" of a masked attention also removes the wire on its keys port.
     * @param changes Object of key to value, e.g. {"log_std": "state", "log_std_min": -20}.
     */
    void SetParameters(Network network, std::string const& name, nlohmann::ordered_json const& changes);
    /**
     * @brief Wires `source` to `target.port`, replacing the wire already on that port.
     * @param port Input port; empty for the single port "in".
     */
    void Connect(Network network, std::string const& source, std::string const& target, std::string const& port);
    /** @brief Removes the wire on `target.port`, if any. */
    void Disconnect(Network network, std::string const& target, std::string const& port);
    /** @brief First free numbered port (in1, in2...) of a concatenation, for a new wire. */
    std::string NextNumberedPort(Network network, std::string const& target) const;

    /** @brief Moves a node; a drag is recorded as one undo step by BeginMove and EndMove. */
    void MoveNode(Network network, std::string const& name, Position position);
    /** @brief Remembers the positions at the start of a drag. */
    void BeginMove();
    /** @brief Records the drag as one undo step if a node moved since BeginMove. */
    void EndMove();

    bool CanUndo() const noexcept { return !undo_.empty(); }
    bool CanRedo() const noexcept { return !redo_.empty(); }
    void Undo();
    void Redo();

    /** @brief Parses the document as training would, against `task`. */
    Validation Validate(schema::TaskSchema const& task) const;

    /** @brief Folder of graph.json, where component libraries are looked up. */
    void SetDirectory(std::filesystem::path directory) { directory_ = std::move(directory); }
    std::filesystem::path const& Directory() const noexcept { return directory_; }
    /** @brief Library references of "components", in file order. */
    std::vector<std::string> const& ComponentReferences() const noexcept { return components_; }
    /** @brief Opens every library of the document (each DLL is loaded once per process). */
    std::vector<ComponentLibrary> ComponentLibraries() const;
    /** @brief Adds a library reference (undoable); a reference already listed is ignored. */
    void AddComponentLibrary(std::string const& reference);
    void RemoveComponentLibrary(std::string const& reference);
    /** @brief Adds a node of a component with its default configuration; returns its name. */
    std::string AddComponentNode(Network network, component::ComponentRef const& component, Position position);

private:
    struct Snapshot final
    {
        NetworkDocument actor;
        NetworkDocument critic;
        NetworkDocument observers;
        std::vector<std::string> components;
        nlohmann::ordered_json teamCritic;
    };

    NetworkDocument& Mutable(Network network) noexcept;
    EditorNode& RequireNode(Network network, std::string const& name);
    /** @brief Saves the current state for Undo and forgets the redo history. */
    void Record();
    /** @brief Pushes a state on the undo history (bounded) and forgets the redo history. */
    void PushUndo(Snapshot snapshot);
    Snapshot Current() const;
    void Restore(Snapshot const& snapshot);
    void PlaceUnpositioned(Network network, std::vector<bool> const& positioned);

    NetworkDocument actor_;
    NetworkDocument critic_;
    NetworkDocument observers_;
    std::vector<std::string> components_;
    /** @brief The "team_critic" section, kept as written (null when absent); follows actor renames. */
    nlohmann::ordered_json teamCritic_;
    std::filesystem::path directory_;
    std::vector<Snapshot> undo_;
    std::vector<Snapshot> redo_;
    std::optional<Snapshot> moveStart_;
};

} // namespace ppokit::editor
