/**
 * @file PPOKit/studio/graph_editor_view.hpp
 * @brief ImGui + imnodes view of a block graph: palette, canvas, inspector, live errors.
 * @details The view draws an editor::GraphDocument and turns user gestures into document
 * operations; it owns no network logic. Validation runs after every change with the same
 * parser as training, so the status bar says exactly whether the graph will train.
 *
 * Gestures: click a palette entry to add a block; drag from an output pin to an input pin
 * to wire (the dashed "+" pin of a concatenation adds a port); Ctrl+click a wire to detach
 * it; Delete removes the selection; Ctrl+Z / Ctrl+Y undo and redo; Ctrl+S saves.
 *
 * It needs a current ImGui context and imnodes context, nothing else: the studio window
 * uses raylib, the tests run it headless.
 */
#pragma once

#include "ppokit/editor/graph_document.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <imgui.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

struct ImNodesEditorContext;

namespace ppokit::studio {

/** @brief Files edited by the view. */
struct GraphFiles final
{
    std::filesystem::path graph;
    /** @brief Layout sidecar; LayoutPathFor(graph) by default. */
    std::filesystem::path layout;
};

/** @brief "graph_blocks.json" -> "graph_blocks.layout.json". */
std::filesystem::path LayoutPathFor(std::filesystem::path const& graph);

/**
 * @brief Opens a network file and its layout; a missing network gives an empty document.
 *        Component libraries are found relative to the network file's folder.
 * @throws std::runtime_error when a file exists but cannot be read.
 */
editor::GraphDocument OpenGraphDocument(std::filesystem::path const& graph, std::filesystem::path const& layout);

class GraphEditorView final
{
public:
    /**
     * @param task Task the graph must fit (inputs and heads).
     * @param document Graph to edit.
     * @param files Where Save writes; an empty graph path disables saving.
     */
    GraphEditorView(schema::TaskSchema task, editor::GraphDocument document, GraphFiles files);
    ~GraphEditorView();
    GraphEditorView(GraphEditorView const&) = delete;
    GraphEditorView& operator=(GraphEditorView const&) = delete;

    /** @brief Draws the whole editor in the current ImGui window and applies the user's edits. */
    void Draw();

    /** @brief Writes graph.json and its layout atomically; returns false and reports on failure. */
    bool Save();

    bool Dirty() const noexcept { return dirty_; }
    /** @brief Shows the actor or the critic. */
    void ShowNetwork(editor::Network network) noexcept { network_ = network; }

    /**
     * @brief The project's components/ folder: its libraries are offered in the palette, and
     *        "Add a DLL..." copies a chosen library there.
     */
    void SetComponentFolder(std::filesystem::path folder);

    /**
     * @brief Shows the network without letting it change (a network compiled into its program).
     * @param reason Shown in place of the palette, e.g. "FlightControl3D compiles this network".
     */
    void SetReadOnly(std::string reason) { readOnlyReason_ = std::move(reason); }
    /** @brief The network cannot be changed here. */
    bool ReadOnly() const noexcept { return !readOnlyReason_.empty(); }

    /** @brief Zoom of the network shown (1 = 100 %). */
    float Zoom() const noexcept;
    /**
     * @brief Scales the canvas by `factor` (clamped to 30 %..250 %) around `pivot`, a point of
     *        the canvas in pixels that stays in place. Positions are saved at 100 %.
     */
    void ZoomCanvas(float factor, ImVec2 pivot);
    /**
     * @brief Zooms and pans so that the whole network fits the canvas (at most 100 %).
     * @details Needs one drawn frame to know the node sizes; before that, the fit is done
     *          at the next frame. Each network is fitted once when it is first shown.
     */
    void FitCanvas();
    editor::GraphDocument const& Document() const noexcept { return document_; }
    editor::Validation const& LastValidation() const noexcept { return validation_; }
    editor::Network CurrentNetwork() const noexcept { return network_; }
    void SelectNetwork(editor::Network network) noexcept { network_ = network; }
    /** @brief Shows a node in the inspector, as a click on it would. */
    void SelectNode(std::string name) { selected_ = std::move(name); }

private:
    /** @brief imnodes identifiers of the nodes and pins drawn this frame. */
    struct PinMap final
    {
        struct Input final
        {
            std::size_t node = 0;
            std::string port;
        };
        std::vector<Input> inputs;
    };

    void DrawToolbar();
    /** @brief Name, output shape, settings and removal of the selected node. */
    void DrawNodeEditing(editor::EditorNode const& node);
    /** @brief Zoom out, zoom percentage (back to 100 %), zoom in, fit. */
    void DrawZoomControls();
    void DrawPalette();
    void DrawCanvas();
    /** @brief Where a block added from the palette appears, at 100 %. */
    editor::Position NewNodePosition() const;
    void DrawNode(std::size_t index);
    void DrawInspector();
    void DrawParameters(editor::EditorNode const& node);
    /** @brief "Separate keys" of a masked attention: adds or removes its keys port. */
    void DrawAttentionKeys(editor::EditorNode const& node);
    /** @brief Log-std of a Gaussian head: a learned parameter, or a clamped dense layer (state). */
    void DrawLogStd(editor::EditorNode const& node);
    /** @brief Actor block whose parameters a critic layer shares (joint optimizer). */
    void DrawShare(editor::EditorNode const& node);
    void DrawStatus();
    void HandleShortcuts();
    void HandleWiring();
    void HandleDeletion();
    void TrackDrags();
    void UpdateSelection(bool canvasClicked);
    /**
     * @brief Records the extent of the nodes just drawn, for FitCanvas.
     * @param settled False while positions just pushed to imnodes settle: the extent is then unknown.
     */
    void MeasureNodes(bool settled);
    /** @brief Applies a fit that waits for node sizes. */
    void ApplyPendingFit();

    /** @brief Applies a document change: revalidates and marks the canvas for repositioning. */
    void Changed();
    /** @brief Wraps a document operation, reporting std::invalid_argument in the status bar. */
    template<typename Operation>
    void Apply(Operation&& operation);

    std::vector<std::string> InputPortsOf(editor::EditorNode const& node) const;
    /** @brief The component of a "component:<name>@<version>" node, or null when no library provides it. */
    component::ComponentRef const* FindComponent(std::string const& type) const;
    /** @brief Reopens the libraries when the document's list changed (not every frame). */
    void RefreshLibraries();
    void DrawComponentPalette();
    /** @brief Libraries of the components folder not used by the graph yet, with a "Use" button. */
    void DrawProjectLibraries();
    /** @brief Reads the components folder again. */
    void ScanComponentFolder();
    /** @brief Copies `library` into the components folder (unless it is there) and uses it. */
    void UseLibrary(std::filesystem::path const& library);
    void DrawComponentParameters(editor::EditorNode const& node);

    schema::TaskSchema task_;
    editor::GraphDocument document_;
    GraphFiles files_;
    editor::Validation validation_;
    /** @brief Actor, critic and observers each have their own canvas. */
    static constexpr std::size_t NetworkCount = 3;
    editor::Network network_ = editor::Network::Actor;
    std::array<ImNodesEditorContext*, NetworkCount> contexts_{};
    /** @brief Positions must be pushed to imnodes (after open, add, undo...). */
    std::array<bool, NetworkCount> positionsStale_{true, true, true};
    /** @brief Canvas zoom of each network (1 = 100 %). */
    std::array<float, NetworkCount> zoom_{1.F, 1.F, 1.F};
    /** @brief Extent of the drawn nodes, in canvas units at 100 %, from the last frame. */
    struct Extent final
    {
        ImVec2 minimum{};
        ImVec2 maximum{};
        bool known = false;
    };
    std::array<Extent, NetworkCount> extents_{};
    /** @brief A fit waits for the node sizes of the next frame. */
    std::array<bool, NetworkCount> fitPending_{true, true, true};
    /** @brief Frames drawn since positions were last pushed to imnodes. */
    std::array<int, NetworkCount> settledFrames_{};
    /** @brief Size of the canvas in pixels, from the last frame. */
    ImVec2 canvasSize_{};
    /** @brief components/ of the project, and the libraries found there. */
    std::filesystem::path componentFolder_;
    std::vector<std::filesystem::path> componentFiles_;
    std::string componentError_;
    PinMap pins_;
    bool dragging_ = false;
    /** @brief Layout when the left button went down on the canvas, to detect real moves. */
    std::string layoutAtDragStart_;
    std::string selected_;
    std::array<char, 64> renameBuffer_{};
    std::array<char, 256> indicesBuffer_{};
    std::string message_;
    /** @brief Why the network cannot be changed; empty when it can. */
    std::string readOnlyReason_;
    bool dirty_ = false;
    std::vector<editor::ComponentLibrary> libraries_;
    std::vector<std::string> librariesFor_;
    std::array<char, 256> libraryBuffer_{};
};

} // namespace ppokit::studio
