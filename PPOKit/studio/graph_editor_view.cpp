/**
 * @file PPOKit/studio/graph_editor_view.cpp
 * @brief Drawing and gesture handling of the block graph editor (ImGui + imnodes).
 *
 * imnodes identifies nodes and pins by integers. They are rebuilt every frame from the
 * document: node i has id i + 1, its output pin OutputPinBase + i, and the k-th input pin
 * drawn this frame InputPinBase + k (PinMap tells which node and port it is). Wire j of
 * the current network has id j. Nothing else is stored, so the view cannot drift from
 * the document after an undo.
 */
#include "graph_editor_view.hpp"

#include "file_dialogs.hpp"
#include "studio_icons.hpp"
#include "studio_theme.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <imgui.h>
#include <imnodes.h>

#include <algorithm>
#include <fstream>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ppokit::studio {
namespace {

using editor::Network;

constexpr int InputPinBase = 100000;
constexpr int OutputPinBase = 200000;
constexpr float PaletteWidth = 210.F;
constexpr float InspectorWidth = 300.F;
constexpr float StatusHeight = 54.F;
constexpr float NodeTextWidth = 190.F;
constexpr float OutputIndent = 40.F;
/** @brief Zoom range of the canvas and factor of one mouse-wheel notch. */
constexpr float MinimumZoom = 0.3F;
constexpr float MaximumZoom = 2.5F;
constexpr float ZoomStep = 1.12F;
/** @brief Distance of a new block from the visible top left corner, in canvas pixels. */
constexpr float NewNodeOffset = 120.F;
/** @brief Free border around the network when it is fitted to the canvas, in pixels. */
constexpr float FitMargin = 30.F;
/** @brief Frames after a position change before node sizes are trusted (ImGui sizes new items late). */
constexpr int SettlingFrames = 2;
/** @brief imnodes sizes scaled with the zoom, so that nodes, pins and grid grow with the text. */
constexpr std::array<ImNodesStyleVar, 9> ZoomedNodeSizes{ImNodesStyleVar_GridSpacing,
    ImNodesStyleVar_NodeCornerRounding, ImNodesStyleVar_NodeBorderThickness, ImNodesStyleVar_LinkThickness,
    ImNodesStyleVar_PinCircleRadius, ImNodesStyleVar_PinQuadSideLength, ImNodesStyleVar_PinTriangleSideLength,
    ImNodesStyleVar_PinHoverRadius, ImNodesStyleVar_PinOffset};

/** @brief Value of a scalar imnodes size in the current style. */
float NodeSize(ImNodesStyle const& style, ImNodesStyleVar variable) noexcept
{
    switch (variable)
    {
    case ImNodesStyleVar_GridSpacing:
        return style.GridSpacing;
    case ImNodesStyleVar_NodeCornerRounding:
        return style.NodeCornerRounding;
    case ImNodesStyleVar_NodeBorderThickness:
        return style.NodeBorderThickness;
    case ImNodesStyleVar_LinkThickness:
        return style.LinkThickness;
    case ImNodesStyleVar_PinCircleRadius:
        return style.PinCircleRadius;
    case ImNodesStyleVar_PinQuadSideLength:
        return style.PinQuadSideLength;
    case ImNodesStyleVar_PinTriangleSideLength:
        return style.PinTriangleSideLength;
    case ImNodesStyleVar_PinHoverRadius:
        return style.PinHoverRadius;
    case ImNodesStyleVar_PinOffset:
        return style.PinOffset;
    default:
        break;
    }
    return 0.F;
}

/** @brief Pushes the imnodes sizes and the font size of `zoom`; PopZoom undoes it. */
void PushZoom(float zoom)
{
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * zoom);
    ImNodesStyle const& style = ImNodes::GetStyle();
    for (ImNodesStyleVar const variable : ZoomedNodeSizes)
    {
        ImNodes::PushStyleVar(variable, NodeSize(style, variable) * zoom);
    }
    ImNodes::PushStyleVar(ImNodesStyleVar_NodePadding,
        ImVec2(style.NodePadding.x * zoom, style.NodePadding.y * zoom));
}

void PopZoom()
{
    ImNodes::PopStyleVar(static_cast<int>(ZoomedNodeSizes.size()) + 1);
    ImGui::PopFont();
}
/** @brief Longest index accepted in a select block (guards the integer conversion). */
constexpr std::size_t MaximumIndexDigits = 6;

constexpr ImU32 ErrorColor = IM_COL32(200, 60, 55, 255);
constexpr ImU32 ErrorHoveredColor = IM_COL32(225, 80, 70, 255);
constexpr ImU32 SetPinColor = IM_COL32(230, 160, 60, 255);
constexpr ImVec4 ValidText{0.45F, 0.85F, 0.45F, 1.F};
constexpr ImVec4 ErrorText{1.F, 0.45F, 0.4F, 1.F};

char const* const Activations[] = {"tanh", "relu", "identity"};
char const* const PoolModes[] = {"mean", "max"};
char const* const LogStdModes[] = {"parameter", "state"};

/** @brief A boolean parameter such as "keys"; false when absent or not a boolean. */
bool FlagParameter(editor::EditorNode const& node, char const* key)
{
    auto const found = node.parameters.find(key);
    return found != node.parameters.end() && found->is_boolean() && found->get<bool>();
}

/** @brief A number parameter such as "log_std_min"; `fallback` when absent or not a number. */
float NumberParameter(editor::EditorNode const& node, char const* key, float fallback)
{
    auto const found = node.parameters.find(key);
    return found != node.parameters.end() && found->is_number() ? found->get<float>() : fallback;
}

/** @brief A text parameter such as "share"; `fallback` when absent or not a string. */
std::string TextParameter(editor::EditorNode const& node, char const* key, char const* fallback)
{
    auto const found = node.parameters.find(key);
    return found != node.parameters.end() && found->is_string() ? found->get<std::string>() : std::string(fallback);
}
/** @brief Default clamp of a state-dependent log-std, as the schema documents it. */
constexpr float DefaultLogStdMinimum = -20.F;
constexpr float DefaultLogStdMaximum = 2.F;

int NetworkIndex(Network network) noexcept
{
    return static_cast<int>(network);
}

/** @brief True when the palette offers `type` in `network`: observers only take input nodes. */
bool OffersBlock(Network network, schema::BlockType type) noexcept
{
    return network != Network::Observers || type == schema::BlockType::Input;
}

/** @brief True when the palette offers a component in `network`: observers in observers only. */
bool OffersComponent(Network network, component::ComponentRef const& component) noexcept
{
    return component.manifest->capabilities.observer == (network == Network::Observers);
}

bool IsHead(schema::BlockType type) noexcept
{
    return type == schema::BlockType::GaussianHead || type == schema::BlockType::BernoulliHead
        || type == schema::BlockType::Value;
}

/**
 * @brief Whether a node draws an output pin: heads and observer components have none; in
 *        the observers canvas only input nodes feed others.
 */
bool HasOutputPin(Network network, editor::BlockInfo const* info) noexcept
{
    if (network == Network::Observers)
    {
        return info != nullptr && info->type == schema::BlockType::Input;
    }
    return info == nullptr || !IsHead(info->type);
}

/** @brief One line on what a component does with gradients, for its palette tooltip. */
char const* ComponentRole(component::Capabilities const& capabilities) noexcept
{
    if (capabilities.observer)
    {
        return "Observer: only watches the signals wired to it (for example to show the world).";
    }
    if (capabilities.trainable)
    {
        return "Trainable: PPOKit's optimizer updates its parameters.";
    }
    if (capabilities.differentiable)
    {
        return "Differentiable, no parameters.";
    }
    return "No gradient: place it before every trainable block.";
}

/** @brief A component port declared [capacity, n]. */
bool IsComponentSetPort(component::ComponentRef const* component, std::string const& port) noexcept
{
    if (component == nullptr)
    {
        return false;
    }
    for (component::PortSpec const& spec : component->manifest->ports)
    {
        if (spec.name == port)
        {
            return spec.shape.size() == 2;
        }
    }
    return false;
}

/** @brief Ports that receive a set: drawn as squares, in another colour. */
bool IsSetPort(schema::BlockType type, std::string const& port) noexcept
{
    switch (type)
    {
    case schema::BlockType::EntityEncoder:
    case schema::BlockType::MaskedPool:
    case schema::BlockType::ConcatSets:
        return true;
    case schema::BlockType::MaskedAttention:
        return port == "set";
    default:
        break;
    }
    return false;
}

/** @brief Port as stored in a wire: the single port "in" is written without a name. */
std::string StoredPort(std::string const& port)
{
    return port == "in" ? std::string() : port;
}

void HelpMarker(char const* text)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.F);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

std::string ShapeText(editor::Validation const& validation, Network network, std::string const& node)
{
    if (network == Network::Observers)
    {
        return {};
    }
    auto const& shapes = network == Network::Actor ? validation.actorShapes : validation.criticShapes;
    auto const shape = shapes.find(node);
    return shape == shapes.end() ? std::string() : graph::Describe(shape->second);
}

/** @brief Parses "0, 1, 3" into indices; nullopt when a token is not a non-negative integer. */
std::optional<nlohmann::ordered_json> ParseIndices(char const* text)
{
    nlohmann::ordered_json indices = nlohmann::ordered_json::array();
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (token.empty() || token.size() > MaximumIndexDigits
            || token.find_first_not_of("0123456789") != std::string::npos)
        {
            return std::nullopt;
        }
        indices.push_back(std::stoul(token));
    }
    if (indices.empty())
    {
        return std::nullopt;
    }
    return indices;
}

std::string FormatIndices(nlohmann::ordered_json const& indices)
{
    std::string text;
    if (!indices.is_array())
    {
        return text;
    }
    for (auto const& index : indices)
    {
        text += (text.empty() ? "" : ", ") + index.dump();
    }
    return text;
}

template<std::size_t Size>
void CopyToBuffer(std::array<char, Size>& buffer, std::string const& text)
{
    std::snprintf(buffer.data(), buffer.size(), "%s", text.c_str());
}

} // namespace

namespace {

std::string ReadWholeFile(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

} // namespace

editor::GraphDocument OpenGraphDocument(std::filesystem::path const& graph, std::filesystem::path const& layout)
{
    std::string const graphText = std::filesystem::exists(graph) ? ReadWholeFile(graph)
                                                                 : std::string(R"({ "schema": "ppokit.graph/2" })");
    std::string const layoutText = std::filesystem::exists(layout) ? ReadWholeFile(layout) : std::string();
    editor::GraphDocument document = editor::GraphDocument::Open(graphText, layoutText);
    document.SetDirectory(std::filesystem::absolute(graph).parent_path());
    return document;
}

std::filesystem::path LayoutPathFor(std::filesystem::path const& graph)
{
    std::filesystem::path layout = graph;
    layout.replace_extension(".layout.json");
    return layout;
}

GraphEditorView::GraphEditorView(schema::TaskSchema task, editor::GraphDocument document, GraphFiles files)
    : task_(std::move(task)), document_(std::move(document)), files_(std::move(files))
{
    if (files_.layout.empty() && !files_.graph.empty())
    {
        files_.layout = LayoutPathFor(files_.graph);
    }
    for (ImNodesEditorContext*& context : contexts_)
    {
        context = ImNodes::EditorContextCreate();
    }
    // Ctrl+click on a wire detaches it (then it is removed, see HandleWiring).
    ImNodes::GetIO().LinkDetachWithModifierClick.Modifier = &ImGui::GetIO().KeyCtrl;
    validation_ = document_.Validate(task_);
}

GraphEditorView::~GraphEditorView()
{
    for (ImNodesEditorContext* context : contexts_)
    {
        ImNodes::EditorContextFree(context);
    }
}

void GraphEditorView::Draw()
{
    RefreshLibraries();
    if (!ReadOnly())
    {
        HandleShortcuts();
    }
    DrawToolbar();
    float const bodyHeight = std::max(100.F, ImGui::GetContentRegionAvail().y - StatusHeight);

    BeginPanel("palette", ImVec2(PaletteWidth, bodyHeight));
    DrawPalette();
    EndPanel();

    ImGui::SameLine();
    float const canvasWidth = std::max(100.F, ImGui::GetContentRegionAvail().x - InspectorWidth);
    ImGui::BeginChild("canvas", ImVec2(canvasWidth, bodyHeight), ImGuiChildFlags_None);
    DrawCanvas();
    ImGui::EndChild();

    ImGui::SameLine();
    BeginPanel("inspector", ImVec2(0.F, bodyHeight));
    DrawInspector();
    EndPanel();

    DrawStatus();
}

void GraphEditorView::DrawToolbar()
{
    char const* const saveHelp = dirty_ ? "Save the network and its layout (Ctrl+S). There are unsaved changes."
                                        : "Save the network and its layout (Ctrl+S).";
    bool const editable = !ReadOnly();
    if (IconButton("save", StudioIcon::Save, saveHelp, editable && !files_.graph.empty() && dirty_))
    {
        Save();
    }
    ImGui::SameLine();
    if (IconButton("undo", StudioIcon::Undo, "Undo (Ctrl+Z)", editable && document_.CanUndo()))
    {
        Apply([this]() { document_.Undo(); });
    }
    ImGui::SameLine();
    if (IconButton("redo", StudioIcon::Redo, "Redo (Ctrl+Y)", editable && document_.CanRedo()))
    {
        Apply([this]() { document_.Redo(); });
    }
    ImGui::SameLine(0.F, ImGui::GetFontSize());
    if (ToggleChip("Actor", network_ == Network::Actor, "Policy network: reads the observation, ends in the action heads."))
    {
        network_ = Network::Actor;
    }
    ImGui::SameLine();
    if (ToggleChip("Critic", network_ == Network::Critic,
            "Value network: may also read the centralized critic input, ends in one value."))
    {
        network_ = Network::Critic;
    }
    ImGui::SameLine();
    if (ToggleChip("Observers", network_ == Network::Observers,
            "Visualiseur blocks: watch observations, world state, actions and rewards after each step. "
            "They never change the networks."))
    {
        network_ = Network::Observers;
    }
    ImGui::SameLine(0.F, ImGui::GetFontSize());
    DrawZoomControls();
    ImGui::SameLine(0.F, ImGui::GetFontSize());
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(MutedText, "%s", files_.graph.empty() ? "(unsaved graph)" : files_.graph.filename().string().c_str());
}

void GraphEditorView::DrawZoomControls()
{
    if (IconButton("zoom-out", StudioIcon::ZoomOut, "Zoom out (mouse wheel down over the canvas)."))
    {
        ZoomCanvas(1.F / ZoomStep, ImVec2(0.F, 0.F));
    }
    ImGui::SameLine();
    char zoomLabel[16];
    std::snprintf(zoomLabel, sizeof zoomLabel, "%.0f %%", 100.F * Zoom());
    if (ImGui::Button(zoomLabel, ImVec2(ImGui::GetFontSize() * 4.F, 0.F)))
    {
        ZoomCanvas(1.F / Zoom(), ImVec2(0.F, 0.F));
    }
    ItemTooltip("Back to 100 %.");
    ImGui::SameLine();
    if (IconButton("zoom-in", StudioIcon::ZoomIn,
            "Zoom in (mouse wheel up over the canvas). Drag the background with the middle button to move."))
    {
        ZoomCanvas(ZoomStep, ImVec2(0.F, 0.F));
    }
    ImGui::SameLine();
    if (IconButton("fit", StudioIcon::Fit, "Show the whole network."))
    {
        FitCanvas();
    }
}

void GraphEditorView::DrawPalette()
{
    if (ReadOnly())
    {
        IconText(StudioIcon::Warning, HealthColor(editor::Health::Watch));
        ImGui::SameLine();
        ImGui::TextColored(HealthColor(editor::Health::Watch), "Read only");
        MutedWrapped(readOnlyReason_.empty()
            ? "An active run owns this configuration. Stop it before changing the network."
            : readOnlyReason_.c_str());
        ImGui::Spacing();
        MutedWrapped("Hover a block for its role; select it to see its settings. Wheel to zoom, middle button to "
                     "move.");
        return;
    }
    ImGui::SeparatorText("Blocks");
    ImGui::TextDisabled("Click to add, hover for help");
    for (editor::BlockInfo const& info : editor::BlockCatalog())
    {
        if (!OffersBlock(network_, info.type))
        {
            continue;
        }
        if (ImGui::Selectable(info.title))
        {
            editor::Position const position = NewNodePosition();
            Apply([&]() { selected_ = document_.AddNode(network_, info.type, position); });
        }
        if (ImGui::BeginItemTooltip())
        {
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 25.F);
            ImGui::TextUnformatted(info.help);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
    DrawComponentPalette();
    ImGui::SeparatorText("Wiring");
    ImGui::TextWrapped("Drag from an output (right) to an input (left). Squares carry sets of entities.");
    ImGui::TextWrapped("Ctrl+click a wire to detach it. Delete removes the selection.");
}

void GraphEditorView::DrawCanvas()
{
    int const networkIndex = NetworkIndex(network_);
    float const zoom = zoom_[networkIndex];
    ImVec2 const origin = ImGui::GetCursorScreenPos();
    canvasSize_ = ImGui::GetContentRegionAvail();
    ImNodes::EditorContextSet(contexts_[networkIndex]);
    PushZoom(zoom);
    ImNodes::BeginNodeEditor();
    editor::NetworkDocument const& network = document_.At(network_);
    if (positionsStale_[networkIndex])
    {
        settledFrames_[networkIndex] = 0;
        // The document keeps positions at 100 %; imnodes draws them at the zoom.
        for (std::size_t index = 0; index < network.nodes.size(); ++index)
        {
            editor::Position const position = network.nodes[index].position;
            ImNodes::SetNodeGridSpacePos(static_cast<int>(index) + 1, ImVec2(position.x * zoom, position.y * zoom));
        }
        positionsStale_[networkIndex] = false;
    }
    pins_.inputs.clear();
    for (std::size_t index = 0; index < network.nodes.size(); ++index)
    {
        DrawNode(index);
    }
    for (std::size_t edge = 0; edge < network.edges.size(); ++edge)
    {
        editor::EditorEdge const& wire = network.edges[edge];
        auto const source = std::find_if(network.nodes.begin(), network.nodes.end(),
            [&wire](editor::EditorNode const& node) { return node.name == wire.source; });
        std::string const port = wire.port.empty() ? "in" : wire.port;
        auto const target = std::find_if(pins_.inputs.begin(), pins_.inputs.end(), [&](PinMap::Input const& pin) {
            return network.nodes[pin.node].name == wire.target && pin.port == port;
        });
        if (source == network.nodes.end() || target == pins_.inputs.end())
        {
            continue; // A wire to a missing node or port: reported by the validation, not drawable.
        }
        bool const faulty = validation_.network == network_ && validation_.edge == edge;
        if (faulty)
        {
            ImNodes::PushColorStyle(ImNodesCol_Link, ErrorColor);
        }
        ImNodes::Link(static_cast<int>(edge),
            OutputPinBase + static_cast<int>(source - network.nodes.begin()),
            InputPinBase + static_cast<int>(target - pins_.inputs.begin()));
        if (faulty)
        {
            ImNodes::PopColorStyle();
        }
    }
    ImNodes::MiniMap(0.15F, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();
    PopZoom();
    ++settledFrames_[networkIndex];
    MeasureNodes(settledFrames_[networkIndex] > SettlingFrames);
    ApplyPendingFit();

    bool const hovered = ImNodes::IsEditorHovered();
    float const wheel = ImGui::GetIO().MouseWheel;
    if (hovered && wheel != 0.F)
    {
        ImVec2 const mouse = ImGui::GetMousePos();
        ZoomCanvas(std::pow(ZoomStep, wheel), ImVec2(mouse.x - origin.x, mouse.y - origin.y));
    }
    bool const canvasClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered;
    if (!ReadOnly())
    {
        TrackDrags();
        HandleWiring();
        HandleDeletion();
    }
    UpdateSelection(canvasClicked);
}

void GraphEditorView::ZoomCanvas(float factor, ImVec2 pivot)
{
    int const networkIndex = NetworkIndex(network_);
    float const before = zoom_[networkIndex];
    float const after = std::clamp(before * factor, MinimumZoom, MaximumZoom);
    if (after == before)
    {
        return;
    }
    // Keep the point under `pivot` (canvas coordinates) in place while the scale changes.
    ImNodes::EditorContextSet(contexts_[networkIndex]);
    ImVec2 const panning = ImNodes::EditorContextGetPanning();
    float const ratio = after / before;
    ImNodes::EditorContextResetPanning(
        ImVec2(pivot.x - (pivot.x - panning.x) * ratio, pivot.y - (pivot.y - panning.y) * ratio));
    zoom_[networkIndex] = after;
    positionsStale_[networkIndex] = true;
}

void GraphEditorView::MeasureNodes(bool settled)
{
    int const networkIndex = NetworkIndex(network_);
    if (!settled)
    {
        // Nodes just moved or created are not measured right for a couple of frames.
        extents_[networkIndex].known = false;
        return;
    }
    float const zoom = zoom_[networkIndex];
    std::size_t const count = document_.At(network_).nodes.size();
    Extent extent;
    for (std::size_t index = 0; index < count; ++index)
    {
        int const id = static_cast<int>(index) + 1;
        ImVec2 const position = ImNodes::GetNodeGridSpacePos(id);
        ImVec2 const size = ImNodes::GetNodeDimensions(id);
        ImVec2 const minimum(position.x / zoom, position.y / zoom);
        ImVec2 const maximum((position.x + size.x) / zoom, (position.y + size.y) / zoom);
        if (!extent.known)
        {
            extent = {minimum, maximum, true};
            continue;
        }
        extent.minimum = ImVec2(std::min(extent.minimum.x, minimum.x), std::min(extent.minimum.y, minimum.y));
        extent.maximum = ImVec2(std::max(extent.maximum.x, maximum.x), std::max(extent.maximum.y, maximum.y));
    }
    extents_[networkIndex] = extent;
}

void GraphEditorView::FitCanvas()
{
    fitPending_[NetworkIndex(network_)] = true;
}

/**
 * @brief Picks the largest zoom (up to 100 %) that shows every node, then pans the network to the top left.
 * @details Runs after a frame, once imnodes knows the node sizes of the network shown.
 */
void GraphEditorView::ApplyPendingFit()
{
    int const networkIndex = NetworkIndex(network_);
    Extent const& extent = extents_[networkIndex];
    if (!fitPending_[networkIndex])
    {
        return;
    }
    if (document_.At(network_).nodes.empty())
    {
        fitPending_[networkIndex] = false; // Nothing to fit.
        return;
    }
    if (!extent.known || canvasSize_.x <= 2.F * FitMargin
        || canvasSize_.y <= 2.F * FitMargin)
    {
        return;
    }
    fitPending_[networkIndex] = false;
    float const width = std::max(extent.maximum.x - extent.minimum.x, 1.F);
    float const height = std::max(extent.maximum.y - extent.minimum.y, 1.F);
    float const fitted = std::min((canvasSize_.x - 2.F * FitMargin) / width, (canvasSize_.y - 2.F * FitMargin) / height);
    float const zoom = std::clamp(fitted, MinimumZoom, 1.F);
    zoom_[networkIndex] = zoom;
    ImNodes::EditorContextSet(contexts_[networkIndex]);
    ImNodes::EditorContextResetPanning(
        ImVec2(FitMargin - extent.minimum.x * zoom, FitMargin - extent.minimum.y * zoom));
    positionsStale_[networkIndex] = true;
}

float GraphEditorView::Zoom() const noexcept
{
    return zoom_[NetworkIndex(network_)];
}

editor::Position GraphEditorView::NewNodePosition() const
{
    // A new block appears near the top left corner of what the canvas shows.
    ImNodes::EditorContextSet(contexts_[NetworkIndex(network_)]);
    ImVec2 const panning = ImNodes::EditorContextGetPanning();
    float const zoom = Zoom();
    return {(NewNodeOffset - panning.x) / zoom, (NewNodeOffset - panning.y) / zoom};
}

/**
 * @brief The inspector follows a single selected node; a click on empty canvas clears it.
 * @details A node chosen from the palette or the status bar stays inspected until then.
 */
void GraphEditorView::UpdateSelection(bool canvasClicked)
{
    editor::NetworkDocument const& network = document_.At(network_);
    if (ImNodes::NumSelectedNodes() == 1)
    {
        int node = 0;
        ImNodes::GetSelectedNodes(&node);
        std::size_t const index = static_cast<std::size_t>(node - 1);
        if (index < network.nodes.size())
        {
            selected_ = network.nodes[index].name;
        }
        return;
    }
    if (canvasClicked && ImNodes::NumSelectedNodes() == 0)
    {
        selected_.clear();
    }
}

void GraphEditorView::DrawNode(std::size_t index)
{
    editor::EditorNode const& node = document_.At(network_).nodes[index];
    editor::BlockInfo const* const info = editor::FindBlock(node.type);
    bool const faulty = validation_.network == network_ && validation_.node == node.name;
    if (faulty)
    {
        ImNodes::PushColorStyle(ImNodesCol_TitleBar, ErrorColor);
        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, ErrorHoveredColor);
        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, ErrorHoveredColor);
    }
    ImNodes::SetNodeDraggable(static_cast<int>(index) + 1, !ReadOnly());
    ImNodes::BeginNode(static_cast<int>(index) + 1);
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(node.name.c_str());
    component::ComponentRef const* const component = info == nullptr ? FindComponent(node.type) : nullptr;
    if (info != nullptr)
    {
        ImGui::TextDisabled("%s", info->title);
    }
    else if (component != nullptr)
    {
        ImGui::TextDisabled("component %s", component->Id().c_str());
    }
    else
    {
        ImGui::TextDisabled("unknown: %s", node.type.c_str());
    }
    ImNodes::EndNodeTitleBar();

    for (std::string const& port : InputPortsOf(node))
    {
        int const pin = InputPinBase + static_cast<int>(pins_.inputs.size());
        pins_.inputs.push_back({index, port});
        bool const set = info != nullptr ? IsSetPort(info->type, port) : IsComponentSetPort(component, port);
        if (set)
        {
            ImNodes::PushColorStyle(ImNodesCol_Pin, SetPinColor);
        }
        ImNodes::BeginInputAttribute(pin, set ? ImNodesPinShape_QuadFilled : ImNodesPinShape_CircleFilled);
        bool const faultyPort = faulty && validation_.field == port;
        if (faultyPort)
        {
            ImGui::TextColored(ErrorText, "%s", port.c_str());
        }
        else
        {
            ImGui::TextUnformatted(port.c_str());
        }
        ImNodes::EndInputAttribute();
        if (set)
        {
            ImNodes::PopColorStyle();
        }
    }

    if (info != nullptr && info->type == schema::BlockType::Input)
    {
        std::string const from = node.parameters.value("from", "");
        ImGui::TextDisabled("%s", from.empty() ? "(choose a group)" : from.c_str());
    }
    if (HasOutputPin(network_, info))
    {
        std::string const shape = ShapeText(validation_, network_, node.name);
        ImNodes::BeginOutputAttribute(OutputPinBase + static_cast<int>(index));
        ImGui::Indent(OutputIndent * Zoom());
        ImGui::TextUnformatted(shape.empty() ? "out" : shape.c_str());
        ImGui::Unindent(OutputIndent * Zoom());
        ImNodes::EndOutputAttribute();
    }
    else
    {
        std::string const shape = ShapeText(validation_, network_, node.name);
        ImGui::TextDisabled("-> %s", shape.empty() ? "output" : shape.c_str());
    }
    if (faulty)
    {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + NodeTextWidth * Zoom());
        ImGui::TextColored(ErrorText, "%s", validation_.detail.c_str());
        ImGui::PopTextWrapPos();
    }
    ImNodes::EndNode();
    if (faulty)
    {
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }
}

void GraphEditorView::TrackDrags()
{
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImNodes::IsEditorHovered())
    {
        document_.BeginMove();
        layoutAtDragStart_ = document_.LayoutJson();
        dragging_ = true;
    }
    // imnodes owns the positions while the user drags; mirror them into the document, at 100 %.
    // After a zoom or a document change, imnodes still holds the old positions until the next
    // frame pushes them: mirroring them now would move every node.
    int const networkIndex = NetworkIndex(network_);
    editor::NetworkDocument const& network = document_.At(network_);
    float const zoom = zoom_[networkIndex];
    for (std::size_t index = 0; index < network.nodes.size() && !positionsStale_[networkIndex]; ++index)
    {
        ImVec2 const position = ImNodes::GetNodeGridSpacePos(static_cast<int>(index) + 1);
        editor::Position const stored = network.nodes[index].position;
        if (position.x != stored.x * zoom || position.y != stored.y * zoom)
        {
            document_.MoveNode(network_, network.nodes[index].name, {position.x / zoom, position.y / zoom});
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && dragging_)
    {
        dirty_ = dirty_ || document_.LayoutJson() != layoutAtDragStart_;
        document_.EndMove();
        dragging_ = false;
    }
}

void GraphEditorView::HandleWiring()
{
    int start = 0;
    int end = 0;
    if (ImNodes::IsLinkCreated(&start, &end))
    {
        int const output = start >= OutputPinBase ? start : end;
        int const input = start >= OutputPinBase ? end : start;
        std::size_t const pin = static_cast<std::size_t>(input - InputPinBase);
        std::size_t const source = static_cast<std::size_t>(output - OutputPinBase);
        editor::NetworkDocument const& network = document_.At(network_);
        if (input >= InputPinBase && input < OutputPinBase && pin < pins_.inputs.size() && source < network.nodes.size())
        {
            std::string const sourceName = network.nodes[source].name;
            std::string const targetName = network.nodes[pins_.inputs[pin].node].name;
            std::string const port = StoredPort(pins_.inputs[pin].port);
            Apply([&]() { document_.Connect(network_, sourceName, targetName, port); });
        }
    }
    int link = 0;
    if (ImNodes::IsLinkDestroyed(&link))
    {
        editor::NetworkDocument const& network = document_.At(network_);
        if (static_cast<std::size_t>(link) < network.edges.size())
        {
            editor::EditorEdge const wire = network.edges[static_cast<std::size_t>(link)];
            Apply([&]() { document_.Disconnect(network_, wire.target, wire.port); });
        }
    }
}

void GraphEditorView::HandleDeletion()
{
    bool const typing = ImGui::GetIO().WantTextInput;
    if (typing || !ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) || !ImGui::IsKeyPressed(ImGuiKey_Delete))
    {
        return;
    }
    editor::NetworkDocument const& network = document_.At(network_);
    std::vector<editor::EditorEdge> wires;
    std::vector<int> links(static_cast<std::size_t>(ImNodes::NumSelectedLinks()));
    if (!links.empty())
    {
        ImNodes::GetSelectedLinks(links.data());
    }
    for (int const link : links)
    {
        if (static_cast<std::size_t>(link) < network.edges.size())
        {
            wires.push_back(network.edges[static_cast<std::size_t>(link)]);
        }
    }
    std::vector<std::string> names;
    std::vector<int> nodes(static_cast<std::size_t>(ImNodes::NumSelectedNodes()));
    if (!nodes.empty())
    {
        ImNodes::GetSelectedNodes(nodes.data());
    }
    for (int const node : nodes)
    {
        if (static_cast<std::size_t>(node - 1) < network.nodes.size())
        {
            names.push_back(network.nodes[static_cast<std::size_t>(node - 1)].name);
        }
    }
    ImNodes::ClearLinkSelection();
    ImNodes::ClearNodeSelection();
    Apply([&]() {
        for (editor::EditorEdge const& wire : wires)
        {
            document_.Disconnect(network_, wire.target, wire.port);
        }
        for (std::string const& name : names)
        {
            document_.RemoveNode(network_, name);
        }
    });
    selected_.clear();
}

void GraphEditorView::DrawInspector()
{
    ImGui::SeparatorText("Inspector");
    editor::NetworkDocument const& network = document_.At(network_);
    auto const node = std::find_if(network.nodes.begin(), network.nodes.end(),
        [this](editor::EditorNode const& candidate) { return candidate.name == selected_; });
    if (node == network.nodes.end())
    {
        ImGui::TextWrapped("%s", ReadOnly() ? "Select a node to see its parameters."
                                            : "Select a node to edit its name and parameters.");
        ImGui::Spacing();
        ImGui::Text("%zu nodes, %zu wires", network.nodes.size(), network.edges.size());
        return;
    }
    editor::BlockInfo const* const info = editor::FindBlock(node->type);
    ImGui::TextUnformatted(info != nullptr ? info->title : node->type.c_str());
    if (info == nullptr && FindComponent(node->type) != nullptr)
    {
        ImGui::PushTextWrapPos(0.F);
        ImGui::TextDisabled("A black-box block of a component library. Its ports and settings come from its "
                            "manifest.");
        ImGui::PopTextWrapPos();
    }
    if (info != nullptr)
    {
        ImGui::PushTextWrapPos(0.F);
        ImGui::TextDisabled("%s", info->help);
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    // Read only: the settings are shown, not editable.
    ImGui::BeginDisabled(ReadOnly());
    DrawNodeEditing(*node);
    ImGui::EndDisabled();
}

void GraphEditorView::DrawNodeEditing(editor::EditorNode const& node)
{
    if (std::string(renameBuffer_.data()) != node.name && !ImGui::IsAnyItemActive())
    {
        CopyToBuffer(renameBuffer_, node.name);
    }
    if (ImGui::InputText("Name", renameBuffer_.data(), renameBuffer_.size(), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        std::string const oldName = node.name;
        std::string const newName = renameBuffer_.data();
        Apply([&]() { document_.RenameNode(network_, oldName, newName); });
        selected_ = message_.empty() ? newName : oldName;
        return;
    }
    HelpMarker("Identifier used in graph.json and in error messages. Press Enter to rename.");
    std::string const shape = ShapeText(validation_, network_, node.name);
    ImGui::Text("Output: %s", shape.empty() ? "(graph not valid yet)" : shape.c_str());
    ImGui::Spacing();
    DrawParameters(node);
    ImGui::Spacing();
    if (ImGui::Button("Remove node"))
    {
        std::string const name = node.name;
        Apply([&]() { document_.RemoveNode(network_, name); });
        selected_.clear();
    }
}

void GraphEditorView::DrawParameters(editor::EditorNode const& node)
{
    editor::BlockInfo const* const info = editor::FindBlock(node.type);
    if (info == nullptr && node.type.rfind("component:", 0) == 0)
    {
        DrawComponentParameters(node);
        return;
    }
    if (info == nullptr)
    {
        ImGui::TextColored(ErrorText, "Unknown block type: remove this node.");
        return;
    }
    std::string const name = node.name;
    auto const setParameter = [&](char const* key, nlohmann::ordered_json const& value) {
        Apply([&]() { document_.SetParameter(network_, name, key, value); });
    };
    bool const hasWidth = info->type == schema::BlockType::Dense || info->type == schema::BlockType::Mlp
        || info->type == schema::BlockType::EntityEncoder;
    if (info->type == schema::BlockType::Input)
    {
        std::string const from = node.parameters.value("from", "");
        if (ImGui::BeginCombo("Group", from.empty() ? "(choose)" : from.c_str()))
        {
            for (std::string const& choice : editor::InputChoices(task_, network_))
            {
                if (ImGui::Selectable(choice.c_str(), choice == from))
                {
                    setParameter("from", choice);
                }
            }
            ImGui::EndCombo();
        }
        HelpMarker("Group of task.json read by this input. A set group gives a masked set of entities.");
    }
    if (info->type == schema::BlockType::Mlp)
    {
        int layers = node.parameters.value("layers", 0);
        if (ImGui::InputInt("Layers", &layers, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue) && layers > 0)
        {
            setParameter("layers", layers);
        }
        HelpMarker("Number of dense layers in the stack.");
    }
    if (hasWidth)
    {
        int width = node.parameters.value("width", 0);
        if (ImGui::InputInt("Width", &width, 8, 32, ImGuiInputTextFlags_EnterReturnsTrue) && width > 0)
        {
            setParameter("width", width);
        }
        HelpMarker("Number of outputs of each layer (per entity for an encoder).");
        std::string const activation = node.parameters.value("activation", "tanh");
        if (ImGui::BeginCombo("Activation", activation.c_str()))
        {
            for (char const* choice : Activations)
            {
                if (ImGui::Selectable(choice, activation == choice))
                {
                    setParameter("activation", choice);
                }
            }
            ImGui::EndCombo();
        }
        HelpMarker("tanh: bounded, smooth. relu: unbounded, sparse. identity: linear.");
    }
    if (info->type == schema::BlockType::MaskedPool)
    {
        std::string const mode = node.parameters.value("mode", "mean");
        if (ImGui::BeginCombo("Mode", mode.c_str()))
        {
            for (char const* choice : PoolModes)
            {
                if (ImGui::Selectable(choice, mode == choice))
                {
                    setParameter("mode", choice);
                }
            }
            ImGui::EndCombo();
        }
        HelpMarker("mean: average situation. max: the most extreme entity on each feature.");
    }
    if (info->type == schema::BlockType::Select)
    {
        std::string const current = FormatIndices(node.parameters.value("indices", nlohmann::ordered_json::array()));
        if (std::string(indicesBuffer_.data()) != current && !ImGui::IsAnyItemActive())
        {
            CopyToBuffer(indicesBuffer_, current);
        }
        if (ImGui::InputText("Indices", indicesBuffer_.data(), indicesBuffer_.size(),
                ImGuiInputTextFlags_EnterReturnsTrue))
        {
            std::optional<nlohmann::ordered_json> const indices = ParseIndices(indicesBuffer_.data());
            if (indices)
            {
                setParameter("indices", *indices);
            }
            else
            {
                message_ = "Indices: write non-negative integers separated by commas, e.g. 0, 3, 4";
            }
        }
        HelpMarker("Features kept from the input vector, in output order, e.g. 0, 3, 4. Press Enter to apply.");
    }
    if (info->type == schema::BlockType::MaskedAttention)
    {
        DrawAttentionKeys(node);
    }
    if (info->type == schema::BlockType::GaussianHead)
    {
        DrawLogStd(node);
    }
    if (network_ == Network::Critic && hasWidth)
    {
        DrawShare(node);
    }
}

void GraphEditorView::DrawAttentionKeys(editor::EditorNode const& node)
{
    std::string const name = node.name;
    bool keys = FlagParameter(node, "keys");
    if (ImGui::Checkbox("Separate keys", &keys))
    {
        nlohmann::ordered_json const value = keys ? nlohmann::ordered_json(true) : nlohmann::ordered_json();
        Apply([&]() { document_.SetParameter(network_, name, "keys", value); });
    }
    HelpMarker("Adds a keys port (a set of the same shape as set): the scores compare the query with the keys, "
               "the output averages the set. Off: the set is its own keys.");
}

void GraphEditorView::DrawLogStd(editor::EditorNode const& node)
{
    std::string const name = node.name;
    std::string const mode = TextParameter(node, "log_std", "parameter");
    if (ImGui::BeginCombo("Log-std", mode.c_str()))
    {
        for (char const* choice : LogStdModes)
        {
            if (ImGui::Selectable(choice, mode == choice) && mode != choice)
            {
                bool const state = std::string(choice) == "state";
                nlohmann::ordered_json changes;
                changes["log_std"] = state ? nlohmann::ordered_json("state") : nlohmann::ordered_json();
                changes["log_std_min"] = state ? nlohmann::ordered_json(DefaultLogStdMinimum) : nlohmann::ordered_json();
                changes["log_std_max"] = state ? nlohmann::ordered_json(DefaultLogStdMaximum) : nlohmann::ordered_json();
                Apply([&]() { document_.SetParameters(network_, name, changes); });
            }
        }
        ImGui::EndCombo();
    }
    HelpMarker("parameter: one learned log-std per action, independent of the state (initial value in training). "
               "state: a dense layer of the head input gives the log-std, clamped to [min, max].");
    if (mode != "state")
    {
        return;
    }
    float minimum = NumberParameter(node, "log_std_min", DefaultLogStdMinimum);
    float maximum = NumberParameter(node, "log_std_max", DefaultLogStdMaximum);
    if (ImGui::InputFloat("Log-std min", &minimum, 0.F, 0.F, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue))
    {
        Apply([&]() { document_.SetParameter(network_, name, "log_std_min", minimum); });
    }
    if (ImGui::InputFloat("Log-std max", &maximum, 0.F, 0.F, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue))
    {
        Apply([&]() { document_.SetParameter(network_, name, "log_std_max", maximum); });
    }
    HelpMarker("Clamp of the log-std; no gradient flows outside it. Min must be below max. Press Enter to apply.");
}

void GraphEditorView::DrawShare(editor::EditorNode const& node)
{
    std::string const name = node.name;
    std::string const share = TextParameter(node, "share", "");
    if (ImGui::BeginCombo("Share", share.empty() ? "(own parameters)" : share.c_str()))
    {
        if (ImGui::Selectable("(own parameters)", share.empty()) && !share.empty())
        {
            Apply([&]() { document_.SetParameter(network_, name, "share", nlohmann::ordered_json()); });
        }
        for (editor::EditorNode const& candidate : document_.At(Network::Actor).nodes)
        {
            if (candidate.type != node.type)
            {
                continue;
            }
            std::string const choice = candidate.name;
            if (ImGui::Selectable(choice.c_str(), choice == share) && choice != share)
            {
                Apply([&]() { document_.SetParameter(network_, name, "share", choice); });
            }
        }
        ImGui::EndCombo();
    }
    HelpMarker("Uses the parameters of an actor layer of the same type, sizes, activation and inputs. Actor and "
               "critic then train with one optimizer: critic learning rate = actor learning rate, value loss "
               "weighted by value_coefficient.");
}

void GraphEditorView::DrawStatus()
{
    ImGui::Separator();
    if (validation_.valid)
    {
        ImGui::TextColored(ValidText, "Graph valid: it trains as drawn.");
    }
    else
    {
        ImGui::TextColored(ErrorText, "%s", validation_.message.c_str());
        bool const onNode = validation_.network && !validation_.node.empty();
        if (onNode)
        {
            ImGui::SameLine();
            if (ImGui::SmallButton("Show"))
            {
                network_ = *validation_.network;
                selected_ = validation_.node;
                ImNodes::EditorContextSet(contexts_[NetworkIndex(network_)]);
                auto const& nodes = document_.At(network_).nodes;
                auto const node = std::find_if(nodes.begin(), nodes.end(),
                    [this](editor::EditorNode const& candidate) { return candidate.name == selected_; });
                if (node != nodes.end())
                {
                    ImNodes::EditorContextMoveToNode(static_cast<int>(node - nodes.begin()) + 1);
                }
            }
        }
    }
    if (!message_.empty())
    {
        ImGui::TextDisabled("%s", message_.c_str());
    }
}

void GraphEditorView::HandleShortcuts()
{
    if (ImGui::GetIO().WantTextInput)
    {
        return; // Ctrl+Z in a text field edits the text, not the graph.
    }
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Z))
    {
        Apply([this]() { document_.Undo(); });
    }
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Y))
    {
        Apply([this]() { document_.Redo(); });
    }
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S) && !files_.graph.empty())
    {
        Save();
    }
}

bool GraphEditorView::Save()
{
    if (ReadOnly())
    {
        message_ = "Save refused: stop the active run before changing network.json.";
        return false;
    }
    try
    {
        std::string const graph = document_.GraphJson();
        std::string const layout = document_.LayoutJson();
        // Commit the graphical sidecar first. If the semantic graph write fails, training
        // still sees its previous compatible graph; a newer layout is harmless.
        artifacts::WriteFileAtomically(
            files_.layout, [&layout](std::ostream& out) { out << layout; }, artifacts::AtomicWriteOptions{},
            "PPOKit studio layout");
        artifacts::WriteFileAtomically(
            files_.graph, [&graph](std::ostream& out) { out << graph; }, artifacts::AtomicWriteOptions{},
            "PPOKit studio graph");
        dirty_ = false;
        message_ = "Saved " + files_.graph.filename().string()
            + (validation_.valid ? "" : " (not valid yet: training will refuse it)");
        return true;
    }
    catch (std::exception const& error)
    {
        message_ = std::string("Save failed: ") + error.what();
        return false;
    }
}

void GraphEditorView::Changed()
{
    validation_ = document_.Validate(task_);
    dirty_ = true;
    positionsStale_ = {true, true};
}

template<typename Operation>
void GraphEditorView::Apply(Operation&& operation)
{
    try
    {
        operation();
        message_.clear();
        Changed();
    }
    catch (std::invalid_argument const& error)
    {
        message_ = error.what();
    }
}

std::vector<std::string> GraphEditorView::InputPortsOf(editor::EditorNode const& node) const
{
    std::optional<schema::BlockType> const type = schema::BlockTypeFromName(node.type);
    if (!type)
    {
        component::ComponentRef const* const component = FindComponent(node.type);
        return component != nullptr ? schema::ComponentInputPorts(*component->manifest) : std::vector<std::string>{};
    }
    if (!schema::HasNumberedPorts(*type))
    {
        bool const separateKeys = FlagParameter(node, "keys");
        return schema::BlockInputPorts(*type, 0, separateKeys);
    }
    // Every wired port (even with gaps, which the validation reports), then one free port
    // so that a new input can be dropped on it.
    std::vector<std::pair<std::size_t, std::string>> wired;
    for (editor::EditorEdge const& edge : document_.At(network_).edges)
    {
        bool const numbered = edge.target == node.name && edge.port.size() > 2 && edge.port.rfind("in", 0) == 0
            && edge.port.find_first_not_of("0123456789", 2) == std::string::npos && edge.port.size() <= 8;
        if (numbered)
        {
            wired.emplace_back(std::stoul(edge.port.substr(2)), edge.port);
        }
    }
    std::sort(wired.begin(), wired.end());
    std::vector<std::string> ports;
    for (auto const& [number, port] : wired)
    {
        ports.push_back(port);
    }
    ports.push_back(document_.NextNumberedPort(network_, node.name));
    return ports;
}

component::ComponentRef const* GraphEditorView::FindComponent(std::string const& type) const
{
    for (editor::ComponentLibrary const& library : libraries_)
    {
        for (component::ComponentRef const& candidate : library.components)
        {
            if (editor::ComponentType(candidate) == type)
            {
                return &candidate;
            }
        }
    }
    return nullptr;
}

void GraphEditorView::RefreshLibraries()
{
    if (document_.ComponentReferences() == librariesFor_)
    {
        return;
    }
    librariesFor_ = document_.ComponentReferences();
    libraries_ = document_.ComponentLibraries();
}

void GraphEditorView::DrawComponentPalette()
{
    ImGui::SeparatorText("Components");
    std::string removed;
    for (editor::ComponentLibrary const& library : libraries_)
    {
        ImGui::PushID(library.reference.c_str());
        ImGui::TextDisabled("%s", library.reference.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("x"))
        {
            removed = library.reference;
        }
        ImGui::SetItemTooltip("Remove this library from graph.json (its nodes stay, and are reported)");
        if (!library.error.empty())
        {
            ImGui::PushTextWrapPos(0.F);
            ImGui::TextColored(ErrorText, "%s", library.error.c_str());
            ImGui::PopTextWrapPos();
        }
        for (component::ComponentRef const& component : library.components)
        {
            if (!OffersComponent(network_, component))
            {
                continue;
            }
            std::string const label = "  " + component.Id();
            if (ImGui::Selectable(label.c_str()))
            {
                editor::Position const position = NewNodePosition();
                Apply([&]() { selected_ = document_.AddComponentNode(network_, component, position); });
            }
            if (ImGui::BeginItemTooltip())
            {
                component::Capabilities const& capabilities = component.manifest->capabilities;
                ImGui::Text("Inputs: %zu, settings: %zu", schema::ComponentInputPorts(*component.manifest).size(),
                    component.manifest->config.size());
                ImGui::TextUnformatted(ComponentRole(capabilities));
                ImGui::EndTooltip();
            }
        }
        ImGui::PopID();
    }
    if (!removed.empty())
    {
        Apply([&]() { document_.RemoveComponentLibrary(removed); });
    }
    DrawProjectLibraries();
    if (NativeDialogsAvailable() && !componentFolder_.empty())
    {
        if (ImGui::Button("Add a DLL..."))
        {
            if (std::optional<std::filesystem::path> const library = PickComponentLibrary(componentFolder_))
            {
                UseLibrary(*library);
            }
        }
        ItemTooltip("Choose a component library: it is copied into the project's components/ folder, and its blocks "
                    "appear above, ready to add and wire.");
        return;
    }
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 3.F);
    bool const entered = ImGui::InputTextWithHint("##library", "library name or path", libraryBuffer_.data(),
        libraryBuffer_.size(), ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((ImGui::SmallButton("Add") || entered) && libraryBuffer_[0] != '\0')
    {
        std::string const reference = libraryBuffer_.data();
        Apply([&]() { document_.AddComponentLibrary(reference); });
        libraryBuffer_[0] = '\0';
    }
    ItemTooltip("A component DLL, by name (trackfilter finds trackfilter.dll or libtrackfilter.so) or path, relative to "
                "network.json. Put it in the project's components/ folder to have it offered above.");
}

void GraphEditorView::SetComponentFolder(std::filesystem::path folder)
{
    componentFolder_ = std::move(folder);
    ScanComponentFolder();
}

void GraphEditorView::ScanComponentFolder()
{
    componentFiles_.clear();
    std::error_code error;
    for (auto const& entry : std::filesystem::directory_iterator(componentFolder_, error))
    {
        std::string const extension = entry.path().extension().string();
        if (entry.is_regular_file(error) && (extension == ".dll" || extension == ".so" || extension == ".dylib"))
        {
            componentFiles_.push_back(entry.path());
        }
    }
    std::sort(componentFiles_.begin(), componentFiles_.end());
}

void GraphEditorView::DrawProjectLibraries()
{
    if (componentFolder_.empty())
    {
        return;
    }
    std::vector<std::string> const& used = document_.ComponentReferences();
    bool offered = false;
    for (std::filesystem::path const& file : componentFiles_)
    {
        std::string const reference = "components/" + file.filename().string();
        if (std::find(used.begin(), used.end(), reference) != used.end())
        {
            continue;
        }
        offered = true;
        ImGui::PushID(reference.c_str());
        ImGui::TextUnformatted(file.filename().string().c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Use"))
        {
            UseLibrary(file);
        }
        ItemTooltip("Load this library: its components appear in the list above.");
        ImGui::PopID();
    }
    if (componentFiles_.empty())
    {
        MutedWrapped("No library in components/ yet.");
    }
    else if (!offered && libraries_.empty())
    {
        MutedWrapped("Every library of components/ is in use.");
    }
    if (ImGui::SmallButton("Rescan components/"))
    {
        ScanComponentFolder();
    }
    ItemTooltip("Look again for libraries copied into the project's components/ folder.");
    if (!componentError_.empty())
    {
        ImGui::PushTextWrapPos(0.F);
        ImGui::TextColored(ErrorText, "%s", componentError_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void GraphEditorView::UseLibrary(std::filesystem::path const& library)
{
    componentError_.clear();
    std::filesystem::path const target = componentFolder_ / library.filename();
    std::error_code error;
    if (!std::filesystem::equivalent(library, target, error))
    {
        std::filesystem::create_directories(componentFolder_, error);
        std::filesystem::copy_file(library, target, std::filesystem::copy_options::overwrite_existing, error);
        if (error)
        {
            componentError_ = "Cannot copy " + library.string() + " into components/: " + error.message();
            return;
        }
    }
    std::string const reference = "components/" + library.filename().string();
    Apply([&]() { document_.AddComponentLibrary(reference); });
    ScanComponentFolder();
}

void GraphEditorView::DrawComponentParameters(editor::EditorNode const& node)
{
    component::ComponentRef const* const component = FindComponent(node.type);
    if (component == nullptr)
    {
        ImGui::PushTextWrapPos(0.F);
        ImGui::TextColored(ErrorText, "No library of this graph provides %s: add its library in the palette.",
            node.type.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    std::string const name = node.name;
    nlohmann::ordered_json config = node.parameters.value("config", nlohmann::ordered_json::object());
    for (component::SettingSpec const& setting : component->manifest->config)
    {
        ImGui::PushID(setting.name.c_str());
        bool changed = false;
        if (setting.type == component::SettingSpec::Type::Bool)
        {
            bool value = config.value(setting.name, setting.defaultValue != 0.0);
            changed = ImGui::Checkbox(setting.name.c_str(), &value);
            config[setting.name] = value;
        }
        else
        {
            double value = config.value(setting.name, setting.defaultValue);
            char const* const format = setting.type == component::SettingSpec::Type::Int ? "%.0f" : "%.6g";
            ImGui::InputScalar(setting.name.c_str(), ImGuiDataType_Double, &value, nullptr, nullptr, format);
            // Written once the field is left (Enter, Tab or a click elsewhere), not at every keystroke.
            changed = ImGui::IsItemDeactivatedAfterEdit();
            value = std::clamp(value, setting.minimum, setting.maximum);
            if (setting.type == component::SettingSpec::Type::Int)
            {
                value = std::round(value);
            }
            config[setting.name] = value;
        }
        ImGui::SetItemTooltip("Default %g, range [%g, %g]. Type a value, then Enter.", setting.defaultValue,
            setting.minimum, setting.maximum);
        if (changed)
        {
            Apply([&]() { document_.SetParameter(network_, name, "config", config); });
        }
        ImGui::PopID();
    }
    if (component->manifest->config.empty())
    {
        ImGui::TextDisabled("No setting.");
    }
}

} // namespace ppokit::studio
