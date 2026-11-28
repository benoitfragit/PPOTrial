/**
 * @file PPOKit/studio/tests/graph_editor_view_tests.cpp
 * @brief The imnodes editor view, run headless: real ImGui and imnodes frames without a
 *        renderer. Draws the Blue graph (every block kind but Bernoulli), a broken graph,
 *        both networks, keyboard undo/redo and saving.
 */
#include "graph_editor_view.hpp"

#include <imgui.h>
#include <imnodes.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using namespace ppokit;
using editor::Network;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::string ReadFile(std::filesystem::path const& path)
{
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

/** @brief ImGui and imnodes contexts for headless frames; the font atlas is built in memory. */
class HeadlessImGui final
{
public:
    HeadlessImGui()
    {
        ImGui::CreateContext();
        ImNodes::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(1600.F, 900.F);
        io.DeltaTime = 1.F / 60.F;
        io.IniFilename = nullptr;
        // Like the studio renderer, so that fonts are rebuilt at every zoom size.
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    }
    ~HeadlessImGui()
    {
        ImNodes::DestroyContext();
        ImGui::DestroyContext();
    }
    HeadlessImGui(HeadlessImGui const&) = delete;
    HeadlessImGui& operator=(HeadlessImGui const&) = delete;

    /** @brief One full frame of the editor in a full-screen window. */
    static void Frame(studio::GraphEditorView& view)
    {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.F, 0.F));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("PPOKit studio", nullptr, ImGuiWindowFlags_NoDecoration);
        view.Draw();
        ImGui::End();
        ImGui::Render();
        AcceptTextures(*ImGui::GetDrawData());
    }

    /** @brief Plays the renderer: every texture ImGui asks for is accepted, none is uploaded. */
    static void AcceptTextures(ImDrawData const& data)
    {
        if (data.Textures == nullptr)
        {
            return;
        }
        for (ImTextureData* texture : *data.Textures)
        {
            if (texture->Status == ImTextureStatus_WantCreate)
            {
                texture->SetTexID(static_cast<ImTextureID>(texture->UniqueID + 1));
                texture->SetStatus(ImTextureStatus_OK);
            }
            else if (texture->Status == ImTextureStatus_WantUpdates)
            {
                texture->SetStatus(ImTextureStatus_OK);
            }
            else if (texture->Status == ImTextureStatus_WantDestroy)
            {
                texture->SetTexID(ImTextureID_Invalid);
                texture->SetStatus(ImTextureStatus_Destroyed);
            }
        }
    }

    /** @brief Presses Ctrl + `key` for one frame, then releases it. */
    static void Chord(studio::GraphEditorView& view, ImGuiKey key)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiMod_Ctrl, true);
        io.AddKeyEvent(key, true);
        Frame(view);
        io.AddKeyEvent(key, false);
        io.AddKeyEvent(ImGuiMod_Ctrl, false);
        Frame(view);
    }
};

void DrawsTheBlueGraph(HeadlessImGui& imgui, schema::TaskSchema const& task)
{
    auto document = editor::GraphDocument::Open(ReadFile(PPOKIT_BLUE_GRAPH_BLOCKS));
    studio::GraphEditorView view(task, std::move(document), {});
    for (int frame = 0; frame < 3; ++frame)
    {
        imgui.Frame(view);
    }
    view.SelectNetwork(Network::Critic);
    for (int frame = 0; frame < 2; ++frame)
    {
        imgui.Frame(view);
    }
    Check(view.LastValidation().valid, "the Blue graph is drawn valid");
    Check(!view.Dirty(), "drawing does not modify the graph");
}

/** @brief The Blue graph opens fitted (below 100 %), zooms around a point, stays unmodified and fits again. */
void ZoomsAndFits(HeadlessImGui& imgui, schema::TaskSchema const& task)
{
    auto document = editor::GraphDocument::Open(ReadFile(PPOKIT_BLUE_GRAPH_BLOCKS));
    studio::GraphEditorView view(task, std::move(document), {});
    for (int frame = 0; frame < 5; ++frame)
    {
        imgui.Frame(view);
    }
    float const fitted = view.Zoom();
    std::string const layout = view.Document().LayoutJson();
    Check(fitted < 1.F && fitted >= 0.3F, "the Blue actor opens fitted to the canvas, below 100 %");
    view.ZoomCanvas(1.5F, ImVec2(100.F, 100.F));
    imgui.Frame(view);
    Check(std::abs(view.Zoom() - fitted * 1.5F) < 1e-4F, "zooming multiplies the zoom");
    view.ZoomCanvas(100.F, ImVec2(0.F, 0.F));
    Check(view.Zoom() == 2.5F, "the zoom stops at 250 %");
    imgui.Frame(view);
    view.FitCanvas();
    for (int frame = 0; frame < 4; ++frame)
    {
        imgui.Frame(view); // The fit waits for the node sizes to settle.
    }
    // Text sizes round differently at 250 %: the measured network differs by a few pixels.
    Check(std::abs(view.Zoom() - fitted) < 0.02F * fitted, "Fit comes back to the fitted zoom");
    Check(!view.Dirty(), "zooming does not modify the graph");
    Check(view.Document().LayoutJson() == layout, "zooming keeps the saved positions (at 100 %)");
}

/** @brief Nodes placed automatically (no layout file) do not overlap in a column. */
void PlacesWithoutOverlap()
{
    auto const document = editor::GraphDocument::Open(ReadFile(PPOKIT_BLUE_GRAPH_BLOCKS));
    std::vector<editor::EditorNode> const& nodes = document.At(Network::Actor).nodes;
    for (std::size_t first = 0; first < nodes.size(); ++first)
    {
        for (std::size_t second = first + 1; second < nodes.size(); ++second)
        {
            bool const sameColumn = nodes[first].position.x == nodes[second].position.x;
            float const distance = std::abs(nodes[first].position.y - nodes[second].position.y);
            Check(!sameColumn || distance >= 140.F,
                nodes[first].name + " and " + nodes[second].name + " are at least one node apart");
        }
    }
}

void DrawsErrors(HeadlessImGui& imgui, schema::TaskSchema const& task)
{
    auto document = editor::GraphDocument::Open(R"({
      "schema": "ppokit.graph/2",
      "actor": { "nodes": { "own": { "type": "input", "from": "observation.own" },
                            "joined": { "type": "concat" },
                            "odd": { "type": "transformer" } },
                 "edges": [ ["own", "joined.in1"], ["own", "joined.in3"], ["own", "missing"] ] }
    })");
    studio::GraphEditorView view(task, std::move(document), {});
    for (int frame = 0; frame < 3; ++frame)
    {
        imgui.Frame(view);
    }
    Check(!view.LastValidation().valid && !view.LastValidation().message.empty(),
        "a broken graph is drawn with its error");
}

void KeyboardUndoRedo(HeadlessImGui& imgui, schema::TaskSchema const& task)
{
    editor::GraphDocument document;
    document.AddNode(Network::Actor, schema::BlockType::Dense, {0, 0});
    studio::GraphEditorView view(task, std::move(document), {});
    imgui.Frame(view);
    Check(view.Document().At(Network::Actor).nodes.size() == 1, "the document is shown as given");
    imgui.Chord(view, ImGuiKey_Z);
    Check(view.Document().At(Network::Actor).nodes.empty(), "Ctrl+Z undoes");
    imgui.Chord(view, ImGuiKey_Y);
    Check(view.Document().At(Network::Actor).nodes.size() == 1 && view.Dirty(), "Ctrl+Y redoes and marks dirty");
}

void Saves(HeadlessImGui& imgui, schema::TaskSchema const& task)
{
    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_studio_view_tests";
    std::filesystem::remove_all(folder);
    std::filesystem::create_directories(folder);
    std::filesystem::path const graphPath = folder / "graph_blocks.json";
    std::string const original = ReadFile(PPOKIT_BLUE_GRAPH_BLOCKS);
    auto document = editor::GraphDocument::Open(original);
    studio::GraphEditorView view(task, std::move(document), {graphPath, {}});
    imgui.Frame(view);
    imgui.Chord(view, ImGuiKey_S);
    Check(std::filesystem::exists(graphPath) && std::filesystem::exists(folder / "graph_blocks.layout.json"),
        "Ctrl+S writes the graph and its layout");
    Check(!view.Dirty(), "saving clears the dirty flag");
    auto const reopened = editor::GraphDocument::Open(ReadFile(graphPath), ReadFile(folder / "graph_blocks.layout.json"));
    Check(reopened.GraphJson() == view.Document().GraphJson(), "the saved graph reopens identical");
    Check(reopened.Validate(task).valid, "the saved graph is valid");
    std::filesystem::remove_all(folder);
}

} // namespace

#ifdef PPOKIT_TEST_COMPONENTS
/** @brief A graph with a component: its node, its settings in the inspector, the library in the palette. */
void DrawsComponents(HeadlessImGui& imgui)
{
    auto const task = schema::LoadTaskSchema(PPOKIT_CORRIDOR_TASK);
    std::string const graph = R"({ "schema": "ppokit.graph/2", "components": ["ppokit_test_components"],
      "actor": { "nodes": { "state": { "type": "input", "from": "observation.state" },
                            "scaled": { "type": "component:scale@1", "config": { "initial": 2 } },
                            "trunk": { "type": "mlp", "layers": 1, "width": 8, "activation": "tanh" },
                            "policy": { "type": "gaussian_head" } },
                 "edges": [ ["state", "scaled"], ["scaled", "trunk"], ["trunk", "policy"] ] },
      "critic": { "nodes": { "state": { "type": "input", "from": "observation.state" },
                             "value": { "type": "value" } },
                  "edges": [ ["state", "value"] ] } })";
    auto document = editor::GraphDocument::Open(graph);
    document.SetDirectory(std::filesystem::path(PPOKIT_TEST_COMPONENTS).parent_path());
    studio::GraphEditorView view(task, std::move(document), {});
    view.SelectNode("scaled");
    for (int frame = 0; frame < 3; ++frame)
    {
        imgui.Frame(view);
    }
    Check(view.LastValidation().valid, "a graph with a component is drawn valid: " + view.LastValidation().message);
    Check(!view.Dirty(), "drawing a component node does not modify the graph");
}
#endif

int main()
{
    try
    {
        HeadlessImGui imgui;
        auto const blue = schema::LoadTaskSchema(PPOKIT_BLUE_TASK);
        DrawsTheBlueGraph(imgui, blue);
        ZoomsAndFits(imgui, blue);
        PlacesWithoutOverlap();
        DrawsErrors(imgui, blue);
        KeyboardUndoRedo(imgui, blue);
        Saves(imgui, blue);
#ifdef PPOKIT_TEST_COMPONENTS
        DrawsComponents(imgui);
#endif
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " editor view check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit studio editor view tests passed\n";
    return 0;
}
