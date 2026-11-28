/**
 * @file PPOKit/studio/main.cpp
 * @brief PPOKit studio: open or create a project folder, then design, train, evaluate and
 *        run it.
 *
 * Usage: ppokit-studio                                   start screen
 *        ppokit-studio <project folder>                  open that project
 *        ppokit-studio --check <task.json> <network.json> validate a network, no window
 *
 * --check exits with 0 when the network trains as written (used by the tests).
 */
#include "graph_editor_view.hpp"
#include "studio_app.hpp"
#include "studio_theme.hpp"

#include "ppokit/component/native_module.hpp"
#include "ppokit/editor/recent_projects.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <imgui.h>
#include <imnodes.h>
#include <implot.h>
#include <raylib.h>
#include <rlImGui.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifndef PPOKIT_PROGRAM_REGISTRY
#define PPOKIT_PROGRAM_REGISTRY ""
#endif

namespace {

int Check(std::filesystem::path const& taskPath, std::filesystem::path const& networkPath)
{
    auto const task = ppokit::schema::LoadTaskSchema(taskPath);
    auto const validation = ppokit::studio::OpenGraphDocument(networkPath, {}).Validate(task);
    std::cout << (validation.valid ? "valid: " + networkPath.string() : validation.message) << '\n';
    return validation.valid ? 0 : 1;
}

/** @brief Opens the window and the ImGui, imnodes and ImPlot contexts, in the studio's look. */
void OpenWindow()
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1500, 920, "PPOKit studio");
    SetTargetFPS(60);
    // Escape must not close the studio with unsaved work: only the window button does.
    SetExitKey(KEY_NULL);
    rlImGuiBeginInitImGui();
    ppokit::studio::LoadStudioFonts(GetWindowScaleDPI().y);
    rlImGuiEndInitImGui();
    ImNodes::CreateContext();
    ImPlot::CreateContext();
    ppokit::studio::ApplyStudioTheme();
}

void CloseWindowAndContexts()
{
    ImPlot::DestroyContext();
    ImNodes::DestroyContext();
    rlImGuiShutdown();
    CloseWindow();
}

/** @brief Programs the projects may launch: those the build registered, and those beside the studio. */
std::vector<std::filesystem::path> KnownPrograms()
{
    std::vector<std::filesystem::path> programs = ppokit::schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY);
    for (std::filesystem::path& program : ppokit::studio::ProgramsIn(GetApplicationDirectory()))
    {
        programs.push_back(std::move(program));
    }
    return programs;
}

/** @brief Frames until the window closes; closing with unsaved changes asks first. */
void FrameLoop(ppokit::studio::StudioApp& app)
{
    while (!app.QuitConfirmed())
    {
        if (WindowShouldClose())
        {
            app.RequestQuit();
        }
        BeginDrawing();
        ClearBackground(Color{7, 10, 16, 255});
        rlImGuiBegin();
        ImGui::SetNextWindowPos(ImVec2{0.F, 0.F}, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2{static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight())},
            ImGuiCond_Always);
        ImGui::Begin("##ppokit-studio", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        app.Draw();
        ImGui::End();
        rlImGuiEnd();
        EndDrawing();
    }
}

int RunStudio(std::filesystem::path const* project)
{
    OpenWindow();
    {
        std::vector<std::filesystem::path> programs = KnownPrograms();
        // The Visualiseurs a network.json names are found among the libraries of the build.
        ppokit::component::SetKnownModules(programs);
        ppokit::studio::StudioApp app(ppokit::editor::DefaultRecentProjectsFile(), std::move(programs));
        if (project != nullptr)
        {
            app.OpenProject(*project);
        }
        FrameLoop(app);
    }
    CloseWindowAndContexts();
    return 0;
}

void PrintUsage()
{
    std::cerr << "usage: ppokit-studio                                    start screen\n"
                 "       ppokit-studio <project folder>                   open that project\n"
                 "       ppokit-studio --check <task.json> <network.json> validate a network, no window\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1)
        {
            return RunStudio(nullptr);
        }
        std::string const first = argv[1];
        if (argc == 4 && first == "--check")
        {
            return Check(argv[2], argv[3]);
        }
        if (argc == 2 && first.rfind("--", 0) != 0)
        {
            std::filesystem::path const project = first;
            return RunStudio(&project);
        }
        PrintUsage();
        return 2;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ppokit-studio: " << error.what() << '\n';
        return 1;
    }
}
