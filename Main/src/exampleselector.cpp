#include "exampleselector.hpp"

#include <imgui.h>
#include <raylib.h>
#include <rlImGui.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ppo::app {
namespace {

void DrawLabel(ImDrawList& draw, ImVec2 position, float size,
               ImU32 color, std::string_view value, float wrapWidth = 0.F)
{
    if (value.empty())
        return;
    draw.AddText(ImGui::GetFont(), size, position, color,
                 value.data(), value.data() + value.size(), wrapWidth);
}

} // namespace

std::optional<std::string> SelectExample(ExampleCatalog const& catalog)
{
    // This window is independent of the simulation and its renderer.
    // Fully release raylib/ImGui before EnvironmentRenderer creates its window.
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1080, 720, "PPOTrial - Select example");
    if (!IsWindowReady())
        return std::nullopt;

    SetWindowMinSize(680, 460);
    SetTargetFPS(60);
    rlImGuiSetup(true);
    ImGui::StyleColorsDark();

    auto& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2{18.F, 18.F};

    std::optional<std::string> selection;
    std::size_t highlighted = 0;
    auto const& entries = catalog.Entries();

    while (!WindowShouldClose() && !selection)
    {
        if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_RIGHT))
            highlighted = (highlighted + 1U) % entries.size();
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_LEFT))
            highlighted = (highlighted + entries.size() - 1U) % entries.size();
        if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER))
            selection = std::string(entries[highlighted].id);

        BeginDrawing();
        ClearBackground(Color{13, 19, 30, 255});
        rlImGuiBegin();

        float const screenWidth = static_cast<float>(GetScreenWidth());
        float const screenHeight = static_cast<float>(GetScreenHeight());
        ImGui::SetNextWindowPos(ImVec2{0.F, 0.F}, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2{screenWidth, screenHeight}, ImGuiCond_Always);
        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.05F, 0.07F, 0.11F, 1.F});

        if (ImGui::Begin("##example-selector", nullptr, flags))
        {
            float const width = std::max(300.F, std::min(880.F, screenWidth - 72.F));
            float const left = std::max(18.F, (screenWidth - width) * 0.5F);
            auto* draw = ImGui::GetWindowDrawList();

            ImGui::Dummy(ImVec2{0.F, std::clamp(screenHeight * 0.08F, 20.F, 70.F)});
            ImGui::SetCursorPosX(left);
            auto const heading = ImGui::GetCursorScreenPos();
            DrawLabel(*draw, heading, 18.F, IM_COL32(129, 180, 255, 255),
                      "PPOTrial  /  EXAMPLES");
            ImGui::Dummy(ImVec2{width, 36.F});

            ImGui::SetCursorPosX(left);
            auto const titlePosition = ImGui::GetCursorScreenPos();
            DrawLabel(*draw, titlePosition, 36.F, IM_COL32(247, 250, 255, 255),
                      "Choose a simulation");
            ImGui::Dummy(ImVec2{width, 58.F});

            ImGui::SetCursorPosX(left);
            auto const subtitlePosition = ImGui::GetCursorScreenPos();
            DrawLabel(*draw, subtitlePosition, 18.F, IM_COL32(172, 186, 207, 255),
                      "Select one of the compiled examples to get started.");
            ImGui::Dummy(ImVec2{width, 46.F});

            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                auto const& example = entries[i];
                ImGui::SetCursorPosX(left);
                ImVec2 const topLeft = ImGui::GetCursorScreenPos();
                ImVec2 const bottomRight{topLeft.x + width, topLeft.y + 124.F};
                ImGui::PushID(static_cast<int>(i));
                bool const pressed = ImGui::InvisibleButton("##example-card",
                                                            ImVec2{width, 124.F});
                bool const hovered = ImGui::IsItemHovered();
                ImGui::PopID();

                if (hovered)
                    highlighted = i;
                if (pressed)
                    selection = std::string(example.id);

                bool const active = highlighted == i;
                draw->AddRectFilled(topLeft, bottomRight,
                    active ? IM_COL32(33, 53, 83, 255) : IM_COL32(25, 34, 49, 255),
                    12.F);
                draw->AddRect(topLeft, bottomRight,
                    active ? IM_COL32(114, 181, 255, 255) : IM_COL32(67, 83, 105, 255),
                    12.F, 0, active ? 2.F : 1.F);

                DrawLabel(*draw, ImVec2{topLeft.x + 24.F, topLeft.y + 15.F},
                          26.F, IM_COL32(248, 251, 255, 255), example.windowTitle);
                DrawLabel(*draw, ImVec2{topLeft.x + 24.F, topLeft.y + 53.F},
                          17.F, IM_COL32(181, 201, 223, 255),
                          example.description.empty() ? example.id : example.description,
                          width - 48.F);
                DrawLabel(*draw, ImVec2{topLeft.x + 24.F, topLeft.y + 99.F},
                          14.F, IM_COL32(135, 184, 247, 255), example.id);
                ImGui::Dummy(ImVec2{0.F, 14.F});
            }

            ImGui::SetCursorPosX(left);
            auto const footer = ImGui::GetCursorScreenPos();
            DrawLabel(*draw, footer, 16.F, IM_COL32(164, 176, 195, 255),
                      "Click to launch   |   Arrow keys to select   |   Enter to launch   |   Esc to exit");
            ImGui::Dummy(ImVec2{width, 32.F});
        }

        ImGui::End();
        ImGui::PopStyleColor();
        rlImGuiEnd();
        EndDrawing();
    }

    rlImGuiShutdown();
    CloseWindow();
    return selection;
}

} // namespace ppo::app
