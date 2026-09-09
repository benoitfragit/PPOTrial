#include "environmentrenderer.hpp"
#include "imgui_internal.h"
#include "implot.h"
#include "rlImGui.h"
#include "raymath.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ppo::renderer {
namespace {
Vector3 World(navigation::Point point, float height = 0)
{ return {static_cast<float>(point.x), height, static_cast<float>(point.y)}; }
std::string EntityName(int index)
{
    if (index == 0) return "Mobile";
    if (index == 1) return "Destination";
    return "Forbidden zone " + std::to_string(index - 1);
}
}
EnvironmentRenderer::EnvironmentRenderer(navigation::Application& application) : m_application(application)
{
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1600, 1000, "PLAYGROUND");
    SetTargetFPS(60);
    rlImGuiSetup(true);
    ImPlot::CreateContext();
    ApplyTheme();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    m_camera.position = {0, 12, 12};
    m_camera.target = {0, 0, 0};
    m_camera.up = {0, 1, 0};
    m_camera.fovy = 45;
    m_camera.projection = CAMERA_PERSPECTIVE;
    LoadSettings();
}
EnvironmentRenderer::~EnvironmentRenderer()
{
    if (m_texture.id) UnloadRenderTexture(m_texture);
    ImPlot::DestroyContext();
    rlImGuiShutdown();
    CloseWindow();
}
void
EnvironmentRenderer::ApplyTheme(void)
{
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.F;
    style.ChildRounding = 10.F;
    style.FrameRounding = 7.F;
    style.PopupRounding = 8.F;
    style.GrabRounding = 6.F;
    style.ScrollbarRounding = 8.F;
    style.TabRounding = 8.F;
    style.WindowPadding = ImVec2(14.F, 12.F);
    style.FramePadding = ImVec2(10.F, 8.F);
    style.ItemSpacing = ImVec2(10.F, 8.F);
    style.ItemInnerSpacing = ImVec2(8.F, 6.F);
    style.WindowBorderSize = 1.F;
    style.FrameBorderSize = 0.F;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.035f, 0.045f, 0.060f, 1.0f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.055f, 0.070f, 0.090f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.055f, 0.070f, 0.090f, 0.98f);
    colors[ImGuiCol_Border] = ImVec4(0.18f, 0.24f, 0.30f, 0.65f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.08f, 0.11f, 0.15f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.16f, 0.22f, 1.0f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.22f, 0.28f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.11f, 0.17f, 0.22f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.18f, 0.28f, 0.36f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.22f, 0.34f, 0.44f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.10f, 0.18f, 0.24f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.14f, 0.25f, 0.33f, 1.0f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.18f, 0.31f, 0.40f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.07f, 0.10f, 0.14f, 1.0f);
    colors[ImGuiCol_TabActive] = ImVec4(0.16f, 0.24f, 0.31f, 1.0f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.21f, 0.32f, 0.41f, 1.0f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.05f, 0.08f, 0.11f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.11f, 0.15f, 1.0f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.42f, 0.67f, 1.0f, 1.0f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.34f, 0.56f, 0.96f, 1.0f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.48f, 0.73f, 1.0f, 1.0f);
    colors[ImGuiCol_Separator] = ImVec4(0.22f, 0.29f, 0.35f, 0.8f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.32f, 0.55f, 0.90f, 0.70f);
}
void
EnvironmentRenderer::BuildDockspace(void)
{
    ImGuiViewport const* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.F);
    ImGui::Begin("##DockHost", nullptr, flags);
    ImGui::PopStyleVar(2);

    ImGuiID const dockspaceId = ImGui::GetID("##MainDockspace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.F, 0.F), ImGuiDockNodeFlags_PassthruCentralNode);
    BuildDefaultLayout(dockspaceId);
    ImGui::End();
}
void
EnvironmentRenderer::BuildDefaultLayout(ImGuiID dockspaceId)
{
    if (m_layoutInitialized)
        return;

    m_layoutInitialized = true;
    if (ImGui::DockBuilderGetNode(dockspaceId) != nullptr
        && ImGui::DockBuilderGetNode(dockspaceId)->IsSplitNode()) return;

    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);

    ImVec2 const size = ImGui::GetMainViewport()->Size;
    ImGui::DockBuilderSetNodeSize(dockspaceId, size);

    ImGuiID dockLeft = dockspaceId;
    ImGuiID dockRight = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Right, 0.27F, nullptr, &dockLeft);
    ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Down, 0.34F, nullptr, &dockLeft);
    ImGuiID dockTopLeft = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Left, 0.26F, nullptr, &dockLeft);

    ImGuiID dockRightBottom = ImGui::DockBuilderSplitNode(dockRight, ImGuiDir_Down, 0.44F, nullptr, &dockRight);
    ImGuiID dockBottomRight = ImGui::DockBuilderSplitNode(dockBottom, ImGuiDir_Right, 0.45F, nullptr, &dockBottom);

    ImGui::DockBuilderDockWindow("Simulation", dockTopLeft);
    ImGui::DockBuilderDockWindow("Entities", dockTopLeft);
    ImGui::DockBuilderDockWindow("Navigation View", dockLeft);
    ImGui::DockBuilderDockWindow("Minimap", dockBottom);
    ImGui::DockBuilderDockWindow("Inspection", dockBottomRight);
    ImGui::DockBuilderDockWindow("Agents", dockRight);
    ImGui::DockBuilderDockWindow("Meta Catalog", dockRightBottom);

    ImGui::DockBuilderFinish(dockspaceId);
}
void EnvironmentRenderer::LoadSettings()
{
    std::ifstream input("NavigationRenderer.json");
    if (!input) return;
    try
    {
        nlohmann::json settings;
        input >> settings;
        m_autoAttachCamera = settings.value("FollowSelection", false);
        m_showGrid = settings.value("Grid", true);
        m_showDomains = settings.value("Zones", true);
        m_showTrails = settings.value("Trails", true);
        m_showMetaCatalog = settings.value("Catalog", true);
        const float zoom = settings.value("MapZoom", 1.F);
        if (std::isfinite(zoom)) m_mapZoom = std::clamp(zoom, 0.3F, 4.F);
    }
    catch (const std::exception& error) { m_error = error.what(); }
}
void EnvironmentRenderer::SaveSettings() const
{
    const nlohmann::json settings = {{"FollowSelection",m_autoAttachCamera}, {"Grid",m_showGrid},
        {"Zones",m_showDomains}, {"Trails",m_showTrails}, {"Catalog",m_showMetaCatalog}, {"MapZoom",m_mapZoom}};
    std::ofstream output("NavigationRenderer.json");
    if (!output || !(output << settings.dump(2))) throw std::runtime_error("Cannot save renderer settings");
}
navigation::Point EnvironmentRenderer::SelectedPosition() const
{
    const auto& environment = m_application.environment();
    if (m_selection == 0) return environment.position();
    if (m_selection == 1) return environment.scenario().goal;
    return environment.scenario().obstacles.at(static_cast<std::size_t>(m_selection - 2)).center;
}
void EnvironmentRenderer::FocusSelection()
{
    const auto focus = World(SelectedPosition());
    m_camera.position = Vector3Add(m_camera.position, Vector3Subtract(focus, m_camera.target));
    m_camera.target = focus;
}
void EnvironmentRenderer::DrawSimulationControls()
{
    ImGui::Begin("Simulation");
    const auto& scenario = m_application.environment().scenario();
    ImGui::TextUnformatted(m_application.status().c_str());
    if (ImGui::Button("PLAY"))
    {
        try { m_application.play(static_cast<navigation::ControlMode>(m_mode), static_cast<std::size_t>(m_rollout), m_checkpoint); m_error.clear(); }
        catch (const std::exception& error) { m_error = error.what(); }
    }
    ImGui::SameLine();
    if (ImGui::Button("PAUSE")) m_application.pause();
    ImGui::SameLine();
    if (ImGui::Button("STOP"))
    {
        try { m_application.stop(); }
        catch (const std::exception& error) { m_error = error.what(); }
    }
    ImGui::SliderFloat("Time scale", &m_timeScale, 0.1F, 20.F);
    ImGui::Text("Step: %.3f s", scenario.dt);
    ImGui::Text("World: %.1f x %.1f", 2 * scenario.half_extent, 2 * scenario.half_extent);
    ImGui::TextWrapped("Manual: arrow keys turn and accelerate while the navigation view is focused.");
    ImGui::Separator();
    ImGui::Checkbox("Grid", &m_showGrid);
    ImGui::Checkbox("Forbidden zones", &m_showDomains);
    ImGui::Checkbox("Trails", &m_showTrails);
    ImGui::Checkbox("Follow selection", &m_autoAttachCamera);
    ImGui::Checkbox("Meta catalog", &m_showMetaCatalog);
    if (ImGui::Button("Focus selection")) FocusSelection();
    if (ImGui::Button("Save display settings"))
    {
        try { SaveSettings(); m_error.clear(); }
        catch (const std::exception& error) { m_error = error.what(); }
    }
    ImGui::TextWrapped("Camera: right drag orbit, middle drag pan, wheel zoom; WASD/QE move, F focus, R reset.");
    if (!m_error.empty()) ImGui::TextWrapped("Error: %s", m_error.c_str());
    ImGui::End();
}
void EnvironmentRenderer::UpdateFreeCamera(bool hovered)
{
    if (m_autoAttachCamera) { FocusSelection(); return; }
    if (!hovered || ImGui::GetIO().WantTextInput) return;
    Vector3 offset = Vector3Subtract(m_camera.position, m_camera.target);
    const float radius = std::max(0.5F, Vector3Length(offset));
    const auto delta = GetMouseDelta();
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
    {
        const float angle = std::atan2(offset.x, offset.z) - delta.x * 0.005F;
        const float elevation = std::clamp(std::asin(std::clamp(offset.y / radius, -0.98F, 0.98F)) - delta.y * 0.005F, -1.35F, 1.35F);
        offset = {std::sin(angle)*std::cos(elevation)*radius, std::sin(elevation)*radius, std::cos(angle)*std::cos(elevation)*radius};
        m_camera.position = Vector3Add(m_camera.target, offset);
    }
    const Vector3 up{0,1,0};
    const auto forward = Vector3Normalize(Vector3Subtract(m_camera.target, m_camera.position));
    const auto right = Vector3Normalize(Vector3CrossProduct(forward, up));
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
    {
        const auto pan = Vector3Add(Vector3Scale(right, -delta.x * 0.002F * radius), Vector3Scale(up, delta.y * 0.002F * radius));
        m_camera.position = Vector3Add(m_camera.position, pan);
        m_camera.target = Vector3Add(m_camera.target, pan);
    }
    const float wheel = GetMouseWheelMove();
    if (wheel != 0)
        m_camera.position = Vector3Add(m_camera.target, Vector3Scale(Vector3Normalize(offset), std::clamp(radius * std::exp(-wheel * 0.15F), 0.75F, 100000.F)));
    const float speed = GetFrameTime() * (IsKeyDown(KEY_LEFT_SHIFT) ? 25.F : 10.F);
    Vector3 translation{};
    if (IsKeyDown(KEY_W)) translation = Vector3Add(translation, Vector3Scale(forward, speed));
    if (IsKeyDown(KEY_S)) translation = Vector3Add(translation, Vector3Scale(forward, -speed));
    if (IsKeyDown(KEY_D)) translation = Vector3Add(translation, Vector3Scale(right, speed));
    if (IsKeyDown(KEY_A)) translation = Vector3Add(translation, Vector3Scale(right, -speed));
    if (IsKeyDown(KEY_E)) translation.y += speed;
    if (IsKeyDown(KEY_Q)) translation.y -= speed;
    m_camera.position = Vector3Add(m_camera.position, translation);
    m_camera.target = Vector3Add(m_camera.target, translation);
    if (IsKeyPressed(KEY_F)) FocusSelection();
    if (IsKeyPressed(KEY_R)) { m_camera.position = {0,12,12}; m_camera.target = {0,0,0}; }
}
void EnvironmentRenderer::RenderTacticalScene(int width, int height)
{
    if (m_texture.id == 0 || m_texture.texture.width != width || m_texture.texture.height != height)
    {
        if (m_texture.id) UnloadRenderTexture(m_texture);
        m_texture = LoadRenderTexture(width, height);
        if (!m_texture.id) throw std::runtime_error("Cannot allocate navigation viewport");
    }
    const auto& environment = m_application.environment();
    const auto& scenario = environment.scenario();
    BeginTextureMode(m_texture);
    ClearBackground({9,12,16,255});
    BeginMode3D(m_camera);
    if (m_showGrid) DrawGrid(100, static_cast<float>(scenario.half_extent / 50));
    const float extent = static_cast<float>(scenario.half_extent);
    DrawCubeWires({0,0,0}, 2*extent, 0.02F, 2*extent, GRAY);
    if (m_showDomains)
        for (const auto& obstacle : scenario.obstacles)
        {
            const float radius = static_cast<float>(obstacle.radius);
            DrawCylinder(World(obstacle.center), radius, radius, 0.1F, 48, Fade(ORANGE, 0.4F));
            DrawCylinderWires(World(obstacle.center), radius, radius, 0.1F, 48, ORANGE);
        }
    DrawCylinder(World(scenario.goal), static_cast<float>(scenario.arrival_radius), static_cast<float>(scenario.arrival_radius), 0.1F, 32, GREEN);
    DrawSphere(World(environment.position(), 0.25F), 0.25F, SKYBLUE);
    const auto& observation = std::get<std::vector<float>>(m_application.observation());
    const auto position = environment.position();
    DrawLine3D(World(position, 0.3F), World({position.x + observation[4], position.y + observation[5]}, 0.3F), WHITE);
    if (m_showTrails)
        for (std::size_t index = 1; index < m_trail.size(); ++index)
            DrawLine3D(World(m_trail[index-1], 0.05F), World(m_trail[index], 0.05F), Fade(SKYBLUE, 0.7F));
    DrawSphereWires(World(SelectedPosition(), 0.3F), 0.4F, 8, 8, YELLOW);
    EndMode3D();
    EndTextureMode();
}
void EnvironmentRenderer::DrawTacticalWindow()
{
    if (ImGui::Begin("Navigation View"))
    {
        const auto& telemetry = m_application.telemetry();
        ImGui::Text("Episodes %zu | success %zu | step %zu | %.0f FPS", telemetry.episodes, telemetry.successes, telemetry.length, ImGui::GetIO().Framerate);
        const ImVec2 size = ImGui::GetContentRegionAvail();
        if (size.x > 1 && size.y > 1)
        {
            UpdateFreeCamera(ImGui::IsWindowHovered());
            RenderTacticalScene(static_cast<int>(size.x), static_cast<int>(size.y));
            rlImGuiImageRenderTextureFit(&m_texture, true);
        }
    }
    ImGui::End();
}
void EnvironmentRenderer::DrawMinimapWindow()
{
    if (ImGui::Begin("Minimap"))
    {
        ImGui::SliderFloat("Zoom", &m_mapZoom, 0.3F, 4.F);
        const auto origin = ImGui::GetCursorScreenPos();
        const auto available = ImGui::GetContentRegionAvail();
        const float side = std::max(1.F, std::min(available.x, available.y));
        ImGui::InvisibleButton("Map", ImVec2(side,side));
        auto* draw = ImGui::GetWindowDrawList();
        draw->PushClipRect(origin, ImVec2(origin.x+side,origin.y+side), true);
        draw->AddRectFilled(origin, ImVec2(origin.x+side,origin.y+side), IM_COL32(12,18,24,255));
        const auto& environment = m_application.environment();
        const auto& scenario = environment.scenario();
        const float scale = side * m_mapZoom / static_cast<float>(2*scenario.half_extent);
        const ImVec2 center(origin.x+side/2,origin.y+side/2);
        for (std::size_t i = 0; i < scenario.obstacles.size()+2; ++i)
        {
            navigation::Point point;
            float radius = 4;
            ImU32 color = IM_COL32(80,170,245,255);
            if (i == 0) point = environment.position();
            else if (i == 1) { point = scenario.goal; color = IM_COL32(80,230,120,255); radius = std::max(4.F, scale*static_cast<float>(scenario.arrival_radius)); }
            else { const auto& zone = scenario.obstacles[i-2]; point = zone.center; radius = scale*static_cast<float>(zone.radius); color = IM_COL32(240,140,60,200); }
            const ImVec2 screen(center.x+static_cast<float>(point.x)*scale, center.y-static_cast<float>(point.y)*scale);
            draw->AddCircleFilled(screen,radius,color);
            if (static_cast<int>(i) == m_selection) draw->AddCircle(screen,radius+3,IM_COL32(255,230,80,255));
            const auto mouse = ImGui::GetIO().MousePos;
            const float dx = mouse.x-screen.x, dy = mouse.y-screen.y;
            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0) && dx*dx+dy*dy <= (radius+5)*(radius+5)) m_selection = static_cast<int>(i);
        }
        draw->PopClipRect();
    }
    ImGui::End();
}
void EnvironmentRenderer::DrawEntitiesWindow()
{
    if (ImGui::Begin("Entities"))
    {
        m_filter.Draw("Filter");
        const auto count = m_application.environment().scenario().obstacles.size()+2;
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto name = EntityName(static_cast<int>(index));
            if (!m_filter.PassFilter(name.c_str())) continue;
            if (ImGui::Selectable(name.c_str(), m_selection == static_cast<int>(index))) m_selection = static_cast<int>(index);
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) { m_selection = static_cast<int>(index); FocusSelection(); }
        }
    }
    ImGui::End();
}
void EnvironmentRenderer::DrawInspectorWindow()
{
    if (ImGui::Begin("Inspection"))
    {
        const auto point = SelectedPosition();
        ImGui::TextUnformatted(EntityName(m_selection).c_str());
        ImGui::Text("Position: %.4f, %.4f", point.x, point.y);
        const auto& scenario = m_application.environment().scenario();
        if (m_selection == 0)
        {
            const auto& values = std::get<std::vector<float>>(m_application.observation());
            ImGui::Text("Heading: %.3f rad", std::atan2(values[5],values[4]));
            ImGui::Text("Speed: %.3f", values[6]*scenario.max_speed);
            const double dx = point.x-scenario.goal.x, dy = point.y-scenario.goal.y;
            ImGui::Text("Distance: %.3f", std::hypot(dx,dy));
        }
        else if (m_selection == 1) ImGui::Text("Arrival radius: %.3f", scenario.arrival_radius);
        else ImGui::Text("Radius: %.3f", scenario.obstacles.at(static_cast<std::size_t>(m_selection-2)).radius);
    }
    ImGui::End();
}
void EnvironmentRenderer::DrawAgentsWindow()
{
    if (ImGui::Begin("Agents"))
    {
        const auto& telemetry = m_application.telemetry();
        if (ImGui::BeginTabBar("AgentTabs"))
        {
            if (ImGui::BeginTabItem("Controls"))
            {
                ImGui::Combo("Mode", &m_mode, "Manual\0Training\0Frozen policy\0");
                ImGui::SliderInt("New learner rollout", &m_rollout, 64, 8192);
                ImGui::InputText("Policy model.pt", m_checkpoint, sizeof(m_checkpoint));
                ImGui::TextWrapped("Stop before switching mode. Rollout applies when creating a learner. Pause/Play resumes the current policy.");
                ImGui::SliderInt("Evaluation episodes", &m_evaluationEpisodes, 1, 1000);
                if (ImGui::Button("Save candidate / evaluate / promote"))
                {
                    try { m_application.evaluate_candidate(static_cast<std::size_t>(m_evaluationEpisodes)); m_error.clear(); }
                    catch (const std::exception& error) { m_error = error.what(); }
                }
                ImGui::Text("Rollout step: %zu", telemetry.rollout_step);
                ImGui::Text("Reward: %.5f | length: %zu", telemetry.reward, telemetry.length);
                ImGui::Text("Terminated: %s | truncated: %s", telemetry.terminated ? "yes":"no", telemetry.truncated ? "yes":"no");
                ImGui::Text("Loss %.5f | actor %.5f", telemetry.loss, telemetry.actor_loss);
                ImGui::Text("Critic %.5f | entropy %.5f", telemetry.critic_loss, telemetry.entropy);
                ImGui::SeparatorText("Environment reward");
                ImGui::TextWrapped("Progress / world extent - 0.001 per step; arrival +1; collision -1. These rules belong to Navigation2D.");
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Snapshot"))
            {
                const auto& values = std::get<std::vector<float>>(m_application.observation());
                for (std::size_t index=0; index<values.size(); ++index) ImGui::Text("observation[%zu] = %.6f", index, values[index]);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Plots"))
            {
                if (ImPlot::BeginPlot("Episode reward", ImVec2(-1,180)))
                {
                    if (!telemetry.rewards.empty()) ImPlot::PlotLine("Reward", telemetry.rewards.data(), static_cast<int>(telemetry.rewards.size()));
                    ImPlot::EndPlot();
                }
                if (ImPlot::BeginPlot("PPO loss", ImVec2(-1,180)))
                {
                    if (!telemetry.losses.empty()) ImPlot::PlotLine("Loss", telemetry.losses.data(), static_cast<int>(telemetry.losses.size()));
                    ImPlot::EndPlot();
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}
void EnvironmentRenderer::DrawMetaCatalogWindow()
{
    if (!m_showMetaCatalog) return;
    if (ImGui::Begin("Meta Catalog", &m_showMetaCatalog))
    {
        ImGui::TextUnformatted("Navigation2D schema");
        ImGui::BulletText("Mobile: position, heading, speed");
        ImGui::BulletText("Destination: position, arrival radius");
        ImGui::BulletText("Forbidden zone: center, radius");
        const auto& observation = std::get<std::vector<float>>(m_application.observation());
        ImGui::Text("Observation: Box float32 [%zu]", observation.size());
        ImGui::TextUnformatted("Action: Box float32 [2], range [-1, 1]");
        ImGui::TextUnformatted("Action[0]: turn; action[1]: acceleration");
        ImGui::TextUnformatted("Info: success, collision, distance");
        ImGui::TextWrapped("The catalog describes the example. The RL framework only sees spaces and transitions.");
    }
    ImGui::End();
}
bool EnvironmentRenderer::Run()
{
    if (WindowShouldClose()) return false;
    BeginDrawing();
    ClearBackground({9,12,16,255});
    rlImGuiBegin();
    BuildDockspace();
    std::vector<float> action{0,0};
    // Text entry and other panels must never steer the mobile.
    const auto* view = ImGui::FindWindowByName("Navigation View");
    if (view && ImGui::GetCurrentContext()->NavWindow == view && !ImGui::GetIO().WantTextInput)
    {
        action[0] = (IsKeyDown(KEY_RIGHT) ? 1.F:0.F) - (IsKeyDown(KEY_LEFT) ? 1.F:0.F);
        action[1] = (IsKeyDown(KEY_UP) ? 1.F:0.F) - (IsKeyDown(KEY_DOWN) ? 1.F:0.F);
    }
    try { m_application.tick(GetFrameTime(), action, m_timeScale); }
    catch (const std::exception& error) { m_application.pause(); m_error = error.what(); }
    const auto& telemetry = m_application.telemetry();
    if (telemetry.episodes != m_lastEpisode || telemetry.length < m_lastStep) m_trail.clear();
    if (telemetry.length != m_lastStep)
    {
        if (m_trail.size() == 5000) m_trail.erase(m_trail.begin());
        m_trail.push_back(m_application.environment().position());
    }
    m_lastEpisode = telemetry.episodes;
    m_lastStep = telemetry.length;
    DrawSimulationControls();
    DrawTacticalWindow();
    DrawMinimapWindow();
    DrawEntitiesWindow();
    DrawInspectorWindow();
    DrawAgentsWindow();
    DrawMetaCatalogWindow();
    rlImGuiEnd();
    EndDrawing();
    return true;
}
}
