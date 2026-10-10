#pragma once
#include "application.hpp"
#include "raylib.h"
#include "imgui.h"
#include <string>
#include <vector>

namespace ppo::renderer {
/// Existing desktop shell ported to the Navigation2D application controller.
class EnvironmentRenderer
{
public:
    explicit EnvironmentRenderer(navigation::Application& application);
    ~EnvironmentRenderer();
    EnvironmentRenderer(const EnvironmentRenderer&) = delete;
    EnvironmentRenderer& operator=(const EnvironmentRenderer&) = delete;
    bool Run();
private:
    void ApplyTheme();
    void BuildDockspace();
    void BuildDefaultLayout(ImGuiID dockspaceId);
    void DrawSimulationControls();
    void DrawTacticalWindow();
    void DrawMinimapWindow();
    void DrawEntitiesWindow();
    void DrawInspectorWindow();
    void DrawAgentsWindow();
    void DrawMetaCatalogWindow();
    void UpdateFreeCamera(bool hovered);
    void FocusSelection();
    void RenderTacticalScene(int width, int height);
    navigation::Point SelectedPosition() const;
    void LoadSettings();
    void SaveSettings() const;
    navigation::Application& m_application;
    Camera3D m_camera{};
    RenderTexture2D m_texture{};
    bool m_layoutInitialized = false;
    bool m_autoAttachCamera = false;
    bool m_showGrid = true;
    bool m_showDomains = true;
    bool m_showLidar = true;
    bool m_showMetaCatalog = true;
    bool m_showTrails = true;
    float m_timeScale = 1;
    float m_mapZoom = 1;
    navigation::EntityId m_selection = 1;
    int m_mode = 0;
    int m_rollout = 2048;
    int m_evaluationEpisodes = 10;
    int m_initializationMode = 0;
    char m_caseManifest[1024] = "scenarios/evaluation.cases";
    char m_checkpoint[1024] = {};
    int m_selectedMode = 0;
    char m_selectedCheckpoint[1024] = {};
    char m_scenarioPath[1024] = "scenarios/multiple.nav";
    ImGuiTextFilter m_filter;
    std::string m_error;
    std::map<navigation::EntityId, std::vector<navigation::Point>> m_trails;
    std::size_t m_lastEpisode = 0;
    std::size_t m_lastStep = 0;
};
}
