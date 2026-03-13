#pragma once

#include <imgui.h>
#include <raylib.h>

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "environment.hpp"

namespace ppo::renderer {

class EnvironmentRenderer
{
public:
    static EnvironmentRenderer& GetInstance(std::string const& title = "PLAYER ENVIRONMENT");

    EnvironmentRenderer(EnvironmentRenderer&) = delete;
    EnvironmentRenderer(EnvironmentRenderer&&) = delete;
    EnvironmentRenderer& operator=(EnvironmentRenderer&) = delete;
    EnvironmentRenderer& operator=(EnvironmentRenderer&&) = delete;

    ~EnvironmentRenderer(void);

    void SetEnvironment(ppo::model::Environment& env) noexcept;
    bool Run(void);

    template<std::size_t N>
    struct CircularBuffer
    {
        CircularBuffer()
        {
            m_data.reserve(N);
            m_data.resize(N);
        }

        void push(float value)
        {
            if (m_size < N)
            {
                m_data[m_size] = value;
                ++m_size;
            }
            else
            {
                std::move(m_data.begin() + 1, m_data.end(), m_data.begin());
                m_data.back() = value;
            }
        }

        float const* data() const noexcept { return m_data.data(); }
        std::size_t size() const noexcept { return m_size; }

        std::vector<float> m_data{};
        std::size_t m_size{ 0 };
        std::string m_name{};
    };

    struct AgentController
    {
        using Plot = CircularBuffer<5000>;
        using Plots = std::array<Plot, 4>;

        Plots m_plots{};
        bool m_started{ false };
        std::vector<float> m_scalings{};
        int m_rollout{ 0 };
        float m_clamp{ 0.F };
        unsigned long long m_id{ 0 };
    };

    struct MissionTelemetry
    {
        unsigned long long m_agentId{ 0 };
        unsigned int m_totalEpisodes{ 0 };
        unsigned int m_successEpisodes{ 0 };
        bool m_previousDone{ false };
        bool m_lastEpisodeSuccess{ false };
    };

private:
    explicit EnvironmentRenderer(std::string const& title);

    std::shared_ptr<AgentController> GetAgentController(unsigned long long id);

    void EnsureTacticalTexture(int width, int height);
    void BuildDockspace(void);
    void ApplyTheme(void);
    void BuildDefaultLayout(ImGuiID dockspaceId);
    void UpdateMissionTelemetry(entt::registry const& snapshot);
    void UpdateTrackedEntity(entt::registry const& snapshot);
    void UpdateAttachedCamera(entt::registry const& snapshot);

    void DrawSimulationControls(void);
    void DrawStatusBar(entt::registry const& snapshot);
    void DrawTrainingControls(std::shared_ptr<AgentController> const& agent,
                              ppo::model::sAgentStateSnaphot const& state);
    void DrawRewardControls(std::shared_ptr<AgentController> const& agent,
                            ppo::model::sAgentStateSnaphot const& state);
    void DrawAgentPlot(std::shared_ptr<AgentController> const& agent,
                       ppo::model::sAgentStateSnaphot const& state);
    void DrawOneAgentPlot(AgentController::Plot& plot);

    void DrawTacticalWindow(entt::registry const& snapshot);
    void DrawMinimapWindow(entt::registry const& snapshot);
    void DrawEntitiesWindow(entt::registry const& snapshot);
    void DrawInspectorWindow(entt::registry const& snapshot);
    void DrawAgentsWindow(entt::registry const& snapshot);
    void DrawMetaCatalogWindow(void);

    void RenderTacticalScene(entt::registry const& snapshot, int width, int height);

    ppo::model::Environment* p_environment{ nullptr };
    std::map<unsigned long long, std::shared_ptr<AgentController>> p_agents{};
    AgentController::Plot* p_pinnedPlot{ nullptr };

    RenderTexture2D m_tacticalTexture{};
    bool m_hasTacticalTexture{ false };
    Camera3D m_camera{};

    std::string m_title{};
    float m_worldScale{ 1000.F };
    float m_areawidth{ 150.F };
    float m_northRange{ 112500.F };
    float m_eastRange{ 150000.F };
    float m_timestepMs{ 20.F };
    float m_cameraDistance{ 18.F };
    float m_cameraHeight{ 8.F };
    float m_domainHeight{ 1.6F };
    float m_mapZoom{ 1.F };
    int m_currentScaleLevel{ 1 };
    bool m_layoutInitialized{ false };
    bool m_autoAttachCamera{ true };
    bool m_showGrid{ true };
    bool m_showDomains{ true };
    bool m_showDestroyed{ true };
    bool m_showMetaCatalog{ true };
    bool m_focusSelectionInMinimap{ false };
    char m_entityFilter[96]{};
    MissionTelemetry m_missionTelemetry{};

    entt::entity m_selectedEntity{ entt::null };
    entt::entity m_trackedEntity{ entt::null };
};

} /* ppo::renderer */
