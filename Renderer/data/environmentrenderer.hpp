/**
* @file Renderer/data/environmentrenderer.hpp
* @brief Declarations for environmentrenderer.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <imgui.h>
#include <raylib.h>

#include <array>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "environment.hpp"
#include "ppo/core/learning_monitor.hpp"

namespace ppo::renderer {

/**
* @brief Main renderer and UI controller for the simulation.
*/
class EnvironmentRenderer
{
public:
    /**
    * @brief Returns the singleton renderer instance.
    * @param title Window title used on first construction.
    * @return Shared renderer singleton.
    */
    static EnvironmentRenderer& GetInstance(std::string const& title = "PLAYER ENVIRONMENT");

    EnvironmentRenderer(EnvironmentRenderer&) = delete;
    EnvironmentRenderer(EnvironmentRenderer&&) = delete;
    EnvironmentRenderer& operator=(EnvironmentRenderer&) = delete;
    EnvironmentRenderer& operator=(EnvironmentRenderer&&) = delete;

    /**
    * @brief Shuts down rendering resources and the UI backend.
    */
    ~EnvironmentRenderer(void);

    /**
    * @brief Attaches the renderer to one simulation environment.
    * @param env Environment to visualize.
    */
    void SetEnvironment(ppo::model::Environment& env) noexcept;
    /**
    * @brief Executes one UI frame.
    * @return False when the application should exit.
    */
    bool Run(void);

    /**
    * @brief Fixed-size float history buffer used by plots.
    * @tparam N Maximum number of samples kept in memory.
    */
    template<std::size_t N>
    struct CircularBuffer
    {
        /**
        * @brief Preallocates the buffer storage.
        */
        CircularBuffer()
        {
            m_data.reserve(N);
            m_data.resize(N);
        }

        /**
        * @brief Appends one value to the history.
        * @param value Sample value to append.
        */
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

        /**
        * @brief Returns a raw pointer to the stored samples.
        * @return Sample buffer pointer.
        */
        float const* data() const noexcept { return m_data.data(); }
        /**
        * @brief Returns the number of valid samples.
        * @return Sample count.
        */
        std::size_t size() const noexcept { return m_size; }
        /**
        * @brief Clears the logical sample count.
        */
        void clear() noexcept { m_size = 0; }

        /**
        * @brief Backing storage for plot samples.
        */
        std::vector<float> m_data{};
        /**
        * @brief Number of valid samples currently stored.
        */
        std::size_t m_size{ 0 };
        /**
        * @brief Plot label.
        */
        std::string m_name{};
    };

    /**
    * @brief UI state associated with one training agent.
    */
    struct AgentController
    {
        /**
        * @brief Plot type used for agent histories.
        */
        using Plot = CircularBuffer<5000>;
        /**
        * @brief Fixed plot collection used by the UI.
        */
        using Plots = std::array<Plot, 5>;

        /**
        * @brief Performance and reward plots.
        */
        Plots m_plots{};
        // Observational only. A data point requires a completed evaluation.
        ppo::core::LearningHistory m_evaluationScores{};
        ppo::core::LearningHistory m_evaluationSuccess{};
        unsigned int m_lastPlottedVersion{ 0U };
        bool m_seenVersion{ false };
        /**
        * @brief True once the controller has received valid data.
        */
        bool m_started{ false };
        /**
        * @brief Reward scaling values shown in the UI.
        */
        std::vector<float> m_scalings{};
        /**
        * @brief Rollout length configured for the agent.
        */
        int m_rollout{ 0 };
        /**
        * @brief Reward clamp value configured for the agent.
        */
        float m_clamp{ 0.F };
        /**
        * @brief Mean reward over the last finished episode.
        */
        float m_rewardMean{ 0.F };
        /**
        * @brief Number of reward samples accumulated for the mean.
        */
        unsigned int m_rewardSampleCount{ 0 };
        /**
        * @brief Last step index included in the reward mean.
        */
        long long m_lastRewardStep{ -1 };
        /**
        * @brief Unique agent identifier.
        */
        unsigned long long m_id{ 0 };
    };

    /**
    * @brief Mission-level telemetry aggregated for the tactical UI.
    */
    struct MissionTelemetry
    {
        /**
        * @brief Agent identifier associated with the telemetry.
        */
        unsigned long long m_agentId{ 0 };
        /**
        * @brief Number of completed episodes.
        */
        unsigned int m_totalEpisodes{ 0 };
        /**
        * @brief Number of successful episodes.
        */
        unsigned int m_successEpisodes{ 0 };
        /**
        * @brief Previous done flag used to detect episode boundaries.
        */
        bool m_previousDone{ false };
        /**
        * @brief Result of the last completed episode.
        */
        bool m_lastEpisodeSuccess{ false };
    };

    /**
    * @brief One timestamped point stored in an entity trail.
    */
    struct TrailPoint
    {
        /**
        * @brief World-space position of the point.
        */
        Vector3 m_position{};
        /**
        * @brief Timestamp in seconds.
        */
        double m_timestamp{ 0.0 };
    };

    /**
    * @brief Runtime state for a transient 3D explosion effect.
    */
    struct ExplosionEffect
    {
        /**
        * @brief Unique event identifier.
        */
        unsigned long long m_eventId{ 0 };
        /**
        * @brief North position in meters.
        */
        float m_x{ 0.F };
        /**
        * @brief East position in meters.
        */
        float m_y{ 0.F };
        /**
        * @brief Down position in meters.
        */
        float m_z{ 0.F };
        /**
        * @brief Time at which the effect started.
        */
        double m_startTime{ 0.0 };
        /**
        * @brief Base visual radius.
        */
        float m_baseRadius{ 0.9F };
        /**
        * @brief Vertical scaling used to make the plume taller.
        */
        float m_heightScale{ 1.F };
        /**
        * @brief Destroyed entity type associated with the effect.
        */
        ppo::model::eEntityType m_type{ ppo::model::eEntityType::None };
    };

    /**
    * @brief Loaded 3D asset with transform metadata.
    */
    struct ModelAsset
    {
        /**
        * @brief Raylib model handle.
        */
        Model m_model{};
        /**
        * @brief Model-space bounding box.
        */
        BoundingBox m_bounds{};
        /**
        * @brief Uniform scale applied at draw time.
        */
        float m_uniformScale{ 1.F };
        /**
        * @brief Vertical offset required to rest on the ground plane.
        */
        float m_groundOffset{ 0.F };
        /**
        * @brief Constant yaw offset used to align the asset with gameplay heading.
        */
        float m_yawOffsetDeg{ 0.F };
        /**
        * @brief True when the asset is loaded and valid.
        */
        bool m_loaded{ false };
    };

private:
    /**
    * @brief Creates the renderer singleton.
    * @param title Window title.
    */
    explicit EnvironmentRenderer(std::string const& title);

    /**
    * @brief Returns the UI controller associated with one agent id.
    * @param id Agent identifier.
    * @return Controller pointer, created on demand.
    */
    std::shared_ptr<AgentController> GetAgentController(unsigned long long id);
    /**
    * @brief Applies one world-scale preset.
    * @param scaleLevel Scale preset index.
    */
    void ApplyScaleLevel(int scaleLevel) noexcept;
    /**
    * @brief Loads external and generated 3D assets.
    */
    void LoadModelAssets(void);
    /**
    * @brief Releases every loaded 3D asset.
    */
    void UnloadModelAssets(void);

    /**
    * @brief Ensures the tactical render texture matches the requested size.
    * @param width Render target width.
    * @param height Render target height.
    */
    void EnsureTacticalTexture(int width, int height);
    /**
    * @brief Builds the root docking layout.
    */
    void BuildDockspace(void);
    /**
    * @brief Applies the ImGui visual theme.
    */
    void ApplyTheme(void);
    /**
    * @brief Builds the default docking layout.
    * @param dockspaceId Root dockspace identifier.
    */
    void BuildDefaultLayout(ImGuiID dockspaceId);
    /**
    * @brief Updates aggregated mission telemetry from the latest snapshot.
    * @param snapshot Read-only environment snapshot.
    */
    void UpdateMissionTelemetry(entt::registry const& snapshot);
    /**
    * @brief Updates entity trails from the latest snapshot.
    * @param snapshot Read-only environment snapshot.
    */
    void UpdateTrails(entt::registry const& snapshot);
    /**
    * @brief Spawns and ages explosion effects from the latest snapshot.
    * @param snapshot Read-only environment snapshot.
    */
    void UpdateExplosionEffects(entt::registry const& snapshot);
    /**
    * @brief Keeps tracked and selected entities valid across snapshots.
    * @param snapshot Read-only environment snapshot.
    */
    void UpdateTrackedEntity(entt::registry const& snapshot);
    /**
    * @brief Updates the attached camera when auto-follow is enabled.
    * @param snapshot Read-only environment snapshot.
    */
    void UpdateAttachedCamera(entt::registry const& snapshot);
    /**
    * @brief Updates free-camera controls from user input.
    * @param snapshot Read-only environment snapshot.
    * @param hovered True when the tactical viewport is hovered.
    * @param viewportOrigin Screen-space viewport origin.
    * @param viewportSize Screen-space viewport size.
    */
    void UpdateFreeCamera(entt::registry const& snapshot, bool hovered, Vector2 viewportOrigin, Vector2 viewportSize);

    /**
    * @brief Draws simulation play, pause, and stop controls.
    */
    void DrawSimulationControls(void);
    /**
    * @brief Draws the top-level simulation status bar.
    * @param snapshot Read-only environment snapshot.
    */
    void DrawStatusBar(entt::registry const& snapshot);
    /**
    * @brief Draws training controls for one agent.
    * @param agent UI controller for the selected agent.
    * @param state Current agent snapshot.
    */
    void DrawTrainingControls(std::shared_ptr<AgentController> const& agent,
                              ppo::model::sAgentStateSnaphot const& state);
    /**
    * @brief Draws reward controls for one agent.
    * @param agent UI controller for the selected agent.
    * @param state Current agent snapshot.
    */
    void DrawRewardControls(std::shared_ptr<AgentController> const& agent,
                            ppo::model::sAgentStateSnaphot const& state);
    /**
    * @brief Draws the plot selection area for one agent.
    * @param agent UI controller for the selected agent.
    * @param state Current agent snapshot.
    */
    void DrawAgentPlot(std::shared_ptr<AgentController> const& agent,
                       ppo::model::sAgentStateSnaphot const& state);
    /** Draws evaluation progress (distinct from PPO training losses). */
    void DrawLearningProgress(std::shared_ptr<AgentController> const& agent,
                              ppo::model::sAgentStateSnaphot const& state);
    /** Collects a new sample only when its source event advances. */
    void UpdateLearningTelemetry(entt::registry const& snapshot);
    /**
    * @brief Draws one plot widget.
    * @param plot Plot descriptor to display.
    */
    void DrawOneAgentPlot(AgentController::Plot& plot);

    /**
    * @brief Draws the tactical 3D viewport window.
    * @param snapshot Read-only environment snapshot.
    */
    void DrawTacticalWindow(entt::registry const& snapshot);
    /**
    * @brief Draws the minimap window.
    * @param snapshot Read-only environment snapshot.
    */
    void DrawMinimapWindow(entt::registry const& snapshot);
    /**
    * @brief Draws the entity list window.
    * @param snapshot Read-only environment snapshot.
    */
    void DrawEntitiesWindow(entt::registry const& snapshot);
    /**
    * @brief Draws the entity inspector window.
    * @param snapshot Read-only environment snapshot.
    */
    void DrawInspectorWindow(entt::registry const& snapshot);
    /**
    * @brief Draws the agent list and metrics window.
    * @param snapshot Read-only environment snapshot.
    */
    void DrawAgentsWindow(entt::registry const& snapshot);
    /**
    * @brief Draws the metadata catalog window.
    */
    void DrawMetaCatalogWindow(void);

    /**
    * @brief Renders the full tactical 3D scene into the render texture.
    * @param snapshot Read-only environment snapshot.
    * @param width Render width.
    * @param height Render height.
    */
    void RenderTacticalScene(entt::registry const& snapshot, int width, int height);

    /**
    * @brief Attached simulation environment.
    */
    ppo::model::Environment* p_environment{ nullptr };
    /**
    * @brief Per-agent UI controller cache.
    */
    std::map<unsigned long long, std::shared_ptr<AgentController>> p_agents{};
    /**
    * @brief Plot currently pinned to the main simulation window.
    */
    AgentController::Plot* p_pinnedPlot{ nullptr };

    /**
    * @brief Off-screen tactical render target.
    */
    RenderTexture2D m_tacticalTexture{};
    /**
    * @brief True when the tactical render target is allocated.
    */
    bool m_hasTacticalTexture{ false };
    /**
    * @brief Active 3D camera.
    */
    Camera3D m_camera{};

    /**
    * @brief Window title.
    */
    std::string m_title{};
    /**
    * @brief Current world-to-render scaling factor.
    */
    float m_worldScale{ 1000.F };
    /**
    * @brief Approximate tactical area width in display units.
    */
    float m_areawidth{ 150.F };
    /**
    * @brief Current north-axis range.
    */
    float m_northRange{ 112500.F };
    /**
    * @brief Current east-axis range.
    */
    float m_eastRange{ 150000.F };
    /**
    * @brief Simulation time step expressed in milliseconds.
    */
    float m_timestepMs{ 20.F };
    /**
    * @brief Camera follow distance.
    */
    float m_cameraDistance{ 12.F };
    /**
    * @brief Camera follow height.
    */
    float m_cameraHeight{ 4.5F };
    /**
    * @brief Free-camera move speed.
    */
    float m_cameraMoveSpeed{ 12.F };
    /**
    * @brief Trail persistence duration in seconds.
    */
    float m_trailDuration{ 12.F };
    /**
    * @brief Display height used for engagement cones.
    */
    float m_domainHeight{ 1.6F };
    /**
    * @brief Minimap zoom factor.
    */
    float m_mapZoom{ 1.F };
    /**
    * @brief Selected unit scale preset.
    */
    int m_currentScaleLevel{ 1 };
    /**
    * @brief True once the docking layout has been initialized.
    */
    bool m_layoutInitialized{ false };
    /**
    * @brief True while the camera auto-attaches to the tracked entity.
    */
    bool m_autoAttachCamera{ true };
    /**
    * @brief True when the tactical grid is visible.
    */
    bool m_showGrid{ true };
    /**
    * @brief True when engagement domains are visible.
    */
    bool m_showDomains{ true };
    /**
    * @brief True when destroyed entities remain visible.
    */
    bool m_showDestroyed{ true };
    /**
    * @brief True when the metadata catalog window is visible.
    */
    bool m_showMetaCatalog{ false };
    bool m_showSimulation{ true };
    bool m_showAgents{ true };
    bool m_showMinimap{ false };
    bool m_showEntities{ false };
    bool m_showInspection{ false };
    /**
    * @brief True when the pseudo-lidar overlay is visible.
    */
    bool m_showPseudoLidar{ false };
    /**
    * @brief True when the minimap follows the current selection.
    */
    bool m_focusSelectionInMinimap{ false };
    /**
    * @brief True when a stop request has been issued from the UI.
    */
    bool m_stopRequested{ false };
    /**
    * @brief Entity-name filter buffer used by the UI.
    */
    char m_entityFilter[96]{};
    /**
    * @brief Aggregated mission telemetry.
    */
    MissionTelemetry m_missionTelemetry{};
    /**
    * @brief World-space movement trails keyed by entity id.
    */
    std::map<unsigned long long, std::deque<TrailPoint>> m_entityTrails{};
    /**
    * @brief Active explosion effects keyed by entity id.
    */
    std::map<unsigned long long, ExplosionEffect> m_explosionEffects{};

    /**
    * @brief Currently selected entity.
    */
    entt::entity m_selectedEntity{ entt::null };
    /**
    * @brief Currently tracked entity.
    */
    entt::entity m_trackedEntity{ entt::null };
    /**
    * @brief Friendly aircraft model asset.
    */
    ModelAsset m_blueAircraftModel{};
    /**
    * @brief Hostile aircraft model asset.
    */
    ModelAsset m_redAircraftModel{};
    /**
    * @brief Objective model asset.
    */
    ModelAsset m_objectiveModel{};
};

} /* ppo::renderer */
