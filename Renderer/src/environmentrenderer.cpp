#include "environmentrenderer.hpp"
#include <imgui.h>
#include <imgui-SFML.h>

#include <implot.h>

namespace ppo::renderer {

void 
SetupModernDarkTheme(void)
{
    ImGuiStyle& style = ImGui::GetStyle();

    // -------------------------------
    // Taille globale des contrôles
    // -------------------------------
    style.ScaleAllSizes(1.2f);   // clé pour éviter les controls rikiki

    // -------------------------------
    // Spacing & padding (lisibilité)
    // -------------------------------
    style.WindowPadding = ImVec2(14, 14);
    style.FramePadding = ImVec2(12, 8);   // HAUTEUR des boutons/sliders
    style.ItemSpacing = ImVec2(12, 10);
    style.ItemInnerSpacing = ImVec2(8, 6);
    style.IndentSpacing = 22.0f;
    style.ScrollbarSize = 18.0f;

    // -------------------------------
    // Arrondis (modern, flat)
    // -------------------------------
    style.WindowRounding = 6.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.TabRounding = 6.0f;

    // -------------------------------
    // Bordures quasi invisibles
    // -------------------------------
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.PopupBorderSize = 1.0f;
    style.TabBorderSize = 0.0f;

    // -------------------------------
    // Couleurs : black / flat / modern
    // -------------------------------
    ImVec4* c = style.Colors;

    c[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.08f, 0.09f, 1.00f);
    c[ImGuiCol_ChildBg] = ImVec4(0.07f, 0.08f, 0.09f, 1.00f);
    c[ImGuiCol_PopupBg] = ImVec4(0.09f, 0.10f, 0.11f, 1.00f);
    c[ImGuiCol_Border] = ImVec4(0.18f, 0.19f, 0.21f, 1.00f);

    c[ImGuiCol_FrameBg] = ImVec4(0.14f, 0.15f, 0.17f, 1.00f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.21f, 0.24f, 1.00f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.28f, 0.32f, 1.00f);

    c[ImGuiCol_Button] = ImVec4(0.18f, 0.19f, 0.22f, 1.00f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.30f, 0.35f, 1.00f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.35f, 0.38f, 0.45f, 1.00f);

    c[ImGuiCol_Header] = ImVec4(0.18f, 0.19f, 0.22f, 1.00f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.30f, 0.35f, 1.00f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.35f, 0.38f, 0.45f, 1.00f);

    c[ImGuiCol_Tab] = ImVec4(0.12f, 0.13f, 0.15f, 1.00f);
    c[ImGuiCol_TabHovered] = ImVec4(0.28f, 0.30f, 0.35f, 1.00f);
    c[ImGuiCol_TabActive] = ImVec4(0.20f, 0.22f, 0.26f, 1.00f);

    c[ImGuiCol_SliderGrab] = ImVec4(0.45f, 0.48f, 0.55f, 1.00f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.55f, 0.60f, 0.70f, 1.00f);

    c[ImGuiCol_CheckMark] = ImVec4(0.65f, 0.70f, 0.80f, 1.00f);

    c[ImGuiCol_Text] = ImVec4(0.92f, 0.94f, 0.97f, 1.00f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.55f, 0.58f, 0.62f, 1.00f);


    ImGuiIO& io = ImGui::GetIO();
    // Utilisation de Segoe UI (police native Windows)
    const char* fontPath = "C:\\Windows\\Fonts\\segoeui.ttf";
    ImFontConfig config;
    config.SizePixels = 16.0f;
    io.Fonts->AddFontFromFileTTF(fontPath, 20.0f, &config, io.Fonts->GetGlyphRangesDefault());

    io.FontGlobalScale = 1.3F;
    io.FontAllowUserScaling = true;
    io.ConfigWindowsResizeFromEdges = true;
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImGui::SFML::UpdateFontTexture();
}

EnvironmentRenderer&
EnvironmentRenderer::GetInstance(std::string const& title)
{
    static EnvironmentRenderer renderer{ title };
    return renderer;
}

EnvironmentRenderer::EnvironmentRenderer(std::string const& title) :
    sf::RenderWindow(sf::VideoMode(1280, 980), title)
{
    if (ImGui::SFML::Init(*this))
        setFramerateLimit(60.F);


    if (p_controlWindow == nullptr)
    {
        p_controlWindow = std::make_unique<sf::RenderWindow>(sf::VideoMode(320, 980), "Training controls");
        
        if (ImGui::SFML::Init(*p_controlWindow))
            p_controlWindow->setFramerateLimit(60.F);
    }

    ImGui::StyleColorsDark();

    SetupModernDarkTheme();

    if (p_plotWindow == nullptr)
    {
        p_plotWindow = std::make_unique<sf::RenderWindow>(sf::VideoMode(320,980), "Training plots");
        
        if (ImGui::SFML::Init(*p_plotWindow))
            p_plotWindow->setFramerateLimit(60.F);
    }

    ImPlot::CreateContext();

    ImGui::StyleColorsDark();

    SetupModernDarkTheme();
}

void 
EnvironmentRenderer::SetEnvironment(ppo::model::Environment& env) noexcept
{
    p_environment = &env;

    if (p_environment != nullptr)
    {
        entt::registry const& snapshot{ p_environment->GetSnapshot() };
        if (snapshot.ctx().contains<ppo::model::sEnvironmentState>())
        {
            auto const& envState{ snapshot.ctx().get<ppo::model::sEnvironmentState>() };
            m_areawidth = envState.m_maxEast / GetScaleWorld();
            m_timestep = envState.m_dt * 1000.F;
        }
    }
}

sf::View const& 
EnvironmentRenderer::GetView(void) const noexcept
{
    return m_view;
}

float
EnvironmentRenderer::GetScaleWorld(void) const noexcept
{
    return m_scaleKm;
}

float 
EnvironmentRenderer::GetLineWidth(void) const noexcept
{
    return m_linewidth;
}

void 
EnvironmentRenderer::DrawSimuControls(void)
{
    if (p_environment == nullptr)
        return;

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14, 8));

    // --- PLAY ---
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.6f, 0.1f, 1.0f));
    if (ImGui::Button("Play"))
    {
        for (auto [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = true;

        p_environment->Play();
    }

    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    // --- PAUSE ---
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.65f, 0.2f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.75f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.75f, 0.55f, 0.15f, 1.0f));
    if (ImGui::Button("Pause"))
    {
        for (auto [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = false;

        p_environment->Pause();
    }

    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    // --- STOP ---
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.25f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.35f, 0.35f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.15f, 0.15f, 1.0f));
    if (ImGui::Button("Stop"))
    {
        for (auto [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = false;

        p_environment->Stop();
    }
    ImGui::PopStyleColor(3);

    ImGui::PopStyleVar(2);
}

void 
EnvironmentRenderer::DrawEnvControls(void)
{
    if (p_environment == nullptr)
        return;

    sf::Vector2u const sizes = getSize();

    bool refreshMaxRanges{ false };
    if (ImGui::SliderFloat("Area width (km)", &m_areawidth, 1.F, 500.F))
        refreshMaxRanges = true;

    if (ImGui::SliderFloat("Time step (ms)", &m_timestep, 1.F, 200.F))
        p_environment->SetTimeStep(m_timestep / 1000.F);

    char const* const scaleLevels[]{ "m", "km", "NM" };
    if (ImGui::Combo("Scale level", &m_currentScaleLevel, scaleLevels, IM_ARRAYSIZE(scaleLevels)))
    {
        if (m_currentScaleLevel == 0)
            m_scaleKm = 1.F;
        else if (m_currentScaleLevel == 1)
            m_scaleKm = 1000.F;
        else if (m_currentScaleLevel == 2)
            m_scaleKm = 1852.F;
        
        refreshMaxRanges = true;
    }

    if (refreshMaxRanges)
    {
        float const ratio{ static_cast<float>(sizes.y) / static_cast<float>(sizes.x) };
        float const east{ m_areawidth * GetScaleWorld() };
        float const north{ east * ratio };

        p_environment->SetMaxRange(north, east);
    }
}

void
EnvironmentRenderer::DrawRewardControls(std::shared_ptr<AgentController> agent, ppo::model::sAgentStateSnaphot const& state)
{
    if (p_environment == nullptr || agent == nullptr)
        return;

    agent->m_clamp    = state.m_common.m_rewardClamp;
    agent->m_scalings = state.m_scalings;

    if (ImGui::BeginTable("##agentinfo", 2,
        ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Properties");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Reward clamp");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::SliderFloat("##clamp", &agent->m_clamp, 0.5F, 10.F))
        {
            ppo::model::sAgentRealTimeCommand command{};
            command.m_id = agent->m_id;
            command.m_rewardClamp = agent->m_clamp;
            command.m_type = ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardClamp;
            p_environment->AddAgentCommand(command);
        }

        for (auto i = 0; i < state.m_scalings.size(); ++i)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text(state.m_rewardnames[i].c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::PushID(i);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::SliderFloat("##i",
                &agent->m_scalings[i], 0.F, 1.F))
            {
                ppo::model::sAgentRealTimeCommand command{};
                command.m_id = agent->m_id;
                command.m_rewardScaling = agent->m_scalings[i];
                command.m_rewardScalingType = i;
                command.m_type = ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardScaling;
                p_environment->AddAgentCommand(command);
            }
            ImGui::PopID();
        }


        ImGui::EndTable();
    }
}

void
EnvironmentRenderer::DrawTrainingControls(std::shared_ptr<AgentController> agent, ppo::model::sAgentStateSnaphot const& state)
{
    if (agent == nullptr || p_environment == nullptr)
        return;

    agent->m_rollout = state.m_common.m_maxStep;

    if (ImGui::SliderInt("Rollout Buffer", &agent->m_rollout, 1, 10000))
    {
        ppo::model::sAgentRealTimeCommand command{};
        command.m_id = agent->m_id;
        command.m_rolloutBuffer = agent->m_rollout;
        command.m_type = ppo::model::sAgentRealTimeCommand::eTypeCommand::RolloutBuffer;
        p_environment->AddAgentCommand(command);
    }
}

void
EnvironmentRenderer::DrawEntityInfo(entt::registry const& snapshot, entt::entity entity)
{
    if (!snapshot.any_of<ppo::model::sEntityState>(entity))
        return;

    auto const [playerState] = snapshot.view<ppo::model::sEntityState>().get(entity);

    std::string playerLabel{ "Entity" };
    playerLabel.append(std::to_string(static_cast<entt::id_type>(playerState.m_entityId)));
    if (ImGui::BeginTable(playerLabel.c_str(), 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Properties");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Entity");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text(std::to_string(static_cast<entt::id_type>(playerState.m_entityId)).c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("North");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_x);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("East");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_y);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("NorthSpeed");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_dx);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("EastSpeed");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_dy);

        if (snapshot.any_of<ppo::model::sFlightModelState>(entity))
        {
            auto const [flightmodelstate] = snapshot.view<ppo::model::sFlightModelState>().get(entity);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Fuel");
            ImGui::TableSetColumnIndex(1);
            ImGui::ProgressBar(flightmodelstate.m_fuel / flightmodelstate.m_maxfuel, ImVec2(-1, 0));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Heading");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_rotation);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Speed");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_speed);

        if (snapshot.any_of<ppo::model::sSNAState>(entity))
        {
            auto const [snastate] = snapshot.view<ppo::model::sSNAState>().get(entity);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("NoEscZone");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f", snastate.m_noescapezone);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("EscZone");
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f", snastate.m_escapezone);
        }

        ImGui::EndTable();
    }
}

void 
EnvironmentRenderer::DrawAgentInfo(ppo::model::sAgentStateSnaphot const& agentState)
{
    std::string const& agentName{ agentState.m_name };
    std::vector<float> rewards{};

    if (ImGui::BeginTable("##agentinfo", 2,
        ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Properties");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent player");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text(agentName.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent version");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%lld", agentState.m_common.m_version);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent maxstep");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%lld", agentState.m_common.m_maxStep);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent step");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%lld", agentState.m_common.m_step);

        if (0 < agentState.m_common.m_maxStep)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Agent rollout");
            ImGui::TableSetColumnIndex(1);
            ImGui::ProgressBar(static_cast<float>(agentState.m_common.m_step) / static_cast<float>(agentState.m_common.m_maxStep), ImVec2(-1, 0));
        }

        if (agentState.m_common.m_evaluationinprogress)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Evaluation progress");
            ImGui::TableSetColumnIndex(1);
            ImGui::ProgressBar(agentState.m_common.m_evaluationprogress, ImVec2(-1, 0));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Evaluation");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%lf", agentState.m_common.m_notation);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent training");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%d", agentState.m_common.m_training);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent reward");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", agentState.m_common.m_reward);

        for (auto i = 0; i <agentState.m_rewards.size(); ++i)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text(agentState.m_rewardnames[i].data());
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f", agentState.m_rewards[i]);
        }

        ImGui::EndTable();
    }
}

void
EnvironmentRenderer::DrawAgentPlot(std::shared_ptr<AgentController> agent, ppo::model::sAgentStateSnaphot const& state)
{
    if (agent == nullptr)
        return;

    if (agent->m_started && !state.m_common.m_training)
    {
        agent->m_plots[0].push(state.m_common.m_loss);
        agent->m_plots[1].push(state.m_common.m_actorloss);
        agent->m_plots[2].push(state.m_common.m_valueloss);
        agent->m_plots[3].push(state.m_common.m_entropyloss);
    }

    if (ImGui::CollapsingHeader(state.m_name.c_str()))
    {
        std::string plotTitle{state.m_name};
        plotTitle.append("-Loss");

        if (ImPlot::BeginPlot(plotTitle.c_str()))
        {
            ImPlot::SetupAxes("t", "y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Loss", agent->m_plots[0].data(), agent->m_plots[0].size());
            ImPlot::EndPlot();
        }

        plotTitle =state.m_name;
        plotTitle.append("-ActorLoss");

        if (ImPlot::BeginPlot(plotTitle.c_str()))
        {
            ImPlot::SetupAxes("t", "y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Loss", agent->m_plots[1].data(), agent->m_plots[1].size());
            ImPlot::EndPlot();
        }

        plotTitle = state.m_name;
        plotTitle.append("-ValueLoss");

        if (ImPlot::BeginPlot(plotTitle.c_str()))
        {
            ImPlot::SetupAxes("t", "y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Loss", agent->m_plots[2].data(), agent->m_plots[2].size());
            ImPlot::EndPlot();
        }     

        plotTitle = state.m_name;
        plotTitle.append("-EntropyLoss");

        if (ImPlot::BeginPlot(plotTitle.c_str()))
        {
            ImPlot::SetupAxes("t", "y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Loss", agent->m_plots[3].data(), agent->m_plots[3].size());
            ImPlot::EndPlot();
        }
    }
}

std::shared_ptr<EnvironmentRenderer::AgentController>
EnvironmentRenderer::GetAgentController(unsigned long long id)
{
    if (id == 0)
        return nullptr;

    auto Iter = p_agents.find(id);

    std::shared_ptr<AgentController> agent{ nullptr };
    if (Iter == p_agents.end())
    {
        agent = std::make_shared<AgentController>();
        agent->m_id = id;
        p_agents[id] = agent;
    }
    else
    {
        agent = Iter->second;
    }

    return agent;
}

bool
EnvironmentRenderer::Run(void)
{
    if (p_environment == nullptr 
    || !sf::RenderWindow::isOpen())
        return false;                  

    sf::Event event;
    while (this->pollEvent(event))
    {
        ImGui::SFML::ProcessEvent(*this, event);

        if (event.type == sf::Event::Closed)
        {
            if (p_controlWindow->isOpen())
                p_controlWindow->close();

            if (p_plotWindow->isOpen())
                p_plotWindow->close();

            sf::RenderWindow::close();
            ImPlot::DestroyContext();
            ImGui::SFML::Shutdown();
            return false;
        }

        if (event.type == sf::Event::MouseWheelScrolled)
        {
            if (event.mouseWheelScroll.delta > 0)
                m_view.zoom(0.9);
            else
                m_view.zoom(1.1);
        }
    }

    if (p_controlWindow->isOpen())
    {
        while (p_controlWindow->pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(*p_controlWindow, event);

            if (event.type == sf::Event::Closed)
            {
                p_controlWindow->close();
                ImGui::SFML::Shutdown(*p_controlWindow);
            }
        }
    }

    if (p_plotWindow->isOpen())
    {
        while (p_plotWindow->pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(*p_plotWindow, event);
            if (event.type == sf::Event::Closed)
            {
                p_plotWindow->close();
                ImGui::SFML::Shutdown(*p_plotWindow);
            }
        }
    }
    
    sf::Time const dt{ m_deltaClock.restart() };
    ImGui::SFML::Update(*this, dt);

    entt::registry& snapshot{ p_environment->GetSnapshot() };

    auto const& envState{ snapshot.ctx().get<ppo::model::sEnvironmentState>() };

    float sx{ envState.m_maxEast / GetScaleWorld() };
    m_linewidth = sx / 250.F;

    if (p_controlWindow->isOpen())
    {
        ImGui::SFML::Update(*p_controlWindow, dt);

        sf::Vector2u controlSizes{ p_controlWindow->getSize() };
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2((float)controlSizes.x, (float)controlSizes.y), ImGuiCond_Always);

        ImGuiWindowFlags flags{ ImGuiWindowFlags_NoTitleBar
                            | ImGuiWindowFlags_NoResize
                            | ImGuiWindowFlags_NoMove
                            | ImGuiWindowFlags_NoCollapse };

        if (ImGui::Begin("Training Controls", nullptr, flags))
        {
            ImGui::SFML::SetCurrentWindow(*p_controlWindow);

            DrawSimuControls();

            if (ImGui::CollapsingHeader("Environment"))
                DrawEnvControls();

            if (ImGui::CollapsingHeader("Entities"))
            {
                auto view = snapshot.view<entt::entity>();

                for (auto const& entity : view)
                    DrawEntityInfo(snapshot, entity);
            }

            auto agentView = snapshot.view<ppo::model::sAgentStateSnaphot>();

            for (auto entity : agentView)
            {
                auto [trainingAgentState] = agentView.get(entity) ;

                auto agent{ GetAgentController(trainingAgentState.m_id) };

                if (agent == nullptr)
                    continue;

                if (ImGui::CollapsingHeader(trainingAgentState.m_name.c_str()))
                {
                    DrawTrainingControls(agent, trainingAgentState);

                    DrawRewardControls(agent, trainingAgentState);

                    DrawAgentInfo(trainingAgentState);
                }
            }
            
        }
        ImGui::End();
    }

    if (p_plotWindow->isOpen())
    {
        ImGui::SFML::Update(*p_plotWindow, dt);
        
        sf::Vector2u plotSizes{ p_plotWindow->getSize() };
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2((float)plotSizes.x, (float)plotSizes.y), ImGuiCond_Always);

        ImGuiWindowFlags flags{ ImGuiWindowFlags_NoTitleBar
                            | ImGuiWindowFlags_NoResize
                            | ImGuiWindowFlags_NoMove
                            | ImGuiWindowFlags_NoCollapse };

        if (ImGui::Begin("Plots Displays", nullptr, flags))
        {
            ImGui::SFML::SetCurrentWindow(*p_plotWindow);

            auto agentView = snapshot.view<ppo::model::sAgentStateSnaphot>();

            for (auto entity : agentView)
            {
                auto [trainingAgentState] = agentView.get(entity);

                auto agent{ GetAgentController(trainingAgentState.m_id) };

                DrawAgentPlot(agent, trainingAgentState);
            }
        }

        ImGui::End();
    }

    ImGui::SFML::SetCurrentWindow(*this);
    sf::RenderWindow::clear();

    sf::Vector2f maxRanges{ envState.m_maxNorth, envState.m_maxEast };
    maxRanges = maxRanges / GetScaleWorld();

    // set the view
    m_view.setCenter({ 0.F, 0.F });
    m_view.setViewport({ 0.F, 0.F, 1.F, 1.F });
    m_view.setRotation(90.F);
    sf::RenderWindow::setView(m_view);

    m_fightingArea.setOutlineColor(sf::Color::Yellow);
    m_fightingArea.setFillColor(sf::Color::Transparent);
    m_fightingArea.setOutlineThickness(GetLineWidth());
    m_fightingArea.setPosition(-maxRanges/2.F);
    m_fightingArea.setOrigin(0, 0);
    m_fightingArea.setSize(maxRanges);
    sf::RenderWindow::draw(m_fightingArea);

    auto view = snapshot.view<entt::entity>();

    for (auto entity : view)
    {
        if (entity == entt::null
        || !snapshot.any_of<ppo::model::sEntityState>(entity))
            continue;

        auto [entityState] = snapshot.view<ppo::model::sEntityState>().get(entity);

        std::shared_ptr<IRenderer> renderer{ nullptr };

        if (auto Iter = p_renderers.find(entity);
            Iter == p_renderers.end())
        {          
            int  symbolheight = 30;
            int  symbolwidth = 30;
            bool symbolfilled = true;

            // creating a renderer for this entity
            switch (entityState.m_type)
            {
            case ppo::model::eEntityType::Objective:
                renderer = std::make_shared<ObjectiveRenderer>(symbolheight, symbolfilled);
                p_renderers[entity] = renderer;
                break;
            case ppo::model::eEntityType::Missile:
            case ppo::model::eEntityType::SAM:
            case ppo::model::eEntityType::Aircraft:
                renderer = std::make_shared<EntityRenderer>(symbolheight, symbolwidth, symbolfilled);
                p_renderers[entity] = renderer;
                break;
            default:
                break;
            }
        }
        else
        {
            renderer = Iter->second;
        }

        if (renderer != nullptr)
            renderer->Run(snapshot, entity);
    }

    // Retour vue default
    sf::RenderWindow::setView(sf::RenderWindow::getDefaultView());

    ImGui::SFML::Render(*this);
    sf::RenderWindow::display();     

    if (p_controlWindow->isOpen())
    {
        p_controlWindow->clear();
        ImGui::SFML::Render(*p_controlWindow);
        p_controlWindow->display();
    }

    if (p_plotWindow->isOpen())
    {
        p_plotWindow->clear();
        ImGui::SFML::Render(*p_plotWindow);
        p_plotWindow->display();
    }
    return true;
}

}