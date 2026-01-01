#include "environmentrenderer.hpp"
#include <imgui.h>
#include <imgui-SFML.h>

namespace ppo::environment {

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
EnvironmentRenderer::GetInstance(int width, int height, std::string const& title)
{
    static EnvironmentRenderer renderer{ width, height, title };
    return renderer;
}

EnvironmentRenderer::EnvironmentRenderer(int width, int height, std::string const& title):
    sf::RenderWindow(sf::VideoMode(width, height), title)
{
    if (ImGui::SFML::Init(*this))
        setFramerateLimit(60.F);

    if (p_controlWindow == nullptr)
    {
        p_controlWindow = std::make_unique<sf::RenderWindow>(sf::VideoMode(640, height), "Training controls");
        
        if (ImGui::SFML::Init(*p_controlWindow))
            p_controlWindow->setFramerateLimit(60.F);
    }

    ImGui::StyleColorsDark();
    SetupModernDarkTheme();
}

void 
EnvironmentRenderer::SetEnvironment(ppo::model::Environment& env) noexcept
{
    p_environment = &env;
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
    float linewidth{ 1.F };
    if (p_environment == nullptr)
        return linewidth;

    model::sEnvironmentState state{ p_environment->GetSnapshot() };
    float sx{ state.m_maxEast / GetScaleWorld() };
    linewidth = sx / 250.F;
    return linewidth;
}

void
EnvironmentRenderer::Display(model::Player& player, sf::Color const color, int height, int width, bool filled)
{
    if (p_environment == nullptr)
        return;

    auto Iter{ p_renderers.find(&player)};
    if (Iter == p_renderers.end())
        p_renderers[&player] = std::make_shared<PlayerRenderer>(&player, height, width, color, filled);
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
        p_environment->Play();

    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    // --- PAUSE ---
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.65f, 0.2f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.75f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.75f, 0.55f, 0.15f, 1.0f));
    if (ImGui::Button("Pause"))
        p_environment->Pause();
    
    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    // --- STOP ---
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.25f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.35f, 0.35f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.15f, 0.15f, 1.0f));
    if (ImGui::Button("Stop")) 
        p_environment->Stop();
    
    ImGui::PopStyleColor(3);

    ImGui::PopStyleVar(2);
}

void 
EnvironmentRenderer::DrawEnvControls(void)
{
    if (p_environment == nullptr)
        return;

    static float areawidth        { 150.F };
    static float timestep         { 2.F };
    static int   currentScaleLevel{ 1 };
    static int   maxStep          { 4096 };

    sf::Vector2u const sizes = getSize();

    bool refreshMaxRanges{ false };
    if (ImGui::SliderFloat("Area width (km)", &areawidth, 1.F, 500.F))
        refreshMaxRanges = true;

    if (ImGui::SliderFloat("Time step (ms)", &timestep, 1.F, 200.F))
        p_environment->SetTimeStep(timestep / 1000.F);

    char const* const scaleLevels[]{ "m", "km", "NM" };
    if (ImGui::Combo("Scale level", &currentScaleLevel, scaleLevels, IM_ARRAYSIZE(scaleLevels)))
    {
        if (currentScaleLevel == 0)
            m_scaleKm = 1.F;
        else if (currentScaleLevel == 1)
            m_scaleKm = 1000.F;
        else if (currentScaleLevel == 2)
            m_scaleKm = 1852.F;
        
        refreshMaxRanges = true;
    }

    if (refreshMaxRanges)
    {
        float const ratio{ static_cast<float>(sizes.y) / static_cast<float>(sizes.x) };
        float const east{ areawidth * GetScaleWorld() };
        float const north{ east * ratio };

        p_environment->SetMaxRange(north, east);
    }
}

void
EnvironmentRenderer::DrawRewardControls(void)
{
    static float rewardClamp        { 5.F };
    static std::array<float, ppo::model::sAgentState::REWARDS.size()> RwardScalings{};

    if (ImGui::SliderFloat("Reward clamp", &rewardClamp, 0.5F, 10.F))
        p_environment->SetRewardClamp(rewardClamp);

    for (auto i = 0; i < ppo::model::sAgentState::REWARDS.size(); ++i)
    {
        if (ImGui::SliderFloat(std::string(ppo::model::sAgentState::REWARDS[i]).c_str(), &RwardScalings[i], 0.F, 1.F))
            p_environment->SetRewardScaling(i, RwardScalings[i]);
    }
}

void
EnvironmentRenderer::DrawTrainingControls(void)
{
    int value{ static_cast<int>(m_rolloutBufferSize) };
    if (ImGui::SliderInt("Rollout Buffer", &value, 1, 10000))
        m_rolloutBufferSize = value;

    static bool training{false};
    if (ImGui::Checkbox("Enable training", &training))
    {
        if (training)
            p_environment->StartTraining(m_rolloutBufferSize);
        else
            p_environment->StopTraining();
    }
}

void
EnvironmentRenderer::DrawPlayerInfo(ppo::model::IPlayerState* const player)
{
    if (player == nullptr)
        return;

    auto const playerState = player->GetSnapshot();
    std::string playerLabel{ "Player" };
    playerLabel.append(std::to_string(playerState.m_playerId));
    if (ImGui::BeginTable(playerLabel.c_str(), 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Properties");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Player");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text(std::to_string(playerState.m_playerId).c_str());

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

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Fuel");
        ImGui::TableSetColumnIndex(1);
        ImGui::ProgressBar(playerState.m_fuel/playerState.m_maxfuel, ImVec2(-1, 0));

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

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("NoEscZone");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_noescapezone);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("EscZone");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", playerState.m_escapezone);

        ImGui::EndTable();
    }
}

void 
EnvironmentRenderer::DrawAgentInfo(ppo::model::PlayerId playerId, std::shared_ptr<IAgentRenderer> agent)
{
    if (agent == nullptr)
        return;

    auto const agentState = agent->GetSnapshot();
    std::string agentLabel{ "Agent-Player" };
    agentLabel.append(std::to_string(playerId));
    if (ImGui::BeginTable(agentLabel.c_str(), 2, 
        ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Properties");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent player");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text(std::to_string(playerId).c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent maxstep");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%lld", agentState.m_maxStep);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent step");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%lld", agentState.m_step);

        if (0 < agentState.m_maxStep)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Agent rollout");
            ImGui::TableSetColumnIndex(1);
            ImGui::ProgressBar(static_cast<float>(agentState.m_step) / static_cast<float>(agentState.m_maxStep), ImVec2(-1, 0));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent training");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%d", agentState.m_training);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent backward");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%d", agentState.m_backward);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Agent reward");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", agentState.m_reward);

        for (auto i = 0; i < agentState.m_rewardVector.size(); ++i)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text(ppo::model::sAgentState::REWARDS[i].data());
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f", agentState.m_rewardVector[i]);
        }

        ImGui::EndTable();
    }
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

            sf::RenderWindow::close();
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
    
    sf::Time const dt{ m_deltaClock.restart() };
    ImGui::SFML::Update(*this, dt);
    if (p_controlWindow->isOpen())
        ImGui::SFML::Update(*p_controlWindow, dt);

    if (p_controlWindow->isOpen())
    {
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

            if (ImGui::CollapsingHeader("Players"))
            {
                for (auto const& [player, _] : p_renderers)
                {
                    if (player == nullptr)
                        continue;

                    DrawPlayerInfo(player);
                }
            }

            if (ImGui::CollapsingHeader("Training"))
            {
                DrawTrainingControls();
                DrawRewardControls();
            }

            if (ImGui::CollapsingHeader("Agents"))
            {
                for (auto const& [playerId, agent] : p_agentRenderers)
                    DrawAgentInfo(playerId, agent);
            }
        }
        ImGui::End();
    }

    ImGui::SFML::SetCurrentWindow(*this);
    sf::RenderWindow::clear();

    model::sEnvironmentState state{ p_environment->GetSnapshot() };
    sf::Vector2f maxRanges{ state.m_maxNorth, state.m_maxEast };
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

    for (auto& [player, renderer]: p_renderers)
        if (renderer != nullptr)
            renderer->Run();

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
    return true;
}

}