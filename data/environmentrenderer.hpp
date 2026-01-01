#pragma once 
#include <SFML/Graphics.hpp>

#include <string>
#include <memory>
#include <map>

#include "environment.hpp"
#include "playerrenderer.hpp"

namespace ppo::environment {

class EnvironmentRenderer : 
    public sf::RenderWindow
{
public:                                      
    static EnvironmentRenderer& GetInstance        (int width=0, int height=0, std::string const& title = "PLAYER ENVIRONMENT");

                                EnvironmentRenderer (EnvironmentRenderer&)  = delete;
                                EnvironmentRenderer (EnvironmentRenderer&&) = delete;
    EnvironmentRenderer&        operator=           (EnvironmentRenderer&)  = delete;
    EnvironmentRenderer&        operator=           (EnvironmentRenderer&&) = delete;

    void                        SetEnvironment      (ppo::model::Environment& env) noexcept;
    bool                        Run                 (void);
    sf::View const&             GetView             (void) const noexcept;
    float                       GetScaleWorld       (void) const noexcept;
    float                       GetLineWidth        (void) const noexcept;
    
    class IAgentRenderer
    {
    public:

        virtual ~IAgentRenderer(void) = default;

        virtual ppo::model::sAgentState GetSnapshot(void) const noexcept = 0;
    };

    template <typename Obj>
    class AgentRenderer final :
        public IAgentRenderer
    {
    public:
        explicit AgentRenderer(Obj* obj):
            p_obj{obj}{ };

        ppo::model::sAgentState GetSnapshot(void) const noexcept
        {
            return p_obj->GetSnapshot();
        };

    private:
        Obj* p_obj{ nullptr };
    };

    template<typename Obj>
    void                        Display(ppo::model::Player& player, sf::Color const color, Obj* obj = nullptr, int height = 30, int width = 30, bool filled = true)
    {
        if (obj != nullptr)
            p_agentRenderers[player.GetPlayerId()] = std::make_shared<AgentRenderer<Obj>>(obj);

        Display(player, color, height, width, filled);
    };

    void                        Display(model::Player& player, sf::Color const color, int height = 30, int width = 30, bool filled = true);

private:
                                EnvironmentRenderer (int width, int height, std::string const& title = "PLAYER ENVIRONMENT");
    
    void                        DrawSimuControls    (void);
    void                        DrawEnvControls     (void);
    void                        DrawRewardControls  (void);
    void                        DrawTrainingControls(void);
    void                        DrawPlayerInfo      (ppo::model::IPlayerState* const player);
    void                        DrawAgentInfo       (ppo::model::PlayerId playerid, std::shared_ptr<IAgentRenderer> agent);

    std::unique_ptr<sf::RenderWindow>   p_controlWindow {nullptr};
    sf::Clock                           m_deltaClock    {};
    sf::View                            m_view          {};
    sf::RectangleShape                  m_fightingArea  {};
    float                               m_scaleKm       { 1000.F };
    ppo::model::Environment*            p_environment   {nullptr};
    using Renderers = std::map<model::IPlayerState*, std::shared_ptr<PlayerRenderer>>;
    Renderers                           p_renderers     {};
    using AgentRenderers = std::map<ppo::model::PlayerId, std::shared_ptr<IAgentRenderer>>;
    AgentRenderers                      p_agentRenderers{};
    long long                           m_rolloutBufferSize{ 4096 };
};

}   /* ppo::environment */