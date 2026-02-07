#pragma once 
#include <SFML/Graphics.hpp>

#include <string>
#include <memory>
#include <map>
#include <vector>

#include "environment.hpp"
#include "entityrenderer.hpp"
#include "objectiverenderer.hpp"
#include "trainingnet.hpp"

namespace ppo::renderer {

class EnvironmentRenderer : 
    public sf::RenderWindow
{
public:                                      
    static EnvironmentRenderer& GetInstance        (std::string const& title = "PLAYER ENVIRONMENT");

                                EnvironmentRenderer (EnvironmentRenderer&)  = delete;
                                EnvironmentRenderer (EnvironmentRenderer&&) = delete;
    EnvironmentRenderer&        operator=           (EnvironmentRenderer&)  = delete;
    EnvironmentRenderer&        operator=           (EnvironmentRenderer&&) = delete;

    void                        SetEnvironment      (ppo::model::Environment& env) noexcept;
    bool                        Run                 (void);
    sf::View const&             GetView             (void) const noexcept;
    float                       GetScaleWorld       (void) const noexcept;
    float                       GetLineWidth        (void) const noexcept;
    
    template<std::size_t N>
    struct CircularBuffer
    {
        CircularBuffer() {
            m_data.reserve(N);
            m_data.resize(N);
        };
           
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

        float const* data() const noexcept
        {
            return m_data.data();
        }

        std::size_t size() const noexcept
        {
            return m_size;
        }

        std::vector<float> m_data{};
        std::size_t m_size{ 0 };
        std::size_t m_index{ 0 };

    };

    struct AgentController {
        using Plot = CircularBuffer<5000>;
        using Plots = std::array<Plot, 4>;

        Plots               m_plots{};
        bool                m_started{ false };
        std::vector<float>  m_scalings{};
        int                 m_rollout{ 0 };
        float               m_clamp{ 0.F };
        unsigned long long  m_id{ 0 };
    };

private:
                                EnvironmentRenderer (std::string const& title = "PLAYER ENVIRONMENT");
    
    std::shared_ptr<AgentController> GetAgentController(unsigned long long id);
    void                        DrawSimuControls    (void);
    void                        DrawEnvControls     (void);
    void                        DrawRewardControls  (std::shared_ptr<AgentController> agent,  ppo::model::sAgentStateSnaphot const& state);
    void                        DrawTrainingControls(std::shared_ptr<AgentController> agent, ppo::model::sAgentStateSnaphot const& state);
    void                        DrawEntityInfo      (entt::registry const& snapshot, entt::entity entity);
    void                        DrawAgentInfo       (ppo::model::sAgentStateSnaphot const& state);
    void                        DrawAgentPlot       (std::shared_ptr<AgentController> agent, ppo::model::sAgentStateSnaphot const& state);

    std::unique_ptr<sf::RenderWindow>   p_controlWindow {nullptr};
    std::unique_ptr<sf::RenderWindow>   p_plotWindow    { nullptr };    
    sf::Clock                           m_deltaClock    {};
    sf::View                            m_view          {};
    sf::RectangleShape                  m_fightingArea  {};
    float                               m_scaleKm       { 1000.F };
    ppo::model::Environment*            p_environment   {nullptr};
    using Renderers = std::map<entt::entity, std::shared_ptr<IRenderer>>;
    Renderers                           p_renderers     {};
    std::map<unsigned long long, std::shared_ptr<AgentController>> p_agents{};

    int                                 m_currentScaleLevel{1};
    float                               m_areawidth        {150};
    float                               m_timestep         {20};
    float                               m_linewidth        {1.F};
};

}   /* ppo::renderer */