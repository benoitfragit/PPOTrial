#pragma once 
#include <SFML/Graphics.hpp>

#include <string>
#include <memory>
#include <map>
#include <vector>

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
    
    class IAgentController
    {
    public:

        virtual ~IAgentController(void) = default;

        virtual std::string const&      GetName     (void) const noexcept = 0;
        virtual void                    SetRewardClamp(void) noexcept = 0;
        virtual float&                  GetRewardClamp(void) noexcept = 0;

        virtual void                    SetRewardScaling(unsigned char type) noexcept = 0;
        virtual std::array<float, ppo::model::sAgentState::REWARDS.size()>&
                                        GetRewardScalings(void) noexcept = 0;

        virtual int&                    GetRolloutBufferSize(void) noexcept = 0;
        virtual ppo::model::sAgentState GetSnapshot (void) const noexcept = 0;
        virtual void                    StartTraining(void) noexcept = 0;
        virtual void                    StopTraining(void) noexcept = 0;
    };

    template <typename Obj>
    class AgentController final :
        public IAgentController
    {
    public:
        explicit AgentController(Obj* obj):
            p_obj{obj}
        {
            if (p_obj)
            {
                m_name = p_obj->GetName();

                auto const state = p_obj->GetSnapshot();
                m_rewardClamp = state.m_rewardClamp;
                for (auto i = 0; i < ppo::model::sAgentState::REWARDS.size(); ++i)
                    m_rewardScalings[i] = state.m_rewardScaling[i];
                m_rolloutBufferSize = state.m_maxStep;
            }
        };

        std::string const& 
            GetName(void) const noexcept
        {
            return m_name;
        };

        void                    SetRewardClamp(void) noexcept
        {
            if (p_obj != nullptr)
                p_obj->SetRewardClampValue(m_rewardClamp);
        };

        float& 
            GetRewardClamp(void) noexcept
        {
            return m_rewardClamp;
        };

        void                    SetRewardScaling(unsigned char type) noexcept
        {
            if (p_obj != nullptr)
                p_obj->SetRewardScaling(type, m_rewardScalings[type]);
        };

        std::array<float, ppo::model::sAgentState::REWARDS.size()>&
            GetRewardScalings(void) noexcept
        {
            return m_rewardScalings;
        };

        int& 
            GetRolloutBufferSize(void) noexcept
        {
            return m_rolloutBufferSize;
        };

        ppo::model::sAgentState 
            GetSnapshot(void) const noexcept
        {
            return p_obj->GetSnapshot();
        };

        void 
            StartTraining(void) noexcept
        {
            if (p_obj != nullptr)
                p_obj->StartTraining(m_rolloutBufferSize);
        };

        void StopTraining(void) noexcept
        {
            if (p_obj != nullptr)
                p_obj->StopTraining();
        };

    private:
        Obj* p_obj{ nullptr };
        std::string m_name{ "" };
        float m_rewardClamp{0.F};
        int m_rolloutBufferSize{ 4096 };
        std::array<float, ppo::model::sAgentState::REWARDS.size()> m_rewardScalings{};
    };

    template<typename Obj>
    void                        Display(ppo::model::Player& player, sf::Color const color, Obj* obj = nullptr, int height = 30, int width = 30, bool filled = true)
    {
        if (obj != nullptr)
            p_agentControllers.push_back(std::make_shared<AgentController<Obj>>(obj));

        Display(player, color, height, width, filled);
    };

    void                        Display(model::Player& player, sf::Color const color, int height = 30, int width = 30, bool filled = true);

private:
                                EnvironmentRenderer (int width, int height, std::string const& title = "PLAYER ENVIRONMENT");
    
    void                        DrawSimuControls    (void);
    void                        DrawEnvControls     (model::sEnvironmentState const& state);
    void                        DrawRewardControls  (std::shared_ptr<IAgentController> agent);
    void                        DrawTrainingControls(std::shared_ptr<IAgentController> agent);
    void                        DrawPlayerInfo      (ppo::model::IPlayerState* const player);
    void                        DrawAgentInfo       (std::shared_ptr<IAgentController> agent);

    std::unique_ptr<sf::RenderWindow>   p_controlWindow {nullptr};
    sf::Clock                           m_deltaClock    {};
    sf::View                            m_view          {};
    sf::RectangleShape                  m_fightingArea  {};
    float                               m_scaleKm       { 1000.F };
    ppo::model::Environment*            p_environment   {nullptr};
    using Renderers = std::map<model::IPlayerState*, std::shared_ptr<PlayerRenderer>>;
    Renderers                           p_renderers     {};
    using AgentControllers = std::vector<std::shared_ptr<IAgentController>>;
    AgentControllers                    p_agentControllers{};

    int                                 m_currentScaleLevel{1};
    float                               m_areawidth        {150};
    float                               m_timestep         {20};
};

}   /* ppo::environment */