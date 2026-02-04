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

    class IAgentController
    {
    public:

        virtual ~IAgentController(void) = default;

        virtual std::string const&      GetName     (void) const noexcept = 0;
        virtual void                    SetRewardClamp(void) noexcept = 0;
        virtual float&                  GetRewardClamp(void) noexcept = 0;

        virtual void                    SetRewardScaling(unsigned char type) noexcept = 0;
        virtual int                     GetNumberOfRewards(void) const noexcept = 0;
        virtual float&                  GetRewardScaling(int type) noexcept = 0;
        virtual std::string const&      GetRewardName(unsigned char type) noexcept = 0;

        virtual int&                    GetRolloutBufferSize(void) noexcept = 0;
        virtual ppo::model::sCommonAgentState GetSnapshot (std::vector<float>& rewards) noexcept = 0;
        virtual void                    SetRolloutBufferSize(void) noexcept = 0;
        using Plot = CircularBuffer<5000>;
        using Plots = std::array<Plot, 4>;
        virtual Plots const& GetPlots(void) const noexcept = 0;

        virtual void Start(void) noexcept = 0;
        virtual void Stop(void) noexcept = 0;   
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

                std::vector<float> rewards{};
                auto const state = p_obj->GetSnapshot(rewards);
                m_rewardScalings = p_obj->GetRewardScalings();
                m_rewardNames    = p_obj->GetRewardNames();
                m_rewardClamp = state.m_rewardClamp;
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

        float& GetRewardScaling(int type) noexcept
        {
            return m_rewardScalings[type];
        };

        std::string const& GetRewardName(unsigned char type) noexcept
        {
            return m_rewardNames[type];
        };

        int                 GetNumberOfRewards(void) const noexcept
        {
            return m_rewardScalings.size();
        };

        int& 
            GetRolloutBufferSize(void) noexcept
        {
            return m_rolloutBufferSize;
        };

        ppo::model::sCommonAgentState 
            GetSnapshot(std::vector<float>& rewards) noexcept
        {
            auto const snapshot{ p_obj->GetSnapshot(rewards) };

            if (m_started && !snapshot.m_training)
            {
                m_plots[0].push(snapshot.m_loss);
                m_plots[1].push(snapshot.m_actorloss);
                m_plots[2].push(snapshot.m_valueloss);
                m_plots[3].push(snapshot.m_entropyloss);
            }

            return snapshot;
        };

        void 
            SetRolloutBufferSize(void) noexcept
        {
            if (p_obj != nullptr)
                p_obj->SetRolloutBufferSize(m_rolloutBufferSize);
        };

        Plots const& GetPlots(void) const noexcept
        {
            return m_plots;
        };

        void Start(void) noexcept final { m_started = true;};
        void Stop (void) noexcept final { m_started = false;};
    private:
        Obj*                     p_obj{ nullptr };
        std::string              m_name{ "" };
        float                    m_rewardClamp{0.F};
        int                      m_rolloutBufferSize{ 4096 };
        std::vector<float>       m_rewardScalings{};
        std::vector<std::string> m_rewardNames{};       
        Plots                    m_plots{};
        bool                     m_started{ false };
    };

    template<typename Obj>
    void                        Display(ppo::model::Entity& player, Obj* obj = nullptr, int height = 30, int width = 30, bool filled = true)
    {
        if (obj != nullptr)
            p_agentControllers.push_back(std::make_shared<AgentController<Obj>>(obj));

        Display(player, height, width, filled);
    };

    void                        Display(model::Entity& player, int height = 30, int width = 30, bool filled = true);

private:
                                EnvironmentRenderer (std::string const& title = "PLAYER ENVIRONMENT");
    
    void                        DrawSimuControls    (void);
    void                        DrawEnvControls     (model::sEnvironmentState const& state);
    void                        DrawRewardControls  (std::shared_ptr<IAgentController> agent);
    void                        DrawTrainingControls(std::shared_ptr<IAgentController> agent);
    void                        DrawEntityInfo      (ppo::model::IEntityState* const player);
    void                        DrawAgentInfo       (std::shared_ptr<IAgentController> agent);
    void                        DrawAgentPlot       (std::shared_ptr<IAgentController> agent); 

    std::unique_ptr<sf::RenderWindow>   p_controlWindow {nullptr};
    std::unique_ptr<sf::RenderWindow>   p_plotWindow    { nullptr };    
    sf::Clock                           m_deltaClock    {};
    sf::View                            m_view          {};
    sf::RectangleShape                  m_fightingArea  {};
    float                               m_scaleKm       { 1000.F };
    ppo::model::Environment*            p_environment   {nullptr};
    using Renderers = std::map<model::IEntityState*, std::shared_ptr<IRenderer>>;
    Renderers                           p_renderers     {};
    using AgentControllers = std::vector<std::shared_ptr<IAgentController>>;
    AgentControllers                    p_agentControllers{};

    int                                 m_currentScaleLevel{1};
    float                               m_areawidth        {150};
    float                               m_timestep         {20};
};

}   /* ppo::renderer */