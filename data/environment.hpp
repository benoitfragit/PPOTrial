#pragma once 
#include "interface.hpp"
#include "player.hpp"

#include<string>
#include <memory>
#include <array>

namespace ppo::model {

class Environment {
public:
    template <typename Obj>
    class AgentWrapper final :
        public IAgent
    {
    public:
        explicit AgentWrapper(Obj* const obj) :
            m_obj{ obj }{};

        void                Init(sEnvironmentState const& environment, Player* const player)
        {
            if (m_obj == nullptr)
                return;

            m_obj->Init(environment, player);
        };

        void                ChooseAction(sEnvironmentState const& environment, Player* const player)
        {
            if (m_obj == nullptr)
                return;
            m_obj->ChooseAction(environment, player);
        };

        void                AddReward(unsigned char type, float reward)        noexcept
        {
            if (m_obj == nullptr)
                return;
            m_obj->AddReward(type, reward);
        };

        void                SetDone(bool done)             noexcept 
        {
            if (m_obj == nullptr)
                return;
            m_obj->SetDone(done);
        };

        void                SetNewState(sEnvironmentState const& environment, Player* const player)
        {
            if (m_obj == nullptr)
                return;
            m_obj->SetNewState(environment, player);
        };

        bool                SetupTraining(void)  noexcept
        {
            if (m_obj == nullptr)
                return false;

            return m_obj->SetupTraining();
        };

        bool IsLastStep(void)                  const noexcept
        {
            if (m_obj == nullptr)
                return false;

            return m_obj->IsLastStep();
        };

    private:
        Obj* const m_obj;
    };

                      Environment     (void);

    virtual           ~Environment    (void);

                      Environment           (Environment&)  = delete;
                      Environment           (Environment&&) = delete;
    Environment&      operator=             (Environment&)  = delete;
    Environment&      operator=             (Environment&&) = delete;

    template<typename T>
    void              AddAgent(Player* const player, T* const obj = nullptr)
    {
        if (obj != nullptr)
            AddAgent(player, std::make_shared<AgentWrapper<T>>(obj));
        else
            AddAgent(player);
    };

    void              AddAgent(Player* const player, std::shared_ptr<IAgent> pAgent = nullptr);

    sEnvironmentState GetSnapshot           (void) const noexcept;

    void              Start                 (void);
    void              Play                  (void);
    void              Pause                 (void);
    void              Stop                  (void);

    void              SetMaxRange           (float north, float east);
    void              SetTimeStep           (float timestep);

private:
    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

}   /* ppo::model */            