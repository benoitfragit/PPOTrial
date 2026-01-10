#pragma once
#include <array>
#include "rlagent.hpp"

namespace ppo::model {

template<typename Env, typename Model, typename AgentState>
class Agent
{
public:
    using StateVector  = std::array<float, Model::STATES.size()>;
    using ActionVector = std::array<float, Model::ACTIONS.size()>;

                             Agent                   (std::string const& name, std::string const& modelpath );

                             ~Agent                  (void) = default;
    
    void                     Init                    (Env const& env, Model* const model);
    bool                     SetupTraining           (void);

    void                     ChooseAction            (Env const& env, Model* const model);
    void                     SetDone                 (bool done)                noexcept;
    void                     SetRewardClampValue     (float clampValue)         noexcept;
    void                     SetRewardScaling        (unsigned char type, float scaling)            noexcept;
    void                     AddReward               (unsigned char type, float reward)             noexcept;
    void                     StartTraining           (long long buffer_dim);
    void                     StopTraining            (void);
    bool                     IsTraining              (void)                  const noexcept;
    bool                     IsLastStep              (void)                  const noexcept;
    void                     SetNewState             (Env const& env, Model* const modelid);
    std::string const&       GetName                 (void)                  const noexcept;

    typename AgentState      GetSnapshot             (void) const noexcept;

private:
    template<typename Ar>
    void                        serialize(Ar& ar);

    void                        UpdateDoubleBuffer(void);

    StateVector                 m_stateVector   {};
    ActionVector                m_actionVector  {};
    std::unique_ptr<RLAgent>    p_rlagent       {nullptr};
    AgentState                  m_internal      {};
    AgentState                  m_buffers[2];
    std::atomic<bool>           m_training{ false };
    std::atomic<long long>      m_maxRolloutBuffer{ 4096 };
    std::atomic<float>          m_rewardScaling[AgentState::REWARDS.size()];
    std::atomic<float>          m_rewardClamp{ 5.F };
    std::atomic<uint32_t>       m_index         { 0 };
    std::string                 m_name{ "" };
    std::string                 m_modelpath{ "" };
};

} /* ppo::model */

#include "agent_def.hpp"