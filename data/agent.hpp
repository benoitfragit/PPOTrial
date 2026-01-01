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

    explicit                 Agent(std::string const& name);

                             ~Agent                  (void) = default;
    
    void                     Init                    (Env const& env, Model* const model);

    void                     ChooseAction            (Env const& env, Model* const model);
    void                     SetDone                 (bool done)                noexcept;
    void                     SetRewardClampValue     (float clampValue)         noexcept;
    void                     SetRewardScaling        (unsigned char type, float scaling)            noexcept;
    void                     AddReward               (unsigned char type, float reward)             noexcept;
    void                     StartTraining           (long long buffer_dim);
    void                     StopTraining            (void);
    bool                     IsTraining              (void)                  const noexcept;
    long long                GetCurrentStep          (void)                  const noexcept;
    void                     SetNewState             (Env const& env, Model* const modelid);

    typename AgentState               GetSnapshot             (void) const noexcept;

private:
    StateVector                 m_stateVector   {};
    ActionVector                m_actionVector  {};
    std::unique_ptr<RLAgent>    p_rlagent       {nullptr};
    AgentState                  m_internal      {};
    AgentState                  m_buffers[2];
    std::atomic<uint32_t>       m_index         { 0 };
    std::string                 m_name{ "" };
};

} /* ppo::model */

#include "agent_def.hpp"