#include "rlagent.hpp"
#include "ppo.hpp"
#include <torch/torch.h>

namespace ppo::model {

struct RLAgent::Impl 
{
    // Implementation details would go here
    Impl(long long state_dim, long long action_dim, long long buffer_dim, float* const stateVector, float* const actionVector) :
        p_stateVector{ stateVector },
        p_actionVector{ actionVector }
    {
        if (!m_randomInitTorch)
        {
            torch::manual_seed(42);
            m_randomInitTorch = true;
        }

        p_ppo = std::make_unique<PPO>(state_dim, action_dim);


        m_maxStep   = buffer_dim;
        m_step      = 0;
        m_stateDim  = state_dim;
        m_actionDim = action_dim;

        m_state_t = torch::from_blob(p_stateVector, { 1, m_stateDim }, torch::kFloat);
    };

    void Init(void);

    void ChooseAction(void);

    void SetNewState(float reward, bool done);

    void UpdatePolicy(void);

    float* p_stateVector { nullptr };
    float* p_actionVector{ nullptr };

    torch::Tensor m_state_t;

    torch::Tensor m_states     ;
    torch::Tensor m_actions    ;
    torch::Tensor m_rewards    ;
    torch::Tensor m_dones      ;
    torch::Tensor m_old_logp; 
    torch::Tensor m_values;
    torch::Tensor m_last_values;

    std::unique_ptr<PPO> p_ppo{ nullptr };

    long long m_stateDim{ 0 };
    long long m_bufferDim{ 0 };
    long long m_actionDim{ 0 };
    long long m_maxStep{ 0 };
    long long m_step{ 0 };

    inline static bool m_randomInitTorch{ false };
};

void 
RLAgent::Impl::Init(void)
{
    m_step = 0;
    // everything we need to save during a rollout sequence
    m_states        = torch::randn({ m_maxStep, m_stateDim });
    m_actions       = torch::tanh(torch::randn({ m_maxStep, m_actionDim }));
    m_rewards       = torch::randn({ m_maxStep });
    m_dones         = torch::zeros({ m_maxStep });
    m_old_logp      = torch::zeros({ m_maxStep });
    m_values        = torch::zeros({ m_maxStep });
    m_last_values   = torch::zeros({ 1 });
}

void
RLAgent::Impl::ChooseAction(void)
{
    // copy the current stateVector into the current state at given step index
    if (p_stateVector== nullptr
    ||  m_step >= m_maxStep
    ||  p_ppo == nullptr)
        return;

    torch::NoGradGuard no_grad;

    // at this step copy the current state coming from outside to the tensor
    auto [action, logpb, values] = p_ppo->SelectAction(m_state_t);

    // save data for next backward pass
    m_states[m_step]    = m_state_t.squeeze(0);
    m_actions[m_step]   = action.squeeze(0);
    m_old_logp[m_step]  = logpb.item<float>();
    m_values[m_step]    = values.item<float>();

    // copy current action buffer into output buffer
    float const* srcActionPtr = action.data_ptr<float>();
    std::memcpy(p_actionVector, srcActionPtr, sizeof(float) * action.size(1));
}

void
RLAgent::Impl::UpdatePolicy(void)
{
    if (p_ppo == nullptr)
        return;

    p_ppo->UpdatePolicy(m_states, m_actions,m_old_logp, m_values, m_rewards, m_dones, m_last_values);
}

void
RLAgent::Impl::SetNewState(float reward, bool done)
{
    if (m_step >= m_maxStep)
        return;

    m_rewards[m_step]      = reward;
    m_dones[m_step]        = done ? 1.0F : 0.0F;
    
    if (m_step < m_maxStep)
        ++m_step;

    // rollout finished, we need to update our policy
    if (m_step == m_maxStep)
    {
        // si c'est un épisode terminal reel
        if (done)
        {
            m_last_values = torch::zeros({ 1 });
        }
        else // sinon episode tronqué (boostrap)
        {
            auto [_, __, last_val] = p_ppo->SelectAction(m_state_t);
            m_last_values = last_val;
        }
        
        UpdatePolicy();
    }
}

RLAgent::RLAgent(long long state_dim, long long action_dim, long long buffer_dim, float* const stateVector, float* const actionVector) :
    p_impl{ std::make_unique<Impl>(state_dim, action_dim, buffer_dim, stateVector, actionVector) }
{
}

RLAgent::~RLAgent(void) = default;

void
RLAgent::Init(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->Init();
}

void
RLAgent::ChooseAction(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->ChooseAction();
}
     
void
RLAgent::SetNewState(float reward, bool done)
{
    if (p_impl == nullptr)
        return;

    p_impl->SetNewState(reward, done);
}

bool 
RLAgent::IsTraining(void) const noexcept
{
    return p_impl != nullptr 
        && p_impl->p_ppo != nullptr 
        && p_impl->m_step < p_impl->m_maxStep;
}

long long 
RLAgent::GetCurrentStep(void) const noexcept
{
    if (p_impl == nullptr)
        return 0;
    return p_impl->m_step;
}

long long  
RLAgent::GetMaxStep(void) const noexcept
{
    if (p_impl == nullptr)
        return 0;
    return p_impl->m_maxStep;
}

}