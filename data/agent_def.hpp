#pragma once
#include "interface.hpp"
#include <array>

#include <vector>
#include <functional>

namespace ppo::model {

template<typename Env, typename Model, typename AgentState>
Agent<Env, Model, AgentState>::Agent(std::string const& name):
    m_name{name}
{
    m_index.store(0, std::memory_order_release);
}

template<typename Env,  typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::Init(Env const& env, Model* const model)
{
    if (model == nullptr)
        return;

    m_internal.m_reward = 0.F;
    m_internal.m_done = false;

    for (auto& reward : m_internal.m_rewardVector)
        reward = 0.F;

    for (int i = 0; i < Model::STATES.size(); ++i)
        m_stateVector[i] = (model->*Model::STATES[i])(env);

    if (p_rlagent != nullptr)
        p_rlagent->Init();
}

template<typename Env,  typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::ChooseAction(Env const& env, Model* const model)
{
    // Compute the input state vector to pass to the model before selecting action
    m_internal.m_reward = 0.F;

    if (model == nullptr)
        return;

    // Run the RL model to get the action
    if (p_rlagent != nullptr)
        p_rlagent->ChooseAction();

    // Apply the action to the player
    for (int i = 0; i < Model::ACTIONS.size(); ++i)
        (model->*Model::ACTIONS[i])(m_actionVector[i]);
}

template<typename Env,  typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::SetDone(bool done)         noexcept
{
    m_internal.m_done |= done;
}

template<typename Env,  typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::AddReward(unsigned char type, float reward)      noexcept
{
    if (type >= AgentState::REWARDS.size())
        return;

    m_internal.m_reward += m_internal.m_rewardScaling[type]*reward;

    m_internal.m_rewardVector[type] += m_internal.m_rewardScaling[type] * reward;
}

template<typename Env, typename Model, typename AgentState>
void  
Agent<Env, Model, AgentState>::SetRewardScaling(unsigned char type, float scaling)            noexcept
{
    if (type < m_internal.m_rewardScaling.size())
        m_internal.m_rewardScaling[type] = scaling;
}

template<typename Env, typename Model, typename AgentState>
void 
Agent<Env, Model, AgentState>::StartTraining(long long buffer_dim)
{
    // if the RL agent is not create thent create one
    if (p_rlagent == nullptr)
    {
        p_rlagent = std::make_unique<RLAgent>(  Model::STATES.size(),
                                                Model::ACTIONS.size(),
                                                static_cast<int64_t>(buffer_dim),
                                                m_stateVector.data(),
                                                m_actionVector.data());
    }

    if (p_rlagent != nullptr)
    {
        // reinit of the RL model
        p_rlagent->Init();
    }

    // reinit of all indicators
    std::fill(m_internal.m_rewardVector.begin(), m_internal.m_rewardVector.end(), 0.F);
}

template<typename Env, typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::StopTraining(void)
{
    if (p_rlagent == nullptr)
        return;

    p_rlagent.reset();
}

template<typename Env, typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::SetNewState(Env const& env, Model* const model)
{
    if (model == nullptr)
        return;

    // building the new state  vetor
    for (int i = 0; i < Model::STATES.size(); ++i)
        m_stateVector[i] = (model->*Model::STATES[i])(env);
    
    float const clampedReward{ std::clamp(m_internal.m_reward, -m_internal.m_rewardClamp, m_internal.m_rewardClamp) };

    if (p_rlagent != nullptr)
    {
        p_rlagent->SetNewState(clampedReward, m_internal.m_done);

        m_internal.m_step    = p_rlagent->GetCurrentStep();
        m_internal.m_maxStep = p_rlagent->GetMaxStep();
        m_internal.m_training = p_rlagent->IsTraining();
    }
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_step          = m_internal.m_step;
    m_buffers[next].m_maxStep       = m_internal.m_maxStep;
    m_buffers[next].m_training      = m_internal.m_training;
    m_buffers[next].m_rewardClamp   = m_internal.m_rewardClamp;
    m_buffers[next].m_done          = m_internal.m_done;
    m_buffers[next].m_backward      = m_internal.m_done;
    m_buffers[next].m_reward        = clampedReward;

    for (auto i = 0; i < AgentState::REWARDS.size(); ++i)
    {
        m_buffers[next].m_rewardVector[i] = m_internal.m_rewardVector[i];
        m_buffers[next].m_rewardScaling[i] = m_internal.m_rewardScaling[i];
    }

    m_index.store(next, std::memory_order_release);
}

template<typename Env, typename Model, typename AgentState>
bool      
Agent<Env, Model, AgentState>::IsTraining(void)                  const noexcept
{
    if (p_rlagent == nullptr)
        return false;

    return p_rlagent->IsTraining();
}

template<typename Env, typename Model, typename AgentState>
long long 
Agent<Env, Model, AgentState>::GetCurrentStep(void)                  const noexcept
{
    if (p_rlagent == nullptr)
        return 0;
    return p_rlagent->GetCurrentStep();
}

template<typename Env, typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::SetRewardClampValue(float clampValue)         noexcept
{
    m_internal.m_rewardClamp = clampValue;
}

template<typename Env, typename Model, typename AgentState>
typename AgentState 
Agent<Env, Model, AgentState>::GetSnapshot(void) const noexcept
{
    return m_buffers[m_index.load(std::memory_order_acquire)];
}


} /* ppo::model */