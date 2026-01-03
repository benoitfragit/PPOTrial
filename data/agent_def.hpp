#pragma once
#include "interface.hpp"
#include "serializer.hpp"
#include <array>

#include <vector>
#include <functional>

namespace ppo::model {

template<typename Env, typename Model, typename AgentState>
Agent<Env, Model, AgentState>::Agent(std::string const& name, std::string const& modelpath):
    m_name{name},
    m_modelpath{modelpath}
{
    m_index.store(0, std::memory_order_release);

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        [this](ppo::serialization::TomlReadArchive& ar)
        {
            serialize(ar);

            for (auto i = 0; i < AgentState::REWARDS.size(); ++i)
                m_rewardScaling[i] = m_internal.m_rewardScaling[i];

            m_rewardClamp = m_internal.m_rewardClamp;
            m_maxRolloutBuffer = m_internal.m_maxStep;
            m_internal.m_ppoparameters.m_bestreward = m_internal.m_bestreward;

            UpdateDoubleBuffer();
        },
        [this](ppo::serialization::TomlWriteArchive& ar)
        {
            serialize(ar);
        }
    );
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

template<typename Env, typename Model, typename AgentState>
bool
Agent<Env, Model, AgentState>::SetupTraining(void)
{
    bool reinitRequest{ false };

    bool const enableTraining { m_training.load(std::memory_order_acquire) };
    std::array<float, sAgentState::REWARDS.size()> rewardScaling{};

    m_internal.m_maxStep     = m_maxRolloutBuffer.load(std::memory_order_acquire);
    m_internal.m_rewardClamp = m_rewardClamp.load(std::memory_order_acquire);

    for (auto i = 0; i < sAgentState::REWARDS.size(); ++i)
        m_internal.m_rewardScaling[i] = m_rewardScaling[i].load(std::memory_order_acquire);

    if (enableTraining)
    {
        if (p_rlagent != nullptr
            && m_internal.m_maxStep != p_rlagent->GetMaxStep())
        {
            p_rlagent.reset();
        }

        // if the RL agent is not create thent create one
        if (p_rlagent == nullptr)
        {
            p_rlagent = std::make_unique<RLAgent>(
                m_modelpath,
                Model::STATES.size(),
                Model::ACTIONS.size(),
                static_cast<int64_t>(m_internal.m_maxStep),
                m_internal.m_ppoparameters,
                m_stateVector.data(),
                m_actionVector.data());
            
            reinitRequest = true;
        }

        if (p_rlagent != nullptr)
            reinitRequest |= !p_rlagent->IsTraining();
    }
    else if (p_rlagent != nullptr)
    {
        p_rlagent.reset();
    }

    return reinitRequest;
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
Agent<Env, Model, AgentState>::StartTraining(long long buffer_dim)
{
    m_maxRolloutBuffer.store(buffer_dim, std::memory_order_release);
    m_training.store(true, std::memory_order_release);
}

template<typename Env, typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::StopTraining(void)
{
    m_training.store(false, std::memory_order_release);
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

        m_internal.m_step     = p_rlagent->GetCurrentStep();
        m_internal.m_training = p_rlagent->IsTraining();
        m_internal.m_reward = clampedReward;

        PPOOutputs const& ppooutputs{ p_rlagent->GetOutputs() };
        m_internal.m_loss        = ppooutputs.m_loss;
        m_internal.m_actorloss   = ppooutputs.m_actorloss;
        m_internal.m_valueloss   = ppooutputs.m_valueloss;
        m_internal.m_entropyloss = ppooutputs.m_entropyloss;
        m_internal.m_bestreward  = ppooutputs.m_bestreward;
        m_internal.m_version     = ppooutputs.m_version;
    }

    UpdateDoubleBuffer();
}

template<typename Env, typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::UpdateDoubleBuffer(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_step = m_internal.m_step;
    m_buffers[next].m_maxStep = m_internal.m_maxStep;
    m_buffers[next].m_training = m_internal.m_training;
    m_buffers[next].m_rewardClamp = m_internal.m_rewardClamp;
    m_buffers[next].m_done = m_internal.m_done;
    m_buffers[next].m_reward = m_internal.m_reward;

    if (p_rlagent != nullptr)
    {
        m_buffers[next].m_loss        = m_internal.m_loss       ;
        m_buffers[next].m_actorloss   = m_internal.m_actorloss  ;
        m_buffers[next].m_valueloss   = m_internal.m_valueloss  ;
        m_buffers[next].m_entropyloss = m_internal.m_entropyloss;
        m_buffers[next].m_bestreward  = m_internal.m_bestreward ;
        m_buffers[next].m_version     = m_internal.m_version;
    }

    for (auto i = 0; i < AgentState::REWARDS.size(); ++i)
    {
        m_buffers[next].m_rewardVector[i] = m_internal.m_rewardVector[i];
        m_buffers[next].m_rewardScaling[i] = m_internal.m_rewardScaling[i];
    }

    m_buffers[next].m_ppoparameters.m_gamma         = m_internal.m_ppoparameters.m_gamma         ;
    m_buffers[next].m_ppoparameters.m_lambda_gae    = m_internal.m_ppoparameters.m_lambda_gae    ;
    m_buffers[next].m_ppoparameters.m_clip_epsilon  = m_internal.m_ppoparameters.m_clip_epsilon  ;
    m_buffers[next].m_ppoparameters.m_value_coef    = m_internal.m_ppoparameters.m_value_coef    ;
    m_buffers[next].m_ppoparameters.m_entropy_coef  = m_internal.m_ppoparameters.m_entropy_coef  ;
    m_buffers[next].m_ppoparameters.m_learningReate = m_internal.m_ppoparameters.m_learningReate ;
    m_buffers[next].m_ppoparameters.m_K_epochs      = m_internal.m_ppoparameters.m_K_epochs      ;
    m_buffers[next].m_ppoparameters.m_batch_size    = m_internal.m_ppoparameters.m_batch_size    ;
    m_buffers[next].m_ppoparameters.m_checkpoint    = m_internal.m_ppoparameters.m_checkpoint    ;
    m_buffers[next].m_ppoparameters.m_bestreward    = m_internal.m_ppoparameters.m_bestreward;

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
void
Agent<Env, Model, AgentState>::SetRewardClampValue(float clampValue)         noexcept
{
    m_rewardClamp.store(clampValue, std::memory_order_relaxed);
}

template<typename Env, typename Model, typename AgentState>
void
Agent<Env, Model, AgentState>::SetRewardScaling(unsigned char type, float scaling)            noexcept
{
    if (type < AgentState::REWARDS.size())
        m_rewardScaling[type].store(scaling, std::memory_order_relaxed);
}

template<typename Env, typename Model, typename AgentState>
typename AgentState 
Agent<Env, Model, AgentState>::GetSnapshot(void) const noexcept
{
    return m_buffers[m_index.load(std::memory_order_acquire)];
}

template<typename Env, typename Model, typename AgentState>
template<typename Ar>
void 
Agent<Env, Model, AgentState>::serialize(Ar& ar)
{
    auto agentSection = ar.begin_section(m_name.c_str());

    agentSection("Rollout", m_internal.m_maxStep);

    auto rewardSection = agentSection.begin_section("Reward");

    rewardSection("Clamping", m_internal.m_rewardClamp);

    rewardSection("Scaling", m_internal.m_rewardScaling);

    rewardSection("Best", m_internal.m_bestreward);

    auto ppoSection = agentSection.begin_section("PPO");

    ppoSection("Gamma",         m_internal.m_ppoparameters.m_gamma        );
    ppoSection("Lambda_GAE",    m_internal.m_ppoparameters.m_lambda_gae   );
    ppoSection("Clip_Epsilon",  m_internal.m_ppoparameters.m_clip_epsilon );
    ppoSection("Value_Coef",    m_internal.m_ppoparameters.m_value_coef   );
    ppoSection("Entropy_Coef",  m_internal.m_ppoparameters.m_entropy_coef );
    ppoSection("Learning_Rate", m_internal.m_ppoparameters.m_learningReate);
    ppoSection("K_epochs",      m_internal.m_ppoparameters.m_K_epochs     );
    ppoSection("Batch_size",    m_internal.m_ppoparameters.m_batch_size   );
    ppoSection("Checkpoint",    m_internal.m_ppoparameters.m_checkpoint   );
}

template<typename Env, typename Model, typename AgentState>
std::string const& 
Agent<Env, Model, AgentState>::GetName(void)                  const noexcept
{
    return m_name;
}


} /* ppo::model */