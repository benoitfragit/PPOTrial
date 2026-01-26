#pragma once
#include "serializer.hpp"
#include <array>

#include <vector>
#include <functional>
#include <string>

namespace ppo::model {

template<typename Policy, typename Model>
TrainingAgent<Policy, Model>::TrainingAgent(Model* model, 
                                            std::string const& name, 
                                            std::string const& modelpath, 
                                            IEvaluator* const evaluator):
    p_model{model},
    p_policy{ std::make_shared<Policy>(Model::STATES.size(), Model::ACTIONS.size()) },
    m_name{name},
    m_modelpath{modelpath},
    p_evaluator{evaluator}
{
    m_index.store(0, std::memory_order_release);

    std::filesystem::path mypath{ m_name };
    mypath.replace_extension(".toml");

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        mypath,
        [this](ppo::serialization::TomlReadArchive& ar)
        {
            serialize(ar);

            for (auto i = 0; i < Model::REWARDS.size(); ++i)
                m_rewardScaling[i] = m_rewardScalings[i];

            m_rewardClamp = m_internal.m_rewardClamp;
            m_maxRolloutBuffer = m_internal.m_maxStep;

            UpdateDoubleBuffer();
        },
        [this](ppo::serialization::TomlWriteArchive& ar)
        {
            serialize(ar);
        }
    );
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::Init(void)
{
    if (p_model == nullptr)
        return;

    m_internal.m_reward = 0.F;
    m_internal.m_done = false;

    for (auto& reward : m_rewardVector)
        reward = 0.F;

    for (int i = 0; i < Model::STATES.size(); ++i)
        m_stateVector[i] = (p_model->*Model::STATES[i])();
}

template<typename Policy, typename Model>
bool
TrainingAgent<Policy, Model>::Update(void) noexcept
{
    bool reinitRequest{ false };

    std::array<float, Model::REWARDS.size()> rewardScaling{};

    m_internal.m_maxStep     = m_maxRolloutBuffer.load(std::memory_order_acquire);
    m_internal.m_rewardClamp = m_rewardClamp.load(std::memory_order_acquire);

    for (auto i = 0; i < Model::REWARDS.size(); ++i)
        m_rewardScalings[i] = m_rewardScaling[i].load(std::memory_order_acquire);

    if (p_rlagent != nullptr
        && m_internal.m_maxStep != p_rlagent->GetMaxStep())
    {
        p_rlagent.reset();
    }

    // if the RL agent is not create thent create one
    if (p_rlagent == nullptr)
    {
        p_rlagent = std::make_unique<RLAgent>(
            p_policy,
            m_modelpath,
            Model::STATES.size(),
            Model::ACTIONS.size(),
            static_cast<int64_t>(m_internal.m_maxStep),
            m_internal,
            m_stateVector.data(),
            m_actionVector.data());
            
        reinitRequest = true;
    }

    if (p_rlagent != nullptr)
    {
        reinitRequest |= !p_rlagent->IsTraining();
            
        if (reinitRequest)
            p_rlagent->Init();
    }

    return reinitRequest;
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::ChooseAction(void)
{
    // Compute the input state vector to pass to the model before selecting action
    m_internal.m_reward = 0.F;

    if (p_model == nullptr)
        return;

    // Run the RL model to get the action
    if (p_rlagent != nullptr)
        p_rlagent->ChooseAction();

    // Apply the action to the player
    for (int i = 0; i < Model::ACTIONS.size(); ++i)
        (p_model->*Model::ACTIONS[i])(m_actionVector[i]);
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::SetDone(bool done)         noexcept
{
    m_internal.m_done |= done;
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::AddReward(unsigned char type, float reward)      noexcept
{
    if (type >= Model::REWARDS.size())
        return;

    m_internal.m_reward += m_rewardScalings[type]*reward;

    m_rewardVector[type] += m_rewardScalings[type] * reward;
}

template<typename Policy, typename Model>
void 
TrainingAgent<Policy, Model>::SetRolloutBufferSize(long long buffer_dim)
{
    m_maxRolloutBuffer.store(buffer_dim, std::memory_order_release);
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::StopTraining(void)
{
    if (p_evaluator != nullptr)
        p_evaluator->Stop();
}

template<typename Policy, typename Model>
bool
TrainingAgent<Policy, Model>::SetNewState(void)
{
    bool ret{ false };

    if (p_model == nullptr)
        return ret;

    ret = RewardFunction();

    // building the new state  vetor
    for (int i = 0; i < Model::STATES.size(); ++i)
        m_stateVector[i] = (p_model->*Model::STATES[i])();
    
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

        if (p_evaluator != nullptr)
        {
            sEvaluatorState state = p_evaluator->GetSnapshot();
            m_internal.m_notation = state.m_notation;
            m_internal.m_evaluationprogress = state.m_progress;
            m_internal.m_evaluationinprogress = state.m_inprogress;
        }

        // if we reach the end of a rollout
        if (!IsTraining())
        {
            // increase version number
            ++m_internal.m_version;

            // check if we need to evaluate this new model
            if (p_evaluator != nullptr
            &&   m_internal.m_version % m_internal.m_checkpoint == 0
            &&  !m_internal.m_evaluationprogress)
            {
                p_evaluator->Stop();

                // save the model in eval
                p_rlagent->Save(m_modelpath);

                // then launch an evaluation
                p_evaluator->Start(m_name, m_modelpath);
            }
        }
    }

    UpdateDoubleBuffer();

    return ret;
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::UpdateDoubleBuffer(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_step = m_internal.m_step;
    m_buffers[next].m_maxStep = m_internal.m_maxStep;
    m_buffers[next].m_checkpoint = m_internal.m_checkpoint;
    m_buffers[next].m_training = m_internal.m_training;
    m_buffers[next].m_rewardClamp = m_internal.m_rewardClamp;
    m_buffers[next].m_done = m_internal.m_done;
    m_buffers[next].m_reward = m_internal.m_reward;
    m_buffers[next].m_notation             = m_internal.m_notation            ;
    m_buffers[next].m_evaluationprogress   = m_internal.m_evaluationprogress  ;
    m_buffers[next].m_evaluationinprogress = m_internal.m_evaluationinprogress;
    m_buffers[next].m_version = m_internal.m_version;

    if (p_rlagent != nullptr)
    {
        m_buffers[next].m_loss        = m_internal.m_loss       ;
        m_buffers[next].m_actorloss   = m_internal.m_actorloss  ;
        m_buffers[next].m_valueloss   = m_internal.m_valueloss  ;
        m_buffers[next].m_entropyloss = m_internal.m_entropyloss;
    }

    for (auto i = 0; i < Model::REWARDS.size(); ++i)
    {
        m_bufferRewardVector[next][i]  = m_rewardVector[i];
        m_bufferRewardScaling[next][i] = m_rewardScalings[i];
    }

    m_buffers[next].m_gamma         = m_internal.m_gamma         ;
    m_buffers[next].m_lambda_gae    = m_internal.m_lambda_gae    ;
    m_buffers[next].m_clip_epsilon  = m_internal.m_clip_epsilon  ;
    m_buffers[next].m_value_coef    = m_internal.m_value_coef    ;
    m_buffers[next].m_entropy_coef  = m_internal.m_entropy_coef  ;
    m_buffers[next].m_learningReate = m_internal.m_learningReate ;
    m_buffers[next].m_K_epochs      = m_internal.m_K_epochs      ;
    m_buffers[next].m_batch_size    = m_internal.m_batch_size    ;
    m_buffers[next].m_target_kl     = m_internal.m_target_kl;

    m_index.store(next, std::memory_order_release);
}

template<typename Policy, typename Model>
bool      
TrainingAgent<Policy, Model>::IsTraining(void)                  const noexcept
{
    if (p_rlagent == nullptr)
        return false;

    return p_rlagent->IsTraining();
}

template<typename Policy, typename Model>
bool
TrainingAgent<Policy, Model>::IsTrained(void)                  const noexcept
{
    return false;
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::SetRewardClampValue(float clampValue)         noexcept
{
    m_rewardClamp.store(clampValue, std::memory_order_relaxed);
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::SetRewardScaling(unsigned char type, float scaling)            noexcept
{
    if (type < Model::REWARDS.size())
        m_rewardScaling[type].store(scaling, std::memory_order_relaxed);
}

template<typename Policy, typename Model>
sCommonAgentState
TrainingAgent<Policy, Model>::GetSnapshot(std::vector<float>& rewards) const noexcept
{
    uint32_t idx = m_index.load(std::memory_order_acquire);
    sCommonAgentState state = m_buffers[idx];

    for (auto i = 0; i < Model::REWARDS.size(); ++i)
        rewards.push_back(m_bufferRewardVector[idx][i]);

    return state;
}

template<typename Policy, typename Model>
template<typename Ar>
void 
TrainingAgent<Policy, Model>::serialize(Ar& ar)
{
    auto agentSection = ar.begin_section(m_name.c_str());

    agentSection("Rollout", m_internal.m_maxStep);
    agentSection("Checkpoint", m_internal.m_checkpoint);
    agentSection("Version", m_internal.m_version);

    auto rewardSection = agentSection.begin_section("Reward");

    rewardSection("Clamping", m_internal.m_rewardClamp);

    rewardSection("Scaling", m_rewardScalings);

    auto ppoSection = agentSection.begin_section("PPO");

    ppoSection("Gamma",         m_internal.m_gamma        );
    ppoSection("Lambda_GAE",    m_internal.m_lambda_gae   );
    ppoSection("Clip_Epsilon",  m_internal.m_clip_epsilon );
    ppoSection("Value_Coef",    m_internal.m_value_coef   );
    ppoSection("Entropy_Coef",  m_internal.m_entropy_coef );
    ppoSection("Learning_Rate", m_internal.m_learningReate);
    ppoSection("K_epochs",      m_internal.m_K_epochs     );
    ppoSection("Batch_size",    m_internal.m_batch_size   );
    ppoSection("Target_KL",     m_internal.m_target_kl    ); 

}

template<typename Policy, typename Model>
std::string const& 
TrainingAgent<Policy, Model>::GetName(void)                  const noexcept
{
    return m_name;
}

template<typename Policy, typename Model>
bool
TrainingAgent<Policy, Model>::IsLastStep(void)                  const noexcept
{
    if (p_rlagent == nullptr)
        return false;

    return (p_rlagent->GetMaxStep() - p_rlagent->GetCurrentStep()) == 1;
}

template<typename Policy, typename Model>
std::vector<float>       
TrainingAgent<Policy, Model>::GetRewardScalings(void) const noexcept
{
    std::vector<float> scalings{};

    for (auto const val : m_rewardScalings)
        scalings.push_back(val);

    return scalings;
}

template<typename Policy, typename Model>
std::vector<std::string> 
TrainingAgent<Policy, Model>::GetRewardNames(void) const noexcept
{
    std::vector<std::string> names{};

    for (auto const& name : Model::REWARDS)
        names.push_back(std::string(name));

    return names;
}

} /* ppo::model */