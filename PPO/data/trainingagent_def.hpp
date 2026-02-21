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
    p_policy{ std::make_shared<Policy>(m_stateVector.size(), m_actionVector.size()) },
    m_name{name},
    m_modelpath{modelpath},
    p_evaluator{evaluator}
{
    std::filesystem::path mypath{ m_name };
    mypath.replace_extension(".json");

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        mypath,
        [this](ppo::serialization::json& ar)
        {
            LoadFromFile(ar);

            if (p_rlagent == nullptr)
            {
                p_rlagent = std::make_unique<RLAgent>(p_policy,
                    m_modelpath,
                    m_stateVector.size(),
                    m_actionVector.size(),
                    static_cast<int64_t>(m_internal.m_maxStep),
                    m_internal,
                    m_stateVector.data(),
                    m_actionVector.data());

                if (p_rlagent != nullptr)
                    p_rlagent->Init();
            }
        },
        [this](ppo::serialization::json& ar)
        {
            SaveToFile(ar);
        }
    );
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::Init(void)
{
    if (p_model == nullptr)
        return;

    if (p_rlagent != nullptr
    &&  !p_rlagent->IsTraining())
        p_rlagent->Init();

    m_internal.m_reward = 0.F;
    m_internal.m_done = false;

    for (auto& reward : m_rewardVector)
        reward = 0.F;

    p_model->SetNewState(m_stateVector);
}

template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::ProcessCommand(sAgentRealTimeCommand const& command) noexcept
{
    bool reinitRequest{ false };

    // command is ignored if it concerns an other agent
    if (command.m_id != m_id
    ||  command.m_id == 0
    ||  command.m_type == sAgentRealTimeCommand::eTypeCommand::None)
        return;

    switch (command.m_type)
    {
        case sAgentRealTimeCommand::eTypeCommand::RewardClamp:
            m_internal.m_rewardClamp = command.m_rewardClamp;
            break;
        case sAgentRealTimeCommand::eTypeCommand::RewardScaling:
        {
            if (command.m_rewardScalingType < m_rewardScalings.size())
                m_rewardScalings[command.m_rewardScalingType] = command.m_rewardScaling;
        }
        break;
        case sAgentRealTimeCommand::eTypeCommand::RolloutBuffer:
            m_internal.m_maxStep = command.m_rolloutBuffer;
            break;
        default:
            break;
    }

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
            m_stateVector.size(),
            m_actionVector.size(),
            static_cast<int64_t>(m_internal.m_maxStep),
            m_internal,
            m_stateVector.data(),
            m_actionVector.data());
           
        if (p_rlagent != nullptr)
            p_rlagent->Init();
    }
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
    p_model->SetNextAction(m_actionVector);
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
bool
TrainingAgent<Policy, Model>::SetNewState(void)
{
    bool ret{ false };

    if (p_model == nullptr)
        return ret;

    ret = RewardFunction();

    // building the new state  vetor    
    p_model->SetNewState(m_stateVector);
    
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

            if (m_internal.m_version % m_internal.m_checkpoint == 0)
            {
                // save the model in eval
                p_rlagent->Save(m_modelpath);

                // check if we need to evaluate this new model
                if (p_evaluator != nullptr
                    && !m_internal.m_evaluationprogress)
                {
                    p_evaluator->Stop();

                    // then launch an evaluation
                    p_evaluator->Start(m_name, m_modelpath);
                }
            }

            ret = true;
        }
    }

    return ret;
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
TrainingAgent<Policy, Model>::LoadFromFile(ppo::serialization::json& ar)
{
    auto agentSection         = ar[m_name];

    m_internal.m_maxStep      = agentSection["Rollout"]   ;
    m_internal.m_checkpoint   = agentSection["Checkpoint"];
    m_internal.m_version      = agentSection["Version"]   ;

    auto rewardSection        = agentSection["Reward"];

    m_internal.m_rewardClamp   = rewardSection["Clamping"];

    m_rewardScalings           = rewardSection["Scaling"];

    auto ppoSection           = agentSection["PPO"];

    m_internal.m_gamma         = ppoSection["Gamma"         ];
    m_internal.m_lambda_gae    = ppoSection["Lambda_GAE"    ];
    m_internal.m_clip_epsilon  = ppoSection["Clip_Epsilon"  ];
    m_internal.m_value_coef    = ppoSection["Value_Coef"    ];
    m_internal.m_entropy_coef  = ppoSection["Entropy_Coef"  ];
    m_internal.m_learningReate = ppoSection["Learning_Rate" ];
    m_internal.m_K_epochs      = ppoSection["K_epochs"      ];
    m_internal.m_batch_size    = ppoSection["Batch_size"    ];
    m_internal.m_target_kl     = ppoSection["Target_KL"     ]; 
}

template<typename Policy, typename Model>
void 
TrainingAgent<Policy, Model>::SaveToFile(ppo::serialization::json& ar)
{
    auto& agentSection         = ar[m_name];

    agentSection["Rollout"]    = m_internal.m_maxStep;
    agentSection["Checkpoint"] = m_internal.m_checkpoint;
    agentSection["Version"]    = m_internal.m_version;

    auto& rewardSection        = agentSection["Reward"];

    rewardSection["Clamping"]  = m_internal.m_rewardClamp;

    rewardSection["Scaling"]   = m_rewardScalings;

    auto& ppoSection           = agentSection["PPO"];

    ppoSection["Gamma"         ] = m_internal.m_gamma        ;
    ppoSection["Lambda_GAE"    ] = m_internal.m_lambda_gae   ;
    ppoSection["Clip_Epsilon"  ] = m_internal.m_clip_epsilon ;
    ppoSection["Value_Coef"    ] = m_internal.m_value_coef   ;
    ppoSection["Entropy_Coef"  ] = m_internal.m_entropy_coef ;
    ppoSection["Learning_Rate" ] = m_internal.m_learningReate;
    ppoSection["K_epochs"      ] = m_internal.m_K_epochs     ;
    ppoSection["Batch_size"    ] = m_internal.m_batch_size   ;
    ppoSection["Target_KL"     ] = m_internal.m_target_kl    ; 
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
sAgentStateSnaphot
TrainingAgent<Policy, Model>::GetSnapshot(void) const noexcept
{
    sAgentStateSnaphot snapshot{};

    memcpy(&snapshot.m_common, &m_internal, sizeof(sCommonAgentState));

    for (auto reward : m_rewardVector)
        snapshot.m_rewards.push_back(reward);

    for (auto scaling : m_rewardScalings)
        snapshot.m_scalings.push_back(scaling);

    for (auto const& name : Model::REWARDS)
        snapshot.m_rewardnames.push_back(std::string(name));

    snapshot.m_name = m_name;
    snapshot.m_id   = m_id;

    return snapshot;
}

} /* ppo::model */