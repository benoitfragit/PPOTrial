/**
* @file PPO/data/trainingagent_def.hpp
* @brief Declarations for trainingagent_def.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include "serializer.hpp"
#include <algorithm>
#include <array>

#include <vector>
#include <cstring>
#include <functional>
#include <string>

namespace ppo::model {

/**
* @brief Builds the training agent and registers its JSON persistence hooks.
*/
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
                bool const loadExistingModel = m_internal.m_version > 0;
                p_rlagent = std::make_unique<TorchLearner>(p_policy,
                    m_modelpath,
                    m_stateVector.size(),
                    m_actionVector.size(),
                    static_cast<int64_t>(m_internal.m_maxStep),
                    m_internal,
                    loadExistingModel);
            }
        },
        [this](ppo::serialization::json& ar)
        {
            SaveToFile(ar);
        }
    );
}

/**
* @brief Resets agent-side episode state and refreshes the observation buffer.
*/
template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::Init(void)
{
    if (p_model == nullptr)
        return;

    if (p_rlagent != nullptr
    &&  p_rlagent->RolloutComplete())
        p_rlagent->ResetRollout();

    m_internal.m_reward = 0.F;
    m_internal.m_done = false;

    for (auto& reward : m_rewardVector)
        reward = 0.F;

    p_model->SetNewState(m_stateVector);
}

/**
* @brief Applies one real-time command and rebuilds the RL backend if needed.
*/
template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::ProcessCommand(sAgentRealTimeCommand const& command) noexcept
{
    bool reinitRequest{ false };

    // Ignore commands that do not target this specific agent instance.
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
        && m_internal.m_maxStep != p_rlagent->RolloutSize())
    {
        p_rlagent.reset();
    }

    // Recreate the RL backend after rollout-size changes or first-time initialization.
    if (p_rlagent == nullptr)
    {
        bool const loadExistingModel = m_internal.m_version > 0;
        p_rlagent = std::make_unique<TorchLearner>(
            p_policy,
            m_modelpath,
            m_stateVector.size(),
            m_actionVector.size(),
            static_cast<int64_t>(m_internal.m_maxStep),
            m_internal,
            loadExistingModel);
    }
}

/**
* @brief Samples the next action and forwards it to the model.
*/
template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::ChooseAction(void)
{
    // Reset the step reward before the reward function accumulates new terms.
    m_internal.m_reward = 0.F;

    if (p_model == nullptr)
        return;

    // Query the RL backend for one new action.
    if (p_rlagent != nullptr)
    {
        auto const action = p_rlagent->ChooseAction(ppo::core::CopyObservation(m_stateVector));
        ppo::core::CopyAction(m_actionVector, action);
    }

    // Forward the chosen action to the domain model.
    p_model->SetNextAction(m_actionVector);
}

/**
* @brief Accumulates the terminal flag for the current episode.
*/
template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::SetDone(bool done)         noexcept
{
    m_internal.m_done |= done;
}

/**
* @brief Applies one scaled reward contribution to the current step.
*/
template<typename Policy, typename Model>
void
TrainingAgent<Policy, Model>::AddReward(unsigned char type, float reward)      noexcept
{
    if (type >= Model::REWARDS.size())
        return;

    m_internal.m_reward += m_rewardScalings[type]*reward;

    m_rewardVector[type] += m_rewardScalings[type] * reward;
}

/**
* @brief Computes reward, advances PPO state, and triggers evaluation when needed.
*/
template<typename Policy, typename Model>
bool
TrainingAgent<Policy, Model>::SetNewState(void)
{
    bool ret{ false };

    if (p_model == nullptr)
        return ret;

    ret = RewardFunction();

    // Rebuild the next observation after the reward function has read the current step.
    p_model->SetNewState(m_stateVector);
    
    float const clampedReward{ std::clamp(m_internal.m_reward, -m_internal.m_rewardClamp, m_internal.m_rewardClamp) };

    if (p_rlagent != nullptr)
    {
        p_rlagent->Observe(ppo::core::Transition{
            ppo::core::CopyObservation(m_stateVector), clampedReward, m_internal.m_done
        });

        m_internal.m_step     = p_rlagent->RolloutStep();
        m_internal.m_training = !p_rlagent->RolloutComplete();
        m_internal.m_reward = clampedReward;

        PPOOutputs const& ppooutputs{ p_rlagent->Outputs() };
        m_internal.m_loss        = ppooutputs.m_loss;
        m_internal.m_actorloss   = ppooutputs.m_actorloss;
        m_internal.m_valueloss   = ppooutputs.m_valueloss;
        m_internal.m_entropyloss = ppooutputs.m_entropyloss;

        if (p_evaluator != nullptr)
        {
            sEvaluatorState state = p_evaluator->GetSnapshot();
            m_internal.m_notation = state.m_notation;
            m_internal.m_latestEvaluationNotation = state.m_latestNotation;
            m_internal.m_completedEvaluations = state.m_completedCampaigns;
            m_internal.m_hasLatestEvaluation = state.m_hasLatestNotation;
            m_internal.m_latestEvaluationSuccessRate = state.m_latestSuccessRate;
            m_internal.m_latestEvaluationSurvivalRate = state.m_latestSurvivalRate;
            m_internal.m_latestEvaluationTestCount = state.m_lastCompletedTestCount;
            m_internal.m_evaluationTestsFinished = state.m_testsFinished;
            m_internal.m_evaluationStepsExecuted = state.m_stepsExecuted;
            m_internal.m_evaluationStepsPerSecond = state.m_stepsPerSecond;
            m_internal.m_evaluationElapsedSeconds = state.m_elapsedSeconds;
            m_internal.m_evaluationprogress = state.m_progress;
            m_internal.m_evaluationinprogress = state.m_inprogress;
        }

        // End-of-rollout logic includes versioning, model export, and optional evaluation.
        if (!IsTraining())
        {
            ++m_internal.m_version;

            if (m_internal.m_version % m_internal.m_checkpoint == 0)
            {
                p_rlagent->Save(m_modelpath);

                // Trigger evaluation only when no previous evaluation is still running.
                if (p_evaluator != nullptr
                    && !m_internal.m_evaluationinprogress)
                {
                    p_evaluator->Stop();
                    p_evaluator->Start(m_name, m_modelpath);
                }
            }

            ret = true;
        }
    }

    return ret;
}

/**
* @brief Returns whether PPO is currently optimizing a rollout.
*/
template<typename Policy, typename Model>
bool      
TrainingAgent<Policy, Model>::IsTraining(void)                  const noexcept
{
    if (p_rlagent == nullptr)
        return false;

    return !p_rlagent->RolloutComplete();
}

/**
* @brief Returns whether the training objective is considered complete.
*/
template<typename Policy, typename Model>
bool
TrainingAgent<Policy, Model>::IsTrained(void)                  const noexcept
{
    return false;
}

/**
* @brief Loads persisted training settings for the agent.
*/
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

/**
* @brief Saves persisted training settings for the agent.
*/
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

/**
* @brief Returns true on the last step of the current rollout.
*/
template<typename Policy, typename Model>
bool
TrainingAgent<Policy, Model>::IsLastStep(void)                  const noexcept
{
    if (p_rlagent == nullptr)
        return false;

    return (p_rlagent->RolloutSize() - p_rlagent->RolloutStep()) == 1;
}

/**
* @brief Builds a UI-friendly snapshot from the current training state.
*/
template<typename Policy, typename Model>
sAgentStateSnaphot
TrainingAgent<Policy, Model>::GetSnapshot(void) const noexcept
{
    sAgentStateSnaphot snapshot{};

    sCommonAgentState commonState{ m_internal };
    if (p_evaluator != nullptr)
    {
        sEvaluatorState const evaluatorState = p_evaluator->GetSnapshot();
        commonState.m_notation = evaluatorState.m_notation;
        commonState.m_latestEvaluationNotation = evaluatorState.m_latestNotation;
        commonState.m_completedEvaluations = evaluatorState.m_completedCampaigns;
        commonState.m_hasLatestEvaluation = evaluatorState.m_hasLatestNotation;
        commonState.m_latestEvaluationSuccessRate = evaluatorState.m_latestSuccessRate;
        commonState.m_latestEvaluationSurvivalRate = evaluatorState.m_latestSurvivalRate;
        commonState.m_latestEvaluationTestCount = evaluatorState.m_lastCompletedTestCount;
        commonState.m_evaluationTestsFinished = evaluatorState.m_testsFinished;
        commonState.m_evaluationStepsExecuted = evaluatorState.m_stepsExecuted;
        commonState.m_evaluationStepsPerSecond = evaluatorState.m_stepsPerSecond;
        commonState.m_evaluationElapsedSeconds = evaluatorState.m_elapsedSeconds;
        commonState.m_evaluationprogress = evaluatorState.m_progress;
        commonState.m_evaluationinprogress = evaluatorState.m_inprogress;
    }

    snapshot.m_common = commonState;

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
