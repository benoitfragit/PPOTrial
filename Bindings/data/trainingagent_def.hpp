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
#include "training_settings.hpp"
#include <stdexcept>
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
    // A single training configuration path for all policy families.
    std::filesystem::path const settingsPath =
        std::filesystem::path("Training") / (m_name + ".json");

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        settingsPath,
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
    std::vector<std::string> rewardNames{};
    rewardNames.reserve(Model::REWARDS.size());
    for (auto const& name : Model::REWARDS)
        rewardNames.emplace_back(name);

    auto const settings = ppo::serialization::ReadTrainingSettings(ar, rewardNames);
    if (settings.rewardWeights.size() != m_rewardScalings.size())
        throw std::invalid_argument("Reward.Scaling count does not match the agent's reward contract");

    m_internal.m_maxStep = settings.rolloutSteps;
    m_internal.m_checkpoint = settings.checkpointEvery;
    m_internal.m_version = settings.modelVersion;
    m_internal.m_rewardClamp = settings.rewardClamp;
    std::copy(settings.rewardWeights.begin(), settings.rewardWeights.end(),
              m_rewardScalings.begin());

    m_internal.m_gamma         = settings.gamma;
    m_internal.m_lambda_gae    = settings.gaeLambda;
    m_internal.m_clip_epsilon  = settings.clipEpsilon;
    m_internal.m_value_coef    = settings.valueCoefficient;
    m_internal.m_entropy_coef  = settings.entropyCoefficient;
    m_internal.m_learningReate = settings.learningRate;
    m_internal.m_K_epochs      = settings.epochs;
    m_internal.m_batch_size    = settings.batchSize;
    m_internal.m_target_kl     = settings.targetKL;
}

/**
* @brief Saves persisted training settings for the agent.
*/
template<typename Policy, typename Model>
void 
TrainingAgent<Policy, Model>::SaveToFile(ppo::serialization::json& ar)
{
    ppo::serialization::TrainingSettings settings{};
    settings.rolloutSteps = m_internal.m_maxStep;
    settings.checkpointEvery = m_internal.m_checkpoint;
    settings.modelVersion = m_internal.m_version;
    settings.rewardClamp = m_internal.m_rewardClamp;
    settings.rewardWeights.assign(m_rewardScalings.begin(), m_rewardScalings.end());

    settings.gamma = m_internal.m_gamma;
    settings.gaeLambda = m_internal.m_lambda_gae;
    settings.clipEpsilon = m_internal.m_clip_epsilon;
    settings.valueCoefficient = m_internal.m_value_coef;
    settings.entropyCoefficient = m_internal.m_entropy_coef;
    settings.learningRate = m_internal.m_learningReate;
    settings.epochs = m_internal.m_K_epochs;
    settings.batchSize = m_internal.m_batch_size;
    settings.targetKL = m_internal.m_target_kl;

    std::vector<std::string> rewardNames{};
    rewardNames.reserve(Model::REWARDS.size());
    for (auto const& name : Model::REWARDS)
        rewardNames.emplace_back(name);
    ppo::serialization::WriteTrainingSettings(ar, settings, rewardNames);
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
