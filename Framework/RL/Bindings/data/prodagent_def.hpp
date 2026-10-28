/**
* @file PPO/data/prodagent_def.hpp
* @brief Declarations for prodagent_def.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <vector>
#include <functional>
#include <memory>
#include <cstring>
#include <stdexcept>

#include <torch/torch.h>

namespace ppo::model {

/**
* @brief Builds the production agent and loads its serialized policy when present.
*/
template<typename Policy, typename Model>
ProdAgent<Policy, Model>::ProdAgent(Model* model, std::string const& name, std::filesystem::path const& modelpath):
    p_model{model},
    m_name{name},
    m_modelpath{modelpath},
    p_policy{ nullptr }
{
    // Production policies are now accessed through the generic IPolicy
    // contract, while retaining the unchanged model implementation.
    auto policy = std::make_unique<TorchProductionPolicy<Policy>>(
        m_stateVector.size(), m_actionVector.size());
    if (std::filesystem::exists(m_modelpath))
    {
        policy->Load(m_modelpath);
        p_policy = std::move(policy);
    }
}

/**
* @brief Initializes the production state buffer from the model.
*/
template<typename Policy,  typename Model>
void
ProdAgent<Policy, Model>::Init(void)
{
    if (p_model == nullptr)
        return;

    p_model->SetNewState(m_stateVector);
}

/**
* @brief Runs inference and copies the result into the live action buffer.
*/
template<typename Policy,  typename Model>
void
ProdAgent<Policy, Model>::ChooseAction(void)
{
    if (p_model == nullptr
    ||  p_policy == nullptr)
        return;

    ppo::core::CopyAction(
        m_actionVector, p_policy->ChooseAction(ppo::core::CopyObservation(m_stateVector)));

    p_model->SetNextAction(m_actionVector);
}

/**
* @brief Refreshes the observation buffer after one environment step.
*/
template<typename Policy, typename Model>
bool
ProdAgent<Policy, Model>::SetNewState(void)
{
    if (p_model == nullptr)
        return false;

    p_model->SetNewState(m_stateVector);

    return false;
}

/**
* @brief Reports that production agents are always inference-ready.
*/
template<typename Policy, typename Model>
bool
ProdAgent<Policy, Model>::IsTrained(void)                  const noexcept
{
    return true;
}


/**
* @brief Returns an empty snapshot because production agents expose no training state.
*/
template<typename Policy, typename Model>
sAgentStateSnaphot
ProdAgent<Policy, Model>::GetSnapshot(void) const noexcept
{
    static const sAgentStateSnaphot snapshot{};
    return snapshot;
}

} /* ppo::model */
