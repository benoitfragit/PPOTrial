#pragma once

#include <array>
#include <filesystem>
#include <vector>
#include <functional>
#include <memory>

#include <torch/torch.h>

namespace ppo::model {

template<typename Policy, typename Model>
ProdAgent<Policy, Model>::ProdAgent(Model* model, std::string const& name, std::filesystem::path const& modelpath):
    p_model{model},
    m_name{name},
    m_modelpath{modelpath},
    p_policy{ std::make_shared<Policy>(m_stateVector.size(), m_actionVector.size()) }
{
    if (p_policy == nullptr)
        return;

    if (std::filesystem::exists(m_modelpath))
    {
        p_policy->load(m_modelpath);
    }
    else
    {
        p_policy.reset();
    }
}

template<typename Policy,  typename Model>
void
ProdAgent<Policy, Model>::Init(void)
{
    if (p_model == nullptr)
        return;

    p_model->SetNewState(m_stateVector);
}

template<typename Policy,  typename Model>
void
ProdAgent<Policy, Model>::ChooseAction(void)
{
    if (p_model == nullptr
    ||  p_policy == nullptr)
        return;

    torch::Tensor actions_t = p_policy->forward(m_states_t);

    // copy current action buffer into output buffer
    float const* srcActionPtr = actions_t.data_ptr<float>();
    std::memcpy(m_actionVector.data(), srcActionPtr, sizeof(float) * actions_t.size(1));

    p_model->SetNextAction(m_actionVector);
}

template<typename Policy, typename Model>
bool
ProdAgent<Policy, Model>::SetNewState(void)
{
    if (p_model == nullptr)
        return false;

    p_model->SetNewState(m_stateVector);

    return false;
}

template<typename Policy, typename Model>
bool
ProdAgent<Policy, Model>::IsTrained(void)                  const noexcept
{
    return true;
}


template<typename Policy, typename Model>
std::string const& 
ProdAgent<Policy, Model>::GetName(void)                  const noexcept
{
    return m_name;
}

} /* ppo::model */