/**
 * @file Examples/Airspace/Agents/Bindings/data/trainingagent_def.hpp
 * @brief Implementation of TrainingAgent.
 */
#pragma once

#include "training_settings.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ppo::model {

namespace detail {

/** @brief Reward names of a model, in its order. */
template<typename Model>
std::vector<std::string> RewardNames()
{
    std::vector<std::string> names;
    names.reserve(Model::REWARDS.size());
    for (auto const& name : Model::REWARDS)
        names.emplace_back(name);
    return names;
}

} // namespace detail

template<typename Model>
TrainingAgent<Model>::TrainingAgent(Model* model, std::string const& name) : p_model{model}, m_name{name}
{
    externalStep_.observation.assign(ObservationSize, 0.F);
    externalStep_.action.assign(ActionSize, 0.F);
    externalStep_.channels.assign(Model::REWARDS.size(), 0.0);
    // One settings file per agent family, read by Serializer::Load.
    std::filesystem::path const settingsPath = std::filesystem::path("Training") / (m_name + ".json");
    settingsRegistration_ = ppo::serialization::Serializer::GetInstance().Register(
        settingsPath, [this](ppo::serialization::json& ar) { LoadFromFile(ar); });
}

template<typename Model>
TrainingAgent<Model>::~TrainingAgent()
{
    ppo::serialization::Serializer::GetInstance().Unregister(settingsRegistration_);
}

template<typename Model>
void TrainingAgent<Model>::Init(void)
{
    if (p_model == nullptr)
        return;
    reward_ = 0.F;
    done_ = false;
    p_model->SetNewState(m_stateVector);
    std::copy(m_stateVector.begin(), m_stateVector.end(), externalStep_.observation.begin());
    externalStep_.terminated = false;
    externalStep_.ended = false;
}

template<typename Model>
void TrainingAgent<Model>::ChooseAction(void)
{
    reward_ = 0.F;
    std::fill(stepChannels_.begin(), stepChannels_.end(), 0.F);
    if (p_model == nullptr)
        return;
    std::copy(externalStep_.action.begin(), externalStep_.action.end(), m_actionVector.begin());
    p_model->SetNextAction(m_actionVector);
}

template<typename Model>
void TrainingAgent<Model>::AddReward(unsigned char type, float reward) noexcept
{
    if (type >= Model::REWARDS.size())
        return;
    reward_ += m_rewardScalings[type] * reward;
    stepChannels_[type] += m_rewardScalings[type] * reward;
}

template<typename Model>
bool TrainingAgent<Model>::SetNewState(void)
{
    if (p_model == nullptr)
        return false;
    bool const ended = RewardFunction();
    // The next observation is read after the reward function has seen the current step.
    p_model->SetNewState(m_stateVector);
    reward_ = std::clamp(reward_, -rewardClamp_, rewardClamp_);
    PublishStep(ended);
    return ended;
}

template<typename Model>
void TrainingAgent<Model>::PublishStep(bool ended) noexcept
{
    std::copy(m_stateVector.begin(), m_stateVector.end(), externalStep_.observation.begin());
    externalStep_.reward = reward_;
    std::copy(stepChannels_.begin(), stepChannels_.end(), externalStep_.channels.begin());
    externalStep_.terminated = done_;
    externalStep_.ended = ended || done_;
}

template<typename Model>
std::size_t TrainingAgent<Model>::RolloutSteps() const noexcept
{
    return static_cast<std::size_t>(std::max<long long>(1, rolloutSteps_));
}

template<typename Model>
void TrainingAgent<Model>::SetRewardWeights(double const* weights) noexcept
{
    for (std::size_t channel = 0; channel < m_rewardScalings.size(); ++channel)
    {
        m_rewardScalings[channel] = static_cast<float>(weights[channel]);
    }
}

template<typename Model>
void TrainingAgent<Model>::RewardWeights(double* weights) const noexcept
{
    std::copy(m_rewardScalings.begin(), m_rewardScalings.end(), weights);
}

template<typename Model>
void TrainingAgent<Model>::LoadFromFile(ppo::serialization::json const& ar)
{
    auto const settings = ppo::serialization::ReadTrainingSettings(ar, detail::RewardNames<Model>());
    if (settings.rewardWeights.size() != m_rewardScalings.size())
        throw std::invalid_argument("Reward.Scaling count does not match the agent's reward contract");
    rolloutSteps_ = settings.rolloutSteps;
    rewardClamp_ = settings.rewardClamp;
    std::copy(settings.rewardWeights.begin(), settings.rewardWeights.end(), m_rewardScalings.begin());
}

} // namespace ppo::model
