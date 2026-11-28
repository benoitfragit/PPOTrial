/**
 * @file training_settings.cpp
 * @brief Single-format typed PPO parameter serialization, C++17.
 */
#include "training_settings.hpp"

#include <stdexcept>
#include <unordered_set>

namespace ppo::serialization {

namespace {
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
void CheckRewardContract(std::vector<std::string> const& names)
{
    if (names.empty())
        throw std::invalid_argument("reward channel names must be provided");

    std::unordered_set<std::string> unique;
    for (auto const& name : names)
        if (name.empty() || !unique.insert(name).second)
            throw std::invalid_argument("reward channel names must be unique and nonempty");
}
}

TrainingSettings ReadTrainingSettings(
    json const& document, std::vector<std::string> const& rewardNames)
{
    CheckRewardContract(rewardNames);
    if (!document.is_object() || !document.contains("formatVersion")
        || !document.at("formatVersion").is_number_integer()
        || document.at("formatVersion").get<int>() != 2)
        throw std::invalid_argument("training settings require formatVersion = 2");

    TrainingSettings settings{};
    settings.rolloutSteps = document.at("training").at("rolloutSteps").get<long long>();

    auto const& rewards = document.at("rewards");
    settings.rewardClamp = rewards.at("clamp").get<float>();
    auto const& weights = rewards.at("weights");
    if (!weights.is_object() || weights.size() != rewardNames.size())
        throw std::invalid_argument("named reward weights do not match the model contract");

    for (auto const& name : rewardNames)
        settings.rewardWeights.push_back(weights.at(name).get<float>());

    if (settings.rolloutSteps <= 0)
        throw std::invalid_argument("the rollout size must be positive");

    return settings;
}
} // namespace ppo::serialization
