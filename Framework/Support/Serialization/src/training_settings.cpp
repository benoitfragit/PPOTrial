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
    auto const& training = document.at("training");
    settings.rolloutSteps = training.at("rolloutSteps").get<long long>();
    settings.checkpointEvery = training.at("checkpointEvery").get<unsigned int>();
    settings.modelVersion = training.at("modelVersion").get<unsigned int>();

    auto const& rewards = document.at("rewards");
    settings.rewardClamp = rewards.at("clamp").get<float>();
    auto const& weights = rewards.at("weights");
    if (!weights.is_object() || weights.size() != rewardNames.size())
        throw std::invalid_argument("named reward weights do not match the model contract");

    for (auto const& name : rewardNames)
        settings.rewardWeights.push_back(weights.at(name).get<float>());

    auto const& ppo = document.at("ppo");
    settings.gamma = ppo.at("gamma").get<float>();
    settings.gaeLambda = ppo.at("gaeLambda").get<float>();
    settings.clipEpsilon = ppo.at("clipEpsilon").get<float>();
    settings.valueCoefficient = ppo.at("valueCoefficient").get<float>();
    settings.entropyCoefficient = ppo.at("entropyCoefficient").get<float>();
    settings.learningRate = ppo.at("learningRate").get<float>();
    settings.epochs = ppo.at("epochs").get<int>();
    settings.batchSize = ppo.at("batchSize").get<int>();
    settings.targetKL = ppo.at("targetKL").get<float>();

    if (settings.rolloutSteps <= 0 || settings.checkpointEvery == 0
        || settings.epochs <= 0 || settings.batchSize <= 0)
        throw std::invalid_argument("rollout, checkpoint, epochs and batch size must be positive");

    return settings;
}

/**
 * @brief Sérialise ou écrit les données associées à WriteTrainingSettings selon les garanties implémentées.
 * @details Peut signaler un état invalide par exception.
 */
void WriteTrainingSettings(
    json& document, TrainingSettings const& settings,
    std::vector<std::string> const& rewardNames)
{
    CheckRewardContract(rewardNames);
    if (settings.rewardWeights.size() != rewardNames.size())
        throw std::invalid_argument("reward channel count changed");

    document["formatVersion"] = 2;
    document["training"] = {
        {"rolloutSteps", settings.rolloutSteps},
        {"checkpointEvery", settings.checkpointEvery},
        {"modelVersion", settings.modelVersion}
    };

    auto& rewards = document["rewards"];
    rewards["clamp"] = settings.rewardClamp;
    auto& named = rewards["weights"];
    named = json::object();
    for (std::size_t i = 0; i < rewardNames.size(); ++i)
        named[rewardNames[i]] = settings.rewardWeights[i];

    document["ppo"] = {
        {"gamma", settings.gamma},
        {"gaeLambda", settings.gaeLambda},
        {"clipEpsilon", settings.clipEpsilon},
        {"valueCoefficient", settings.valueCoefficient},
        {"entropyCoefficient", settings.entropyCoefficient},
        {"learningRate", settings.learningRate},
        {"epochs", settings.epochs},
        {"batchSize", settings.batchSize},
        {"targetKL", settings.targetKL}
    };
}
} // namespace ppo::serialization
