/**
 * @file training_settings.hpp
 * @brief Application-independent training settings parsed from the version 2 JSON.
 *
 * Reward channel identities are supplied by the model in their actual order.
 */
#pragma once
#include "serializer.hpp"
#include <vector>
#include <string>

namespace ppo::serialization {

/** @brief The training settings of an Airspace agent (settings/Training/<agent>.json). */
struct TrainingSettings
{
    /** @brief Steps of one rollout: the episode's step limit under ppokit-run. */
    long long rolloutSteps{ 0 };
    /** @brief Bound of the step's reward, |reward| <= rewardClamp. */
    float rewardClamp{ 0.F };
    /** @brief Weights of the reward channels, in the model's order. */
    std::vector<float> rewardWeights{};
};

/** Parse the sole supported training format (formatVersion = 2).
 *  @throws std::exception on missing, invalid or mismatched values.
 */
TrainingSettings ReadTrainingSettings(
    json const& document, std::vector<std::string> const& rewardNames);

} // namespace ppo::serialization
