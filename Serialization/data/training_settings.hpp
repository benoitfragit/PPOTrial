/**
 * @file training_settings.hpp
 * @brief Application-independent PPO settings parsed from the version 2 JSON.
 *
 * Reward channel identities are supplied by the model in their actual order.
 */
#pragma once
#include "serializer.hpp"
#include <vector>
#include <string>

namespace ppo::serialization {

struct TrainingSettings
{
    long long rolloutSteps{ 0 };
    unsigned int checkpointEvery{ 0 };
    unsigned int modelVersion{ 0 };
    float rewardClamp{ 0.F };
    std::vector<float> rewardWeights{};

    float gamma{ 0.F };
    float gaeLambda{ 0.F };
    float clipEpsilon{ 0.F };
    float valueCoefficient{ 0.F };
    float entropyCoefficient{ 0.F };
    float learningRate{ 0.F };
    int epochs{ 0 };
    int batchSize{ 0 };
    float targetKL{ 0.F };
};

/** Parse the sole supported training format (formatVersion = 2).
 *  @throws std::exception on missing, invalid or mismatched values.
 */
TrainingSettings ReadTrainingSettings(
    json const& document, std::vector<std::string> const& rewardNames);

/** Persist exactly the version 2 format. */
void WriteTrainingSettings(
    json& document, TrainingSettings const& settings,
    std::vector<std::string> const& rewardNames);

} // namespace ppo::serialization
