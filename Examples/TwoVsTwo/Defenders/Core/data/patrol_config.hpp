#pragma once

#include "patrol_core.hpp"
#include "patrol_training_settings.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace ppo::examples::twovstwo::defenders {

inline nlohmann::json ReadPatrolJson(std::filesystem::path const& path)
{
    std::ifstream stream{path};
    if (!stream)
        throw std::runtime_error("Cannot open patrol configuration: " + path.string());
    nlohmann::json document;
    stream >> document;
    if (document.at("formatVersion").get<int>() != 2)
        throw std::runtime_error("Unsupported patrol configuration version");
    return document;
}

inline Parameters LoadPatrolParameters(std::filesystem::path const& settings)
{
    auto const world = ReadPatrolJson(settings / "Environment.json").at("environment");
    auto const flight = ReadPatrolJson(settings / "Platforms/Mobile/defender_car.json")
        .at("parameters").at("FlightModel");
    Parameters parameters{};
    parameters.halfNorth = 0.5F * world.at("maxNorth").get<float>();
    parameters.halfEast = 0.5F * world.at("maxEast").get<float>();
    parameters.dt = world.at("timeStep").get<float>();
    parameters.cruiseSpeed = flight.at("CruiseSpeed").get<float>();
    parameters.maxSpeed = flight.at("MaxSpeed").get<float>();
    parameters.maxAcceleration = flight.at("MaxAcceleration").get<float>();
    parameters.maxRotationRate = flight.at("MaxRotAcceleration").get<float>();
    if (!Valid(parameters))
        throw std::invalid_argument("Invalid patrol world or vehicle parameters");
    return parameters;
}

/**
 * @brief Loads the configurable embedded MAPPO rollout and reward settings.
 * @param settings Root resources/settings directory.
 * @return Validated blue training settings.
 */
inline PatrolTrainingSettings
LoadPatrolTrainingSettings(std::filesystem::path const& settings)
{
    auto const document = ReadPatrolJson(settings / "Training/BluePatrol.json");
    auto const& training = document.at("training");
    auto const& rewards = document.at("rewards");
    auto const& weights = rewards.at("weights");
    auto const& evaluation = document.at("evaluation");

    PatrolTrainingSettings result{};
    result.rolloutSteps = training.at("rolloutSteps").get<std::size_t>();
    result.maxUpdates = training.at("maxUpdates").get<std::uint64_t>();
    result.checkpointEvery = training.at("checkpointEvery").get<std::uint64_t>();
    result.seed = training.at("seed").get<std::uint32_t>();
    result.evaluation.episodes = evaluation.at("episodes").get<std::size_t>();
    result.evaluation.maxSteps = evaluation.at("maxSteps").get<std::size_t>();
    result.evaluation.seed = evaluation.at("seed").get<std::uint32_t>();
    result.evaluation.minimumTrackingFraction =
        evaluation.at("minimumTrackingFraction").get<float>();
    result.rewardClamp = rewards.at("clamp").get<float>();
    for (std::size_t i = 0; i < kRewardChannelNames.size(); ++i)
        result.rewardWeights[i] = weights.at(kRewardChannelNames[i]).get<float>();

    if (!Valid(result) || !weights.is_object()
        || weights.size() != kRewardChannelNames.size())
        throw std::invalid_argument("Invalid BluePatrol MAPPO training settings");
    return result;
}

} // namespace ppo::examples::twovstwo::defenders
