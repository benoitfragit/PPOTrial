#pragma once

#include "patrol_core.hpp"
#include "patrol_training_settings.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>

namespace ppo::examples::bluedefense::defenders {

namespace patrol_config_detail {

inline float FiniteNumber(nlohmann::json const& object, char const* key)
{
    auto const item = object.find(key);
    if (item == object.end() || !item->is_number())
        throw std::invalid_argument(
            std::string{"Missing numeric patrol setting: "} + key);
    float const value = item->get<float>();
    if (!std::isfinite(value))
        throw std::invalid_argument(
            std::string{"Nonfinite patrol setting: "} + key);
    return value;
}

inline PatrolRange Range(nlohmann::json const& object, char const* key)
{
    auto const item = object.find(key);
    if (item == object.end() || !item->is_object())
        throw std::invalid_argument(
            std::string{"Missing patrol range: "} + key);
    PatrolRange const result{
        FiniteNumber(*item, "min"), FiniteNumber(*item, "max")};
    if (result.minimum > result.maximum)
        throw std::invalid_argument(
            std::string{"Invalid patrol range (min > max): "} + key);
    return result;
}

} // namespace patrol_config_detail

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

/**
 * @brief Parses and validates episode-level racetrack and wedge ranges.
 * @param document Full BluePatrol JSON document.
 * @return Validated ranges expressed in metres and radians.
 */
inline PatrolGeometrySettings
ParsePatrolGeometrySettings(nlohmann::json const& document)
{
    constexpr float degreesToRadians = kPatrolPi / 180.F;
    auto const patrolItem = document.find("patrol");
    if (patrolItem == document.end() || !patrolItem->is_object())
        throw std::invalid_argument("Missing BluePatrol patrol settings");
    auto const racetrackItem = patrolItem->find("racetrack");
    auto const formationItem = patrolItem->find("formation");
    if (racetrackItem == patrolItem->end() || !racetrackItem->is_object()
        || formationItem == patrolItem->end() || !formationItem->is_object())
        throw std::invalid_argument(
            "Missing BluePatrol racetrack or formation settings");

    PatrolGeometrySettings result{};
    result.lengthMeters = patrol_config_detail::Range(
        *racetrackItem, "lengthMeters");
    result.widthMeters = patrol_config_detail::Range(
        *racetrackItem, "widthMeters");
    result.headingRadians = patrol_config_detail::Range(
        *racetrackItem, "headingDegrees");
    result.headingRadians.minimum *= degreesToRadians;
    result.headingRadians.maximum *= degreesToRadians;
    result.anchorNorthMeters = patrol_config_detail::Range(
        *racetrackItem, "anchorNorthMeters");
    result.anchorEastMeters = patrol_config_detail::Range(
        *racetrackItem, "anchorEastMeters");
    result.boundaryMarginMeters = patrol_config_detail::FiniteNumber(
        *racetrackItem, "boundaryMarginMeters");
    auto const attempts = racetrackItem->find("maximumSamplingAttempts");
    auto const clockwise = racetrackItem->find("clockwise");
    if (attempts == racetrackItem->end() || !attempts->is_number_unsigned()
        || clockwise == racetrackItem->end() || !clockwise->is_boolean())
        throw std::invalid_argument(
            "Invalid racetrack sampling attempts or direction");
    result.maximumSamplingAttempts = attempts->get<std::size_t>();
    result.clockwise = clockwise->get<bool>();

    result.separationDistanceMeters = patrol_config_detail::Range(
        *formationItem, "separationDistanceMeters");
    result.separationAngleRadians = patrol_config_detail::Range(
        *formationItem, "separationAngleDegrees");
    result.separationAngleRadians.minimum *= degreesToRadians;
    result.separationAngleRadians.maximum *= degreesToRadians;
    if (!Valid(result))
        throw std::invalid_argument("Invalid BluePatrol patrol geometry ranges");
    return result;
}

/**
 * @brief Loads and validates the episode-level racetrack and wedge ranges.
 * @param settings Root resources/settings directory.
 * @return Validated ranges expressed in metres and radians.
 */
inline PatrolGeometrySettings
LoadPatrolGeometrySettings(std::filesystem::path const& settings)
{
    return ParsePatrolGeometrySettings(
        ReadPatrolJson(settings / "Training/BluePatrol.json"));
}

inline Parameters LoadPatrolParameters(std::filesystem::path const& settings)
{
    auto const world = ReadPatrolJson(settings / "Environment.json").at("environment");
    auto const flight = ReadPatrolJson(settings / "Platforms/Mobile/defender_car.json")
        .at("parameters").at("FlightModel");
    auto const patrolDocument = ReadPatrolJson(
        settings / "Training/BluePatrol.json");
    Parameters parameters{};
    parameters.halfNorth = 0.5F * world.at("maxNorth").get<float>();
    parameters.halfEast = 0.5F * world.at("maxEast").get<float>();
    parameters.dt = world.at("timeStep").get<float>();
    parameters.cruiseSpeed = flight.at("CruiseSpeed").get<float>();
    parameters.maxSpeed = flight.at("MaxSpeed").get<float>();
    parameters.maxAcceleration = flight.at("MaxAcceleration").get<float>();
    parameters.maxRotationRate = flight.at("MaxRotAcceleration").get<float>();
    auto const& shaping = patrolDocument.at("rewards").at("shaping");
    parameters.patrolProgressWeight = patrol_config_detail::FiniteNumber(
        shaping, "patrolProgressWeight");
    parameters.patrolCrossTrackWeight = patrol_config_detail::FiniteNumber(
        shaping, "patrolCrossTrackWeight");
    parameters.patrolCrossTrackScale = patrol_config_detail::FiniteNumber(
        shaping, "patrolCrossTrackScaleMeters");
    parameters.formationErrorWeight = patrol_config_detail::FiniteNumber(
        shaping, "formationErrorWeight");
    parameters.formationTolerance = patrol_config_detail::FiniteNumber(
        shaping, "formationToleranceFraction");
    parameters.controlEffortWeight = patrol_config_detail::FiniteNumber(
        shaping, "controlEffortWeight");
    auto const& defense = patrolDocument.at("patrol").at("defense");
    parameters.defense.activationRisk = patrol_config_detail::FiniteNumber(
        defense, "activationRisk");
    parameters.defense.deactivationRisk = patrol_config_detail::FiniteNumber(
        defense, "deactivationRisk");
    parameters.defense.activationDistanceMeters =
        patrol_config_detail::FiniteNumber(
            defense, "activationDistanceMeters");
    parameters.defense.deactivationDistanceMeters =
        patrol_config_detail::FiniteNumber(
            defense, "deactivationDistanceMeters");
    parameters.defense.protectedRadiusMeters =
        patrol_config_detail::FiniteNumber(defense, "protectedRadiusMeters");
    parameters.defense.timeToIntrusionSeconds =
        patrol_config_detail::FiniteNumber(defense, "timeToIntrusionSeconds");
    parameters.defense.teammateThreatDistanceMeters =
        patrol_config_detail::FiniteNumber(
            defense, "teammateThreatDistanceMeters");
    parameters.defense.teammateVulnerabilityWeight =
        patrol_config_detail::FiniteNumber(
            defense, "teammateVulnerabilityWeight");

    PatrolGeometrySettings const geometrySettings =
        LoadPatrolGeometrySettings(settings);
    PatrolSamplingConstraints const constraints{
        parameters.halfNorth, parameters.halfEast, parameters.cruiseSpeed,
        parameters.maxRotationRate, parameters.collisionDistance};
    std::mt19937 validationRng{0U};
    auto const geometry = SamplePatrolGeometry(
        geometrySettings, constraints, validationRng);
    if (!geometry)
        throw std::invalid_argument(
            "BluePatrol geometry has no feasible sample within the attempt limit");
    parameters.geometry = *geometry;
    if (!Valid(parameters))
        throw std::invalid_argument("Invalid patrol world or vehicle parameters");
    return parameters;
}

/**
 * @brief Parses the configurable embedded MAPPO rollout and reward settings.
 * @param document Full BluePatrol JSON document.
 * @return Validated blue training settings.
 */
inline PatrolTrainingSettings
ParsePatrolTrainingSettings(nlohmann::json const& document)
{
    auto const& training = document.at("training");
    auto const& rewards = document.at("rewards");
    auto const& weights = rewards.at("weights");
    auto const& evaluation = document.at("evaluation");

    PatrolTrainingSettings result{};
    result.rolloutSteps = training.at("rolloutSteps").get<std::size_t>();
    result.maxUpdates = training.at("maxUpdates").get<std::uint64_t>();
    result.checkpointEvery = training.at("checkpointEvery").get<std::uint64_t>();
    // Old configurations retain their original save/evaluation cadence.
    result.saveEveryUpdates = training.value(
        "saveEveryUpdates", result.checkpointEvery);
    result.seed = training.at("seed").get<std::uint32_t>();
    result.evaluation.episodes = evaluation.at("episodes").get<std::size_t>();
    result.evaluation.maxSteps = evaluation.at("maxSteps").get<std::size_t>();
    result.evaluation.seed = evaluation.at("seed").get<std::uint32_t>();
    result.evaluation.minimumTrackingFraction =
        evaluation.at("minimumTrackingFraction").get<float>();
    result.rewardClamp = rewards.at("clamp").get<float>();
    for (std::size_t i = 0; i < kRewardChannelNames.size(); ++i)
        result.rewardWeights[i] = weights.at(kRewardChannelNames[i]).get<float>();

    // Optional to preserve existing blue configuration files.
    auto const generalized = rewards.find("generalized");
    if (generalized != rewards.end())
    {
        if (!generalized->is_object())
            throw std::invalid_argument("Invalid generalized blue reward settings");
        result.ownLossWeight = patrol_config_detail::FiniteNumber(
            *generalized, "ownLossWeight");
        result.lowSpeedRiskWeight = patrol_config_detail::FiniteNumber(
            *generalized, "lowSpeedRiskWeight");
    }

    if (!Valid(result) || !weights.is_object()
        || weights.size() != kRewardChannelNames.size())
        throw std::invalid_argument("Invalid BluePatrol MAPPO training settings");
    return result;
}

/** @brief Loads validated blue MAPPO settings from the resource directory. */
inline PatrolTrainingSettings
LoadPatrolTrainingSettings(std::filesystem::path const& settings)
{
    return ParsePatrolTrainingSettings(
        ReadPatrolJson(settings / "Training/BluePatrol.json"));
}

} // namespace ppo::examples::bluedefense::defenders
