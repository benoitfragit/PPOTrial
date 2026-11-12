#pragma once

#include "flight_types.hpp"
#include <filesystem>
#include <string>

namespace ppo::examples::flight {

struct Runway final
{
    double latitudeDegrees{}, longitudeDegrees{}, elevationMeters{}, headingDegrees{};
    double lengthMeters{}, widthMeters{}, thresholdHeightMeters{};
    Vec3 origin{}, velocity{}, angularVelocity{};
    double pitch{}, roll{};
};
struct Range final { double minimum{}, maximum{}; };
struct RunwayRandomization final
{
    Range northMeters{}, eastMeters{}, headingOffsetDegrees{}, deckHeightMeters{}, speedMps{};
    Range heaveAmplitudeMeters{}, wavePeriodSeconds{}, pitchAmplitudeDegrees{}, rollAmplitudeDegrees{}, phaseDegrees{};
    double seaHeightMeters{};
};
struct MissionSettings final
{
    double physicsDt{}, maxSeconds{}, maxDistanceMeters{}, maxAltitudeMeters{};
    std::size_t actionSubsteps{};
    double switchDistanceMeters{}, ilsRangeMeters{}, ilsSectorDegrees{}, glideDegrees{};
    double localizerCaptureDegrees{}, glideCaptureDegrees{};
    double patrolSpeed{}, approachSpeed{}, flareSpeed{}, flareHeightMeters{};
    double maxTouchdownSink{}, maxTouchdownSpeed{}, maxTouchdownRollDegrees{};
    double maxTouchdownPitchDegrees{}, minTouchdownPitchDegrees{}, maxTouchdownHeadingDegrees{};
    double gearDownMinimum{}, contactHeightMeters{}, stoppedSpeed{}, stoppedHoldSeconds{};
    double racetrackCenterAlong{}, racetrackCenterCross{}, racetrackHalfStraight{};
    double racetrackRadius{}, patrolAltitude{}, waypointRadius{};
    std::size_t turnSamples{}, straightSamples{};
};
struct InitialSettings final
{
    bool approach{}, gearDown{};
    Range distanceMeters{}, bearingDegrees{}, heightMeters{}, speedMps{}, headingOffsetDegrees{};
    Range rollDegrees{}, pitchDegrees{}, slopeDegrees{};
    double throttle{}, flap{}, propeller{};
};
struct Normalization final
{
    double distance{}, altitude{}, speed{}, rate{}, path{}, localizerDegrees{}, glideDegrees{};
};
struct RewardSettings final
{
    std::array<double, RewardCount> weights{};
    double pathTolerance{}, localizerToleranceDegrees{}, glideToleranceDegrees{};
    double speedTolerance{}, slopeToleranceDegrees{};
};
struct TrainingSettings final
{
    std::uint32_t seed{};
    std::size_t rolloutSteps{}, minibatch{}, epochs{}, maxUpdates{}, checkpointEvery{}, hiddenWidth{};
    double actorRate{}, criticRate{}, gamma{}, lambda{}, clip{}, entropy{}, valueWeight{};
    double minLogStd{}, maxLogStd{}, initialLogStd{}, maxGradientNorm{};
    double adamBeta1{}, adamBeta2{}, adamEpsilon{};
    bool resume{};
    std::string checkpointDirectory;
};
struct EvaluationSettings final { std::uint32_t seed{}; std::size_t episodes{}; };
struct Config final
{
    Runway runway{};
    RunwayRandomization runwayRandomization{};
    MissionSettings mission{};
    InitialSettings initial{};
    Normalization normalization{};
    RewardSettings reward{};
    TrainingSettings training{};
    EvaluationSettings evaluation{};
    std::string aircraft;
    std::string checkpointSignature;
    double StepSeconds() const noexcept { return mission.physicsDt * static_cast<double>(mission.actionSubsteps); }
};

/** Strict versioned JSON. Missing, unknown and non-finite fields are errors. */
Config LoadConfig(std::filesystem::path const& path);
void Validate(Config const& config);

} // namespace ppo::examples::flight
