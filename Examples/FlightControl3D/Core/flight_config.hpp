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
    double captureHoldSeconds{},captureHysteresis{},pathCorridorMeters{},maximumWaypointStepMeters{};
    double approachHeadingToleranceDegrees{},approachCrossToleranceMeters{},approachAltitudeToleranceMeters{};
    std::size_t requiredLaps{};
    double patrolSpeed{}, approachSpeed{}, flareSpeed{}, flareHeightMeters{};
    double maxTouchdownSink{}, maxTouchdownSpeed{}, maxTouchdownRollDegrees{};
    double maxTouchdownPitchDegrees{}, minTouchdownPitchDegrees{}, maxTouchdownHeadingDegrees{};
    double gearDownMinimum{}, contactHeightMeters{}, stoppedSpeed{}, stoppedHoldSeconds{};
    double racetrackCenterAlong{}, racetrackCenterCross{}, racetrackHalfStraight{};
    double racetrackRadius{}, patrolAltitude{}, waypointRadius{};
    /** @brief Time horizon used to place the continuous rally target on the racetrack. */
    double patrolLookaheadSeconds{};
    std::size_t turnSamples{}, straightSamples{};
};
struct InitialSettings final
{
    bool approach{}, gearDown{};
    /** Long-run LOC/glide share; the remainder is balanced between left and right. */
    double runwayAlignedProbability{};
    Range distanceMeters{}, bearingDegrees{}, heightMeters{}, speedMps{}, headingOffsetDegrees{};
    Range rollDegrees{}, pitchDegrees{}, slopeDegrees{};
    double throttle{}, flap{}, approachFlap{},rolloutFlap{},propeller{};
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
    double convergenceSeconds{}, ilsDecaySeconds{};
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
struct EvaluationSettings final
{
    std::uint32_t seed{};
    std::size_t episodes{}, everyUpdates{};
    double minimumSuccessRate{};
};
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
/** Warm-start training may change horizontal starts, their mixture, entry phase and rewards; BEST requires an exact contract. */
bool CompatibleTrainingSignature(std::string const& stored,std::string const& current);

} // namespace ppo::examples::flight
