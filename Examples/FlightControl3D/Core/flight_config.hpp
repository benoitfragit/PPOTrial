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
    /** @brief Enables racetrack guidance and fallback; false keeps airborne flight in approach. */
    bool patrolEnabled{true};
    double physicsDt{}, maxSeconds{}, maxDistanceMeters{}, maxAltitudeMeters{};
    std::size_t actionSubsteps{};
    double switchDistanceMeters{}, ilsRangeMeters{}, ilsSectorDegrees{}, glideDegrees{};
    double localizerCaptureDegrees{}, glideCaptureDegrees{};
    double captureHoldSeconds{},captureHysteresis{},pathCorridorMeters{},maximumWaypointStepMeters{};
    double patrolHeadingToleranceDegrees{};
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
    double throttle{}, flap{}, approachFlap{},rolloutFlap{},propeller{},mixture{1.0};
};
struct Normalization final
{
    double distance{}, altitude{}, speed{}, rate{}, path{}, glideDegrees{};
    /** Independent SI scales for lateral error, vertical speed and target flight-path angle. */
    double cross{}, verticalSpeed{}, slopeDegrees{};
};
struct RewardSettings final
{
    std::array<double, RewardCount> weights{};
    double pathTolerance{}, localizerToleranceDegrees{}, glideToleranceDegrees{};
    double speedTolerance{}, slopeToleranceDegrees{};
    /** @brief Fraction of LOC quality supplied by the bounded intercept-course quality. */
    double localizerCourseBlend{};
    /** @brief Gain from signed LOC angle to the desired intercept course. */
    double localizerInterceptGain{};
    /** @brief Absolute bound for the desired LOC intercept course, in degrees. */
    double localizerInterceptLimitDegrees{};
};
struct CaptureStartSettings final
{
    double speedToleranceMps{},rollDegrees{},slopeToleranceDegrees{},localizerHeightOffsetMeters{};
    Range pitchDegrees{};
};
struct TrainingSettings final
{
    std::uint32_t seed{};
    std::size_t rolloutSteps{}, minibatch{}, epochs{}, maxUpdates{}, checkpointEvery{}, hiddenWidth{};
    /** @brief Joint-training multiplier; capture lessons keep the nominal objective. */
    double ilsCaptureScale{1.0};
    CurriculumStage curriculumStage{CurriculumStage::Joint};
    double locReminderProbability{0.30}, glideOffsetMeters{120.0}, gateHoldSeconds{10.0};
    double gateHeadingDegrees{5.0}, gateRouteDegrees{5.0}, gateSlopeDegrees{1.5};
    double gateSpeedToleranceMps{},gateRollDegrees{},gateRateDegreesPerSecond{},captureMaxSeconds{};
    CaptureStartSettings captureStart{};
    double actorRate{}, criticRate{}, gamma{}, lambda{}, clip{}, entropy{}, valueWeight{}, targetKl{};
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
/**
 * @brief Checks whether a stored learner can warm-start the current training contract.
 * @param stored Serialized checkpoint signature from the stored learner.
 * @param current Serialized checkpoint signature for the requested session.
 * @return True for compatible training-only changes; BEST still requires an exact contract.
 *
 * Compatible changes include spawn position ranges, their mixture, entry phase,
 * patrol activation, reward settings, Gaussian exploration bounds and the
 * terminal distance boundary. New rollouts never reuse old likelihoods.
 */
bool CompatibleTrainingSignature(std::string const& stored,std::string const& current);

} // namespace ppo::examples::flight
