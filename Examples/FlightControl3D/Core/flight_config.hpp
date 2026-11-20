#pragma once

#include "flight_types.hpp"
#include <filesystem>
#include <string>

namespace ppo::examples::flight {

/** @brief Static runway pose and dimensions; the origin is the threshold centre on the deck surface. */
struct Runway final
{
    double latitudeDegrees{}, longitudeDegrees{}, elevationMeters{}, headingDegrees{};
    double lengthMeters{}, widthMeters{}, thresholdHeightMeters{};
    /** @brief Extra landable surface around each runway edge, in metres. */
    double landingMarginMeters{};
    Vec3 origin{};
};
struct Range final { double minimum{}, maximum{}; };
/** @brief Per-episode runway pose randomization. */
struct RunwayRandomization final
{
    Range northMeters{}, eastMeters{}, headingOffsetDegrees{}, deckHeightMeters{};
    double seaHeightMeters{};
};
struct MissionSettings final
{
    double physicsDt{};
    /** @brief First-order continuous control filter time constant, in seconds. */
    double controlFilterTimeConstantSeconds{};
    /** @brief Fraction of the requested rudder command available while airborne. */
    double airborneRudderAuthority{1.0};
    double maxSeconds{}, maxDistanceMeters{}, maxAltitudeMeters{};
    std::size_t actionSubsteps{};
    double glideDegrees{};
    double approachSpeed{}, flareSpeed{}, flareHeightMeters{};
    double maxTouchdownSink{}, maxTouchdownSpeed{}, maxTouchdownRollDegrees{};
    double maxTouchdownPitchDegrees{}, minTouchdownPitchDegrees{}, maxTouchdownHeadingDegrees{};
    double gearDownMinimum{}, contactHeightMeters{}, stoppedSpeed{}, stoppedHoldSeconds{};
};
/** @brief Already-landed start used to practise braking on the runway. */
struct RolloutStartSettings final
{
    Range speedMps{}, alongMeters{}, crossMeters{};
};
/** @brief Airborne start envelope around the 3-degree approach plane, relative to the runway. */
struct InitialSettings final
{
    Range distanceMeters{}, bearingDegrees{}, headingOffsetDegrees{};
    /** @brief Uniform height offset from the approach plane, in metres. */
    double heightOffsetMeters{};
    double speedToleranceMps{}, rollDegrees{}, slopeToleranceDegrees{};
    Range pitchDegrees{};
    RolloutStartSettings rolloutStart{};
    /**
     * @brief Trimmed 3-degree approach controls: the initial airborne commands
     * and the controls a zero actor output requests.
     */
    double throttle{}, elevatorTrim{};
    double approachFlap{}, rolloutFlap{}, propeller{}, mixture{1.0};
};
struct RewardSettings final
{
    std::array<double, RewardCount> weights{};
    double pathTolerance{};
};
/**
 * @brief Adaptive start-difficulty curriculum used only by training episodes.
 *
 * Difficulty 0 is a short, aligned final approach; difficulty 1 is the full
 * configured start envelope. Validations always use the full envelope.
 */
struct CurriculumSettings final
{
    /** @brief Start distance range at difficulty 0, in metres. */
    Range firstDistanceMeters{};
    /** @brief Number of difficulty levels, the last one being the full task. */
    std::size_t levels{1};
    /** @brief Stop rate over the window that promotes to the next level. */
    double promotionSuccessRate{};
    /** @brief Airborne training episodes per promotion decision. */
    std::size_t windowEpisodes{1};
    /** @brief Share of training episodes that start already landed, to practise braking. */
    double rolloutStartShare{};
};
struct TrainingSettings final
{
    std::uint32_t seed{};
    /** @brief Draw a fresh environment seed for each training session when enabled. */
    bool randomizeEnvironmentSeed{};
    std::size_t rolloutSteps{}, minibatch{}, epochs{}, maxUpdates{}, checkpointEvery{}, hiddenWidth{};
    double actorRate{}, criticRate{}, gamma{}, lambda{}, clip{}, entropy{}, valueWeight{};
    /** @brief Target for the sampled old-to-new minibatch KL; skip above 1.5 times this value. */
    double targetKl{0.02};
    double adamBeta1{}, adamBeta2{}, adamEpsilon{};
    bool resume{};
    std::string checkpointDirectory;
    CurriculumSettings curriculum{};
};
/** @brief Optional bounded telemetry used to diagnose failed convergence. */
struct DiagnosticsSettings final
{
    bool enabled{};
    std::size_t logEveryUpdates{1};
    double saturationThreshold{0.98};
};
/** @brief Desktop execution controls; headless execution is always unthrottled. */
struct RuntimeSettings final
{
    bool maxSpeed{};
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
    RewardSettings reward{};
    TrainingSettings training{};
    EvaluationSettings evaluation{};
    DiagnosticsSettings diagnostics{};
    RuntimeSettings runtime{};
    /** @brief JSBSim aircraft model name. */
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
 * @return True when only reward weights or start ranges differ; BEST still requires an exact contract.
 */
bool CompatibleTrainingSignature(std::string const& stored, std::string const& current);

} // namespace ppo::examples::flight
