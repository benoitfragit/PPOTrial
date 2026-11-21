/**
 * @file flight_config.cpp
 * @brief Reads the scenario JSON section by section, then validates the cross-field rules.
 *
 * Role in the chain: first stage of every run; produces the immutable Config and its
 * checkpoint signature (the canonical JSON of every task-defining section plus
 * contractVersion). The signature is what checkpoints and BEST manifests compare.
 *
 * Not here: no default values are invented. A missing, unknown, mistyped or non-finite
 * field is an error, so a scenario can never silently change meaning.
 *
 * Read next: flight_config.hpp (field meanings and units), flight_reward_tuning.cpp
 * (runtime edits of reward weights), Training/flight_model_store.cpp (signature checks).
 */
#include "flight_config.hpp"
#include "flight_aircraft_performance.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ppo::examples::flight {
namespace {
using Json = nlohmann::json;

/** @brief Version of the observation, action, guidance, reward and spawn semantics stored in checkpoints. */
constexpr int ContractVersion = 7;
/** @brief Accepted value of the scenario "formatVersion" field. */
constexpr int ConfigurationFormatVersion = 2;
/**
 * @brief Spawn-distribution fields of "initial" that a warm start may change.
 *
 * Every other initial field (trimmed controls, flaps, propeller, mixture and any
 * field added later) sets the aircraft dynamics or the meaning of an actor output,
 * so a checkpoint is only compatible when it matches exactly.
 */
constexpr std::array<char const*, 9> WarmStartSpawnFields{"distanceMeters", "bearingDegrees",
    "headingOffsetDegrees", "heightOffsetMeters", "speedToleranceMps", "rollDegrees", "pitchDegrees",
    "slopeToleranceDegrees", "rolloutStart"};

// Upper bounds of the integer fields. They only reject absurd values; the
// physical and numerical rules live in the Validate* functions below.
constexpr std::size_t MaximumActionSubsteps = 100;
constexpr std::size_t MaximumCurriculumLevels = 100;
constexpr std::size_t MaximumCurriculumStages = 16;
constexpr std::size_t MaximumCurriculumWindowEpisodes = 100000;
constexpr std::size_t MaximumRolloutSteps = 65536;
constexpr std::size_t MaximumMinibatch = 65536;
constexpr std::size_t MaximumEpochs = 100;
constexpr std::size_t MaximumUpdateCount = 100000000;
constexpr std::size_t MaximumHiddenWidth = 256;
constexpr std::size_t MaximumEvaluationEpisodes = 1000000;

// Validation limits, with their origin.
/** @brief Latitude limit of the local tangent frame, in degrees (avoids the poles). */
constexpr double MaximumAbsoluteLatitudeDegrees = 85.0;
constexpr double MaximumAbsoluteLongitudeDegrees = 180.0;
constexpr double FullTurnDegrees = 360.0;
constexpr double HalfTurnDegrees = 180.0;
constexpr double QuarterTurnDegrees = 90.0;
/** @brief Largest accepted landable margin around the runway, in metres. */
constexpr double MaximumLandingMarginMeters = 10000.0;
/** @brief Accepted JSBSim step range, in seconds (1 kHz to 50 Hz). */
constexpr double MinimumPhysicsDtSeconds = 0.001;
constexpr double MaximumPhysicsDtSeconds = 0.02;
/** @brief Longest policy action, in seconds. */
constexpr double MaximumActionSeconds = 0.5;
constexpr double MaximumFilterTimeConstantSeconds = 10.0;
/** @brief Longest episode, one simulated day, in seconds. */
constexpr double MaximumEpisodeSeconds = 86400.0;
/** @brief Steepest accepted glide slope (and slope plus tolerance), in degrees. */
constexpr double MaximumGlideDegrees = 10.0;
/** @brief Largest accepted initial bank amplitude, in degrees. */
constexpr double MaximumInitialRollDegrees = 30.0;
/** @brief Largest accepted absolute initial pitch, in degrees. */
constexpr double MaximumAbsoluteInitialPitchDegrees = 15.0;
/** @brief Rolling starts stay within width / RolloutCrossWidthDivisor of the centreline (a quarter width). */
constexpr double RolloutCrossWidthDivisor = 4.0;
/** @brief Highest stage start or gate above the deck, in metres (below the observed height range of 2 km). */
constexpr double MaximumStageHeightMeters = 1500.0;
/** @brief Largest stage lateral offset, in metres (below the observed cross range of 5 km). */
constexpr double MaximumStageCrossMeters = 4000.0;
/** @brief Smallest distance between route gates, in metres (more than three gate radii). */
constexpr double MinimumGateSpacingMeters = 1000.0;
/** @brief Shortest first curriculum start distance, in metres. */
constexpr double MinimumFirstCurriculumDistanceMeters = 1000.0;
/** @brief Accepted saturation threshold range of the diagnostics. */
constexpr double MinimumSaturationThreshold = 0.5;
constexpr double MaximumSaturationThreshold = 1.0;

/** @brief Throws std::invalid_argument with a "FlightControl3D: " prefix when the condition fails. */
void Require(bool condition, char const* message)
{
    if (!condition)
    {
        throw std::invalid_argument(std::string{"FlightControl3D: "} + message);
    }
}

/** @brief Requires an object with exactly the listed keys. */
void Keys(Json const& object, std::initializer_list<char const*> names)
{
    Require(object.is_object() && object.size() == names.size(), "unexpected or missing JSON fields");
    for (auto const* name : names)
    {
        Require(object.contains(name), name);
    }
}

/** @brief Reads a finite JSON number (booleans are rejected). */
double Number(Json const& object, char const* key)
{
    auto const& value = object.at(key);
    Require(value.is_number() && !value.is_boolean(), key);
    double const result = value.get<double>();
    Require(std::isfinite(result), key);
    return result;
}

/** @brief Reads a JSON integer in [0, limit]. */
std::size_t Integer(Json const& object, char const* key, std::size_t limit)
{
    auto const& value = object.at(key);
    Require(value.is_number_unsigned() || value.is_number_integer(), key);
    auto const result = value.get<std::int64_t>();
    Require(result >= 0 && static_cast<std::uint64_t>(result) <= limit, key);
    return static_cast<std::size_t>(result);
}

/** @brief Reads a JSON boolean. */
bool Boolean(Json const& object, char const* key)
{
    Require(object.at(key).is_boolean(), key);
    return object.at(key).get<bool>();
}

/** @brief Reads a finite two-element [minimum, maximum] array with minimum <= maximum. */
Range Bounds(Json const& object, char const* key)
{
    auto const& value = object.at(key);
    Require(value.is_array() && value.size() == 2 && value[0].is_number() && value[1].is_number(), key);
    Range const range{value[0].get<double>(), value[1].get<double>()};
    Require(std::isfinite(range.minimum) && std::isfinite(range.maximum) && range.minimum <= range.maximum,
        key);
    return range;
}

// ---------------------------------------------------------------------------
// Reading, one function per JSON section. Fields are read in a fixed order so
// that the first reported error is stable.
// ---------------------------------------------------------------------------

Runway ReadRunway(Json const& section)
{
    Keys(section, {"latitudeDegrees", "longitudeDegrees", "elevationMeters", "headingDegrees", "lengthMeters",
        "widthMeters", "thresholdHeightMeters", "landingMarginMeters"});
    Runway runway{};
    runway.latitudeDegrees = Number(section, "latitudeDegrees");
    runway.longitudeDegrees = Number(section, "longitudeDegrees");
    runway.elevationMeters = Number(section, "elevationMeters");
    runway.headingDegrees = Number(section, "headingDegrees");
    runway.lengthMeters = Number(section, "lengthMeters");
    runway.widthMeters = Number(section, "widthMeters");
    runway.thresholdHeightMeters = Number(section, "thresholdHeightMeters");
    runway.landingMarginMeters = Number(section, "landingMarginMeters");
    return runway;
}

RunwayRandomization ReadRunwayRandomization(Json const& section)
{
    Keys(section, {"northMeters", "eastMeters", "headingOffsetDegrees", "deckHeightMeters",
        "seaHeightMeters"});
    return {Bounds(section, "northMeters"), Bounds(section, "eastMeters"), Bounds(section,
        "headingOffsetDegrees"),
        Bounds(section, "deckHeightMeters"), Number(section, "seaHeightMeters")};
}

MissionSettings ReadMission(Json const& section)
{
    Keys(section, {"physicsDt", "controlFilterTimeConstantSeconds", "airborneRudderAuthority",
        "actionSubsteps",
        "maxSeconds", "maxDistanceMeters", "maxAltitudeMeters", "glideDegrees", "approachSpeed", "flareSpeed",
        "flareHeightMeters", "maxTouchdownSink", "maxTouchdownSpeed", "maxTouchdownRollDegrees",
        "maxTouchdownPitchDegrees", "minTouchdownPitchDegrees", "maxTouchdownHeadingDegrees",
        "gearDownMinimum", "contactHeightMeters", "stoppedSpeed", "stoppedHoldSeconds"});
    MissionSettings mission{};
    mission.physicsDt = Number(section, "physicsDt");
    mission.controlFilterTimeConstantSeconds = Number(section, "controlFilterTimeConstantSeconds");
    mission.airborneRudderAuthority = Number(section, "airborneRudderAuthority");
    mission.maxSeconds = Number(section, "maxSeconds");
    mission.maxDistanceMeters = Number(section, "maxDistanceMeters");
    mission.maxAltitudeMeters = Number(section, "maxAltitudeMeters");
    mission.glideDegrees = Number(section, "glideDegrees");
    mission.approachSpeed = Number(section, "approachSpeed");
    mission.flareSpeed = Number(section, "flareSpeed");
    mission.flareHeightMeters = Number(section, "flareHeightMeters");
    mission.maxTouchdownSink = Number(section, "maxTouchdownSink");
    mission.maxTouchdownSpeed = Number(section, "maxTouchdownSpeed");
    mission.maxTouchdownRollDegrees = Number(section, "maxTouchdownRollDegrees");
    mission.maxTouchdownPitchDegrees = Number(section, "maxTouchdownPitchDegrees");
    mission.minTouchdownPitchDegrees = Number(section, "minTouchdownPitchDegrees");
    mission.maxTouchdownHeadingDegrees = Number(section, "maxTouchdownHeadingDegrees");
    mission.gearDownMinimum = Number(section, "gearDownMinimum");
    mission.contactHeightMeters = Number(section, "contactHeightMeters");
    mission.stoppedSpeed = Number(section, "stoppedSpeed");
    mission.stoppedHoldSeconds = Number(section, "stoppedHoldSeconds");
    mission.actionSubsteps = Integer(section, "actionSubsteps", MaximumActionSubsteps);
    return mission;
}

InitialSettings ReadInitial(Json const& section)
{
    Keys(section, {"distanceMeters", "bearingDegrees", "headingOffsetDegrees", "heightOffsetMeters",
        "speedToleranceMps", "rollDegrees", "pitchDegrees", "slopeToleranceDegrees", "rolloutStart",
        "throttle", "elevatorTrim", "approachFlap", "rolloutFlap", "propeller", "mixture"});
    InitialSettings initial{};
    initial.distanceMeters = Bounds(section, "distanceMeters");
    initial.bearingDegrees = Bounds(section, "bearingDegrees");
    initial.headingOffsetDegrees = Bounds(section, "headingOffsetDegrees");
    initial.heightOffsetMeters = Number(section, "heightOffsetMeters");
    initial.speedToleranceMps = Number(section, "speedToleranceMps");
    initial.rollDegrees = Number(section, "rollDegrees");
    initial.pitchDegrees = Bounds(section, "pitchDegrees");
    initial.slopeToleranceDegrees = Number(section, "slopeToleranceDegrees");
    auto const& rollout = section.at("rolloutStart");
    Keys(rollout, {"speedMps", "alongMeters", "crossMeters"});
    initial.rolloutStart = {Bounds(rollout, "speedMps"), Bounds(rollout, "alongMeters"),
        Bounds(rollout, "crossMeters")};
    initial.throttle = Number(section, "throttle");
    initial.elevatorTrim = Number(section, "elevatorTrim");
    initial.approachFlap = Number(section, "approachFlap");
    initial.rolloutFlap = Number(section, "rolloutFlap");
    initial.propeller = Number(section, "propeller");
    initial.mixture = Number(section, "mixture");
    return initial;
}

RewardSettings ReadReward(Json const& section)
{
    Keys(section, {"weights", "pathTolerance"});
    auto const& weights = section.at("weights");
    Require(weights.is_object() && weights.size() == RewardCount, "reward channel mismatch");
    RewardSettings reward{};
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        reward.weights[channel] = Number(weights, RewardNames[channel]);
    }
    reward.pathTolerance = Number(section, "pathTolerance");
    return reward;
}

/** @brief Reads the kind name of a stage: "approach", "pattern" or "holding". */
StageKind ReadStageKind(Json const& stage)
{
    auto const& value = stage.at("kind");
    Require(value.is_string(), "curriculum stage kind");
    auto const kind = value.get<std::string>();
    if (kind == "approach")
    {
        return StageKind::Approach;
    }
    if (kind == "pattern")
    {
        return StageKind::Pattern;
    }
    Require(kind == "holding", "curriculum stage kind must be approach, pattern or holding");
    return StageKind::Holding;
}

/** @brief Reads one curriculum stage; each kind has its own exact set of fields. */
StageSettings ReadStage(Json const& section)
{
    Require(section.is_object() && section.contains("kind") && section.contains("name")
        && section.at("name").is_string(), "curriculum stage name and kind");
    StageSettings stage{};
    stage.name = section.at("name").get<std::string>();
    stage.kind = ReadStageKind(section);
    if (stage.kind == StageKind::Approach)
    {
        Keys(section, {"name", "kind", "distanceMeters", "bearingDegrees", "headingOffsetDegrees",
            "heightAboveGlideMeters", "gearUpShare"});
        stage.distanceMeters = Bounds(section, "distanceMeters");
        stage.bearingDegrees = Bounds(section, "bearingDegrees");
        stage.heightMeters = Bounds(section, "heightAboveGlideMeters");
    }
    else if (stage.kind == StageKind::Pattern)
    {
        Keys(section, {"name", "kind", "startAlongMeters", "downwindOffsetMeters", "headingOffsetDegrees",
            "heightMeters", "gearUpShare"});
        stage.startAlongMeters = Bounds(section, "startAlongMeters");
        stage.lateralMeters = Bounds(section, "downwindOffsetMeters");
        stage.heightMeters = Bounds(section, "heightMeters");
    }
    else
    {
        Keys(section, {"name", "kind", "startAlongMeters", "startCrossMeters", "headingOffsetDegrees",
            "heightMeters", "fixAlongMeters", "outboundMeters", "widthMeters", "gearUpShare"});
        stage.startAlongMeters = Bounds(section, "startAlongMeters");
        stage.lateralMeters = Bounds(section, "startCrossMeters");
        stage.heightMeters = Bounds(section, "heightMeters");
        stage.fixAlongMeters = Number(section, "fixAlongMeters");
        stage.outboundMeters = Number(section, "outboundMeters");
        stage.holdingWidthMeters = Number(section, "widthMeters");
    }
    stage.headingOffsetDegrees = Bounds(section, "headingOffsetDegrees");
    stage.gearUpShare = Number(section, "gearUpShare");
    return stage;
}

CurriculumSettings ReadCurriculum(Json const& section)
{
    Keys(section, {"firstDistanceMeters", "levels", "promotionSuccessRate", "windowEpisodes",
        "rolloutStartShare", "stages"});
    CurriculumSettings curriculum{Bounds(section, "firstDistanceMeters"),
        Integer(section, "levels", MaximumCurriculumLevels), Number(section, "promotionSuccessRate"),
        Integer(section, "windowEpisodes", MaximumCurriculumWindowEpisodes),
        Number(section, "rolloutStartShare"), {}};
    auto const& stages = section.at("stages");
    Require(stages.is_array() && stages.size() <= MaximumCurriculumStages, "curriculum stages");
    for (auto const& stage : stages)
    {
        curriculum.stages.push_back(ReadStage(stage));
    }
    return curriculum;
}

TrainingSettings ReadTraining(Json const& section)
{
    Keys(section, {"seed", "randomizeEnvironmentSeed", "rolloutSteps", "minibatch", "epochs", "maxUpdates",
        "checkpointEvery", "hiddenWidth", "actorRate", "criticRate", "gamma", "lambda", "clip", "entropy",
        "targetKl", "adamBeta1", "adamBeta2", "adamEpsilon", "resume", "checkpointDirectory",
        "curriculum"});
    TrainingSettings training{};
    training.seed = static_cast<std::uint32_t>(Integer(section, "seed", UINT32_MAX));
    training.randomizeEnvironmentSeed = Boolean(section, "randomizeEnvironmentSeed");
    training.rolloutSteps = Integer(section, "rolloutSteps", MaximumRolloutSteps);
    training.minibatch = Integer(section, "minibatch", MaximumMinibatch);
    training.epochs = Integer(section, "epochs", MaximumEpochs);
    training.maxUpdates = Integer(section, "maxUpdates", MaximumUpdateCount);
    training.checkpointEvery = Integer(section, "checkpointEvery", MaximumUpdateCount);
    training.hiddenWidth = Integer(section, "hiddenWidth", MaximumHiddenWidth);
    training.actorRate = Number(section, "actorRate");
    training.criticRate = Number(section, "criticRate");
    training.gamma = Number(section, "gamma");
    training.lambda = Number(section, "lambda");
    training.clip = Number(section, "clip");
    training.entropy = Number(section, "entropy");
    training.targetKl = Number(section, "targetKl");
    training.adamBeta1 = Number(section, "adamBeta1");
    training.adamBeta2 = Number(section, "adamBeta2");
    training.adamEpsilon = Number(section, "adamEpsilon");
    training.resume = Boolean(section, "resume");
    Require(section.at("checkpointDirectory").is_string(), "checkpointDirectory");
    training.checkpointDirectory = section.at("checkpointDirectory").get<std::string>();
    training.curriculum = ReadCurriculum(section.at("curriculum"));
    return training;
}

EvaluationSettings ReadEvaluation(Json const& section)
{
    Keys(section, {"seed", "episodes", "everyUpdates", "minimumSuccessRate"});
    return {static_cast<std::uint32_t>(Integer(section, "seed", UINT32_MAX)),
        Integer(section, "episodes", MaximumEvaluationEpisodes), Integer(section, "everyUpdates",
            MaximumUpdateCount),
        Number(section, "minimumSuccessRate")};
}

DiagnosticsSettings ReadDiagnostics(Json const& section)
{
    Keys(section, {"enabled", "logEveryUpdates", "saturationThreshold"});
    return {Boolean(section, "enabled"), Integer(section, "logEveryUpdates", MaximumUpdateCount),
        Number(section, "saturationThreshold")};
}

RuntimeSettings ReadRuntime(Json const& section)
{
    Keys(section, {"maxSpeed"});
    RuntimeSettings runtime{};
    runtime.maxSpeed = Boolean(section, "maxSpeed");
    return runtime;
}

/**
 * @brief Builds the checkpoint contract from every setting that changes the learned task.
 * @param root Complete scenario document.
 * @return Canonical JSON text; training, evaluation and runtime settings are excluded.
 */
std::string Signature(Json root)
{
    for (auto const* section : {"training", "evaluation", "diagnostics", "runtime"})
    {
        root.erase(section);
    }
    root["contractVersion"] = ContractVersion;
    return root.dump();
}

// ---------------------------------------------------------------------------
// Validation, one function per concern. Each Require message is user-facing.
// ---------------------------------------------------------------------------

/** @brief Largest absolute bound of a range. */
double MaximumMagnitude(Range range) noexcept
{
    return std::max(std::abs(range.minimum), std::abs(range.maximum));
}

void ValidateRunway(Config const& config)
{
    auto const& runway = config.runway;
    auto const& randomization = config.runwayRandomization;
    Require(std::abs(runway.latitudeDegrees) < MaximumAbsoluteLatitudeDegrees
        && std::abs(runway.longitudeDegrees) <= MaximumAbsoluteLongitudeDegrees,
            "runway geodetic coordinates");
    Require(runway.headingDegrees >= 0 && runway.headingDegrees < FullTurnDegrees && runway.lengthMeters > 0
        && runway.widthMeters > 0 && runway.widthMeters < runway.lengthMeters
            && runway.thresholdHeightMeters >= 0,
        "runway geometry");
    Require(runway.landingMarginMeters >= 0 && runway.landingMarginMeters <= MaximumLandingMarginMeters,
        "landing area margin");
    Require(randomization.headingOffsetDegrees.minimum >= -HalfTurnDegrees
        && randomization.headingOffsetDegrees.maximum <= HalfTurnDegrees
        && MaximumMagnitude(randomization.northMeters) <= config.mission.maxDistanceMeters
        && MaximumMagnitude(randomization.eastMeters) <= config.mission.maxDistanceMeters,
        "bounded runway spawn position/course");
    Require(randomization.deckHeightMeters.minimum > randomization.seaHeightMeters,
        "deck must remain above sea");
}

void ValidateMission(Config const& config)
{
    auto const& mission = config.mission;
    Require(mission.physicsDt >= MinimumPhysicsDtSeconds && mission.physicsDt <= MaximumPhysicsDtSeconds
        && mission.actionSubsteps > 0 && config.StepSeconds() <= MaximumActionSeconds,
            "integration timestep");
    Require(mission.controlFilterTimeConstantSeconds > 0
        && mission.controlFilterTimeConstantSeconds <= MaximumFilterTimeConstantSeconds,
        "control filter time constant");
    Require(mission.airborneRudderAuthority >= 0 && mission.airborneRudderAuthority <= 1,
        "airborne rudder authority within [0,1]");
    Require(mission.maxSeconds > config.StepSeconds() && mission.maxSeconds <= MaximumEpisodeSeconds
        && mission.maxDistanceMeters > config.runway.lengthMeters
        && mission.maxAltitudeMeters > mission.flareHeightMeters, "episode limits");
    Require(mission.glideDegrees > 0 && mission.glideDegrees < MaximumGlideDegrees, "approach slope");
    // The glide plane crosses the threshold at its crossing height and meets
    // the runway at the aim point, which must lie well inside the runway.
    double const glideRadians = mission.glideDegrees * DegreesToRadians;
    double const aimAlong = config.runway.thresholdHeightMeters / std::tan(glideRadians);
    Require(config.runway.thresholdHeightMeters > 0 && aimAlong <= config.runway.lengthMeters / 2,
        "aim point must lie on the first half of the runway");
    Require(mission.approachSpeed > mission.flareSpeed && mission.flareSpeed > mission.stoppedSpeed
        && mission.stoppedSpeed >= 0 && mission.flareHeightMeters > mission.contactHeightMeters,
        "target speeds and flare");
    // Both reference speeds must keep the 1.3 Vs stall reserve of the aircraft model.
    Require(mission.flareSpeed >= MinimumCalibratedSpeed(config.performance, config.initial.approachFlap)
        && mission.approachSpeed >= MinimumCalibratedSpeed(config.performance, 0),
        "calibrated speeds must retain the model stall reserve");
    Require(mission.maxTouchdownSink > 0 && mission.maxTouchdownSpeed >= mission.flareSpeed
        && mission.maxTouchdownRollDegrees > 0
        && mission.minTouchdownPitchDegrees < mission.maxTouchdownPitchDegrees
        && mission.maxTouchdownPitchDegrees < QuarterTurnDegrees
        && mission.maxTouchdownHeadingDegrees > 0 && mission.maxTouchdownHeadingDegrees < QuarterTurnDegrees
        && mission.gearDownMinimum > 0 && mission.gearDownMinimum <= 1 && mission.contactHeightMeters > 0
        && mission.stoppedHoldSeconds > 0, "landing limits");
}

/** @brief Checks that every airborne start stays above the flare height and below the altitude limit. */
void ValidateApproachStartHeights(Config const& config)
{
    auto const& initial = config.initial;
    auto const& mission = config.mission;
    auto const& runway = config.runway;
    double const gradient = std::tan(mission.glideDegrees * DegreesToRadians);
    // Lowest start: shortest distance at the widest bearing, minus the height offset.
    double const minimumHeight = runway.thresholdHeightMeters
        + initial.distanceMeters.minimum * std::cos(initial.bearingDegrees.maximum * DegreesToRadians)
            * gradient
        - initial.heightOffsetMeters;
    double const maximumHeight = runway.thresholdHeightMeters + initial.distanceMeters.maximum * gradient
        + initial.heightOffsetMeters;
    Require(minimumHeight > mission.flareHeightMeters && maximumHeight < mission.maxAltitudeMeters,
        "approach starts must remain airborne inside the altitude envelope");
}

void ValidateInitial(Config const& config)
{
    auto const& initial = config.initial;
    auto const& mission = config.mission;
    auto const& runway = config.runway;
    Require(initial.distanceMeters.minimum > runway.lengthMeters
        && initial.distanceMeters.maximum <= mission.maxDistanceMeters, "bounded airborne spawn distance");
    Require(initial.bearingDegrees.minimum > 0 && initial.bearingDegrees.maximum < QuarterTurnDegrees
        && initial.headingOffsetDegrees.minimum >= -QuarterTurnDegrees
        && initial.headingOffsetDegrees.maximum <= QuarterTurnDegrees, "spawn course");
    Require(initial.speedToleranceMps > 0
        && mission.approachSpeed - initial.speedToleranceMps > mission.flareSpeed, "spawn speed envelope");
    Require(initial.rollDegrees >= 0 && initial.rollDegrees <= MaximumInitialRollDegrees
        && initial.pitchDegrees.minimum >= -MaximumAbsoluteInitialPitchDegrees
        && initial.pitchDegrees.maximum <= MaximumAbsoluteInitialPitchDegrees
            && initial.slopeToleranceDegrees >= 0
        && mission.glideDegrees + initial.slopeToleranceDegrees <= MaximumGlideDegrees
        && initial.heightOffsetMeters >= 0, "spawn attitude envelope");
    ValidateApproachStartHeights(config);
    auto const& rollout = initial.rolloutStart;
    Require(rollout.speedMps.minimum > mission.stoppedSpeed
        && rollout.speedMps.maximum <= mission.maxTouchdownSpeed,
        "rollout speed envelope");
    Require(rollout.alongMeters.minimum >= 0 && rollout.alongMeters.maximum < runway.lengthMeters
        && rollout.crossMeters.minimum >= -runway.widthMeters / RolloutCrossWidthDivisor
        && rollout.crossMeters.maximum <= runway.widthMeters / RolloutCrossWidthDivisor,
            "rollout spawn envelope");
    Require(initial.throttle >= 0 && initial.throttle <= 1 && initial.elevatorTrim > -1
        && initial.elevatorTrim < 1
        && initial.approachFlap >= 0 && initial.approachFlap <= 1
        && initial.rolloutFlap >= 0 && initial.rolloutFlap <= 1 && initial.propeller >= 0
            && initial.propeller <= 1
        && initial.mixture >= 0 && initial.mixture <= 1, "initial and phase controls");
}

void ValidateReward(Config const& config)
{
    auto const& reward = config.reward;
    Require(reward.pathTolerance > 0, "reward distance scale");
    for (double const weight : reward.weights)
    {
        Require(std::isfinite(weight), "finite reward weights");
    }
    for (auto const channel : {Reward::Path, Reward::Direction, Reward::VerticalSpeed, Reward::Speed,
             Reward::Touchdown, Reward::Stop})
    {
        Require(reward.weights[static_cast<std::size_t>(channel)] >= 0, "non-negative reward weight");
    }
    Require(reward.weights[static_cast<std::size_t>(Reward::Crash)] <= 0,
        "crash weight must be non-positive");
}

void ValidateTraining(Config const& config)
{
    auto const& training = config.training;
    Require(training.rolloutSteps >= 2 && training.minibatch > 0
        && training.minibatch <= training.rolloutSteps
        && training.rolloutSteps % training.minibatch == 0 && training.epochs > 0 && training.maxUpdates > 0
        && training.checkpointEvery > 0, "training capacities");
    Require(training.hiddenWidth == 32 || training.hiddenWidth == 64 || training.hiddenWidth == 128
        || training.hiddenWidth == 256, "hiddenWidth must be 32, 64, 128 or 256");
    Require(training.actorRate > 0 && training.criticRate > 0 && training.gamma > 0 && training.gamma <= 1
        && training.lambda >= 0 && training.lambda <= 1 && training.clip > 0 && training.clip < 1
        && training.entropy >= 0, "PPO hyperparameters");
    Require(training.targetKl > 0 && training.targetKl <= 1, "PPO target KL");
    Require(training.adamBeta1 >= 0 && training.adamBeta1 < 1 && training.adamBeta2 >= 0
        && training.adamBeta2 < 1
        && training.adamEpsilon > 0, "Adam hyperparameters");
}

/** @brief Height of the glide plane at a distance before the threshold, in metres. */
double GlideHeightAt(Config const& config, double distanceMeters) noexcept
{
    return config.runway.thresholdHeightMeters
        + distanceMeters * std::tan(config.mission.glideDegrees * DegreesToRadians);
}

/**
 * @brief Checks one curriculum stage: starts inside the envelope and the observation ranges,
 *        above the flare height, and route gates far enough apart to be passed one by one.
 */
void ValidateStage(Config const& config, StageSettings const& stage)
{
    auto const& mission = config.mission;
    Require(!stage.name.empty() && stage.gearUpShare >= 0 && stage.gearUpShare <= 1
        && MaximumMagnitude(stage.headingOffsetDegrees) <= HalfTurnDegrees,
        "curriculum stage name, gear or heading");
    if (stage.kind == StageKind::Approach)
    {
        double const largestBearing = MaximumMagnitude(stage.bearingDegrees);
        Require(stage.distanceMeters.minimum >= MinimumFirstCurriculumDistanceMeters
            && stage.distanceMeters.maximum < mission.maxDistanceMeters && largestBearing < QuarterTurnDegrees
            && stage.distanceMeters.maximum * std::sin(largestBearing * DegreesToRadians) < MaximumStageCrossMeters,
            "approach stage distance or bearing");
        Require(GlideHeightAt(config, stage.distanceMeters.minimum) + stage.heightMeters.minimum
            > mission.flareHeightMeters
            && GlideHeightAt(config, stage.distanceMeters.maximum) + stage.heightMeters.maximum
            < MaximumStageHeightMeters, "approach stage height");
        return;
    }
    Require(stage.heightMeters.minimum > mission.flareHeightMeters
        && stage.heightMeters.maximum < MaximumStageHeightMeters
        && MaximumMagnitude(stage.startAlongMeters) < mission.maxDistanceMeters
        && MaximumMagnitude(stage.lateralMeters) < MaximumStageCrossMeters, "pattern or holding stage start");
    if (stage.kind == StageKind::Pattern)
    {
        // The start lies before the downwind end, and the base gate keeps away from the final one.
        Require(stage.lateralMeters.minimum * PatternBaseOffsetFraction >= MinimumGateSpacingMeters
            && stage.startAlongMeters.minimum >= PatternDownwindEndAlongMeters + MinimumGateSpacingMeters
            && -PatternFinalAlongMeters < mission.maxDistanceMeters, "pattern stage geometry");
        return;
    }
    // A holding start is inbound to the fix, before it.
    Require(stage.startAlongMeters.maximum <= stage.fixAlongMeters - MinimumGateSpacingMeters
        && stage.fixAlongMeters <= -MinimumGateSpacingMeters
        && stage.outboundMeters >= MinimumGateSpacingMeters && stage.holdingWidthMeters >= MinimumGateSpacingMeters
        && std::hypot(stage.fixAlongMeters - stage.outboundMeters, stage.holdingWidthMeters)
        < mission.maxDistanceMeters
        && stage.holdingWidthMeters < MaximumStageCrossMeters, "holding stage geometry");
}

void ValidateCurriculum(Config const& config)
{
    auto const& curriculum = config.training.curriculum;
    Require(curriculum.firstDistanceMeters.minimum >= MinimumFirstCurriculumDistanceMeters
        && curriculum.firstDistanceMeters.maximum <= config.initial.distanceMeters.minimum
        && curriculum.levels >= 1 && curriculum.windowEpisodes >= 1
        && curriculum.promotionSuccessRate > 0 && curriculum.promotionSuccessRate <= 1
        && curriculum.rolloutStartShare >= 0 && curriculum.rolloutStartShare < 1, "training curriculum");
    double const firstHeight = config.runway.thresholdHeightMeters
        + curriculum.firstDistanceMeters.minimum * std::tan(config.mission.glideDegrees * DegreesToRadians);
    Require(firstHeight > config.mission.flareHeightMeters,
        "the first curriculum starts must remain above the flare height");
    for (auto const& stage : curriculum.stages)
    {
        ValidateStage(config, stage);
    }
}

/**
 * @brief Checks the evaluation budget and that evaluation seeds never overlap the training seed.
 *
 * Episode i of a campaign uses seed evaluation.seed + i, so the whole seed range must fit
 * in 32 bits and must not contain training.seed.
 */
void ValidateEvaluation(Config const& config)
{
    auto const& training = config.training;
    auto const& evaluation = config.evaluation;
    Require(evaluation.episodes > 0
        && evaluation.episodes - 1U <= std::numeric_limits<std::uint32_t>::max() - evaluation.seed
        && (training.seed < evaluation.seed || training.seed - evaluation.seed >= evaluation.episodes)
        && evaluation.everyUpdates > 0
        && evaluation.minimumSuccessRate >= 0 && evaluation.minimumSuccessRate <= 1,
        "independent evaluation seed, budget and success threshold");
}

void ValidateDiagnosticsAndPaths(Config const& config)
{
    Require(config.diagnostics.logEveryUpdates > 0
        && config.diagnostics.saturationThreshold >= MinimumSaturationThreshold
        && config.diagnostics.saturationThreshold <= MaximumSaturationThreshold,
            "diagnostic logging settings");
    auto const directory = std::filesystem::path(config.training.checkpointDirectory);
    Require(!directory.empty() && !directory.is_absolute(), "checkpointDirectory must be relative");
    for (auto const& component : directory)
    {
        Require(component != "..", "checkpointDirectory traversal");
    }
}

/** @brief Learning-side rules, in the historical order of their error messages. */
void ValidateLearning(Config const& config)
{
    ValidateReward(config);
    ValidateTraining(config);
    ValidateCurriculum(config);
    ValidateEvaluation(config);
    ValidateDiagnosticsAndPaths(config);
}
} // namespace

Config LoadConfig(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    Require(static_cast<bool>(stream), "cannot open configuration");
    Json root;
    stream >> root;
    Keys(root, {"formatVersion", "aircraft", "runway", "runwayRandomization", "mission", "initial",
        "reward", "training", "evaluation", "diagnostics", "runtime"});
    Require(Integer(root, "formatVersion", ConfigurationFormatVersion) == ConfigurationFormatVersion,
        "unsupported formatVersion");
    Config config{};
    Require(root.at("aircraft").is_string(), "aircraft");
    config.aircraft = root.at("aircraft").get<std::string>();
    if (auto const* const performance = FindAircraftPerformance(config.aircraft))
    {
        config.performance = *performance;
    }
    config.runway = ReadRunway(root.at("runway"));
    config.runwayRandomization = ReadRunwayRandomization(root.at("runwayRandomization"));
    config.mission = ReadMission(root.at("mission"));
    config.initial = ReadInitial(root.at("initial"));
    config.reward = ReadReward(root.at("reward"));
    config.training = ReadTraining(root.at("training"));
    config.evaluation = ReadEvaluation(root.at("evaluation"));
    config.diagnostics = ReadDiagnostics(root.at("diagnostics"));
    config.runtime = ReadRuntime(root.at("runtime"));
    config.checkpointSignature = Signature(root);
    Validate(config);
    return config;
}

bool CompatibleTrainingSignature(std::string const& stored, std::string const& current)
{
    if (stored == current)
    {
        return true;
    }
    auto previous = Json::parse(stored, nullptr, false);
    auto next = Json::parse(current, nullptr, false);
    if (!previous.is_object() || !next.is_object())
    {
        return false;
    }
    // Reward weights and spawn distributions change what is learned next, not the
    // network contract: such a checkpoint can warm-start training. All other
    // fields, including the rest of "initial", must match exactly.
    for (auto* signature : {&previous, &next})
    {
        auto const initial = signature->find("initial");
        if (initial == signature->end() || !initial->is_object())
        {
            return false;
        }
        for (char const* spawnField : WarmStartSpawnFields)
        {
            initial->erase(spawnField);
        }
        signature->erase("reward");
    }
    return previous == next;
}

void Validate(Config const& config)
{
    auto const* const performance = FindAircraftPerformance(config.aircraft);
    Require(performance != nullptr && performance->name == config.performance.name,
        "runtime aircraft must be t6texan2 or pc7, with its performance data");
    ValidateRunway(config);
    ValidateMission(config);
    ValidateInitial(config);
    ValidateLearning(config);
}
} // namespace ppo::examples::flight
