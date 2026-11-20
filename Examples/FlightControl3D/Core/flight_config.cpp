#include "flight_config.hpp"
#include "flight_t6_performance.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ppo::examples::flight {
namespace {
using Json = nlohmann::json;

/** @brief Version of the observation, action, guidance, reward and spawn semantics stored in checkpoints. */
constexpr int ContractVersion = 5;
constexpr int ConfigurationFormatVersion = 2;

void Require(bool condition, char const* message)
{
    if (!condition) throw std::invalid_argument(std::string{"FlightControl3D: "} + message);
}

/** @brief Requires an object with exactly the listed keys. */
void Keys(Json const& object, std::initializer_list<char const*> names)
{
    Require(object.is_object() && object.size() == names.size(), "unexpected or missing JSON fields");
    for (auto const* name : names) Require(object.contains(name), name);
}

double Number(Json const& object, char const* key)
{
    auto const& value = object.at(key);
    Require(value.is_number() && !value.is_boolean(), key);
    double const result = value.get<double>();
    Require(std::isfinite(result), key);
    return result;
}

std::size_t Integer(Json const& object, char const* key, std::size_t limit)
{
    auto const& value = object.at(key);
    Require(value.is_number_unsigned() || value.is_number_integer(), key);
    auto const result = value.get<std::int64_t>();
    Require(result >= 0 && static_cast<std::uint64_t>(result) <= limit, key);
    return static_cast<std::size_t>(result);
}

bool Boolean(Json const& object, char const* key)
{
    Require(object.at(key).is_boolean(), key);
    return object.at(key).get<bool>();
}

Range Bounds(Json const& object, char const* key)
{
    auto const& value = object.at(key);
    Require(value.is_array() && value.size() == 2 && value[0].is_number() && value[1].is_number(), key);
    Range const range{value[0].get<double>(), value[1].get<double>()};
    Require(std::isfinite(range.minimum) && std::isfinite(range.maximum) && range.minimum <= range.maximum, key);
    return range;
}

Runway ReadRunway(Json const& r)
{
    Keys(r, {"latitudeDegrees", "longitudeDegrees", "elevationMeters", "headingDegrees", "lengthMeters",
        "widthMeters", "thresholdHeightMeters", "landingMarginMeters"});
    Runway runway{};
    runway.latitudeDegrees = Number(r, "latitudeDegrees");
    runway.longitudeDegrees = Number(r, "longitudeDegrees");
    runway.elevationMeters = Number(r, "elevationMeters");
    runway.headingDegrees = Number(r, "headingDegrees");
    runway.lengthMeters = Number(r, "lengthMeters");
    runway.widthMeters = Number(r, "widthMeters");
    runway.thresholdHeightMeters = Number(r, "thresholdHeightMeters");
    runway.landingMarginMeters = Number(r, "landingMarginMeters");
    return runway;
}

RunwayRandomization ReadRunwayRandomization(Json const& d)
{
    Keys(d, {"northMeters", "eastMeters", "headingOffsetDegrees", "deckHeightMeters", "seaHeightMeters"});
    return {Bounds(d, "northMeters"), Bounds(d, "eastMeters"), Bounds(d, "headingOffsetDegrees"),
        Bounds(d, "deckHeightMeters"), Number(d, "seaHeightMeters")};
}

MissionSettings ReadMission(Json const& m)
{
    Keys(m, {"physicsDt", "controlFilterTimeConstantSeconds", "airborneRudderAuthority", "actionSubsteps",
        "maxSeconds", "maxDistanceMeters", "maxAltitudeMeters", "glideDegrees", "approachSpeed", "flareSpeed",
        "flareHeightMeters", "maxTouchdownSink", "maxTouchdownSpeed", "maxTouchdownRollDegrees",
        "maxTouchdownPitchDegrees", "minTouchdownPitchDegrees", "maxTouchdownHeadingDegrees",
        "gearDownMinimum", "contactHeightMeters", "stoppedSpeed", "stoppedHoldSeconds"});
    MissionSettings mission{};
#define READ_M(field) mission.field = Number(m, #field)
    READ_M(physicsDt); READ_M(controlFilterTimeConstantSeconds); READ_M(airborneRudderAuthority);
    READ_M(maxSeconds); READ_M(maxDistanceMeters); READ_M(maxAltitudeMeters); READ_M(glideDegrees);
    READ_M(approachSpeed); READ_M(flareSpeed); READ_M(flareHeightMeters);
    READ_M(maxTouchdownSink); READ_M(maxTouchdownSpeed); READ_M(maxTouchdownRollDegrees);
    READ_M(maxTouchdownPitchDegrees); READ_M(minTouchdownPitchDegrees); READ_M(maxTouchdownHeadingDegrees);
    READ_M(gearDownMinimum); READ_M(contactHeightMeters); READ_M(stoppedSpeed); READ_M(stoppedHoldSeconds);
#undef READ_M
    mission.actionSubsteps = Integer(m, "actionSubsteps", 100);
    return mission;
}

InitialSettings ReadInitial(Json const& i)
{
    Keys(i, {"distanceMeters", "bearingDegrees", "headingOffsetDegrees", "heightOffsetMeters",
        "speedToleranceMps", "rollDegrees", "pitchDegrees", "slopeToleranceDegrees", "rolloutStart",
        "throttle", "elevatorTrim", "approachFlap", "rolloutFlap", "propeller", "mixture"});
    InitialSettings initial{};
    initial.distanceMeters = Bounds(i, "distanceMeters");
    initial.bearingDegrees = Bounds(i, "bearingDegrees");
    initial.headingOffsetDegrees = Bounds(i, "headingOffsetDegrees");
    initial.heightOffsetMeters = Number(i, "heightOffsetMeters");
    initial.speedToleranceMps = Number(i, "speedToleranceMps");
    initial.rollDegrees = Number(i, "rollDegrees");
    initial.pitchDegrees = Bounds(i, "pitchDegrees");
    initial.slopeToleranceDegrees = Number(i, "slopeToleranceDegrees");
    auto const& rollout = i.at("rolloutStart");
    Keys(rollout, {"speedMps", "alongMeters", "crossMeters"});
    initial.rolloutStart = {Bounds(rollout, "speedMps"), Bounds(rollout, "alongMeters"),
        Bounds(rollout, "crossMeters")};
    initial.throttle = Number(i, "throttle");
    initial.elevatorTrim = Number(i, "elevatorTrim");
    initial.approachFlap = Number(i, "approachFlap");
    initial.rolloutFlap = Number(i, "rolloutFlap");
    initial.propeller = Number(i, "propeller");
    initial.mixture = Number(i, "mixture");
    return initial;
}

RewardSettings ReadReward(Json const& w)
{
    Keys(w, {"weights", "pathTolerance"});
    auto const& weights = w.at("weights");
    Require(weights.is_object() && weights.size() == RewardCount, "reward channel mismatch");
    RewardSettings reward{};
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
        reward.weights[channel] = Number(weights, RewardNames[channel]);
    reward.pathTolerance = Number(w, "pathTolerance");
    return reward;
}

CurriculumSettings ReadCurriculum(Json const& c)
{
    Keys(c, {"firstDistanceMeters", "levels", "promotionSuccessRate", "windowEpisodes", "rolloutStartShare"});
    return {Bounds(c, "firstDistanceMeters"), Integer(c, "levels", 100), Number(c, "promotionSuccessRate"),
        Integer(c, "windowEpisodes", 100000), Number(c, "rolloutStartShare")};
}

TrainingSettings ReadTraining(Json const& t)
{
    Keys(t, {"seed", "randomizeEnvironmentSeed", "rolloutSteps", "minibatch", "epochs", "maxUpdates",
        "checkpointEvery", "hiddenWidth", "actorRate", "criticRate", "gamma", "lambda", "clip", "entropy",
        "valueWeight", "targetKl", "adamBeta1", "adamBeta2", "adamEpsilon", "resume", "checkpointDirectory",
        "curriculum"});
    TrainingSettings training{};
    training.seed = static_cast<std::uint32_t>(Integer(t, "seed", UINT32_MAX));
    training.randomizeEnvironmentSeed = Boolean(t, "randomizeEnvironmentSeed");
    training.rolloutSteps = Integer(t, "rolloutSteps", 65536);
    training.minibatch = Integer(t, "minibatch", 65536);
    training.epochs = Integer(t, "epochs", 100);
    training.maxUpdates = Integer(t, "maxUpdates", 100000000);
    training.checkpointEvery = Integer(t, "checkpointEvery", 100000000);
    training.hiddenWidth = Integer(t, "hiddenWidth", 256);
#define READ_T(field) training.field = Number(t, #field)
    READ_T(actorRate); READ_T(criticRate); READ_T(gamma); READ_T(lambda); READ_T(clip); READ_T(entropy);
    READ_T(valueWeight); READ_T(targetKl); READ_T(adamBeta1); READ_T(adamBeta2); READ_T(adamEpsilon);
#undef READ_T
    training.resume = Boolean(t, "resume");
    Require(t.at("checkpointDirectory").is_string(), "checkpointDirectory");
    training.checkpointDirectory = t.at("checkpointDirectory").get<std::string>();
    training.curriculum = ReadCurriculum(t.at("curriculum"));
    return training;
}

EvaluationSettings ReadEvaluation(Json const& e)
{
    Keys(e, {"seed", "episodes", "everyUpdates", "minimumSuccessRate"});
    return {static_cast<std::uint32_t>(Integer(e, "seed", UINT32_MAX)), Integer(e, "episodes", 1000000),
        Integer(e, "everyUpdates", 100000000), Number(e, "minimumSuccessRate")};
}

/**
 * @brief Builds the checkpoint contract from every setting that changes the learned task.
 * @param root Complete scenario document.
 * @return Canonical JSON text; training, evaluation and runtime settings are excluded.
 */
std::string Signature(Json root)
{
    for (auto const* section : {"training", "evaluation", "diagnostics", "runtime"}) root.erase(section);
    root["contractVersion"] = ContractVersion;
    return root.dump();
}

void ValidateRunway(Config const& c)
{
    auto const& r = c.runway;
    auto const& d = c.runwayRandomization;
    Require(std::abs(r.latitudeDegrees) < 85 && std::abs(r.longitudeDegrees) <= 180, "runway geodetic coordinates");
    Require(r.headingDegrees >= 0 && r.headingDegrees < 360 && r.lengthMeters > 0 && r.widthMeters > 0
        && r.widthMeters < r.lengthMeters && r.thresholdHeightMeters >= 0, "runway geometry");
    Require(r.landingMarginMeters >= 0 && r.landingMarginMeters <= 10000, "landing area margin");
    Require(d.headingOffsetDegrees.minimum >= -180 && d.headingOffsetDegrees.maximum <= 180
        && std::max(std::abs(d.northMeters.minimum), std::abs(d.northMeters.maximum)) <= c.mission.maxDistanceMeters
        && std::max(std::abs(d.eastMeters.minimum), std::abs(d.eastMeters.maximum)) <= c.mission.maxDistanceMeters,
        "bounded runway spawn position/course");
    Require(d.deckHeightMeters.minimum > d.seaHeightMeters, "deck must remain above sea");
}

void ValidateMission(Config const& c)
{
    auto const& m = c.mission;
    Require(m.physicsDt >= 0.001 && m.physicsDt <= 0.02 && m.actionSubsteps > 0 && c.StepSeconds() <= 0.5,
        "integration timestep");
    Require(m.controlFilterTimeConstantSeconds > 0 && m.controlFilterTimeConstantSeconds <= 10,
        "control filter time constant");
    Require(m.airborneRudderAuthority >= 0 && m.airborneRudderAuthority <= 1, "airborne rudder authority within [0,1]");
    Require(m.maxSeconds > c.StepSeconds() && m.maxSeconds <= 86400 && m.maxDistanceMeters > c.runway.lengthMeters
        && m.maxAltitudeMeters > m.flareHeightMeters, "episode limits");
    Require(m.glideDegrees > 0 && m.glideDegrees < 10, "approach slope");
    // The glide plane crosses the threshold at its crossing height and meets
    // the runway at the aim point, which must lie well inside the runway.
    double const aimAlong = c.runway.thresholdHeightMeters / std::tan(m.glideDegrees * DegreesToRadians);
    Require(c.runway.thresholdHeightMeters > 0 && aimAlong <= c.runway.lengthMeters / 2,
        "aim point must lie on the first half of the runway");
    Require(m.approachSpeed > m.flareSpeed && m.flareSpeed > m.stoppedSpeed && m.stoppedSpeed >= 0
        && m.flareHeightMeters > m.contactHeightMeters, "target speeds and flare");
    Require(m.flareSpeed >= T6MinimumCalibratedSpeed(c.initial.approachFlap)
        && m.approachSpeed >= T6MinimumCalibratedSpeed(0), "T6 calibrated speeds must retain the model stall reserve");
    Require(m.maxTouchdownSink > 0 && m.maxTouchdownSpeed >= m.flareSpeed && m.maxTouchdownRollDegrees > 0
        && m.minTouchdownPitchDegrees < m.maxTouchdownPitchDegrees && m.maxTouchdownPitchDegrees < 90
        && m.maxTouchdownHeadingDegrees > 0 && m.maxTouchdownHeadingDegrees < 90
        && m.gearDownMinimum > 0 && m.gearDownMinimum <= 1 && m.contactHeightMeters > 0
        && m.stoppedHoldSeconds > 0, "landing limits");
}

void ValidateInitial(Config const& c)
{
    auto const& i = c.initial;
    auto const& m = c.mission;
    auto const& r = c.runway;
    Require(i.distanceMeters.minimum > r.lengthMeters && i.distanceMeters.maximum <= m.maxDistanceMeters,
        "bounded airborne spawn distance");
    Require(i.bearingDegrees.minimum > 0 && i.bearingDegrees.maximum < 90
        && i.headingOffsetDegrees.minimum >= -90 && i.headingOffsetDegrees.maximum <= 90, "spawn course");
    Require(i.speedToleranceMps > 0 && m.approachSpeed - i.speedToleranceMps > m.flareSpeed,
        "spawn speed envelope");
    Require(i.rollDegrees >= 0 && i.rollDegrees <= 30 && i.pitchDegrees.minimum >= -15
        && i.pitchDegrees.maximum <= 15 && i.slopeToleranceDegrees >= 0
        && m.glideDegrees + i.slopeToleranceDegrees <= 10 && i.heightOffsetMeters >= 0, "spawn attitude envelope");
    double const gradient = std::tan(m.glideDegrees * DegreesToRadians);
    double const minimumHeight = r.thresholdHeightMeters
        + i.distanceMeters.minimum * std::cos(i.bearingDegrees.maximum * DegreesToRadians) * gradient
        - i.heightOffsetMeters;
    double const maximumHeight = r.thresholdHeightMeters + i.distanceMeters.maximum * gradient + i.heightOffsetMeters;
    Require(minimumHeight > m.flareHeightMeters && maximumHeight < m.maxAltitudeMeters,
        "approach starts must remain airborne inside the altitude envelope");
    auto const& rollout = i.rolloutStart;
    Require(rollout.speedMps.minimum > m.stoppedSpeed && rollout.speedMps.maximum <= m.maxTouchdownSpeed,
        "rollout speed envelope");
    Require(rollout.alongMeters.minimum >= 0 && rollout.alongMeters.maximum < r.lengthMeters
        && rollout.crossMeters.minimum >= -r.widthMeters / 4 && rollout.crossMeters.maximum <= r.widthMeters / 4,
        "rollout spawn envelope");
    Require(i.throttle >= 0 && i.throttle <= 1 && i.elevatorTrim > -1 && i.elevatorTrim < 1
        && i.approachFlap >= 0 && i.approachFlap <= 1
        && i.rolloutFlap >= 0 && i.rolloutFlap <= 1 && i.propeller >= 0 && i.propeller <= 1
        && i.mixture >= 0 && i.mixture <= 1, "initial and phase controls");
}

void ValidateLearning(Config const& c)
{
    auto const& w = c.reward;
    Require(w.pathTolerance > 0, "reward distance scale");
    for (double weight : w.weights) Require(std::isfinite(weight), "finite reward weights");
    for (auto channel : {Reward::Path, Reward::Direction, Reward::VerticalSpeed, Reward::Speed,
             Reward::Touchdown, Reward::Stop})
        Require(w.weights[static_cast<std::size_t>(channel)] >= 0, "non-negative reward weight");
    Require(w.weights[static_cast<std::size_t>(Reward::Crash)] <= 0, "crash weight must be non-positive");
    auto const& t = c.training;
    Require(t.rolloutSteps >= 2 && t.minibatch > 0 && t.minibatch <= t.rolloutSteps && t.rolloutSteps % t.minibatch == 0
        && t.epochs > 0 && t.maxUpdates > 0 && t.checkpointEvery > 0, "training capacities");
    Require(t.hiddenWidth == 32 || t.hiddenWidth == 64 || t.hiddenWidth == 128 || t.hiddenWidth == 256,
        "hiddenWidth must be 32, 64, 128 or 256");
    Require(t.actorRate > 0 && t.criticRate > 0 && t.gamma > 0 && t.gamma <= 1 && t.lambda >= 0 && t.lambda <= 1
        && t.clip > 0 && t.clip < 1 && t.entropy >= 0 && t.valueWeight > 0, "PPO hyperparameters");
    Require(t.targetKl > 0 && t.targetKl <= 1, "PPO target KL");
    Require(t.adamBeta1 >= 0 && t.adamBeta1 < 1 && t.adamBeta2 >= 0 && t.adamBeta2 < 1 && t.adamEpsilon > 0,
        "Adam hyperparameters");
    auto const& curriculum = t.curriculum;
    Require(curriculum.firstDistanceMeters.minimum >= 1000
        && curriculum.firstDistanceMeters.maximum <= c.initial.distanceMeters.minimum
        && curriculum.levels >= 1 && curriculum.windowEpisodes >= 1
        && curriculum.promotionSuccessRate > 0 && curriculum.promotionSuccessRate <= 1
        && curriculum.rolloutStartShare >= 0 && curriculum.rolloutStartShare < 1, "training curriculum");
    double const firstHeight = c.runway.thresholdHeightMeters
        + curriculum.firstDistanceMeters.minimum * std::tan(c.mission.glideDegrees * DegreesToRadians);
    Require(firstHeight > c.mission.flareHeightMeters, "the first curriculum starts must remain above the flare height");
    Require(c.evaluation.episodes > 0
        && c.evaluation.episodes - 1U <= std::numeric_limits<std::uint32_t>::max() - c.evaluation.seed
        && (t.seed < c.evaluation.seed || t.seed - c.evaluation.seed >= c.evaluation.episodes)
        && c.evaluation.everyUpdates > 0
        && c.evaluation.minimumSuccessRate >= 0 && c.evaluation.minimumSuccessRate <= 1,
        "independent evaluation seed, budget and success threshold");
    Require(c.diagnostics.logEveryUpdates > 0 && c.diagnostics.saturationThreshold >= 0.5
        && c.diagnostics.saturationThreshold <= 1.0, "diagnostic logging settings");
    auto const directory = std::filesystem::path(t.checkpointDirectory);
    Require(!directory.empty() && !directory.is_absolute(), "checkpointDirectory must be relative");
    for (auto const& component : directory) Require(component != "..", "checkpointDirectory traversal");
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
    config.runway = ReadRunway(root.at("runway"));
    config.runwayRandomization = ReadRunwayRandomization(root.at("runwayRandomization"));
    config.mission = ReadMission(root.at("mission"));
    config.initial = ReadInitial(root.at("initial"));
    config.reward = ReadReward(root.at("reward"));
    config.training = ReadTraining(root.at("training"));
    config.evaluation = ReadEvaluation(root.at("evaluation"));
    auto const& diagnostics = root.at("diagnostics");
    Keys(diagnostics, {"enabled", "logEveryUpdates", "saturationThreshold"});
    config.diagnostics = {Boolean(diagnostics, "enabled"), Integer(diagnostics, "logEveryUpdates", 100000000),
        Number(diagnostics, "saturationThreshold")};
    auto const& runtime = root.at("runtime");
    Keys(runtime, {"maxSpeed"});
    config.runtime.maxSpeed = Boolean(runtime, "maxSpeed");
    config.checkpointSignature = Signature(root);
    Validate(config);
    return config;
}

bool CompatibleTrainingSignature(std::string const& stored, std::string const& current)
{
    if (stored == current) return true;
    auto previous = Json::parse(stored, nullptr, false);
    auto next = Json::parse(current, nullptr, false);
    if (!previous.is_object() || !next.is_object()) return false;
    // Reward weights and start ranges change what is learned next, not the
    // network contract: such a checkpoint can warm-start training. The trimmed
    // controls define what an actor output means, so they must still match.
    for (auto* signature : {&previous, &next})
    {
        auto const initial = signature->find("initial");
        if (initial == signature->end() || !initial->is_object()) return false;
        Json const actionReference{{"throttle", initial->value("throttle", Json())},
            {"elevatorTrim", initial->value("elevatorTrim", Json())}};
        signature->erase("reward");
        signature->erase("initial");
        (*signature)["actionReference"] = actionReference;
    }
    return previous == next;
}

void Validate(Config const& c)
{
    Require(c.aircraft == "t6texan2", "runtime aircraft must be t6texan2");
    ValidateRunway(c);
    ValidateMission(c);
    ValidateInitial(c);
    ValidateLearning(c);
}
} // namespace ppo::examples::flight
