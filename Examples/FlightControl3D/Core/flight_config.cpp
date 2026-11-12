#include "flight_config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace ppo::examples::flight {
namespace {
using Json = nlohmann::json;

void Require(bool condition, char const* message)
{
    if (!condition) throw std::invalid_argument(std::string{"FlightControl3D: "} + message);
}

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
    auto const& value=object.at(key);
    Require(value.is_array() && value.size()==2 && value[0].is_number() && value[1].is_number(),key);
    Range const range{value[0].get<double>(),value[1].get<double>()};
    Require(std::isfinite(range.minimum) && std::isfinite(range.maximum) && range.minimum<=range.maximum,key);
    return range;
}
} // namespace

Config LoadConfig(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    Require(static_cast<bool>(stream), "cannot open configuration");
    Json root;
    stream >> root;
    Keys(root, {"formatVersion", "aircraft", "runway", "runwayRandomization", "mission", "initial", "normalization", "reward", "training", "evaluation"});
    Require(Integer(root, "formatVersion", 1) == 1, "unsupported formatVersion");
    Config config{};
    config.aircraft = root.at("aircraft").get<std::string>();
    auto const& r = root.at("runway");
    Keys(r, {"latitudeDegrees", "longitudeDegrees", "elevationMeters", "headingDegrees", "lengthMeters", "widthMeters", "thresholdHeightMeters"});
    config.runway = {Number(r,"latitudeDegrees"), Number(r,"longitudeDegrees"), Number(r,"elevationMeters"),
        Number(r,"headingDegrees"), Number(r,"lengthMeters"), Number(r,"widthMeters"), Number(r,"thresholdHeightMeters")};
    auto const& d=root.at("runwayRandomization");
    Keys(d,{"northMeters","eastMeters","headingOffsetDegrees","deckHeightMeters","speedMps","heaveAmplitudeMeters",
        "wavePeriodSeconds","pitchAmplitudeDegrees","rollAmplitudeDegrees","phaseDegrees","seaHeightMeters"});
    config.runwayRandomization={Bounds(d,"northMeters"),Bounds(d,"eastMeters"),Bounds(d,"headingOffsetDegrees"),Bounds(d,"deckHeightMeters"),
        Bounds(d,"speedMps"),Bounds(d,"heaveAmplitudeMeters"),Bounds(d,"wavePeriodSeconds"),Bounds(d,"pitchAmplitudeDegrees"),
        Bounds(d,"rollAmplitudeDegrees"),Bounds(d,"phaseDegrees"),Number(d,"seaHeightMeters")};
    auto const& m = root.at("mission");
    Keys(m, {"physicsDt", "actionSubsteps", "maxSeconds", "maxDistanceMeters", "maxAltitudeMeters",
        "switchDistanceMeters", "ilsRangeMeters", "ilsSectorDegrees", "glideDegrees",
        "localizerCaptureDegrees", "glideCaptureDegrees", "captureHoldSeconds", "captureHysteresis", "requiredLaps", "pathCorridorMeters", "maximumWaypointStepMeters",
        "patrolHeadingToleranceDegrees", "approachHeadingToleranceDegrees", "approachCrossToleranceMeters", "approachAltitudeToleranceMeters", "patrolSpeed", "approachSpeed", "flareSpeed", "flareHeightMeters",
        "maxTouchdownSink", "maxTouchdownSpeed", "maxTouchdownRollDegrees", "maxTouchdownPitchDegrees", "minTouchdownPitchDegrees", "maxTouchdownHeadingDegrees",
        "gearDownMinimum", "contactHeightMeters", "stoppedSpeed", "stoppedHoldSeconds", "racetrackCenterAlong", "racetrackCenterCross",
        "racetrackHalfStraight", "racetrackRadius", "patrolAltitude", "waypointRadius", "patrolLookaheadSeconds", "turnSamples", "straightSamples"});
    auto& mission = config.mission;
#define READ_M(field) mission.field = Number(m, #field)
    READ_M(physicsDt); READ_M(maxSeconds); READ_M(maxDistanceMeters); READ_M(maxAltitudeMeters);
    READ_M(switchDistanceMeters); READ_M(ilsRangeMeters); READ_M(ilsSectorDegrees); READ_M(glideDegrees);
    READ_M(localizerCaptureDegrees); READ_M(glideCaptureDegrees); READ_M(patrolSpeed); READ_M(approachSpeed);
    READ_M(captureHoldSeconds);READ_M(captureHysteresis);READ_M(pathCorridorMeters);READ_M(maximumWaypointStepMeters);
    READ_M(patrolHeadingToleranceDegrees);
    READ_M(approachHeadingToleranceDegrees);READ_M(approachCrossToleranceMeters);READ_M(approachAltitudeToleranceMeters);
    READ_M(flareSpeed); READ_M(flareHeightMeters); READ_M(maxTouchdownSink); READ_M(maxTouchdownSpeed);
    READ_M(maxTouchdownRollDegrees); READ_M(maxTouchdownPitchDegrees); READ_M(minTouchdownPitchDegrees); READ_M(maxTouchdownHeadingDegrees);
    READ_M(gearDownMinimum); READ_M(contactHeightMeters); READ_M(stoppedSpeed); READ_M(stoppedHoldSeconds);
    READ_M(racetrackCenterAlong); READ_M(racetrackCenterCross); READ_M(racetrackHalfStraight);
    READ_M(racetrackRadius); READ_M(patrolAltitude); READ_M(waypointRadius); READ_M(patrolLookaheadSeconds);
#undef READ_M
    mission.actionSubsteps = Integer(m, "actionSubsteps", 100);
    mission.turnSamples = Integer(m, "turnSamples", MaximumWaypoints);
    mission.straightSamples = Integer(m, "straightSamples", MaximumWaypoints);
    mission.requiredLaps=Integer(m,"requiredLaps",100);
    auto const& i = root.at("initial");
    Keys(i, {"approach", "gearDown", "runwayAlignedProbability", "distanceMeters", "bearingDegrees", "heightMeters", "speedMps", "headingOffsetDegrees",
        "rollDegrees", "pitchDegrees", "slopeDegrees", "throttle", "flap", "approachFlap", "rolloutFlap", "propeller"});
    auto& initial = config.initial;
    initial.approach = Boolean(i, "approach"); initial.gearDown = Boolean(i, "gearDown");
    initial.runwayAlignedProbability=Number(i,"runwayAlignedProbability");
    initial.distanceMeters=Bounds(i,"distanceMeters"); initial.bearingDegrees=Bounds(i,"bearingDegrees");
    initial.heightMeters=Bounds(i,"heightMeters"); initial.speedMps=Bounds(i,"speedMps");
    initial.headingOffsetDegrees=Bounds(i,"headingOffsetDegrees"); initial.rollDegrees=Bounds(i,"rollDegrees");
    initial.pitchDegrees=Bounds(i,"pitchDegrees"); initial.slopeDegrees=Bounds(i,"slopeDegrees");
    initial.throttle=Number(i,"throttle"); initial.flap=Number(i,"flap"); initial.propeller=Number(i,"propeller");
    initial.approachFlap=Number(i,"approachFlap");initial.rolloutFlap=Number(i,"rolloutFlap");
    auto const& n = root.at("normalization");
    Keys(n, {"distance", "altitude", "speed", "rate", "path", "glideDegrees"});
    config.normalization = {Number(n,"distance"), Number(n,"altitude"), Number(n,"speed"), Number(n,"rate"),
        Number(n,"path"), Number(n,"glideDegrees")};
    auto const& w = root.at("reward");
    Keys(w, {"weights", "pathTolerance", "localizerToleranceDegrees", "glideToleranceDegrees", "speedTolerance", "slopeToleranceDegrees",
        "localizerCourseBlend", "localizerInterceptGain", "localizerInterceptLimitDegrees"});
    auto const& weights = w.at("weights");
    Require(weights.is_object() && weights.size() == RewardCount, "reward channel mismatch");
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
        config.reward.weights[channel] = Number(weights, RewardNames[channel]);
    config.reward.pathTolerance = Number(w,"pathTolerance");
    config.reward.localizerToleranceDegrees = Number(w,"localizerToleranceDegrees");
    config.reward.glideToleranceDegrees = Number(w,"glideToleranceDegrees");
    config.reward.speedTolerance = Number(w,"speedTolerance");
    config.reward.slopeToleranceDegrees = Number(w,"slopeToleranceDegrees");
    config.reward.localizerCourseBlend = Number(w,"localizerCourseBlend");
    config.reward.localizerInterceptGain = Number(w,"localizerInterceptGain");
    config.reward.localizerInterceptLimitDegrees = Number(w,"localizerInterceptLimitDegrees");
    auto const& t = root.at("training");
    Keys(t, {"seed", "rolloutSteps", "minibatch", "epochs", "maxUpdates", "checkpointEvery", "hiddenWidth", "actorRate", "criticRate", "gamma", "lambda",
        "clip", "entropy", "valueWeight", "minLogStd", "maxLogStd", "initialLogStd", "maxGradientNorm", "adamBeta1", "adamBeta2", "adamEpsilon",
        "ilsCaptureScale", "curriculum", "resume", "checkpointDirectory"});
    auto& training = config.training;
    training.seed = static_cast<std::uint32_t>(Integer(t,"seed", UINT32_MAX));
    training.rolloutSteps = Integer(t,"rolloutSteps", 65536);
    training.minibatch = Integer(t,"minibatch", 65536);
    training.epochs = Integer(t,"epochs", 100);
    training.maxUpdates = Integer(t,"maxUpdates", 100000000);
    training.checkpointEvery = Integer(t,"checkpointEvery", 100000000);
    training.hiddenWidth = Integer(t,"hiddenWidth", 128);
#define READ_T(field) training.field = Number(t, #field)
    READ_T(ilsCaptureScale);
    auto const& curriculum=t.at("curriculum");
    Keys(curriculum, {"stage","locReminderProbability","glideOffsetMeters","gateHoldSeconds"});
    Require(curriculum.at("stage").is_string(),"curriculum stage");
    auto const stage=curriculum.at("stage").get<std::string>();
    Require(stage=="loc" || stage=="glide" || stage=="joint","curriculum stage");
    training.curriculumStage=stage=="loc" ? CurriculumStage::Loc
        : stage=="glide" ? CurriculumStage::Glide : CurriculumStage::Joint;
    training.locReminderProbability=Number(curriculum,"locReminderProbability");
    training.glideOffsetMeters=Number(curriculum,"glideOffsetMeters");
    training.gateHoldSeconds=Number(curriculum,"gateHoldSeconds");
    READ_T(actorRate); READ_T(criticRate); READ_T(gamma); READ_T(lambda); READ_T(clip); READ_T(entropy);
    READ_T(valueWeight); READ_T(minLogStd); READ_T(maxLogStd); READ_T(initialLogStd); READ_T(maxGradientNorm);
    READ_T(adamBeta1); READ_T(adamBeta2); READ_T(adamEpsilon);
#undef READ_T
    training.resume = Boolean(t,"resume");
    training.checkpointDirectory = t.at("checkpointDirectory").get<std::string>();
    auto const& e = root.at("evaluation");
    Keys(e,{"seed","episodes","everyUpdates","minimumSuccessRate"});
    config.evaluation.seed=static_cast<std::uint32_t>(Integer(e,"seed",UINT32_MAX));
    config.evaluation.episodes=Integer(e,"episodes",1000000);
    config.evaluation.everyUpdates=Integer(e,"everyUpdates",100000000);
    config.evaluation.minimumSuccessRate=Number(e,"minimumSuccessRate");
    auto semantics = root;
    semantics.erase("training");
    semantics.erase("evaluation");
    semantics["distribution"] = {training.minLogStd,training.maxLogStd};
    semantics["rewardVersion"] = 11;
    semantics["guidanceVersion"] = 13;
    semantics["spawnVersion"] = 2;
    config.checkpointSignature = semantics.dump();
    Validate(config);
    return config;
}

bool CompatibleTrainingSignature(std::string const& stored,std::string const& current)
{
    if(stored==current) return true;
    auto previous=Json::parse(stored,nullptr,false),next=Json::parse(current,nullptr,false);
    if(!previous.is_object() || !next.is_object() || !previous.contains("initial") || !next.contains("initial")
        || !previous["initial"].is_object() || !next["initial"].is_object()) return false;
    for(auto const* signature : {&previous,&next})
    {
        if(!signature->contains("reward") || !(*signature)["reward"].is_object()
            || !(*signature)["initial"].contains("approach") || !(*signature)["initial"]["approach"].is_boolean()) return false;
        if(signature->contains("rewardVersion") && (*signature)["rewardVersion"]!=2
            && (*signature)["rewardVersion"]!=3 && (*signature)["rewardVersion"]!=4
            && (*signature)["rewardVersion"]!=5 && (*signature)["rewardVersion"]!=6
            && (*signature)["rewardVersion"]!=7 && (*signature)["rewardVersion"]!=8
            && (*signature)["rewardVersion"]!=9 && (*signature)["rewardVersion"]!=10
            && (*signature)["rewardVersion"]!=11) return false;
        if(signature->contains("guidanceVersion") && (*signature)["guidanceVersion"]!=2
            && (*signature)["guidanceVersion"]!=3 && (*signature)["guidanceVersion"]!=4
            && (*signature)["guidanceVersion"]!=5 && (*signature)["guidanceVersion"]!=6
            && (*signature)["guidanceVersion"]!=7 && (*signature)["guidanceVersion"]!=8
            && (*signature)["guidanceVersion"]!=9 && (*signature)["guidanceVersion"]!=10
            && (*signature)["guidanceVersion"]!=11 && (*signature)["guidanceVersion"]!=12
            && (*signature)["guidanceVersion"]!=13) return false;
        if(signature->contains("spawnVersion") && (*signature)["spawnVersion"]!=2) return false;
    }
    for(char const* name : {"distanceMeters","bearingDegrees","headingOffsetDegrees"})
    {
        if(!previous["initial"].contains(name) || !next["initial"].contains(name)) return false;
        previous["initial"].erase(name); next["initial"].erase(name);
    }
    // These changes keep the network dimensions and may warm-start training.
    // Old scores are never reused: production compares the complete signature.
    previous["initial"].erase("runwayAlignedProbability"); next["initial"].erase("runwayAlignedProbability");
    previous["initial"].erase("approach"); next["initial"].erase("approach");
    if(!previous.contains("normalization") || !next.contains("normalization")
        || !previous["normalization"].is_object() || !next["normalization"].is_object()) return false;
    // V13 uses sin(localizer) and removes the obsolete configurable scale.
    previous["normalization"].erase("localizerDegrees");
    next["normalization"].erase("localizerDegrees");
    if(!previous.contains("mission") || !next.contains("mission")
        || !previous["mission"].is_object() || !next["mission"].is_object()) return false;
    previous["mission"].erase("patrolLookaheadSeconds"); next["mission"].erase("patrolLookaheadSeconds");
    previous["mission"].erase("patrolHeadingToleranceDegrees"); next["mission"].erase("patrolHeadingToleranceDegrees");
    previous["mission"].erase("waypointRadius"); next["mission"].erase("waypointRadius");
    previous["mission"].erase("turnSamples"); next["mission"].erase("turnSamples");
    previous["mission"].erase("straightSamples"); next["mission"].erase("straightSamples");
    previous["mission"].erase("approachHeadingToleranceDegrees"); next["mission"].erase("approachHeadingToleranceDegrees");
    previous["mission"].erase("racetrackCenterCross"); next["mission"].erase("racetrackCenterCross");
    // Episode-distance changes preserve the network contract and can warm-start
    // training, but the complete signature still forces BEST revalidation.
    previous["mission"].erase("maxDistanceMeters"); next["mission"].erase("maxDistanceMeters");
    previous.erase("reward"); next.erase("reward");
    previous.erase("rewardVersion"); next.erase("rewardVersion");
    previous.erase("guidanceVersion"); next.erase("guidanceVersion");
    previous.erase("spawnVersion"); next.erase("spawnVersion");
    return previous==next;
}

void Validate(Config const& c)
{
    auto const& i=c.initial;
    Require(std::isfinite(i.runwayAlignedProbability) && i.runwayAlignedProbability>=0 && i.runwayAlignedProbability<=1,
        "runwayAlignedProbability must be finite and within [0,1]");
    auto const& d=c.runwayRandomization;
    std::array<Range,18> const ranges{i.distanceMeters,i.bearingDegrees,i.heightMeters,i.speedMps,i.headingOffsetDegrees,
        i.rollDegrees,i.pitchDegrees,i.slopeDegrees,d.northMeters,d.eastMeters,d.headingOffsetDegrees,d.deckHeightMeters,
        d.speedMps,d.heaveAmplitudeMeters,d.wavePeriodSeconds,d.pitchAmplitudeDegrees,d.rollAmplitudeDegrees,d.phaseDegrees};
    for(auto range:ranges) Require(std::isfinite(range.minimum) && std::isfinite(range.maximum) && range.minimum<=range.maximum,"ordered finite randomization ranges");
    Require(c.aircraft == "c310", "this adapter supports the pinned c310 model");
    auto const& r = c.runway;
    Require(std::abs(r.latitudeDegrees) < 85 && std::abs(r.longitudeDegrees) <= 180, "runway geodetic coordinates");
    Require(r.headingDegrees >= 0 && r.headingDegrees < 360 && r.lengthMeters > 0 && r.widthMeters > 0
        && r.widthMeters < r.lengthMeters && r.thresholdHeightMeters >= 0, "runway geometry");
    auto const& m = c.mission;
    Require(m.physicsDt >= 0.001 && m.physicsDt <= 0.02 && m.actionSubsteps > 0 && c.StepSeconds() <= 0.5, "integration timestep");
    Require(m.maxSeconds > c.StepSeconds() && m.maxSeconds <= 86400 && m.maxDistanceMeters > r.lengthMeters
        && m.maxAltitudeMeters > m.patrolAltitude, "episode limits");
    Require(m.switchDistanceMeters > 0 && m.ilsRangeMeters >= m.switchDistanceMeters && m.ilsRangeMeters < m.maxDistanceMeters
        && m.ilsSectorDegrees > 0 && m.ilsSectorDegrees < 90 && m.glideDegrees > 0 && m.glideDegrees < 10, "ILS envelope");
    Require(m.localizerCaptureDegrees > 0 && m.localizerCaptureDegrees < m.ilsSectorDegrees
        && m.glideCaptureDegrees > 0 && m.glideCaptureDegrees < m.glideDegrees, "ILS capture tolerances");
    Require(m.captureHoldSeconds>=m.physicsDt && m.captureHysteresis>=1 && m.pathCorridorMeters>0
        && m.maximumWaypointStepMeters>0 && m.patrolHeadingToleranceDegrees>0 && m.patrolHeadingToleranceDegrees<90
        && m.approachHeadingToleranceDegrees>0 && m.approachHeadingToleranceDegrees<90
        && m.approachCrossToleranceMeters>0 && m.approachAltitudeToleranceMeters>0,"navigation and capture confirmation");
    Require(m.patrolSpeed > 0 && m.approachSpeed > m.flareSpeed && m.flareSpeed > m.stoppedSpeed
        && m.flareHeightMeters > r.thresholdHeightMeters && m.stoppedSpeed >= 0, "target speeds and flare");
    Require(m.maxTouchdownSink > 0 && m.maxTouchdownSpeed >= m.flareSpeed && m.maxTouchdownRollDegrees > 0
        && m.minTouchdownPitchDegrees < m.maxTouchdownPitchDegrees && m.maxTouchdownPitchDegrees < 90
        && m.maxTouchdownHeadingDegrees>0 && m.maxTouchdownHeadingDegrees<90
        && m.gearDownMinimum > 0 && m.gearDownMinimum <= 1 && m.contactHeightMeters > 0 && m.stoppedHoldSeconds > 0, "landing limits");
    Require(m.racetrackHalfStraight > 0 && m.racetrackRadius > 0 && m.patrolAltitude > m.flareHeightMeters
        && m.waypointRadius > 0 && m.patrolLookaheadSeconds > 0 && m.patrolLookaheadSeconds <= 60
        && m.turnSamples >= 4 && m.straightSamples >= 1
        && 2*(m.turnSamples + m.straightSamples) <= MaximumWaypoints, "racetrack geometry/capacity");
    Require(std::abs(m.racetrackCenterCross-m.racetrackRadius)<=1e-9,
        "inbound racetrack straight must align with the localizer");
    double const minimumSegment=std::min(2*m.racetrackHalfStraight/static_cast<double>(m.straightSamples),
        2*m.racetrackRadius*std::sin(Pi/(2*static_cast<double>(m.turnSamples))));
    Require(2*m.waypointRadius<minimumSegment && m.maximumWaypointStepMeters<minimumSegment,"non-overlapping sequential waypoint gates");
    Require(std::hypot(m.racetrackCenterAlong, m.racetrackCenterCross) + m.racetrackHalfStraight + m.racetrackRadius
        < m.maxDistanceMeters, "racetrack outside episode bounds");
    Require(i.distanceMeters.minimum>r.lengthMeters && i.distanceMeters.maximum<=m.ilsRangeMeters
        && i.heightMeters.minimum>m.flareHeightMeters && i.heightMeters.maximum<m.maxAltitudeMeters
        && i.speedMps.minimum>m.flareSpeed && i.speedMps.maximum<=c.normalization.speed,
        "bounded airborne spawn distance/height/speed");
    Require(i.bearingDegrees.minimum>0 && i.bearingDegrees.maximum<90
        && i.headingOffsetDegrees.minimum>=-180 && i.headingOffsetDegrees.maximum<=180
        && i.rollDegrees.minimum>=-30 && i.rollDegrees.maximum<=30 && i.pitchDegrees.minimum>=-15 && i.pitchDegrees.maximum<=15
        && i.slopeDegrees.minimum>=-10 && i.slopeDegrees.maximum<=10,"reasonable spawn attitude/course");
    double const maximumHeadingJitter=std::max(std::abs(i.headingOffsetDegrees.minimum),std::abs(i.headingOffsetDegrees.maximum));
    Require(maximumHeadingJitter<i.bearingDegrees.minimum
        && i.bearingDegrees.maximum+maximumHeadingJitter<=m.approachHeadingToleranceDegrees,
        "left/right starts must point inward within the LOC capture heading envelope");
    Require(i.throttle >= 0 && i.throttle <= 1 && i.flap >= 0 && i.flap <= 1 && i.approachFlap>=0 && i.approachFlap<=1
        && i.rolloutFlap>=0 && i.rolloutFlap<=1 && i.propeller >= 0 && i.propeller <= 1, "initial and phase controls");
    Require(d.heaveAmplitudeMeters.minimum>=0 && d.heaveAmplitudeMeters.maximum<=5 && d.wavePeriodSeconds.minimum>=3
        && d.wavePeriodSeconds.maximum<=60 && d.pitchAmplitudeDegrees.minimum>=0 && d.pitchAmplitudeDegrees.maximum<=2
        && d.rollAmplitudeDegrees.minimum>=0 && d.rollAmplitudeDegrees.maximum<=2
        && d.speedMps.minimum>=0 && d.speedMps.maximum<=15,"reasonable runway motion/houle");
    Require(d.headingOffsetDegrees.minimum>=-180 && d.headingOffsetDegrees.maximum<=180
        && std::max(std::abs(d.northMeters.minimum),std::abs(d.northMeters.maximum))<=m.maxDistanceMeters
        && std::max(std::abs(d.eastMeters.minimum),std::abs(d.eastMeters.maximum))<=m.maxDistanceMeters,
        "bounded runway spawn position/course");
    double const maximumTilt=std::sin(d.pitchAmplitudeDegrees.maximum*DegreesToRadians)+std::sin(d.rollAmplitudeDegrees.maximum*DegreesToRadians);
    double const minimumGlideHeight=r.thresholdHeightMeters+i.distanceMeters.minimum*std::tan(m.glideDegrees*DegreesToRadians);
    double const maximumGlideHeight=r.thresholdHeightMeters+i.distanceMeters.maximum*std::tan(m.glideDegrees*DegreesToRadians);
    Require(minimumGlideHeight>m.flareHeightMeters && maximumGlideHeight<m.maxAltitudeMeters,
        "LOC/glide starts must remain inside the airborne altitude envelope");
    double const minimumInitialHeight=std::min(i.heightMeters.minimum,minimumGlideHeight);
    Require(minimumInitialHeight-i.distanceMeters.maximum*maximumTilt+d.deckHeightMeters.minimum-d.heaveAmplitudeMeters.maximum>d.seaHeightMeters,
        "all randomized aircraft starts must remain above the sea");
    Require(d.deckHeightMeters.minimum-d.heaveAmplitudeMeters.maximum-r.lengthMeters*std::sin(d.pitchAmplitudeDegrees.maximum*DegreesToRadians)/2
        > d.seaHeightMeters,"deck must remain above sea");
    auto const& n = c.normalization;
    Require(n.distance > 0 && n.altitude > 0 && n.speed > 0 && n.rate > 0 && n.path > 0
        && n.glideDegrees > 0, "normalization scales");
    auto const& w = c.reward;
    Require(w.pathTolerance > 0 && w.localizerToleranceDegrees > 0 && w.glideToleranceDegrees > 0
        && w.speedTolerance > 0 && w.slopeToleranceDegrees > 0, "reward tolerances");
    Require(std::isfinite(w.localizerCourseBlend) && w.localizerCourseBlend >= 0
        && w.localizerCourseBlend <= 1, "localizer course blend within [0,1]");
    Require(std::isfinite(w.localizerInterceptGain) && w.localizerInterceptGain >= 1
        && w.localizerInterceptGain <= 4, "localizer intercept gain within [1,4]");
    Require(std::isfinite(w.localizerInterceptLimitDegrees)
        && w.localizerInterceptLimitDegrees > i.bearingDegrees.maximum+maximumHeadingJitter
        && w.localizerInterceptLimitDegrees < 90, "localizer intercept limit must cover initial bearings");
    for (double weight : w.weights) Require(std::isfinite(weight), "finite reward weights");
    Require(w.weights[static_cast<std::size_t>(Reward::Crash)] <= 0, "crash weight must be non-positive");
    auto const& t = c.training;
    Require(t.rolloutSteps >= 2 && t.minibatch > 0 && t.minibatch <= t.rolloutSteps && t.rolloutSteps % t.minibatch == 0
        && t.epochs > 0 && t.maxUpdates > 0 && t.checkpointEvery > 0, "training capacities");
    Require(t.hiddenWidth == 32 || t.hiddenWidth == 64 || t.hiddenWidth == 128, "hiddenWidth must be 32, 64 or 128");
    Require(t.actorRate > 0 && t.criticRate > 0 && t.gamma > 0 && t.gamma <= 1 && t.lambda >= 0 && t.lambda <= 1
        && t.clip > 0 && t.clip < 1 && t.entropy >= 0 && t.valueWeight > 0 && t.maxGradientNorm > 0, "PPO hyperparameters");
    Require(t.minLogStd >= -10 && t.maxLogStd <= 2 && t.minLogStd < t.initialLogStd && t.initialLogStd < t.maxLogStd,
        "Gaussian log standard deviations");
    Require(t.adamBeta1 >= 0 && t.adamBeta1 < 1 && t.adamBeta2 >= 0 && t.adamBeta2 < 1 && t.adamEpsilon > 0,
        "Adam hyperparameters");
    Require(std::isfinite(t.locReminderProbability) && t.locReminderProbability>=0
        && t.locReminderProbability<=1 && std::isfinite(t.glideOffsetMeters)
        && t.glideOffsetMeters>0 && t.glideOffsetMeters<minimumGlideHeight-m.flareHeightMeters
        && std::isfinite(t.gateHoldSeconds) && t.gateHoldSeconds>=m.physicsDt
        && (t.curriculumStage==CurriculumStage::Joint || t.gateHoldSeconds<=m.maxSeconds),
        "curriculum settings");
    Require(t.ilsCaptureScale>=1.0
        && m.localizerCaptureDegrees*t.ilsCaptureScale<m.ilsSectorDegrees
        && m.glideCaptureDegrees*t.ilsCaptureScale<m.glideDegrees,
        "training ILS capture scale");
    Require(c.evaluation.episodes>0 && c.evaluation.seed!=t.seed && c.evaluation.everyUpdates>0
        && c.evaluation.minimumSuccessRate>=0 && c.evaluation.minimumSuccessRate<=1,
        "independent evaluation seed, budget and success threshold");
    auto const directory = std::filesystem::path(t.checkpointDirectory);
    Require(!directory.empty() && !directory.is_absolute(), "checkpointDirectory must be relative");
    for (auto const& component : directory) Require(component != "..", "checkpointDirectory traversal");
}
} // namespace ppo::examples::flight
