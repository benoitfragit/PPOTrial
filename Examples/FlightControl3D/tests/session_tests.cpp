#include "test_support.hpp"
#include "flight_session.hpp"
#include <fstream>
#include <iomanip>
#include <iterator>
#include <nlohmann/json.hpp>
using namespace ppo::examples::flight;

std::string ReadFile(std::filesystem::path const& path)
{
    std::ifstream stream(path,std::ios::binary);
    return {std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
}
void DesktopContinuation(Config c)
{
    c.training.rolloutSteps=8;c.training.minibatch=4;c.training.epochs=1;c.training.maxUpdates=1;
    c.training.resume=true;c.training.checkpointEvery=1;c.mission.maxSeconds=1;c.evaluation.episodes=2;
    TemporaryDirectory directory("flight-desktop-resume-");
    FlightModelStore store(c,directory.path);
    {
        FlightSession session(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
        Check(!session.Snapshot().resumed && session.Snapshot().learning.updates==0,"missing latest creates a first policy");
        for(int i=0;i<3;++i) session.Tick();
        session.ContinueTraining();
        Check(session.Learner().Collected()==3,"Play during a partial rollout preserves the collector");
        for(int i=0;i<1000 && !session.Finished();++i) session.Tick();
        Check(session.Finished() && session.Snapshot().learning.updates==1,"desktop block completes validation");
        auto const before=ReadFile(store.Latest());
        session.ContinueTraining();
        auto const comparison=directory.path/"continued.txt";session.Learner().Save(comparison);
        Check(!session.Finished() && session.Snapshot().trainingUpdateLimit==2 && ReadFile(comparison)==before,
            "second Play extends the budget without resetting weights, Adam or RNG");
        for(int i=0;i<1000 && !session.Finished();++i) session.Tick();
        Check(session.Finished() && session.Snapshot().learning.updates==2,"second block advances the same model");
    }
    FlightSession relaunched(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
    Check(relaunched.Snapshot().resumed && relaunched.Snapshot().learning.updates==2
        && relaunched.Snapshot().trainingUpdateLimit==3 && !relaunched.Snapshot().validating,
        "relaunch resumes latest with a fresh additional update budget");
    auto const comparison=directory.path/"relaunched.txt";relaunched.Learner().Save(comparison);
    Check(ReadFile(comparison)==ReadFile(store.Latest()),"relaunch restores the complete training checkpoint exactly");
    {std::ofstream broken(store.Latest());broken<<"broken";}
    Rejects([&] {FlightSession invalid(c,true,directory.path);},"corrupt latest must not silently create a new policy");
}
enum class TrainingContractChange
{
    HorizontalStarts, Rewards, LegacyStarts, LegacyGuidance, AlignedProbability,
    EpisodeBoundary, ApproachOnly
};
void CommittedTrainingCheckpoint(Config c)
{
    std::filesystem::path const checkpoint{PPO_FLIGHT_COMMITTED_CHECKPOINT};
    if(!std::filesystem::exists(checkpoint)) return;
    // Legacy snapshots contain the old tanh mapping and must not be replayed as clamped policies.
    std::ifstream stream(checkpoint);
    std::string magic,signature;
    std::size_t observations{},actions{},width{};
    std::uint64_t updates{};
    Check(static_cast<bool>(stream>>magic>>observations>>actions>>width>>updates>>std::quoted(signature)),
        "read committed checkpoint signature");
    if(!CompatibleTrainingSignature(signature,c.checkpointSignature))
    {
        Check(!nlohmann::json::parse(signature).contains("actionMappingVersion"),
            "only the historical tanh checkpoint may be incompatible in this fixture");
        Rejects([&] { FlightLearner rejected(c);rejected.Load(checkpoint); },
            "old tanh checkpoint cannot warm-start clamped controls");
        return;
    }
    c.training.resume=true;
    auto const modelDirectory=checkpoint.parent_path().parent_path();
    FlightSession resumed(c,true,modelDirectory,std::nullopt,TrainingBudget::AdditionalUpdates);
    auto const snapshot=resumed.Snapshot();
    Check(snapshot.resumed && snapshot.learning.updates>0
        && snapshot.trainingUpdateLimit==snapshot.learning.updates+c.training.maxUpdates,
        "compatible FlightControl3D checkpoint is a complete additional-training warm start");
    Check(resumed.Learner().Collected()==0,
        "committed FlightControl3D checkpoint starts at an update boundary");
}
void ChangedTrainingContract(Config c,TrainingContractChange change)
{
    c.training.rolloutSteps=8;c.training.minibatch=4;c.training.epochs=1;c.training.maxUpdates=1;
    c.training.resume=true;c.training.checkpointEvery=1;c.mission.maxSeconds=1;c.evaluation.episodes=2;
    TemporaryDirectory directory("flight-changed-starts-");
    auto old=c;
    auto signature=nlohmann::json::parse(c.checkpointSignature);
    if(change==TrainingContractChange::Rewards)
    {
        signature.erase("rewardVersion");signature["reward"].erase("convergenceSeconds");
        signature["reward"].erase("ilsDecaySeconds");
        signature["reward"]["weights"]["Path"]=0.02;signature["reward"]["weights"]["Speed"]=0.02;
        signature["reward"]["weights"]["Slope"]=0.02;signature["initial"]["approach"]=false;
    }
    else if(change==TrainingContractChange::HorizontalStarts)
    {signature["initial"]["distanceMeters"]={7000,14000};signature["initial"]["bearingDegrees"]={-180,180};}
    else if(change==TrainingContractChange::LegacyStarts) signature["initial"].erase("runwayAlignedProbability");
    else if(change==TrainingContractChange::LegacyGuidance)
    {
        signature.erase("guidanceVersion");
        signature["mission"].erase("patrolLookaheadSeconds");
        signature["mission"]["approachHeadingToleranceDegrees"]=60;
        signature["mission"]["racetrackCenterCross"]=0;
    }
    else if(change==TrainingContractChange::AlignedProbability)
        signature["initial"]["runwayAlignedProbability"]=0.25;
    else if(change==TrainingContractChange::ApproachOnly)
    {
        signature["guidanceVersion"]=13;
        signature["mission"].erase("patrolEnabled");
        signature["initial"]["distanceMeters"]={5000,14000};
        signature["initial"]["bearingDegrees"]={12,30};
        signature["initial"]["heightMeters"]={400,1000};
        signature["initial"]["headingOffsetDegrees"]={-3,3};
    }
    else
        signature["mission"]["maxDistanceMeters"]=30000.0;
    old.checkpointSignature=signature.dump();
    FlightModelStore oldStore(old,directory.path);
    FlightLearner oldPolicy(old);oldPolicy.Save(oldStore.Latest());
    ValidationMetrics successful{};successful.episodes=2;successful.touchdowns=2;successful.stops=2;
    successful.touchdownSumMeters=100;successful.stopSumMeters=600;
    Check(oldStore.Promote(oldPolicy,successful),"controlled old-distribution BEST fixture");
    auto const pointer=directory.path/"best/current.json";
    auto const previous=ReadFile(pointer);
    auto const generation=directory.path/"best"/nlohmann::json::parse(previous).at("checkpoint").get<std::string>();
    FlightSession resumed(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
    Check(resumed.Snapshot().resumed && !resumed.Snapshot().hasBest
        && ReadFile(pointer)==previous,"changed training contract resumes while preserving the old BEST without reusing its score");
    Check(resumed.Learner().Collected()==0,"new episode starts with an empty collector");
    FlightLearner comparison(c);comparison.Load(oldStore.Latest());
    Check(comparison.Evaluate({})==oldPolicy.Evaluate({}),"warm start preserves learned policy outputs");
    Rejects([&] {FlightSession inference(c,false,directory.path);},"inference rejects BEST validated on a different contract");
    for(int i=0;i<1000 && !resumed.Finished();++i) resumed.Tick();
    Check(resumed.Finished() && ReadFile(pointer)==previous && std::filesystem::exists(generation),
        "failed revalidation keeps the previous BEST generation and pointer intact");
    FlightModelStore newStore(c,directory.path);successful.update=resumed.Snapshot().learning.updates;
    Check(newStore.Promote(resumed.Learner(),successful) && newStore.BestMetrics().has_value()
        && std::filesystem::exists(generation),"a newly validated candidate preserves the old generation");
    auto incompatible=c;signature=nlohmann::json::parse(c.checkpointSignature);signature["mission"]["glideDegrees"]=4;
    incompatible.checkpointSignature=signature.dump();
    Rejects([&] {FlightLearner rejected(incompatible);rejected.Load(oldStore.Latest());},"guidance changes remain incompatible");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["initial"]["speedMps"]={60,80};
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature)
        && !CompatibleTrainingSignature("broken",c.checkpointSignature),"other observation/physics changes remain incompatible");
    signature=nlohmann::json::parse(c.checkpointSignature);signature.erase("actionMappingVersion");
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "legacy tanh checkpoint cannot warm-start clamped controls");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["actionMappingVersion"]=1;
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "different action mapping versions are incompatible");
    signature=nlohmann::json::parse(c.checkpointSignature);signature.erase("reward");
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),"missing reward contract is not a warm start");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["rewardVersion"]=2;
    signature["reward"].erase("ilsDecaySeconds");
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V2 remains usable as a training warm start");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["rewardVersion"]=3;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V3 remains usable as a training warm start");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["rewardVersion"]=4;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V4 remains usable as a training warm start");
    signature["rewardVersion"]=5;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V5 remains usable as a training warm start");
    signature["rewardVersion"]=6;
    signature["reward"].erase("localizerConvergenceScale");
    signature["reward"].erase("localizerSustainScale");
    signature["reward"].erase("localizerCourseBlend");
    signature["reward"].erase("localizerInterceptGain");
    signature["reward"].erase("localizerInterceptLimitDegrees");
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V6 remains usable as a training warm start");
    signature["rewardVersion"]=7;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V7 remains usable as a training warm start");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["rewardVersion"]=8;
    signature["reward"].erase("localizerCourseBlend");
    signature["reward"].erase("localizerInterceptGain");
    signature["reward"].erase("localizerInterceptLimitDegrees");
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V8 remains usable as a training warm start");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["rewardVersion"]=9;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "previous reward V9 remains usable as a training warm start");
    signature["rewardVersion"]=10;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "current reward V10 remains compatible");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["rewardVersion"]=999;
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),"unknown shaping contract is rejected");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["guidanceVersion"]=2;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V2 remains usable as a training warm start");
    signature["guidanceVersion"]=3;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V3 remains usable as a training warm start");
    signature["guidanceVersion"]=4;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V4 remains usable as a training warm start");
    signature["guidanceVersion"]=5;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V5 remains usable as a training warm start");
    signature["guidanceVersion"]=6;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V6 remains usable as a training warm start");
    signature["guidanceVersion"]=7;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V7 remains usable as a training warm start");
    signature["guidanceVersion"]=8;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V8 remains usable as a training warm start");
    signature["guidanceVersion"]=9;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V9 remains usable as a training warm start");
    signature["guidanceVersion"]=10;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V10 remains usable as a training warm start");
    signature["guidanceVersion"]=11;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V11 remains usable as a training warm start");
    signature["guidanceVersion"]=12;
    signature["normalization"]["localizerDegrees"]=35.0;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V12 with the retired LOC scale remains usable as a training warm start");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["guidanceVersion"]=13;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "guidance V13 remains a compatible warm start");
    signature["guidanceVersion"]=14;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "approach-only guidance V14 remains compatible");
    signature["mission"]["patrolEnabled"]=true;
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "explicitly re-enabling patrol keeps a compatible training architecture");
    signature["mission"]["patrolEnabled"]=1;
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "non-boolean patrol setting is rejected from warm-start contracts");
    signature=nlohmann::json::parse(c.checkpointSignature);
    signature["guidanceVersion"]=999;
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),"unknown guidance contract is rejected");
    signature=nlohmann::json::parse(c.checkpointSignature);signature.erase("spawnVersion");
    Check(CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),
        "legacy random placement remains usable as a training warm start");
    signature=nlohmann::json::parse(c.checkpointSignature);signature["spawnVersion"]=999;
    Check(!CompatibleTrainingSignature(signature.dump(),c.checkpointSignature),"unknown spawn contract is rejected");
}
int main()
{
    try
    {
        auto c=LoadConfig(PPO_FLIGHT_CONFIG);
        CommittedTrainingCheckpoint(c);
        DesktopContinuation(c);
        for(auto change : {TrainingContractChange::HorizontalStarts,TrainingContractChange::Rewards,
            TrainingContractChange::LegacyStarts,TrainingContractChange::LegacyGuidance,
            TrainingContractChange::AlignedProbability,TrainingContractChange::EpisodeBoundary,
            TrainingContractChange::ApproachOnly})
            ChangedTrainingContract(c,change);
        c.training.rolloutSteps=16;c.training.minibatch=8;c.training.epochs=1;c.training.maxUpdates=1;
        c.mission.maxSeconds=2;c.evaluation.episodes=2;c.evaluation.everyUpdates=1;
        TemporaryDirectory directory("flight-session-");
        FlightModelStore store(c,directory.path);auto const checkpoint=store.Latest();
        FlightSession training(c,true,directory.path);
        auto const display=training.Snapshot();
        Check(display.plan.count==0 && display.mission.phase==Phase::Approach && !display.guidance.targetValid,
            "desktop and validation snapshots carry no disabled racetrack or guidance point");
        Check(display.localizerCaptureDegrees==c.mission.localizerCaptureDegrees*c.training.ilsCaptureScale
            && display.glideCaptureDegrees==c.mission.glideCaptureDegrees*c.training.ilsCaptureScale
            && display.gearDownMinimum==c.mission.gearDownMinimum,
            "training snapshot carries widened ILS angles and the physical gear threshold");
        for(std::size_t i=0;i<16;++i) training.Tick();
        Check(!training.Finished() && training.Snapshot().validating && training.Snapshot().learning.updates==1,"final update requires completed validation");
        Check(training.Snapshot().localizerCaptureDegrees==c.mission.localizerCaptureDegrees
            && training.Snapshot().glideCaptureDegrees==c.mission.glideCaptureDegrees,
            "validation restores the nominal ILS capture tolerances");
        auto const original=ReadFile(checkpoint);
        Check(!original.empty(),"completed training checkpoint exists");
        for(std::size_t i=0;i<1000 && !training.Finished();++i) training.Tick();
        Check(training.Finished() && training.Snapshot().validationCampaigns==1 && training.Snapshot().lastValidation.episodes==2,"complete independent candidate validation");
        Check(!store.BestMetrics() && !training.Snapshot().hasBest,"failed candidate is never promoted as BEST");
        MissionState capture{};capture.phase=Phase::Approach;
        capture.capturedLocalizer=true;capture.capturedGlide=true;
        AircraftState aligned{};
        aligned.airspeed=c.mission.approachSpeed;
        aligned.heading=c.runway.headingDegrees*DegreesToRadians;
        Guidance guidance{};
        guidance.targetSpeed=c.mission.approachSpeed;
        guidance.targetSlope=-c.mission.glideDegrees*DegreesToRadians;
        guidance.relativeVelocity={c.mission.approachSpeed,0,
            -c.mission.approachSpeed*std::tan(c.mission.glideDegrees*DegreesToRadians)};
        Check(CaptureGateQualified(c,CurriculumStage::Loc,capture,aligned,guidance)
            && CaptureGateQualified(c,CurriculumStage::Glide,capture,aligned,guidance),
            "ideal stabilized LOC and Glide qualify for their gates");
        auto rotated=c;rotated.runway.headingDegrees+=25;
        auto rotatedAircraft=aligned;
        rotatedAircraft.heading=rotated.runway.headingDegrees*DegreesToRadians;
        Check(CaptureGateQualified(rotated,CurriculumStage::Loc,capture,rotatedAircraft,guidance)
            && !CaptureGateQualified(c,CurriculumStage::Loc,capture,rotatedAircraft,guidance),
            "gate heading uses the actual randomized runway rather than default JSON heading");
        aligned.heading+=20*DegreesToRadians;
        Check(!CaptureGateQualified(c,CurriculumStage::Loc,capture,aligned,guidance),
            "radio capture with wrong runway heading cannot clear LOC gate");
        aligned.heading=c.runway.headingDegrees*DegreesToRadians;
        guidance.relativeVelocity.east=20;
        Check(!CaptureGateQualified(c,CurriculumStage::Loc,capture,aligned,guidance),
            "radio capture with lateral ground track cannot clear LOC gate");
        guidance.relativeVelocity.east=0;guidance.relativeVelocity.up=5;
        Check(!CaptureGateQualified(c,CurriculumStage::Glide,capture,aligned,guidance)
            && CaptureGateQualified(c,CurriculumStage::Loc,capture,aligned,guidance),
            "climbing on Glide is a regression even with correct LOC");
        guidance.relativeVelocity.up=-c.mission.approachSpeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
        capture.capturedGlide=false;
        Check(!CaptureGateQualified(c,CurriculumStage::Glide,capture,aligned,guidance),
            "unconfirmed glide cannot pass Glide gate");
        // A capture-only curriculum gate must never mint a landing BEST.
        auto gateConfig=c;gateConfig.training.curriculumStage=CurriculumStage::Loc;
        gateConfig.training.gateHoldSeconds=0.1;gateConfig.training.resume=false;
        TemporaryDirectory gateDirectory("flight-session-gate-");
        FlightSession gateTraining(gateConfig,true,gateDirectory.path);
        for(std::size_t step=0;step<600 && !gateTraining.Finished();++step) gateTraining.Tick();
        Check(gateTraining.Finished() && gateTraining.Snapshot().gateEvaluated==gateConfig.evaluation.episodes
            && !std::filesystem::exists(gateDirectory.path/"best/current.json"),
            "independent LOC gate evaluation cannot promote an unvalidated landing BEST");
        training.Learner().Save(directory.path/"after-validation.txt");
        Check(ReadFile(directory.path/"after-validation.txt")==original,"candidate validation leaves weights, Adam and RNG frozen");
        Rejects([&] { FlightSession noBest(c,false,directory.path); },"production must not fall back to unvalidated training");
        // Controlled successful metrics exercise promotion; no learned success is claimed.
        ValidationMetrics successful{};successful.update=1;successful.episodes=2;successful.touchdowns=2;successful.stops=2;
        successful.touchdownSumMeters=100;successful.stopSumMeters=600;
        Check(store.Promote(training.Learner(),successful),"first complete eligible BEST");
        auto const pointer=directory.path/"best/current.json";
        auto const initialBest=ReadFile(pointer);
        auto partial=successful;partial.episodes=1;
        Check(!store.Promote(training.Learner(),partial) && ReadFile(pointer)==initialBest,"partial validation cannot replace BEST");
        auto worse=successful;worse.stopSumMeters=800;
        Check(!store.Promote(training.Learner(),worse) && !store.Promote(training.Learner(),successful),"worse/tied candidate retains incumbent");
        FlightSession evaluation(c,false,directory.path),sameEvaluation(c,false,directory.path);
        Check(evaluation.Snapshot().aircraft.position.north==sameEvaluation.Snapshot().aircraft.position.north,"independent evaluation seed repeatability");
        Check(evaluation.Snapshot().aircraft.position.north!=training.Snapshot().aircraft.position.north,"evaluation uses independent starts");
        for(std::size_t i=0;i<1000 && !evaluation.Finished();++i) evaluation.Tick();
        Check(evaluation.Finished() && evaluation.Snapshot().episodes==2 && evaluation.Snapshot().learning.updates==1,"bounded inference without PPO updates");
        Check(ReadFile(checkpoint)==original && ReadFile(pointer)==initialBest,"evaluation never writes latest or BEST");
        auto improved=successful;improved.stopSumMeters=500;
        Check(store.Promote(training.Learner(),improved) && store.BestMetrics()->MeanStop()==250,"equal success prefers shorter safe stop");
        auto const newBest=ReadFile(pointer);
        c.training.resume=true;
        FlightSession resumed(c,true,directory.path);
        Check(resumed.Snapshot().validating,"resume with exhausted training budget finishes pending validation");
        for(std::size_t i=0;i<1000 && !resumed.Finished();++i) resumed.Tick();
        Check(resumed.Finished() && ReadFile(pointer)==newBest,"failed resumed validation preserves BEST");
        auto const manifest=nlohmann::json::parse(newBest);
        auto const bestFile=directory.path/"best"/manifest.at("checkpoint").get<std::string>();
        { std::ofstream stream(bestFile,std::ios::app);stream<<"corruption"; }
        Rejects([&] { store.BestMetrics(); },"BEST checksum mismatch must fail closed");
        Check(ReadFile(pointer)==newBest,"failed BEST read never rewrites metadata");
    }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n';return 1; }
    std::cout<<"Shared desktop/headless session regressions passed\n";
}
