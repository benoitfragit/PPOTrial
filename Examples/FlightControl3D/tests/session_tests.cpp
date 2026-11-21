/**
 * @file session_tests.cpp
 * @brief Regressions of FlightSession: resumption, contracts, validation, BEST, reward
 *        tuning, crash snapshot and curriculum.
 *
 * Protects: desktop continuation without losing a rollout, warm starts across reward
 * changes, the frozen learner during validation, BEST promotion and its immutability, the
 * reward-tuning round trip through the scenario file, and the curriculum persistence.
 *
 * Read next: Runtime/flight_session.cpp, Training/flight_model_store.cpp.
 */
#include "test_support.hpp"
#include "flight_session.hpp"
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
using namespace ppo::examples::flight;

namespace {
std::string ReadFile(std::filesystem::path const& path)
{
    std::ifstream stream(path,std::ios::binary);
    return {std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
}
/** @brief A tiny, fast training contract on the production scenario. */
Config SmallConfig()
{
    auto c=LoadConfig(PPO_FLIGHT_CONFIG);
    c.training.rolloutSteps=8;c.training.minibatch=4;c.training.epochs=1;c.training.maxUpdates=1;
    c.training.hiddenWidth=32;c.training.resume=true;c.training.checkpointEvery=1;
    c.training.randomizeEnvironmentSeed=false;c.training.curriculum.rolloutStartShare=0;
    c.mission.maxSeconds=1;c.evaluation.episodes=2;c.evaluation.everyUpdates=1;
    return c;
}
ValidationMetrics Successful(Config const& c,std::uint64_t update)
{
    // Controlled manifest fixture, not a claim about a trained landing policy.
    ValidationMetrics metrics{};
    metrics.update=update;metrics.episodes=c.evaluation.episodes;
    metrics.touchdowns=metrics.stops=metrics.episodes;
    metrics.touchdownSumMeters=100;metrics.stopSumMeters=600;
    return metrics;
}
void DesktopContinuation()
{
    auto const c=SmallConfig();
    TemporaryDirectory directory("flight-desktop-resume-");
    FlightModelStore store(c,directory.path);
    {
        FlightSession session(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
        Check(!session.Snapshot().resumed && session.Snapshot().learning.updates==0,"a missing latest creates a first policy");
        for(int i=0;i<3;++i) session.Tick();
        session.ContinueTraining();
        Check(session.Learner().Collected()==3,"Play during a partial rollout preserves the collector");
        for(int i=0;i<1000 && !session.Finished();++i) session.Tick();
        Check(session.Finished() && session.Snapshot().learning.updates==1,"a desktop block completes its validation");
        auto const before=ReadFile(store.Latest());
        session.ContinueTraining();
        auto const comparison=directory.path/"continued.txt";session.Learner().Save(comparison);
        Check(!session.Finished() && session.Snapshot().trainingUpdateLimit==2 && ReadFile(comparison)==before,
            "a second Play extends the budget without resetting weights, Adam or RNG");
        for(int i=0;i<1000 && !session.Finished();++i) session.Tick();
        Check(session.Finished() && session.Snapshot().learning.updates==2,"the second block advances the same model");
    }
    FlightSession relaunched(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
    Check(relaunched.Snapshot().resumed && relaunched.Snapshot().learning.updates==2
        && relaunched.Snapshot().trainingUpdateLimit==3 && !relaunched.Snapshot().validating,
        "a relaunch resumes latest with a fresh additional update budget");
    auto const comparison=directory.path/"relaunched.txt";relaunched.Learner().Save(comparison);
    Check(SameLearnerCheckpoint(comparison,store.Latest()),"a relaunch preserves weights, Adam and the future RNG sequence");
    {std::ofstream broken(store.Latest());broken<<"broken";}
    Rejects([&] {FlightSession invalid(c,true,directory.path);},"a corrupt latest must not silently create a new policy");
}
/** @brief A warm-startable contract change resumes the policy but never reuses the old BEST score. */
void ChangedTrainingContract()
{
    auto const c=SmallConfig();
    TemporaryDirectory directory("flight-changed-contract-");
    auto old=c;
    auto signature=nlohmann::json::parse(c.checkpointSignature);
    signature["reward"]["weights"]["Path"]=1.0;signature["initial"]["distanceMeters"]={8000,9000};
    old.checkpointSignature=signature.dump();
    FlightModelStore oldStore(old,directory.path);
    FlightLearner oldPolicy(old);oldPolicy.Save(oldStore.Latest());
    Check(oldStore.Promote(oldPolicy,Successful(old,0)),"controlled old-contract BEST fixture");
    auto const pointer=directory.path/"best/current.json";
    auto const previous=ReadFile(pointer);
    auto const generation=directory.path/"best"/nlohmann::json::parse(previous).at("checkpoint").get<std::string>();
    FlightSession resumed(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
    Check(resumed.Snapshot().resumed && !resumed.Snapshot().hasBest && ReadFile(pointer)==previous,
        "a changed training contract resumes while preserving the old BEST without reusing its score");
    FlightLearner comparison(c);comparison.Load(oldStore.Latest());
    Check(comparison.Evaluate({})==oldPolicy.Evaluate({}),"a warm start preserves the learned policy outputs");
    Rejects([&] {FlightSession inference(c,false,directory.path);},"inference rejects a BEST validated on another contract");
    for(int i=0;i<1000 && !resumed.Finished();++i) resumed.Tick();
    Check(resumed.Finished() && ReadFile(pointer)==previous && std::filesystem::exists(generation),
        "a failed revalidation keeps the previous BEST generation and pointer intact");
    FlightModelStore newStore(c,directory.path);
    Check(newStore.Promote(resumed.Learner(),Successful(c,resumed.Snapshot().learning.updates))
        && newStore.BestMetrics().has_value() && std::filesystem::exists(generation),
        "a newly validated candidate preserves the old generation");
    auto incompatible=c;signature=nlohmann::json::parse(c.checkpointSignature);signature["mission"]["glideDegrees"]=4;
    incompatible.checkpointSignature=signature.dump();
    Rejects([&] {FlightLearner rejected(incompatible);rejected.Load(oldStore.Latest());},"guidance changes remain incompatible");
}
/** @brief Training validates, promotes only eligible candidates and evaluation never writes. */
void TrainingValidationAndBest()
{
    auto c=SmallConfig();
    c.training.rolloutSteps=16;c.training.minibatch=8;c.mission.maxSeconds=2;
    TemporaryDirectory directory("flight-session-");
    FlightModelStore store(c,directory.path);auto const checkpoint=store.Latest();
    FlightSession training(c,true,directory.path);
    auto const display=training.Snapshot();
    Check(display.mission.phase==Phase::Approach && display.guidance.targetValid
        && display.gearDownMinimum==c.mission.gearDownMinimum,"the training snapshot carries the approach and gear threshold");
    for(std::size_t i=0;i<16;++i) training.Tick();
    Check(!training.Finished() && training.Snapshot().validating && training.Snapshot().learning.updates==1,
        "the final update requires a completed validation");
    auto const original=ReadFile(checkpoint);
    Check(!original.empty(),"the completed training checkpoint exists");
    for(std::size_t i=0;i<1000 && !training.Finished();++i) training.Tick();
    auto const finished=training.Snapshot();
    Check(training.Finished() && finished.validationCampaigns==1 && finished.lastValidation.episodes==2
        && finished.gateEvaluated==2 && finished.gateSuccesses==finished.lastValidation.stops,
        "a complete independent candidate validation is reported");
    Check(finished.curriculumLevel==0 && finished.lastValidation.difficulty==0,
        "the validation campaign evaluates the current curriculum level");
    Check(!store.BestMetrics() && !finished.hasBest,"a failed candidate is never promoted as BEST");
    training.Learner().Save(directory.path/"after-validation.txt");
    Check(ReadFile(directory.path/"after-validation.txt")==original,"validation leaves weights, Adam and RNG frozen");
    Rejects([&] { FlightSession noBest(c,false,directory.path); },"production must not fall back to unvalidated training");
    auto const successful=Successful(c,1);
    Check(store.Promote(training.Learner(),successful),"first complete eligible BEST");
    auto const pointer=directory.path/"best/current.json";
    auto const initialBest=ReadFile(pointer);
    auto partial=successful;partial.episodes=1;
    Check(!store.Promote(training.Learner(),partial) && ReadFile(pointer)==initialBest,"a partial validation cannot replace BEST");
    auto worse=successful;worse.stopSumMeters=800;
    Check(!store.Promote(training.Learner(),worse) && !store.Promote(training.Learner(),successful),
        "a worse or tied candidate retains the incumbent");
    FlightSession evaluation(c,false,directory.path),sameEvaluation(c,false,directory.path);
    Check(evaluation.Snapshot().aircraft.position.north==sameEvaluation.Snapshot().aircraft.position.north,
        "independent evaluation seeds are repeatable");
    for(std::size_t i=0;i<1000 && !evaluation.Finished();++i) evaluation.Tick();
    Check(evaluation.Finished() && evaluation.Snapshot().episodes==2 && evaluation.Snapshot().learning.updates==1,
        "bounded inference runs without PPO updates");
    Check(ReadFile(checkpoint)==original && ReadFile(pointer)==initialBest,"evaluation never writes latest or BEST");
    auto improved=successful;improved.stopSumMeters=500;
    auto const pending=directory.path/"best/candidate.pending";
    { std::ofstream stale(pending);stale<<"interrupted promotion"; }
    Check(store.Promote(training.Learner(),improved) && store.BestMetrics()->MeanStop()==250,
        "equal success prefers a shorter safe stop");
    auto pendingBackup=pending;pendingBackup+=".previous";
    Check(!std::filesystem::exists(pending) && !std::filesystem::exists(pendingBackup),
        "a promotion leaves no pending candidate or backup behind");
    // A harder curriculum campaign supersedes an easier one, whatever their stop rates.
    auto easier=improved;easier.difficulty=0.4;
    auto harder=improved;harder.difficulty=0.6;harder.stopSumMeters=1200;
    Check(Better(harder,easier) && !Better(easier,harder),"a harder validated campaign always ranks higher");
    Check(!store.Promote(training.Learner(),easier) && store.BestMetrics()->difficulty==1,
        "an easier campaign never replaces a full-task BEST");
    auto const newBest=ReadFile(pointer);
    Check(nlohmann::json::parse(newBest).at("difficulty").get<double>()==1,"the BEST manifest records its difficulty");
    FlightSession resumed(c,true,directory.path);
    Check(resumed.Snapshot().hasBest && resumed.Snapshot().bestDifficulty==1 && resumed.Snapshot().bestUpdate==1,
        "the snapshot reports the difficulty and update of the BEST campaign");
    Check(resumed.Snapshot().validating,"a resume with an exhausted budget finishes the pending validation");
    for(std::size_t i=0;i<1000 && !resumed.Finished();++i) resumed.Tick();
    Check(resumed.Finished() && ReadFile(pointer)==newBest,"a failed resumed validation preserves BEST");
    auto const manifest=nlohmann::json::parse(newBest);
    auto const bestFile=directory.path/"best"/manifest.at("checkpoint").get<std::string>();
    { std::ofstream stream(bestFile,std::ios::app);stream<<"corruption"; }
    Rejects([&] { store.BestMetrics(); },"a BEST checksum mismatch fails closed");
    Check(ReadFile(pointer)==newBest,"a failed BEST read never rewrites metadata");
}
/** @brief Every reward control round-trips independently through JSON. */
void RewardControlsRoundTrip()
{
    TemporaryDirectory directory("flight-reward-controls-");
    auto const scenario=directory.path/"scenario.json";
    std::filesystem::copy_file(PPO_FLIGHT_CONFIG,scenario);
    auto config=LoadConfig(scenario);
    for(std::size_t index=0;index<RewardCount;++index)
    {
        double const value=static_cast<Reward>(index)==Reward::Crash ? -0.1 : 0.1;
        auto const next=WithRewardTuning(config,index,value);
        SaveRewardTuning(scenario,index,value);
        auto const loaded=LoadConfig(scenario);
        Check(loaded.checkpointSignature==next.checkpointSignature && Near(RewardTuningValue(loaded.reward,index),value),
            "each reward persists independently");
        for(std::size_t other=0;other<RewardCount;++other)
            if(other!=index) Check(Near(loaded.reward.weights[other],config.reward.weights[other]),
                "other weights remain unchanged");
        config=loaded;
    }
}
/** @brief Reward edits preserve the policy, persist settings and wait for a PPO boundary. */
void RewardTuning()
{
    TemporaryDirectory directory("flight-reward-tuning-");
    auto const scenario=directory.path/"scenario.json";
    std::filesystem::copy_file(PPO_FLIGHT_CONFIG,scenario);
    auto config=LoadConfig(scenario);
    config.training.rolloutSteps=4;config.training.minibatch=4;config.training.epochs=1;config.training.hiddenWidth=32;
    config.training.maxUpdates=3;config.training.resume=false;config.evaluation.everyUpdates=100;
    FlightSession session(config,true,directory.path/"model",std::nullopt,TrainingBudget::TotalUpdates,scenario);
    auto const direction=static_cast<std::size_t>(Reward::Direction);
    auto const path=static_cast<std::size_t>(Reward::Path);
    auto const before=directory.path/"before.txt",after=directory.path/"after.txt";
    session.Learner().Save(before);
    Check(session.SetRewardTuning(direction,0.1),"the direction weight can be edited at a rollout boundary");
    session.Learner().Save(after);
    std::ifstream oldFile(before),newFile(after);
    std::string oldHeader,newHeader,oldSignature,newSignature;
    std::getline(oldFile,oldHeader);std::getline(newFile,newHeader);
    std::getline(oldFile,oldSignature);std::getline(newFile,newSignature);
    std::string const oldData((std::istreambuf_iterator<char>(oldFile)),{});
    std::string const newData((std::istreambuf_iterator<char>(newFile)),{});
    Check(oldHeader==newHeader && oldData==newData && oldSignature!=newSignature
        && Near(LoadConfig(scenario).reward.weights[direction],0.1),"editing a weight preserves the network and optimizer");
    session.Tick();
    Check(session.SetRewardTuning(path,12) && session.Snapshot().rewardTuningPending
        && Near(LoadConfig(scenario).reward.weights[path],12),"mid-rollout edits are queued and saved");
    for(int step=0;step<3;++step) session.Tick();
    auto const state=session.Snapshot();
    Check(!state.rewardTuningPending && Near(state.rewardSettings.weights[path],12)
        && state.learning.hasRewardMeans && state.learning.updates==1,"queued edits activate after the rollout boundary");
}
/** @brief Crash poses and details remain available to the renderer across episode resets. */
void CrashRenderingSnapshot()
{
    auto c=SmallConfig();
    // Impossible attitude limits turn the first rolling start into a deterministic crash.
    c.mission.minTouchdownPitchDegrees=5;c.mission.maxTouchdownPitchDegrees=10;
    c.mission.maxSeconds=0.2;c.training.rolloutSteps=64;c.training.minibatch=8;
    c.training.curriculum.rolloutStartShare=0.9;
    TemporaryDirectory directory("flight-crash-render-snapshot-");
    FlightSession session(c,true,directory.path);
    Check(session.Snapshot().crashEvent==0,"a new session has no crash event");
    for(int tick=0;tick<50 && session.Snapshot().crashEvent==0;++tick) session.Tick();
    auto const first=session.Snapshot();
    auto const& pitch=first.crashDiagnostic.conditions[static_cast<std::size_t>(CrashReason::PitchLow)];
    Check(first.crashEvent==1 && IsFinite(first.crashAircraft) && first.hasCrashDiagnostic
        && pitch.failed && Near(pitch.limit,c.mission.minTouchdownPitchDegrees),
        "the terminal crash publishes the physical aircraft pose and the failed limit");
    session.ResetEpisode();
    auto const reset=session.Snapshot();
    Check(reset.hasCrashDiagnostic && reset.crashDiagnostic.conditions[static_cast<std::size_t>(CrashReason::PitchLow)].failed,
        "the last crash details survive the next episode reset");
}
/** @brief The curriculum promotes on a full successful window and never past the full task. */
void CurriculumProgression()
{
    CurriculumSettings settings{};
    settings.firstDistanceMeters={1500,3000};settings.levels=3;settings.windowEpisodes=2;
    settings.promotionSuccessRate=0.5;settings.rolloutStartShare=0.1;
    FlightCurriculum curriculum(settings);
    Check(curriculum.Level()==0 && curriculum.Difficulty()==0,"training starts at the easiest level");
    Check(!curriculum.Record(false) && !curriculum.Record(false) && curriculum.Level()==0 && curriculum.WindowEpisodes()==0,
        "a failed window keeps the level and starts a new window");
    Check(!curriculum.Record(true) && curriculum.Record(false) && curriculum.Level()==1 && Near(curriculum.Difficulty(),0.5),
        "a window at the promotion rate advances one level");
    Check(!curriculum.Record(true) && curriculum.Record(true) && curriculum.Level()==2 && curriculum.Difficulty()==1,
        "the last level is the full task");
    for(int episode=0;episode<5;++episode) Check(!curriculum.Record(true),"the full task is never exceeded");
    curriculum.Restore(10);
    Check(curriculum.Level()==2 && curriculum.WindowEpisodes()==0,"a restored level is clamped to the configured levels");
    settings.levels=1;
    Check(FlightCurriculum(settings).Difficulty()==1,"a single-level curriculum trains on the full task");

    TemporaryDirectory directory("flight-curriculum-level-");
    auto const path=directory.path/"latest/curriculum_level.txt";
    Check(!LoadCurriculumLevel(path),"a missing level file has no saved level");
    SaveCurriculumLevel(path,3);SaveCurriculumLevel(path,4);
    Check(LoadCurriculumLevel(path)==std::optional<std::size_t>{4},"the saved level round-trips and is replaced");
    {std::ofstream broken(path);broken<<"4 extra";}
    Rejects([&] { LoadCurriculumLevel(path); },"a malformed level file fails closed");
}

/** @brief The level is saved with latest and restored only when training resumes it. */
void CurriculumPersistence()
{
    auto c=SmallConfig();
    TemporaryDirectory directory("flight-curriculum-session-");
    FlightModelStore store(c,directory.path);
    {
        FlightSession session(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
        Check(session.Snapshot().curriculumLevel==0 && session.Snapshot().curriculumLevels==CurriculumLevels(c.training.curriculum),
            "a new policy starts at the first curriculum level");
        for(int i=0;i<1000 && !session.Finished();++i) session.Tick();
        Check(LoadCurriculumLevel(store.CurriculumLevel()).has_value(),"the level is saved together with latest");
    }
    SaveCurriculumLevel(store.CurriculumLevel(),2);
    FlightSession resumed(c,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
    Check(resumed.Snapshot().resumed && resumed.Snapshot().curriculumLevel==2,"a resumed session restores its level");
    auto fresh=c;fresh.training.resume=false;
    FlightSession restarted(fresh,true,directory.path,std::nullopt,TrainingBudget::AdditionalUpdates);
    Check(restarted.Snapshot().curriculumLevel==0,"a new policy does not inherit a saved level");
}
} // namespace

int main()
{
    try
    {
        DesktopContinuation();
        ChangedTrainingContract();
        TrainingValidationAndBest();
        RewardControlsRoundTrip();
        RewardTuning();
        CrashRenderingSnapshot();
        CurriculumProgression();
        CurriculumPersistence();
    }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n';return 1; }
    std::cout<<"Shared desktop/headless session regressions passed\n";
}
