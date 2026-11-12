#include "test_support.hpp"
#include "flight_session.hpp"
#include <fstream>
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
void ChangedStarts(Config c)
{
    c.training.rolloutSteps=8;c.training.minibatch=4;c.training.epochs=1;c.training.maxUpdates=1;
    c.training.resume=true;c.training.checkpointEvery=1;c.mission.maxSeconds=1;c.evaluation.episodes=2;
    TemporaryDirectory directory("flight-changed-starts-");
    auto old=c;
    auto signature=nlohmann::json::parse(c.checkpointSignature);
    signature["initial"]["distanceMeters"]={7000,14000};signature["initial"]["bearingDegrees"]={-180,180};
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
        && ReadFile(pointer)==previous,"new starts resume training while preserving the old BEST without reusing its score");
    Check(resumed.Learner().Collected()==0,"new episode starts with an empty collector");
    FlightLearner comparison(c);comparison.Load(oldStore.Latest());
    Check(comparison.Evaluate({})==oldPolicy.Evaluate({}),"changed spawn ranges preserve learned policy outputs");
    Rejects([&] {FlightSession inference(c,false,directory.path);},"inference rejects BEST validated on different starts");
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
        && !CompatibleTrainingSignature("broken",c.checkpointSignature),"only horizontal spawn changes are accepted");
}
int main()
{
    try
    {
        auto c=LoadConfig(PPO_FLIGHT_CONFIG);
        DesktopContinuation(c);ChangedStarts(c);
        c.training.rolloutSteps=16;c.training.minibatch=8;c.training.epochs=1;c.training.maxUpdates=1;
        c.mission.maxSeconds=2;c.evaluation.episodes=2;c.evaluation.everyUpdates=1;
        TemporaryDirectory directory("flight-session-");
        FlightModelStore store(c,directory.path);auto const checkpoint=store.Latest();
        FlightSession training(c,true,directory.path);
        auto const display=training.Snapshot();
        Check(display.localizerCaptureDegrees==c.mission.localizerCaptureDegrees
            && display.glideCaptureDegrees==c.mission.glideCaptureDegrees,"render snapshot carries configured ILS envelope angles");
        for(std::size_t i=0;i<16;++i) training.Tick();
        Check(!training.Finished() && training.Snapshot().validating && training.Snapshot().learning.updates==1,"final update requires completed validation");
        auto const original=ReadFile(checkpoint);
        Check(!original.empty(),"completed training checkpoint exists");
        for(std::size_t i=0;i<1000 && !training.Finished();++i) training.Tick();
        Check(training.Finished() && training.Snapshot().validationCampaigns==1 && training.Snapshot().lastValidation.episodes==2,"complete independent candidate validation");
        Check(!store.BestMetrics() && !training.Snapshot().hasBest,"failed candidate is never promoted as BEST");
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
