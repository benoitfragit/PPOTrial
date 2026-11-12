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
int main()
{
    try
    {
        auto c=LoadConfig(PPO_FLIGHT_CONFIG);
        c.training.rolloutSteps=16;c.training.minibatch=8;c.training.epochs=1;c.training.maxUpdates=1;
        c.mission.maxSeconds=2;c.evaluation.episodes=2;c.evaluation.everyUpdates=1;
        TemporaryDirectory directory("flight-session-");
        FlightModelStore store(c,directory.path);auto const checkpoint=store.Latest();
        FlightSession training(c,true,directory.path);
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
