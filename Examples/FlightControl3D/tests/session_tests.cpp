#include "test_support.hpp"
#include "flight_session.hpp"
#include <fstream>
#include <iterator>
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
        c.mission.maxSeconds=2;c.evaluation.episodes=2;
        TemporaryDirectory directory("flight-session-");auto const checkpoint=directory.path/"flight.txt";
        FlightSession training(c,true,checkpoint);
        for(std::size_t i=0;i<16;++i) training.Tick();
        Check(training.Finished() && training.Snapshot().learning.updates==1,"bounded complete training session");
        auto const original=ReadFile(checkpoint);
        Check(!original.empty(),"completed training checkpoint exists");
        FlightSession evaluation(c,false,checkpoint),sameEvaluation(c,false,checkpoint);
        Check(evaluation.Snapshot().aircraft.position.north==sameEvaluation.Snapshot().aircraft.position.north,"independent evaluation seed repeatability");
        Check(evaluation.Snapshot().aircraft.position.north!=training.Snapshot().aircraft.position.north,"evaluation uses independent starts");
        for(std::size_t i=0;i<1000 && !evaluation.Finished();++i) evaluation.Tick();
        Check(evaluation.Finished() && evaluation.Snapshot().episodes==2 && evaluation.Snapshot().learning.updates==1,"bounded inference without PPO updates");
        Check(ReadFile(checkpoint)==original && !std::filesystem::exists(checkpoint.string()+".previous"),"evaluation never writes checkpoint");
        c.training.resume=true;
        FlightSession resumed(c,true,checkpoint);
        Check(resumed.Finished(),"resume respects existing update budget");
    }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n';return 1; }
    std::cout<<"Shared desktop/headless session regressions passed\n";
}
