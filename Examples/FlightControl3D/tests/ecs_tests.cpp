#include "test_support.hpp"
#include "flight_example.hpp"
#include "flight_session.hpp"
#include "environment.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>

using namespace ppo::examples::flight;

namespace {
FlightSnapshot WaitSnapshot(ppo::model::Environment const& environment,std::uint64_t updates,bool finished)
{
    for(int attempt=0;attempt<5000;++attempt)
    {
        std::optional<FlightSnapshot> state;
        environment.ReadSnapshot([&state](entt::registry const& registry)
        {
            if(registry.ctx().contains<FlightSnapshot>()) state=registry.ctx().get<FlightSnapshot>();
        });
        if(state && state->learning.updates==updates && state->finished==finished) return *state;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("FlightControl3D: timed out waiting for ECS transport snapshot");
}
}
int main()
{
    try
    {
        TemporaryDirectory directory("flight-ecs-");
        auto const configPath=directory.path/"settings/Scenarios/FlightControl3D.json";
        std::filesystem::create_directories(configPath.parent_path());
        std::ifstream source(PPO_FLIGHT_CONFIG);nlohmann::json json;source>>json;
        json["training"]["rolloutSteps"]=8;json["training"]["minibatch"]=4;json["training"]["epochs"]=1;
        json["training"]["maxUpdates"]=1;json["training"]["checkpointEvery"]=1;json["training"]["resume"]=true;
        json["evaluation"]["episodes"]=2;json["evaluation"]["everyUpdates"]=1;json["mission"]["maxSeconds"]=0.3;
        {std::ofstream output(configPath);output<<json;}
        auto const example=Definition();
        {
            ppo::model::Environment environment;
            Check(example.loadScenario(configPath,environment),"native Flight example loads");
            environment.Start();
            auto const preview=WaitSnapshot(environment,0,false);
            Check(preview.mission.steps==0 && !preview.resumed,"paused preview does not train or advance physics");
            environment.Play();WaitSnapshot(environment,1,true);
            environment.Stop();environment.Play();
            auto const continued=WaitSnapshot(environment,2,true);
            Check(continued.trainingUpdateLimit==2,"Stop/Play continues a completed block through real ECS callbacks");
            environment.Stop();
        }
        {
            ppo::model::Environment relaunched;
            Check(example.loadScenario(configPath,relaunched),"same example relaunches");
            relaunched.Start();
            auto const resumed=WaitSnapshot(relaunched,2,false);
            Check(resumed.resumed && resumed.trainingUpdateLimit==3,"fresh desktop environment resumes latest with another block");
            relaunched.Stop();
        }
    }
    catch(std::exception const& error) {std::cerr<<error.what()<<'\n';return 1;}
    std::cout<<"Flight ECS preview, Stop/Play and relaunch passed\n";
}
