/**
 * @file ecs_tests.cpp
 * @brief Integration checks of the desktop ECS bridge (desktop builds only).
 *
 * Protects: the step pacing of the generic Environment with the flight session, and the
 * reward-scaling command round trip from the desktop to the scenario file.
 *
 * Read next: Runtime/flight_example.cpp.
 */
#include "test_support.hpp"
#include "flight_example.hpp"
#include "flight_session.hpp"
#include "environment.hpp"
#include "../../../Framework/Simulation/Env/src/step_pacing.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>

using namespace ppo::examples::flight;

namespace {
/** @brief Desktop iterations retain the historical throttle and max speed has none. */
void StepPacing()
{
    using namespace std::chrono_literals;
    using ppo::model::detail::StepDelay;
    Check(StepDelay(false)==2ms,"desktop iterations use the historical short throttle");
    Check(StepDelay(true)==0ms,"max-speed mode remains unthrottled");
}
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
void CheckLearningDiagnostics(ppo::model::Environment const& environment)
{
    bool found=false;
    environment.ReadSnapshot([&found](entt::registry const& registry)
    {
        auto const stats=registry.ctx().get<FlightSnapshot>().learning;
        registry.view<ppo::model::sAgentStateSnaphot>().each([&found,&stats](auto const& agent)
        {
            Check(agent.m_common.m_actorloss==static_cast<float>(stats.policyLoss)
                && agent.m_common.m_valueloss==static_cast<float>(stats.valueLoss)
                && agent.m_common.m_loss==static_cast<float>(stats.policyLoss+stats.valueLoss),
                "shared UI receives actual PPO losses and their unweighted total");
            found=true;
        });
    });
    Check(found,"Flight learning diagnostics are published to the shared UI");
}
/** @brief UI reward commands reach the T-6 session and persist without a training step. */
void RewardCommand()
{
    TemporaryDirectory directory("flight-ecs-reward-");
    auto const path=directory.path/"settings/Scenarios/FlightControl3D.json";
    std::filesystem::create_directories(path.parent_path());
    std::ifstream source(PPO_FLIGHT_CONFIG);nlohmann::json json;source>>json;
    json["training"]["resume"]=false;json["runtime"]["maxSpeed"]=false;
    {std::ofstream output(path);output<<json;}
    ppo::model::Environment environment;
    Check(Definition().loadScenario(path,environment),"Joint reward-control scenario loads");
    environment.Start();WaitSnapshot(environment,0,false);
    unsigned long long id{};
    environment.ReadSnapshot([&id](entt::registry const& registry)
    {
        registry.view<ppo::model::sAgentStateSnaphot>().each([&id](auto const& agent)
        {
            id=agent.m_id;
            Check(agent.m_scalings.size()==RewardCount
                && agent.m_rewardnames[0]=="Path" && agent.m_rewardnames[1]=="Direction",
                "the shared UI exposes exactly the reward channels");
        });
    });
    Check(id!=0,"Joint agent exposes a reward command identifier");
    ppo::model::sAgentRealTimeCommand command{};
    command.m_id=id;command.m_type=ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardScaling;
    command.m_rewardScalingType=1;command.m_rewardScaling=12;
    environment.AddAgentCommand(command);
    bool applied=false;
    for(int attempt=0;attempt<5000 && !applied;++attempt)
    {
        environment.ReadSnapshot([&applied](entt::registry const& registry)
        {
            if(registry.ctx().contains<FlightSnapshot>())
            {
                auto const& state=registry.ctx().get<FlightSnapshot>();
                auto const direction=static_cast<std::size_t>(Reward::Direction);
                applied=Near(state.activeRewardSettings.weights[direction],12)
                    && state.rewardTuningRevisions[direction]==1 && !state.rewardTuningPending;
            }
        });
        if(!applied) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(applied && Near(LoadConfig(path).reward.weights[static_cast<std::size_t>(Reward::Direction)],12),
        "the visible direction control updates its correct reward channel");
    command.m_rewardScalingType=0;command.m_rewardScaling=0.125F;
    environment.AddAgentCommand(command);
    environment.Stop();
    Check(Near(LoadConfig(path).reward.weights[static_cast<std::size_t>(Reward::Path)],0.125,1e-12),
        "closing the worker flushes the threshold-distance weight");
}
}
int main()
{
    try
    {
        StepPacing();
        TemporaryDirectory directory("flight-ecs-");
        auto const configPath=directory.path/"settings/Scenarios/FlightControl3D.json";
        std::filesystem::create_directories(configPath.parent_path());
        std::ifstream source(PPO_FLIGHT_CONFIG);nlohmann::json json;source>>json;
        json["training"]["rolloutSteps"]=8;json["training"]["minibatch"]=4;json["training"]["epochs"]=1;
        json["training"]["maxUpdates"]=1;json["training"]["checkpointEvery"]=1;json["training"]["resume"]=true;
        json["training"]["hiddenWidth"]=32;json["training"]["randomizeEnvironmentSeed"]=false;json["evaluation"]["episodes"]=2;json["evaluation"]["everyUpdates"]=1;json["mission"]["maxSeconds"]=0.3;
        {std::ofstream output(configPath);output<<json;}
        auto const example=Definition();
        RewardCommand();
        {
            ppo::model::Environment environment;
            Check(example.loadScenario(configPath,environment),"native Flight example loads");
            environment.Start();
            auto const preview=WaitSnapshot(environment,0,false);
            Check(preview.mission.steps==0 && !preview.resumed,"paused preview does not train or advance physics");
            environment.Play();WaitSnapshot(environment,1,true);
            CheckLearningDiagnostics(environment);
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
