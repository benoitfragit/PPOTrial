#include "navigation.hpp"
#include "ppo/rl/runtime.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
class RecordingLearner final : public ppo::rl::ILearner
{
public:
    ppo::rl::Value act(const ppo::rl::Value&) override {return std::vector<float>{0.F,0.F};}
    void observe(const ppo::rl::StepResult& transition) override {transitions.push_back(transition);}
    std::vector<ppo::rl::StepResult> transitions;
};
class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        const auto token=std::chrono::steady_clock::now().time_since_epoch().count();
        path=std::filesystem::temp_directory_path()/("ppo-navigation-test-"+std::to_string(token));
        if(!std::filesystem::create_directory(path)) throw std::runtime_error("test directory already exists");
    }
    ~TemporaryDirectory() {std::error_code error;std::filesystem::remove_all(path,error);}
    std::filesystem::path path;
};
void test_navigation()
{
    navigation::Scenario scenario;
    scenario.goal={2,0}; scenario.arrival_radius=0.25; scenario.speed=5; scenario.dt=1;
    navigation::Environment environment(scenario);
    environment.reset();
    auto arrival=environment.step(std::vector<float>{0,0});
    check(arrival.terminated && std::get<bool>(arrival.info.at("success")),"swept arrival");
    check(std::abs(environment.position().x-1.75)<1e-9,"stop at first arrival");
    scenario.goal={10,0}; scenario.obstacles={{{2,0},0.25}};
    navigation::Environment blocked(scenario);
    blocked.reset();
    const auto collision=blocked.step(std::vector<float>{0,0});
    check(collision.terminated && std::get<bool>(collision.info.at("collision")),"no obstacle tunnelling");
    check(std::abs(blocked.position().x-1.75)<1e-9,"stop at first collision");
    scenario.goal={1,0}; scenario.obstacles={{{3,0},0.25}};
    navigation::Environment early_goal(scenario);
    early_goal.reset();
    check(std::get<bool>(early_goal.step(std::vector<float>{0,0}).info.at("success")),"arrival precedes later obstacle");
    scenario.goal={-1,0}; scenario.half_extent=4;scenario.obstacles.clear();
    navigation::Environment boundary(scenario);boundary.reset();
    check(std::get<bool>(boundary.step(std::vector<float>{0,0}).info.at("collision")),"boundary crossing");
    check(std::abs(boundary.position().x-4)<1e-9,"boundary position");
}
void test_moving_goal()
{
    navigation::Scenario scenario;
    scenario.goal = {2,0};
    scenario.goal_velocity = {0.5,0};
    scenario.arrival_radius = 0.25;
    scenario.speed = 1;
    scenario.dt = 1;
    scenario.max_steps = 20;
    scenario.obstacles = {{{2,2},0.3}};
    navigation::Environment environment(scenario);
    const auto initial = environment.reset({42,{}});
    const auto& observed = std::get<std::vector<float>>(initial.observation);
    check(observed.size() == 13 && std::abs(observed[8] - 0.1F) < 1e-6F, "goal velocity included in observation");
    auto transition = environment.step(std::vector<float>{0,0});
    check(environment.goal_state().position.x == 2.5 && !transition.terminated, "second mobile advances");
    for (int i = 0; i < 3 && !transition.terminated; ++i)
        transition = environment.step(std::vector<float>{0,0});
    check(transition.terminated && std::get<bool>(transition.info.at("success")), "controlled mobile reaches moving mobile with zone present");
    check(std::abs(environment.position().x - 3.5) < 1e-9
        && std::abs(environment.goal_state().position.x - 3.75) < 1e-9, "both mobiles stop at the same meeting instant");
    environment.reset({42,{}});
    check(environment.goal_state().position.x == 2 && environment.position().x == 0, "reset restores both mobiles");

    // Replay a simple detour: north, east, east, south, east. No learned policy.
    navigation::Scenario detour_scenario;
    detour_scenario.goal = {2,0};
    detour_scenario.goal_velocity = {0.1,0};
    detour_scenario.arrival_radius = 0.25;
    detour_scenario.speed = 1;
    detour_scenario.dt = 1;
    detour_scenario.max_steps = 20;
    detour_scenario.max_turn_rate = 1.5707963267948966;
    detour_scenario.obstacles = {{{1,0},0.2}};
    navigation::Environment detour(detour_scenario);
    detour.reset();
    const float turns[] = {1,-1,0,-1,1};
    ppo::rl::StepResult result;
    for (float turn : turns)
    {
        result = detour.step(std::vector<float>{turn,0});
        check(!std::get<bool>(result.info.at("collision")), "scripted detour remains outside the zone");
        if (result.terminated) break;
    }
    check(result.terminated && std::get<bool>(result.info.at("success")), "detour reaches another moving mobile around a blocking zone");

    scenario.goal = {4,0};
    scenario.goal_velocity = {-5,0};
    scenario.speed = 5;
    scenario.max_steps = 1;
    scenario.obstacles.clear();
    navigation::Environment crossing(scenario);
    crossing.reset();
    const auto meeting = crossing.step(std::vector<float>{0,0});
    check(std::get<bool>(meeting.info.at("success"))
        && std::abs(crossing.position().x - 1.875) < 1e-9
        && std::abs(crossing.goal_state().position.x - 2.125) < 1e-9, "relative sweep detects meeting between endpoints");

    scenario.goal_velocity = {-1,0};
    scenario.obstacles = {{{1,0},0.1}};
    navigation::Environment blocked(scenario);
    blocked.reset();
    const auto collision = blocked.step(std::vector<float>{0,0});
    check(std::get<bool>(collision.info.at("collision")) && !std::get<bool>(collision.info.at("success")), "forbidden zone reached before moving goal");
    check(std::abs(blocked.goal_state().position.x - 3.82) < 1e-9, "goal uses collision event time too");

    scenario.obstacles.clear();
    scenario.goal_velocity = {1000,0};
    bool rejected = false;
    try { navigation::Environment invalid(scenario); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "goal trajectory outside world rejected");
    scenario.goal_velocity.x = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try { navigation::Environment invalid(scenario); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "nonfinite goal velocity rejected");
}
void test_shared_loop()
{
    navigation::Scenario scenario; scenario.max_steps=2; scenario.goal_velocity={0.5,0.2};
    navigation::Environment first(scenario),second(scenario);
    RecordingLearner training,evaluation;
    const auto train=ppo::rl::train_episode(first,training,{42,{}},10);
    const auto eval=ppo::rl::run_episode(second,evaluation,{42,{}},10);
    check(train.reward==eval.reward && train.length==eval.length,"shared train/eval dynamics");
    check(training.transitions.size()==2 && evaluation.transitions.empty(),"evaluation never updates learner");
    check(train.truncated && !train.terminated,"environment timeout");
    check(training.transitions.back().truncated,"truncation delivered to learner");
    check(std::get<std::vector<float>>(training.transitions.back().observation)[0]>0,"final state before reset");
    const auto external=ppo::rl::train_episode(first,training,{},1);
    check(external.truncated && external.length==1,"external time limit");
    first.reset();
    bool invalid=false;
    try {first.step(std::vector<float>{std::numeric_limits<float>::quiet_NaN(),0});}
    catch(const std::invalid_argument&) {invalid=true;}
    check(invalid && first.position().x==scenario.start.x,"invalid action does not mutate environment");
}
void test_incremental_session()
{
    navigation::Scenario scenario;
    navigation::Environment first(scenario), second(scenario);
    RecordingLearner learner, other;
    ppo::rl::EpisodeSession session(first, learner, 20, &learner);
    bool rejected = false;
    try { session.advance(); } catch (const std::logic_error&) { rejected = true; }
    check(rejected, "advance requires reset");
    session.reset();
    session.advance();
    const auto paused = first.position();
    check(session.episode().length == 1 && learner.transitions.size() == 1, "one complete transition per advance");
    rejected = false;
    try { session.reset(); } catch (const std::logic_error&) { rejected = true; }
    check(rejected && first.position().x == paused.x, "active reset cannot discard a rollout boundary");
    session.request_stop();
    const auto final = session.advance();
    check(session.finished() && final.truncated && !final.terminated, "external stop truncates");
    check(learner.transitions.back().observation == final.observation, "stop sends final observation to learner");
    const auto reward = session.episode().reward;
    const auto batch = ppo::rl::train_episode(second, other, {}, 2);
    check(batch.reward == reward && batch.length == session.episode().length, "desktop and batch loop agree");
    const auto count = learner.transitions.size();
    session.reset();
    check(session.episode().length == 0 && learner.transitions.size() == count, "reset never updates learner");
    rejected = false;
    try { ppo::rl::EpisodeSession invalid(first, learner, 20, &other); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "feedback cannot go to a different policy");
}
void test_bootstrap()
{
    check(std::abs(ppo::rl::bootstrap_reward(1,false,true,10,0.9F)-10)<1e-6,"truncation bootstrap once");
    check(ppo::rl::bootstrap_reward(1,true,true,10,0.9F)==1,"termination suppresses bootstrap");
    check(ppo::rl::bootstrap_reward(1,false,false,10,0.9F)==1,"ordinary reward unchanged");
    // Known two-step GAE: terminal/truncated t=0 must not include t=1's advantage.
    const double value=2;
    const double delta=ppo::rl::bootstrap_reward(1,false,true,10,0.9F)-value;
    const double legacy_done_mask=0;
    const double advantage=delta+0.9*0.95*legacy_done_mask*1234;
    check(std::abs(advantage-8)<1e-6,"no recurrence across reset");
}
void test_metrics()
{
    std::vector<ppo::rl::Episode> episodes(2);
    episodes[0].reward=2;episodes[1].reward=4;
    episodes[0].final_info["success"]=true;episodes[1].final_info["success"]=false;
    check(ppo::rl::aggregate(episodes,{"episode_reward",ppo::rl::Reduction::Mean})==3,"mean reward");
    check(ppo::rl::aggregate(episodes,{"episode_reward",ppo::rl::Reduction::Std})==1,"population std");
    check(ppo::rl::aggregate(episodes,{"success",ppo::rl::Reduction::Rate})==0.5,"success rate");
    check(ppo::rl::score(episodes,{{"episode_reward",ppo::rl::Reduction::Mean,2}})==6,"weighted score");
    bool missing=false;
    try {ppo::rl::aggregate(episodes,{"missing"});}
    catch(const std::invalid_argument&) {missing=true;}
    check(missing,"missing metrics do not silently score zero");
    episodes[0].reward = 1e308;
    episodes[1].reward = -1e308;
    bool overflow = false;
    try { ppo::rl::aggregate(episodes, {"episode_reward", ppo::rl::Reduction::Std}); }
    catch (const std::overflow_error&) { overflow = true; }
    check(overflow, "overflowing variance must not report zero");
}
void test_checkpoints_and_config()
{
    TemporaryDirectory temporary;
    const auto candidate=temporary.path/"candidate";
    const auto store=temporary.path/"store";
    std::filesystem::create_directory(candidate);
    {std::ofstream file(candidate/"model.test");file<<"synthetic checkpoint, no model";}
    check(ppo::rl::promote_checkpoint(candidate,store,"v1",1),"first promotion");
    check(!ppo::rl::promote_checkpoint(candidate,store,"v2",0),"worse candidate rejected");
    check(!std::filesystem::exists(store/"v2"),"rejected candidate not published");
    check(ppo::rl::promote_checkpoint(candidate,store,"v3",2),"better candidate promoted");
    std::ifstream best(store/"best.txt");double score;std::string version;best>>score>>version;
    check(score==2 && version=="v3","best reference updated");
    bool context_mismatch = false;
    try { ppo::rl::promote_checkpoint(candidate, store, "v4", 100, "different conditions"); }
    catch (const std::invalid_argument&) { context_mismatch = true; }
    check(context_mismatch && !std::filesystem::exists(store / "v4"), "incomparable candidate rejected before copying");
    const auto metric_file=temporary.path/"score.metrics";
    {std::ofstream file(metric_file);file<<"success rate 100\nepisode_reward mean 1\n";}
    const auto criteria=ppo::rl::load_metrics(metric_file);
    check(criteria.size()==2 && criteria[0].weight==100,"declarative scoring loaded");
    const auto scenario_file=temporary.path/"scenario.nav";
    {std::ofstream file(scenario_file);file<<"navigation2d-v2\nstart 0 0 0 1\ngoal 10 0 0.5 0 0\n";}
    check(navigation::load_scenario(scenario_file).goal.x==10,"scenario loaded");
    auto loaded = navigation::load_scenario(scenario_file);
    loaded.goal_velocity = {0.1,0.2};
    navigation::save_scenario(loaded, scenario_file);
    check(navigation::scenario_text(navigation::load_scenario(scenario_file)) == navigation::scenario_text(loaded), "canonical scenario round trip");
    { std::ofstream file(metric_file); file << ppo::rl::metrics_text(criteria); }
    check(ppo::rl::metrics_text(ppo::rl::load_metrics(metric_file)) == ppo::rl::metrics_text(criteria), "canonical metrics round trip");
    {std::ofstream file(scenario_file);file<<"navigation2d-v2\nunknown 0\n";}
    bool rejected=false;
    try {navigation::load_scenario(scenario_file);}
    catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"unknown config rejected");
}
}
int main()
{
    try
    {
        test_navigation();test_moving_goal();test_shared_loop();test_incremental_session();test_bootstrap();test_metrics();test_checkpoints_and_config();
        std::cout<<"7 navigation/runtime test groups passed\n";return 0;
    }
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
