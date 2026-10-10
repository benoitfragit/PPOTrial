#include "navigation.hpp"
#include "ppo/rl/runtime.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void set_goal_velocity(navigation::Scenario& scenario, navigation::Point velocity)
{
    scenario.mobiles[1].initial.heading = std::atan2(velocity.y, velocity.x);
    scenario.mobiles[1].initial.speed = std::hypot(velocity.x, velocity.y);
}
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
    scenario.mobiles[1].initial.position={2,0}; scenario.mobiles[0].arrival_radius = scenario.mobiles[1].arrival_radius = 0.25; scenario.mobiles[0].initial.speed=5; scenario.dt=1;
    navigation::Environment environment(scenario);
    environment.reset();
    auto arrival=environment.step(std::vector<float>{0,0});
    check(arrival.terminated && std::get<bool>(arrival.info.at("success")),"swept arrival");
    check(std::abs(environment.position().x-1.75)<1e-9,"stop at first arrival");
    scenario.mobiles[1].initial.position={10,0}; scenario.zones={{100,"Zone",{2,0},0.25}};
    navigation::Environment blocked(scenario);
    blocked.reset();
    const auto collision=blocked.step(std::vector<float>{0,0});
    check(collision.terminated && std::get<bool>(collision.info.at("collision")),"no obstacle tunnelling");
    check(std::abs(blocked.position().x-1.75)<1e-9,"stop at first collision");
    scenario.mobiles[1].initial.position={1,0}; scenario.zones={{100,"Zone",{3,0},0.25}};
    navigation::Environment early_goal(scenario);
    early_goal.reset();
    check(std::get<bool>(early_goal.step(std::vector<float>{0,0}).info.at("success")),"arrival precedes later obstacle");
    scenario.mobiles[1].initial.position={-1,0}; scenario.half_extent=4;scenario.zones.clear();
    navigation::Environment boundary(scenario);boundary.reset();
    check(std::get<bool>(boundary.step(std::vector<float>{0,0}).info.at("collision")),"boundary crossing");
    check(std::abs(boundary.position().x-4)<1e-9,"boundary position");
}
void test_moving_goal()
{
    navigation::Scenario scenario;
    scenario.mobiles[1].initial.position = {2,0};
    set_goal_velocity(scenario, {0.5,0});
    scenario.mobiles[0].arrival_radius = scenario.mobiles[1].arrival_radius = 0.25;
    scenario.mobiles[0].initial.speed = 1;
    scenario.dt = 1;
    scenario.max_steps = 20;
    scenario.zones = {{100,"Zone",{2,2},0.3}};
    navigation::Environment environment(scenario);
    const auto initial = environment.reset({42,{}});
    const auto& observed = std::get<std::vector<float>>(initial.observation);
    check(observed.size() == 13 + navigation::kLidarBinCount && std::abs(observed[8] - 0.1F) < 1e-6F, "goal velocity included in observation");
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
    detour_scenario.mobiles[1].initial.position = {2,0};
    set_goal_velocity(detour_scenario, {0.1,0});
    detour_scenario.mobiles[0].arrival_radius = detour_scenario.mobiles[1].arrival_radius = 0.25;
    detour_scenario.mobiles[0].initial.speed = 1;
    detour_scenario.dt = 1;
    detour_scenario.max_steps = 20;
    detour_scenario.max_turn_rate = 1.5707963267948966;
    detour_scenario.zones = {{100,"Zone",{1,0},0.2}};
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

    scenario.mobiles[1].initial.position = {4,0};
    set_goal_velocity(scenario, {-5,0});
    scenario.mobiles[0].initial.speed = 5;
    scenario.max_steps = 1;
    scenario.zones.clear();
    navigation::Environment crossing(scenario);
    crossing.reset();
    const auto meeting = crossing.step(std::vector<float>{0,0});
    check(std::get<bool>(meeting.info.at("success"))
        && std::abs(crossing.position().x - 1.875) < 1e-9
        && std::abs(crossing.goal_state().position.x - 2.125) < 1e-9, "relative sweep detects meeting between endpoints");

    set_goal_velocity(scenario, {-1,0});
    scenario.zones = {{100,"Zone",{1,0},0.1}};
    navigation::Environment blocked(scenario);
    blocked.reset();
    const auto collision = blocked.step(std::vector<float>{0,0});
    check(std::get<bool>(collision.info.at("collision")) && !std::get<bool>(collision.info.at("success")), "forbidden zone reached before moving goal");
    check(std::abs(blocked.goal_state().position.x - 3.82) < 1e-9, "goal uses collision event time too");

    scenario.zones.clear();
    set_goal_velocity(scenario, {1000,0});
    bool rejected = false;
    try { navigation::Environment invalid(scenario); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "goal speed above configured limits rejected");
    scenario.mobiles[1].initial.speed = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try { navigation::Environment invalid(scenario); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "nonfinite goal velocity rejected");
}
void test_lidar()
{
    navigation::Scenario scenario;
    scenario.half_extent = 10;
    scenario.mobiles[0].initial = {{0,0}, 0, 0};
    scenario.mobiles[1].initial.position = {8,8};
    scenario.zones = {{100, "Forward obstacle", {5,0}, 1}};
    navigation::Environment environment(scenario);
    const auto first = environment.reset();
    const auto scan = environment.lidar(1);
    // The central ray points along the mobile's heading (positive x).
    check(std::abs(scan.distances[30] - 4.0) < 1e-6, "LiDAR hits circular zone before boundary");
    check(std::abs(scan.normalized[30] - 0.8F) < 1e-6F,
          "LiDAR preserves legacy d/(d+radius) normalization");
    check(std::abs(scan.distances[0] - 10.0) < 1e-6, "LiDAR sees west work-area boundary");
    check(std::abs(scan.normalized[0] - static_cast<float>(10.0 / (10.0 + std::sqrt(200.0)))) < 1e-6F,
          "work-area boundary uses its diagonal reference");
    check(std::abs(scan.distances[60] - scan.distances[0]) < 1e-6,
          "first and final 360-degree rays coincide");
    navigation::Scenario closer = scenario;
    closer.zones.push_back({101, "Closer", {3,0}, 0.5});
    navigation::Environment competing(closer);
    const auto nearest = competing.lidar(1);
    check(std::abs(nearest.distances[30] - 2.5) < 1e-6,
          "nearest of multiple circles determines raw range");
    check(std::abs(nearest.normalized[30] - static_cast<float>(2.5 / 3.5)) < 1e-6F,
          "normalized LiDAR value follows physically nearest obstacle");
    navigation::Scenario rotated = scenario;
    rotated.mobiles[0].initial.heading = 1.5707963267948966;
    rotated.zones[0].center = {0,5};
    navigation::Environment turned(rotated);
    check(std::abs(turned.lidar(1).distances[30] - 4.0) < 1e-6,
          "LiDAR rays rotate with mobile heading");
    const auto& obs = std::get<std::vector<float>>(first.observation);
    check(obs.size() == 13 + navigation::kLidarBinCount, "LiDAR is in learner observation");
    check(std::abs(obs[13+30] - scan.normalized[30]) < 1e-6, "observation uses same scan");
    check(environment.lidar(2).normalized.size() == navigation::kLidarBinCount,
          "each independent mobile has its own scan");
    bool rejects_zone = false;
    try { (void)environment.lidar(100); } catch (const std::invalid_argument&) { rejects_zone = true; }
    check(rejects_zone, "zones cannot masquerade as sensing mobiles");

    scenario.zones.clear();
    navigation::Environment boundaries(scenario);
    const auto empty = boundaries.lidar(1);
    check(std::abs(empty.distances[30] - 10.0) < 1e-6,
          "boundary alone is visible to LiDAR");
    check(empty.normalized[30] < 1, "boundary range is normalized for the learner");
    scenario.zones = {{100,"Inside", {0,0}, 1}};
    // Valid topology forbids spawning a mobile inside an obstacle.
    bool invalid = false;
    try { navigation::Environment forbidden(scenario); } catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "inside-zone initial state is rejected");
}
void test_avoidance_reward()
{
    navigation::Scenario near_zone;
    near_zone.mobiles[0].initial = {{0,0}, 0, 0};
    near_zone.mobiles[1].initial.position = {8,8};
    near_zone.zones = {{100,"Nearby zone",{1.5,0},0.4}};
    navigation::Scenario far_zone = near_zone;
    far_zone.zones[0].center = {5,5};
    navigation::Environment near(near_zone), far(far_zone);
    near.reset(); far.reset();
    const auto near_reward = near.step(std::vector<float>{0,0}).reward;
    const auto far_reward = far.step(std::vector<float>{0,0}).reward;
    check(near_reward < far_reward, "early zone proximity incurs dense safety penalty");

    navigation::Scenario edge = near_zone;
    edge.zones.clear();
    edge.mobiles[0].initial.position = {99.5,0};
    navigation::Scenario middle = edge;
    middle.mobiles[0].initial.position = {0,0};
    navigation::Environment boundary(edge), clear(middle);
    boundary.reset(); clear.reset();
    check(boundary.step(std::vector<float>{0,0}).reward
        < clear.step(std::vector<float>{0,0}).reward,
        "approaching work-area edge incurs dense safety penalty");
}
void test_episode_initialization()
{
    navigation::Scenario baseline;
    baseline.zones = {{100,"Circle",{5,5},1}};
    navigation::Environment environment(baseline);
    environment.configure_initialization(navigation::InitializationMode::Randomized);
    const auto first = environment.reset({1234, {}});
    const auto pos1 = environment.position();
    const auto goal1 = environment.goal_state().position;
    const auto second = environment.reset({1234, {}});
    check(environment.position().x == pos1.x && environment.position().y == pos1.y
          && environment.goal_state().position.x == goal1.x
          && environment.goal_state().position.y == goal1.y,
          "random initialization reproduces all entities with the same seed");
    check(std::get<std::vector<float>>(first.observation)
              == std::get<std::vector<float>>(second.observation),
          "random seeded reset reproduces observations including LiDAR");
    environment.reset({1235, {}});
    check(environment.position().x != pos1.x || environment.position().y != pos1.y,
          "a different seed changes randomized initial state");

    auto other = baseline;
    other.mobiles[0].initial.position = {-10,-10};
    other.mobiles[1].initial.position = {25,10};
    environment.configure_initialization(navigation::InitializationMode::PresetCases, {baseline, other});
    environment.reset({1,{}});
    check(environment.position().x == -10 && environment.goal_state().position.x == 25,
          "preset evaluation case swaps complete mobile world");
    environment.reset({0,{}});
    check(environment.position().x == 0 && environment.goal_state().position.x == 10,
          "preset selection uses the supplied episode index");

    auto invalid = other;
    invalid.mobiles[1].id = 999;
    bool rejected = false;
    try { environment.configure_initialization(navigation::InitializationMode::PresetCases,{invalid}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && environment.episode_cases().size() == 2,
          "invalid preset topology is rejected atomically");
    environment.configure_initialization(navigation::InitializationMode::Fixed);
    environment.reset();
    check(environment.position().x == 0 && environment.goal_state().position.x == 10,
          "fixed initialization restores the original world");

    TemporaryDirectory directory;
    navigation::save_scenario(baseline, directory.path / "case-a.nav");
    navigation::save_scenario(other, directory.path / "case-b.nav");
    {
        std::ofstream manifest(directory.path / "evaluation.cases");
        manifest << "# Evaluation scenarios\\ncase-a.nav\\ncase-b.nav\\n";
    }
    const auto cases = navigation::load_episode_cases(directory.path / "evaluation.cases");
    check(cases.size() == 2 && cases[1].mobiles[0].initial.position.x == -10,
          "case manifest loads multiple complete named scenarios");
}
void test_shared_loop()
{
    navigation::Scenario scenario; scenario.max_steps=2; set_goal_velocity(scenario, {0.5,0.2});
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
    check(invalid && first.position().x==scenario.mobiles[0].initial.position.x,"invalid action does not mutate environment");
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
    {std::ofstream file(scenario_file);file<<"navigation2d-v3\nmobile 1 \"Mobile\" 0 0 0 1 2 0.5\nmobile 2 \"Goal\" 10 0 0 0 1 0.5\n";}
    check(navigation::load_scenario(scenario_file).mobiles[1].initial.position.x==10,"scenario loaded");
    auto loaded = navigation::load_scenario(scenario_file);
    set_goal_velocity(loaded, {0.1,0.2});
    navigation::save_scenario(loaded, scenario_file);
    check(navigation::scenario_text(navigation::load_scenario(scenario_file)) == navigation::scenario_text(loaded), "canonical scenario round trip");
    { std::ofstream file(metric_file); file << ppo::rl::metrics_text(criteria); }
    check(ppo::rl::metrics_text(ppo::rl::load_metrics(metric_file)) == ppo::rl::metrics_text(criteria), "canonical metrics round trip");
    {std::ofstream file(scenario_file);file<<"navigation2d-v3\nunknown 0\n";}
    bool rejected=false;
    try {navigation::load_scenario(scenario_file);}
    catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"unknown config rejected");
    { std::ofstream file(scenario_file); file << "navigation2d-v3\nmobile -1 \"Invalid\" 0 0 0 1 2 0.5\n"; }
    rejected=false;
    try { navigation::load_scenario(scenario_file); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"negative entity IDs rejected rather than wrapping");
}
void test_ecs_world()
{
    navigation::Scenario scenario;
    scenario.dt = 1; scenario.max_steps = 5;
    scenario.mobiles[1].initial.position = {20,0};
    scenario.mobiles.push_back({3,"Learner B",{{0,10},0,2},4,0.5});
    scenario.mobiles.push_back({4,"Goal B",{{20,10},0,0},3,0.5});
    scenario.mobiles.push_back({5,"Manual",{{-10,-10},0,1},0,0.5});
    scenario.zones.push_back({100,"Zone",{5,5},1});
    navigation::Environment environment(scenario);
    check(environment.entities().size()==6, "N mobiles and zones are entities");
    std::size_t mobiles=0, zones=0;
    for (const auto entity:environment.registry().view<const navigation::Identity>())
    {
        if (environment.registry().all_of<navigation::Mobile>(entity)) ++mobiles;
        if (environment.registry().all_of<navigation::ForbiddenZone>(entity)) ++zones;
    }
    check(mobiles==5 && zones==1, "real EnTT components describe all entities");
    auto secondary=std::make_unique<RecordingLearner>();
    auto* learner_b=secondary.get();
    environment.set_policy(3,std::move(secondary),true);
    auto frozen=std::make_unique<RecordingLearner>();
    auto* prod=frozen.get();
    environment.set_policy(2,std::move(frozen));
    RecordingLearner primary;
    ppo::rl::EpisodeSession session(environment,primary,5,&primary);
    session.reset();
    const auto handle=environment.handle(3);
    environment.set_manual_action(5,{0,1});
    session.advance();
    check(environment.state(1).position.x==1 && environment.state(3).position.x==2
        && environment.state(5).position.x==-7, "one simultaneous world step advances independent controllers");
    check(primary.transitions.size()==1 && learner_b->transitions.size()==1 && prod->transitions.empty(), "only attached learners receive their own feedback");
    check(std::get<std::vector<float>>(learner_b->transitions[0].observation)[1]==0.1F,
          "secondary learner sees its own position, not primary observation");
    environment.request_stop(); session.request_stop(); session.advance();
    check(primary.transitions.back().truncated && learner_b->transitions.back().truncated, "external world stop closes all learners");
    session.reset();
    check(environment.handle(3)==handle && environment.state(3).position.x==0 && environment.entities().size()==6,
          "reset restores every entity without changing identities or duplicating entities");

    navigation::Scenario event_scenario=scenario;
    event_scenario.zones.push_back({101,"Secondary blocker",{1,10},0.2});
    navigation::Environment event(event_scenario);
    auto event_learner=std::make_unique<RecordingLearner>();
    auto* event_record=event_learner.get();
    event.set_policy(3,std::move(event_learner),true); event.reset();
    const auto result=event.step(std::vector<float>{0,0});
    check(result.truncated && !result.terminated && event_record->transitions.back().terminated,
          "another entity's terminal event truncates the primary's unfinished episode");
    check(std::abs(event.state(1).position.x-0.4)<1e-9 && std::abs(event.state(3).position.x-0.8)<1e-9,
          "all entities stop at the same earliest event time");

    navigation::Environment editable(scenario);
    editable.create_mobile({42,"Extra",{{-20,20},0,0},0,0.5});
    check(editable.entities().size()==7, "create additional entity");
    editable.remove_entity(42);
    bool rejected=false;
    try { editable.handle(42); } catch (const std::out_of_range&) { rejected=true; }
    check(rejected, "removed identity is invalid");
    rejected=false;
    try { editable.remove_entity(2); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected && editable.entities().size()==6, "cannot remove referenced destination; edit is atomic");
    editable.create_zone({102,"New zone",{-30,30},2});
    check(std::get<std::vector<float>>(editable.reset().observation).size()==16 + navigation::kLidarBinCount, "zone entity changes observation space on reset");
    editable.step(std::vector<float>{0,0});
    rejected=false;
    try { editable.create_mobile({43,"Late",{{-40,20},0,0},0,0.5}); } catch (const std::logic_error&) { rejected=true; }
    check(rejected, "topology cannot change during an episode");
    std::reverse(scenario.mobiles.begin(),scenario.mobiles.end());
    navigation::Environment reversed(scenario); reversed.reset();
    check(reversed.entities().front()==1, "entity processing order is stable across declaration order");
}

}
int main(int argc, char** argv)
{
    try
    {
        if (argc > 1)
        {
            for (int index=1; index<argc; ++index)
            {
                navigation::Environment environment(navigation::load_scenario(argv[index]));
                environment.reset();
                environment.step(std::vector<float>{0,0});
                check(environment.entities().size() >= 2, "packaged scenario creates ECS entities");
            }
            std::cout << "Packaged scenarios parsed and stepped without models\n";
            return 0;
        }
        test_navigation();test_moving_goal();test_lidar();test_avoidance_reward();test_episode_initialization();test_shared_loop();test_incremental_session();test_bootstrap();test_metrics();test_checkpoints_and_config();test_ecs_world();
        std::cout<<"11 navigation/runtime test groups passed\n";return 0;
    }
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
