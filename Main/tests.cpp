#include "application.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
struct Counts
{
    int creations = 0, loads = 0, actions = 0, feedback = 0, evaluation_actions = 0, saves = 0;
};
class FakeLearner final : public ppo::rl::ILearner
{
public:
    explicit FakeLearner(Counts& counts) : counts_(counts) {}
    ppo::rl::Value act(const ppo::rl::Value&) override
    {
        ++counts_.actions;
        return std::vector<float>{0,0};
    }
    void observe(const ppo::rl::StepResult&) override { ++counts_.feedback; }
private:
    Counts& counts_;
};
class FakePolicy final : public ppo::rl::IPolicy
{
public:
    explicit FakePolicy(Counts& counts) : counts_(counts) {}
    ppo::rl::Value act(const ppo::rl::Value&) override
    {
        ++counts_.evaluation_actions;
        return std::vector<float>{0,0};
    }
private:
    Counts& counts_;
};
class FakeBackend final : public navigation::IPolicyBackend
{
public:
    Counts counts;
    navigation::TrainingParameters last_training_parameters{};
    std::vector<std::filesystem::path> loaded_paths;
    std::unique_ptr<ppo::rl::ILearner> create_learner_configured(
        std::size_t observations, std::size_t actions, std::size_t rollout,
        const std::filesystem::path& root, const navigation::TrainingParameters& settings) override
    {
        navigation::validate_training_parameters(settings);
        last_training_parameters = settings;
        return create_learner(observations, actions, rollout, root);
    }
    std::unique_ptr<ppo::rl::ILearner> create_learner(
        std::size_t, std::size_t, std::size_t, const std::filesystem::path&) override
    {
        ++counts.creations;
        return std::make_unique<FakeLearner>(counts);
    }
    std::unique_ptr<ppo::rl::IPolicy> load_policy(
        std::size_t, std::size_t, const std::filesystem::path& path) override
    {
        loaded_paths.push_back(path);
        ++counts.loads;
        return std::make_unique<FakePolicy>(counts);
    }
    navigation::BackendTelemetry telemetry(const ppo::rl::ILearner&) const override { return {}; }
    void save(ppo::rl::ILearner&, const std::filesystem::path& path) override
    {
        ++counts.saves;
        std::filesystem::create_directories(path / "eval");
        std::ofstream output(path / "eval" / "model.pt");
        output << "synthetic test data; not a neural model";
        output.close();
        check(static_cast<bool>(output), "write fake checkpoint");
    }
};
class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        path = std::filesystem::temp_directory_path() / ("ppo-application-test-"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(path), "create temporary directory");
        std::ofstream output(path / "score.metrics");
        output << "episode_reward mean 1\n";
    }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
    std::filesystem::path path;
};
void test_controls()
{
    TemporaryDirectory directory;
    FakeBackend backend;
    navigation::Scenario scenario;
    scenario.mobiles[1].initial.position = {90,0};
    navigation::Application application(scenario, directory.path / "score.metrics", directory.path / "store", backend);
    check(!application.running() && backend.counts.creations == 0 && backend.counts.loads == 0, "launch must not create a neural policy");
    application.play(navigation::ControlMode::Training, 64, {});
    application.tick(scenario.dt, {0,0}, 1);
    check(backend.counts.actions == 1 && backend.counts.feedback == 1, "complete training transition");
    application.pause();
    application.tick(10, {0,0}, 1);
    check(backend.counts.actions == 1, "training paused");
    application.play(navigation::ControlMode::Training, 64, {});
    application.tick(scenario.dt, {0,0}, 1);
    check(backend.counts.creations == 1 && backend.counts.feedback == 2, "resume preserves learner");
    application.evaluate_candidate(2);
    application.pause();
    application.tick(1, {0,0}, 1);
    check(backend.counts.evaluation_actions == 0 && application.evaluating(), "pause suspends pending evaluation");
    application.play(navigation::ControlMode::Training, 64, {});
    application.tick(1, {0,0}, 1);
    check(backend.counts.evaluation_actions == 64, "evaluation resumes incrementally");
    application.pause();
    application.tick(1, {0,0}, 1);
    check(backend.counts.evaluation_actions == 64 && backend.counts.feedback == 2, "paused evaluation does not step or update learner");
    application.stop();
    check(!application.evaluating() && !std::filesystem::exists(directory.path / "store/prod/mobile-1/best.txt"), "cancelled evaluation is not promoted");
    bool rejected = false;
    try { application.play(static_cast<navigation::ControlMode>(100), 64, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid mode is not silently treated as manual");
}
void test_training_settings()
{
    TemporaryDirectory directory;
    FakeBackend backend;
    navigation::Scenario scenario;
    scenario.mobiles[1].initial.position = {90,0};
    navigation::Application application(scenario, directory.path / "score.metrics",
                                        directory.path / "store", backend);
    auto parameters = application.training_parameters();
    parameters.learning_rate = 0.0005F;
    parameters.gamma = 0.97F;
    parameters.epochs = 7;
    application.set_training_parameters(parameters);
    check(application.training_parameters().gamma == 0.97F, "PPO settings accepted before training");
    application.play(navigation::ControlMode::Training, 128, {});
    check(backend.last_training_parameters.learning_rate == 0.0005F
          && backend.last_training_parameters.epochs == 7,
          "primary learner receives new PPO hyperparameters");
    bool rejected = false;
    parameters.gamma = 2;
    try { application.set_training_parameters(parameters); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "out-of-range PPO options rejected even while running");
    parameters.gamma = 0.99F;
    rejected = false;
    try { application.set_training_parameters(parameters); }
    catch (const std::logic_error&) { rejected = true; }
    check(rejected, "optimizer options cannot silently mutate a running learner");
}
void test_evaluation_context()
{
    TemporaryDirectory directory;
    FakeBackend backend;
    navigation::Scenario scenario;
    scenario.max_steps = 2;
    navigation::Application first(scenario, directory.path / "score.metrics", directory.path / "store", backend);
    first.play(navigation::ControlMode::Training, 64, {});
    first.evaluate_candidate(1);
    first.tick(0, {0,0}, 1);
    check(!first.evaluating() && std::filesystem::exists(directory.path / "store/prod/mobile-1/best.txt"), "fake evaluation promotes first candidate");
    scenario.mobiles[1].initial.position.x = 11;
    navigation::Application different(scenario, directory.path / "score.metrics", directory.path / "store", backend);
    different.play(navigation::ControlMode::Training, 64, {});
    different.evaluate_candidate(1);
    bool rejected = false;
    try { different.tick(0, {0,0}, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && !different.evaluating(), "different scenarios cannot share a best score");
    check(backend.counts.feedback == 0, "evaluation never updates either learner");
}
class RecordingGoal final : public ppo::rl::ILearner
{
public:
    ppo::rl::Value last_observation;
    int calls = 0, feedback = 0;
    ppo::rl::Value action = std::vector<float>{0, 1};
    ppo::rl::Value act(const ppo::rl::Value& observation) override
    {
        ++calls;
        last_observation = observation;
        return action;
    }
    void observe(const ppo::rl::StepResult&) override { ++feedback; }
};
void test_goal_policy()
{
    navigation::Scenario scenario;
    scenario.dt = 1;
    scenario.max_steps = 2;
    navigation::Environment environment(scenario);
    auto goal = std::make_unique<RecordingGoal>();
    auto* recording = goal.get();
    environment.set_goal_policy(std::move(goal));
    environment.reset();
    environment.step(std::vector<float>{0,0});
    const auto& view = std::get<std::vector<float>>(recording->last_observation);
    check(std::abs(view[0] - 0.1F) < 1e-6F && view[2] == 0, "goal sees itself first and controlled mobile second, before motion");
    check(std::abs(view[8] - 0.2F) < 1e-6F, "goal sees other mobile velocity");
    check(environment.goal_state().position.x == 12 && environment.goal_state().speed == 2, "goal policy controls acceleration");
    check(recording->calls == 1 && recording->feedback == 0, "goal policy only acts, never learns");
    const auto observation = std::get<std::vector<float>>(environment.reset().observation);
    check(environment.goal_state().position.x == 10 && observation[8] == 0, "reset restores goal motion");
    recording->action = std::vector<float>{9,0};
    bool rejected = false;
    try { environment.step(std::vector<float>{0,1}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && environment.state().speed == 1 && environment.goal_state().position.x == 10,
          "invalid goal action rejected before either state changes");
    navigation::Scenario blocked;
    blocked.dt = 1; blocked.max_steps = 2;
    blocked.zones.push_back({100,"Zone",{11,0},0.2});
    navigation::Environment bounded(blocked);
    bounded.set_goal_policy(std::make_unique<RecordingGoal>());
    bounded.reset();
    const auto result = bounded.step(std::vector<float>{0,0});
    check(result.terminated && std::get<bool>(result.info.at("goal_collision")) && !std::get<bool>(result.info.at("success")),
          "goal policy collision ends episode without meeting success");
    check(std::abs(bounded.goal_state().position.x-10.8) < 1e-10, "goal cannot tunnel through forbidden zone");
}
void test_pinned_goal_checkpoint()
{
    TemporaryDirectory directory;
    FakeBackend backend;
    const auto source = directory.path / "prod-model.pt";
    { std::ofstream output(source); output << "N-1 synthetic"; }
    navigation::Scenario scenario;
    scenario.max_steps = 2;
    navigation::Application app(scenario, directory.path / "score.metrics", directory.path / "store", backend);
    app.set_goal_checkpoint(source);
    const auto pinned = app.goal_checkpoint();
    check(pinned != source && std::filesystem::exists(pinned), "selected Prod is privately pinned");
    { std::ofstream output(source); output << "N synthetic"; }
    std::ifstream input(pinned);
    std::string bytes; std::getline(input, bytes);
    check(bytes == "N-1 synthetic", "later source replacement cannot change pinned goal");
    app.play(navigation::ControlMode::Training, 64, {});
    app.tick(scenario.dt, {0,0}, 1);
    check(backend.counts.feedback == 1 && backend.counts.evaluation_actions == 1, "training updates only main learner while frozen goal acts");
    bool rejected = false;
    try { app.set_goal_checkpoint({}); }
    catch (const std::logic_error&) { rejected = true; }
    check(rejected, "cannot switch goal during episode");
    app.evaluate_candidate(1);
    check(backend.counts.loads == 3 && backend.loaded_paths[1].filename() == "model.pt", "evaluation independently loads pinned goal and candidate");
    app.tick(0, {0,0}, 1);
    check(backend.counts.feedback == 1 && backend.counts.evaluation_actions == 5, "evaluation drives both frozen policies without learner update");
    check(app.goal_checkpoint() == pinned, "promotion does not switch N-1 goal");
    app.stop();
    app.set_goal_checkpoint({});
    check(app.goal_checkpoint().empty(), "explicit return to linear goal mode");
}

void test_multiple_controllers()
{
    TemporaryDirectory directory;
    FakeBackend backend;
    const auto source=directory.path/"prod-model.pt";
    { std::ofstream output(source); output<<"synthetic frozen policy"; }
    navigation::Scenario scenario;
    scenario.max_steps=2;
    scenario.mobiles.push_back({3,"Learner B",{{0,20},0,1},4,0.5});
    scenario.mobiles.push_back({4,"Prod B",{{10,20},0,0},3,0.5});
    scenario.mobiles.push_back({5,"Manual",{{-10,-10},0,1},0,0.5});
    scenario.zones.push_back({100,"Zone",{5,5},1});
    navigation::Application app(scenario,directory.path/"score.metrics",directory.path/"store",backend);
    app.configure_mobile(3,navigation::ControlMode::Training,64);
    app.configure_mobile(2,navigation::ControlMode::Policy,64,source);
    app.configure_mobile(4,navigation::ControlMode::Policy,64,source);
    app.play(navigation::ControlMode::Training,64,{});
    app.tick(scenario.dt,{0,1},1,5);
    check(backend.counts.creations==2 && backend.counts.actions==2 && backend.counts.feedback==2
        && backend.counts.evaluation_actions==2, "two learners and two frozen policies act in the same world");
    check(app.environment().state(5).speed>1 && app.telemetry(3).length==1 && app.telemetry(5).length==1,
          "selected manual entity and per-agent telemetry are independent");
    const auto before=app.environment().state(3);
    app.evaluate_candidate(1,3);
    check(backend.counts.loads==6 && backend.counts.saves==2, "evaluation freezes selected learner and every other controller");
    app.tick(0,{0,0},1);
    check(backend.counts.feedback==2 && backend.counts.evaluation_actions==10,
          "evaluating a non-primary learner never updates any training instance");
    check(app.environment().state(3).position.x==before.position.x
        && std::filesystem::exists(directory.path/"store/prod/mobile-3/best.txt"),
          "independent evaluation publishes only the selected learner");
    app.stop();
    check(app.telemetry(3).truncated && backend.counts.feedback==4, "Stop closes every active learner");
    bool rejected=false;
    try { app.replace_scenario(scenario); } catch (const std::logic_error&) { rejected=true; }
    check(rejected, "topology replacement cannot silently discard learners");
}

}
int main()
{
    try
    {
        test_controls();test_training_settings();
        test_evaluation_context();
        test_goal_policy();
        test_pinned_goal_checkpoint();
        test_multiple_controllers();
        std::cout << "6 application control test groups passed (fake backend only)\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
