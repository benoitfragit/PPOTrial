#include "application.hpp"
#include <chrono>
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
    std::unique_ptr<ppo::rl::ILearner> create_learner(
        std::size_t, std::size_t, std::size_t, const std::filesystem::path&) override
    {
        ++counts.creations;
        return std::make_unique<FakeLearner>(counts);
    }
    std::unique_ptr<ppo::rl::IPolicy> load_policy(
        std::size_t, std::size_t, const std::filesystem::path&) override
    {
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
    scenario.goal = {90,0};
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
    check(!application.evaluating() && !std::filesystem::exists(directory.path / "store/prod/best.txt"), "cancelled evaluation is not promoted");
    bool rejected = false;
    try { application.play(static_cast<navigation::ControlMode>(100), 64, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid mode is not silently treated as manual");
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
    check(!first.evaluating() && std::filesystem::exists(directory.path / "store/prod/best.txt"), "fake evaluation promotes first candidate");
    scenario.goal.x = 11;
    navigation::Application different(scenario, directory.path / "score.metrics", directory.path / "store", backend);
    different.play(navigation::ControlMode::Training, 64, {});
    different.evaluate_candidate(1);
    bool rejected = false;
    try { different.tick(0, {0,0}, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && !different.evaluating(), "different scenarios cannot share a best score");
    check(backend.counts.feedback == 0, "evaluation never updates either learner");
}
}
int main()
{
    try
    {
        test_controls();
        test_evaluation_context();
        std::cout << "2 application control test groups passed (fake backend only)\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
