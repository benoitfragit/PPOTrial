#include "application.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace navigation {
namespace {
class ManualPolicy final : public ppo::rl::IPolicy
{
public:
    std::vector<float> action{0,0};
    ppo::rl::Value act(const ppo::rl::Value&) override { return action; }
};
void append_sample(std::vector<float>& values, float value)
{
    if (values.size() == 5000) values.erase(values.begin());
    values.push_back(value);
}
}
struct Application::Impl
{
    Environment environment;
    IPolicyBackend& backend;
    ppo::rl::ILearner* learner = nullptr; // owned by policy
    bool evaluation_paused = false;
    std::string evaluation_context;
    std::filesystem::path metrics, checkpoints;
    std::unique_ptr<ppo::rl::IPolicy> policy;
    std::unique_ptr<ppo::rl::EpisodeSession> session;
    ControlMode mode = ControlMode::Manual;
    std::filesystem::path loaded_checkpoint;
    std::filesystem::path goal_snapshot;
    Telemetry telemetry;
    bool running = false;
    double elapsed = 0;
    std::string status = "Stopped";
    std::unique_ptr<Environment> evaluation_environment;
    std::unique_ptr<ppo::rl::IPolicy> evaluation_policy;
    std::unique_ptr<ppo::rl::EpisodeSession> evaluation_session;
    std::vector<ppo::rl::Episode> evaluation_episodes;
    std::vector<ppo::rl::Metric> evaluation_metrics;
    std::size_t evaluation_count = 0;
    std::filesystem::path candidate;
    std::string version;
    Impl(Scenario scenario, std::filesystem::path metrics_path, std::filesystem::path root, IPolicyBackend& policy_backend)
        : environment(std::move(scenario)), backend(policy_backend), metrics(std::move(metrics_path)), checkpoints(std::move(root))
    {
        policy = std::make_unique<ManualPolicy>();
        session = std::make_unique<ppo::rl::EpisodeSession>(environment, *policy, environment.scenario().max_steps);
        session->reset();
    }
    void record()
    {
        const auto& episode = session->episode();
        telemetry.reward = episode.reward;
        telemetry.length = episode.length;
        telemetry.terminated = episode.terminated;
        telemetry.truncated = episode.truncated;
        if (learner != nullptr)
        {
            const auto outputs = backend.telemetry(*learner);
            telemetry.rollout_step = outputs.rollout_step;
            telemetry.loss = outputs.loss;
            telemetry.actor_loss = outputs.actor_loss;
            telemetry.critic_loss = outputs.critic_loss;
            telemetry.entropy = outputs.entropy;
        }
        if (!session->finished()) return;
        ++telemetry.episodes;
        const auto found = episode.final_info.find("success");
        if (found != episode.final_info.end() && std::get<bool>(found->second)) ++telemetry.successes;
        append_sample(telemetry.rewards, static_cast<float>(episode.reward));
        if (mode == ControlMode::Training) append_sample(telemetry.losses, telemetry.loss);
    }
    void evaluate_tick()
    {
        // A bounded number of transitions per frame; evaluation never touches the learner.
        for (int step = 0; step < 64 && evaluation_session; ++step)
        {
            evaluation_session->advance();
            if (!evaluation_session->finished()) continue;
            evaluation_episodes.push_back(evaluation_session->episode());
            if (evaluation_episodes.size() < evaluation_count)
            {
                evaluation_session->reset({static_cast<std::uint64_t>(evaluation_episodes.size()), {}});
                continue;
            }
            const double value = ppo::rl::score(evaluation_episodes, evaluation_metrics);
            const bool promoted = ppo::rl::promote_checkpoint(candidate, checkpoints / "prod", version, value, evaluation_context);
            status = "Evaluation score " + std::to_string(value) + (promoted ? " - promoted to best" : " - best retained");
            evaluation_session.reset();
            evaluation_policy.reset();
            evaluation_environment.reset();
        }
    }
};
Application::Application(Scenario scenario, std::filesystem::path metrics, std::filesystem::path checkpoints,
                         IPolicyBackend& backend)
    : impl_(std::make_unique<Impl>(std::move(scenario), std::move(metrics), std::move(checkpoints), backend)) {}
Application::~Application() = default;
void Application::set_goal_checkpoint(const std::filesystem::path& checkpoint)
{
    if (impl_->running || evaluating() || (!impl_->session->finished() && impl_->session->episode().length != 0))
        throw std::logic_error("Stop the current episode before changing the goal controller");
    std::unique_ptr<ppo::rl::IPolicy> policy;
    std::filesystem::path snapshot;
    if (!checkpoint.empty())
    {
        const auto directory = impl_->checkpoints / "goal-snapshots";
        std::filesystem::create_directories(directory);
        const auto serial = std::chrono::system_clock::now().time_since_epoch().count();
        snapshot = std::filesystem::absolute(directory / ("goal-" + std::to_string(serial) + ".pt"));
        std::filesystem::copy_file(checkpoint, snapshot); // no overwrite; later Prod promotion cannot replace this snapshot
        try
        {
            const auto dimensions = std::get<std::vector<float>>(impl_->session->observation()).size();
            policy = impl_->backend.load_policy(dimensions, 2, snapshot);
            if (!policy) throw std::runtime_error("backend returned no goal policy");
        }
        catch (...)
        {
            std::error_code error;
            std::filesystem::remove(snapshot, error);
            throw;
        }
    }
    auto session = std::make_unique<ppo::rl::EpisodeSession>(impl_->environment, *impl_->policy,
        impl_->environment.scenario().max_steps, impl_->learner);
    impl_->environment.set_goal_policy(std::move(policy));
    session->reset();
    impl_->session = std::move(session);
    impl_->goal_snapshot = std::move(snapshot);
    impl_->elapsed = 0;
    impl_->status = impl_->goal_snapshot.empty() ? "Goal: constant velocity" : "Goal: pinned frozen policy";
}
const std::filesystem::path& Application::goal_checkpoint() const noexcept { return impl_->goal_snapshot; }
void Application::play(ControlMode mode, std::size_t rollout, const std::filesystem::path& checkpoint)
{
    if (mode != ControlMode::Manual && mode != ControlMode::Training && mode != ControlMode::Policy)
        throw std::invalid_argument("invalid control mode");
    if (evaluating())
    {
        impl_->evaluation_paused = false;
        impl_->status = "Evaluating frozen candidate";
        return;
    }
    if (mode != impl_->mode || (mode == ControlMode::Policy && checkpoint != impl_->loaded_checkpoint))
    {
        if (!impl_->session->finished() && impl_->session->episode().length != 0)
            throw std::logic_error("Stop the current episode before changing mode");
        const auto dimensions = std::get<std::vector<float>>(impl_->session->observation()).size();
        std::unique_ptr<ppo::rl::IPolicy> next;
        ppo::rl::ILearner* learner = nullptr;
        if (mode == ControlMode::Training)
        {
            auto training = impl_->backend.create_learner(dimensions, 2, rollout, impl_->checkpoints / "work");
            learner = training.get();
            next = std::move(training);
        }
        else if (mode == ControlMode::Policy)
            next = impl_->backend.load_policy(dimensions, 2, checkpoint);
        else next = std::make_unique<ManualPolicy>();
        if (!next) throw std::runtime_error("backend returned no policy");
        auto session = std::make_unique<ppo::rl::EpisodeSession>(impl_->environment, *next,
                                                               impl_->environment.scenario().max_steps, learner);
        session->reset();
        impl_->session = std::move(session);
        impl_->policy = std::move(next);
        impl_->learner = learner;
        impl_->mode = mode;
        impl_->loaded_checkpoint = checkpoint;
        impl_->telemetry = {};
    }
    if (impl_->session->finished()) impl_->session->reset();
    impl_->running = true;
    impl_->status = "Running";
}
void Application::pause() { impl_->running = false; impl_->elapsed = 0; impl_->evaluation_paused = true; impl_->status = "Paused"; }
void Application::stop()
{
    pause();
    if (evaluating())
    {
        impl_->evaluation_session.reset();
        impl_->evaluation_policy.reset();
        impl_->evaluation_environment.reset();
        impl_->status = "Evaluation cancelled; candidate not promoted";
        return;
    }
    if (!impl_->session->finished() && impl_->session->episode().length != 0)
    {
        impl_->session->request_stop();
        impl_->session->advance();
        impl_->record();
    }
    impl_->status = "Stopped";
}
void Application::tick(double elapsed, const std::vector<float>& manual_action, double time_scale)
{
    if (!std::isfinite(elapsed) || elapsed < 0 || !std::isfinite(time_scale) || time_scale <= 0)
        throw std::invalid_argument("invalid desktop clock");
    if (evaluating())
    {
        if (impl_->evaluation_paused) return;
        try { impl_->evaluate_tick(); }
        catch (...)
        {
            impl_->evaluation_session.reset();
            impl_->evaluation_policy.reset();
            impl_->evaluation_environment.reset();
            impl_->status = "Evaluation failed; best reference unchanged unless publication completed";
            throw;
        }
        return;
    }
    if (!impl_->running) return;
    if (auto* manual = dynamic_cast<ManualPolicy*>(impl_->policy.get())) manual->action = manual_action;
    const double dt = impl_->environment.scenario().dt;
    impl_->elapsed = std::min(impl_->elapsed + elapsed * time_scale, 64 * dt);
    while (impl_->elapsed >= dt)
    {
        if (impl_->session->finished()) impl_->session->reset();
        impl_->session->advance();
        impl_->record();
        impl_->elapsed -= dt;
    }
}
void Application::evaluate_candidate(std::size_t episodes)
{
    if (evaluating()) throw std::logic_error("evaluation already running");
    auto* learner = impl_->learner;
    if (!learner) throw std::logic_error("create a training policy first");
    if (episodes == 0 || episodes > 10000) throw std::invalid_argument("invalid evaluation episode count");
    pause();
    auto metrics = ppo::rl::load_metrics(impl_->metrics);
    const auto serial = std::chrono::system_clock::now().time_since_epoch().count();
    const std::string version = "candidate-" + std::to_string(serial);
    const auto candidate = impl_->checkpoints / "candidates" / version;
    if (std::filesystem::exists(candidate)) throw std::runtime_error("candidate already exists");
    impl_->backend.save(*learner, candidate);
    save_scenario(impl_->environment.scenario(), candidate / "scenario.nav");
    // Serialize the parsed criteria, not a file which may change after it was read.
    const auto criteria_text = ppo::rl::metrics_text(metrics);
    std::ofstream criteria(candidate / "score.metrics");
    criteria << criteria_text;
    criteria.close();
    if (!criteria) throw std::runtime_error("cannot save candidate scoring criteria");
    auto environment = std::make_unique<Environment>(impl_->environment.scenario());
    const auto dimensions = std::get<std::vector<float>>(impl_->session->observation()).size();
    if (!impl_->goal_snapshot.empty())
    {
        const auto goal_archive = candidate / "goal-model.pt";
        std::filesystem::copy_file(impl_->goal_snapshot, goal_archive);
        auto goal_policy = impl_->backend.load_policy(dimensions, 2, goal_archive);
        if (!goal_policy) throw std::runtime_error("backend returned no evaluation goal policy");
        environment->set_goal_policy(std::move(goal_policy));
    }
    auto policy = impl_->backend.load_policy(dimensions, 2, candidate / "eval" / "model.pt");
    if (!policy) throw std::runtime_error("backend returned no evaluation policy");
    auto session = std::make_unique<ppo::rl::EpisodeSession>(*environment, *policy, environment->scenario().max_steps);
    session->reset({0, {}});
    const auto evaluation_context = scenario_text(impl_->environment.scenario()) + criteria_text
        + "episodes " + std::to_string(episodes) + "\nseed-sequence 0..episodes-1\n"
        + "goal-controller " + (impl_->goal_snapshot.empty() ? std::string("linear") : impl_->goal_snapshot.generic_string()) + "\n";
    std::ofstream context(candidate / "evaluation-context.txt", std::ios::binary);
    context << evaluation_context;
    context.close();
    if (!context) throw std::runtime_error("cannot save candidate evaluation context");
    impl_->evaluation_environment = std::move(environment);
    impl_->evaluation_policy = std::move(policy);
    impl_->evaluation_session = std::move(session);
    impl_->evaluation_metrics = std::move(metrics);
    impl_->evaluation_context = evaluation_context;
    impl_->evaluation_paused = false;
    impl_->evaluation_episodes.clear();
    impl_->evaluation_count = episodes;
    impl_->version = version;
    impl_->candidate = candidate;
    impl_->status = "Evaluating frozen candidate";
}
const Environment& Application::environment() const noexcept { return impl_->environment; }
const ppo::rl::Value& Application::observation() const noexcept { return impl_->session->observation(); }
const Telemetry& Application::telemetry() const noexcept { return impl_->telemetry; }
bool Application::running() const noexcept { return impl_->running; }
bool Application::evaluating() const noexcept { return impl_->evaluation_session != nullptr; }
ControlMode Application::mode() const noexcept { return impl_->mode; }
const std::string& Application::status() const noexcept { return impl_->status; }
}
