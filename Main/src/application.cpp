#include "application.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace navigation
{
namespace
{
class ManualPolicy final : public ppo::rl::IPolicy
{
public:
    std::vector<float> action{0, 0};
    ppo::rl::Value act(const ppo::rl::Value&) override
    {
        return action;
    }
};
void append_sample(std::vector<float>& values, float value)
{
    if (values.size() == 5000)
        values.erase(values.begin());
    values.push_back(value);
}
} // namespace
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
    std::filesystem::path primary_snapshot;
    struct MobileControl
    {
        ControlMode mode = ControlMode::Manual;
        ppo::rl::ILearner* learner = nullptr; // owned by environment controller
        std::filesystem::path snapshot;
        Telemetry telemetry;
    };
    std::map<EntityId, MobileControl> controls;
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
    std::filesystem::path evaluation_store;
    Impl(Scenario scenario, std::filesystem::path metrics_path, std::filesystem::path root,
         IPolicyBackend& policy_backend)
        : environment(std::move(scenario)), backend(policy_backend), metrics(std::move(metrics_path)),
          checkpoints(std::move(root))
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
        for (auto& item : controls)
        {
            auto& control = item.second;
            auto& values = control.telemetry;
            const auto& result = environment.transition(item.first);
            if (episode.length == 1)
            {
                values.reward = 0;
                values.length = 0;
            }
            values.reward += result.reward;
            ++values.length;
            values.terminated = result.terminated;
            values.truncated = result.truncated;
            if (control.learner)
            {
                const auto outputs = backend.telemetry(*control.learner);
                values.rollout_step = outputs.rollout_step;
                values.loss = outputs.loss;
                values.actor_loss = outputs.actor_loss;
                values.critic_loss = outputs.critic_loss;
                values.entropy = outputs.entropy;
            }
            if (result.terminated || result.truncated)
            {
                ++values.episodes;
                if (std::get<bool>(result.info.at("success")))
                    ++values.successes;
                append_sample(values.rewards, static_cast<float>(values.reward));
                if (control.learner)
                    append_sample(values.losses, values.loss);
            }
        }
        if (!session->finished())
            return;
        ++telemetry.episodes;
        const auto found = episode.final_info.find("success");
        if (found != episode.final_info.end() && std::get<bool>(found->second))
            ++telemetry.successes;
        append_sample(telemetry.rewards, static_cast<float>(episode.reward));
        if (mode == ControlMode::Training)
            append_sample(telemetry.losses, telemetry.loss);
    }
    void evaluate_tick()
    {
        // A bounded number of transitions per frame; evaluation never touches the learner.
        for (int step = 0; step < 64 && evaluation_session; ++step)
        {
            evaluation_session->advance();
            if (!evaluation_session->finished())
                continue;
            evaluation_episodes.push_back(evaluation_session->episode());
            if (evaluation_episodes.size() < evaluation_count)
            {
                evaluation_session->reset({static_cast<std::uint64_t>(evaluation_episodes.size()), {}});
                continue;
            }
            const double value = ppo::rl::score(evaluation_episodes, evaluation_metrics);
            const bool promoted =
                ppo::rl::promote_checkpoint(candidate, evaluation_store, version, value, evaluation_context);
            status =
                "Evaluation score " + std::to_string(value) + (promoted ? " - promoted to best" : " - best retained");
            evaluation_session.reset();
            evaluation_policy.reset();
            evaluation_environment.reset();
        }
    }
};
Application::Application(Scenario scenario, std::filesystem::path metrics, std::filesystem::path checkpoints,
                         IPolicyBackend& backend)
    : impl_(std::make_unique<Impl>(std::move(scenario), std::move(metrics), std::move(checkpoints), backend))
{
}
Application::~Application() = default;
void Application::set_goal_checkpoint(const std::filesystem::path& checkpoint)
{
    const auto& env = impl_->environment;
    const auto goal = env.registry().get<Destination>(env.handle(env.scenario().controlled)).entity;
    configure_mobile(goal, checkpoint.empty() ? ControlMode::Manual : ControlMode::Policy, 2048, checkpoint);
}
const std::filesystem::path& Application::goal_checkpoint() const noexcept
{
    const auto& env = impl_->environment;
    const auto goal = env.registry().get<Destination>(env.handle(env.scenario().controlled)).entity;
    const auto found = impl_->controls.find(goal);
    static const std::filesystem::path empty;
    return found == impl_->controls.end() ? empty : found->second.snapshot;
}
void Application::configure_mobile(EntityId mobile, ControlMode mode, std::size_t rollout,
                                   const std::filesystem::path& checkpoint)
{
    if (impl_->running || evaluating() || (!impl_->session->finished() && impl_->session->episode().length != 0))
        throw std::logic_error("Stop the world episode before changing a controller");
    const auto& world = impl_->environment.registry();
    const auto handle = impl_->environment.handle(mobile);
    if (!world.all_of<Mobile>(handle) || mobile == impl_->environment.scenario().controlled)
        throw std::invalid_argument("use primary mode controls for the primary mobile");
    if (mode != ControlMode::Manual && mode != ControlMode::Training && mode != ControlMode::Policy)
        throw std::invalid_argument("invalid mobile control mode");
    if (mode != ControlMode::Manual && world.get<Destination>(handle).entity == no_entity)
        throw std::invalid_argument("policy-controlled mobile needs a destination");
    const auto dimensions = std::get<std::vector<float>>(impl_->session->observation()).size();
    std::unique_ptr<ppo::rl::IPolicy> policy;
    Impl::MobileControl control;
    control.mode = mode;
    if (mode == ControlMode::Policy)
    {
        const auto directory = impl_->checkpoints / "controller-snapshots";
        std::filesystem::create_directories(directory);
        const auto serial = std::chrono::system_clock::now().time_since_epoch().count();
        control.snapshot = std::filesystem::absolute(
            directory / ("mobile-" + std::to_string(mobile) + "-" + std::to_string(serial) + ".pt"));
        std::filesystem::copy_file(checkpoint, control.snapshot);
        try
        {
            policy = impl_->backend.load_policy(dimensions, 2, control.snapshot);
            if (!policy)
                throw std::runtime_error("backend returned no frozen policy");
        }
        catch (...)
        {
            std::error_code error;
            std::filesystem::remove(control.snapshot, error);
            throw;
        }
    }
    else if (mode == ControlMode::Training)
    {
        auto learner = impl_->backend.create_learner(
            dimensions, 2, rollout, impl_->checkpoints / "work" / ("mobile-" + std::to_string(mobile)));
        if (!learner)
            throw std::runtime_error("backend returned no learner");
        control.learner = learner.get();
        policy = std::move(learner);
    }
    auto session = std::make_unique<ppo::rl::EpisodeSession>(impl_->environment, *impl_->policy,
                                                             impl_->environment.scenario().max_steps, impl_->learner);
    impl_->environment.set_policy(mobile, std::move(policy), mode == ControlMode::Training);
    impl_->controls.insert_or_assign(mobile, std::move(control));
    session->reset();
    impl_->session = std::move(session);
    impl_->elapsed = 0;
    impl_->status = "Mobile controller configured";
}
ControlMode Application::mobile_mode(EntityId mobile) const
{
    impl_->environment.handle(mobile);
    if (mobile == impl_->environment.scenario().controlled)
        return impl_->mode;
    const auto found = impl_->controls.find(mobile);
    return found == impl_->controls.end() ? ControlMode::Manual : found->second.mode;
}
const std::filesystem::path& Application::mobile_checkpoint(EntityId mobile) const
{
    impl_->environment.handle(mobile);
    if (mobile == impl_->environment.scenario().controlled)
        return impl_->loaded_checkpoint;
    const auto found = impl_->controls.find(mobile);
    static const std::filesystem::path empty;
    return found == impl_->controls.end() ? empty : found->second.snapshot;
}
const Telemetry& Application::telemetry(EntityId mobile) const
{
    impl_->environment.handle(mobile);
    if (mobile == impl_->environment.scenario().controlled)
        return impl_->telemetry;
    const auto found = impl_->controls.find(mobile);
    static const Telemetry empty;
    return found == impl_->controls.end() ? empty : found->second.telemetry;
}
void Application::replace_scenario(Scenario scenario)
{
    if (impl_->running || evaluating() || impl_->mode != ControlMode::Manual)
        throw std::logic_error("stop and detach policies before replacing the scenario");
    for (const auto& item : impl_->controls)
        if (item.second.mode != ControlMode::Manual)
            throw std::logic_error("detach every mobile policy before replacing the scenario");
    auto next = std::make_unique<Impl>(std::move(scenario), impl_->metrics, impl_->checkpoints, impl_->backend);
    impl_ = std::move(next);
}
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
        std::filesystem::path snapshot;
        ppo::rl::ILearner* learner = nullptr;
        if (mode == ControlMode::Training)
        {
            auto training = impl_->backend.create_learner(
                dimensions, 2, rollout,
                impl_->checkpoints / "work" / ("mobile-" + std::to_string(impl_->environment.scenario().controlled)));
            learner = training.get();
            next = std::move(training);
        }
        else if (mode == ControlMode::Policy)
        {
            const auto directory = impl_->checkpoints / "controller-snapshots";
            std::filesystem::create_directories(directory);
            const auto serial = std::chrono::system_clock::now().time_since_epoch().count();
            snapshot = std::filesystem::absolute(directory / ("primary-" + std::to_string(serial) + ".pt"));
            std::filesystem::copy_file(checkpoint, snapshot);
            try
            {
                next = impl_->backend.load_policy(dimensions, 2, snapshot);
            }
            catch (...)
            {
                std::error_code error;
                std::filesystem::remove(snapshot, error);
                throw;
            }
        }
        else
            next = std::make_unique<ManualPolicy>();
        if (!next)
            throw std::runtime_error("backend returned no policy");
        auto session = std::make_unique<ppo::rl::EpisodeSession>(impl_->environment, *next,
                                                                 impl_->environment.scenario().max_steps, learner);
        session->reset();
        impl_->session = std::move(session);
        impl_->policy = std::move(next);
        impl_->learner = learner;
        impl_->mode = mode;
        impl_->loaded_checkpoint = checkpoint;
        impl_->primary_snapshot = std::move(snapshot);
        impl_->telemetry = {};
    }
    if (impl_->session->finished())
        impl_->session->reset();
    impl_->running = true;
    impl_->status = "Running";
}
void Application::pause()
{
    impl_->running = false;
    impl_->elapsed = 0;
    impl_->evaluation_paused = true;
    impl_->status = "Paused";
}
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
        impl_->environment.request_stop();
        impl_->session->request_stop();
        impl_->session->advance();
        impl_->record();
    }
    impl_->status = "Stopped";
}
void Application::tick(double elapsed, const std::vector<float>& manual_action, double time_scale,
                       EntityId manual_mobile)
{
    if (!std::isfinite(elapsed) || elapsed < 0 || !std::isfinite(time_scale) || time_scale <= 0)
        throw std::invalid_argument("invalid desktop clock");
    if (evaluating())
    {
        if (impl_->evaluation_paused)
            return;
        try
        {
            impl_->evaluate_tick();
        }
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
    if (!impl_->running)
        return;
    if (manual_mobile == no_entity)
        manual_mobile = impl_->environment.scenario().controlled;
    if (auto* manual = dynamic_cast<ManualPolicy*>(impl_->policy.get()))
        manual->action =
            manual_mobile == impl_->environment.scenario().controlled ? manual_action : std::vector<float>{0, 0};
    const double dt = impl_->environment.scenario().dt;
    impl_->elapsed = std::min(impl_->elapsed + elapsed * time_scale, 64 * dt);
    while (impl_->elapsed >= dt)
    {
        if (impl_->session->finished())
            impl_->session->reset();
        for (const auto& mobile : impl_->environment.scenario().mobiles)
            if (mobile.id != impl_->environment.scenario().controlled && mobile_mode(mobile.id) == ControlMode::Manual)
            {
                const auto commands = mobile.id == manual_mobile ? manual_action : std::vector<float>{0, 0};
                if (commands.size() != 2)
                    throw std::invalid_argument("invalid manual action shape");
                impl_->environment.set_manual_action(mobile.id, {commands[0], commands[1]});
            }
        impl_->session->advance();
        impl_->record();
        impl_->elapsed -= dt;
    }
}
void Application::evaluate_candidate(std::size_t episodes, EntityId evaluated_mobile)
{
    if (evaluating())
        throw std::logic_error("evaluation already running");
    if (evaluated_mobile == no_entity)
        evaluated_mobile = impl_->environment.scenario().controlled;
    impl_->environment.handle(evaluated_mobile);
    const auto control = impl_->controls.find(evaluated_mobile);
    auto* learner = evaluated_mobile == impl_->environment.scenario().controlled
                        ? impl_->learner
                        : (control == impl_->controls.end() ? nullptr : control->second.learner);
    if (!learner)
        throw std::logic_error("selected mobile must have a training policy");
    if (episodes == 0 || episodes > 10000)
        throw std::invalid_argument("invalid evaluation episode count");
    pause();
    auto metrics = ppo::rl::load_metrics(impl_->metrics);
    const auto serial = std::chrono::system_clock::now().time_since_epoch().count();
    const std::string version = "candidate-" + std::to_string(serial);
    const auto candidate = impl_->checkpoints / "candidates" / version;
    if (std::filesystem::exists(candidate))
        throw std::runtime_error("candidate already exists");
    impl_->backend.save(*learner, candidate);
    auto evaluated_scenario = impl_->environment.scenario();
    evaluated_scenario.controlled = evaluated_mobile;
    save_scenario(evaluated_scenario, candidate / "scenario.nav");
    // Serialize the parsed criteria, not a file which may change after it was read.
    const auto criteria_text = ppo::rl::metrics_text(metrics);
    std::ofstream criteria(candidate / "score.metrics");
    criteria << criteria_text;
    criteria.close();
    if (!criteria)
        throw std::runtime_error("cannot save candidate scoring criteria");
    auto environment = std::make_unique<Environment>(evaluated_scenario);
    const auto dimensions = std::get<std::vector<float>>(impl_->session->observation()).size();
    std::string controller_context;
    for (const auto& definition : impl_->environment.scenario().mobiles)
    {
        const auto mobile = definition.id;
        if (mobile == evaluated_mobile)
            continue;
        const auto found = impl_->controls.find(mobile);
        const bool primary = mobile == impl_->environment.scenario().controlled;
        const auto mode =
            primary ? impl_->mode : (found == impl_->controls.end() ? ControlMode::Manual : found->second.mode);
        if (mode == ControlMode::Manual)
            continue;
        auto* background_learner = primary ? impl_->learner : found->second.learner;
        const auto& snapshot = primary ? impl_->primary_snapshot : found->second.snapshot;
        const auto directory = candidate / "controllers" / std::to_string(mobile);
        std::filesystem::create_directories(directory);
        std::filesystem::path archive;
        if (background_learner)
        {
            impl_->backend.save(*background_learner, directory);
            archive = directory / "eval" / "model.pt";
            controller_context += "learner-snapshot " + std::to_string(mobile) + " " + version + "\n";
        }
        else
        {
            archive = directory / "model.pt";
            std::filesystem::copy_file(snapshot, archive);
            controller_context += "frozen-snapshot " + std::to_string(mobile) + " " + snapshot.generic_string() + "\n";
        }
        auto controller = impl_->backend.load_policy(dimensions, 2, archive);
        if (!controller)
            throw std::runtime_error("backend returned no evaluation controller");
        environment->set_policy(mobile, std::move(controller));
    }
    auto policy = impl_->backend.load_policy(dimensions, 2, candidate / "eval" / "model.pt");
    if (!policy)
        throw std::runtime_error("backend returned no evaluation policy");
    auto session = std::make_unique<ppo::rl::EpisodeSession>(*environment, *policy, environment->scenario().max_steps);
    session->reset({0, {}});
    const auto evaluation_context = scenario_text(evaluated_scenario) + criteria_text + "episodes " +
                                    std::to_string(episodes) + "\nseed-sequence 0..episodes-1\n" + controller_context;
    std::ofstream context(candidate / "evaluation-context.txt", std::ios::binary);
    context << evaluation_context;
    context.close();
    if (!context)
        throw std::runtime_error("cannot save candidate evaluation context");
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
    impl_->evaluation_store = impl_->checkpoints / "prod" / ("mobile-" + std::to_string(evaluated_mobile));
    impl_->status = "Evaluating frozen candidate";
}
const Environment& Application::environment() const noexcept
{
    return impl_->environment;
}
const ppo::rl::Value& Application::observation() const noexcept
{
    return impl_->session->observation();
}
const Telemetry& Application::telemetry() const noexcept
{
    return impl_->telemetry;
}
bool Application::running() const noexcept
{
    return impl_->running;
}
bool Application::evaluating() const noexcept
{
    return impl_->evaluation_session != nullptr;
}
ControlMode Application::mode() const noexcept
{
    return impl_->mode;
}
const std::string& Application::status() const noexcept
{
    return impl_->status;
}
} // namespace navigation
