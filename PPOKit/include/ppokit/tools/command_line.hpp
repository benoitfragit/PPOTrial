/**
 * @file PPOKit/include/ppokit/tools/command_line.hpp
 * @brief The standard command line of a PPOKit example: check, train, eval.
 * @details An example is a project folder (schema/project.hpp): task.json, network.json,
 * training.json, eval/, components/, checkpoints/, and its task written in C++. Its main is
 * one call:
 *
 * @code
 * int main(int argc, char** argv)
 * {
 *     return ppokit::tools::RunCommandLine<ObservationSize, ActionSize>(argc, argv, MakeTask, ExampleFolder);
 * }
 * @endcode
 *
 * @verbatim
 *   my_example check                  reads and validates every file, says what is wrong in plain words
 *   my_example train [--updates N]    trains; resumes checkpoints/latest.ppokit when it exists
 *   my_example eval [--suite FILE]    evaluates checkpoints/policy.ppokit; exit code 0 when the gates pass
 *   my_example play [--episodes N]    plays the policy and prints every episode (inference)
 * @endverbatim
 * Every command takes `--project DIR` to work on another copy of the project (the studio
 * passes it); the default is the example's own folder.
 *
 * Training uses everything training.json asks for: the PPO parameters, the probes, and the
 * curriculum. The task opts into more by having these members (all optional):
 * - `void Configure(std::vector<double> const& scenario)`: receives each episode's
 *   scenario from the curriculum (values in task.json "scenario" order);
 * - `bool Succeeded() const`: whether the episode that just ended counts as a success
 *   for promotion (otherwise: terminated episodes count);
 * - `double const* LastRewardChannels() const`: weighted value of every task.json reward
 *   channel for the last step, for per-channel reward curves;
 * - `std::vector<double> RewardWeights() const` and `void SetRewardWeights(std::vector<double> const&)`:
 *   the studio can change the weights while the run trains (Reward weights tab).
 *
 * While it trains, the run is live (ppokit/live/live_channel.hpp): the studio attaches
 * at any time, follows the journal and the rewards of the window being collected, and can
 * pause, resume, stop and change weights through --control. Older TCP and file clients opt in to --legacy-control.
 */
#pragma once

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/eval/evaluation.hpp"
#include "ppokit/eval/evaluator.hpp"
#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/graph/probes.hpp"
#include "ppokit/live/live_control.hpp"
#include "ppokit/runtime/collector.hpp"
#include "ppokit/runtime/curriculum.hpp"
#include "ppokit/runtime/reward_ledger.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/runtime/weight_request.hpp"
#include "ppokit/ipc/step_control.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ppokit::tools {

namespace detail {

/** @brief Whether the task takes curriculum scenarios. */
template<typename Task, typename = void>
struct Configurable : std::false_type
{
};
template<typename Task>
struct Configurable<Task, std::void_t<decltype(std::declval<Task&>().Configure(std::declval<std::vector<double> const&>()))>>
    : std::true_type
{
};

/** @brief Whether the task says which episodes are successes. */
template<typename Task, typename = void>
struct ReportsSuccess : std::false_type
{
};
template<typename Task>
struct ReportsSuccess<Task, std::void_t<decltype(bool(std::declval<Task const&>().Succeeded()))>> : std::true_type
{
};

/** @brief Whether the studio may change the task's reward weights. */
template<typename Task, typename = void>
struct TunableWeights : std::false_type
{
};
template<typename Task>
struct TunableWeights<Task,
    std::void_t<decltype(std::vector<double>(std::declval<Task const&>().RewardWeights())),
        decltype(std::declval<Task&>().SetRewardWeights(std::declval<std::vector<double> const&>()))>>
    : std::true_type
{
};

/**
 * @brief The example's task as the collector sees it, with the curriculum around its episodes.
 * @details Each Reset draws the next scenario and hands it to the task; each episode end
 * is recorded by the curriculum. The reward channels pass through. The scenario vector is
 * sized on the first draw: no allocation afterwards.
 */
template<typename Task>
class ScenarioTask final
{
public:
    ScenarioTask(Task& task, runtime::Curriculum* curriculum, std::uint32_t seed)
        : task_(task), curriculum_(curriculum), rng_(seed)
    {
    }

    auto Reset()
    {
        if constexpr (Configurable<Task>::value)
        {
            if (curriculum_ != nullptr)
            {
                draw_ = curriculum_->Sample(rng_, scenario_);
                task_.Configure(scenario_);
            }
        }
        return task_.Reset();
    }

    template<typename Action, typename Observation>
    auto Step(Action const& action, Observation& next)
    {
        auto const outcome = task_.Step(action, next);
        if (curriculum_ != nullptr && runtime::EndsEpisode(outcome.end))
        {
            bool success = outcome.end == runtime::StepEnd::Terminated;
            if constexpr (ReportsSuccess<Task>::value)
            {
                success = task_.Succeeded();
            }
            if (curriculum_->Record(draw_, success))
            {
                std::cout << "curriculum: promoted to stage " << curriculum_->Stage() + 1 << "/"
                          << curriculum_->Stages() << " '" << curriculum_->StageName() << "'\n";
            }
        }
        return outcome;
    }

    template<typename T = Task>
    auto LastRewardChannels() const -> decltype(std::declval<T const&>().LastRewardChannels())
    {
        return task_.LastRewardChannels();
    }

private:
    Task& task_;
    runtime::Curriculum* curriculum_;
    std::mt19937 rng_;
    std::vector<double> scenario_;
    runtime::ScenarioDraw draw_;
};

/** @brief The files of an example: its own, or those named on the command line. */
struct Files final
{
    std::filesystem::path task;
    std::filesystem::path graph;
    std::filesystem::path train;
    std::filesystem::path suite;
    std::filesystem::path model;
};

/** @brief A mistake on the command line: printed with the usage, exit code 2. */
class UsageError final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/** @brief Arguments after the subcommand: "--name value" options and "--name" flags. */
class Arguments final
{
public:
    Arguments(int argc, char** argv, std::vector<std::string> const& options, std::vector<std::string> const& flags)
    {
        for (int index = 2; index < argc; ++index)
        {
            std::string const name = argv[index];
            if (Contains(flags, name))
            {
                flags_.push_back(name);
            }
            else if (Contains(options, name))
            {
                if (index + 1 == argc)
                {
                    throw UsageError(name + " needs a value");
                }
                values_.emplace_back(name, argv[++index]);
            }
            else
            {
                throw UsageError("unknown argument '" + name + "'");
            }
        }
    }

    bool Flag(std::string const& name) const { return Contains(flags_, name); }

    std::string Text(std::string const& name, std::string const& fallback) const
    {
        for (auto const& [key, value] : values_)
        {
            if (key == name)
            {
                return value;
            }
        }
        return fallback;
    }

    /** @throws UsageError when the value is not a whole number of at least `minimum`. */
    std::uint64_t Count(std::string const& name, std::uint64_t fallback, std::uint64_t minimum) const
    {
        std::string const text = Text(name, "");
        if (text.empty())
        {
            return fallback;
        }
        std::uint64_t value = 0;
        for (char const digit : text)
        {
            if (digit < '0' || digit > '9')
            {
                throw UsageError(name + " takes a whole number, not '" + text + "'");
            }
            value = value * 10 + static_cast<std::uint64_t>(digit - '0');
        }
        if (value < minimum)
        {
            throw UsageError(name + " must be at least " + std::to_string(minimum));
        }
        return value;
    }

private:
    static bool Contains(std::vector<std::string> const& names, std::string const& name)
    {
        for (auto const& candidate : names)
        {
            if (candidate == name)
            {
                return true;
            }
        }
        return false;
    }

    std::vector<std::pair<std::string, std::string>> values_;
    std::vector<std::string> flags_;
};

inline std::vector<std::string> FileOptions(std::vector<std::string> more)
{
    more.emplace_back("--project");
    return more;
}

/** @brief The files of the project named by --project (default: the example's folder). */
inline Files ReadFiles(Arguments const& arguments, std::filesystem::path const& folder)
{
    schema::Project project;
    project.root = arguments.Text("--project", folder.string());
    if (!std::filesystem::is_directory(project.root))
    {
        throw std::runtime_error("no project folder at " + project.root.string());
    }
    Files files;
    files.task = project.Task();
    files.graph = project.Network();
    files.train = project.Training();
    files.suite = arguments.Text("--suite", (project.Evaluations() / "smoke.json").string());
    files.model = project.Checkpoints();
    return files;
}

/** @brief Task, networks and training plan, read and validated together. */
struct Setup final
{
    schema::TaskSchema task;
    schema::BlockGraphSchema graphs;
    schema::TrainingPlan plan;
    artifacts::ArtifactIdentity identity;
};

template<std::size_t ObservationSize, std::size_t ActionSize>
Setup ReadSetup(Files const& files)
{
    Setup setup;
    setup.task = schema::LoadTaskSchema(files.task);
    if (setup.task.observation.size != ObservationSize || setup.task.actionSize != ActionSize)
    {
        throw std::runtime_error(files.task.string() + " declares " + std::to_string(setup.task.observation.size) +
                                 " observation values and " + std::to_string(setup.task.actionSize) +
                                 " action values, the C++ task " + std::to_string(ObservationSize) + " and " +
                                 std::to_string(ActionSize) +
                                 ": rebuild (the layout header is generated from task.json) or fix the task");
    }
    setup.graphs = schema::LoadBlockGraph(files.graph, setup.task);
    if (std::filesystem::exists(files.train))
    {
        setup.plan = schema::LoadTrainingPlan(files.train, setup.task);
    }
    setup.identity = schema::ArtifactIdentityOf(setup.task, setup.graphs);
    return setup;
}

template<std::size_t ObservationSize, std::size_t ActionSize>
graph::GraphLearner<ObservationSize, ActionSize> MakeLearner(Setup const& setup)
{
    return graph::GraphLearner<ObservationSize, ActionSize>(schema::BuildGraph(setup.graphs.actor),
        schema::BuildGraph(setup.graphs.critic), schema::MakePolicyLayout(setup.graphs, setup.task),
        setup.plan.ppo.Settings());
}

/**
 * @brief `checkpoints/stop.request`: written by the studio's Stop button; training saves a
 *        checkpoint and ends after the current update.
 */
inline std::filesystem::path StopRequestPath(std::filesystem::path const& checkpoints)
{
    return checkpoints / "stop.request";
}

inline bool StopRequested(std::filesystem::path const& checkpoints)
{
    std::error_code error;
    return std::filesystem::exists(StopRequestPath(checkpoints), error);
}

/** @brief `checkpoints/curriculum.txt`: name of the stage reached, so that a resume continues there. */
inline std::filesystem::path StagePath(std::filesystem::path const& model)
{
    return model / "curriculum.txt";
}

inline void SaveStage(std::filesystem::path const& model, runtime::Curriculum const& curriculum)
{
    artifacts::WriteFileAtomically(
        StagePath(model), [&](std::ostream& stream) { stream << curriculum.StageName() << '\n'; },
        artifacts::AtomicWriteOptions{}, "PPOKit curriculum stage");
}

inline void RestoreStage(std::filesystem::path const& model, runtime::Curriculum& curriculum)
{
    std::ifstream stream(StagePath(model));
    std::string name;
    if (!std::getline(stream, name))
    {
        return;
    }
    if (auto const stage = curriculum.StageIndex(name))
    {
        curriculum.Restore(*stage);
        std::cout << "Curriculum continues at stage " << *stage + 1 << "/" << curriculum.Stages() << " '" << name
                  << "'.\n";
    }
    else
    {
        std::cout << "Curriculum: stage '" << name << "' is no longer in training.json, starting at the first stage.\n";
    }
}

inline int Check(Setup const& setup, Files const& files)
{
    std::string suite = "no evaluation suite (" + files.suite.string() + " is missing)";
    if (std::filesystem::exists(files.suite))
    {
        eval::EvaluationSuite const loaded = eval::LoadEvaluationSuite(files.suite);
        eval::RequireDisjointSeeds(loaded, {setup.plan.ppo.Settings().seed});
        suite = "evaluation suite '" + loaded.name + "'";
    }
    std::string curriculum = "no curriculum";
    if (setup.plan.curriculum)
    {
        curriculum = std::to_string(setup.plan.curriculum->stages.size()) + " curriculum stages";
    }
    std::cout << "All files are valid: task '" << setup.task.name << "', " << setup.graphs.actor.blocks.size()
              << " actor blocks, " << setup.graphs.critic.blocks.size() << " critic blocks, " << curriculum << ", "
              << suite << ".\n";
    return 0;
}

inline void PrintUsage(std::ostream& stream, char const* program)
{
    stream << "Usage:\n"
           << "  " << program << " check                 validate task.json, network.json, training.json and eval/\n"
           << "  " << program << " train [options]       train; resumes checkpoints/latest.ppokit when it exists\n"
           << "      --updates N      updates to run in this session (default 100)\n"
           << "      --save-every N   updates between two checkpoints (default 10)\n"
           << "      --fresh          start over instead of resuming\n"
           << "  " << program << " eval [options]        evaluate checkpoints/policy.ppokit; exit code 0 when the gates pass\n"
           << "      --suite FILE     evaluation suite (default eval/smoke.json)\n"
           << "      --policy FILE    checkpoint to evaluate (default checkpoints/policy.ppokit)\n"
           << "      --max-steps N    bound on an episode (default 100000)\n"
           << "  " << program << " play [options]        play the policy and print every episode\n"
           << "      --episodes N     episodes to play (default 5)\n"
           << "      --seed N         seed of the first episode (default 9001)\n"
           << "Every command takes --project DIR (default: the example's own folder).\n"
           << "Open the project in ppokit-studio to edit it, train it and watch it live.\n";
}

/** @brief Reward channels of the journal: those of task.json when the task reports them, else one total. */
template<typename Task>
std::vector<runtime::RunChannel> JournalChannels(Task const& task, schema::TaskSchema const& schema)
{
    std::vector<runtime::RunChannel> channels;
    if constexpr (runtime::HasRewardChannels<ScenarioTask<Task>>::value)
    {
        std::vector<double> weights;
        if constexpr (TunableWeights<Task>::value)
        {
            weights = task.RewardWeights();
        }
        for (std::size_t channel = 0; channel < schema.rewards.size(); ++channel)
        {
            auto const& reward = schema.rewards[channel];
            double const weight = weights.size() == schema.rewards.size() ? weights[channel] : reward.weight;
            channels.push_back({reward.name, reward.label, weight});
        }
    }
    else
    {
        channels.push_back({"reward", "Reward", 1.0});
    }
    return channels;
}

/**
 * @brief Reward weights in force after update `updates` in an earlier journal of this run,
 *        empty when there is none or its channels differ (studio changes survive a resume).
 */
inline std::vector<double> WeightsInForce(std::filesystem::path const& journal, std::uint64_t updates,
    std::vector<runtime::RunChannel> const& channels)
{
    std::vector<double> weights;
    try
    {
        runtime::RunLog const log = runtime::LoadRunLog(journal);
        if (log.header.channels.size() != channels.size())
        {
            return {};
        }
        for (std::size_t channel = 0; channel < channels.size(); ++channel)
        {
            if (log.header.channels[channel].name != channels[channel].name)
            {
                return {};
            }
            weights.push_back(log.header.channels[channel].weight);
        }
        for (runtime::WeightChange const& change : log.weightChanges)
        {
            if (change.fromUpdate <= updates + 1 && change.weights.size() == weights.size())
            {
                weights = change.weights;
            }
        }
    }
    catch (std::exception const&)
    {
        return {};
    }
    return weights;
}

/**
 * @brief Applies a studio request to the task's weights, or says why not.
 * @return The answer for the studio; the task, ledger and journal change only when applied.
 */
template<typename Task>
runtime::WeightAnswer ApplyWeightRequest(Task& task, runtime::WeightRequest const& request,
    runtime::RewardLedger& ledger, runtime::RunLogWriter& journal, std::uint64_t fromUpdate)
{
    std::vector<double> const previous = task.RewardWeights();
    std::vector<double> next = previous;
    for (auto const& [channel, weight] : request.weights)
    {
        next.at(channel) = weight;
    }
    try
    {
        task.SetRewardWeights(next);
    }
    catch (std::exception const& error)
    {
        return {request.id, false, std::string("refused by the task: ") + error.what()};
    }
    ledger.SetWeights(next);
    journal.WriteWeights({fromUpdate, previous, next});
    return {request.id, true, "applied from update " + std::to_string(fromUpdate)};
}

/** @brief Saves what resumes the run (training state, curriculum stage) and the policy. */
template<typename Learner>
void SaveAll(Learner& learner, std::filesystem::path const& latest, std::filesystem::path const& policy,
    artifacts::ArtifactIdentity const& identity, std::filesystem::path const& model, runtime::Curriculum const* stages)
{
    learner.SaveTraining(latest, identity);
    learner.SavePolicy(policy, identity);
    if (stages != nullptr)
    {
        SaveStage(model, *stages);
    }
}

/** @brief Options of `train`, read before any file so that a typo is reported first. */
struct TrainOptions final
{
    std::uint64_t updates = 100;
    std::uint64_t saveEvery = 10;
    bool fresh = false;
    bool legacy = false;
    std::string endpoint;
};

inline TrainOptions ReadTrainOptions(Arguments const& arguments)
{
    TrainOptions options;
    options.updates = arguments.Count("--updates", options.updates, 1);
    options.saveEvery = arguments.Count("--save-every", options.saveEvery, 1);
    options.fresh = arguments.Flag("--fresh");
    options.legacy = arguments.Flag("--legacy-control");
    options.endpoint = arguments.Text("--control", "");
    if (options.legacy && !options.endpoint.empty()) throw UsageError("select one control protocol");
    return options;
}

template<std::size_t ObservationSize, std::size_t ActionSize, typename TaskFactory>
int Train(Setup const& setup, Files const& files, TrainOptions const& options, TaskFactory& makeTask)
{
    using Learner = graph::GraphLearner<ObservationSize, ActionSize>;
    using Task = std::decay_t<decltype(makeTask(std::uint32_t{}))>;
    std::uint64_t const session = options.updates;
    std::uint64_t const saveEvery = options.saveEvery;
    core::PpoSettings const settings = setup.plan.ppo.Settings();

    Learner learner = MakeLearner<ObservationSize, ActionSize>(setup);
    std::filesystem::create_directories(files.model);
    // A stop asked of an earlier run must not end this one.
    if (options.legacy) std::filesystem::remove(StopRequestPath(files.model));
    std::filesystem::path const latest = files.model / "latest.ppokit";
    std::filesystem::path const policy = files.model / "policy.ppokit";
    std::filesystem::path const journalPath = files.model / "metrics.jsonl";
    bool const resume = !options.fresh && std::filesystem::exists(latest);
    std::uint64_t updates = 0;
    if (resume)
    {
        try
        {
            learner.LoadTraining(latest, setup.identity);
        }
        catch (artifacts::ArtifactMismatch const& mismatch)
        {
            throw std::runtime_error(std::string(mismatch.what()) +
                                     "\nTo train these networks from scratch instead, run: train --fresh");
        }
        updates = artifacts::ReadArtifactHeader(latest).updates;
        std::cout << "Resuming from " << latest.string() << " at update " << updates << ".\n";
    }

    Task task = makeTask(settings.seed);
    std::optional<runtime::Curriculum> curriculum;
    if (setup.plan.curriculum)
    {
        if constexpr (Configurable<Task>::value)
        {
            curriculum.emplace(*setup.plan.curriculum);
            if (resume)
            {
                RestoreStage(files.model, *curriculum);
            }
        }
        else
        {
            std::cout << "training.json has a curriculum, but the task has no Configure(scenario): it is ignored.\n";
        }
    }
    runtime::Curriculum* const stages = curriculum ? &*curriculum : nullptr;
    ScenarioTask<Task> episodes(task, stages, settings.seed + 1);

    runtime::RunHeader header;
    header.task = setup.task.name;
    header.channels = JournalChannels(task, setup.task);
    header.settings = runtime::PpoSettingsValues(settings);
    header.settings.emplace_back("rollout_steps", static_cast<double>(setup.plan.ppo.RolloutSteps()));
    if (setup.plan.probeEvery > 0)
    {
        header.settings.emplace_back("probe_every", static_cast<double>(setup.plan.probeEvery));
    }
    constexpr bool tunable = TunableWeights<Task>::value && runtime::HasRewardChannels<ScenarioTask<Task>>::value;
    header.acceptsWeightRequests = tunable && (options.legacy || !options.endpoint.empty());
    if constexpr (tunable)
    {
        if (resume)
        {
            // Weights changed from the studio in an earlier session stay in force.
            std::vector<double> const inForce = WeightsInForce(journalPath, updates, header.channels);
            if (!inForce.empty() && inForce != task.RewardWeights())
            {
                task.SetRewardWeights(inForce);
                for (std::size_t channel = 0; channel < inForce.size(); ++channel)
                {
                    header.channels[channel].weight = inForce[channel];
                }
                std::cout << "Reward weights changed during earlier sessions are kept.\n";
            }
        }
    }
    std::vector<double> weights;
    std::vector<std::string> names;
    for (auto const& channel : header.channels)
    {
        weights.push_back(channel.weight);
        names.push_back(channel.name);
    }
    runtime::RewardLedger ledger(weights, 1);
    runtime::RunLogWriter journal = resume
        ? runtime::RunLogWriter::Resume(journalPath, header, updates, &ledger)
        : runtime::RunLogWriter(journalPath, header);
    std::optional<runtime::WeightRequestInbox> inbox;
    std::optional<live::LiveServer> server;
    std::optional<live::LiveControl> legacy;
    std::optional<ipc::Worker> worker;
    std::optional<ipc::StepControl> step;
    if (options.legacy)
    {
        inbox.emplace(journalPath, names);
        server.emplace(files.model, journalPath, names.size());
        journal.SetSink(&*server);
        legacy.emplace(*server, ledger, updates);
    }
    else if (!options.endpoint.empty())
    {
        worker.emplace(options.endpoint, tunable);
        step.emplace(*worker);
    }

    typename Learner::Buffer buffer(setup.plan.ppo.RolloutSteps());
    runtime::RolloutCollector<ScenarioTask<Task>, Learner, typename Learner::Buffer> collector(episodes, learner);
    collector.Track(&ledger);
    if (legacy) collector.Watch(&*legacy);
    if (step) collector.Watch(&*step);
    graph::GraphProbes actorProbes(learner.Actor());
    graph::GraphProbes criticProbes(learner.Critic());

    std::uint64_t const target = updates + session;
    auto const start = std::chrono::steady_clock::now();
    std::cout << "Training '" << setup.task.name << "' up to update " << target << ".\n"
              << "Watch it live: open the project in ppokit-studio (journal " << journalPath.string() << ")\n";
    while (updates < target)
    {
        if (!collector.Collect(buffer))
        {
            // Stopped from the studio during the collection: the partial rollout is
            // dropped, the networks are those of the last update.
            if (worker) worker->Saving();
            SaveAll(learner, latest, policy, setup.identity, files.model, stages);
            if (legacy) legacy->Finish();
            if (worker) worker->Finish();
            std::cout << "Stopped on request at update " << updates << ", checkpoint saved." << std::endl;
            return 0;
        }
        bool const probed = schema::ProbesDue(setup.plan.probeEvery, updates + 1);
        if (probed)
        {
            learner.SetProbes(&actorProbes, &criticProbes);
        }
        core::UpdateStatistics const statistics = learner.Update(buffer);
        learner.SetProbes(nullptr, nullptr);
        updates = statistics.updates;

        double const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        runtime::RewardSummary const rewards = ledger.Summarize();
        journal.WriteUpdate({updates, elapsed, runtime::PpoMetrics(statistics), rewards});
        if (legacy) legacy->AtUpdate(updates, rewards.episodes);
        if (probed)
        {
            journal.WriteProbes({updates, "actor", actorProbes.Summary()});
            journal.WriteProbes({updates, "critic", criticProbes.Summary()});
        }
        std::cout << "update " << updates << "/" << target << "  episodes " << rewards.episodes << "  mean return "
                  << rewards.episodeReturnMean << "  explained variance " << statistics.explainedVariance
                  << std::endl; // flushed: the studio shows the console live

        if constexpr (tunable)
        {
            if (auto const request = inbox ? inbox->Take() : std::nullopt)
            {
                runtime::WeightAnswer const answer = ApplyWeightRequest(task, *request, ledger, journal, updates + 1);
                inbox->Answer(answer);
                std::cout << "reward weights: " << answer.message << std::endl;
            }
            if (auto const request = worker ? worker->TakeWeights() : std::nullopt)
                worker->Answer(ApplyWeightRequest(task, *request, ledger, journal, updates + 1));
        }
        if (legacy)
        {
            while (auto const request = legacy->TakeWeights())
            {
                if constexpr (tunable) server->Answer(ApplyWeightRequest(task, *request, ledger, journal, updates + 1));
                else server->Answer({request->id, false, "this task does not take weight changes"});
            }
        }
        bool const stop = (options.legacy && StopRequested(files.model)) || (legacy && legacy->StopRequested())
            || (worker && worker->Stopping());
        if (updates % saveEvery == 0 || updates == target || stop)
        {
            SaveAll(learner, latest, policy, setup.identity, files.model, stages);
        }
        if (stop)
        {
            if (options.legacy) std::filesystem::remove(StopRequestPath(files.model));
            if (legacy) legacy->Finish();
            if (worker) worker->Finish();
            std::cout << "Stopped on request at update " << updates << ", checkpoint saved." << std::endl;
            return 0;
        }
    }
    if (legacy) legacy->Finish();
    if (worker) worker->Finish();
    std::cout << "Done. " << latest.string() << " resumes training (train), " << policy.string()
              << " is the policy to evaluate (eval).\n";
    return 0;
}

/** @brief Run a complete deterministic campaign or return an empty result after a supervised stop. */
template<typename TaskFactory, typename Policy>
std::optional<eval::EvaluationReport> ControlledEvaluation(eval::EvaluationSuite const& suite, TaskFactory& task,
    Policy& policy, std::size_t limit, Arguments const& args)
{
    auto const endpoint = args.Text("--control", "");
    if (endpoint.empty()) return eval::Evaluate(suite, task, policy, limit);
    ipc::Worker worker(endpoint, false);
    ipc::StepControl watcher(worker);
    try
    {
        auto report = eval::Evaluate(suite, task, policy, limit, &watcher);
        worker.Finish();
        return report;
    }
    catch (eval::EvaluationStopped const&)
    {
        worker.Finish();
        std::cout << "Stopped on request at a simulation step." << std::endl;
        return std::nullopt;
    }
}

template<std::size_t ObservationSize, std::size_t ActionSize, typename TaskFactory>
int Evaluate(Setup const& setup, Files const& files, Arguments const& arguments, TaskFactory& makeTask)
{
    std::filesystem::path const policy = arguments.Text("--policy", (files.model / "policy.ppokit").string());
    if (!std::filesystem::exists(policy))
    {
        throw std::runtime_error("no policy at " + policy.string() + ": train first, or name one with --policy");
    }
    if (!std::filesystem::exists(files.suite))
    {
        throw std::runtime_error("no evaluation suite at " + files.suite.string() + ": write one or name it with --suite");
    }
    auto learner = MakeLearner<ObservationSize, ActionSize>(setup);
    learner.LoadPolicy(policy, setup.identity);
    eval::EvaluationSuite const suite = eval::LoadEvaluationSuite(files.suite);
    eval::RequireDisjointSeeds(suite, {setup.plan.ppo.Settings().seed});
    std::size_t const maximumSteps = static_cast<std::size_t>(arguments.Count("--max-steps", 100000, 1));
    auto const result = ControlledEvaluation(suite, makeTask, learner, maximumSteps, arguments);
    if (!result) return 0;
    auto const& report = *result;
    std::cout << report.ToJson() << '\n';
    if (report.Passed())
    {
        std::cout << "PASSED: every gate of '" << suite.name << "' holds.\n";
        return 0;
    }
    std::cout << "FAILED: a gate of '" << suite.name << "' does not hold (see \"gates\" above).\n";
    return 1;
}

/** @brief Plays the policy on a few episodes and prints each one: the inference command. */
template<std::size_t ObservationSize, std::size_t ActionSize, typename TaskFactory>
int Play(Setup const& setup, Files const& files, Arguments const& arguments, TaskFactory& makeTask)
{
    std::filesystem::path const policy = files.model / "policy.ppokit";
    if (!std::filesystem::exists(policy))
    {
        throw std::runtime_error("no policy at " + policy.string() + ": train first");
    }
    auto learner = MakeLearner<ObservationSize, ActionSize>(setup);
    learner.LoadPolicy(policy, setup.identity);
    eval::EvaluationSuite suite;
    suite.name = "play";
    std::uint64_t const episodes = arguments.Count("--episodes", 5, 1);
    std::uint64_t const firstSeed = arguments.Count("--seed", 9001, 0);
    for (std::uint64_t episode = 0; episode < episodes; ++episode)
    {
        suite.seeds.push_back(static_cast<std::uint32_t>(firstSeed + episode));
    }
    suite.episodesPerSeed = 1;
    auto const campaign = ControlledEvaluation(suite, makeTask, learner, 100000, arguments);
    if (!campaign) return 0;
    auto const& report = *campaign;
    for (eval::EpisodeResult const& result : report.episodes)
    {
        std::cout << "episode seed " << result.seed << ": return " << result.episodeReturn << ", " << result.length
                  << " steps, " << (result.terminated ? "ended by the task" : "out of time") << '\n';
    }
    return 0;
}

} // namespace detail

/**
 * @brief Runs `check`, `train`, `eval`, `play` or `help` for an example.
 * @param makeTask Callable: seed -> a task (Reset, Step; the optional members are listed in the file comment).
 * @param folder The example's project folder (used when --project is not given).
 * @return The exit code: 0 on success, 1 on failure (with the reason printed), 2 for a usage error.
 */
template<std::size_t ObservationSize, std::size_t ActionSize, typename TaskFactory>
int RunCommandLine(int argc, char** argv, TaskFactory makeTask, std::filesystem::path const& folder)
{
    std::string const command = argc > 1 ? argv[1] : "help";
    if (command == "help" || command == "--help" || command == "-h")
    {
        detail::PrintUsage(std::cout, argv[0]);
        return 0;
    }
    try
    {
        if (command == "check")
        {
            detail::Arguments const arguments(argc, argv, detail::FileOptions({"--suite"}), {});
            detail::Files const files = detail::ReadFiles(arguments, folder);
            return detail::Check(detail::ReadSetup<ObservationSize, ActionSize>(files), files);
        }
        if (command == "train")
        {
            detail::Arguments const arguments(argc, argv, detail::FileOptions({"--updates", "--save-every", "--control"}), {"--fresh", "--legacy-control"});
            detail::TrainOptions const options = detail::ReadTrainOptions(arguments);
            detail::Files const files = detail::ReadFiles(arguments, folder);
            detail::Setup const setup = detail::ReadSetup<ObservationSize, ActionSize>(files);
            return detail::Train<ObservationSize, ActionSize>(setup, files, options, makeTask);
        }
        if (command == "eval")
        {
            detail::Arguments const arguments(argc, argv, detail::FileOptions({"--suite", "--policy", "--max-steps", "--control"}), {});
            detail::Files const files = detail::ReadFiles(arguments, folder);
            detail::Setup const setup = detail::ReadSetup<ObservationSize, ActionSize>(files);
            return detail::Evaluate<ObservationSize, ActionSize>(setup, files, arguments, makeTask);
        }
        if (command == "play")
        {
            detail::Arguments const arguments(argc, argv, detail::FileOptions({"--episodes", "--seed", "--control"}), {});
            detail::Files const files = detail::ReadFiles(arguments, folder);
            detail::Setup const setup = detail::ReadSetup<ObservationSize, ActionSize>(files);
            return detail::Play<ObservationSize, ActionSize>(setup, files, arguments, makeTask);
        }
        throw detail::UsageError("unknown command '" + command + "'");
    }
    catch (detail::UsageError const& error)
    {
        std::cerr << argv[0] << ": " << error.what() << "\n\n";
        detail::PrintUsage(std::cerr, argv[0]);
        return 2;
    }
    catch (std::exception const& error)
    {
        std::cerr << argv[0] << " " << command << ": " << error.what() << '\n';
        return 1;
    }
}

} // namespace ppokit::tools
