/**
 * @file PPOKit/tools/ppokit_run.cpp
 * @brief ppokit-run: trains, evaluates or plays a project from its two DLLs.
 * @details The project's environment DLL (ppokit/environment/environment_abi.h) simulates
 * the example; its network DLL (ppokit/network/network_abi.h) decides and learns. ppokit-run
 * owns the loop between them and everything around it: the checkpoints, the run journal,
 * the live channel with the studio (pause, play, stop, weights), and the observers of
 * network.json (Visualiseur blocks), called after every step.
 *
 * @verbatim
 * ppokit-run train --project DIR [--updates N | --until-update N] [--save-every N] [--fresh | --resume] [--until-passed SUITE]
 *                  [--evaluate-every N] [--max-seconds S]
 * ppokit-run eval  --project DIR [--suite NAME] [--best | --model ROLE | --policy FILE]
 * ppokit-run play  --project DIR [--episodes N] [--best [--suite NAME] | --model ROLE | --policy FILE] [--realtime [--step-seconds S]]
 * ppokit-run transfer --project DIR --from FILE
 * @endverbatim
 *
 * transfer: a checkpoint trained in another environment (task.json "contract", e.g. the T-6
 * for the PC-7) is refused by train, eval and play; `transfer` makes it this project's
 * starting point on purpose: the project's training checkpoint (or policy, for a policy
 * artifact) becomes a copy of FILE under this project's contract, tensors unchanged, checked
 * by loading it. Nothing is overwritten.
 *
 * The libraries are named in project.json ("libraries": {"environment": ..., "network": ...})
 * and found in the project folder or among the libraries the build registered.
 *
 * Evaluation suites (eval/<suite>.json, ppokit/eval/evaluation.hpp) run on demand (eval) and, when
 * they say so ("during_training"), every few updates and after the last one: a campaign
 * plays the policy just saved, deterministically, on the suite's seeds; its metrics and
 * gates go to the journal (and live to the studio), and a passing campaign that ranks
 * above the BEST becomes the BEST (checkpoints/best/, ppokit/eval/best_store.hpp).
 * `--until-passed SUITE` ends the training at the first campaign of SUITE that passes its
 * gates and reaches its goal; without one by the last update, ppokit-run exits with code 3
 * (so does eval when a gate fails). `--evaluate-every N` replaces the rhythm of the suites
 * run during training; `--max-seconds S` ends the training at the first update boundary
 * after S seconds, as if it were the last update (campaigns included). eval and play act with
 * the project's policy, the BEST (--best), or a policy file of its own (--policy, relative to
 * the project folder, e.g. a production policy kept apart).
 *
 * Teams: when task.json allows several agents ("agents" max above 1), the environment and
 * the network DLLs work on rows of agents and the network DLL is a team learner (MAPPO):
 * it chooses every present agent's action at once and updates by itself when its rollout
 * is full. A team's learner saves only between rollouts, so a stop asked during a
 * collection takes effect at the next update.
 */
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/component/native_module.hpp"
#include "ppokit/environment/environment_library.hpp"
#include "ppokit/eval/best_store.hpp"
#include "ppokit/eval/evaluation.hpp"
#include "ppokit/ipc/control.hpp"
#include "ppokit/network/network_library.hpp"
#include "ppokit/runtime/observers.hpp"
#include "ppokit/runtime/reward_ledger.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/runtime/run_state.hpp"
#include "ppokit/runtime/writer_lock.hpp"
#include "ppokit/runtime/publication.hpp"
#include "ppokit/runtime/weight_request.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/policy_reference.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include "ppokit/artifacts/sha256.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace ppokit;

/** @brief A mistake on the command line: printed with the usage, exit code 2. */
class UsageError final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct Options final
{
    std::string command;
    fs::path project;
    std::optional<std::uint64_t> updates;
    /** @brief With --until-update: the update to reach (absolute), instead of `updates` more. */
    std::optional<std::uint64_t> untilUpdate;
    std::optional<std::uint64_t> saveEvery;
    bool fresh = false;
    bool requireResume = false;
    std::optional<std::uint64_t> episodes;
    bool realtime = false;
    double stepSeconds = 0.02;
    /** @brief Train until a campaign of this suite passes (empty: no such goal). */
    std::string untilPassed;
    /** @brief The suite of eval, or whose BEST play uses (empty: every suite / the first with a BEST). */
    std::string suite;
    /** @brief Evaluate or play the BEST policy instead of policy.ppokit. */
    bool best = false;
    /** @brief Evaluate or play this policy file (relative to the project folder) instead. */
    fs::path policy;
    std::string model;
    fs::path evaluationState;
    std::string controlEndpoint;
    /** @brief Updates between two campaigns of the suites run during training; 0 keeps theirs. */
    std::uint64_t evaluateEvery = 0;
    /** @brief Training time budget in seconds; 0 for none. */
    double maxSeconds = 0.0;
    /** @brief transfer: the checkpoint of another environment to start from. */
    fs::path from;
};

/** @brief Exit code of a training that never passed its --until-passed suite, or an evaluation that failed. */
constexpr int NotPassedExitCode = 3;
/** @brief An evaluation episode longer than this is an error: the environment must end its episodes. */
constexpr std::size_t MaximumEvaluationSteps = 10000000;

std::uint64_t ReadCount(std::string const& option, std::string const& text)
{
    std::uint64_t count = 0;
    auto const parsed = std::from_chars(text.data(), text.data() + text.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || count == 0)
        throw UsageError(option + " requires a positive 64-bit integer: '" + text + "'");
    return count;
}

Options ParseOptions(int argc, char** argv)
{
    if (argc < 2)
    {
        throw UsageError("give a command: train, eval, play or transfer");
    }
    Options options;
    options.command = argv[1];
    if (options.command != "train" && options.command != "eval" && options.command != "play"
        && options.command != "transfer")
    {
        throw UsageError("unknown command '" + options.command + "'");
    }
    for (int index = 2; index < argc; ++index)
    {
        std::string const argument = argv[index];
        bool const hasValue = index + 1 < argc;
        if (argument == "--fresh")
        {
            options.fresh = true;
        }
        else if (argument == "--resume")
        {
            options.requireResume = true;
        }
        else if (argument == "--realtime")
        {
            options.realtime = true;
        }
        else if (argument == "--best")
        {
            options.best = true;
        }
        else if (!hasValue)
        {
            throw UsageError("unknown argument or missing value: '" + argument + "'");
        }
        else if (argument == "--project")
        {
            options.project = argv[++index];
        }
        else if (argument == "--updates")
        {
            options.updates = ReadCount(argument, argv[++index]);
        }
        else if (argument == "--until-update")
        {
            options.untilUpdate = ReadCount(argument, argv[++index]);
        }
        else if (argument == "--save-every")
        {
            options.saveEvery = ReadCount(argument, argv[++index]);
        }
        else if (argument == "--episodes")
        {
            options.episodes = ReadCount(argument, argv[++index]);
        }
        else if (argument == "--step-seconds")
        {
            options.stepSeconds = std::stod(argv[++index]);
        }
        else if (argument == "--until-passed")
        {
            options.untilPassed = argv[++index];
        }
        else if (argument == "--suite")
        {
            options.suite = argv[++index];
        }
        else if (argument == "--control")
        {
            options.controlEndpoint = argv[++index];
        }
        else if (argument == "--environment-state")
        {
            options.evaluationState = argv[++index];
        }
        else if (argument == "--model")
        {
            options.model = argv[++index];
            if (options.model.empty()) throw UsageError("--model requires a role");
        }
        else if (argument == "--policy")
        {
            options.policy = argv[++index];
        }
        else if (argument == "--evaluate-every")
        {
            options.evaluateEvery = ReadCount(argument, argv[++index]);
        }
        else if (argument == "--max-seconds")
        {
            options.maxSeconds = static_cast<double>(ReadCount(argument, argv[++index]));
        }
        else if (argument == "--from")
        {
            options.from = argv[++index];
        }
        else
        {
            throw UsageError("unknown argument '" + argument + "'");
        }
    }
    if (options.project.empty())
    {
        throw UsageError("--project DIR is required");
    }
    if ((options.fresh || options.requireResume) && options.command != "train")
        throw UsageError("--fresh and --resume apply only to train");
    if (options.fresh && options.requireResume) throw UsageError("choose one of --fresh and --resume");
    if (options.untilUpdate && options.updates)
    {
        throw UsageError("--updates and --until-update both say when to stop: give one");
    }
    if (!options.evaluationState.empty() && options.command != "eval" && options.command != "play")
        throw UsageError("--environment-state is an explicit context for play or eval");
    auto const selections = static_cast<int>(options.best) + !options.policy.empty() + !options.model.empty();
    if (selections > 1) throw UsageError("select one of --best, --policy and --model");
    if (selections && options.command != "play" && options.command != "eval")
        throw UsageError("model selection applies to play and eval");
    if (options.model == "BEST") options.best = true;
    if ((options.command == "transfer") != !options.from.empty())
    {
        throw UsageError("transfer takes --from FILE, and only transfer does");
    }
    if (!std::isfinite(options.stepSeconds) || options.stepSeconds <= 0.0 || options.stepSeconds > 86400.0)
        throw UsageError("--step-seconds must be finite and in (0, 86400]");
    return options;
}

void PrintUsage(std::ostream& stream)
{
    stream << "ppokit-run train --project DIR [--updates N | --until-update N] [--save-every N] [--fresh | --resume] [--until-passed SUITE]\n"
              "                 [--evaluate-every N] [--max-seconds S]\n"
              "ppokit-run eval  --project DIR [--suite NAME] [--best | --model ROLE | --policy FILE] [--environment-state FILE]\n"
              "ppokit-run play  --project DIR [--episodes N] [--best [--suite NAME] | --model ROLE | --policy FILE]\n"
              "                 [--realtime [--step-seconds S]]\n"
              "ppokit-run transfer --project DIR --from FILE\n"
              "train, eval, play: [--control PORT/SESSION] from the Studio\n";
}

/** @brief A project, its task and observers, and its two libraries, checked together. */
using RunSetupNetwork = std::shared_ptr<network::NetworkLibrary const>;

struct RunSetup final
{
    schema::Project project;
    schema::TaskSchema task;
    schema::BlockGraphSchema graphs;
    core::PpoSettings settings;
    std::optional<PpokitRunProfileV1> profile;
    std::shared_ptr<environment::EnvironmentLibrary const> environment;
    std::shared_ptr<network::NetworkLibrary const> network;
    /** @brief Suites of eval/, in file-name order. */
    std::vector<eval::EvaluationSuite> suites;
    std::vector<schema::PolicyReference> references;
};

/** @brief The suites of the project's eval/ folder, read with the task's episode metrics. */
std::vector<eval::EvaluationSuite> LoadSuites(schema::Project const& project, schema::TaskSchema const& task)
{
    std::vector<fs::path> files;
    if (fs::is_directory(project.Evaluations()))
    {
        for (fs::directory_entry const& entry : fs::directory_iterator(project.Evaluations()))
        {
            if (entry.path().extension() == ".json")
            {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    std::vector<eval::EvaluationSuite> suites;
    for (fs::path const& file : files)
    {
        suites.push_back(eval::LoadEvaluationSuite(file, task.metrics));
    }
    return suites;
}

/** @brief The suite named `name`, or null. */
eval::EvaluationSuite const* FindSuite(std::vector<eval::EvaluationSuite> const& suites, std::string const& name)
{
    for (eval::EvaluationSuite const& suite : suites)
    {
        if (suite.name == name)
        {
            return &suite;
        }
    }
    return nullptr;
}

/** @brief The library `name` of the project, or a message saying where it was looked for. */
fs::path RequireLibrary(schema::Project const& project, std::string const& name, char const* role)
{
    if (name.empty())
    {
        throw std::runtime_error(project.Manifest().string() + ": \"libraries\" has no " + role
            + " (e.g. \"libraries\": {\"environment\": \"reach_environment\", \"network\": \"reach_network\"})");
    }
    fs::path const found = schema::FindLibrary(project, name, schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
    if (found.empty())
    {
        throw std::runtime_error(std::string(role) + " library '" + name
            + "' not found in the project folder nor among the libraries of the build");
    }
    return found;
}

/** @brief Resolve the project policy and suite independently, as in the Studio.
 * Explicit CLI selections override their respective defaults. Pacing, episode counts
 * and budgets remain opt-in on the CLI.
 */
void ApplyProjectPolicy(schema::Project const& project, Options& options)
{
    if (options.command != "eval" && options.command != "play")
        return;
    auto const mode = options.command == "eval" ? schema::ProjectCommand::Evaluate : schema::ProjectCommand::Inference;
    auto const selection = schema::DeclaredPolicy(project, mode);
    if (options.suite.empty())
    {
        options.suite = selection.suite;
    }
    if (!options.best && options.model.empty() && options.policy.empty())
    {
        options.model = selection.role;
        options.policy = selection.file;
        options.best = options.model == "BEST";
    }
}

RunSetup Prepare(Options& options)
{
    RunSetup setup;
    setup.project = schema::LoadProject(options.project);
    ApplyProjectPolicy(setup.project, options);
    // Component libraries of network.json (the Visualiseurs) may be named like the project's libraries.
    component::SetKnownModules(schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
    setup.task = schema::LoadTaskSchema(setup.project.Task());
    // A team's networks may live entirely in their DLL: network.json then only adds observers, if any.
    if (fs::exists(setup.project.Network()))
    {
        setup.graphs = schema::LoadBlockGraph(setup.project.Network(), setup.task);
    }
    if (fs::exists(setup.project.Training()))
    {
        setup.settings = schema::LoadTrainingPlan(setup.project.Training(), setup.task).ppo.Settings();
    }
    setup.environment = environment::EnvironmentLibrary::Open(
        RequireLibrary(setup.project, setup.project.environmentLibrary, "environment").string(), {}, setup.project.environmentRunProfile);
    setup.network = network::NetworkLibrary::Open(
        RequireLibrary(setup.project, setup.project.networkLibrary, "network").string(), {}, &setup.task);
    PpokitEnvironmentInfo const& world = setup.environment->Info();
    PpokitNetworkInfo const& networks = setup.network->Info();
    std::size_t const worldSize = setup.task.world ? setup.task.world->size : 0;
    bool const fits = world.task && networks.task && setup.task.name == world.task && setup.task.name == networks.task
        && world.observationSize == setup.task.observation.size && world.actionSize == setup.task.actionSize
        && world.rewardChannels == setup.task.rewards.size() && world.worldSize == worldSize
        && networks.observationSize == world.observationSize && networks.actionSize == world.actionSize
        && world.metricCount == setup.task.metrics.size() && world.agentCapacity == setup.task.maximumAgents
        && networks.agentCapacity == world.agentCapacity;
    if (!fits)
    {
        throw std::runtime_error("the environment (" + std::string(world.task ? world.task : "<unnamed>") + "), the networks ("
            + std::string(networks.task ? networks.task : "<unnamed>") + ") and task.json do not have the same sizes or agents: rebuild the libraries");
    }
    auto const taskContract = schema::SemanticHash(setup.task);
    setup.environment->VerifyTaskContract(taskContract);
    setup.network->VerifyTaskContract(taskContract);
    if (setup.project.environmentRunProfile)
        setup.profile = setup.environment->RunProfile(setup.project.root, 0);
    setup.references = schema::LoadPolicyReferences(setup.project.root);
    setup.suites = LoadSuites(setup.project, setup.task);
    for (eval::EvaluationSuite& suite : setup.suites)
    {
        if (options.evaluateEvery != 0 && suite.everyUpdates != 0)
        {
            suite.everyUpdates = static_cast<std::size_t>(options.evaluateEvery);
        }
    }
    for (std::string const* named : {&options.untilPassed, &options.suite})
    {
        if (!named->empty() && FindSuite(setup.suites, *named) == nullptr)
        {
            throw UsageError("no evaluation suite '" + *named + "' in " + setup.project.Evaluations().string());
        }
    }
    if (!options.untilPassed.empty() && FindSuite(setup.suites, options.untilPassed)->everyUpdates == 0)
    {
        throw UsageError("suite '" + options.untilPassed
            + "' does not run during training: give it \"during_training\": {\"every_updates\": N}");
    }
    return setup;
}

/**
 * @brief The buffers of the run loop, agentCapacity rows each: the state the agents act on
 *        (`current`), and what the step returns.
 */
struct StepSignals final
{
    std::size_t agents;
    std::size_t channelCount;
    std::vector<float> current;
    std::vector<std::uint8_t> currentAvailable;
    std::vector<float> observation;
    std::vector<std::uint8_t> available;
    std::vector<float> world;
    std::vector<float> action;
    std::vector<double> rewards;
    std::vector<double> channels;
    /** @brief The channels in float, as the observers read them. */
    std::vector<float> channelValues;

    explicit StepSignals(PpokitEnvironmentInfo const& info)
        : agents(info.agentCapacity), channelCount(info.rewardChannels),
          current(agents * info.observationSize), currentAvailable(agents),
          observation(agents * info.observationSize), available(agents), world(agents * info.worldSize),
          action(agents * info.actionSize), rewards(agents), channels(agents * info.rewardChannels),
          channelValues(agents * info.rewardChannels)
    {
    }

    float* World() noexcept { return world.empty() ? nullptr : world.data(); }

    /** @brief Starts an episode into `current`. */
    void Reset(environment::Environment& environment)
    {
        environment.Reset(current.data(), currentAvailable.data(), World());
    }

    /**
     * @brief Publish the physical step to observers, then prepare the state used by the next decision.
     * A truncation bootstraps on the final observation returned by Step. Only a continuing episode
     * has a BeginStep, whose observation and availability must also be the learner's bootstrap inputs.
     */
    runtime::StepEnd Step(environment::Environment& environment, runtime::ObserverSet& observers)
    {
        auto const end = environment.Step(action.data(), observation.data(), available.data(), World(), rewards.data(),
            channels.data());
        observers.Notify(Observed());
        if (!runtime::EndsEpisode(end))
        {
            environment.BeginStep(observation.data(), available.data(), World());
        }
        return end;
    }

    /** @brief The step's reward for the team: the mean over the agents that acted (the reward of one agent). */
    double TeamReward() const noexcept
    {
        double sum = 0.0;
        std::size_t acted = 0;
        for (std::size_t agent = 0; agent < agents; ++agent)
        {
            if (currentAvailable[agent] != 0)
            {
                sum += rewards[agent];
                ++acted;
            }
        }
        return acted == 0 ? 0.0 : sum / static_cast<double>(acted);
    }

    /** @brief Once the learner and reward ledger have consumed the acting rows, rotate the decision buffers. */
    void Advance() noexcept
    {
        current.swap(observation);
        currentAvailable.swap(available);
    }

    /** @brief The step as the observers see it. */
    runtime::ObservedStep Observed() noexcept
    {
        std::copy(channels.begin(), channels.end(), channelValues.begin());
        runtime::ObservedStep step;
        step.agents = agents;
        step.observation = observation.data();
        step.world = world.empty() ? nullptr : world.data();
        step.actions = action.data();
        step.rewards = channelValues.data();
        return step;
    }
};

/** @brief Deterministic actions of a saved policy, for a single agent or a team. */
class Actor
{
public:
    virtual ~Actor() = default;
    /** @brief Actions of the present agents of `signals.current` into `signals.action`. */
    virtual void Act(StepSignals& signals) = 0;
};

class SingleActor final : public Actor
{
public:
    SingleActor(RunSetupNetwork const& network, fs::path const& project, fs::path const& policyFile)
        : policy_(network, project, policyFile)
    {
    }
    void Act(StepSignals& signals) override { policy_.Act(signals.current.data(), signals.action.data(), true); }

private:
    network::Policy policy_;
};

class TeamActor final : public Actor
{
public:
    TeamActor(RunSetupNetwork const& network, fs::path const& project, fs::path const& policyFile)
        : policy_(network, project, policyFile)
    {
    }
    void Act(StepSignals& signals) override
    {
        policy_.Act(signals.current.data(), signals.currentAvailable.data(), signals.action.data());
    }

private:
    network::TeamPolicy policy_;
};

/** @brief Open the actor and verify that its file still matches the selected generation. */
std::unique_ptr<Actor> OpenActor(RunSetup const& setup, fs::path const& file, std::string const& digest)
{
    std::unique_ptr<Actor> actor;
    if (setup.network->Info().agentCapacity > 1)
        actor = std::make_unique<TeamActor>(setup.network, setup.project.root, file);
    else
        actor = std::make_unique<SingleActor>(setup.network, setup.project.root, file);
    if (artifacts::Sha256OfFile(file) != digest)
        throw std::runtime_error("policy changed while opening: " + file.string());
    return actor;
}

/** @brief Weights in force at update `updates` of an earlier journal of this run, if any. */
std::vector<double> WeightsInForce(fs::path const& journal, std::uint64_t updates, std::size_t channels)
{
    if (!fs::exists(journal))
    {
        return {};
    }
    runtime::RunLog const log = runtime::LoadRunLog(journal);
    if (log.header.channels.size() != channels)
    {
        return {};
    }
    std::vector<double> weights;
    for (runtime::RunChannel const& channel : log.header.channels)
    {
        weights.push_back(channel.weight);
    }
    for (runtime::WeightChange const& change : log.weightChanges)
    {
        if (change.fromUpdate <= updates + 1 && change.weights.size() == channels)
        {
            weights = change.weights;
        }
    }
    return weights;
}

/** @brief The studio's weights, applied to the environment, the ledger and the journal. */
runtime::WeightAnswer ApplyWeights(runtime::WeightRequest const& request, environment::Environment& environment,
    runtime::RewardLedger& ledger, runtime::RunLogWriter& journal, std::uint64_t fromUpdate)
{
    std::vector<double> const previous = ledger.Weights();
    std::vector<double> next = previous;
    for (auto const& [channel, weight] : request.weights)
    {
        if (channel >= next.size() || !std::isfinite(weight))
        {
            return {request.id, false, "Refused: unknown channel or non-finite weight"};
        }
        next[channel] = weight;
    }
    try
    {
        environment.SetWeights(next.data());
    }
    catch (std::exception const& error)
    {
        return {request.id, false, std::string("Refused by the environment: ") + error.what()};
    }
    ledger.SetWeights(next);
    journal.WriteWeights({fromUpdate, previous, next});
    return {request.id, true, "Applied from update " + std::to_string(fromUpdate)};
}

/** @brief A single agent's update, in the journal's names. */
runtime::NamedValues Metrics(PpokitUpdateStatistics const& statistics)
{
    core::UpdateStatistics converted;
    converted.updates = statistics.updates;
    converted.policyLoss = statistics.policyLoss;
    converted.valueLoss = statistics.valueLoss;
    converted.entropy = statistics.entropy;
    converted.approximateKl = statistics.approximateKl;
    converted.clipFraction = statistics.clipFraction;
    converted.explainedVariance = statistics.explainedVariance;
    converted.actorKlSkips = static_cast<std::size_t>(statistics.actorKlSkips);
    return runtime::PpoMetrics(converted);
}

/** @brief A team learner's update, in the journal's names (those of an in-process team learner). */
runtime::NamedValues TeamMetrics(PpokitUpdateStatistics const& statistics)
{
    runtime::TeamUpdateStatistics converted;
    converted.update = statistics.updates;
    converted.policyLoss = static_cast<float>(statistics.policyLoss);
    converted.valueLoss = static_cast<float>(statistics.valueLoss);
    converted.entropy = static_cast<float>(statistics.entropy);
    converted.approximateKl = static_cast<float>(statistics.approximateKl);
    converted.clipFraction = static_cast<float>(statistics.clipFraction);
    converted.explainedVariance = static_cast<float>(statistics.explainedVariance);
    converted.actorKlSkips = static_cast<std::size_t>(statistics.actorKlSkips);
    converted.advantageAbsMean = static_cast<float>(statistics.advantageAbsMean);
    converted.actionStdMean = static_cast<float>(statistics.actionStdMean);
    converted.learnableRows = static_cast<std::size_t>(statistics.learnableRows);
    converted.criticRows = static_cast<std::size_t>(statistics.criticRows);
    return runtime::PpoMetrics(converted);
}

/** @brief A team learner's settings, in the journal's names. */
runtime::NamedValues TeamSettingsValues(PpokitTeamSettings const& settings)
{
    runtime::TeamPpoSettings converted;
    converted.gamma = static_cast<float>(settings.gamma);
    converted.lambda = static_cast<float>(settings.lambda);
    converted.clip = static_cast<float>(settings.clip);
    converted.entropyCoefficient = static_cast<float>(settings.entropyCoefficient);
    converted.epochs = settings.epochs;
    converted.minibatch = settings.minibatch;
    return runtime::PpoSettingsValues(converted);
}

/** @brief Probes of an update grouped by network, as `probes` events (actor first). */
std::vector<runtime::ProbeRecord> ProbeRecords(std::uint64_t update, std::vector<PpokitBlockProbe> const& probes)
{
    std::vector<runtime::ProbeRecord> records;
    for (PpokitBlockProbe const& probe : probes)
    {
        if (records.empty() || records.back().network != probe.network)
        {
            records.push_back({update, probe.network, {}});
        }
        runtime::BlockProbe block;
        block.block = probe.block;
        block.type = probe.type;
        block.samples = static_cast<std::size_t>(probe.samples);
        block.mean = probe.mean;
        block.deviation = probe.deviation;
        block.minimum = probe.minimum;
        block.maximum = probe.maximum;
        block.deadShare = probe.deadShare;
        block.saturatedShare = probe.saturatedShare;
        block.hasGradient = probe.hasGradient != 0;
        block.gradientNorm = probe.gradientNorm;
        records.back().blocks.push_back(std::move(block));
    }
    return records;
}

/** @brief How one played episode went. */
struct PlayedEpisode final
{
    double episodeReturn = 0.0;
    std::size_t steps = 0;
    runtime::StepEnd end = runtime::StepEnd::Running;
};

/** @brief Pacing of a played episode: as fast as possible, or one step every `stepSeconds`. */
struct Pace final
{
    bool realtime = false;
    double stepSeconds = 0.0;
};

/** @brief Safe cancellation of play or a campaign. Partial episodes are never reported or promoted. */
struct RunStopped {};

void CheckPlaying(ipc::Worker* control)
{
    if (control && !control->Continue()) throw RunStopped{};
}

/** @brief Keep a stopped or paused real-time simulation responsive even with a long step interval. */
void PaceStep(Pace const& pace, ipc::Worker* control)
{
    if (!pace.realtime) return;
    auto const duration = std::chrono::duration<double>(pace.stepSeconds);
    if (!control) { std::this_thread::sleep_for(duration); return; }
    auto const end = std::chrono::steady_clock::now() + duration;
    auto now = std::chrono::steady_clock::now();
    while (now < end)
    {
        CheckPlaying(control);
        std::this_thread::sleep_until(std::min<decltype(end)>(end, now + std::chrono::milliseconds(10)));
        now = std::chrono::steady_clock::now();
    }
}

/**
 * @brief Plays one episode from `signals.current` (the first state) with the policy's
 *        deterministic actions; the observers see every step.
 * @throws std::runtime_error when the episode does not end within MaximumEvaluationSteps.
 */
PlayedEpisode PlayEpisode(environment::Environment& environment, Actor& actor, StepSignals& signals,
    runtime::ObserverSet& observers, Pace const& pace, ipc::Worker* control)
{
    PlayedEpisode played;
    while (!runtime::EndsEpisode(played.end))
    {
        CheckPlaying(control);
        if (played.steps == MaximumEvaluationSteps)
        {
            throw std::runtime_error("an episode exceeded " + std::to_string(MaximumEvaluationSteps)
                + " steps: the environment must terminate or truncate its episodes");
        }
        actor.Act(signals);
        played.end = signals.Step(environment, observers);
        played.episodeReturn += signals.TeamReward();
        ++played.steps;
        if (!runtime::EndsEpisode(played.end))
        {
            signals.Advance();
        }
        PaceStep(pace, control);
    }
    return played;
}

/** @brief Apply weight overrides only after validating the full channel vector. */
std::vector<double> EffectiveWeights(environment::Environment& environment, std::size_t channels,
    std::vector<double> const& requested)
{
    std::vector<double> weights(channels);
    environment.Weights(weights.data());
    if (requested.empty() || requested == weights) return weights;
    if (requested.size() != channels || !std::all_of(requested.begin(), requested.end(),
            [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("evaluation reward vector is incompatible with the task");
    environment.SetWeights(requested.data());
    environment.Weights(weights.data());
    return weights;
}

/**
 * @brief One campaign of `suite`: for each seed, an environment created with it; each episode
 *        starts with reset_evaluation (its place in the campaign) and reports its metrics.
 * @param trainingState The environment's training state the campaign is played at (the
 *        training run's, e.g. its curriculum level); nothing: as the environment starts.
 */
eval::EvaluationReport RunCampaign(RunSetup const& setup, eval::EvaluationSuite const& suite, Actor& actor,
    runtime::ObserverSet& observers, std::optional<std::string> const& trainingState,
    std::vector<double> const& weights, ipc::Worker* control, nlohmann::json* context = nullptr)
{
    eval::EvaluationReport report;
    report.suite = suite.name;
    StepSignals signals(setup.environment->Info());
    std::vector<double> metrics(setup.task.metrics.size());
    auto const total = static_cast<std::uint32_t>(suite.Episodes());
    std::uint32_t index = 0;
    std::unique_ptr<environment::Environment> instance;
    for (std::uint32_t const seed : suite.seeds)
    {
        CheckPlaying(control);
        if (!instance || !suite.campaignEnvironment)
        {
            instance = std::make_unique<environment::Environment>(setup.environment, setup.project.root, seed);
            instance->SetLearning(false);
            if (trainingState) instance->RestoreTrainingState(*trainingState);
            auto const effective = EffectiveWeights(*instance, setup.task.rewards.size(), weights);
            if (context && index == 0)
            {
                (*context)["effective_environment_state"] = instance->TrainingState();
                (*context)["reward_weights"] = effective;
            }
        }
        auto& environment = *instance;
        for (std::size_t repeat = 0; repeat < suite.episodesPerSeed; ++repeat)
        {
            environment.ResetEvaluation(
                seed, index, total, signals.current.data(), signals.currentAvailable.data(), signals.World());
            PlayedEpisode const played = PlayEpisode(environment, actor, signals, observers, Pace{}, control);
            eval::EpisodeResult result;
            result.seed = seed;
            result.episodeReturn = played.episodeReturn;
            result.length = played.steps;
            result.terminated = played.end == runtime::StepEnd::Terminated;
            if (!metrics.empty())
            {
                environment.EpisodeMetrics(metrics.data());
                result.metrics = metrics;
            }
            report.episodes.push_back(std::move(result));
            ++index;
        }
    }
    eval::Summarize(suite, report);
    return report;
}

/** @brief Gates or goal conditions as the journal writes them. */
std::vector<runtime::EvaluationGate> ToConditions(std::vector<eval::GateResult> const& results)
{
    std::vector<runtime::EvaluationGate> conditions;
    for (eval::GateResult const& result : results)
    {
        char const* comparison = result.gate.comparison == eval::Comparison::GreaterOrEqual ? ">=" : "<=";
        conditions.push_back({result.gate.metric, comparison, result.gate.threshold, result.value, result.passed});
    }
    return conditions;
}

/** @brief The campaign as a journal event: built-in metrics, then the aggregates. */
runtime::EvaluationRecord ToRecord(std::uint64_t update, eval::EvaluationReport const& report, bool promoted)
{
    runtime::EvaluationRecord record;
    record.update = update;
    record.suite = report.suite;
    record.episodes = report.episodes.size();
    record.passed = report.Passed();
    record.promoted = promoted;
    for (std::string const& name : eval::BuiltInMetrics())
    {
        record.metrics.emplace_back(name, report.Metric(name));
    }
    for (eval::NamedValue const& aggregate : report.aggregates)
    {
        record.metrics.emplace_back(aggregate.name, aggregate.value);
    }
    record.goalReached = report.GoalReached();
    record.gates = ToConditions(report.gates);
    record.goal = ToConditions(report.goal);
    return record;
}

/** @brief One line per campaign: its verdict, then each gate with its value. */
void PrintCampaign(eval::EvaluationReport const& report, std::optional<std::uint64_t> update, bool promoted)
{
    std::cout << "evaluation '" << report.suite << "'";
    if (update)
    {
        std::cout << " at update " << *update;
    }
    std::cout << ": " << report.episodes.size() << " episodes, " << (report.Passed() ? "passed" : "not passed");
    if (promoted)
    {
        std::cout << ", new BEST";
    }
    if (!report.goal.empty())
    {
        std::cout << (report.GoalReached() ? ", goal reached" : ", goal not reached");
    }
    for (std::vector<eval::GateResult> const* conditions : {&report.gates, &report.goal})
    {
        for (eval::GateResult const& gate : *conditions)
        {
            std::cout << "  [" << gate.gate.metric << " " << gate.value << (gate.passed ? " ok" : " not met") << "]";
        }
    }
    std::cout << std::endl;
}

/** @brief Whether a suite run during training has a campaign at this update (and always after the last). */
bool CampaignDue(eval::EvaluationSuite const& suite, std::uint64_t updates, bool last) noexcept
{
    return suite.everyUpdates != 0 && (last || updates % suite.everyUpdates == 0);
}

bool AnyCampaignDue(std::vector<eval::EvaluationSuite> const& suites, std::uint64_t updates, bool last) noexcept
{
    for (eval::EvaluationSuite const& suite : suites)
    {
        if (CampaignDue(suite, updates, last))
        {
            return true;
        }
    }
    return false;
}

/** @brief A campaign's result for the training environment's curriculum (see the environment ABI). */
std::string CampaignProgress(std::uint64_t updates, eval::EvaluationReport const& report)
{
    nlohmann::json aggregates = nlohmann::json::object();
    for (std::string const& name : eval::BuiltInMetrics())
    {
        aggregates[name] = report.Metric(name);
    }
    for (eval::NamedValue const& aggregate : report.aggregates)
    {
        aggregates[aggregate.name] = aggregate.value;
    }
    nlohmann::json const progress{{"suite", report.suite}, {"update", updates}, {"passed", report.Passed()},
        {"aggregates", aggregates}};
    return progress.dump();
}

/** @brief Independent reasons to finish after the scheduled campaigns of an update. */
struct CampaignOutcome final
{
    bool goalReached = false;
    bool curriculumEnded = false;
};

/**
 * @brief Evaluate the just-published generation with its own actor, environment state and weights,
 *        and tell the training environment each result (its curriculum may pass a stage on it).
 * @return Whether the requested suite or the environment's curriculum ended training.
 */
CampaignOutcome RunDueCampaigns(RunSetup const& setup, Options const& options, std::uint64_t updates, bool last,
    environment::Environment& training, runtime::ObserverSet& observers, runtime::RunLogWriter& journal,
    ipc::Worker* control)
{
    auto const generation = runtime::VerifyPublication(setup.project);
    if (!generation || generation->update != updates || !generation->rewardWeights)
        throw std::runtime_error("training campaign requires the complete publication of update " + std::to_string(updates));
    auto const file = setup.project.TrainingPolicy();
    CampaignOutcome outcome;
    std::string publishedState = generation->state;
    for (eval::EvaluationSuite const& suite : setup.suites)
    {
        if (!CampaignDue(suite, updates, last))
        {
            continue;
        }
        auto const actor = OpenActor(setup, file, generation->policyDigest);
        auto report = RunCampaign(setup, suite, *actor, observers, generation->state,
            *generation->rewardWeights, control);
        report.policySha256 = generation->policyDigest;
        bool const promoted = !suite.best.empty()
            && eval::OfferBest(setup.project.Checkpoints(), suite, updates, file, report, generation->state,
                *generation->rewardWeights);
        journal.WriteEvaluation(ToRecord(updates, report, promoted));
        PrintCampaign(report, updates, promoted);
        // Every due suite observes the original evaluation context. The training state,
        // however, must include each completed campaign even if Stop interrupts a later one.
        if (training.TrainingProgress(updates, CampaignProgress(updates, report)))
        {
            outcome.curriculumEnded = true;
        }
        std::string const state = training.TrainingState();
        if (state != publishedState)
        {
            // Reuse the published artifact bytes: a curriculum transition does not update weights.
            runtime::PublishGeneration(setup.project, updates, state, *generation->rewardWeights,
                [&](fs::path const& learner, fs::path const& policy) {
                    fs::copy_file(setup.project.LatestCheckpoint(), learner);
                    fs::copy_file(file, policy);
                    if (artifacts::Sha256OfFile(learner) != generation->trainingDigest
                        || artifacts::Sha256OfFile(policy) != generation->policyDigest)
                    {
                        throw std::runtime_error("published artifacts changed during curriculum state publication");
                    }
                });
            publishedState = state;
        }
        outcome.goalReached = outcome.goalReached || (suite.name == options.untilPassed && report.GoalReached());
    }
    return outcome;
}

/**
 * @brief What learns in a training run: a single-agent learner (decide, append, update when
 *        full) or a team learner (act, observe; it updates by itself).
 */
class Trainee
{
public:
    virtual ~Trainee() = default;
    /** @brief One step from `signals.current`: actions, the environment's step, the record. */
    virtual runtime::StepEnd Collect(environment::Environment& environment, StepSignals& signals,
        runtime::ObserverSet& observers) = 0;
    /** @brief Whether the rollout is complete: time for Update. */
    virtual bool Ready() const = 0;
    /** @brief The update (made now, or by the team learner at its last step). */
    virtual PpokitUpdateStatistics Update() = 0;
    /** @brief Write both members of a completed update into the unpublished staging directory. */
    virtual void Save(fs::path const& training, fs::path const& policy) = 0;
    virtual void Load(fs::path const& trainingFile) = 0;
    virtual std::uint64_t Updates() const = 0;
    /** @brief The PPO settings of the journal header (`plan`: training.json's). */
    virtual runtime::NamedValues SettingsValues(core::PpoSettings const& plan) const = 0;
    /** @brief The update's metrics, in the journal's names. */
    virtual runtime::NamedValues UpdateMetrics(PpokitUpdateStatistics const& statistics) const = 0;
    /** @brief Probes of the last update, when the learner made some. */
    virtual std::vector<runtime::ProbeRecord> UpdateProbes(std::uint64_t update) const = 0;
};

class SingleTrainee final : public Trainee
{
public:
    SingleTrainee(RunSetupNetwork const& network, fs::path const& project) : learner_(network, project) {}

    runtime::StepEnd Collect(environment::Environment& environment, StepSignals& signals,
        runtime::ObserverSet& observers) override
    {
        // Decide on the current observation, step, and store the transition as the
        // in-process collector does.
        double logProbability = 0.0;
        double value = 0.0;
        learner_.Decide(signals.current.data(), signals.action.data(), logProbability, value);
        auto const end = signals.Step(environment, observers);
        PpokitTransition transition{};
        transition.observation = signals.current.data();
        transition.action = signals.action.data();
        transition.logProbability = logProbability;
        transition.value = value;
        transition.reward = signals.rewards[0];
        transition.end = static_cast<std::uint32_t>(end);
        if (end != runtime::StepEnd::Terminated)
        {
            transition.nextValue = learner_.Value(signals.observation.data());
        }
        learner_.Append(transition);
        return end;
    }
    bool Ready() const override { return learner_.Full(); }
    PpokitUpdateStatistics Update() override { return learner_.Update(); }
    void Save(fs::path const& training, fs::path const& policy) override { learner_.Save(training, policy); }
    void Load(fs::path const& trainingFile) override { learner_.Load(trainingFile); }
    std::uint64_t Updates() const override { return learner_.Updates(); }
    runtime::NamedValues SettingsValues(core::PpoSettings const& plan) const override
    {
        return runtime::PpoSettingsValues(plan);
    }
    runtime::NamedValues UpdateMetrics(PpokitUpdateStatistics const& statistics) const override
    {
        return Metrics(statistics);
    }
    std::vector<runtime::ProbeRecord> UpdateProbes(std::uint64_t /*update*/) const override { return {}; }

private:
    network::Learner learner_;
};

class TeamTrainee final : public Trainee
{
public:
    TeamTrainee(RunSetupNetwork const& network, fs::path const& project)
        : learner_(network, project), graphBackend_(network->Info().backend != nullptr
              && std::string(network->Info().backend) == PPOKIT_GRAPH_DYNAMIC_BACKEND) {}

    runtime::StepEnd Collect(environment::Environment& environment, StepSignals& signals,
        runtime::ObserverSet& observers) override
    {
        learner_.Act(signals.current.data(), signals.currentAvailable.data(), signals.action.data(), false);
        auto const end = signals.Step(environment, observers);
        updated_ = learner_.Observe(signals.observation.data(), signals.available.data(), signals.rewards.data(),
            static_cast<std::uint32_t>(end));
        return end;
    }
    bool Ready() const override { return updated_; }
    PpokitUpdateStatistics Update() override
    {
        updated_ = false;
        return learner_.Statistics();
    }
    void Save(fs::path const& training, fs::path const& policy) override { learner_.Save(training, policy); }
    void Load(fs::path const& trainingFile) override { learner_.Load(trainingFile); }
    std::uint64_t Updates() const override { return learner_.Updates(); }
    runtime::NamedValues SettingsValues(core::PpoSettings const& plan) const override
    {
        auto values = TeamSettingsValues(learner_.Settings());
        if (graphBackend_)
        {
            // These graph optimizer and distribution settings come from training.json;
            // native team backends may instead own their settings independently.
            values.emplace_back("target_kl", plan.targetKl);
            values.emplace_back("max_gradient_norm", plan.maxGradientNorm);
            values.emplace_back("actor_learning_rate", plan.actorLearningRate);
            values.emplace_back("critic_learning_rate", plan.criticLearningRate);
            values.emplace_back("initial_log_std", plan.initialLogStd);
            values.emplace_back("minimum_log_std", plan.minimumLogStd);
            values.emplace_back("maximum_log_std", plan.maximumLogStd);
        }
        return values;
    }
    runtime::NamedValues UpdateMetrics(PpokitUpdateStatistics const& statistics) const override
    {
        return TeamMetrics(statistics);
    }
    std::vector<runtime::ProbeRecord> UpdateProbes(std::uint64_t update) const override
    {
        return ProbeRecords(update, learner_.Probes());
    }

private:
    network::TeamLearner learner_;
    bool graphBackend_ = false;
    bool updated_ = false;
};

std::unique_ptr<Trainee> CreateTrainee(RunSetupNetwork const& network, fs::path const& project)
{
    if (network->Info().agentCapacity > 1)
    {
        return std::make_unique<TeamTrainee>(network, project);
    }
    return std::make_unique<SingleTrainee>(network, project);
}

/** @brief Records the step's weighted channels of every agent that acted; closes their episodes at its end. */
void RecordStep(runtime::RewardLedger& ledger, StepSignals const& signals, std::vector<std::uint8_t>& acted,
    bool endsEpisode)
{
    for (std::size_t agent = 0; agent < signals.agents; ++agent)
    {
        if (signals.currentAvailable[agent] != 0)
        {
            ledger.RecordWeighted(agent, signals.channels.data() + agent * signals.channelCount, signals.rewards[agent]);
            acted[agent] = 1;
        }
    }
    if (!endsEpisode)
    {
        return;
    }
    for (std::size_t agent = 0; agent < signals.agents; ++agent)
    {
        if (acted[agent] != 0)
        {
            ledger.EndEpisode(agent);
            acted[agent] = 0;
        }
    }
}

/**
 * @brief The files of a training run: its checkpoint, policy, journal, environment states,
 *        and the BEST of each suite with its generations.
 */
std::vector<fs::path> RunFiles(RunSetup const& setup)
{
    schema::Project const& project = setup.project;
    fs::path const checkpoints = project.Checkpoints();
    std::vector<fs::path> files{project.LatestCheckpoint(), project.TrainingPolicy(), project.Journal(),
        runtime::TrainingStateFolder(checkpoints), checkpoints / "generation.json"};
    auto const best = eval::BestRunFiles(checkpoints, setup.suites, project.TrainingPolicy());
    files.insert(files.end(), best.begin(), best.end());
    return files;
}

/**
 * @brief A run that does not resume starts from nothing of an earlier one: its files are
 *        archived (checkpoints/archive/run-...), never overwritten.
 */
void ArchiveFormerRun(RunSetup const& setup)
{
    if (std::optional<fs::path> const archive = runtime::ArchiveRun(setup.project.Checkpoints(), RunFiles(setup)))
    {
        std::cout << "The former run is archived in " << archive->string() << ".\n";
    }
}

/**
 * @brief Gives a resumed run's environment the state saved with its checkpoint.
 * @throws std::runtime_error when states were saved but none of the checkpoint's update.
 */
void RestoreResumedEnvironment(environment::Environment& environment, fs::path const& checkpoints,
    std::uint64_t updates)
{
    std::vector<runtime::SavedTrainingState> const saved = runtime::ReadTrainingStates(checkpoints);
    if (saved.empty())
    {
        // A checkpoint of an earlier ppokit-run: the environment keeps the state it read itself.
        std::cout << "No environment state saved with this checkpoint: the environment keeps its own.\n";
        return;
    }
    std::optional<std::string> const state = runtime::TrainingStateOfUpdate(saved, updates);
    if (!state)
    {
        throw std::runtime_error(runtime::TrainingStateFolder(checkpoints).string()
            + " has no environment state of update " + std::to_string(updates)
            + ", the update of the checkpoint: the checkpoint and the states are of different runs");
    }
    environment.RestoreTrainingState(*state);
}

/** @brief Stage both artifacts and publish them with the state of their update. */
void SaveRun(Trainee& trainee, schema::Project const& project, environment::Environment const& environment,
    std::vector<double> const& weights)
{
    runtime::PublishGeneration(project, trainee.Updates(), environment.TrainingState(), weights,
        [&](fs::path const& training, fs::path const& policy) { trainee.Save(training, policy); });
}

int Train(RunSetup const& setup, Options const& options, ipc::Worker* control)
{
    schema::Project const& project = setup.project;
    runtime::WriterLock const writer({project.Checkpoints() / "generation.json", project.LatestCheckpoint(), project.TrainingPolicy(),
        project.Journal()});
    fs::path const checkpoints = project.Checkpoints();
    fs::path const journalPath = project.Journal();
    fs::create_directories(checkpoints);
    runtime::RecoverPublication(project);
    bool const haveCheckpoint = fs::is_regular_file(project.LatestCheckpoint());
    if (options.requireResume && !haveCheckpoint)
        throw std::runtime_error("--resume requires the training checkpoint at " + project.LatestCheckpoint().string());
    bool const autoResume = !setup.profile || setup.profile->resumeIfAvailable != 0;
    bool const resume = haveCheckpoint && !options.fresh && (options.requireResume || autoResume);
    // A manifest commits this project to a complete generation. Losing the learner must never
    // turn a requested/default continuation into an implicit fresh run.
    if (!haveCheckpoint && !options.fresh && fs::exists(checkpoints / "generation.json"))
        throw std::runtime_error("published training checkpoint is missing: " + project.LatestCheckpoint().string());
    if (!resume)
    {
        ArchiveFormerRun(setup);
    }
    std::unique_ptr<Trainee> const trainee = CreateTrainee(setup.network, project.root);
    std::optional<runtime::PublishedGeneration> generation;
    if (resume)
    {
        generation = runtime::VerifyPublication(project);
        trainee->Load(project.LatestCheckpoint());
        if (generation && generation->update != trainee->Updates())
            throw std::runtime_error("learner update differs from publication");
        std::cout << "Resuming from " << project.LatestCheckpoint().string() << " at update " << trainee->Updates()
                  << ".\n";
    }
    auto const startUpdate = trainee->Updates();
    auto absoluteLimit = options.untilUpdate;
    if (!absoluteLimit && !options.updates && setup.profile) absoluteLimit = setup.profile->maxUpdates;
    constexpr std::uint64_t defaultAdditionalUpdates = 100;
    constexpr std::uint64_t defaultSaveInterval = 10;
    std::uint64_t const additional = options.updates.value_or(defaultAdditionalUpdates);
    if (!absoluteLimit && additional > std::numeric_limits<std::uint64_t>::max() - startUpdate)
        throw UsageError("requested training update would overflow");
    std::uint64_t const target = absoluteLimit ? *absoluteLimit : startUpdate + additional;
    std::uint64_t const saveEvery = options.saveEvery.value_or(setup.profile ? setup.profile->checkpointEvery : defaultSaveInterval);
    std::uint64_t environmentSeed = setup.settings.seed + 1;
    if (setup.profile)
    {
        if (setup.profile->randomizeTrainingSeed != 0) environmentSeed = std::random_device{}();
        else if (startUpdate == 0) environmentSeed = setup.profile->trainingSeed;
        else environmentSeed = setup.environment->RunProfile(project.root, startUpdate).trainingSeed;
    }
    environment::Environment environment(setup.environment, project.root, environmentSeed);
    environment.SetLearning(true);
    if (generation) environment.RestoreTrainingState(generation->state);
    else if (resume) RestoreResumedEnvironment(environment, checkpoints, startUpdate);
    else
    {
        // A fresh run: what the environment may have read of an earlier one (a curriculum) goes.
        environment.RestoreTrainingState({});
    }
    std::uint64_t updates = trainee->Updates();
    std::size_t const channels = setup.task.rewards.size();
    std::vector<double> restoredWeights;
    if (generation && generation->rewardWeights) restoredWeights = *generation->rewardWeights;
    else if (resume) restoredWeights = WeightsInForce(journalPath, updates, channels);
    auto const weights = EffectiveWeights(environment, channels, restoredWeights);

    runtime::RunHeader header;
    header.task = setup.task.name;
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
        header.channels.push_back(
            {setup.task.rewards[channel].name, setup.task.rewards[channel].label, weights[channel]});
    }
    header.settings = trainee->SettingsValues(setup.settings);
    header.settings.emplace_back("environment_seed", static_cast<double>(environmentSeed));
    header.settings.emplace_back("checkpoint_every", static_cast<double>(saveEvery));
    header.acceptsWeightRequests = control != nullptr;
    std::size_t const agents = setup.environment->Info().agentCapacity;
    runtime::RewardLedger ledger(weights, agents);
    runtime::RunLogWriter journal = resume ? runtime::RunLogWriter::Resume(journalPath, header, updates, &ledger)
                                           : runtime::RunLogWriter(journalPath, header);
    runtime::ObserverSet observers(setup.task, setup.graphs.observers, agents, setup.settings.seed);

    StepSignals signals(setup.environment->Info());
    std::vector<std::uint8_t> acted(agents, 0);
    // JSBSim resumes at a new episode. Defer reset until a transition actually needs it,
    // so a boundary checkpoint retains the stream for the next start.
    bool needsReset = true;
    std::optional<std::uint64_t> published;
    if (resume) published = startUpdate;
    auto const start = std::chrono::steady_clock::now();
    std::cout << "Training environment seed " << environmentSeed << ", checkpoint every " << saveEvery
              << ", budget " << (absoluteLimit ? "absolute" : "additional") << ".\n";
    std::cout << "Training '" << setup.task.name << "' up to update " << target << ".\n";
    while (updates < target)
    {
        if (control && !control->Continue())
        {
            // No transition has been collected for the next update.
            if (published != updates)
            {
                control->Saving();
                SaveRun(*trainee, project, environment, ledger.Weights());
            }
            std::cout << "Stopped on request at update " << updates << "." << std::endl;
            return 0;
        }
        while (!trainee->Ready())
        {
            if (needsReset) signals.Reset(environment);
            auto const end = trainee->Collect(environment, signals, observers);
            needsReset = runtime::EndsEpisode(end);
            RecordStep(ledger, signals, acted, needsReset);
            if (!needsReset) signals.Advance();
            // Stop is latched. Publish only after all collected transitions have been optimized.
            if (control) (void)control->Continue();
        }
        PpokitUpdateStatistics const statistics = trainee->Update();
        updates = statistics.updates;
        // Before the save of this update: a curriculum the environment owns moves with it.
        bool const curriculumEnded = environment.TrainingProgress(updates, {});
        double const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        runtime::RewardSummary const rewards = ledger.Summarize();
        journal.WriteUpdate({updates, elapsed, trainee->UpdateMetrics(statistics), rewards});
        for (runtime::ProbeRecord const& probes : trainee->UpdateProbes(updates))
        {
            journal.WriteProbes(probes);
        }
        if (control) (void)control->Continue();
        std::cout << "update " << updates << "/" << target << "  episodes " << rewards.episodes << "  mean return "
                  << rewards.episodeReturnMean << "  explained variance " << statistics.explainedVariance
                  << std::endl;
        if (control)
        {
            if (auto const request = control->TakeWeights())
                control->Answer(ApplyWeights(*request, environment, ledger, journal, updates + 1));
        }
        bool const stop = control && control->Stopping();
        bool const outOfTime = options.maxSeconds > 0.0 && elapsed >= options.maxSeconds;
        bool const last = updates == target || outOfTime || curriculumEnded;
        bool const campaigns = !stop && AnyCampaignDue(setup.suites, updates, last);
        if (updates % saveEvery == 0 || last || stop || campaigns)
        {
            if (stop) control->Saving();
            SaveRun(*trainee, project, environment, ledger.Weights());
            published = updates;
        }
        if (stop || (control && control->Stopping()))
        {
            // Stop may have arrived after the save decision but before a due campaign.
            if (published != updates)
            {
                control->Saving();
                SaveRun(*trainee, project, environment, ledger.Weights());
            }
            std::cout << "Stopped on request at update " << updates << ", checkpoint saved." << std::endl;
            return 0;
        }
        CampaignOutcome campaignOutcome;
        if (campaigns)
        {
            campaignOutcome = RunDueCampaigns(setup, options, updates, last, environment, observers, journal, control);
        }
        if (campaignOutcome.goalReached)
        {
            std::cout << "Suite '" << options.untilPassed << "' passed at update " << updates << ", checkpoint saved."
                      << std::endl;
            return 0;
        }
        if (outOfTime)
        {
            std::cout << "Time budget of " << options.maxSeconds << " s spent at update " << updates
                      << ", checkpoint saved." << std::endl;
            break;
        }
        if (curriculumEnded || campaignOutcome.curriculumEnded)
        {
            std::cout << "The environment's curriculum ended the training at update " << updates
                      << ", checkpoint saved." << std::endl;
            break;
        }
    }
    std::cout << "Done. " << project.LatestCheckpoint().string() << " resumes training, " << project.TrainingPolicy().string()
              << " is the policy.\n";
    if (!options.untilPassed.empty())
    {
        std::cout << "Suite '" << options.untilPassed << "' did not pass by update " << updates << ".\n";
        return NotPassedExitCode;
    }
    return 0;
}

/**
 * @brief The update a policy file was saved at, when it is a PPOKit artifact (a team's DLL
 *        may keep its own file format).
 */
std::optional<std::uint64_t> PolicyUpdate(fs::path const& policyFile)
{
    try
    {
        return artifacts::ReadArtifactHeader(policyFile).updates;
    }
    catch (std::exception const&)
    {
        return std::nullopt;
    }
}

/** @brief Policy generation and evaluation context resolved together. */
struct SelectedPolicy final
{
    fs::path file;
    std::optional<std::uint64_t> update;
    std::optional<std::string> state;
    std::string source;
    std::string digest;
    std::string role;
    schema::PolicyReference const* reference = nullptr;
    std::vector<double> weights;
};

/** @brief Recognize a published file even when the selected path is a symlink or a hard link. */
bool RefersToPublishedFile(fs::path const& selected, fs::path const& published)
{
    return fs::is_regular_file(published) && fs::equivalent(selected, published);
}

/** @brief Resolve checkpoints written before generation manifests from their update-indexed sidecars. */
void ReadLegacyPolicyContext(SelectedPolicy& policy, fs::path const& checkpoints, bool hasExplicitState)
{
    auto const saved = runtime::ReadTrainingStates(checkpoints);
    if (saved.empty()) return;
    if (policy.update) policy.state = runtime::TrainingStateOfUpdate(saved, *policy.update);
    else if (saved.size() == 1) policy.state = saved.front().state;
    if (!policy.state && !hasExplicitState)
        throw std::runtime_error("no matching environment state for the current policy");
    policy.source = "checkpoint";
}

/** @brief Bind a packaged actor to its immutable provenance before opening the native network. */
void SelectReference(SelectedPolicy& policy, schema::PolicyReference const& reference)
{
    if (!fs::is_regular_file(reference.file) || artifacts::Sha256OfFile(reference.file) != reference.sha256)
        throw std::runtime_error(reference.role + " policy missing or SHA-256 mismatch: " + reference.file.string());
    if (auto const header = PolicyUpdate(reference.file); header && reference.artifactUpdate && header != reference.artifactUpdate)
        throw std::runtime_error(reference.role + " declared artifact update differs from policy header");
    policy.file = reference.file;
    policy.update = reference.sourceUpdate;
    policy.state = reference.environmentState;
    policy.source = "reference";
    policy.digest = reference.sha256;
    policy.role = reference.role;
    policy.reference = &reference;
}

schema::PolicyReference const* FindReference(RunSetup const& setup, std::string const& role)
{
    for (auto const& reference : setup.references)
        if (reference.role == role) return &reference;
    return nullptr;
}

/** @brief Resolve a live promotion first. Only its own suite may use the packaged initial BEST. */
void SelectBest(SelectedPolicy& result, RunSetup const& setup, Options const& options, eval::EvaluationSuite const* suite)
{
    auto const checkpoints = setup.project.Checkpoints();
    for (auto const& candidate : setup.suites)
    {
        if (suite && candidate.name != suite->name) continue;
        auto const best = eval::ReadBest(checkpoints, candidate.name);
        if (!best)
        {
            if (fs::exists(eval::BestManifest(checkpoints, candidate.name)))
                throw std::runtime_error("unusable BEST publication for " + candidate.name);
            continue;
        }
        result.file = best->policy;
        result.update = best->update;
        result.digest = best->sha256;
        result.state = best->environmentState;
        result.weights = best->rewardWeights;
        result.source = "best";
        result.role = "BEST";
        if (!result.state && options.evaluationState.empty())
            throw std::runtime_error("historical BEST has no saved context: specify --environment-state FILE");
        auto const headerUpdate = PolicyUpdate(result.file);
        if (headerUpdate && headerUpdate != result.update)
            throw std::runtime_error("BEST update differs from the policy header");
        std::cout << "BEST of '" << best->report.suite << "': update " << best->update << ".\n";
        return;
    }
    auto const reference = FindReference(setup, "HISTORICAL_BEST");
    if (reference && FindSuite(setup.suites, reference->suite) && (!suite || reference->suite == suite->name))
    {
        SelectReference(result, *reference);
        return;
    }
    throw std::runtime_error("no BEST policy yet in " + (checkpoints / "best").string());
}

SelectedPolicy ChoosePolicy(RunSetup const& setup, Options const& options, eval::EvaluationSuite const* suite)
{
    SelectedPolicy result;
    auto const checkpoints = setup.project.Checkpoints();
    if (options.best)
    {
        SelectBest(result, setup, options, suite);
    }
    else if (!options.model.empty() && options.model != "LATEST")
    {
        auto const reference = FindReference(setup, options.model);
        if (!reference) throw UsageError("no policy role '" + options.model + "' declared in models.json");
        SelectReference(result, *reference);
    }
    else
    {
        result.file = setup.project.Policy();
        if (options.model == "LATEST") result.file = setup.project.TrainingPolicy();
        if (!options.policy.empty()) result.file = setup.project.root / options.policy;
        if (!fs::is_regular_file(result.file))
            throw std::runtime_error("no policy at " + result.file.string());
        result.update = PolicyUpdate(result.file);
        result.source = "project defaults";
        result.role = options.policy.empty() ? "PROJECT" : "CUSTOM";
        for (auto const& reference : setup.references)
            if (RefersToPublishedFile(result.file, reference.file))
            {
                SelectReference(result, reference);
                break;
            }
        // A training checkpoint also contains an actor that the graph backend can open for inference.
        bool const trainingFile = RefersToPublishedFile(result.file, setup.project.LatestCheckpoint());
        if (!result.reference && (trainingFile || RefersToPublishedFile(result.file, setup.project.TrainingPolicy())))
        {
            result.role = "LATEST";
            auto const generation = runtime::VerifyPublication(setup.project);
            if (generation)
            {
                if (result.update && result.update != generation->update)
                    throw std::runtime_error("policy update differs from publication");
                result.update = generation->update;
                result.digest = trainingFile ? generation->trainingDigest : generation->policyDigest;
                result.state = generation->state;
                result.source = "checkpoint";
                if (generation->rewardWeights) result.weights = *generation->rewardWeights;
                else result.weights = WeightsInForce(setup.project.Journal(), generation->update, setup.task.rewards.size());
            }
            else
            {
                ReadLegacyPolicyContext(result, checkpoints, !options.evaluationState.empty());
            }
        }
    }
    if (!options.evaluationState.empty())
    {
        std::ifstream input(options.evaluationState);
        if (!input) throw std::runtime_error("cannot open evaluation context " + options.evaluationState.string());
        // Same wrapper as checkpoints/environment_state/<update>.json. The state itself is environment-specific.
        auto const override = nlohmann::json::parse(input);
        result.state = override.at("state").get<std::string>();
        if (override.contains("reward_weights")) result.weights = override.at("reward_weights").get<std::vector<double>>();
        result.source = "explicit";
    }
    if (result.digest.empty()) result.digest = artifacts::Sha256OfFile(result.file);
    if (artifacts::Sha256OfFile(result.file) != result.digest)
        throw std::runtime_error("resolved policy SHA-256 mismatch at " + result.file.string());
    std::cout << "Model " << result.role << "  " << fs::absolute(result.file).lexically_normal().string()
              << "\nSHA-256 " << result.digest << "  update "
              << (result.update ? std::to_string(*result.update) : "unknown") << "  contract " << setup.task.contract
              << "\nEnvironment " << setup.environment->Path().string() << "\nNetwork " << setup.network->Path().string() << std::endl;
    return result;
}

/** @brief Plays the policy: deterministic actions, observers called each step. */
int Play(RunSetup const& setup, Options const& options, ipc::Worker* control)
{
    eval::EvaluationSuite const* suite = FindSuite(setup.suites, options.suite);
    auto const selected = ChoosePolicy(setup, options, suite);
    auto const actor = OpenActor(setup, selected.file, selected.digest);
    std::cout << "Policy " << selected.digest << ", context " << selected.source << ".\n";
    auto const episodes = options.episodes.value_or(setup.profile ? setup.profile->inferenceEpisodes : 5U);
    auto const initialSeed = setup.profile ? setup.profile->inferenceSeed : setup.settings.seed + 7919;
    if (setup.profile && (episodes > std::numeric_limits<std::uint32_t>::max()
        || episodes - 1 > std::numeric_limits<std::uint32_t>::max() - initialSeed))
        throw UsageError("inference episode count or seed exceeds the native profile range");
    environment::Environment environment(setup.environment, setup.project.root, initialSeed);
    // Playing shows the policy: what only training moves (a curriculum) stays as it is, at
    // the last save of the training run.
    environment.SetLearning(false);
    if (selected.state)
    {
        environment.RestoreTrainingState(*selected.state);
    }
    (void)EffectiveWeights(environment, setup.task.rewards.size(), selected.weights);
    runtime::ObserverSet observers(
        setup.task, setup.graphs.observers, setup.environment->Info().agentCapacity, setup.settings.seed);
    StepSignals signals(setup.environment->Info());
    double returnSum = 0.0;
    std::size_t played = 0;
    while (played < episodes)
    {
        CheckPlaying(control);
        if (setup.profile)
        {
            auto const index = static_cast<std::uint32_t>(played);
            environment.ResetInference(initialSeed + index, index, static_cast<std::uint32_t>(episodes),
                signals.current.data(), signals.currentAvailable.data(), signals.World());
        }
        else
        {
            signals.Reset(environment);
        }
        PlayedEpisode const episode =
            PlayEpisode(environment, *actor, signals, observers, Pace{options.realtime, options.stepSeconds}, control);
        returnSum += episode.episodeReturn;
        ++played;
        std::cout << "episode " << played << ": return " << episode.episodeReturn << ", " << episode.steps
                  << " steps, " << (episode.end == runtime::StepEnd::Terminated ? "terminated" : "truncated")
                  << std::endl;
    }
    std::cout << "mean return over " << played << " episodes: " << returnSum / static_cast<double>(played)
              << std::endl;
    for (std::size_t index = 0; index < observers.Size(); ++index)
    {
        if (!observers.Active(index))
        {
            std::cout << "observer '" << observers.Name(index) << "' stopped: " << observers.Error(index) << '\n';
        }
    }
    return 0;
}

/**
 * @brief Runs the suites (or the one of --suite) on the policy, prints each verdict and
 *        writes checkpoints/evaluation_<suite>.json.
 * @return 0 when every gate passed, NotPassedExitCode otherwise.
 */
int Evaluate(RunSetup const& setup, Options const& options, ipc::Worker* control)
{
    if (setup.suites.empty())
    {
        throw std::runtime_error(setup.project.Evaluations().string()
            + " has no evaluation suite: create one (eval/<suite>.json) to evaluate the policy");
    }
    runtime::ObserverSet observers(
        setup.task, setup.graphs.observers, setup.environment->Info().agentCapacity, setup.settings.seed);
    bool passed = true;
    for (eval::EvaluationSuite const& suite : setup.suites)
    {
        if (!options.suite.empty() && suite.name != options.suite)
        {
            continue;
        }
        auto const selected = ChoosePolicy(setup, options, &suite);
        auto const& policyFile = selected.file;
        std::unique_ptr<Actor> const actor = OpenActor(setup, policyFile, selected.digest);
        nlohmann::json context = nlohmann::json::object();
        auto report = RunCampaign(setup, suite, *actor, observers, selected.state, selected.weights, control, &context);
        report.policySha256 = selected.digest;
        PrintCampaign(report, selected.update, false);
        fs::path const file = eval::EvaluationReportPath(setup.project.Checkpoints(), suite.name);
        fs::create_directories(file.parent_path());
        auto result = nlohmann::json::parse(report.ToJson());
        result.update(context);
        result["evaluation_suite_sha256"] = artifacts::Sha256Hex(eval::SuiteToJson(suite));
        result["policy_role"] = selected.role;
        result["policy_file"] = fs::absolute(selected.file).lexically_normal().string();
        if (selected.reference)
            result["policy_origin"] = {{"commit", selected.reference->sourceCommit}, {"path", selected.reference->sourcePath},
                {"sha256", selected.reference->sourceSha256}};
        result["update"] = selected.update ? nlohmann::json(*selected.update) : nlohmann::json(nullptr);
        result["context_source"] = selected.source;
        result["environment_state"] = selected.state ? nlohmann::json(*selected.state) : nlohmann::json(nullptr);
        result["seeds"] = suite.seeds;
        result["environment_scope"] = suite.campaignEnvironment ? "campaign" : "seed";
        result["episodes_per_seed"] = suite.episodesPerSeed;
        result["task_contract"] = setup.task.contract;
        if (auto const settings = setup.project.SettingsFile())
            result["scenario_sha256"] = artifacts::Sha256OfFile(*settings);
        if (fs::is_regular_file(setup.project.Training()))
            result["training_sha256"] = artifacts::Sha256OfFile(setup.project.Training());
        runtime::WriterLock const reportLock({setup.project.Checkpoints() / ".evaluation" / suite.name / "report"});
        std::string const text = result.dump(2) + "\n";
        artifacts::WriteFileAtomically(
            file, [&](std::ostream& out) { out << text; }, artifacts::AtomicWriteOptions{}, "evaluation report");
        passed = passed && report.Passed();
    }
    if (passed)
    {
        return 0;
    }
    return NotPassedExitCode;
}

/**
 * @brief Makes a checkpoint of another environment this project's starting point, on
 *        purpose: copied under this project's contract, then loaded to check it fits.
 */
int Transfer(RunSetup const& setup, Options const& options)
{
    fs::path const source = options.from.is_absolute() ? options.from : fs::current_path() / options.from;
    artifacts::ArtifactHeader const header = artifacts::ReadArtifactHeader(source);
    bool const training = header.kind == artifacts::ArtifactKind::Training;
    fs::path const target = training ? setup.project.LatestCheckpoint() : setup.project.Policy();
    runtime::WriterLock const writer({target, setup.project.Checkpoints() / "generation.json"});
    if (fs::exists(target) || fs::exists(setup.project.Checkpoints() / ".publication-ready"))
        throw std::runtime_error("nothing is overwritten: transfer destination is already in use: " + target.string());
    std::string const& contract = setup.task.contract;
    artifacts::RetargetArtifact(source, target, contract);
    try
    {
        if (training)
        {
            CreateTrainee(setup.network, setup.project.root)->Load(target);
        }
        else
        {
            OpenActor(setup, target, artifacts::Sha256OfFile(target));
        }
    }
    catch (...)
    {
        std::error_code ignored;
        fs::remove(target, ignored);
        throw;
    }
    std::cout << "Transferred " << source.string() << " (" << artifacts::ArtifactKindName(header.kind) << ", update "
              << header.updates << ", environment '" << header.identity.contract << "') to " << target.string()
              << " for the environment '" << contract << "'.\n";
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::unique_ptr<ipc::Worker> control;
    try
    {
        Options options = ParseOptions(argc, argv);
        if (!options.controlEndpoint.empty())
            control = std::make_unique<ipc::Worker>(options.controlEndpoint, options.command == "train");
        RunSetup const setup = Prepare(options);
        if (options.command == "transfer") return Transfer(setup, options);
        int result = 0;
        try
        {
            if (options.command == "train") result = Train(setup, options, control.get());
            else if (options.command == "eval") result = Evaluate(setup, options, control.get());
            else result = Play(setup, options, control.get());
        }
        catch (RunStopped const&)
        {
            std::cout << "Stopped on request at a simulation step." << std::endl;
        }
        if (control) control->Finish();
        return result;
    }
    catch (UsageError const& error)
    {
        if (control) control->Fail(error.what());
        std::cerr << "ppokit-run: " << error.what() << "\n";
        PrintUsage(std::cerr);
        return 2;
    }
    catch (std::exception const& error)
    {
        if (control) control->Fail(error.what());
        std::cerr << "ppokit-run: " << error.what() << '\n';
        return 1;
    }
}
