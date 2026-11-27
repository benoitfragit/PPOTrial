/**
 * @file PPOKit/src/network/graph_network_library.cpp
 * @brief Shared graph PPO module and fixed-contract compatibility adapter.
 * @details The shared module binds a task's semantic identity when each policy or learner
 * is created. The fixed-contract adapter remains for native ABI regression tests.
 * Dimensions and topology are loaded from task.json and network.json.
 * Exceptions never cross the C boundary: they become the handle's error.
 */
#include "ppokit/network/network_abi.h"

#if !defined(PPOKIT_DYNAMIC_GRAPH)
#include PPOKIT_TASK_LAYOUT
#endif

#include "ppokit/component/native_module.hpp"
#include "ppokit/graph/graph_learner.hpp"
#if defined(PPOKIT_DYNAMIC_GRAPH)
#include "ppokit/graph/team_graph_learner.hpp"
#endif
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

#if !defined(PPOKIT_DYNAMIC_GRAPH)
using Layout = PPOKIT_TASK_NAMESPACE::TaskLayout;
constexpr std::size_t ObservationSize = Layout::Observation::Size;
constexpr std::size_t ActionSize = Layout::Action::Size;
#endif
using Learner = ppokit::graph::GraphLearnerCore;
using Rollout = ppokit::runtime::DynamicRolloutBuffer;
using Transition = ppokit::runtime::DynamicTransition;

/** @brief Task, networks and PPO settings of a project folder, checked against this DLL. */
struct ProjectNetworks final
{
    ppokit::schema::TaskSchema task;
    ppokit::schema::BlockGraphSchema graphs;
    ppokit::core::PpoSettings settings;
    /** @brief Steps per rollout (training.json), or a default without one. */
    std::size_t rolloutSteps = 2048;
    /** @brief Interval between probed updates; zero disables probes. */
    std::size_t probeEvery = 0;
    ppokit::artifacts::ArtifactIdentity identity;
};

/**
 * @brief Reads the project folder (task.json, network.json, training.json) and checks its task.
 * @param team True for the networks of a team (agents.max above 1), false for one agent.
 */
ProjectNetworks ReadProject(std::filesystem::path const& folder, std::uint64_t expectedContract, bool team = false)
{
    ppokit::schema::Project files;
    files.root = folder;
    ProjectNetworks project;
    project.task = ppokit::schema::LoadTaskSchema(files.Task());
    if (ppokit::schema::SemanticHash(project.task) != expectedContract)
    {
#if defined(PPOKIT_DYNAMIC_GRAPH)
        throw std::runtime_error(files.Task().string() + ": bound graph task contract differs from task.json");
#else
        throw std::runtime_error(files.Task().string() + " is not the task this network DLL was built for ("
            + std::string(Layout::Name) + "): rebuild the DLL");
#endif
    }
    if ((project.task.maximumAgents > 1) != team)
    {
        throw std::runtime_error(files.Task().string() + (team ? ": a team learner needs agents.max above 1"
                                                               : ": a team task has no single-agent networks"));
    }
#if defined(PPOKIT_PROGRAM_REGISTRY)
    // Initialize the module's registry once; multiple projects can open this DLL concurrently.
    static std::once_flag registryOnce;
    std::call_once(registryOnce, [] {
        ppokit::component::SetKnownModules(ppokit::schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
    });
#endif
    project.graphs = ppokit::schema::LoadBlockGraph(files.Network(), project.task);
    if (std::filesystem::exists(files.Training()))
    {
        ppokit::schema::TrainingPlan const plan = ppokit::schema::LoadTrainingPlan(files.Training(), project.task);
        project.settings = plan.ppo.Settings();
        project.rolloutSteps = plan.ppo.RolloutSteps();
        project.probeEvery = plan.probeEvery;
    }
    project.identity = ppokit::schema::ArtifactIdentityOf(project.task, project.graphs);
    return project;
}

Learner MakeLearner(ProjectNetworks const& project)
{
    return Learner(ppokit::schema::BuildGraph(project.graphs.actor), ppokit::schema::BuildGraph(project.graphs.critic),
        ppokit::schema::MakePolicyLayout(project.graphs, project.task), project.settings,
        project.task.observation.size, project.task.actionSize);
}

void CopyError(std::string const& message, char* error, std::size_t errorSize) noexcept
{
    if (error != nullptr && errorSize > 0)
    {
        std::snprintf(error, errorSize, "%s", message.c_str());
    }
}

/** @brief Allocate the transition scratch row once, outside the collection loop. */
Transition MakeTransition(ProjectNetworks const& project)
{
    Transition row;
    row.observation.resize(project.task.observation.size);
    row.action.resize(project.task.actionSize);
    return row;
}

/** @brief Runs `body`; an exception becomes return code 1 and `error`. */
template<typename Body>
int Guarded(std::string& error, Body&& body) noexcept
{
    try
    {
        body();
        return 0;
    }
    catch (std::exception const& failure)
    {
        error = failure.what();
    }
    catch (...)
    {
        error = "unknown exception";
    }
    return 1;
}

ppokit::runtime::StepEnd ToStepEnd(std::uint32_t end)
{
    switch (end)
    {
    case PPOKIT_STEP_RUNNING:
        return ppokit::runtime::StepEnd::Running;
    case PPOKIT_STEP_TERMINATED:
        return ppokit::runtime::StepEnd::Terminated;
    case PPOKIT_STEP_TRUNCATED:
        return ppokit::runtime::StepEnd::Truncated;
    default:
        throw std::invalid_argument("unknown step end " + std::to_string(end));
    }
}

} // namespace

struct PpokitPolicy final
{
    ProjectNetworks project;
    Learner learner;
    std::string error;
};

struct PpokitLearner final
{
    ProjectNetworks project;
    Learner learner;
    Rollout buffer;
    Transition row;
    std::string error;
};

#if defined(PPOKIT_DYNAMIC_GRAPH)
/** @brief A team's actor for inference: the team learner of the project, with a one-step rollout. */
struct PpokitTeamPolicy final
{
    explicit PpokitTeamPolicy(ProjectNetworks networks)
        : project(std::move(networks)), learner(project.task, project.graphs, project.settings, 1)
    {
    }

    ProjectNetworks project;
    ppokit::graph::TeamGraphLearner learner;
    std::string error;
};

struct PpokitTeamLearner final
{
    explicit PpokitTeamLearner(ProjectNetworks networks)
        : project(std::move(networks)),
          learner(project.task, project.graphs, project.settings, project.rolloutSteps)
    {
        if (project.probeEvery != 0)
        {
            actorProbes = std::make_unique<ppokit::graph::GraphProbes>(learner.Models().Actor());
            criticProbes = std::make_unique<ppokit::graph::GraphProbes>(learner.Models().Critic());
            ScheduleProbes();
        }
    }

    /** @brief Enable probes for the next scheduled update, after creation or resume. */
    void ScheduleProbes() noexcept
    {
        bool const enabled = project.probeEvery != 0
            && learner.Updates() % project.probeEvery == project.probeEvery - 1;
        if (enabled)
        {
            actorProbes->Reset();
            criticProbes->Reset();
        }
        learner.Models().SetProbes(enabled ? actorProbes.get() : nullptr, enabled ? criticProbes.get() : nullptr);
        probing = enabled;
    }

    /** @brief Preserve the completed update's probe results before scheduling the next one. */
    void FinishUpdate()
    {
        actorSummary.clear();
        criticSummary.clear();
        if (probing)
        {
            actorSummary = actorProbes->Summary();
            criticSummary = criticProbes->Summary();
        }
        ScheduleProbes();
    }

    ProjectNetworks project;
    ppokit::graph::TeamGraphLearner learner;
    std::unique_ptr<ppokit::graph::GraphProbes> actorProbes;
    std::unique_ptr<ppokit::graph::GraphProbes> criticProbes;
    std::vector<ppokit::runtime::BlockProbe> actorSummary;
    std::vector<ppokit::runtime::BlockProbe> criticSummary;
    bool probing = false;
    /** @brief Message of the last failed call; also set by the const probe query. */
    mutable std::string error;
};
#endif

extern "C" {

PPOKIT_NETWORK_EXPORT void ppokit_network_info(PpokitNetworkInfo* info)
{
    info->abi = PPOKIT_NETWORK_ABI;
    info->agentCapacity = 1;
#if defined(PPOKIT_DYNAMIC_GRAPH)
    info->observationSize = 0;
    info->actionSize = 0;
    info->task = nullptr;
    info->backend = PPOKIT_GRAPH_DYNAMIC_BACKEND;
#else
    info->observationSize = static_cast<std::uint32_t>(ObservationSize);
    info->actionSize = static_cast<std::uint32_t>(ActionSize);
    info->task = Layout::Name;
    info->backend = "graph";
#endif
}

#if defined(PPOKIT_DYNAMIC_GRAPH)
PPOKIT_NETWORK_EXPORT PpokitPolicy* ppokit_graph_policy_open(char const* project, char const* policyFile,
    std::uint64_t expectedContract, char* error, std::size_t errorSize)
#else
PPOKIT_NETWORK_EXPORT PpokitPolicy* ppokit_policy_open(char const* project, char const* policyFile, char* error,
    std::size_t errorSize)
#endif
{
#if !defined(PPOKIT_DYNAMIC_GRAPH)
    constexpr auto expectedContract = Layout::SemanticHash;
#endif
    try
    {
        ProjectNetworks networks = ReadProject(project, expectedContract);
        networks.settings = ppokit::core::InferenceSettings(networks.settings);
        Learner learner = MakeLearner(networks);
        learner.LoadPolicy(policyFile, networks.identity);
        return new PpokitPolicy{std::move(networks), std::move(learner), {}};
    }
    catch (std::exception const& failure)
    {
        CopyError(failure.what(), error, errorSize);
    }
    return nullptr;
}

PPOKIT_NETWORK_EXPORT int ppokit_policy_act(PpokitPolicy* policy, float const* observation, float* action,
    int deterministic)
{
    return Guarded(policy->error, [&] {
        if (deterministic != 0)
        {
            policy->learner.Mean(observation, action);
        }
        else
        {
            double logProbability = 0.0;
            double value = 0.0;
            policy->learner.Decide(observation, action, logProbability, value);
        }
    });
}

PPOKIT_NETWORK_EXPORT char const* ppokit_policy_error(PpokitPolicy const* policy)
{
    return policy->error.c_str();
}

PPOKIT_NETWORK_EXPORT void ppokit_policy_close(PpokitPolicy* policy)
{
    delete policy;
}

#if defined(PPOKIT_DYNAMIC_GRAPH)
PPOKIT_NETWORK_EXPORT PpokitLearner* ppokit_graph_learner_create(char const* project, std::uint64_t seed,
    std::uint64_t expectedContract, char* error, std::size_t errorSize)
#else
PPOKIT_NETWORK_EXPORT PpokitLearner* ppokit_learner_create(char const* project, std::uint64_t seed, char* error,
    std::size_t errorSize)
#endif
{
#if !defined(PPOKIT_DYNAMIC_GRAPH)
    constexpr auto expectedContract = Layout::SemanticHash;
#endif
    try
    {
        ProjectNetworks networks = ReadProject(project, expectedContract);
        if (seed != 0)
        {
            networks.settings.seed = seed;
        }
        Learner learner = MakeLearner(networks);
        Rollout buffer(networks.rolloutSteps, networks.task.observation.size, networks.task.actionSize);
        auto row = MakeTransition(networks);
        return new PpokitLearner{std::move(networks), std::move(learner), std::move(buffer), std::move(row), {}};
    }
    catch (std::exception const& failure)
    {
        CopyError(failure.what(), error, errorSize);
    }
    return nullptr;
}

PPOKIT_NETWORK_EXPORT int ppokit_learner_decide(PpokitLearner* learner, float const* observation, float* action,
    double* logProbability, double* value)
{
    return Guarded(learner->error, [&] {
        learner->learner.Decide(observation, action, *logProbability, *value);
    });
}

PPOKIT_NETWORK_EXPORT int ppokit_learner_value(PpokitLearner* learner, float const* observation, double* value)
{
    return Guarded(learner->error, [&] { *value = learner->learner.Value(observation); });
}

PPOKIT_NETWORK_EXPORT int ppokit_learner_append(PpokitLearner* learner, PpokitTransition const* transition)
{
    return Guarded(learner->error, [&] {
        auto& row = learner->row;
        std::copy_n(transition->observation, row.observation.size(), row.observation.begin());
        std::copy_n(transition->action, row.action.size(), row.action.begin());
        row.logProbability = transition->logProbability;
        row.value = transition->value;
        row.reward = transition->reward;
        row.nextValue = transition->nextValue;
        row.end = ToStepEnd(transition->end);
        learner->buffer.Append(row);
    });
}

PPOKIT_NETWORK_EXPORT int ppokit_learner_full(PpokitLearner const* learner)
{
    return learner->buffer.Full() ? 1 : 0;
}

PPOKIT_NETWORK_EXPORT int ppokit_learner_update(PpokitLearner* learner, PpokitUpdateStatistics* statistics)
{
    return Guarded(learner->error, [&] {
        ppokit::core::UpdateStatistics const result = learner->learner.Update(learner->buffer);
        statistics->updates = result.updates;
        statistics->policyLoss = result.policyLoss;
        statistics->valueLoss = result.valueLoss;
        statistics->entropy = result.entropy;
        statistics->approximateKl = result.approximateKl;
        statistics->clipFraction = result.clipFraction;
        statistics->explainedVariance = result.explainedVariance;
        statistics->actorKlSkips = result.actorKlSkips;
    });
}

PPOKIT_NETWORK_EXPORT int ppokit_learner_save(PpokitLearner* learner, char const* trainingFile, char const* policyFile)
{
    return Guarded(learner->error, [&] {
        if (trainingFile != nullptr)
        {
            if (learner->buffer.Size() != 0)
            {
                throw std::logic_error("PPOKit graph learner: training checkpoint requires an empty rollout (update boundary)");
            }
            learner->learner.SaveTraining(trainingFile, learner->project.identity);
        }
        if (policyFile != nullptr)
        {
            learner->learner.SavePolicy(policyFile, learner->project.identity);
        }
    });
}

PPOKIT_NETWORK_EXPORT int ppokit_learner_load(PpokitLearner* learner, char const* trainingFile)
{
    return Guarded(learner->error, [&] {
        if (learner->buffer.Size() != 0)
        {
            throw std::logic_error("PPOKit graph learner: loading training state requires an empty rollout (update boundary)");
        }
        learner->learner.LoadTraining(trainingFile, learner->project.identity);
    });
}

PPOKIT_NETWORK_EXPORT std::uint64_t ppokit_learner_updates(PpokitLearner const* learner)
{
    return learner->learner.Updates();
}

PPOKIT_NETWORK_EXPORT char const* ppokit_learner_error(PpokitLearner const* learner)
{
    return learner->error.c_str();
}

PPOKIT_NETWORK_EXPORT void ppokit_learner_destroy(PpokitLearner* learner)
{
    delete learner;
}

#if defined(PPOKIT_DYNAMIC_GRAPH)
PPOKIT_NETWORK_EXPORT PpokitTeamPolicy* ppokit_graph_team_policy_open(char const* project, char const* policyFile,
    std::uint64_t expectedContract, char* error, std::size_t errorSize)
{
    try
    {
        ProjectNetworks networks = ReadProject(project, expectedContract, true);
        networks.settings = ppokit::core::InferenceSettings(networks.settings);
        auto policy = std::make_unique<PpokitTeamPolicy>(std::move(networks));
        policy->learner.LoadPolicy(policyFile, policy->project.identity);
        return policy.release();
    }
    catch (std::exception const& failure)
    {
        CopyError(failure.what(), error, errorSize);
    }
    return nullptr;
}

PPOKIT_NETWORK_EXPORT int ppokit_team_policy_act(PpokitTeamPolicy* policy, float const* observations,
    std::uint8_t const* available, float* actions)
{
    return Guarded(policy->error, [&] { policy->learner.Act(observations, available, actions, true); });
}

PPOKIT_NETWORK_EXPORT char const* ppokit_team_policy_error(PpokitTeamPolicy const* policy)
{
    return policy->error.c_str();
}

PPOKIT_NETWORK_EXPORT void ppokit_team_policy_close(PpokitTeamPolicy* policy)
{
    delete policy;
}

PPOKIT_NETWORK_EXPORT PpokitTeamLearner* ppokit_graph_team_learner_create(char const* project, std::uint64_t seed,
    std::uint64_t expectedContract, char* error, std::size_t errorSize)
{
    try
    {
        ProjectNetworks networks = ReadProject(project, expectedContract, true);
        if (seed != 0)
        {
            networks.settings.seed = static_cast<std::uint32_t>(seed);
        }
        return new PpokitTeamLearner(std::move(networks));
    }
    catch (std::exception const& failure)
    {
        CopyError(failure.what(), error, errorSize);
    }
    return nullptr;
}

PPOKIT_NETWORK_EXPORT int ppokit_team_learner_act(PpokitTeamLearner* learner, float const* observations,
    std::uint8_t const* available, float* actions, int deterministic)
{
    return Guarded(learner->error,
        [&] { learner->learner.Act(observations, available, actions, deterministic != 0); });
}

PPOKIT_NETWORK_EXPORT int ppokit_team_learner_observe(PpokitTeamLearner* learner, float const* observations,
    std::uint8_t const* available, double const* rewards, std::uint32_t end, int* updated)
{
    return Guarded(learner->error, [&] {
        *updated = learner->learner.Observe(observations, available, rewards, ToStepEnd(end)) ? 1 : 0;
        if (*updated != 0) learner->FinishUpdate();
    });
}

PPOKIT_NETWORK_EXPORT int ppokit_team_learner_statistics(PpokitTeamLearner const* learner,
    PpokitUpdateStatistics* statistics)
{
    ppokit::runtime::TeamUpdateStatistics const& last = learner->learner.LastUpdate();
    *statistics = PpokitUpdateStatistics{};
    statistics->updates = learner->learner.Updates();
    statistics->policyLoss = last.policyLoss;
    statistics->valueLoss = last.valueLoss;
    statistics->entropy = last.entropy;
    statistics->approximateKl = last.approximateKl;
    statistics->clipFraction = last.clipFraction;
    statistics->explainedVariance = last.explainedVariance;
    statistics->actorKlSkips = last.actorKlSkips;
    statistics->advantageAbsMean = last.advantageAbsMean;
    statistics->actionStdMean = last.actionStdMean;
    statistics->learnableRows = last.learnableRows;
    statistics->criticRows = last.criticRows;
    return 0;
}

PPOKIT_NETWORK_EXPORT int ppokit_team_learner_settings(PpokitTeamLearner const* learner,
    PpokitTeamSettings* settings)
{
    ppokit::runtime::TeamPpoSettings const& team = learner->learner.Settings();
    settings->gamma = team.gamma;
    settings->lambda = team.lambda;
    settings->clip = team.clip;
    settings->entropyCoefficient = team.entropyCoefficient;
    settings->epochs = static_cast<std::uint32_t>(team.epochs);
    settings->minibatch = static_cast<std::uint32_t>(team.minibatch);
    return 0;
}

PPOKIT_NETWORK_EXPORT std::uint32_t ppokit_team_learner_probe_count(PpokitTeamLearner const* learner)
{
    return static_cast<std::uint32_t>(learner->actorSummary.size() + learner->criticSummary.size());
}

PPOKIT_NETWORK_EXPORT int ppokit_team_learner_probe(PpokitTeamLearner const* learner, std::uint32_t index,
    PpokitBlockProbe* output)
{
    if (index >= ppokit_team_learner_probe_count(learner))
    {
        learner->error = "no probe " + std::to_string(index) + " in the last update";
        return 1;
    }
    bool const actor = index < learner->actorSummary.size();
    auto const& probe = actor ? learner->actorSummary[index] : learner->criticSummary[index - learner->actorSummary.size()];
    *output = PpokitBlockProbe{};
    output->network = actor ? "actor" : "critic";
    output->block = probe.block.c_str();
    output->type = probe.type.c_str();
    output->samples = probe.samples;
    output->mean = probe.mean;
    output->deviation = probe.deviation;
    output->minimum = probe.minimum;
    output->maximum = probe.maximum;
    output->deadShare = probe.deadShare;
    output->saturatedShare = probe.saturatedShare;
    output->hasGradient = probe.hasGradient ? 1 : 0;
    output->gradientNorm = probe.gradientNorm;
    return 0;
}

PPOKIT_NETWORK_EXPORT int ppokit_team_learner_save(PpokitTeamLearner* learner, char const* trainingFile,
    char const* policyFile)
{
    return Guarded(learner->error, [&] {
        if (trainingFile != nullptr)
        {
            learner->learner.SaveTraining(trainingFile, learner->project.identity);
        }
        if (policyFile != nullptr)
        {
            learner->learner.SavePolicy(policyFile, learner->project.identity);
        }
    });
}

PPOKIT_NETWORK_EXPORT int ppokit_team_learner_load(PpokitTeamLearner* learner, char const* trainingFile)
{
    return Guarded(learner->error, [&] {
        learner->learner.LoadTraining(trainingFile, learner->project.identity);
        learner->actorSummary.clear();
        learner->criticSummary.clear();
        learner->ScheduleProbes();
    });
}

PPOKIT_NETWORK_EXPORT std::uint64_t ppokit_team_learner_updates(PpokitTeamLearner const* learner)
{
    return learner->learner.Updates();
}

PPOKIT_NETWORK_EXPORT char const* ppokit_team_learner_error(PpokitTeamLearner const* learner)
{
    return learner->error.c_str();
}

PPOKIT_NETWORK_EXPORT void ppokit_team_learner_destroy(PpokitTeamLearner* learner)
{
    delete learner;
}
#endif

} // extern "C"
