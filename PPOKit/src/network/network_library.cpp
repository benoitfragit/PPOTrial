/**
 * @file PPOKit/src/network/network_library.cpp
 * @brief Loading a network DLL and calling its C API from C++.
 */
#include "ppokit/network/network_library.hpp"

#include "ppokit/component/native_module.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ppokit::network {

namespace {

constexpr std::size_t ErrorCapacity = 1024;

/** @brief Finds one function of the C API, or says which is missing. */
template<typename Function>
void Bind(void* module, char const* name, Function& function, std::filesystem::path const& path)
{
    function = reinterpret_cast<Function>(component::NativeSymbol(module, name));
    if (function == nullptr)
    {
        throw std::runtime_error(path.string() + ": not a PPOKit network library (no " + name
            + "; build it with ppokit_add_network_library)");
    }
}

/** @brief Refuses a single-agent object on a team DLL, or the reverse. */
void RequireKind(NetworkLibrary const& library, bool team)
{
    bool const isTeam = library.Info().agentCapacity > 1;
    if (isTeam != team)
    {
        throw std::runtime_error(library.Path().string() + (isTeam ? ": the networks of a team (agentCapacity "
            + std::to_string(library.Info().agentCapacity) + ") have no single-agent policy or learner"
            : ": single-agent networks have no team policy or learner"));
    }
}

} // namespace

std::shared_ptr<NetworkLibrary const> NetworkLibrary::Open(std::string const& reference,
    std::filesystem::path const& directory, schema::TaskSchema const* task)
{
    std::shared_ptr<NetworkLibrary> library(new NetworkLibrary());
    library->path_ = component::ResolveNativeModule(reference, directory, "network library");
    library->handle_ = component::OpenNativeModule(library->path_);
    void* const module = library->handle_;
    std::filesystem::path const& path = library->path_;
    NetworkFunctions& f = library->functions_;
    Bind(module, "ppokit_network_info", f.info, path);
    f.info(&library->info_);
    // The version first: a DLL of another ABI may lack functions of this one.
    if (library->info_.abi != PPOKIT_NETWORK_ABI)
    {
        throw std::runtime_error(path.string() + ": network ABI " + std::to_string(library->info_.abi)
            + ", this program reads " + std::to_string(PPOKIT_NETWORK_ABI) + ": rebuild it");
    }
    // The shared graph module has no compiled task. Bind the validated task per client.
    bool const dynamicGraph = library->info_.backend != nullptr
        && std::string_view(library->info_.backend) == PPOKIT_GRAPH_DYNAMIC_BACKEND;
    if (dynamicGraph)
    {
        if (task == nullptr || task->maximumAgents == 0 || task->observation.size == 0 || task->actionSize == 0
            || task->maximumAgents > std::numeric_limits<std::uint32_t>::max()
            || task->observation.size > std::numeric_limits<std::uint32_t>::max()
            || task->actionSize > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error(path.string() + ": the shared graph backend requires a valid task");
        }
        library->taskName_ = task->name;
        library->graphContract_ = schema::SemanticHash(*task);
        library->info_.task = library->taskName_.c_str();
        library->info_.agentCapacity = static_cast<std::uint32_t>(task->maximumAgents);
        library->info_.observationSize = static_cast<std::uint32_t>(task->observation.size);
        library->info_.actionSize = static_cast<std::uint32_t>(task->actionSize);
    }
    if (library->info_.agentCapacity == 0)
    {
        throw std::runtime_error(path.string() + ": the networks act for at least one agent");
    }
    if (library->info_.agentCapacity == 1)
    {
        if (dynamicGraph)
            Bind(module, "ppokit_graph_policy_open", f.graphPolicyOpen, path);
        else
            Bind(module, "ppokit_policy_open", f.policyOpen, path);
        Bind(module, "ppokit_policy_act", f.policyAct, path);
        Bind(module, "ppokit_policy_error", f.policyError, path);
        Bind(module, "ppokit_policy_close", f.policyClose, path);
        if (dynamicGraph)
            Bind(module, "ppokit_graph_learner_create", f.graphLearnerCreate, path);
        else
            Bind(module, "ppokit_learner_create", f.learnerCreate, path);
        Bind(module, "ppokit_learner_decide", f.learnerDecide, path);
        Bind(module, "ppokit_learner_value", f.learnerValue, path);
        Bind(module, "ppokit_learner_append", f.learnerAppend, path);
        Bind(module, "ppokit_learner_full", f.learnerFull, path);
        Bind(module, "ppokit_learner_update", f.learnerUpdate, path);
        Bind(module, "ppokit_learner_save", f.learnerSave, path);
        Bind(module, "ppokit_learner_load", f.learnerLoad, path);
        Bind(module, "ppokit_learner_updates", f.learnerUpdates, path);
        Bind(module, "ppokit_learner_error", f.learnerError, path);
        Bind(module, "ppokit_learner_destroy", f.learnerDestroy, path);
    }
    else
    {
        if (dynamicGraph)
            Bind(module, "ppokit_graph_team_policy_open", f.graphTeamPolicyOpen, path);
        else
            Bind(module, "ppokit_team_policy_open", f.teamPolicyOpen, path);
        Bind(module, "ppokit_team_policy_act", f.teamPolicyAct, path);
        Bind(module, "ppokit_team_policy_error", f.teamPolicyError, path);
        Bind(module, "ppokit_team_policy_close", f.teamPolicyClose, path);
        if (dynamicGraph)
            Bind(module, "ppokit_graph_team_learner_create", f.graphTeamLearnerCreate, path);
        else
            Bind(module, "ppokit_team_learner_create", f.teamLearnerCreate, path);
        Bind(module, "ppokit_team_learner_act", f.teamLearnerAct, path);
        Bind(module, "ppokit_team_learner_observe", f.teamLearnerObserve, path);
        Bind(module, "ppokit_team_learner_statistics", f.teamLearnerStatistics, path);
        Bind(module, "ppokit_team_learner_settings", f.teamLearnerSettings, path);
        Bind(module, "ppokit_team_learner_probe_count", f.teamLearnerProbeCount, path);
        Bind(module, "ppokit_team_learner_probe", f.teamLearnerProbe, path);
        Bind(module, "ppokit_team_learner_save", f.teamLearnerSave, path);
        Bind(module, "ppokit_team_learner_load", f.teamLearnerLoad, path);
        Bind(module, "ppokit_team_learner_updates", f.teamLearnerUpdates, path);
        Bind(module, "ppokit_team_learner_error", f.teamLearnerError, path);
        Bind(module, "ppokit_team_learner_destroy", f.teamLearnerDestroy, path);
    }
    return library;
}

void NetworkLibrary::VerifyTaskContract(std::uint64_t expected) const
{
    if (graphContract_)
    {
        if (*graphContract_ != expected)
            throw std::runtime_error(path_.string() + ": bound graph task contract differs from task.json");
        return;
    }
    component::RequireNativeTaskContract(handle_, path_, expected);
}

NetworkLibrary::~NetworkLibrary()
{
    if (handle_ != nullptr)
    {
        component::CloseNativeModule(handle_);
    }
}

Policy::Policy(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project,
    std::filesystem::path const& policyFile)
    : library_(std::move(library))
{
    RequireKind(*library_, false);
    std::array<char, ErrorCapacity> error{};
    auto const& functions = library_->Functions();
    if (library_->graphContract_)
    {
        policy_ = functions.graphPolicyOpen(project.string().c_str(), policyFile.string().c_str(),
            *library_->graphContract_, error.data(), error.size());
    }
    else
    {
        policy_ = functions.policyOpen(project.string().c_str(), policyFile.string().c_str(), error.data(), error.size());
    }
    if (policy_ == nullptr)
    {
        throw std::runtime_error(std::string("policy: ") + error.data());
    }
}

Policy::~Policy()
{
    library_->Functions().policyClose(policy_);
}

void Policy::Act(float const* observation, float* action, bool deterministic)
{
    if (library_->Functions().policyAct(policy_, observation, action, deterministic ? 1 : 0) != 0)
    {
        throw std::runtime_error(std::string("policy: ") + library_->Functions().policyError(policy_));
    }
}

Learner::Learner(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project,
    std::uint64_t seed)
    : library_(std::move(library))
{
    RequireKind(*library_, false);
    std::array<char, ErrorCapacity> error{};
    auto const& functions = library_->Functions();
    if (library_->graphContract_)
    {
        learner_ = functions.graphLearnerCreate(project.string().c_str(), seed, *library_->graphContract_, error.data(), error.size());
    }
    else
    {
        learner_ = functions.learnerCreate(project.string().c_str(), seed, error.data(), error.size());
    }
    if (learner_ == nullptr)
    {
        throw std::runtime_error(std::string("learner: ") + error.data());
    }
}

Learner::~Learner()
{
    library_->Functions().learnerDestroy(learner_);
}

void Learner::Require(int code, char const* call) const
{
    if (code != 0)
    {
        throw std::runtime_error(std::string("learner ") + call + ": " + library_->Functions().learnerError(learner_));
    }
}

void Learner::Decide(float const* observation, float* action, double& logProbability, double& value)
{
    Require(library_->Functions().learnerDecide(learner_, observation, action, &logProbability, &value), "decide");
}

double Learner::Value(float const* observation)
{
    double value = 0.0;
    Require(library_->Functions().learnerValue(learner_, observation, &value), "value");
    return value;
}

void Learner::Append(PpokitTransition const& transition)
{
    Require(library_->Functions().learnerAppend(learner_, &transition), "append");
}

bool Learner::Full() const
{
    return library_->Functions().learnerFull(learner_) != 0;
}

PpokitUpdateStatistics Learner::Update()
{
    PpokitUpdateStatistics statistics{};
    Require(library_->Functions().learnerUpdate(learner_, &statistics), "update");
    return statistics;
}

void Learner::Save(std::filesystem::path const& trainingFile, std::filesystem::path const& policyFile)
{
    std::string const training = trainingFile.string();
    std::string const policy = policyFile.string();
    Require(library_->Functions().learnerSave(learner_, training.empty() ? nullptr : training.c_str(),
                policy.empty() ? nullptr : policy.c_str()),
        "save");
}

void Learner::Load(std::filesystem::path const& trainingFile)
{
    Require(library_->Functions().learnerLoad(learner_, trainingFile.string().c_str()), "load");
}

std::uint64_t Learner::Updates() const
{
    return library_->Functions().learnerUpdates(learner_);
}

TeamPolicy::TeamPolicy(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project,
    std::filesystem::path const& policyFile)
    : library_(std::move(library))
{
    RequireKind(*library_, true);
    std::array<char, ErrorCapacity> error{};
    auto const& functions = library_->Functions();
    if (library_->graphContract_)
    {
        policy_ = functions.graphTeamPolicyOpen(project.string().c_str(), policyFile.string().c_str(),
            *library_->graphContract_, error.data(), error.size());
    }
    else
    {
        policy_ = functions.teamPolicyOpen(project.string().c_str(), policyFile.string().c_str(), error.data(),
            error.size());
    }
    if (policy_ == nullptr)
    {
        throw std::runtime_error(std::string("team policy: ") + error.data());
    }
}

TeamPolicy::~TeamPolicy()
{
    library_->Functions().teamPolicyClose(policy_);
}

void TeamPolicy::Act(float const* observations, std::uint8_t const* available, float* actions)
{
    if (library_->Functions().teamPolicyAct(policy_, observations, available, actions) != 0)
    {
        throw std::runtime_error(std::string("team policy: ") + library_->Functions().teamPolicyError(policy_));
    }
}

TeamLearner::TeamLearner(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project,
    std::uint64_t seed)
    : library_(std::move(library))
{
    RequireKind(*library_, true);
    std::array<char, ErrorCapacity> error{};
    auto const& functions = library_->Functions();
    if (library_->graphContract_)
    {
        learner_ = functions.graphTeamLearnerCreate(project.string().c_str(), seed, *library_->graphContract_,
            error.data(), error.size());
    }
    else
    {
        learner_ = functions.teamLearnerCreate(project.string().c_str(), seed, error.data(), error.size());
    }
    if (learner_ == nullptr)
    {
        throw std::runtime_error(std::string("team learner: ") + error.data());
    }
}

TeamLearner::~TeamLearner()
{
    library_->Functions().teamLearnerDestroy(learner_);
}

void TeamLearner::Require(int code, char const* call) const
{
    if (code != 0)
    {
        throw std::runtime_error(
            std::string("team learner ") + call + ": " + library_->Functions().teamLearnerError(learner_));
    }
}

void TeamLearner::Act(float const* observations, std::uint8_t const* available, float* actions, bool deterministic)
{
    Require(library_->Functions().teamLearnerAct(learner_, observations, available, actions, deterministic ? 1 : 0),
        "act");
}

bool TeamLearner::Observe(float const* observations, std::uint8_t const* available, double const* rewards,
    std::uint32_t end)
{
    int updated = 0;
    Require(library_->Functions().teamLearnerObserve(learner_, observations, available, rewards, end, &updated),
        "observe");
    return updated != 0;
}

PpokitUpdateStatistics TeamLearner::Statistics() const
{
    PpokitUpdateStatistics statistics{};
    Require(library_->Functions().teamLearnerStatistics(learner_, &statistics), "statistics");
    return statistics;
}

PpokitTeamSettings TeamLearner::Settings() const
{
    PpokitTeamSettings settings{};
    Require(library_->Functions().teamLearnerSettings(learner_, &settings), "settings");
    return settings;
}

std::vector<PpokitBlockProbe> TeamLearner::Probes() const
{
    std::uint32_t const count = library_->Functions().teamLearnerProbeCount(learner_);
    std::vector<PpokitBlockProbe> probes(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        Require(library_->Functions().teamLearnerProbe(learner_, index, &probes[index]), "probe");
    }
    return probes;
}

void TeamLearner::Save(std::filesystem::path const& trainingFile, std::filesystem::path const& policyFile)
{
    std::string const training = trainingFile.string();
    std::string const policy = policyFile.string();
    Require(library_->Functions().teamLearnerSave(learner_, training.empty() ? nullptr : training.c_str(),
                policy.empty() ? nullptr : policy.c_str()),
        "save");
}

void TeamLearner::Load(std::filesystem::path const& trainingFile)
{
    Require(library_->Functions().teamLearnerLoad(learner_, trainingFile.string().c_str()), "load");
}

std::uint64_t TeamLearner::Updates() const
{
    return library_->Functions().teamLearnerUpdates(learner_);
}

} // namespace ppokit::network
