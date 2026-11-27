/**
 * @file PPOKit/include/ppokit/network/network_library.hpp
 * @brief C++ side of a network DLL (ppokit/network/network_abi.h): load it, then use its
 *        policy or learner as RAII objects. The client never sees the backend.
 *
 * @code
 * // task is the validated schema::TaskSchema of the project.
 * auto const library = ppokit::network::NetworkLibrary::Open("ppokit_graph_network", {}, &task);
 * ppokit::network::Policy policy(library, "projects/T6", "projects/T6/checkpoints/policy.ppokit");
 * policy.Act(observation, action, true);
 * @endcode
 */
#pragma once

#include "ppokit/network/network_abi.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::schema { struct TaskSchema; }

namespace ppokit::network {

/** @brief Function table of a loaded network DLL. */
struct NetworkFunctions final
{
    decltype(&ppokit_network_info) info = nullptr;
    decltype(&ppokit_policy_open) policyOpen = nullptr;
    PpokitGraphPolicyOpen graphPolicyOpen = nullptr;
    decltype(&ppokit_policy_act) policyAct = nullptr;
    decltype(&ppokit_policy_error) policyError = nullptr;
    decltype(&ppokit_policy_close) policyClose = nullptr;
    decltype(&ppokit_learner_create) learnerCreate = nullptr;
    PpokitGraphLearnerCreate graphLearnerCreate = nullptr;
    decltype(&ppokit_learner_decide) learnerDecide = nullptr;
    decltype(&ppokit_learner_value) learnerValue = nullptr;
    decltype(&ppokit_learner_append) learnerAppend = nullptr;
    decltype(&ppokit_learner_full) learnerFull = nullptr;
    decltype(&ppokit_learner_update) learnerUpdate = nullptr;
    decltype(&ppokit_learner_save) learnerSave = nullptr;
    decltype(&ppokit_learner_load) learnerLoad = nullptr;
    decltype(&ppokit_learner_updates) learnerUpdates = nullptr;
    decltype(&ppokit_learner_error) learnerError = nullptr;
    decltype(&ppokit_learner_destroy) learnerDestroy = nullptr;
    decltype(&ppokit_team_policy_open) teamPolicyOpen = nullptr;
    PpokitGraphTeamPolicyOpen graphTeamPolicyOpen = nullptr;
    decltype(&ppokit_team_policy_act) teamPolicyAct = nullptr;
    decltype(&ppokit_team_policy_error) teamPolicyError = nullptr;
    decltype(&ppokit_team_policy_close) teamPolicyClose = nullptr;
    decltype(&ppokit_team_learner_create) teamLearnerCreate = nullptr;
    PpokitGraphTeamLearnerCreate graphTeamLearnerCreate = nullptr;
    decltype(&ppokit_team_learner_act) teamLearnerAct = nullptr;
    decltype(&ppokit_team_learner_observe) teamLearnerObserve = nullptr;
    decltype(&ppokit_team_learner_statistics) teamLearnerStatistics = nullptr;
    decltype(&ppokit_team_learner_settings) teamLearnerSettings = nullptr;
    decltype(&ppokit_team_learner_probe_count) teamLearnerProbeCount = nullptr;
    decltype(&ppokit_team_learner_probe) teamLearnerProbe = nullptr;
    decltype(&ppokit_team_learner_save) teamLearnerSave = nullptr;
    decltype(&ppokit_team_learner_load) teamLearnerLoad = nullptr;
    decltype(&ppokit_team_learner_updates) teamLearnerUpdates = nullptr;
    decltype(&ppokit_team_learner_error) teamLearnerError = nullptr;
    decltype(&ppokit_team_learner_destroy) teamLearnerDestroy = nullptr;
};

/** @brief A loaded network DLL; it stays loaded while a policy or a learner uses it. */
class NetworkLibrary final
{
public:
    /**
     * @brief Loads a network DLL.
     * @param reference A path, or a bare name looked up in `directory` with the platform's
     *        suffix (.dll, .so, .dylib).
     * @throws std::runtime_error when it is missing, is not a network DLL, or was built for
     *         another ABI. A single-agent DLL (agentCapacity 1) has the policy and learner
     *         functions bound, a team DLL the team functions.
     * @param task Required for the shared graph backend; its validated contract and dimensions
     *        are bound to this library instance, never to mutable DLL-wide state.
     */
    static std::shared_ptr<NetworkLibrary const> Open(std::string const& reference,
        std::filesystem::path const& directory = {}, schema::TaskSchema const* task = nullptr);

    /** @brief Check the immutable semantic identity of the loaded module against task.json. */
    void VerifyTaskContract(std::uint64_t expected) const;

    ~NetworkLibrary();
    NetworkLibrary(NetworkLibrary const&) = delete;
    NetworkLibrary& operator=(NetworkLibrary const&) = delete;

    PpokitNetworkInfo const& Info() const noexcept { return info_; }
    NetworkFunctions const& Functions() const noexcept { return functions_; }
    std::filesystem::path const& Path() const noexcept { return path_; }

private:
    friend class Policy;
    friend class Learner;
    friend class TeamPolicy;
    friend class TeamLearner;
    NetworkLibrary() = default;

    std::filesystem::path path_;
    void* handle_ = nullptr;
    NetworkFunctions functions_;
    PpokitNetworkInfo info_{};
    std::string taskName_;
    std::optional<std::uint64_t> graphContract_;
};

/** @brief A policy for inference. */
class Policy final
{
public:
    /** @throws std::runtime_error with the DLL's reason. */
    Policy(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project,
        std::filesystem::path const& policyFile);
    ~Policy();
    Policy(Policy const&) = delete;
    Policy& operator=(Policy const&) = delete;

    /** @brief Writes the action for `observation`; the mean when `deterministic`. @throws std::runtime_error. */
    void Act(float const* observation, float* action, bool deterministic);

private:
    std::shared_ptr<NetworkLibrary const> library_;
    PpokitPolicy* policy_ = nullptr;
};

/** @brief A learner: decide, append, update, save, resume. Every call throws the DLL's reason. */
class Learner final
{
public:
    Learner(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project, std::uint64_t seed = 0);
    ~Learner();
    Learner(Learner const&) = delete;
    Learner& operator=(Learner const&) = delete;

    /** @brief Samples an action; writes its log-probability and the critic value. */
    void Decide(float const* observation, float* action, double& logProbability, double& value);
    double Value(float const* observation);
    void Append(PpokitTransition const& transition);
    bool Full() const;
    PpokitUpdateStatistics Update();
    /** @brief Empty paths skip a file. */
    void Save(std::filesystem::path const& trainingFile, std::filesystem::path const& policyFile);
    void Load(std::filesystem::path const& trainingFile);
    std::uint64_t Updates() const;
    PpokitNetworkInfo const& Info() const noexcept { return library_->Info(); }

private:
    /** @brief Throws the learner's error when `code` is not 0. */
    void Require(int code, char const* call) const;

    std::shared_ptr<NetworkLibrary const> library_;
    PpokitLearner* learner_ = nullptr;
};

/** @brief A team policy for inference: deterministic actions of the present agents. */
class TeamPolicy final
{
public:
    /** @throws std::runtime_error with the DLL's reason. */
    TeamPolicy(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project,
        std::filesystem::path const& policyFile);
    ~TeamPolicy();
    TeamPolicy(TeamPolicy const&) = delete;
    TeamPolicy& operator=(TeamPolicy const&) = delete;

    /** @brief agentCapacity rows of observations and actions. @throws std::runtime_error. */
    void Act(float const* observations, std::uint8_t const* available, float* actions);

private:
    std::shared_ptr<NetworkLibrary const> library_;
    PpokitTeamPolicy* policy_ = nullptr;
};

/** @brief A team learner: act, observe (updates by itself), save, resume. Every call throws the DLL's reason. */
class TeamLearner final
{
public:
    TeamLearner(std::shared_ptr<NetworkLibrary const> library, std::filesystem::path const& project,
        std::uint64_t seed = 0);
    ~TeamLearner();
    TeamLearner(TeamLearner const&) = delete;
    TeamLearner& operator=(TeamLearner const&) = delete;

    /** @brief Sampled actions (recorded) or, when `deterministic`, the means (nothing recorded). */
    void Act(float const* observations, std::uint8_t const* available, float* actions, bool deterministic);
    /** @return True when the step completed the rollout and an update was made. */
    bool Observe(float const* observations, std::uint8_t const* available, double const* rewards,
        std::uint32_t end);
    PpokitUpdateStatistics Statistics() const;
    PpokitTeamSettings Settings() const;
    /** @brief Probes of the last update, actor's then critic's (empty when it was not probed). */
    std::vector<PpokitBlockProbe> Probes() const;
    /** @brief Empty paths skip a file. */
    void Save(std::filesystem::path const& trainingFile, std::filesystem::path const& policyFile);
    void Load(std::filesystem::path const& trainingFile);
    std::uint64_t Updates() const;
    PpokitNetworkInfo const& Info() const noexcept { return library_->Info(); }

private:
    void Require(int code, char const* call) const;

    std::shared_ptr<NetworkLibrary const> library_;
    PpokitTeamLearner* learner_ = nullptr;
};

} // namespace ppokit::network
