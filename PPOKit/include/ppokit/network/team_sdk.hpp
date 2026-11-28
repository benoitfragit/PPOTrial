/**
 * @file PPOKit/include/ppokit/network/team_sdk.hpp
 * @brief C++ SDK to write the network DLL of a team (network_abi.h, ppokit_team_*): two
 *        classes in, the C API out.
 * @details Header-only, C++17. The DLL keeps its own networks and learning engine (e.g. a
 * MAPPO learner with a centralized critic); ppokit-run sees only rows of agents. Every
 * exception is caught before the C boundary and becomes the handle's error text.
 *
 * @code
 * class TeamLearner final
 * {
 * public:
 *     TeamLearner(std::filesystem::path const& project, std::uint64_t seed);   // seed 0: the project's
 *     // Sampled actions (the step then awaits Observe), or the means when deterministic (nothing recorded).
 *     void Act(float const* observations, std::uint8_t const* available, float* actions, bool deterministic);
 *     // The step after the sampled actions; returns true when it completed the rollout and an update was made.
 *     bool Observe(float const* observations, std::uint8_t const* available, double const* rewards,
 *                  ppokit::runtime::StepEnd end);
 *     PpokitUpdateStatistics Statistics() const;                                // of the last update
 *     PpokitTeamSettings Settings() const;                                      // PPO settings, for the journal
 *     std::size_t ProbeCount() const;                                           // blocks probed in the last update
 *     PpokitBlockProbe Probe(std::size_t index) const;                          // (0 when not probed)
 *     void Save(std::filesystem::path const& trainingFile,                      // empty: skipped
 *               std::filesystem::path const& policyFile) const;
 *     void Load(std::filesystem::path const& trainingFile);
 *     std::uint64_t Updates() const;
 * };
 * class TeamPolicy final
 * {
 * public:
 *     TeamPolicy(std::filesystem::path const& project, std::filesystem::path const& policyFile);
 *     void Act(float const* observations, std::uint8_t const* available, float* actions);   // deterministic
 * };
 * PPOKIT_TEAM_NETWORK(TeamLearner, TeamPolicy, "blue_patrol", 9, ObservationSize, ActionSize, "rltools")
 * @endcode
 */
#pragma once

// A native team DLL is bound to its task with ppokit_bind_task_contract(TARGET name TASK task.json).

#include "ppokit/network/network_abi.h"
#include "ppokit/runtime/rollout.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace ppokit::network::team_sdk {

/** @brief What a C handle points to: the object and the text of its last error. */
template<typename Object>
struct Handle final
{
    std::unique_ptr<Object> object;
    std::string error;
};

inline void CopyError(std::string const& message, char* error, std::size_t errorSize) noexcept
{
    if (error != nullptr && errorSize > 0)
    {
        std::snprintf(error, errorSize, "%s", message.c_str());
    }
}

/** @brief Builds an object behind a C handle; an exception becomes `error` and a null handle. */
template<typename Opaque, typename Object, typename... Arguments>
Opaque* Create(char* error, std::size_t errorSize, Arguments&&... arguments) noexcept
{
    try
    {
        auto handle = std::make_unique<Handle<Object>>();
        handle->object = std::make_unique<Object>(std::forward<Arguments>(arguments)...);
        return reinterpret_cast<Opaque*>(handle.release());
    }
    catch (std::exception const& failure)
    {
        CopyError(failure.what(), error, errorSize);
    }
    catch (...)
    {
        CopyError("unknown exception", error, errorSize);
    }
    return nullptr;
}

/** @brief Runs `body` on a handle's object; an exception becomes return code 1 and the handle's error. */
template<typename Object, typename Opaque, typename Body>
int Guarded(Opaque const* opaque, Body&& body) noexcept
{
    auto* const handle = reinterpret_cast<Handle<Object>*>(const_cast<Opaque*>(opaque));
    try
    {
        body(*handle->object);
        return 0;
    }
    catch (std::exception const& failure)
    {
        handle->error = failure.what();
    }
    catch (...)
    {
        handle->error = "unknown exception";
    }
    return 1;
}

template<typename Object, typename Opaque>
char const* Error(Opaque const* opaque) noexcept
{
    return reinterpret_cast<Handle<Object> const*>(opaque)->error.c_str();
}

template<typename Object, typename Opaque>
void Destroy(Opaque* opaque) noexcept
{
    delete reinterpret_cast<Handle<Object>*>(opaque);
}

/** @brief The C end code as a StepEnd. @throws std::invalid_argument for an unknown code. */
inline runtime::StepEnd ToStepEnd(std::uint32_t end)
{
    switch (end)
    {
    case PPOKIT_STEP_RUNNING:
        return runtime::StepEnd::Running;
    case PPOKIT_STEP_TERMINATED:
        return runtime::StepEnd::Terminated;
    case PPOKIT_STEP_TRUNCATED:
        return runtime::StepEnd::Truncated;
    default:
        throw std::invalid_argument("unknown step end " + std::to_string(end));
    }
}

/** @brief A C path argument, empty when null. */
inline std::filesystem::path OptionalPath(char const* path)
{
    return path == nullptr ? std::filesystem::path{} : std::filesystem::path(path);
}

} // namespace ppokit::network::team_sdk

/** @brief Exports `Learner` and `Policy` as this DLL's team networks. */
#define PPOKIT_TEAM_NETWORK(Learner, Policy, TaskName, Agents, ObservationSize, ActionSize, Backend)                \
    extern "C" PPOKIT_NETWORK_EXPORT void ppokit_network_info(PpokitNetworkInfo* info)                              \
    {                                                                                                               \
        static_assert((Agents) > 1, "a team network acts for more than one agent");                                 \
        info->abi = PPOKIT_NETWORK_ABI;                                                                             \
        info->agentCapacity = static_cast<std::uint32_t>(Agents);                                                   \
        info->observationSize = static_cast<std::uint32_t>(ObservationSize);                                        \
        info->actionSize = static_cast<std::uint32_t>(ActionSize);                                                  \
        info->task = TaskName;                                                                                      \
        info->backend = Backend;                                                                                    \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT PpokitTeamPolicy* ppokit_team_policy_open(                                     \
        char const* project, char const* policyFile, char* error, std::size_t errorSize)                            \
    {                                                                                                               \
        return ppokit::network::team_sdk::Create<PpokitTeamPolicy, Policy>(                                         \
            error, errorSize, std::filesystem::path(project), std::filesystem::path(policyFile));                  \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_policy_act(                                                    \
        PpokitTeamPolicy* policy, float const* observations, std::uint8_t const* available, float* actions)        \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Policy>(                                                          \
            policy, [&](Policy& self) { self.Act(observations, available, actions); });                            \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT char const* ppokit_team_policy_error(PpokitTeamPolicy const* policy)          \
    {                                                                                                               \
        return ppokit::network::team_sdk::Error<Policy>(policy);                                                    \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT void ppokit_team_policy_close(PpokitTeamPolicy* policy)                       \
    {                                                                                                               \
        ppokit::network::team_sdk::Destroy<Policy>(policy);                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT PpokitTeamLearner* ppokit_team_learner_create(                                 \
        char const* project, std::uint64_t seed, char* error, std::size_t errorSize)                                \
    {                                                                                                               \
        return ppokit::network::team_sdk::Create<PpokitTeamLearner, Learner>(                                       \
            error, errorSize, std::filesystem::path(project), seed);                                                \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_learner_act(PpokitTeamLearner* learner,                       \
        float const* observations, std::uint8_t const* available, float* actions, int deterministic)               \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Learner>(learner,                                                 \
            [&](Learner& self) { self.Act(observations, available, actions, deterministic != 0); });               \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_learner_observe(PpokitTeamLearner* learner,                   \
        float const* observations, std::uint8_t const* available, double const* rewards, std::uint32_t end,        \
        int* updated)                                                                                               \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Learner>(learner, [&](Learner& self) {                           \
            *updated = self.Observe(observations, available, rewards, ppokit::network::team_sdk::ToStepEnd(end))   \
                ? 1                                                                                                 \
                : 0;                                                                                                \
        });                                                                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_learner_statistics(                                            \
        PpokitTeamLearner const* learner, PpokitUpdateStatistics* statistics)                                       \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Learner>(                                                         \
            learner, [&](Learner& self) { *statistics = self.Statistics(); });                                     \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_learner_settings(                                              \
        PpokitTeamLearner const* learner, PpokitTeamSettings* settings)                                             \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Learner>(                                                         \
            learner, [&](Learner& self) { *settings = self.Settings(); });                                         \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT std::uint32_t ppokit_team_learner_probe_count(PpokitTeamLearner const* learner) \
    {                                                                                                               \
        auto const* handle = reinterpret_cast<ppokit::network::team_sdk::Handle<Learner> const*>(learner);        \
        return static_cast<std::uint32_t>(handle->object->ProbeCount());                                           \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_learner_probe(                                                 \
        PpokitTeamLearner const* learner, std::uint32_t index, PpokitBlockProbe* probe)                             \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Learner>(                                                         \
            learner, [&](Learner& self) { *probe = self.Probe(index); });                                          \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_learner_save(                                                  \
        PpokitTeamLearner* learner, char const* trainingFile, char const* policyFile)                               \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Learner>(learner, [&](Learner& self) {                           \
            self.Save(ppokit::network::team_sdk::OptionalPath(trainingFile),                                        \
                ppokit::network::team_sdk::OptionalPath(policyFile));                                               \
        });                                                                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT int ppokit_team_learner_load(PpokitTeamLearner* learner, char const* file)    \
    {                                                                                                               \
        return ppokit::network::team_sdk::Guarded<Learner>(                                                         \
            learner, [&](Learner& self) { self.Load(std::filesystem::path(file)); });                              \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT std::uint64_t ppokit_team_learner_updates(PpokitTeamLearner const* learner)   \
    {                                                                                                               \
        auto const* handle = reinterpret_cast<ppokit::network::team_sdk::Handle<Learner> const*>(learner);        \
        return handle->object->Updates();                                                                           \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT char const* ppokit_team_learner_error(PpokitTeamLearner const* learner)       \
    {                                                                                                               \
        return ppokit::network::team_sdk::Error<Learner>(learner);                                                  \
    }                                                                                                               \
    extern "C" PPOKIT_NETWORK_EXPORT void ppokit_team_learner_destroy(PpokitTeamLearner* learner)                  \
    {                                                                                                               \
        ppokit::network::team_sdk::Destroy<Learner>(learner);                                                       \
    }
