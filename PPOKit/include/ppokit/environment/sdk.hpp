/**
 * @file PPOKit/include/ppokit/environment/sdk.hpp
 * @brief C++ SDK to write an environment DLL: a class in, the C API out.
 * @details Header-only, C++17. Write a class with the members below, then export it once
 * with PPOKIT_ENVIRONMENT, giving the sizes of its task (usually the generated layout's).
 * Bind the DLL to its task at build time with ppokit_bind_task_contract(TARGET name TASK task.json).
 * The SDK catches every exception before the C boundary and turns it into the error text.
 *
 * @code
 * class Reach final
 * {
 * public:
 *     Reach(std::filesystem::path const& project, std::uint64_t seed);
 *     void Reset(float* observation, float* world);            // world: WorldSize values, or null
 *     ppokit::runtime::StepEnd Step(float const* action, float* observation, float* world,
 *                                   double* rewardChannels);    // weighted value of each channel
 *     void SetWeights(double const* weights);
 *     void Weights(double* weights) const;
 * };
 * PPOKIT_ENVIRONMENT(Reach, "reach", Layout::Observation::Size, Layout::Action::Size, 3, 0)
 * @endcode
 *
 * Optional members, for the evaluation suites (eval/):
 *
 * @code
 *     static constexpr std::size_t MetricCount = 2;                 // task.json "metrics"
 *     void EpisodeMetrics(double* metrics) const;                   // after the episode ended
 *     void ResetEvaluation(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes,
 *                          float* observation, float* world);       // a campaign's episode
 * @endcode
 *
 * Without ResetEvaluation an evaluation episode starts with Reset (the environment was
 * created with the suite's seed); without MetricCount the environment reports no metric.
 *
 * Optional, for state that only training moves (a curriculum): ppokit-run says whether the
 * episodes that follow feed a learner (train) or only play a policy (play, evaluation).
 *
 * @code
 *     void SetLearning(bool learning);
 * @endcode
 *
 * Optional, when that state must follow the training run (ppokit-run saves it with the run's
 * checkpoint, restores it when the run resumes, and restores the empty state for a fresh run):
 *
 * @code
 *     std::string TrainingState() const;                            // any text
 *     void RestoreTrainingState(std::string const& state);          // "" = a fresh run
 * @endcode
 *
 * A team of agents (task.json "agents" max above 1) declares its capacity and works on
 * rows of agents; `available` (one byte per agent) says which agents are present:
 *
 * @code
 *     static constexpr std::size_t Agents = 9;                      // rows of every buffer
 *     void Reset(float* observations, std::uint8_t* available, float* world);
 *     ppokit::runtime::StepEnd Step(float const* actions, float* observations, std::uint8_t* available,
 *                                   float* world, double* rewards, double* rewardChannels);
 *     void ResetEvaluation(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes,
 *                          float* observations, std::uint8_t* available, float* world);   // optional
 *     // Optional: the world changes between two steps (rules before a step); rewrite what the
 *     // agents act on. Called before each step except an episode's first.
 *     void BeginStep(float* observations, std::uint8_t* available, float* world);
 * @endcode
 *
 * A single-agent class keeps the members above: its agent is always present and its reward
 * is the sum of its channels.
 */
#pragma once

#include "ppokit/environment/environment_abi.h"
#include "ppokit/runtime/rollout.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ppokit::environment::sdk {

/** @brief Whether Environment declares MetricCount. */
template<typename Environment, typename = void>
struct HasMetrics : std::false_type
{
};
template<typename Environment>
struct HasMetrics<Environment, std::void_t<decltype(Environment::MetricCount)>> : std::true_type
{
};

/** @brief Whether Environment declares Agents: a team, working on rows of agents. */
template<typename Environment, typename = void>
struct IsTeam : std::false_type
{
};
template<typename Environment>
struct IsTeam<Environment, std::void_t<decltype(Environment::Agents)>> : std::true_type
{
};

/** @brief Rows of agents of Environment's buffers (1 for a single agent). */
template<typename Environment>
constexpr std::size_t AgentCapacityOf() noexcept
{
    if constexpr (IsTeam<Environment>::value)
    {
        return Environment::Agents;
    }
    else
    {
        return 1;
    }
}

/** @brief Whether a team has BeginStep(observations, available, world). */
template<typename Environment, typename = void>
struct HasBeginStep : std::false_type
{
};
template<typename Environment>
struct HasBeginStep<Environment,
    std::void_t<decltype(std::declval<Environment&>().BeginStep(static_cast<float*>(nullptr),
        static_cast<std::uint8_t*>(nullptr), static_cast<float*>(nullptr)))>> : std::true_type
{
};

/** @brief Before a step that is not an episode's first: BeginStep when the class has it, else nothing. */
template<typename Environment>
void BeginNextStep(Environment& environment, float* observations, std::uint8_t* available, float* world)
{
    if constexpr (HasBeginStep<Environment>::value)
    {
        environment.BeginStep(observations, available, world);
    }
}

/** @brief Whether a single agent has ResetEvaluation(seed, episode, episodes, observation, world). */
template<typename Environment, typename = void>
struct HasEvaluationReset : std::false_type
{
};
template<typename Environment>
struct HasEvaluationReset<Environment,
    std::void_t<decltype(std::declval<Environment&>().ResetEvaluation(std::uint32_t{}, std::uint32_t{},
        std::uint32_t{}, static_cast<float*>(nullptr), static_cast<float*>(nullptr)))>> : std::true_type
{
};

/** @brief Whether a team has ResetEvaluation(seed, episode, episodes, observations, available, world). */
template<typename Environment, typename = void>
struct HasTeamEvaluationReset : std::false_type
{
};
template<typename Environment>
struct HasTeamEvaluationReset<Environment,
    std::void_t<decltype(std::declval<Environment&>().ResetEvaluation(std::uint32_t{}, std::uint32_t{},
        std::uint32_t{}, static_cast<float*>(nullptr), static_cast<std::uint8_t*>(nullptr),
        static_cast<float*>(nullptr)))>> : std::true_type
{
};

/** @brief Whether Environment has SetLearning(bool). */
template<typename Environment, typename = void>
struct HasLearning : std::false_type
{
};
template<typename Environment>
struct HasLearning<Environment, std::void_t<decltype(std::declval<Environment&>().SetLearning(bool{}))>>
    : std::true_type
{
};

/** @brief Tells the class whether it feeds a learner, when it has SetLearning; else nothing. */
template<typename Environment>
void ApplyLearning(Environment& environment, bool learning)
{
    if constexpr (HasLearning<Environment>::value)
    {
        environment.SetLearning(learning);
    }
}

/**
 * @brief Whether Environment has `bool TrainingProgress(updates, campaign)` (a curriculum it
 *        owns; true when the curriculum ends the training).
 */
template<typename Environment, typename = void>
struct HasTrainingProgress : std::false_type
{
};
template<typename Environment>
struct HasTrainingProgress<Environment,
    std::void_t<decltype(bool(std::declval<Environment&>().TrainingProgress(std::uint64_t{},
        std::declval<char const*>())))>> : std::true_type
{
};

/** @brief Hands the run's progress to a class that owns a curriculum; others ignore it. */
template<typename Environment>
bool ApplyTrainingProgress(Environment& environment, std::uint64_t updates, char const* campaign)
{
    if constexpr (HasTrainingProgress<Environment>::value)
    {
        return environment.TrainingProgress(updates, campaign);
    }
    else
    {
        return false;
    }
}

/** @brief Whether Environment has TrainingState() and RestoreTrainingState(state). */
template<typename Environment, typename = void>
struct HasTrainingState : std::false_type
{
};
template<typename Environment>
struct HasTrainingState<Environment,
    std::void_t<decltype(std::string(std::declval<Environment const&>().TrainingState())),
        decltype(std::declval<Environment&>().RestoreTrainingState(std::string{}))>> : std::true_type
{
};

/** @brief The class's training state, or the empty state when it has none. */
template<typename Environment>
std::string TrainingStateOf(Environment const& environment)
{
    if constexpr (HasTrainingState<Environment>::value)
    {
        return environment.TrainingState();
    }
    else
    {
        return {};
    }
}

/**
 * @brief Gives the class the training state of `size` bytes at `state` (NULL for size 0);
 *        a class without one accepts only the empty state.
 */
template<typename Environment>
void RestoreTrainingStateOf(Environment& environment, char const* state, std::size_t size)
{
    if (state == nullptr && size != 0)
    {
        throw std::invalid_argument("a training state of " + std::to_string(size) + " bytes without its text");
    }
    std::string const text = size == 0 ? std::string() : std::string(state, size);
    if constexpr (HasTrainingState<Environment>::value)
    {
        environment.RestoreTrainingState(text);
    }
    else if (!text.empty())
    {
        throw std::runtime_error("this environment keeps no training state");
    }
}

/** @brief Copies `state` to the caller's buffer when it fits; says its size in any case. */
inline void WriteTrainingState(std::string const& state, char* buffer, std::size_t capacity, std::size_t* size) noexcept
{
    *size = state.size();
    if (buffer != nullptr && capacity >= state.size())
    {
        std::memcpy(buffer, state.data(), state.size());
    }
}

/** @brief Number of episode metrics of Environment (0 without MetricCount). */
template<typename Environment>
constexpr std::size_t MetricCountOf() noexcept
{
    if constexpr (HasMetrics<Environment>::value)
    {
        return Environment::MetricCount;
    }
    else
    {
        return 0;
    }
}

/** @brief Starts an episode; a single agent is always present. */
template<typename Environment>
void ResetEpisode(Environment& environment, float* observations, std::uint8_t* available, float* world)
{
    if constexpr (IsTeam<Environment>::value)
    {
        environment.Reset(observations, available, world);
    }
    else
    {
        environment.Reset(observations, world);
        available[0] = 1;
    }
}

/** @brief Starts a campaign's episode: ResetEvaluation when the class has it, else Reset. */
template<typename Environment>
void ResetForEvaluation(Environment& environment, std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes,
    float* observations, std::uint8_t* available, float* world)
{
    if constexpr (HasTeamEvaluationReset<Environment>::value)
    {
        environment.ResetEvaluation(seed, episode, episodes, observations, available, world);
    }
    else if constexpr (HasEvaluationReset<Environment>::value)
    {
        environment.ResetEvaluation(seed, episode, episodes, observations, world);
        available[0] = 1;
    }
    else
    {
        ResetEpisode(environment, observations, available, world);
    }
}

/** @brief One step; a single agent's reward is the sum of its channels. */
template<typename Environment>
runtime::StepEnd StepAgents(Environment& environment, float const* actions, float* observations,
    std::uint8_t* available, float* world, double* rewards, double* rewardChannels, std::size_t channels)
{
    if constexpr (IsTeam<Environment>::value)
    {
        return environment.Step(actions, observations, available, world, rewards, rewardChannels);
    }
    else
    {
        runtime::StepEnd const end = environment.Step(actions, observations, world, rewardChannels);
        double reward = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            reward += rewardChannels[channel];
        }
        rewards[0] = reward;
        available[0] = 1;
        return end;
    }
}

/** @brief Writes the episode metrics when the class reports some. */
template<typename Environment>
void WriteEpisodeMetrics(Environment const& environment, double* metrics)
{
    if constexpr (HasMetrics<Environment>::value)
    {
        environment.EpisodeMetrics(metrics);
    }
}

/** @brief What the C handle points to: the environment and the text of its last error. */
template<typename Environment>
struct Handle final
{
    std::unique_ptr<Environment> environment;
    std::string error;
};

inline void CopyError(std::string const& message, char* error, std::size_t errorSize) noexcept
{
    if (error != nullptr && errorSize > 0)
    {
        std::snprintf(error, errorSize, "%s", message.c_str());
    }
}

/** @brief Runs `body` on a handle; an exception becomes return code 1 and the handle's error. */
template<typename Environment, typename Body>
int Guarded(PpokitEnvironment* opaque, Body&& body) noexcept
{
    auto* const handle = reinterpret_cast<Handle<Environment>*>(opaque);
    try
    {
        body(*handle->environment);
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

template<typename Environment>
PpokitEnvironment* Create(char const* project, std::uint64_t seed, char* error, std::size_t errorSize) noexcept
{
    try
    {
        auto handle = std::make_unique<Handle<Environment>>();
        handle->environment = std::make_unique<Environment>(std::filesystem::path(project), seed);
        return reinterpret_cast<PpokitEnvironment*>(handle.release());
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

template<typename Environment>
int Step(PpokitEnvironment* opaque, float const* actions, float* observations, std::uint8_t* available, float* world,
    double* rewards, double* rewardChannels, std::size_t channels, std::uint32_t* end) noexcept
{
    return Guarded<Environment>(opaque, [&](Environment& environment) {
        *end = static_cast<std::uint32_t>(
            StepAgents(environment, actions, observations, available, world, rewards, rewardChannels, channels));
    });
}

template<typename Environment>
char const* Error(PpokitEnvironment const* opaque) noexcept
{
    return reinterpret_cast<Handle<Environment> const*>(opaque)->error.c_str();
}

template<typename Environment>
void Destroy(PpokitEnvironment* opaque) noexcept
{
    delete reinterpret_cast<Handle<Environment>*>(opaque);
}

} // namespace ppokit::environment::sdk

/** @brief Exports `Class` as this DLL's environment, with the sizes of its task. */
#define PPOKIT_ENVIRONMENT(Class, TaskName, ObservationSize, ActionSize, RewardChannels, WorldSize)                \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT void ppokit_environment_info(PpokitEnvironmentInfo* info)                   \
    {                                                                                                               \
        info->abi = PPOKIT_ENVIRONMENT_ABI;                                                                         \
        info->agentCapacity = static_cast<std::uint32_t>(ppokit::environment::sdk::AgentCapacityOf<Class>());      \
        info->observationSize = static_cast<std::uint32_t>(ObservationSize);                                        \
        info->actionSize = static_cast<std::uint32_t>(ActionSize);                                                  \
        info->rewardChannels = static_cast<std::uint32_t>(RewardChannels);                                          \
        info->worldSize = static_cast<std::uint32_t>(WorldSize);                                                    \
        info->metricCount = static_cast<std::uint32_t>(ppokit::environment::sdk::MetricCountOf<Class>());          \
        info->task = TaskName;                                                                                      \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT PpokitEnvironment* ppokit_environment_create(                              \
        char const* project, std::uint64_t seed, char* error, std::size_t errorSize)                                \
    {                                                                                                               \
        return ppokit::environment::sdk::Create<Class>(project, seed, error, errorSize);                            \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_reset(                                               \
        PpokitEnvironment* environment, float* observations, std::uint8_t* available, float* world)                \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(environment, [&](Class& self) {                            \
            ppokit::environment::sdk::ResetEpisode(self, observations, available, world);                          \
        });                                                                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_reset_evaluation(PpokitEnvironment* environment,   \
        std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes, float* observations,                    \
        std::uint8_t* available, float* world)                                                                      \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(environment, [&](Class& self) {                            \
            ppokit::environment::sdk::ResetForEvaluation(                                                           \
                self, seed, episode, episodes, observations, available, world);                                     \
        });                                                                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_begin_step(                                          \
        PpokitEnvironment* environment, float* observations, std::uint8_t* available, float* world)                \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(environment, [&](Class& self) {                            \
            ppokit::environment::sdk::BeginNextStep(self, observations, available, world);                         \
        });                                                                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_episode_metrics(                                    \
        PpokitEnvironment const* environment, double* metrics)                                                      \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(const_cast<PpokitEnvironment*>(environment),               \
            [&](Class& self) { ppokit::environment::sdk::WriteEpisodeMetrics(self, metrics); });                    \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_step(PpokitEnvironment* environment,               \
        float const* actions, float* observations, std::uint8_t* available, float* world, double* rewards,         \
        double* rewardChannels, std::uint32_t* end)                                                                 \
    {                                                                                                               \
        return ppokit::environment::sdk::Step<Class>(environment, actions, observations, available, world,         \
            rewards, rewardChannels, static_cast<std::size_t>(RewardChannels), end);                                \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_set_learning(                                       \
        PpokitEnvironment* environment, int learning)                                                               \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(environment,                                                \
            [&](Class& self) { ppokit::environment::sdk::ApplyLearning(self, learning != 0); });                    \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_training_state(                                     \
        PpokitEnvironment const* environment, char* buffer, std::size_t capacity, std::size_t* size)              \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(const_cast<PpokitEnvironment*>(environment),               \
            [&](Class& self) {                                                                                      \
                ppokit::environment::sdk::WriteTrainingState(                                                       \
                    ppokit::environment::sdk::TrainingStateOf(self), buffer, capacity, size);                       \
            });                                                                                                     \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_restore_training_state(                             \
        PpokitEnvironment* environment, char const* state, std::size_t size)                                        \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(environment, [&](Class& self) {                            \
            ppokit::environment::sdk::RestoreTrainingStateOf(self, state, size);                                    \
        });                                                                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_training_progress(                                  \
        PpokitEnvironment* environment, std::uint64_t updates, char const* campaign, int* finished)                 \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(environment, [&](Class& self) {                            \
            bool const ended = ppokit::environment::sdk::ApplyTrainingProgress(self, updates, campaign);           \
            *finished = ended ? 1 : 0;                                                                              \
        });                                                                                                         \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_set_weights(                                        \
        PpokitEnvironment* environment, double const* weights)                                                      \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(                                                            \
            environment, [&](Class& self) { self.SetWeights(weights); });                                          \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_weights(                                            \
        PpokitEnvironment const* environment, double* weights)                                                      \
    {                                                                                                               \
        return ppokit::environment::sdk::Guarded<Class>(const_cast<PpokitEnvironment*>(environment),               \
            [&](Class& self) { self.Weights(weights); });                                                           \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT char const* ppokit_environment_error(PpokitEnvironment const* environment) \
    {                                                                                                               \
        return ppokit::environment::sdk::Error<Class>(environment);                                                 \
    }                                                                                                               \
    extern "C" PPOKIT_ENVIRONMENT_EXPORT void ppokit_environment_destroy(PpokitEnvironment* environment)           \
    {                                                                                                               \
        ppokit::environment::sdk::Destroy<Class>(environment);                                                      \
    }
