/**
 * @file PPOKit/include/ppokit/environment/environment_abi.h
 * @brief Stable C API of an environment DLL: the simulation of one example, as ppokit-run
 *        drives it (reset, step, reward channels, world state for the observers).
 * @details An example writes its environment as a class and exports it with the C++ SDK
 * (ppokit/environment/sdk.hpp), which fills this API. ppokit-run knows nothing else of the
 * example. Every buffer belongs to the caller. Functions return 0 on success; on failure
 * ppokit_environment_error explains.
 *
 * Agents: an environment holds up to agentCapacity agents (task.json "agents" max; 1 for
 * a single agent). Observations, actions, rewards, reward channels and the world are laid
 * out agent after agent, always agentCapacity rows (the world: what the observers see of
 * each agent's surroundings, worldSize values per row); `available` (one byte per agent) says which rows
 * hold a present agent: one that observes now and acts at the next step. A row of an absent
 * agent is ignored. Every present agent acts at the same step; the team's episode ends for
 * all at once. A single-agent environment is the case agentCapacity = 1.
 *
 * Evaluation (the suites of eval/): for each seed of a suite, ppokit-run creates an
 * environment with that seed, starts each episode with ppokit_environment_reset_evaluation
 * (the episode's place in the campaign lets the environment spread its starts, e.g. a share
 * of ground starts), and reads ppokit_environment_episode_metrics after the episode ends.
 *
 * Learning: ppokit-run says, with ppokit_environment_set_learning, whether the episodes that
 * follow feed a learner (train) or only show a policy (play, evaluation). State that only
 * training moves, such as a curriculum of start difficulty, stays as it is while not learning.
 *
 * Training state: that state belongs to the training run. ppokit-run reads it
 * (ppokit_environment_training_state) each time it saves the run's checkpoint and keeps it
 * beside it, gives it back when the run resumes (ppokit_environment_restore_training_state),
 * and gives the empty state to a run that starts afresh. An environment without such state
 * writes an empty state and accepts only the empty one.
 *
 * Curriculum: the common runtime owns stages, promotion and persistence. The environment
 * only applies the selected project-specific payload through
 * ppokit_environment_apply_curriculum_stage.
 */
#ifndef PPOKIT_ENVIRONMENT_ABI_H
#define PPOKIT_ENVIRONMENT_ABI_H

#include "ppokit/abi/step_end.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief ABI version; ppokit-run refuses a DLL of another one. */
#define PPOKIT_ENVIRONMENT_ABI 7u

#if defined(_WIN32)
#define PPOKIT_ENVIRONMENT_EXPORT __declspec(dllexport)
#else
#define PPOKIT_ENVIRONMENT_EXPORT __attribute__((visibility("default")))
#endif

/** @brief Sizes of the signals of a step, as task.json declares them. */
typedef struct PpokitEnvironmentInfo
{
    uint32_t abi;
    /** @brief Rows of agents of every buffer (1 for a single agent). */
    uint32_t agentCapacity;
    /** @brief Values of one agent's observation (one row). */
    uint32_t observationSize;
    uint32_t actionSize;
    uint32_t rewardChannels;
    /** @brief Values of one row of the world: task.json "world" (0 when the task has none). */
    uint32_t worldSize;
    /** @brief Number of task.json "metrics" (0 when the task has none). */
    uint32_t metricCount;
    /** @brief Task name of task.json. */
    const char* task;
} PpokitEnvironmentInfo;

typedef struct PpokitEnvironment PpokitEnvironment;

PPOKIT_ENVIRONMENT_EXPORT void ppokit_environment_info(PpokitEnvironmentInfo* info);
/**
 * @brief Creates the environment of a project.
 * @param project Project folder (its settings file, task.json...).
 * @param seed Seed of the episodes.
 */
PPOKIT_ENVIRONMENT_EXPORT PpokitEnvironment* ppokit_environment_create(const char* project, uint64_t seed,
    char* error, size_t errorSize);
/**
 * @brief Starts an episode: writes the first observations, the agents present and the world
 *        (world may be NULL).
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_reset(PpokitEnvironment* environment, float* observations,
    uint8_t* available, float* world);
/**
 * @brief Starts episode `episode` of an evaluation campaign of `episodes` episodes, from
 *        `seed`: the same arguments give the same episode. Writes the first observation and world.
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_reset_evaluation(PpokitEnvironment* environment, uint32_t seed,
    uint32_t episode, uint32_t episodes, float* observations, uint8_t* available, float* world);
/**
 * @brief Before each step that follows a step that did not end the episode: rewrites the
 *        observations, agents present and world the agents act on, when the world changes
 *        between two steps (e.g. rules applied before a step, an agent appearing). An
 *        environment without such changes leaves the buffers as its last step wrote them.
 *        The first step of an episode acts on what the reset wrote.
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_begin_step(PpokitEnvironment* environment, float* observations,
    uint8_t* available, float* world);
/**
 * @brief Metrics of the episode that just ended (metricCount values, task.json order): 0 or
 *        1 for a flag, a number or NaN (does not apply) for a number.
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_episode_metrics(const PpokitEnvironment* environment,
    double* metrics);
/**
 * @brief Applies the actions of the agents present (as `available` said before this step).
 * @param observations, available Receive the next observations and the agents now present.
 * @param rewards Receives the reward each agent of the step learns from (agentCapacity values;
 *        0 for an agent that did not act).
 * @param rewardChannels Receives the weighted value of every reward channel of the step, per
 *        agent (agentCapacity x rewardChannels). An agent's reward is the sum of its channels
 *        unless the task bounds it (e.g. a clamp), which `rewards` already applies.
 * @param end Receives PPOKIT_STEP_RUNNING (0), _TERMINATED (1) or _TRUNCATED (2), for the team.
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_step(PpokitEnvironment* environment, const float* actions,
    float* observations, uint8_t* available, float* world, double* rewards, double* rewardChannels, uint32_t* end);
/**
 * @brief Whether the episodes that follow feed a learner (1: train) or only play a policy
 *        (0: play, evaluation). Set by ppokit-run right after create; until then, 1.
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_set_learning(PpokitEnvironment* environment, int learning);
/**
 * @brief The training state (text the environment reads back), for the run's checkpoint.
 * @param buffer Receives the state when `capacity` is enough; may be NULL when `capacity` is 0.
 * @param size Receives the size of the state in bytes (no terminating zero).
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_training_state(const PpokitEnvironment* environment, char* buffer,
    size_t capacity, size_t* size);
/**
 * @brief Restores a training state written by ppokit_environment_training_state; the empty
 *        state (`size` 0) is the state of a run that starts afresh.
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_restore_training_state(PpokitEnvironment* environment,
    const char* state, size_t size);
/**
 * @brief Applies a stage selected by PPOKit's common curriculum engine.
 * @param name Stable stage name.
 * @param payload Project-specific JSON object from the common plan. Environments without
 *        stage settings accept only the empty object.
 */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_apply_curriculum_stage(PpokitEnvironment* environment,
    const char* name, const char* payload);
/** @brief Weights of the reward channels for the following steps. */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_set_weights(PpokitEnvironment* environment, const double* weights);
/** @brief Current weights (rewardChannels values). */
PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_weights(const PpokitEnvironment* environment, double* weights);
PPOKIT_ENVIRONMENT_EXPORT const char* ppokit_environment_error(const PpokitEnvironment* environment);
PPOKIT_ENVIRONMENT_EXPORT void ppokit_environment_destroy(PpokitEnvironment* environment);

#ifdef __cplusplus
}
#endif

#endif /* PPOKIT_ENVIRONMENT_ABI_H */
