/**
 * @file PPOKit/include/ppokit/network/network_abi.h
 * @brief Stable C API of a network DLL: the networks of one project, for inference and
 *        for training, whatever computes them inside.
 * @details Declarative single-agent tasks share ppokit_graph_network. Native backends
 * can also expose this ABI for a fixed task. A client sees
 * only plain C types, so any compiler or language can load it. Every buffer belongs to
 * the caller; a call never allocates on the caller's behalf. Functions return 0 on
 * success; on failure the handle's `error` function explains.
 *
 * For a shared graph module, info.backend is PPOKIT_GRAPH_DYNAMIC_BACKEND and its dimensions
 * and task pointer are unset. The client binds a validated task and passes its semantic hash
 * to ppokit_graph_policy_open or ppokit_graph_learner_create (ppokit_graph_team_policy_open or
 * ppokit_graph_team_learner_create for a team task). The module validates the hash before
 * opening a handle; all subsequent calls use the ordinary single-agent (or team) ABI.
 *
 * Inference with a fixed-task module, in a few lines:
 * @code
 * PpokitPolicy* policy = ppokit_policy_open("projects/T6", "projects/T6/checkpoints/policy.ppokit", error, sizeof error);
 * ppokit_policy_act(policy, observation, action, 1);   // 1: deterministic (the mean)
 * ppokit_policy_close(policy);
 * @endcode
 *
 * Training (one trajectory stream): decide, step the environment, append, and update
 * when the rollout is full:
 * @code
 * PpokitLearner* learner = ppokit_learner_create("projects/T6", 0, error, sizeof error);
 * ppokit_learner_decide(learner, observation, action, &logProbability, &value);
 * ... environment step ...
 * ppokit_learner_append(learner, &transition);
 * if (ppokit_learner_full(learner)) ppokit_learner_update(learner, &statistics);
 * @endcode
 *
 * A team (agentCapacity above 1, e.g. MAPPO with a centralized critic) exports the
 * ppokit_team_* functions instead: the learner chooses the actions of every present agent
 * at once, then observes the step that followed and updates by itself when its rollout is
 * full. Buffers have agentCapacity rows, as the environment ABI lays them out; `available`
 * says which rows hold a present agent. How the critic is fed stays inside the learner.
 * @code
 * PpokitTeamLearner* learner = ppokit_team_learner_create("projects/BluePatrol", 0, error, sizeof error);
 * ppokit_team_learner_act(learner, observations, available, actions, 0);
 * ... environment step: observations, available, rewards, end ...
 * ppokit_team_learner_observe(learner, observations, available, rewards, end, &updated);
 * if (updated) ppokit_team_learner_statistics(learner, &statistics);
 * @endcode
 */
#ifndef PPOKIT_NETWORK_ABI_H
#define PPOKIT_NETWORK_ABI_H

#include "ppokit/abi/step_end.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief ABI version; a client refuses a DLL of another one. */
#define PPOKIT_NETWORK_ABI 2u

/**
 * @brief Shared graph module with contract-checked creation functions (version 2: also the
 *        team policy and learner of a team task).
 */
#define PPOKIT_GRAPH_DYNAMIC_BACKEND "graph/dynamic/2"

#if defined(_WIN32)
#define PPOKIT_NETWORK_EXPORT __declspec(dllexport)
#else
#define PPOKIT_NETWORK_EXPORT __attribute__((visibility("default")))
#endif

/** @brief What the DLL computes. */
typedef struct PpokitNetworkInfo
{
    uint32_t abi;
    /** @brief 1: the ppokit_policy_* and ppokit_learner_* functions; above 1: ppokit_team_*. */
    uint32_t agentCapacity;
    /** @brief Values of one agent's observation. */
    uint32_t observationSize;
    /** @brief Values of one agent's action. */
    uint32_t actionSize;
    /** @brief Task name of task.json. */
    const char* task;
    /** @brief Engine inside, e.g. "graph" (PPOKit graph engine) or "rltools". */
    const char* backend;
} PpokitNetworkInfo;

/** @brief One step of experience, in the caller's memory. */
typedef struct PpokitTransition
{
    /** @brief observationSize values, as given to ppokit_learner_decide. */
    const float* observation;
    /** @brief actionSize values, as ppokit_learner_decide wrote them. */
    const float* action;
    double logProbability;
    double value;
    /** @brief Total (weighted) reward of the step. */
    double reward;
    /** @brief Continuing step: critic value at the next decision, after begin_step.
     * Truncation: value at the final observation before reset. Ignored on termination. */
    double nextValue;
    uint32_t end;
} PpokitTransition;

/** @brief Diagnostics of one update. */
typedef struct PpokitUpdateStatistics
{
    uint64_t updates;
    double policyLoss;
    double valueLoss;
    double entropy;
    double approximateKl;
    double clipFraction;
    double explainedVariance;
    uint64_t actorKlSkips;
    /* A team learner's diagnostics (0 for a single agent). */
    /** @brief Mean |advantage| of the learnable rows, before normalization. */
    double advantageAbsMean;
    /** @brief Mean standard deviation of the actions after the update. */
    double actionStdMean;
    uint64_t learnableRows;
    uint64_t criticRows;
} PpokitUpdateStatistics;

/** @brief PPO settings a team learner trains with (its own settings file). */
typedef struct PpokitTeamSettings
{
    double gamma;
    double lambda;
    double clip;
    double entropyCoefficient;
    uint32_t epochs;
    uint32_t minibatch;
} PpokitTeamSettings;

/** @brief Activity of one block of a network during a probed update (see ppokit/runtime/probe_record.hpp). */
typedef struct PpokitBlockProbe
{
    /** @brief "actor" or "critic". Strings stay valid until the learner's next call. */
    const char* network;
    const char* block;
    const char* type;
    uint64_t samples;
    double mean;
    double deviation;
    double minimum;
    double maximum;
    double deadShare;
    double saturatedShare;
    int hasGradient;
    double gradientNorm;
} PpokitBlockProbe;

typedef struct PpokitPolicy PpokitPolicy;
typedef struct PpokitLearner PpokitLearner;
typedef struct PpokitTeamPolicy PpokitTeamPolicy;
typedef struct PpokitTeamLearner PpokitTeamLearner;

/** @brief Shared graph export ppokit_graph_policy_open; expectedContract is the task semantic hash. */
typedef PpokitPolicy* (*PpokitGraphPolicyOpen)(const char* project, const char* policyFile,
    uint64_t expectedContract, char* error, size_t errorSize);
/** @brief Shared graph export ppokit_graph_learner_create; expectedContract is the task semantic hash. */
typedef PpokitLearner* (*PpokitGraphLearnerCreate)(const char* project, uint64_t seed,
    uint64_t expectedContract, char* error, size_t errorSize);
/** @brief Shared graph export ppokit_graph_team_policy_open, for a team task. */
typedef PpokitTeamPolicy* (*PpokitGraphTeamPolicyOpen)(const char* project, const char* policyFile,
    uint64_t expectedContract, char* error, size_t errorSize);
/** @brief Shared graph export ppokit_graph_team_learner_create, for a team task. */
typedef PpokitTeamLearner* (*PpokitGraphTeamLearnerCreate)(const char* project, uint64_t seed,
    uint64_t expectedContract, char* error, size_t errorSize);

/** @brief Fills `info`; never fails. */
PPOKIT_NETWORK_EXPORT void ppokit_network_info(PpokitNetworkInfo* info);

/**
 * @brief Opens a policy for inference.
 * @param project Folder of task.json and network.json (the networks' structure).
 * @param policyFile policy.ppokit (or a training checkpoint) holding the weights.
 * @return The policy, or NULL with the reason in `error`.
 */
PPOKIT_NETWORK_EXPORT PpokitPolicy* ppokit_policy_open(const char* project, const char* policyFile, char* error,
    size_t errorSize);
/** @brief Writes actionSize values: the mean when `deterministic`, else a sample. */
PPOKIT_NETWORK_EXPORT int ppokit_policy_act(PpokitPolicy* policy, const float* observation, float* action,
    int deterministic);
PPOKIT_NETWORK_EXPORT const char* ppokit_policy_error(const PpokitPolicy* policy);
PPOKIT_NETWORK_EXPORT void ppokit_policy_close(PpokitPolicy* policy);

/**
 * @brief Creates a learner with new networks, PPO settings from training.json.
 * @param project Folder of task.json, network.json and training.json.
 * @param seed 0 for the seed of training.json, else this one.
 */
PPOKIT_NETWORK_EXPORT PpokitLearner* ppokit_learner_create(const char* project, uint64_t seed, char* error,
    size_t errorSize);
/** @brief Samples an action; writes its log-probability and the critic value. */
PPOKIT_NETWORK_EXPORT int ppokit_learner_decide(PpokitLearner* learner, const float* observation, float* action,
    double* logProbability, double* value);
PPOKIT_NETWORK_EXPORT int ppokit_learner_value(PpokitLearner* learner, const float* observation, double* value);
/** @brief Stores one step; fails when the rollout is full (update first). */
PPOKIT_NETWORK_EXPORT int ppokit_learner_append(PpokitLearner* learner, const PpokitTransition* transition);
/** @brief 1 when the rollout is full: time to update. */
PPOKIT_NETWORK_EXPORT int ppokit_learner_full(const PpokitLearner* learner);
/** @brief One PPO update on the rollout, which is then emptied. */
PPOKIT_NETWORK_EXPORT int ppokit_learner_update(PpokitLearner* learner, PpokitUpdateStatistics* statistics);
/** @brief Saves training state and/or policy; NULL skips a file. Training requires an empty rollout. */
PPOKIT_NETWORK_EXPORT int ppokit_learner_save(PpokitLearner* learner, const char* trainingFile, const char* policyFile);
/** @brief Resumes from a training state saved by the same networks. */
PPOKIT_NETWORK_EXPORT int ppokit_learner_load(PpokitLearner* learner, const char* trainingFile);
/** @brief Updates made, including those of a resumed state. */
PPOKIT_NETWORK_EXPORT uint64_t ppokit_learner_updates(const PpokitLearner* learner);
PPOKIT_NETWORK_EXPORT const char* ppokit_learner_error(const PpokitLearner* learner);
PPOKIT_NETWORK_EXPORT void ppokit_learner_destroy(PpokitLearner* learner);

/* ---- Team (agentCapacity above 1) ---------------------------------------------------- */

/** @brief Opens a team policy for inference (deterministic actions). */
PPOKIT_NETWORK_EXPORT PpokitTeamPolicy* ppokit_team_policy_open(const char* project, const char* policyFile,
    char* error, size_t errorSize);
/** @brief Writes the actions of the present agents (agentCapacity x actionSize; absent rows 0). */
PPOKIT_NETWORK_EXPORT int ppokit_team_policy_act(PpokitTeamPolicy* policy, const float* observations,
    const uint8_t* available, float* actions);
PPOKIT_NETWORK_EXPORT const char* ppokit_team_policy_error(const PpokitTeamPolicy* policy);
PPOKIT_NETWORK_EXPORT void ppokit_team_policy_close(PpokitTeamPolicy* policy);

/**
 * @brief Creates a team learner with new networks, settings from the project.
 * @param seed 0 for the seed of the project, else this one.
 */
PPOKIT_NETWORK_EXPORT PpokitTeamLearner* ppokit_team_learner_create(const char* project, uint64_t seed,
    char* error, size_t errorSize);
/**
 * @brief Actions of the present agents. `deterministic` 0: sampled, and the step awaits its
 *        observation; 1: the means, nothing recorded (only between two updates' rollouts).
 */
PPOKIT_NETWORK_EXPORT int ppokit_team_learner_act(PpokitTeamLearner* learner, const float* observations,
    const uint8_t* available, float* actions, int deterministic);
/**
 * @brief Records the step after the sampled actions: next decision observations and availability (after
 *        begin_step for a continuing episode, before reset at an episode boundary),
 *        each agent's reward, the team's end. Updates when the rollout is full.
 * @param updated Receives 1 when an update was made.
 */
PPOKIT_NETWORK_EXPORT int ppokit_team_learner_observe(PpokitTeamLearner* learner, const float* observations,
    const uint8_t* available, const double* rewards, uint32_t end, int* updated);
/** @brief Diagnostics of the last update. */
PPOKIT_NETWORK_EXPORT int ppokit_team_learner_statistics(const PpokitTeamLearner* learner,
    PpokitUpdateStatistics* statistics);
/** @brief The PPO settings the learner trains with, for the run journal. */
PPOKIT_NETWORK_EXPORT int ppokit_team_learner_settings(const PpokitTeamLearner* learner, PpokitTeamSettings* settings);
/** @brief Blocks probed during the last update: 0 when the learner did not probe it (its own rhythm). */
PPOKIT_NETWORK_EXPORT uint32_t ppokit_team_learner_probe_count(const PpokitTeamLearner* learner);
/** @brief Probe `index` (below the count) of the last update. */
PPOKIT_NETWORK_EXPORT int ppokit_team_learner_probe(const PpokitTeamLearner* learner, uint32_t index,
    PpokitBlockProbe* probe);
/** @brief Saves the training state (resumable) and/or the policy; NULL skips a file. Between rollouts only. */
PPOKIT_NETWORK_EXPORT int ppokit_team_learner_save(PpokitTeamLearner* learner, const char* trainingFile,
    const char* policyFile);
/** @brief Resumes from a training state saved by the same networks. */
PPOKIT_NETWORK_EXPORT int ppokit_team_learner_load(PpokitTeamLearner* learner, const char* trainingFile);
PPOKIT_NETWORK_EXPORT uint64_t ppokit_team_learner_updates(const PpokitTeamLearner* learner);
PPOKIT_NETWORK_EXPORT const char* ppokit_team_learner_error(const PpokitTeamLearner* learner);
PPOKIT_NETWORK_EXPORT void ppokit_team_learner_destroy(PpokitTeamLearner* learner);

#ifdef __cplusplus
}
#endif

#endif /* PPOKIT_NETWORK_ABI_H */
