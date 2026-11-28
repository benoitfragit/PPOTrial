/**
 * @file ppokit/environment/run_profile.h
 * @brief Optional environment/1 extension of the environment C ABI.
 * An opting-in project requires both versioned symbols. Buffers belong to the caller.
 */
#ifndef PPOKIT_ENVIRONMENT_RUN_PROFILE_H
#define PPOKIT_ENVIRONMENT_RUN_PROFILE_H

#include "ppokit/environment/environment_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PpokitRunProfileV1
{
    uint64_t maxUpdates;       /**< Absolute training target, including completed updates. */
    uint64_t checkpointEvery;  /**< Publish every N complete updates. */
    uint32_t trainingSeed;     /**< Deterministic environment seed for the supplied learner update. */
    uint32_t randomizeTrainingSeed;
    uint32_t resumeIfAvailable;
    uint32_t inferenceSeed;    /**< First episode of the project's inference campaign. */
    uint32_t inferenceEpisodes;
} PpokitRunProfileV1;

/** @brief ppokit_environment_run_profile_v1 reads the native scenario without starting a simulation. */
typedef int PpokitReadRunProfileV1(const char* project, uint64_t update, PpokitRunProfileV1* profile,
    char* error, size_t errorSize);

/** @brief ppokit_environment_reset_inference_v1 starts a played episode without advancing training. */
typedef int PpokitResetInferenceV1(PpokitEnvironment* environment, uint32_t seed, uint32_t episode,
    uint32_t episodes, float* observations, uint8_t* available, float* world);

/** @brief Optional scenario editor/1: reports whether an existing scalar at a JSON pointer is editable. */
typedef int PpokitScenarioEditableV1(const char* pointer);

/** @brief Optional scenario editor/1: validate a candidate with the provider's native parser.
 * The project is the original project; candidate is a temporary JSON file. Do not train,
 * step the environment, or modify project files. Zero means accepted, nonzero is an error.
 */
typedef int PpokitValidateScenarioV1(const char* project, const char* candidate, char* error, size_t errorSize);

#ifdef __cplusplus
}
#endif
#endif
