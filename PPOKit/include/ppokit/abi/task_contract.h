/**
 * @file task_contract.h
 * @brief Compiled semantic task identity shared by native environment and network modules.
 * @details A separate entry point keeps the existing C info structures binary compatible.
 */
#pragma once
#include <stdint.h>

#if defined(_WIN32) && defined(PPOKIT_DEFINE_TASK_CONTRACT)
#define PPOKIT_TASK_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
#define PPOKIT_TASK_EXPORT __declspec(dllimport)
#else
#define PPOKIT_TASK_EXPORT __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/** @return The versioned semantic hash of the task used to build this module. */
PPOKIT_TASK_EXPORT uint64_t ppokit_task_semantic_hash(void);
#ifdef __cplusplus
}
#endif
