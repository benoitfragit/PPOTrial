/**
 * @file PPOKit/include/ppokit/component/abi.h
 * @brief C ABI of the black-box components a DLL provides to PPOKit graphs.
 * @details Plain C so that any compiler can build a component: no STL, no exception and
 * no shared allocation crosses this boundary. PPOKit owns every buffer (inputs, outputs,
 * cache, gradients), so a call never allocates on PPOKit's behalf. Functions return 0 on
 * success; on failure `lastError` explains. Write components with the C++ SDK
 * (ppokit/component/sdk.hpp), which fills this table from a class.
 *
 * A DLL exports one function, `ppokit_components`, listing its components. PPOKit asks
 * for its ABI version; a DLL built for another major version answers with an error.
 */
#ifndef PPOKIT_COMPONENT_ABI_H
#define PPOKIT_COMPONENT_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief ABI version; a DLL refuses another one. */
#define PPOKIT_COMPONENT_ABI 1u

#if defined(_WIN32)
#define PPOKIT_EXPORT __declspec(dllexport)
#else
#define PPOKIT_EXPORT __attribute__((visibility("default")))
#endif

/** @brief Element types of a buffer. */
enum PpokitDtype
{
    PPOKIT_DTYPE_F32 = 0,
    PPOKIT_DTYPE_F64 = 1,
    PPOKIT_DTYPE_I32 = 2,
    /** @brief One byte per element, 0 or 1 (presence masks). */
    PPOKIT_DTYPE_BOOL = 3
};

/** @brief A tensor owned by PPOKit: `batch` samples of `shape`, contiguous, row-major. */
typedef struct PpokitBuffer
{
    void* data;
    uint32_t dtype;
    uint32_t rank;
    const uint32_t* shape;
} PpokitBuffer;

/** @brief One component: its manifest and its functions. Optional functions are NULL. */
typedef struct PpokitComponentApi
{
    uint32_t abi;
    /** @brief Name and version, as written in graph.json: "component:<name>@<version>". */
    const char* name;
    uint32_t version;
    /** @brief Ports, capabilities and configuration (schema "ppokit.component/1"). */
    const char* manifestJson;

    /**
     * @brief Creates an instance.
     * @param configJson Validated configuration, e.g. {"alpha":0.2}.
     * @param resolvedShapesJson Symbolic dimensions resolved by the wiring, e.g. {"N":10}.
     * @param seed Seed of the instance (initial parameters, random draws).
     * @return The instance, or NULL with the reason in `error`.
     */
    void* (*create)(const char* configJson, const char* resolvedShapesJson, uint64_t seed, char* error,
        size_t errorSize);
    void (*destroy)(void* instance);

    /** @brief Inputs in manifest order of the "in" ports, outputs in order of the "out" ports. */
    int (*forward)(void* instance, uint32_t batch, const PpokitBuffer* inputs, PpokitBuffer* outputs, void* cache);
    /** @brief Bytes of cache forward may fill for backward (0: none). */
    size_t (*cacheSize)(void* instance, uint32_t batch);

    /**
     * @brief Differentiable components: ADDS d loss / d input to `inputGradients` (one per
     *        "in" port, masks excluded: their pointer is NULL) and d loss / d parameter to
     *        `parameterGradients` (trainable components). NULL otherwise.
     */
    int (*backward)(void* instance, uint32_t batch, const PpokitBuffer* outputGradients, const void* cache,
        PpokitBuffer* inputGradients, float* parameterGradients);

    /** @brief Trainable components: their parameters, which PPOKit's optimizer updates. NULL otherwise. */
    int (*parameters)(void* instance, float** data, size_t* count);

    /** @brief Stateful components: episode start of an agent, and the state for checkpoints. NULL otherwise. */
    int (*reset)(void* instance, uint32_t agentSlot, uint64_t seed);
    size_t (*stateSize)(void* instance);
    int (*saveState)(void* instance, void* buffer, size_t size);
    int (*loadState)(void* instance, const void* buffer, size_t size);

    /** @brief The reason of the last failure of this instance. */
    int (*lastError)(void* instance, char* error, size_t errorSize);
} PpokitComponentApi;

/** @brief Signature of the one function a component DLL exports. */
typedef int (*PpokitComponentsFunction)(uint32_t requestedAbi, const PpokitComponentApi* const** components,
    size_t* count);

/** @brief Name of that function. */
#define PPOKIT_COMPONENTS_ENTRY "ppokit_components"

#ifdef __cplusplus
}
#endif

#endif /* PPOKIT_COMPONENT_ABI_H */
