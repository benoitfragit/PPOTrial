/**
 * @file module_task_contract.cpp
 * @brief Export the build-time task contract of one native module.
 */
#define PPOKIT_DEFINE_TASK_CONTRACT
#include "ppokit/abi/task_contract.h"
#include PPOKIT_TASK_LAYOUT

extern "C" PPOKIT_TASK_EXPORT uint64_t ppokit_task_semantic_hash(void)
{
    return PPOKIT_TASK_NAMESPACE::TaskLayout::SemanticHash;
}
