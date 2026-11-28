/**
 * @file PPOKit/tests/components/not_a_component.cpp
 * @brief A DLL without the PPOKit entry point, to check the loader's message.
 */
#include "ppokit/component/abi.h"

extern "C" PPOKIT_EXPORT int some_other_function()
{
    return 42;
}
