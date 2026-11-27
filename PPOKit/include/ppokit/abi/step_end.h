/**
 * @file PPOKit/include/ppokit/abi/step_end.h
 * @brief How a step ended, shared by the C APIs of network and environment DLLs.
 */
#ifndef PPOKIT_ABI_STEP_END_H
#define PPOKIT_ABI_STEP_END_H

#ifdef __cplusplus
extern "C" {
#endif

enum PpokitStepEnd
{
    PPOKIT_STEP_RUNNING = 0,
    /** @brief The episode reached a terminal state: no future value. */
    PPOKIT_STEP_TERMINATED = 1,
    /** @brief The episode was cut (time limit): its future is the critic's estimate. */
    PPOKIT_STEP_TRUNCATED = 2
};

#ifdef __cplusplus
}
#endif

#endif /* PPOKIT_ABI_STEP_END_H */
