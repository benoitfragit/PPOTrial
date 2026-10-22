#pragma once

#include <utility>

namespace ppo::core {

// Reusable evaluation campaign scheduler; no dependency on policy, simulator,
// scoring formula, checkpoint format or thread management.
//
// continueRunning(): cancellation checked before each test and step.
// resetTest(index): constructs a fresh test state; false aborts the campaign.
// stepTest(index, step): returns true when this test terminates early.
// finishTest(index, step): called once for each successfully reset test.
// step is deliberately the original zero-based loop index, even on early exit.
// Callbacks are synchronous and execute on the caller's thread.
template<typename Continue, typename Reset, typename Step, typename Complete>
void RunEvaluation(int testCount,
                   int maxSteps,
                   Continue&& continueRunning,
                   Reset&& resetTest,
                   Step&& stepTest,
                   Complete&& finishTest)
{
    for (int test = 0; test < testCount; ++test)
    {
        if (!continueRunning())
            break;
        if (!resetTest(test))
            break;

        int step = 0;
        for (; step < maxSteps; ++step)
        {
            if (!continueRunning())
                break;
            if (stepTest(test, step))
                break;
        }
        finishTest(test, step);
    }
}

} // namespace ppo::core
