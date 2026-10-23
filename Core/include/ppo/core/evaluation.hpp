#pragma once

#include <utility>

namespace ppo::core {

// Domain-independent evaluation campaign status. A partial campaign must
// never be mistaken for a valid comparison or model-promotion decision.
struct EvaluationResult
{
    int finishedTests = 0;
    bool completed = false;
};

// Execute tests in caller order. Synchronous, with no thread, model or scoring
// dependencies. On mid-test cancellation, finishTest still receives the
// original zero-based step index (preserving evaluator accounting).
template<typename Continue, typename Reset, typename Step, typename Complete>
EvaluationResult RunEvaluation(int testCount,
                               int maxSteps,
                               Continue&& continueRunning,
                               Reset&& resetTest,
                               Step&& stepTest,
                               Complete&& finishTest)
{
    EvaluationResult result{};
    if (testCount <= 0 || maxSteps <= 0)
        return result;

    bool aborted = false;
    for (int test = 0; test < testCount; ++test)
    {
        if (!continueRunning() || !resetTest(test))
        {
            aborted = true;
            break;
        }

        int step = 0;
        for (; step < maxSteps; ++step)
        {
            if (!continueRunning())
            {
                aborted = true;
                break;
            }
            if (stepTest(test, step))
                break;
        }
        finishTest(test, step);
        ++result.finishedTests;

        if (aborted)
            break;
    }

    result.completed = !aborted && result.finishedTests == testCount;
    return result;
}

} // namespace ppo::core
