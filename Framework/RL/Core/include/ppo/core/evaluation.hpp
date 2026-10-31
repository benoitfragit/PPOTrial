/**
 * @file Framework/RL/Core/include/ppo/core/evaluation.hpp
 * @brief Interface ou implémentation du composant evaluation.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include <algorithm>
#include <utility>

namespace ppo::core {

// Domain-independent evaluation campaign status. A partial campaign must
// never be mistaken for a valid comparison or model-promotion decision.
/** @brief Déclare le type EvaluationResult, utilisé par Interface ou implémentation du composant evaluation. */
struct EvaluationResult
{
    int finishedTests = 0;
    bool completed = false;
};

// Progress within one test is an upper-bound work fraction, not the
// score nor a completion criterion. Short-circuited tests count as complete
// only after finishTest has run.
/**
 * @brief Réalise l'opération EvaluationProgress; voir les préconditions et effets dans son corps C++.
 * @details Applique des bornes numériques.
 */
inline float EvaluationProgress(int completedTests,
                                int stepsInCurrentTest,
                                int totalTests,
                                int maxSteps) noexcept
{
    if (totalTests <= 0 || maxSteps <= 0)
        return 0.F;

    double const finished = static_cast<double>(std::clamp(completedTests, 0, totalTests));
    if (finished >= totalTests)
        return 1.F;

    double const partial = static_cast<double>(std::clamp(stepsInCurrentTest, 0, maxSteps))
        / static_cast<double>(maxSteps);
    return static_cast<float>((finished + partial) / static_cast<double>(totalTests));
}

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
