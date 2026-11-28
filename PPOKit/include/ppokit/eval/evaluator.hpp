/**
 * @file PPOKit/include/ppokit/eval/evaluator.hpp
 * @brief Runs an evaluation suite with a deterministic policy.
 * @details The task is created per seed by a factory, so that every evaluation episode
 * is reproducible from its seed alone. The policy only needs `Mean(observation)`: it is
 * never sampled, and evaluating does not touch the training random stream.
 *
 * @code
 * MyTask MakeTask(std::uint32_t seed);   // any callable: seed -> task
 * auto report = ppokit::eval::Evaluate(suite, MakeTask, learner, maximumSteps);
 * if (!report.Passed()) { std::cerr << report.ToJson(); }
 * @endcode
 */
#pragma once

#include "ppokit/eval/evaluation.hpp"
#include "ppokit/runtime/collector.hpp"

#include <cstddef>
#include <stdexcept>

namespace ppokit::eval {

/** @brief The caller requested an interruption before a complete report existed. */
struct EvaluationStopped final {};

/**
 * @brief Plays `suite.episodesPerSeed` episodes on a fresh task for every seed.
 * @param makeTask Callable returning a task (Reset/Step, see runtime/collector.hpp) for a seed.
 * @param policy Object with `Mean(observation)` returning the deterministic action.
 * @param maximumSteps Safety bound on an episode; reaching it is an error, because a task
 *        must end its episodes itself (terminate or truncate).
 * @throws std::runtime_error when an episode exceeds `maximumSteps`.
 */
template<typename TaskFactory, typename Policy>
EvaluationReport Evaluate(EvaluationSuite const& suite, TaskFactory&& makeTask, Policy& policy,
    std::size_t maximumSteps, runtime::StepWatcher* watcher = nullptr)
{
    if (suite.campaignEnvironment)
        throw std::invalid_argument("campaign scope requires an environment with explicit per-episode reseeding");
    EvaluationReport report;
    report.suite = suite.name;
    for (std::uint32_t const seed : suite.seeds)
    {
        auto task = makeTask(seed);
        for (std::size_t episode = 0; episode < suite.episodesPerSeed; ++episode)
        {
            EpisodeResult result;
            result.seed = seed;
            auto observation = task.Reset();
            bool ended = false;
            while (!ended)
            {
                if (watcher && !watcher->AfterStep()) throw EvaluationStopped{};
                if (result.length == maximumSteps)
                {
                    throw std::runtime_error("PPOKit evaluation: an episode exceeded the step bound; "
                                             "the task must terminate or truncate its episodes");
                }
                decltype(observation) next{};
                auto const outcome = task.Step(policy.Mean(observation), next);
                result.episodeReturn += static_cast<double>(outcome.reward);
                ++result.length;
                ended = runtime::EndsEpisode(outcome.end);
                result.terminated = outcome.end == runtime::StepEnd::Terminated;
                observation = next;
            }
            report.episodes.push_back(result);
        }
    }
    Summarize(suite, report);
    return report;
}

} // namespace ppokit::eval
