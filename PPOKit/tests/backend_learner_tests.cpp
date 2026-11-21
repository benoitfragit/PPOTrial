/**
 * @file PPOKit/tests/backend_learner_tests.cpp
 * @brief End-to-end PPOKit training on a toy task: task.json + graph.json -> generated
 *        networks -> Owned collection -> GaussianMlpLearner updates.
 *
 * This is a smoke test of the training engine, not a convergence claim: the corridor is
 * solved by always moving forward, which any working PPO finds within a few updates.
 */
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "corridor_networks.hpp"
#include "ppokit/backend/rltools/gaussian_mlp_learner.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace {

using ppokit::backend::rltools::GaussianMlpLearner;
using ppokit::backend::rltools::PpoSettings;
using ppokit::runtime::RolloutCollector;
using ppokit::runtime::StepEnd;
using ppokit::runtime::StepOutcome;
using Learner = GaussianMlpLearner<ppokit::tests::corridor::networks::Actor, ppokit::tests::corridor::networks::Critic>;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/**
 * @brief Corridor from 0 to 3: each step moves by half the clipped action and is
 *        rewarded by its progress minus 0.05; reaching 3 gives +1 and terminates; the
 *        episode is truncated after 20 steps.
 */
class CorridorTask final
{
public:
    static constexpr float Goal = 3.F;
    static constexpr float StepScale = 0.5F;
    static constexpr int StepLimit = 20;

    std::array<float, 2> Reset()
    {
        position_ = 0.F;
        steps_ = 0;
        return Observe();
    }

    StepOutcome<double> Step(std::array<float, 1> const& action, std::array<float, 2>& nextObservation)
    {
        // "map": "clip" in task.json: the sample is clipped before it moves the agent.
        float const progress = StepScale * std::clamp(action[0], -1.F, 1.F);
        position_ += progress;
        ++steps_;
        nextObservation = Observe();
        double const reward = static_cast<double>(progress) - 0.05;
        if (position_ >= Goal)
        {
            return {reward + 1.0, StepEnd::Terminated};
        }
        if (steps_ == StepLimit)
        {
            return {reward, StepEnd::Truncated};
        }
        return {reward, StepEnd::Running};
    }

private:
    std::array<float, 2> Observe() const noexcept
    {
        return {position_ / Goal, static_cast<float>(steps_) / static_cast<float>(StepLimit)};
    }

    float position_ = 0.F;
    int steps_ = 0;
};

/** @brief Steps needed by the deterministic policy to reach the goal, or -1 if it never does. */
int DeterministicStepsToGoal(Learner& learner)
{
    CorridorTask task;
    auto observation = task.Reset();
    for (int step = 1; step <= CorridorTask::StepLimit; ++step)
    {
        std::array<float, 2> next{};
        auto const outcome = task.Step(learner.Mean(observation), next);
        if (outcome.end == StepEnd::Terminated)
        {
            return step;
        }
        observation = next;
    }
    return -1;
}

PpoSettings CorridorSettings()
{
    PpoSettings settings;
    settings.actorLearningRate = 3e-3;
    settings.criticLearningRate = 3e-3;
    settings.minibatch = 64;
    settings.epochs = 4;
    settings.seed = 7;
    return settings;
}

void LearnsTheCorridor()
{
    CorridorTask task;
    Learner learner(CorridorSettings());
    Learner::Buffer buffer(512);
    RolloutCollector<CorridorTask, Learner, Learner::Buffer> collector(task, learner);
    collector.Collect(buffer);
    double const firstReturn = collector.Statistics().MeanReturn();
    double lastReturn = firstReturn;
    for (int update = 0; update < 30; ++update)
    {
        auto const statistics = learner.Update(buffer);
        Check(std::isfinite(statistics.policyLoss) && std::isfinite(statistics.valueLoss), "finite losses");
        if (update % 10 == 0)
        {
            std::cout << "update " << statistics.updates << ": return " << lastReturn << ", value loss "
                      << statistics.valueLoss << ", kl " << statistics.approximateKl << ", clip "
                      << statistics.clipFraction << ", explained variance " << statistics.explainedVariance
                      << ", log-std " << learner.LogStd(0) << '\n';
        }
        collector.ResetStatistics();
        collector.Collect(buffer);
        lastReturn = collector.Statistics().MeanReturn();
    }
    int const steps = DeterministicStepsToGoal(learner);
    std::cout << "corridor: mean return " << firstReturn << " -> " << lastReturn << ", deterministic policy reaches "
              << "the goal in " << steps << " steps (optimum 6)\n";
    Check(lastReturn > firstReturn + 0.3, "training raises the mean episode return");
    Check(steps > 0 && steps <= 8, "the deterministic policy walks to the goal");
}

void SameSeedSameTraining()
{
    auto const run = []() {
        CorridorTask task;
        Learner learner(CorridorSettings());
        Learner::Buffer buffer(128);
        RolloutCollector<CorridorTask, Learner, Learner::Buffer> collector(task, learner);
        collector.Collect(buffer);
        auto const statistics = learner.Update(buffer);
        return std::array<double, 3>{statistics.policyLoss, statistics.valueLoss, learner.LogStd(0)};
    };
    Check(run() == run(), "a seed fully determines weights, sampling and shuffling");
}

} // namespace

int main()
{
    try
    {
        LearnsTheCorridor();
        SameSeedSameTraining();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " learner check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit rl_tools learner tests passed\n";
    return EXIT_SUCCESS;
}
