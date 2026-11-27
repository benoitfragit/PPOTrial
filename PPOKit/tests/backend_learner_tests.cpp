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
#include "ppokit/eval/evaluator.hpp"
#include "support/corridor_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ppokit::backend::rltools::GaussianMlpLearner;
using ppokit::backend::rltools::PpoSettings;
using ppokit::runtime::RolloutCollector;
using ppokit::tests::CorridorTask;
using ppokit::tests::DeterministicStepsToGoal;
using ppokit::tests::MakeCorridor;
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

    // Independent-seed evaluation of the frozen policy, gated by the suite file.
    auto const suite = ppokit::eval::LoadEvaluationSuite(PPOKIT_CORRIDOR_SMOKE_SUITE);
    ppokit::eval::RequireDisjointSeeds(suite, {CorridorSettings().seed});
    auto const report = ppokit::eval::Evaluate(suite, MakeCorridor, learner, CorridorTask::StepLimit);
    Check(report.Passed(), "the trained policy passes the corridor_smoke gates");
    if (!report.Passed())
    {
        std::cerr << report.ToJson() << '\n';
    }
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

/**
 * @brief One update on the corridor, probed or not.
 * @return Policy loss, value loss, approximate KL and log-std after the update.
 */
std::array<double, 4> CorridorUpdate(bool probed, std::vector<ppokit::runtime::BlockProbe>* actorSummary)
{
    CorridorTask task;
    Learner learner(CorridorSettings());
    Learner::ActorProbes actorProbes;
    Learner::CriticProbes criticProbes;
    if (probed)
    {
        learner.SetProbes(&actorProbes, &criticProbes);
    }
    Learner::Buffer buffer(128);
    RolloutCollector<CorridorTask, Learner, Learner::Buffer> collector(task, learner);
    collector.Collect(buffer);
    ppokit::core::UpdateStatistics const statistics = learner.Update(buffer);
    if (actorSummary != nullptr)
    {
        *actorSummary = actorProbes.Summary();
    }
    return {statistics.policyLoss, statistics.valueLoss, statistics.approximateKl, learner.LogStd(0)};
}

/** @brief True when a probe saw outputs and a gradient. */
bool Observed(ppokit::runtime::BlockProbe const& probe)
{
    return probe.samples > 0 && std::isfinite(probe.deviation) && probe.hasGradient && probe.gradientNorm > 0.0;
}

/**
 * @brief Probes read the rl_tools layers without changing anything: a probed update gives
 *        bit-identical statistics and log-std, and the probes see every layer.
 */
void ProbesAreReadOnly()
{
    std::vector<ppokit::runtime::BlockProbe> summary;
    Check(CorridorUpdate(true, &summary) == CorridorUpdate(false, nullptr),
        "a probed update is bit-identical to an unprobed one");
    Check(summary.size() == Learner::ActorProbes::Layers, "one probe per actor layer");
    Check(!summary.empty() && summary.front().block == "layer 1" && summary.back().block == "output",
        "layers are named in order, the last one 'output'");
    Check(std::all_of(summary.begin(), summary.end(), Observed), "every actor layer has outputs and a gradient size");
    Check(!summary.empty() && summary.front().type.rfind("dense", 0) == 0, "layers report their activation");
}

} // namespace

/** @brief The gradient norm limit is a graph-learner option: this learner refuses it rather than ignore it. */
void RefusesGradientNormLimit()
{
    PpoSettings settings;
    settings.maxGradientNorm = 0.5;
    bool refused = false;
    try
    {
        Learner const learner(settings);
    }
    catch (std::invalid_argument const& error)
    {
        refused = std::string(error.what()).find("max_gradient_norm") != std::string::npos;
    }
    Check(refused, "rl_tools learner: max_gradient_norm is refused, not ignored");
}

int main()
{
    try
    {
        LearnsTheCorridor();
        SameSeedSameTraining();
        ProbesAreReadOnly();
        RefusesGradientNormLimit();
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
