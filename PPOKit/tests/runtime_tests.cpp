/**
 * @file PPOKit/tests/runtime_tests.cpp
 * @brief Rollout buffer, GAE over episodes, advantage normalization, explained variance
 *        and Owned-mode collection on a toy corridor task.
 */
#include "ppokit/runtime/collector.hpp"
#include "ppokit/runtime/rollout.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using ppokit::runtime::PolicyDecision;
using ppokit::runtime::RolloutBuffer;
using ppokit::runtime::RolloutCollector;
using ppokit::runtime::StepEnd;
using ppokit::runtime::StepOutcome;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

bool Near(double left, double right, double tolerance = 1e-12) noexcept
{
    return std::abs(left - right) <= tolerance;
}

using Buffer = RolloutBuffer<double, 1, 1>;
using Row = Buffer::Row;

/**
 * @brief One-dimensional corridor: move from 0 towards the goal at 3.
 * Each step costs 0.1; reaching the goal gives +1 and terminates; the episode is
 * truncated after six steps.
 */
class CorridorTask final
{
public:
    static constexpr float Goal = 3.F;
    static constexpr std::size_t StepLimit = 6;

    std::array<float, 1> Reset()
    {
        position_ = 0.F;
        steps_ = 0;
        ++resets;
        return {position_};
    }

    StepOutcome<double> Step(std::array<float, 1> const& action, std::array<float, 1>& nextObservation)
    {
        position_ += action[0];
        ++steps_;
        nextObservation = {position_};
        if (position_ >= Goal)
        {
            return {1.0, StepEnd::Terminated};
        }
        if (steps_ == StepLimit)
        {
            return {-0.1, StepEnd::Truncated};
        }
        return {-0.1, StepEnd::Running};
    }

    std::size_t resets = 0;

private:
    float position_ = 0.F;
    std::size_t steps_ = 0;
};

/** @brief Deterministic policy with a fixed step and a critic proportional to the position. */
class FixedStepPolicy final
{
public:
    explicit FixedStepPolicy(float step) noexcept : step_(step) {}

    PolicyDecision<double, 1> Decide(std::array<float, 1> const& observation)
    {
        return {{step_}, -0.5, Value(observation)};
    }
    double Value(std::array<float, 1> const& observation) { return static_cast<double>(observation[0]) / 3.0; }

private:
    float step_;
};

void CollectsEpisodesAcrossRollouts()
{
    CorridorTask task;
    FixedStepPolicy policy(1.F);
    RolloutCollector<CorridorTask, FixedStepPolicy, Buffer> collector(task, policy);
    Buffer buffer(7);
    collector.Collect(buffer);
    Check(buffer.Full(), "collection fills the buffer");
    Check(collector.Statistics().episodes == 2 && collector.Statistics().terminated == 2,
        "two complete episodes of three steps");
    Check(Near(collector.Statistics().MeanReturn(), 0.8) && Near(collector.Statistics().MeanLength(), 3.0),
        "episode return -0.1 - 0.1 + 1 and length 3");
    Check(buffer[2].end == StepEnd::Terminated && buffer[2].nextValue == 0.0,
        "a terminated step is never bootstrapped");
    Check(buffer[6].end == StepEnd::Running && Near(buffer[6].nextValue, 1.0 / 3.0),
        "the last stored step bootstraps on the next observation");
    buffer.Clear();
    collector.Collect(buffer);
    Check(buffer[0].observation[0] == 1.F, "the interrupted episode continues in the next rollout");
    Check(task.resets == 5, "one reset per started episode, none at rollout boundaries");
}

void TruncationBootstraps()
{
    CorridorTask task;
    FixedStepPolicy policy(0.25F);
    RolloutCollector<CorridorTask, FixedStepPolicy, Buffer> collector(task, policy);
    Buffer buffer(6);
    collector.Collect(buffer);
    Check(buffer[5].end == StepEnd::Truncated && Near(buffer[5].nextValue, 1.5 / 3.0),
        "a truncated step bootstraps on the critic");
    Check(collector.Statistics().truncated == 1, "truncation counted");
}

void AdvantagesFollowGae()
{
    Buffer buffer(4);
    // Episode A: two steps ending in a time limit; episode B: two steps, still running.
    Row row;
    row.reward = 1.0;
    row.value = 0.5;
    row.nextValue = 0.25;
    buffer.Append(row);
    row.end = StepEnd::Truncated;
    row.nextValue = 2.0;
    buffer.Append(row);
    row.end = StepEnd::Running;
    row.nextValue = 0.75;
    buffer.Append(row);
    row.reward = -1.0;
    buffer.Append(row);
    constexpr double Gamma = 0.9;
    constexpr double Lambda = 0.8;
    buffer.ComputeAdvantages(Gamma, Lambda);
    double const last = -1.0 + Gamma * 0.75 - 0.5;
    double const third = 1.0 + Gamma * 0.75 - 0.5 + Gamma * Lambda * last;
    double const truncated = 1.0 + Gamma * 2.0 - 0.5;
    double const first = 1.0 + Gamma * 0.25 - 0.5 + Gamma * Lambda * truncated;
    Check(Near(buffer[3].advantage, last), "final row starts the trace from zero");
    Check(Near(buffer[2].advantage, third), "trace inside an episode");
    Check(Near(buffer[1].advantage, truncated), "truncation keeps its bootstrap and cuts the trace");
    Check(Near(buffer[0].advantage, first), "trace inside the truncated episode");
    Check(Near(buffer[0].valueTarget, 0.5 + first), "value target = value + advantage");
}

void NormalizesLearnableRowsOnly()
{
    Buffer buffer(4);
    Row row;
    for (double const reward : {1.0, 2.0, 3.0, 100.0})
    {
        row.reward = reward;
        row.end = StepEnd::Terminated;
        row.learnable = reward < 50.0;
        buffer.Append(row);
    }
    buffer.ComputeAdvantages(0.99, 0.95);
    buffer.NormalizeAdvantages();
    double mean = 0;
    double meanSquare = 0;
    for (std::size_t index = 0; index < 3; ++index)
    {
        mean += buffer[index].advantage / 3.0;
        meanSquare += buffer[index].advantage * buffer[index].advantage / 3.0;
    }
    Check(Near(mean, 0.0, 1e-12) && Near(meanSquare, 1.0, 1e-6), "learnable advantages: zero mean, unit variance");
    Check(buffer[3].advantage == 100.0, "a row that is not learnable keeps its advantage");
}

void ExplainedVarianceRange()
{
    Buffer perfect(3);
    Row row;
    for (double const reward : {1.0, 2.0, 4.0})
    {
        row.reward = reward;
        row.value = reward;
        row.end = StepEnd::Terminated;
        perfect.Append(row);
    }
    perfect.ComputeAdvantages(0.99, 0.95);
    Check(Near(ppokit::runtime::ExplainedVariance(perfect), 1.0), "a critic equal to the targets explains everything");
    Buffer constant(3);
    for (double const reward : {1.0, 2.0, 4.0})
    {
        row.reward = reward;
        row.value = 0.0;
        constant.Append(row);
    }
    constant.ComputeAdvantages(0.99, 0.95);
    Check(Near(ppokit::runtime::ExplainedVariance(constant), 0.0), "a constant critic explains nothing");
}

void RejectsInvalidUse()
{
    bool rejected = false;
    try
    {
        Buffer invalid(0);
    }
    catch (std::invalid_argument const&)
    {
        rejected = true;
    }
    Check(rejected, "zero capacity rejected");
    Buffer buffer(1);
    Row row;
    row.reward = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try
    {
        buffer.Append(row);
    }
    catch (std::invalid_argument const&)
    {
        rejected = true;
    }
    Check(rejected && buffer.Size() == 0, "non-finite reward rejected before it corrupts the rollout");
    row.reward = 0.0;
    buffer.Append(row);
    rejected = false;
    try
    {
        buffer.Append(row);
    }
    catch (std::logic_error const&)
    {
        rejected = true;
    }
    Check(rejected, "appending to a full buffer is an error");
}

} // namespace

int main()
{
    CollectsEpisodesAcrossRollouts();
    TruncationBootstraps();
    AdvantagesFollowGae();
    NormalizesLearnableRowsOnly();
    ExplainedVarianceRange();
    RejectsInvalidUse();
    if (failures != 0)
    {
        std::cerr << failures << " runtime check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit runtime tests passed\n";
    return EXIT_SUCCESS;
}
