/**
 * @file PPOKit/tests/support/corridor_task.hpp
 * @brief Toy corridor task shared by the learner tests (rl_tools and graph learners).
 */
#pragma once

#include "ppokit/runtime/collector.hpp"
#include "ppokit/runtime/rollout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace ppokit::tests {

using runtime::StepEnd;
using runtime::StepOutcome;

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

/** @brief The corridor has no randomness: every seed gives the same task. */
inline CorridorTask MakeCorridor(std::uint32_t /*seed*/)
{
    return {};
}

/** @brief Steps needed by the deterministic policy to reach the goal, or -1 if it never does. */
template<typename Learner>
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

} // namespace ppokit::tests
