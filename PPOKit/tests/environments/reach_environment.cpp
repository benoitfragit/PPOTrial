/**
 * @file PPOKit/tests/environments/reach_environment.cpp
 * @brief The reach task (examples/reach/task.hpp) as an environment DLL for ppokit-run,
 *        written with the environment SDK, plus a world state for the observers (the
 *        position of the agent and of its target) and two episode metrics for the
 *        evaluation suites: "reached" (flag) and "distance" (to the target at the end).
 *        Its training state counts the episodes trained (learning on) since the run started,
 *        so ppokit-run's tests see whether a run resumed it, started it afresh or archived it.
 */
#include "task.hpp"

#include "ppokit/environment/sdk.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using reach_example::ReachTask;
using Layout = reach_example::Layout;

/** @brief World of task.json: agent x, y and target x, y, recovered from the observation. */
constexpr std::size_t WorldSize = 4;
constexpr std::size_t Channels = Layout::Reward::Count;

class ReachEnvironment final
{
public:
    /** @brief "reached" and "distance", in the order of the test's task.json "metrics". */
    static constexpr std::size_t MetricCount = 2;

    ReachEnvironment(std::filesystem::path const& project, std::uint64_t seed)
        : task_(static_cast<std::uint32_t>(seed)), deferClock_(std::filesystem::exists(project / "defer-clock.fixture"))
    {
    }

    void Reset(float* observation, float* world)
    {
        auto const first = task_.Reset();
        std::copy(first.begin(), first.end(), observation);
        WriteWorld(first, world);
    }

    ppokit::runtime::StepEnd Step(float const* action, float* observation, float* world, double* rewardChannels)
    {
        reach_example::Action applied{};
        std::copy(action, action + applied.size(), applied.begin());
        reach_example::Observation next{};
        auto const outcome = task_.Step(applied, next);
        std::copy(next.begin(), next.end(), observation);
        std::copy(task_.LastRewardChannels(), task_.LastRewardChannels() + Channels, rewardChannels);
        WriteWorld(next, world);
        reached_ = outcome.end == ppokit::runtime::StepEnd::Terminated;
        distance_ = std::hypot(static_cast<double>(next[0]), static_cast<double>(next[1]));
        if (learning_ && outcome.end != ppokit::runtime::StepEnd::Running)
        {
            ++trainedEpisodes_;
        }
        if (deferClock_ && outcome.end == ppokit::runtime::StepEnd::Running)
        {
            auto& clock = observation[Layout::Observation::Clock::Offset];
            pendingClock_ = clock;
            clock = 0.F;
        }
        return outcome.end;
    }

    /** @brief In the deferred fixture the clock becomes visible at the next decision boundary. */
    void BeginStep(float* observation, std::uint8_t* /*available*/, float* /*world*/)
    {
        if (!deferClock_) return;
        if (!pendingClock_) throw std::logic_error("BeginStep requires a continuing transition");
        observation[Layout::Observation::Clock::Offset] = *pendingClock_;
        pendingClock_.reset();
    }

    void SetLearning(bool learning) noexcept { learning_ = learning; }

    /** @brief The episodes trained since the run started. */
    std::string TrainingState() const
    {
        return std::to_string(trainedEpisodes_);
    }

    void RestoreTrainingState(std::string const& state)
    {
        trainedEpisodes_ = state.empty() ? 0 : std::stoull(state);
    }

    void EpisodeMetrics(double* metrics) const
    {
        metrics[0] = reached_ ? 1.0 : 0.0;
        metrics[1] = distance_;
    }

    void SetWeights(double const* weights) { task_.SetRewardWeights({weights, weights + Channels}); }

    void Weights(double* weights) const
    {
        std::vector<double> const current = task_.RewardWeights();
        std::copy(current.begin(), current.end(), weights);
    }

private:
    bool learning_ = true;
    unsigned long long trainedEpisodes_ = 0;
    /** @brief The target's offset is the observation's first two values; the agent is at the origin of it. */
    static void WriteWorld(reach_example::Observation const& observation, float* world) noexcept
    {
        if (world == nullptr)
        {
            return;
        }
        world[0] = 0.F;
        world[1] = 0.F;
        world[2] = observation[0];
        world[3] = observation[1];
    }

    ReachTask task_;
    bool const deferClock_;
    std::optional<float> pendingClock_;
    bool reached_ = false;
    double distance_ = 0.0;
};

} // namespace

PPOKIT_ENVIRONMENT(ReachEnvironment, "reach", Layout::Observation::Size, Layout::Action::Size, Channels, WorldSize)
