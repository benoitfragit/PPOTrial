/**
 * @file PPOKit/include/ppokit/runtime/collector.hpp
 * @brief Owned-mode collection: PPOKit drives the task step by step and fills a rollout.
 * @details In Owned mode the example only writes a task, the physics and rewards of
 * one episode; PPOKit owns the loop. The task and the policy are plain classes with
 * the members below (no base class, no virtual call in the hot loop):
 *
 * @code
 * struct MyTask
 * {
 *     // Starts an episode and returns its first observation.
 *     std::array<float, ObservationSize> Reset();
 *     // Applies a raw action, writes the next observation, returns reward and episode end.
 *     ppokit::runtime::StepOutcome<double> Step(std::array<float, ActionSize> const& action,
 *         std::array<float, ObservationSize>& nextObservation);
 * };
 * struct MyPolicy
 * {
 *     // Samples an action and returns it with its log-probability and the critic value.
 *     ppokit::runtime::PolicyDecision<double, ActionSize> Decide(std::array<float, ObservationSize> const& observation);
 *     // Critic value of an observation, used to bootstrap truncated episodes.
 *     double Value(std::array<float, ObservationSize> const& observation);
 * };
 * @endcode
 *
 * A task may also expose its weighted reward channels of the last step, for the run
 * journal (see Track):
 *
 * @code
 *     double const* LastRewardChannels() const; // RewardLedger::Channels() values
 * @endcode
 */
#pragma once

#include "ppokit/runtime/reward_ledger.hpp"
#include "ppokit/runtime/rollout.hpp"
#include "ppokit/runtime/step_watcher.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ppokit::runtime {

/** @brief What the policy decided for one observation. */
template<typename Scalar, std::size_t ActionSize>
struct PolicyDecision final
{
    /** @brief Raw sampled action, before any clipping. */
    std::array<float, ActionSize> action{};
    Scalar logProbability{};
    /** @brief Critic value of the observation the decision was made on. */
    Scalar value{};
};

/** @brief Result of one task step. */
template<typename Scalar>
struct StepOutcome final
{
    /** @brief Total (weighted) reward of the step. */
    Scalar reward{};
    StepEnd end = StepEnd::Running;
};

/** @brief Counters of the episodes finished during collection, for logs and curriculum gates. */
struct EpisodeStatistics final
{
    std::size_t episodes = 0;
    std::size_t terminated = 0;
    std::size_t truncated = 0;
    double returnSum = 0;
    double lengthSum = 0;

    double MeanReturn() const noexcept { return episodes == 0 ? 0.0 : returnSum / static_cast<double>(episodes); }
    double MeanLength() const noexcept { return episodes == 0 ? 0.0 : lengthSum / static_cast<double>(episodes); }
};

/** @brief Whether a task exposes `double const* LastRewardChannels() const`. */
template<typename Task, typename = void>
struct HasRewardChannels : std::false_type
{
};
template<typename Task>
struct HasRewardChannels<Task, std::enable_if_t<std::is_same_v<
    decltype(std::declval<Task const&>().LastRewardChannels()), double const*>>> : std::true_type
{
};

/**
 * @brief Fills rollouts by running a task with a policy, carrying episodes across rollouts.
 * @details An episode interrupted because the buffer is full continues at the next
 * Collect call: rollout boundaries do not end episodes. Its last stored transition is
 * still bootstrapped on the critic value of the next observation.
 */
template<typename Task, typename Policy, typename Buffer>
class RolloutCollector final
{
public:
    using Row = typename Buffer::Row;
    using Scalar = typename Row::ScalarType;
    using Observation = typename Row::Observation;

    RolloutCollector(Task& task, Policy& policy) noexcept : task_(task), policy_(policy) {}

    /**
     * @brief Runs the task until `buffer` is full.
     * @return False when the watcher interrupted the collection (stop): the buffer is partial.
     */
    bool Collect(Buffer& buffer)
    {
        if (!episodeRunning_)
        {
            StartEpisode();
        }
        while (!buffer.Full())
        {
            CollectStep(buffer);
            if (watcher_ != nullptr && !watcher_->AfterStep())
            {
                return false;
            }
        }
        return true;
    }

    /** @brief Calls `watcher` after every step (live control, observers), or stops with null. */
    void Watch(StepWatcher* watcher) noexcept { watcher_ = watcher; }

    EpisodeStatistics const& Statistics() const noexcept { return statistics_; }
    void ResetStatistics() noexcept { statistics_ = {}; }

    /**
     * @brief Records every collected step in `ledger` (stream 0), or stops with null.
     * @details The task's weighted channels when it exposes LastRewardChannels, otherwise
     * the total reward as a single channel. The ledger must outlive the collection.
     */
    void Track(RewardLedger* ledger) noexcept { ledger_ = ledger; }

private:
    void StartEpisode()
    {
        observation_ = task_.Reset();
        episodeReturn_ = 0;
        episodeLength_ = 0;
        episodeRunning_ = true;
    }

    void CollectStep(Buffer& buffer)
    {
        Row row;
        row.observation = observation_;
        auto const decision = policy_.Decide(observation_);
        row.action = decision.action;
        row.logProbability = decision.logProbability;
        row.value = decision.value;

        Observation nextObservation{};
        auto const outcome = task_.Step(decision.action, nextObservation);
        row.reward = outcome.reward;
        row.end = outcome.end;
        // A terminated episode has no future: the critic is not even queried.
        if (outcome.end != StepEnd::Terminated)
        {
            row.nextValue = policy_.Value(nextObservation);
        }
        buffer.Append(row);

        episodeReturn_ += static_cast<double>(outcome.reward);
        ++episodeLength_;
        if (ledger_ != nullptr)
        {
            double const total = static_cast<double>(row.reward);
            if constexpr (HasRewardChannels<Task>::value)
            {
                ledger_->RecordWeighted(0, task_.LastRewardChannels(), total);
            }
            else
            {
                ledger_->RecordWeighted(0, &total, total);
            }
        }
        if (EndsEpisode(outcome.end))
        {
            if (ledger_ != nullptr)
            {
                ledger_->EndEpisode(0);
            }
            RecordEpisode(outcome.end);
            StartEpisode();
        }
        else
        {
            observation_ = nextObservation;
        }
    }

    void RecordEpisode(StepEnd end) noexcept
    {
        ++statistics_.episodes;
        if (end == StepEnd::Terminated)
        {
            ++statistics_.terminated;
        }
        else
        {
            ++statistics_.truncated;
        }
        statistics_.returnSum += episodeReturn_;
        statistics_.lengthSum += static_cast<double>(episodeLength_);
    }

    Task& task_;
    Policy& policy_;
    RewardLedger* ledger_ = nullptr;
    StepWatcher* watcher_ = nullptr;
    Observation observation_{};
    bool episodeRunning_ = false;
    double episodeReturn_ = 0;
    std::size_t episodeLength_ = 0;
    EpisodeStatistics statistics_;
};

} // namespace ppokit::runtime
