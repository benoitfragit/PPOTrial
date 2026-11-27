/**
 * @file PPOKit/include/ppokit/runtime/rollout.hpp
 * @brief On-policy rollout storage: transitions, GAE, advantage normalization, explained variance.
 * @details One PPO update consumes a fixed number of transitions collected with the
 * current policy. The buffer is allocated once, at construction, so collecting a
 * transition never allocates (AGENTS.md: no allocation in per-step loops).
 *
 * Typical use, once per update:
 * @code
 * buffer.Append(transition);                 // until buffer.Full()
 * buffer.ComputeAdvantages(gamma, lambda);   // GAE, backwards over the rollout
 * auto const explained = ExplainedVariance(buffer);
 * buffer.NormalizeAdvantages();              // zero mean, unit variance
 * // ... PPO epochs over buffer[i] ...
 * buffer.Clear();
 * @endcode
 */
#pragma once

#include "ppokit/core/gae.hpp"
#include "ppokit/runtime/advantages.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ppokit::runtime {

/** @brief How a transition relates to the end of its episode. */
enum class StepEnd : std::uint8_t
{
    /** @brief The episode continues. */
    Running,
    /** @brief The episode really ended (crash, landing, mission outcome): the future is worth zero. */
    Terminated,
    /** @brief The episode was stopped from outside (time limit): the future still had value. */
    Truncated,
};

/** @brief True when the transition is the last one of its episode. */
constexpr bool EndsEpisode(StepEnd end) noexcept
{
    return end != StepEnd::Running;
}

/**
 * @brief Compute GAE for interleaved, independent trajectory streams.
 * @param rows Step-major contiguous storage, with one row per stream and step.
 * @param steps Number of recorded steps per stream (zero is allowed).
 * @param streams Number of independent streams (positive).
 * @return Number of rows eligible for the actor update.
 * @details The caller owns storage for steps * streams rows. Preserve stream-major
 * traversal and the row scalar type; no allocation, conversion or cross-stream trace.
 */
template<typename Row>
std::size_t ComputeTrajectoryAdvantages(Row* rows, std::size_t steps, std::size_t streams,
    typename Row::ScalarType gamma, typename Row::ScalarType lambda) noexcept
{
    using Scalar = typename Row::ScalarType;
    std::size_t learnable = 0;
    for (std::size_t stream = 0; stream < streams; ++stream)
    {
        Scalar followingAdvantage{};
        for (std::size_t step = steps; step > 0; --step)
        {
            Row& row = rows[(step - 1) * streams + stream];
            row.advantage = core::GaeStep(row.reward, row.value, row.nextValue, followingAdvantage, gamma,
                lambda, row.end == StepEnd::Terminated, row.end == StepEnd::Truncated);
            row.valueTarget = row.value + row.advantage;
            followingAdvantage = row.advantage;
            learnable += row.learnable ? 1 : 0;
        }
    }
    return learnable;
}

/**
 * @brief One step of experience as PPO needs it.
 * @tparam Scalar float or double, used for rewards, values and advantages.
 */
template<typename Scalar, std::size_t ObservationSize, std::size_t ActionSize>
struct Transition final
{
    using ScalarType = Scalar;
    using Observation = std::array<float, ObservationSize>;
    using Action = std::array<float, ActionSize>;

    std::array<float, ObservationSize> observation{};
    /** @brief Raw sampled action (before clipping to the environment bounds). */
    std::array<float, ActionSize> action{};
    /** @brief log pi_old(action | observation) at collection time. */
    Scalar logProbability{};
    Scalar reward{};
    /** @brief Critic estimate V(observation) at collection time. */
    Scalar value{};
    /** @brief Critic estimate of the next observation; ignored when the episode terminated. */
    Scalar nextValue{};
    StepEnd end = StepEnd::Running;
    /** @brief False for samples that must not train the actor (for instance a destroyed MAPPO agent). */
    bool learnable = true;
    /** @brief Filled by ComputeAdvantages. */
    Scalar advantage{};
    /** @brief Critic regression target value + advantage, filled by ComputeAdvantages. */
    Scalar valueTarget{};
};

namespace detail {

/** @brief Preallocated step-major rows and a committed-step cursor, shared by rollout adapters. */
template<typename Row>
class RolloutRows final
{
public:
    RolloutRows(std::size_t streams, std::size_t steps, Row const& initial = {})
        : streams_(streams), steps_(steps), rows_(CheckedCapacity(streams, steps), initial)
    {
    }

    std::size_t Streams() const noexcept { return streams_; }
    std::size_t Steps() const noexcept { return steps_; }
    std::size_t Capacity() const noexcept { return rows_.size(); }
    std::size_t Collected() const noexcept { return collected_; }
    bool Full() const noexcept { return collected_ == steps_; }
    void Clear() noexcept { collected_ = 0; }

    Row* Data() noexcept { return rows_.data(); }
    Row const* Data() const noexcept { return rows_.data(); }
    Row& operator[](std::size_t index) noexcept { return rows_[index]; }
    Row const& operator[](std::size_t index) const noexcept { return rows_[index]; }

    /** @brief Check both coordinates before flattening them; an invalid stream cannot alias a later step. */
    Row& At(std::size_t step, std::size_t stream)
    {
        if (step >= steps_ || stream >= streams_)
            throw std::out_of_range("PPOKit rollout: step or stream out of range");
        return rows_[step * streams_ + stream];
    }

    /** @brief Access an uncommitted row at the collection cursor. */
    Row& Current(std::size_t stream)
    {
        RequireSpace();
        return At(collected_, stream);
    }

    /** @brief Commit exactly one complete step after the adapter has validated its rows. */
    void Commit()
    {
        RequireSpace();
        ++collected_;
    }

private:
    static std::size_t CheckedCapacity(std::size_t streams, std::size_t steps)
    {
        if (streams == 0 || steps == 0)
            throw std::invalid_argument("PPOKit rollout: dimensions must be positive");
        if (steps > std::numeric_limits<std::size_t>::max() / streams)
            throw std::length_error("PPOKit rollout: row count overflows");
        return streams * steps;
    }

    void RequireSpace() const
    {
        if (Full()) throw std::logic_error("PPOKit rollout: buffer full, run the update first");
    }

    std::size_t streams_;
    std::size_t steps_;
    std::vector<Row> rows_;
    std::size_t collected_ = 0;
};

} // namespace detail

/**
 * @brief Fixed-capacity buffer of one rollout.
 * @details Transitions are stored in collection order: ComputeAdvantages walks them
 * backwards, so one buffer must hold consecutive steps of a single trajectory stream
 * (one agent, or one environment). Episodes may start and end anywhere inside it.
 */
template<typename RowType>
class RolloutStorage
{
public:
    using Row = RowType;
    using Scalar = typename Row::ScalarType;

    /** @brief Preallocate all rows, including any runtime-sized observation and action storage. */
    explicit RolloutStorage(std::size_t capacity, Row const& initial = {})
        : rows_(1, capacity, initial) {}

    std::size_t Capacity() const noexcept { return rows_.Capacity(); }
    std::size_t Size() const noexcept { return rows_.Collected(); }
    bool Full() const noexcept { return rows_.Full(); }
    void Clear() noexcept { rows_.Clear(); }

    Row const& operator[](std::size_t index) const noexcept { return rows_[index]; }

    /**
     * @brief Stores one transition.
     * @throws std::logic_error when the buffer is full.
     * @throws std::invalid_argument when a reward, value or log-probability is not finite:
     *         a NaN would silently corrupt every advantage of the rollout.
     */
    void Append(Row const& row)
    {
        Row& destination = rows_.Current(0);
        if (!std::isfinite(row.reward) || !std::isfinite(row.value) || !std::isfinite(row.nextValue)
            || !std::isfinite(row.logProbability))
        {
            throw std::invalid_argument("PPOKit rollout: non-finite reward, value or log-probability");
        }
        if (row.observation.size() != destination.observation.size()
            || row.action.size() != destination.action.size())
        {
            throw std::invalid_argument("PPOKit rollout: transition dimensions differ from the buffer");
        }
        destination = row;
        rows_.Commit();
    }

    /**
     * @brief Generalized Advantage Estimation over the stored transitions.
     * @details The recursion restarts at each episode end. Rows after the last stored
     * one are unknown, so the trace of the final row starts from zero: its advantage
     * is a one-step estimate bootstrapped on its nextValue.
     * @return Number of rows eligible to train the actor.
     */
    std::size_t ComputeAdvantages(Scalar gamma, Scalar lambda) noexcept
    {
        return ComputeTrajectoryAdvantages(rows_.Data(), Size(), 1, gamma, lambda);
    }

    /**
     * @brief Shifts and scales the advantages of learnable rows to zero mean and unit variance.
     * @details Uses the population variance; rows that are not learnable keep their value.
     * Normalizing makes the actor step size independent of the reward scale.
     */
    void NormalizeAdvantages() noexcept
    {
        NormalizeLearnableAdvantages(rows_.Data(), Size());
    }

    /** @brief Critic quality over the stored rows (see ExplainedVarianceOver). */
    Scalar ExplainedVariance() const noexcept { return ExplainedVarianceOf(rows_.Data(), Size()); }

private:
    detail::RolloutRows<Row> rows_;
};

/** @brief Fixed-dimension adapter for existing collectors. */
template<typename Scalar, std::size_t ObservationSize, std::size_t ActionSize>
class RolloutBuffer final : public RolloutStorage<Transition<Scalar, ObservationSize, ActionSize>>
{
    using Storage = RolloutStorage<Transition<Scalar, ObservationSize, ActionSize>>;
public:
    explicit RolloutBuffer(std::size_t capacity) : Storage(capacity) {}
};

/** @brief Transition with dimensions declared by the loaded task, not by the compiler. */
struct DynamicTransition final
{
    using ScalarType = double;
    std::vector<float> observation;
    std::vector<float> action;
    double logProbability{};
    double reward{};
    double value{};
    double nextValue{};
    StepEnd end = StepEnd::Running;
    bool learnable = true;
    double advantage{};
    double valueTarget{};
};

/** @brief Runtime-sized rollout; all transition vectors are allocated at construction. */
class DynamicRolloutBuffer final : public RolloutStorage<DynamicTransition>
{
    using Storage = RolloutStorage<DynamicTransition>;
public:
    DynamicRolloutBuffer(std::size_t capacity, std::size_t observationSize, std::size_t actionSize)
        : Storage(capacity, MakeRow(observationSize, actionSize)) {}

private:
    static DynamicTransition MakeRow(std::size_t observationSize, std::size_t actionSize)
    {
        if (observationSize == 0 || actionSize == 0)
            throw std::invalid_argument("PPOKit rollout: dimensions must be positive");
        DynamicTransition row;
        row.observation.resize(observationSize);
        row.action.resize(actionSize);
        return row;
    }
};

/**
 * @brief Fraction of the variance of the value targets that the critic explains.
 * @details 1 - Var(target - value) / Var(target), computed before the update on all
 * stored rows. 1 is a perfect critic, 0 a critic no better than a constant, and a
 * negative value a critic worse than a constant. Returns 0 when the targets do not vary.
 * Call it after ComputeAdvantages and before NormalizeAdvantages.
 */
template<typename Scalar, std::size_t ObservationSize, std::size_t ActionSize>
Scalar ExplainedVariance(RolloutBuffer<Scalar, ObservationSize, ActionSize> const& buffer) noexcept
{
    return buffer.ExplainedVariance();
}

} // namespace ppokit::runtime
