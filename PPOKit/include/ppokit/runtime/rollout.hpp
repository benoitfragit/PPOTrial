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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

/**
 * @brief Fixed-capacity buffer of one rollout.
 * @details Transitions are stored in collection order: ComputeAdvantages walks them
 * backwards, so one buffer must hold consecutive steps of a single trajectory stream
 * (one agent, or one environment). Episodes may start and end anywhere inside it.
 */
template<typename Scalar, std::size_t ObservationSize, std::size_t ActionSize>
class RolloutBuffer final
{
public:
    using Row = Transition<Scalar, ObservationSize, ActionSize>;

    /** @brief Small constant added to the variance before its square root. */
    static constexpr Scalar AdvantageEpsilon = Scalar(1e-8);

    /** @throws std::invalid_argument when the capacity is zero. */
    explicit RolloutBuffer(std::size_t capacity)
    {
        if (capacity == 0)
        {
            throw std::invalid_argument("PPOKit rollout: the capacity must be positive");
        }
        rows_.resize(capacity);
    }

    std::size_t Capacity() const noexcept { return rows_.size(); }
    std::size_t Size() const noexcept { return size_; }
    bool Full() const noexcept { return size_ == rows_.size(); }
    void Clear() noexcept { size_ = 0; }

    Row const& operator[](std::size_t index) const noexcept { return rows_[index]; }

    /**
     * @brief Stores one transition.
     * @throws std::logic_error when the buffer is full.
     * @throws std::invalid_argument when a reward, value or log-probability is not finite:
     *         a NaN would silently corrupt every advantage of the rollout.
     */
    void Append(Row const& row)
    {
        if (Full())
        {
            throw std::logic_error("PPOKit rollout: buffer full, run the update first");
        }
        if (!std::isfinite(row.reward) || !std::isfinite(row.value) || !std::isfinite(row.nextValue)
            || !std::isfinite(row.logProbability))
        {
            throw std::invalid_argument("PPOKit rollout: non-finite reward, value or log-probability");
        }
        rows_[size_] = row;
        ++size_;
    }

    /**
     * @brief Generalized Advantage Estimation over the stored transitions.
     * @details The recursion restarts at each episode end. Rows after the last stored
     * one are unknown, so the trace of the final row starts from zero: its advantage
     * is a one-step estimate bootstrapped on its nextValue.
     */
    void ComputeAdvantages(Scalar gamma, Scalar lambda) noexcept
    {
        Scalar followingAdvantage{};
        for (std::size_t index = size_; index > 0; --index)
        {
            Row& row = rows_[index - 1];
            row.advantage = core::GaeStep(row.reward, row.value, row.nextValue, followingAdvantage, gamma, lambda,
                row.end == StepEnd::Terminated, row.end == StepEnd::Truncated);
            row.valueTarget = row.value + row.advantage;
            followingAdvantage = row.advantage;
        }
    }

    /**
     * @brief Shifts and scales the advantages of learnable rows to zero mean and unit variance.
     * @details Uses the population variance; rows that are not learnable keep their value.
     * Normalizing makes the actor step size independent of the reward scale.
     */
    void NormalizeAdvantages() noexcept
    {
        std::size_t learnable = 0;
        Scalar sum{};
        for (std::size_t index = 0; index < size_; ++index)
        {
            if (rows_[index].learnable)
            {
                sum += rows_[index].advantage;
                ++learnable;
            }
        }
        if (learnable == 0)
        {
            return;
        }
        Scalar const mean = sum / static_cast<Scalar>(learnable);
        Scalar squaredDeviations{};
        for (std::size_t index = 0; index < size_; ++index)
        {
            if (rows_[index].learnable)
            {
                Scalar const centered = rows_[index].advantage - mean;
                squaredDeviations += centered * centered;
            }
        }
        Scalar const scale =
            Scalar(1) / std::sqrt(squaredDeviations / static_cast<Scalar>(learnable) + AdvantageEpsilon);
        for (std::size_t index = 0; index < size_; ++index)
        {
            if (rows_[index].learnable)
            {
                rows_[index].advantage = (rows_[index].advantage - mean) * scale;
            }
        }
    }

private:
    std::vector<Row> rows_;
    std::size_t size_ = 0;
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
    std::size_t const count = buffer.Size();
    if (count == 0)
    {
        return Scalar(0);
    }
    Scalar targetSum{};
    Scalar residualSum{};
    for (std::size_t index = 0; index < count; ++index)
    {
        targetSum += buffer[index].valueTarget;
        residualSum += buffer[index].valueTarget - buffer[index].value;
    }
    Scalar const targetMean = targetSum / static_cast<Scalar>(count);
    Scalar const residualMean = residualSum / static_cast<Scalar>(count);
    Scalar targetVariance{};
    Scalar residualVariance{};
    for (std::size_t index = 0; index < count; ++index)
    {
        Scalar const target = buffer[index].valueTarget - targetMean;
        Scalar const residual = buffer[index].valueTarget - buffer[index].value - residualMean;
        targetVariance += target * target;
        residualVariance += residual * residual;
    }
    if (targetVariance <= RolloutBuffer<Scalar, ObservationSize, ActionSize>::AdvantageEpsilon)
    {
        return Scalar(0);
    }
    Scalar const explained = Scalar(1) - residualVariance / targetVariance;
    return std::isfinite(explained) ? explained : Scalar(0);
}

} // namespace ppokit::runtime
