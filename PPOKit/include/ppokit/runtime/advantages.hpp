/**
 * @file PPOKit/include/ppokit/runtime/advantages.hpp
 * @brief Advantage normalization and explained variance shared by every PPOKit rollout.
 * @details Rows are any transition type with `advantage`, `value`, `valueTarget` (of type
 * Row::ScalarType) and `learnable`. RolloutBuffer (one trajectory stream) and TeamRollout
 * (synchronized agents) both use these functions, so single-agent and MAPPO updates
 * normalize and diagnose identically.
 */
#pragma once

#include <cmath>
#include <cstddef>

namespace ppokit::runtime {

/** @brief Small constant that keeps normalization and explained variance finite. */
template<typename Scalar>
constexpr Scalar AdvantageEpsilon = Scalar(1e-8);

/**
 * @brief Shifts and scales the advantages of learnable rows to zero mean and unit variance.
 * @details Population variance; rows that are not learnable keep their value. Normalizing
 * makes the actor step size independent of the reward scale.
 * @return Number of learnable rows (nothing changes when it is zero).
 */
template<typename Row>
std::size_t NormalizeLearnableAdvantages(Row* rows, std::size_t count) noexcept
{
    using Scalar = typename Row::ScalarType;
    std::size_t learnable = 0;
    Scalar mean{};
    for (std::size_t index = 0; index < count; ++index)
    {
        if (rows[index].learnable)
        {
            mean += rows[index].advantage;
            ++learnable;
        }
    }
    if (learnable == 0)
    {
        return 0;
    }
    mean /= static_cast<Scalar>(learnable);
    Scalar variance{};
    for (std::size_t index = 0; index < count; ++index)
    {
        if (rows[index].learnable)
        {
            Scalar const centered = rows[index].advantage - mean;
            variance += centered * centered;
        }
    }
    Scalar const scale = Scalar(1) / std::sqrt(variance / static_cast<Scalar>(learnable) + AdvantageEpsilon<Scalar>);
    for (std::size_t index = 0; index < count; ++index)
    {
        if (rows[index].learnable)
        {
            rows[index].advantage = (rows[index].advantage - mean) * scale;
        }
    }
    return learnable;
}

/**
 * @brief Fraction of the variance of the value targets that the critic explains, over the
 *        rows `rowAt(0) .. rowAt(count - 1)`, in that order.
 * @details 1 - Var(target - value) / Var(target) over all rows, learnable or not (every
 * row trains the critic). 1 is a perfect critic, 0 no better than a constant, negative
 * worse than a constant. Returns 0 when the targets do not vary. Call it after the
 * advantages are computed and before they are normalized.
 */
template<typename Scalar, typename RowAt>
Scalar ExplainedVarianceOver(std::size_t count, RowAt const& rowAt) noexcept
{
    if (count == 0)
    {
        return Scalar(0);
    }
    Scalar const inverse = Scalar(1) / static_cast<Scalar>(count);
    Scalar targetMean{};
    Scalar residualMean{};
    for (std::size_t index = 0; index < count; ++index)
    {
        auto const& row = rowAt(index);
        targetMean += row.valueTarget;
        residualMean += row.valueTarget - row.value;
    }
    targetMean *= inverse;
    residualMean *= inverse;
    Scalar targetVariance{};
    Scalar residualVariance{};
    for (std::size_t index = 0; index < count; ++index)
    {
        auto const& row = rowAt(index);
        Scalar const target = row.valueTarget - targetMean;
        Scalar const residual = row.valueTarget - row.value - residualMean;
        targetVariance += target * target;
        residualVariance += residual * residual;
    }
    targetVariance *= inverse;
    residualVariance *= inverse;
    if (targetVariance <= AdvantageEpsilon<Scalar>)
    {
        return Scalar(0);
    }
    Scalar const explained = Scalar(1) - residualVariance / targetVariance;
    return std::isfinite(explained) ? explained : Scalar(0);
}

/** @brief ExplainedVarianceOver the `count` contiguous rows starting at `rows`. */
template<typename Row>
typename Row::ScalarType ExplainedVarianceOf(Row const* rows, std::size_t count) noexcept
{
    return ExplainedVarianceOver<typename Row::ScalarType>(
        count, [rows](std::size_t index) -> Row const& { return rows[index]; });
}

} // namespace ppokit::runtime
