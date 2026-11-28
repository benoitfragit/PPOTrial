/**
 * @file PPOKit/include/ppokit/runtime/probe_moments.hpp
 * @brief Running statistics of what one network block outputs, shared by every probe
 *        (graph engine, rl_tools MLPs).
 */
#pragma once

#include "ppokit/runtime/probe_record.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace ppokit::runtime {

/** @brief Threshold of a saturated output, for tanh and sigmoid blocks. */
inline constexpr double ProbeSaturationLevel = 0.99;

/** @brief Mean, spread, range, dead and saturated shares of observed outputs. No allocation. */
struct ProbeMoments final
{
    std::size_t count = 0;
    double sum = 0.0;
    double squares = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    std::size_t zeros = 0;
    std::size_t saturated = 0;

    void Add(double value) noexcept
    {
        sum += value;
        squares += value * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        zeros += value == 0.0 ? 1 : 0;
        saturated += std::abs(value) >= ProbeSaturationLevel ? 1 : 0;
        ++count;
    }

    void Add(float const* values, std::size_t width) noexcept
    {
        for (std::size_t index = 0; index < width; ++index)
        {
            Add(static_cast<double>(values[index]));
        }
    }

    /** @brief Writes the output statistics of `probe` (its gradient fields are left alone). */
    void Fill(BlockProbe& probe) const noexcept
    {
        probe.samples = count;
        if (count == 0)
        {
            return;
        }
        double const total = static_cast<double>(count);
        probe.mean = sum / total;
        probe.deviation = std::sqrt(std::max(squares / total - probe.mean * probe.mean, 0.0));
        probe.minimum = minimum;
        probe.maximum = maximum;
        probe.deadShare = static_cast<double>(zeros) / total;
        probe.saturatedShare = static_cast<double>(saturated) / total;
    }
};

} // namespace ppokit::runtime
