#pragma once
#include <chrono>

namespace ppo::model::detail {

/**
 * @brief Returns the desktop throttle, or no delay at maximum speed.
 * @param headless Whether the simulation runs without pacing.
 * @return Delay after each simulation iteration.
 */
inline std::chrono::steady_clock::duration StepDelay(bool headless) noexcept
{
    if (headless) return std::chrono::steady_clock::duration::zero();
    return std::chrono::milliseconds(2);
}

} // namespace ppo::model::detail
