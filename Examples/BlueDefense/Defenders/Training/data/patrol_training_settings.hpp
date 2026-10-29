#pragma once

#include "patrol_core.hpp"
#include "patrol_evaluation_settings.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Returns whether a rollout length has a compiled RLtools backend.
 * @param steps Number of ECS team steps requested per MAPPO update.
 * @return True for supported power-of-two rollout presets.
 */
inline constexpr bool IsSupportedRolloutSize(std::size_t steps) noexcept
{
    return steps == 128U || steps == 256U || steps == 512U
        || steps == 1024U || steps == 2048U || steps == 4096U;
}

/**
 * @brief Runtime settings for the embedded blue MAPPO trainer.
 */
struct PatrolTrainingSettings final
{
    std::size_t rolloutSteps{2048U};
    std::uint64_t maxUpdates{300U};
    std::uint64_t checkpointEvery{10U};
    std::uint32_t seed{42U};
    float rewardClamp{100.F};
    RewardValues rewardWeights{1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F};
    PatrolEvaluationSettings evaluation{};
};

/**
 * @brief Validates one blue MAPPO configuration.
 * @param settings Candidate settings.
 * @return True when every value can be applied safely.
 */
inline bool Valid(PatrolTrainingSettings const& settings) noexcept
{
    if (!IsSupportedRolloutSize(settings.rolloutSteps)
        || settings.maxUpdates == 0U || settings.checkpointEvery == 0U
        || !std::isfinite(settings.rewardClamp) || settings.rewardClamp <= 0.F
        || !Valid(settings.evaluation))
        return false;
    return std::all_of(settings.rewardWeights.begin(), settings.rewardWeights.end(),
        [](float weight) { return std::isfinite(weight) && weight >= 0.F; });
}

} // namespace ppo::examples::bluedefense::defenders
