#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Deterministic test-suite settings shared by patrol evaluators.
 */
struct PatrolEvaluationSettings final
{
    std::size_t episodes{8U};
    std::size_t maxSteps{4096U};
    std::uint32_t seed{4242U};
    float minimumCircuits{1.F};
    float trackingTolerance{0.10F};
    float minimumTrackingFraction{0.80F};
    float minimumHeadingCosine{0.60F};
    float minimumSpeedFraction{0.50F};
    float collisionDistance{1000.F};
    float restrictedNorth{0.F};
    float restrictedEast{0.F};
    float restrictedRadius{0.F};
};

/**
 * @brief Validates deterministic patrol evaluation settings.
 * @param settings Candidate settings.
 * @return True when the campaign is finite and numerically valid.
 */
inline bool Valid(PatrolEvaluationSettings const& settings) noexcept
{
    return settings.episodes > 0U && settings.episodes <= 10000U
        && settings.maxSteps > 0U && settings.maxSteps <= 1000000U
        && std::isfinite(settings.minimumCircuits)
        && settings.minimumCircuits >= 0.F
        && std::isfinite(settings.trackingTolerance)
        && settings.trackingTolerance > 0.F
        && std::isfinite(settings.minimumTrackingFraction)
        && settings.minimumTrackingFraction >= 0.F
        && settings.minimumTrackingFraction <= 1.F
        && std::isfinite(settings.minimumHeadingCosine)
        && settings.minimumHeadingCosine >= -1.F
        && settings.minimumHeadingCosine <= 1.F
        && std::isfinite(settings.minimumSpeedFraction)
        && settings.minimumSpeedFraction >= 0.F
        && settings.minimumSpeedFraction <= 1.F
        && std::isfinite(settings.collisionDistance)
        && settings.collisionDistance > 0.F
        && std::isfinite(settings.restrictedNorth)
        && std::isfinite(settings.restrictedEast)
        && std::isfinite(settings.restrictedRadius)
        && settings.restrictedRadius >= 0.F;
}

} // namespace ppo::examples::bluedefense::defenders
