/**
 * @file Examples/BlueDefense/Evaluation/generalized_evaluation_case.hpp
 * @brief The reset of a case of the BlueDefense evaluation suite: its seed, patrol side and
 *        racetrack phase (blue_environment's reset_evaluation).
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace ppo::examples::bluedefense::defenders {

/** @brief Reproducible reset parameters for one generalized evaluation case. */
struct GeneralizedEvaluationCase final
{
    /** @brief Seed applied before the complete ECS episode reset. */
    std::uint32_t seed{};
    /** @brief Mirrored patrol direction, always -1 or +1. */
    int side{1};
    /** @brief Desired dimensionless phase on the next sampled racetrack. */
    float phaseFraction{};
};

/**
 * @brief The reset case of the campaign's episode `episodeIndex`, from that episode's seed:
 *        alternating patrol sides, and the four quarter-circuit phases in turn.
 * @param seed Seed of the episode (the campaign's first seed plus the episode index).
 * @param episodeIndex Zero-based place of the episode in its campaign.
 */
inline GeneralizedEvaluationCase GeneralizedEvaluationCaseOf(
    std::uint32_t const seed, std::size_t const episodeIndex) noexcept
{
    GeneralizedEvaluationCase result{};
    result.seed = seed;
    result.side = episodeIndex % 2U == 0U ? 1 : -1;
    result.phaseFraction =
        static_cast<float>((episodeIndex / 2U) % 4U) / 4.F;
    return result;
}

} // namespace ppo::examples::bluedefense::defenders
