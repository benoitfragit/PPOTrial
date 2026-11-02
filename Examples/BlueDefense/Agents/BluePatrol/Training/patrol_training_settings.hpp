/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/patrol_training_settings.hpp
 * @brief Implémentation des calculs et contrats d'entraînement patrol training settings.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
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
 * @brief Optional Blue-only P-versus-zero patrol task.
 *
 * V3 uses observation v10 and starts fresh. These runtime settings are carried
 * by JSON, while checkpoint headers independently validate the observation
 * semantics. The curriculum fingerprint validates its own scheduling contract.
 */
struct PatrolNoThreatSettings final
{
    bool enabled{true};
    float positionToleranceMeters{2500.F};
    float speedToleranceFraction{0.20F};
    float headingToleranceRadians{0.35F};
    float stepRewardScale{0.002F};
    float minimumSeparationMeters{2500.F};
    /** @brief Anticipation in observations; measurement targets stay instantaneous. */
    float lookaheadSeconds{3.F};
    /** @brief Strict by default; a small V3 transition budget is explicit in JSON. */
    float minimumQualifiedFraction{1.F};
    /** @brief Cross-track distance that arms the irreversible FOLLOW mode. */
    float rallyEnterDistanceMeters{2500.F};
    /** @brief Wider release threshold used by the RALLY corridor latch. */
    float rallyExitDistanceMeters{3000.F};
    /** @brief N=0 simulated-time horizon; other scenarios keep their own limit. */
    float maxEpisodeSeconds{480.F};
};

/** @brief Validate dimensioned patrol tolerances and bounded shaping. */
inline bool Valid(PatrolNoThreatSettings const& settings) noexcept
{
    return std::isfinite(settings.positionToleranceMeters)
        && settings.positionToleranceMeters > 0.F
        && std::isfinite(settings.speedToleranceFraction)
        && settings.speedToleranceFraction > 0.F
        && settings.speedToleranceFraction <= 1.F
        && std::isfinite(settings.headingToleranceRadians)
        && settings.headingToleranceRadians > 0.F
        && settings.headingToleranceRadians <= 3.14159265358979323846F
        && std::isfinite(settings.stepRewardScale)
        && settings.stepRewardScale >= 0.F
        && settings.stepRewardScale <= 0.02F
        && std::isfinite(settings.minimumSeparationMeters)
        && settings.minimumSeparationMeters > 0.F
        && std::isfinite(settings.lookaheadSeconds)
        && settings.lookaheadSeconds >= 0.F && settings.lookaheadSeconds <= 30.F
        && std::isfinite(settings.minimumQualifiedFraction)
        && settings.minimumQualifiedFraction >= 0.8F
        && settings.minimumQualifiedFraction <= 1.F
        && std::isfinite(settings.rallyEnterDistanceMeters)
        && settings.rallyEnterDistanceMeters > 0.F
        && std::isfinite(settings.rallyExitDistanceMeters)
        && settings.rallyExitDistanceMeters
            > settings.rallyEnterDistanceMeters
        && std::isfinite(settings.maxEpisodeSeconds)
        && settings.maxEpisodeSeconds > 0.F;
}

/**
 * @brief Converts the N=0 simulated-time horizon to a bounded step count.
 * @param settings Valid no-threat patrol settings.
 * @param dt Simulation step duration in seconds.
 * @param fallbackSteps Existing scenario or evaluation step horizon.
 * @return The earlier positive horizon, rounded up to the first expired step.
 */
inline std::size_t NoThreatEpisodeLimitSteps(
    PatrolNoThreatSettings const& settings, float const dt,
    std::size_t const fallbackSteps) noexcept
{
    if (!Valid(settings) || !std::isfinite(dt) || dt <= 0.F
        || fallbackSteps == 0U)
    {
        return fallbackSteps;
    }
    double const exact = static_cast<double>(settings.maxEpisodeSeconds)
        / static_cast<double>(dt);
    double const bounded = std::min(
        std::ceil(exact), static_cast<double>(fallbackSteps));
    return static_cast<std::size_t>(std::max(1.0, bounded));
}

/**
 * @brief Runtime settings for the embedded blue MAPPO trainer.
 */
struct PatrolTrainingSettings final
{
    std::size_t rolloutSteps{2048U};
    std::uint64_t maxUpdates{1000U};
    std::uint64_t checkpointEvery{10U}; // Deterministic evaluation cadence.
    std::uint64_t saveEveryUpdates{10U}; // Independent full-state checkpoint cadence.
    std::uint32_t seed{42U};
    float rewardClamp{100.F};
    // Generalized blue-only channels; legacy weights retain their mapping.
    float ownLossWeight{24.F};
    float lowSpeedRiskWeight{5.F};
    RewardValues rewardWeights{1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F};
    PatrolEvaluationSettings evaluation{};
    PatrolNoThreatSettings noThreatPatrol{};
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
        || settings.saveEveryUpdates == 0U
        || !std::isfinite(settings.rewardClamp) || settings.rewardClamp <= 0.F
        || !std::isfinite(settings.ownLossWeight) || settings.ownLossWeight < 0.F
        || !std::isfinite(settings.lowSpeedRiskWeight) || settings.lowSpeedRiskWeight < 0.F
        || !Valid(settings.evaluation) || !Valid(settings.noThreatPatrol))
        return false;
    return std::all_of(settings.rewardWeights.begin(), settings.rewardWeights.end(),
        [](float weight) { return std::isfinite(weight) && weight >= 0.F; });
}

} // namespace ppo::examples::bluedefense::defenders
