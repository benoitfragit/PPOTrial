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
 * Disabled in legacy configurations to preserve the Run 16 actor's input and
 * reward contract unless the new training configuration explicitly opts in.
 * These settings are runtime state, never serialized in the V6 optimizer.
 */
struct PatrolNoThreatSettings final
{
    bool enabled{false};
    float positionToleranceMeters{2500.F};
    float speedToleranceFraction{0.20F};
    float headingToleranceRadians{0.35F};
    float stepRewardScale{0.002F};
    float minimumSeparationMeters{2500.F};
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
        && settings.minimumSeparationMeters > 0.F;
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
