/**
 * @file Examples/BlueDefense/Evaluation/generalized_evaluation_metrics.hpp
 * @brief Comptabilisation des scores et métriques par épisode et campagne.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "generalized_cooperative_reward.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Final deterministic sample of one generalized P-versus-N episode.
 *
 * Quality values are indexed by semantic responsibility, never by a red
 * target. Destroyed participants remain active and are scored as zero quality.
 */
struct GeneralizedEvaluationSample final
{
    /** @brief Episode membership, final survival, and objective state. */
    GeneralizedCombatStateMask finalState{};
    /** @brief Normalized leader patrol quality in [0, 1]. */
    float leaderPatrolQuality{};
    /** @brief Normalized formation quality by immutable teammate slot. */
    std::array<float, kMaximumFormationSlots> formationSlotQuality{};
    /** @brief True when physics or world resolution failed the mission. */
    bool missionFailed{};
    /** @brief True when the episode exhausted its deterministic step limit. */
    bool timeout{};
    /** @brief Qualified Pvs0 patrol success, not a mere empty hostile set. */
    bool patrolCompleted{};
};

/** @brief Result of adding one generalized evaluation sample. */
enum class GeneralizedEvaluationStatus : std::uint8_t
{
    Recorded,
    InvalidRoles,
    InvalidMasks,
    InvalidQuality
};

/** @brief Normalized results of one evaluation episode, as a campaign averages them. */
struct GeneralizedEpisodeEvaluation final
{
    /** @brief Fraction of the episode's P blue mobiles still alive. */
    double blueSurvival{};
    /** @brief Fraction of the episode's N red mobiles neutralized (1 for N = 0). */
    double redNeutralization{};
    /** @brief Leader patrol quality, 0 when the leader was lost. */
    double patrolQuality{};
    /** @brief Mean quality of the surviving teammates' formation slots (1 for P = 1). */
    double formationQuality{};
    bool objectiveSurvived{};
    /** @brief Objective safe, no red remaining, a blue alive (and the patrol completed for N = 0). */
    bool success{};
    bool timeout{};
};

/**
 * @brief Stable weighted score in [0, 100] of campaign means (or of one episode's values:
 *        the score of a campaign is the mean of its episodes' scores).
 */
inline double GeneralizedEvaluationScore(double objectiveSurvival, double missionSuccess,
    double blueSurvival, double redNeutralization, double leaderPatrolQuality,
    double formationQuality) noexcept
{
    double const score = 25.0 * objectiveSurvival
        + 20.0 * missionSuccess
        + 15.0 * blueSurvival
        + 15.0 * redNeutralization
        + 15.0 * leaderPatrolQuality
        + 10.0 * formationQuality;
    return std::clamp(score, 0.0, 100.0);
}

/** @brief Score of one episode (GeneralizedEvaluationScore of its values). */
inline double GeneralizedEpisodeScore(GeneralizedEpisodeEvaluation const& episode) noexcept
{
    return GeneralizedEvaluationScore(episode.objectiveSurvived ? 1.0 : 0.0,
        episode.success ? 1.0 : 0.0, episode.blueSurvival, episode.redNeutralization,
        episode.patrolQuality, episode.formationQuality);
}

/**
 * @brief Validates one complete episode and computes its normalized results.
 * @param roles Stable leader and teammate-slot assignment.
 * @param sample Final masks and responsibility qualities.
 * @param result Receives the results when the sample is valid.
 * @return Recorded, or the rejected contract category (result untouched).
 *
 * Survival and neutralization are divided by episode P and N, so a 9v10 case cannot
 * dominate a 1v0 case. For N=0, red neutralization is complete by definition. For P=1,
 * formation responsibility is complete because no teammate slot exists.
 */
inline GeneralizedEvaluationStatus EvaluateGeneralizedEpisode(
    BlueTeamRoles const& roles,
    GeneralizedEvaluationSample const& sample,
    GeneralizedEpisodeEvaluation& result) noexcept
{
    if (!roles.IsValid())
        return GeneralizedEvaluationStatus::InvalidRoles;
    std::size_t const blueCount = roles.ActiveBlueCount();
    if (!generalized_reward_detail::ValidMasks(
            sample.finalState, sample.finalState, blueCount))
    {
        return GeneralizedEvaluationStatus::InvalidMasks;
    }
    if (!std::isfinite(sample.leaderPatrolQuality)
        || sample.leaderPatrolQuality < 0.F
        || sample.leaderPatrolQuality > 1.F)
    {
        return GeneralizedEvaluationStatus::InvalidQuality;
    }
    for (std::size_t slot = 0U; slot + 1U < blueCount; ++slot)
    {
        float const quality = sample.formationSlotQuality[slot];
        if (!std::isfinite(quality) || quality < 0.F || quality > 1.F)
            return GeneralizedEvaluationStatus::InvalidQuality;
    }

    std::size_t const redCount = generalized_reward_detail::Count(
        sample.finalState.redActive);
    std::size_t const survivingBlue = generalized_reward_detail::Count(
        sample.finalState.blueAlive);
    std::size_t const survivingRed = generalized_reward_detail::Count(
        sample.finalState.redAlive);

    GeneralizedEpisodeEvaluation episode{};
    episode.blueSurvival = static_cast<double>(survivingBlue)
        / static_cast<double>(blueCount);
    episode.redNeutralization = redCount > 0U
        ? static_cast<double>(redCount - survivingRed)
            / static_cast<double>(redCount)
        : 1.0;

    std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
    episode.patrolQuality = sample.finalState.blueAlive[leaderIndex]
        ? static_cast<double>(sample.leaderPatrolQuality) : 0.0;
    episode.formationQuality = 1.0;
    if (blueCount > 1U)
    {
        double sum = 0.0;
        for (std::size_t slot = 0U; slot + 1U < blueCount; ++slot)
        {
            bool alive = false;
            for (std::size_t physicalIndex = 0U;
                 physicalIndex < blueCount; ++physicalIndex)
            {
                BlueRoleAssignment const assignment =
                    *roles.Assignment(physicalIndex);
                if (assignment.role == BlueRole::Teammate
                    && assignment.formationSlot == slot)
                {
                    alive = sample.finalState.blueAlive[physicalIndex];
                    break;
                }
            }
            if (alive)
                sum += sample.formationSlotQuality[slot];
        }
        episode.formationQuality = sum / static_cast<double>(blueCount - 1U);
    }

    episode.objectiveSurvived = sample.finalState.objectiveAvailable;
    episode.success = episode.objectiveSurvived && !sample.missionFailed
        && survivingBlue > 0U && survivingRed == 0U
        && (redCount > 0U || sample.patrolCompleted);
    episode.timeout = sample.timeout;
    result = episode;
    return GeneralizedEvaluationStatus::Recorded;
}

static_assert(std::is_trivially_copyable_v<GeneralizedEvaluationSample>,
    "Generalized evaluation samples must remain fixed-capacity values");

} // namespace ppo::examples::bluedefense::defenders
