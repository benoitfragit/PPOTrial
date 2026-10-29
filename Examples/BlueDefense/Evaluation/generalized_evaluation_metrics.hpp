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
};

/** @brief Final normalized metrics of a deterministic evaluation campaign. */
struct GeneralizedEvaluationMetrics final
{
    /** @brief Number of valid episodes included in this report. */
    std::size_t episodes{};
    /** @brief Mean fraction of active P blue mobiles still alive. */
    float blueSurvivalRate{};
    /** @brief Mean fraction of active N red mobiles neutralized. */
    float redNeutralizationRate{};
    /** @brief Fraction of episodes where the fixed objective survived. */
    float objectiveSurvivalRate{};
    /** @brief Mean surviving-leader patrol responsibility quality. */
    float leaderPatrolQuality{};
    /** @brief Mean surviving-teammate formation-slot quality. */
    float formationQuality{};
    /** @brief Fraction of episodes with objective safe and no red remaining. */
    float missionSuccessRate{};
    /** @brief Fraction of episodes that reached their step limit. */
    float timeoutRate{};
    /** @brief Stable weighted score in [0, 100]. */
    float score{};
};

/** @brief Result of adding one generalized evaluation sample. */
enum class GeneralizedEvaluationStatus : std::uint8_t
{
    Recorded,
    InvalidRoles,
    InvalidMasks,
    InvalidQuality
};

/**
 * @brief Allocation-free deterministic campaign metric accumulator.
 *
 * Survival and neutralization are divided by episode P and N before campaign
 * averaging, so a 9v10 case cannot dominate a 1v0 case. For N=0, red
 * neutralization is complete by definition. For P=1, formation responsibility
 * is complete because no teammate slot exists.
 */
class GeneralizedEvaluationAccumulator final
{
public:
    /** @brief Clears all recorded episodes and accumulated metrics. */
    void Reset() noexcept
    {
        *this = {};
    }

    /**
     * @brief Validates and records one complete episode.
     * @param roles Stable leader and teammate-slot assignment.
     * @param sample Final masks and responsibility qualities.
     * @return Recorded, or the rejected contract category without mutation.
     */
    GeneralizedEvaluationStatus Record(
        BlueTeamRoles const& roles,
        GeneralizedEvaluationSample const& sample) noexcept
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

        double const blueSurvival = static_cast<double>(survivingBlue)
            / static_cast<double>(blueCount);
        double const redNeutralization = redCount > 0U
            ? static_cast<double>(redCount - survivingRed)
                / static_cast<double>(redCount)
            : 1.0;

        std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
        double const patrolQuality = sample.finalState.blueAlive[leaderIndex]
            ? static_cast<double>(sample.leaderPatrolQuality) : 0.0;
        double formationQuality = 1.0;
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
            formationQuality = sum / static_cast<double>(blueCount - 1U);
        }

        bool const objectiveSurvived =
            sample.finalState.objectiveAvailable;
        bool const success = objectiveSurvived && !sample.missionFailed
            && survivingRed == 0U;

        ++episodes_;
        blueSurvivalSum_ += blueSurvival;
        redNeutralizationSum_ += redNeutralization;
        objectiveSurvivalSum_ += objectiveSurvived ? 1.0 : 0.0;
        patrolQualitySum_ += patrolQuality;
        formationQualitySum_ += formationQuality;
        missionSuccessSum_ += success ? 1.0 : 0.0;
        timeoutSum_ += sample.timeout ? 1.0 : 0.0;
        return GeneralizedEvaluationStatus::Recorded;
    }

    /**
     * @brief Returns a normalized snapshot without mutating the campaign.
     * @return Zero metrics when no episode was recorded.
     */
    GeneralizedEvaluationMetrics Metrics() const noexcept
    {
        GeneralizedEvaluationMetrics result{};
        result.episodes = episodes_;
        if (episodes_ == 0U)
            return result;
        double const inverseCount = 1.0 / static_cast<double>(episodes_);
        result.blueSurvivalRate = static_cast<float>(
            blueSurvivalSum_ * inverseCount);
        result.redNeutralizationRate = static_cast<float>(
            redNeutralizationSum_ * inverseCount);
        result.objectiveSurvivalRate = static_cast<float>(
            objectiveSurvivalSum_ * inverseCount);
        result.leaderPatrolQuality = static_cast<float>(
            patrolQualitySum_ * inverseCount);
        result.formationQuality = static_cast<float>(
            formationQualitySum_ * inverseCount);
        result.missionSuccessRate = static_cast<float>(
            missionSuccessSum_ * inverseCount);
        result.timeoutRate = static_cast<float>(timeoutSum_ * inverseCount);

        double const score = 25.0 * result.objectiveSurvivalRate
            + 20.0 * result.missionSuccessRate
            + 15.0 * result.blueSurvivalRate
            + 15.0 * result.redNeutralizationRate
            + 15.0 * result.leaderPatrolQuality
            + 10.0 * result.formationQuality;
        result.score = static_cast<float>(std::clamp(score, 0.0, 100.0));
        return result;
    }

private:
    std::size_t episodes_{};
    double blueSurvivalSum_{};
    double redNeutralizationSum_{};
    double objectiveSurvivalSum_{};
    double patrolQualitySum_{};
    double formationQualitySum_{};
    double missionSuccessSum_{};
    double timeoutSum_{};
};

static_assert(std::is_trivially_copyable_v<GeneralizedEvaluationSample>,
    "Generalized evaluation samples must remain fixed-capacity values");
static_assert(std::is_trivially_copyable_v<GeneralizedEvaluationAccumulator>,
    "Generalized evaluation metrics must not allocate dynamic storage");

} // namespace ppo::examples::bluedefense::defenders
