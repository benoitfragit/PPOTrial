#pragma once

#include "generalized_evaluation_metrics.hpp"
#include "patrol_core.hpp"
#include "patrol_evaluation_settings.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ppo::examples::bluedefense::defenders {

/** @brief Reproducible reset parameters for one generalized evaluation case. */
struct GeneralizedEvaluationCase final
{
    /** @brief Seed applied before the complete ECS episode reset. */
    std::uint32_t seed{};
    /** @brief Mirrored patrol direction, always -1 or +1. */
    int side{1};
    /** @brief Leader start arclength on the sampled racetrack, in metres. */
    float arclength{};
};

/**
 * @brief Owns one deterministic P-versus-N evaluation campaign.
 *
 * The live ECS keeps responsibility for physics, red RLMission controllers,
 * restricted zones and termination. This fixed-capacity object only provides
 * a reproducible reset schedule and aggregates the normalized episode samples.
 */
class GeneralizedEvaluationCampaign final
{
public:
    /**
     * @brief Creates an idle campaign.
     * @param settings Valid evaluation cadence and finite episode limits.
     */
    explicit GeneralizedEvaluationCampaign(PatrolEvaluationSettings settings)
        : settings_(settings)
    {
        if (!Valid(settings_))
            throw std::invalid_argument(
                "Invalid generalized patrol evaluation settings");
    }

    /** @brief Starts a fresh campaign while retaining the promoted best. */
    void Start() noexcept
    {
        accumulator_.Reset();
        inProgress_ = true;
        improved_ = false;
        episodeIndex_ = 0U;
        episodeSteps_ = 0U;
        stepsExecuted_ = 0U;
        startedAt_ = Clock::now();
        elapsedSeconds_ = 0.F;
    }

    /** @brief Cancels a partial campaign without publishing partial metrics. */
    void Cancel() noexcept
    {
        inProgress_ = false;
        improved_ = false;
    }

    /**
     * @brief Returns the current reproducible reset case.
     * @param parameters Patrol geometry used for phase-to-distance conversion.
     * @return Seed, mirror side and quarter-circuit phase.
     */
    GeneralizedEvaluationCase CurrentCase(Parameters const& parameters) const
    {
        if (!inProgress_ || episodeIndex_ >= settings_.episodes)
            throw std::logic_error(
                "No active generalized patrol evaluation case");
        GeneralizedEvaluationCase result{};
        result.seed = settings_.seed
            + static_cast<std::uint32_t>(episodeIndex_);
        result.side = episodeIndex_ % 2U == 0U ? 1 : -1;
        result.arclength = CircuitLength(parameters)
            * static_cast<float>((episodeIndex_ / 2U) % 4U) / 4.F;
        return result;
    }

    /** @brief Records one resolved live-ECS step for progress diagnostics. */
    void ObserveStep()
    {
        if (!inProgress_)
            throw std::logic_error(
                "Generalized patrol evaluation is not active");
        ++episodeSteps_;
        ++stepsExecuted_;
    }

    /**
     * @brief Completes the current case and publishes a final campaign if full.
     * @param roles Stable leader and formation-slot assignment for this reset.
     * @param sample Normalized final P/N episode sample.
     * @return True when another deterministic reset case remains.
     */
    bool FinishEpisode(BlueTeamRoles const& roles,
                       GeneralizedEvaluationSample const& sample)
    {
        if (!inProgress_)
            throw std::logic_error(
                "Cannot finish an idle generalized patrol evaluation");
        if (accumulator_.Record(roles, sample)
            != GeneralizedEvaluationStatus::Recorded)
        {
            throw std::invalid_argument(
                "Invalid generalized patrol evaluation sample");
        }
        ++episodeIndex_;
        episodeSteps_ = 0U;
        if (episodeIndex_ < settings_.episodes)
            return true;

        latest_ = accumulator_.Metrics();
        improved_ = !std::isfinite(bestScore_)
            || latest_.score > bestScore_;
        if (improved_)
            bestScore_ = latest_.score;
        hasLatest_ = true;
        ++completedCampaigns_;
        elapsedSeconds_ = std::chrono::duration<float>(
            Clock::now() - startedAt_).count();
        inProgress_ = false;
        return false;
    }

    /**
     * @brief Restores the score paired with a previously promoted v8 actor.
     * @param score Finite score in [0, 100].
     */
    void RestoreBestScore(float const score) noexcept
    {
        if (std::isfinite(score) && score >= 0.F && score <= 100.F)
            bestScore_ = score;
    }

    /** @brief Returns whether evaluation currently owns the blue actions. */
    bool InProgress() const noexcept { return inProgress_; }
    /** @brief Returns whether the latest complete campaign beat the best. */
    bool Improved() const noexcept { return improved_; }
    /** @brief Returns whether a complete campaign is available. */
    bool HasLatest() const noexcept { return hasLatest_; }
    /** @brief Returns the configured deterministic case count. */
    std::size_t TestCount() const noexcept { return settings_.episodes; }
    /** @brief Returns completed cases in the active or latest campaign. */
    std::size_t TestsFinished() const noexcept
    {
        return inProgress_ ? episodeIndex_ : latest_.episodes;
    }
    /** @brief Returns all live-ECS steps consumed by the active/latest run. */
    std::uint64_t StepsExecuted() const noexcept { return stepsExecuted_; }
    /** @brief Returns the number of fully published campaigns. */
    std::uint64_t CompletedCampaigns() const noexcept
    {
        return completedCampaigns_;
    }
    /** @brief Returns the latest normalized P/N metrics. */
    GeneralizedEvaluationMetrics Metrics() const noexcept { return latest_; }
    /** @brief Returns the promoted historical score in [0, 100]. */
    float BestScore() const noexcept
    {
        return std::isfinite(bestScore_) ? bestScore_ : 0.F;
    }

    /** @brief Returns active/latest progress in [0, 1]. */
    float Progress() const noexcept
    {
        if (!inProgress_)
            return hasLatest_ ? 1.F : 0.F;
        float const partial = static_cast<float>(std::min(
            episodeSteps_, settings_.maxSteps))
            / static_cast<float>(settings_.maxSteps);
        return std::clamp((static_cast<float>(episodeIndex_) + partial)
            / static_cast<float>(settings_.episodes), 0.F, 1.F);
    }

    /** @brief Returns active/latest wall-clock duration in seconds. */
    float ElapsedSeconds() const noexcept
    {
        if (!inProgress_)
            return elapsedSeconds_;
        return std::chrono::duration<float>(Clock::now() - startedAt_).count();
    }

    /** @brief Returns active/latest evaluation throughput. */
    float StepsPerSecond() const noexcept
    {
        float const seconds = ElapsedSeconds();
        return seconds > 1.e-6F
            ? static_cast<float>(stepsExecuted_) / seconds : 0.F;
    }

private:
    using Clock = std::chrono::steady_clock;

    PatrolEvaluationSettings settings_{};
    GeneralizedEvaluationAccumulator accumulator_{};
    GeneralizedEvaluationMetrics latest_{};
    Clock::time_point startedAt_{};
    std::size_t episodeIndex_{};
    std::size_t episodeSteps_{};
    std::uint64_t stepsExecuted_{};
    std::uint64_t completedCampaigns_{};
    float bestScore_{std::numeric_limits<float>::quiet_NaN()};
    float elapsedSeconds_{};
    bool inProgress_{};
    bool improved_{};
    bool hasLatest_{};
};

} // namespace ppo::examples::bluedefense::defenders
