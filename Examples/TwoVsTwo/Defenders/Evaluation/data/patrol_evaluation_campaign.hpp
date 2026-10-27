#pragma once

#include "patrol_core.hpp"
#include "patrol_evaluation_settings.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ppo::examples::twovstwo::defenders {

/**
 * @brief Reproducible initial conditions for one live ECS evaluation episode.
 */
struct PatrolEvaluationCase final
{
    std::uint32_t seed{};
    int side{1};
    float arclength{};
    bool secondAttackerDestroyed{};
};

/**
 * @brief Runs and scores one deterministic blue-team evaluation campaign.
 *
 * Physics and red RLMission inference stay in the real ECS. This class owns
 * only the fixed case schedule and metric aggregation, keeping evaluation
 * policy separate from the MAPPO training loop.
 */
class PatrolEvaluationCampaign final
{
public:
    /**
     * @brief Creates an idle campaign.
     * @param settings Valid deterministic evaluation settings.
     */
    explicit PatrolEvaluationCampaign(PatrolEvaluationSettings settings)
        : settings_(settings)
    {
        if (!Valid(settings_))
            throw std::invalid_argument("Invalid blue patrol evaluation settings");
    }

    /**
     * @brief Starts a fresh campaign while retaining the historical best.
     * @param combatMode True when red RLMission attackers are present.
     */
    void Start(bool combatMode) noexcept
    {
        combatMode_ = combatMode;
        inProgress_ = true;
        improved_ = false;
        episodeIndex_ = 0U;
        episodeSteps_ = 0U;
        testsFinished_ = 0U;
        timeoutTests_ = 0U;
        successes_ = 0U;
        stepsExecuted_ = 0U;
        survivingGuardFractionSum_ = 0.F;
        objectiveSurvivals_ = 0U;
        trackingGood_ = 0U;
        trackingTotal_ = 0U;
        episodeTrackingGood_ = 0U;
        episodeTrackingTotal_ = 0U;
        episodeProgressMeters_.fill(0.F);
        previousProgress_.fill(0.F);
        initialThreats_ = 0U;
        episodeStarted_ = false;
        startedAt_ = Clock::now();
        elapsedSeconds_ = 0.F;
    }

    /**
     * @brief Restores a score promoted by an earlier process.
     * @param score Finite score in the range [0, 100].
     */
    void RestoreBestScore(float score) noexcept
    {
        if (std::isfinite(score) && score >= 0.F && score <= 100.F)
            bestScore_ = score;
    }

    /** @brief Cancels an incomplete campaign without publishing its metrics. */
    void Cancel() noexcept
    {
        inProgress_ = false;
        episodeStarted_ = false;
        improved_ = false;
    }

    /**
     * @brief Returns the deterministic case currently awaiting initialization.
     * @param parameters Patrol geometry used to convert phase to arclength.
     * @return Fixed side, phase, seed and 2v1/2v2 setup.
     */
    PatrolEvaluationCase CurrentCase(Parameters const& parameters) const
    {
        if (!inProgress_ || episodeIndex_ >= settings_.episodes)
            throw std::logic_error("No active blue patrol evaluation case");
        PatrolEvaluationCase result{};
        result.seed = settings_.seed + static_cast<std::uint32_t>(episodeIndex_);
        result.side = episodeIndex_ % 2U == 0U ? 1 : -1;
        result.arclength = CircuitLength(parameters)
            * static_cast<float>((episodeIndex_ / 2U) % 4U) / 4.F;
        // Consecutive mirrored cases share the same threat count. The eight
        // default cases cover both 2v2 and 2v1 at two phases each.
        result.secondAttackerDestroyed = ((episodeIndex_ / 2U) % 2U) != 0U;
        return result;
    }

    /**
     * @brief Records the initialized state of the current episode.
     * @param initial State after every ECS entity has been reset.
     */
    void BeginEpisode(State const& initial)
    {
        if (!inProgress_)
            throw std::logic_error("Blue patrol evaluation is not active");
        initialThreats_ = static_cast<std::size_t>(std::count_if(
            initial.attackers.begin(), initial.attackers.end(),
            [](Mobile const& attacker) { return attacker.available; }));
        episodeSteps_ = 0U;
        episodeTrackingGood_ = 0U;
        episodeTrackingTotal_ = 0U;
        episodeProgressMeters_.fill(0.F);
        previousProgress_ = initial.patrolProgress;
        episodeStarted_ = true;
    }

    /**
     * @brief Adds one post-physics state to live tracking statistics.
     * @param parameters Patrol geometry and target speed.
     * @param state Latest resolved ECS state.
     */
    void ObserveStep(Parameters const& parameters, State const& state)
    {
        if (!inProgress_ || !episodeStarted_)
            throw std::logic_error("Blue patrol evaluation episode was not initialized");
        ++episodeSteps_;
        ++stepsExecuted_;
        for (std::size_t role = 0; role < state.guards.size(); ++role)
        {
            Mobile const& guard = state.guards[role];
            PatrolProjection const projection = ProjectOnPatrolCurve(
                parameters.geometry, role, guard.north, guard.east);
            float const error = projection.distance
                / std::hypot(parameters.halfNorth, parameters.halfEast);
            float const heading = std::cos(
                guard.heading - projection.pose.heading);
            ++episodeTrackingTotal_;
            if (guard.available && std::isfinite(error)
                && error <= settings_.trackingTolerance
                && heading >= settings_.minimumHeadingCosine
                && guard.speed >= settings_.minimumSpeedFraction
                                   * parameters.cruiseSpeed)
                ++episodeTrackingGood_;

            float const delta = std::remainder(
                state.patrolProgress[role] - previousProgress_[role],
                CircuitLength(parameters));
            float const maximumPhysicalProgress =
                2.F * parameters.maxSpeed * parameters.dt + 1.F;
            if (std::abs(delta) <= maximumPhysicalProgress)
                episodeProgressMeters_[role] += std::max(delta, 0.F);
            previousProgress_[role] = state.patrolProgress[role];
        }
    }

    /**
     * @brief Completes one case and, if needed, the complete campaign.
     * @param parameters Patrol geometry used by the patrol-only distance test.
     * @param finalState Last resolved ECS state.
     * @param timeout True when the case exhausted its configured step budget.
     * @return True when another deterministic case remains.
     */
    bool FinishEpisode(Parameters const& parameters, State const& finalState,
                       bool timeout)
    {
        if (!inProgress_ || !episodeStarted_)
            throw std::logic_error("Cannot finish an idle blue patrol evaluation");
        std::size_t const guardsAlive = static_cast<std::size_t>(std::count_if(
            finalState.guards.begin(), finalState.guards.end(),
            [](Mobile const& guard) { return guard.available; }));
        bool const threatsDefeated = initialThreats_ > 0U
            && !HasActiveThreat(finalState);
        float const episodeTracking = episodeTrackingTotal_ > 0U
            ? static_cast<float>(episodeTrackingGood_)
                / static_cast<float>(episodeTrackingTotal_)
            : 0.F;
        float const travelled = 0.5F
            * (episodeProgressMeters_[0] + episodeProgressMeters_[1]);
        bool const patrolDistanceReached = travelled >=
            settings_.minimumCircuits * CircuitLength(parameters);
        bool const safe = finalState.objectiveAvailable && !finalState.failed;
        bool const success = combatMode_
            ? safe && threatsDefeated
            : safe && timeout && patrolDistanceReached
                && episodeTracking >= settings_.minimumTrackingFraction;

        successes_ += success ? 1U : 0U;
        timeoutTests_ += timeout ? 1U : 0U;
        objectiveSurvivals_ += finalState.objectiveAvailable ? 1U : 0U;
        survivingGuardFractionSum_ += 0.5F * static_cast<float>(guardsAlive);
        trackingGood_ += episodeTrackingGood_;
        trackingTotal_ += episodeTrackingTotal_;
        ++testsFinished_;
        ++episodeIndex_;
        episodeStarted_ = false;

        if (episodeIndex_ < settings_.episodes)
            return true;

        FinishCampaign();
        return false;
    }

    /** @brief Returns whether deterministic evaluation currently owns actions. */
    bool InProgress() const noexcept { return inProgress_; }
    /** @brief Returns whether the most recent score beat the historical best. */
    bool Improved() const noexcept { return improved_; }
    /** @brief Returns whether at least one campaign has completed. */
    bool HasLatest() const noexcept { return hasLatest_; }
    /** @brief Returns the configured number of cases. */
    std::size_t TestCount() const noexcept { return settings_.episodes; }
    /** @brief Returns completed cases in the active/latest campaign. */
    std::size_t TestsFinished() const noexcept { return testsFinished_; }
    /** @brief Returns time-limited cases in the active/latest campaign. */
    std::size_t TimeoutTests() const noexcept { return timeoutTests_; }
    /** @brief Returns resolved ECS steps in the active/latest campaign. */
    std::uint64_t StepsExecuted() const noexcept { return stepsExecuted_; }
    /** @brief Returns the latest complete score in [0, 100]. */
    float LatestScore() const noexcept { return latestScore_; }
    /** @brief Returns the promoted historical score in [0, 100]. */
    float BestScore() const noexcept
    {
        return std::isfinite(bestScore_) ? bestScore_ : 0.F;
    }
    /** @brief Returns the latest complete mission success ratio. */
    float SuccessRate() const noexcept { return successRate_; }
    /** @brief Returns the latest mean blue-guard survival ratio. */
    float SurvivalRate() const noexcept { return survivalRate_; }
    /** @brief Returns the latest timeout ratio. */
    float TimeoutRate() const noexcept { return timeoutRate_; }
    /** @brief Returns the number of completed campaigns. */
    std::uint64_t CompletedCampaigns() const noexcept
    {
        return completedCampaigns_;
    }

    /**
     * @brief Returns campaign progress, including the current episode.
     * @return Ratio in [0, 1].
     */
    float Progress() const noexcept
    {
        if (!inProgress_)
            return hasLatest_ ? 1.F : 0.F;
        float const partial = static_cast<float>(std::min(
            episodeSteps_, settings_.maxSteps))
            / static_cast<float>(settings_.maxSteps);
        return std::clamp((static_cast<float>(testsFinished_) + partial)
            / static_cast<float>(settings_.episodes), 0.F, 1.F);
    }

    /** @brief Returns elapsed wall time for the active/latest campaign. */
    float ElapsedSeconds() const noexcept
    {
        if (!inProgress_)
            return elapsedSeconds_;
        return std::chrono::duration<float>(Clock::now() - startedAt_).count();
    }

    /** @brief Returns measured live evaluation throughput. */
    float StepsPerSecond() const noexcept
    {
        float const seconds = ElapsedSeconds();
        return seconds > 1.e-6F
            ? static_cast<float>(stepsExecuted_) / seconds : 0.F;
    }

private:
    using Clock = std::chrono::steady_clock;

    void FinishCampaign() noexcept
    {
        float const count = static_cast<float>(settings_.episodes);
        successRate_ = static_cast<float>(successes_) / count;
        survivalRate_ = survivingGuardFractionSum_ / count;
        objectiveSurvivalRate_ = static_cast<float>(objectiveSurvivals_) / count;
        timeoutRate_ = static_cast<float>(timeoutTests_) / count;
        trackingRate_ = trackingTotal_ > 0U
            ? static_cast<float>(trackingGood_) / static_cast<float>(trackingTotal_)
            : 0.F;
        // Mission completion and objective protection dominate patrol quality;
        // survival and tracking break ties between otherwise useful policies.
        latestScore_ = 60.F * successRate_
            + 20.F * objectiveSurvivalRate_
            + 10.F * survivalRate_
            + 10.F * trackingRate_;
        improved_ = !std::isfinite(bestScore_) || latestScore_ > bestScore_;
        if (improved_)
            bestScore_ = latestScore_;
        hasLatest_ = true;
        ++completedCampaigns_;
        elapsedSeconds_ = std::chrono::duration<float>(
            Clock::now() - startedAt_).count();
        inProgress_ = false;
    }

    PatrolEvaluationSettings settings_{};
    Clock::time_point startedAt_{};
    std::size_t episodeIndex_{};
    std::size_t episodeSteps_{};
    std::size_t testsFinished_{};
    std::size_t timeoutTests_{};
    std::size_t successes_{};
    std::size_t initialThreats_{};
    std::size_t objectiveSurvivals_{};
    std::size_t episodeTrackingGood_{};
    std::size_t episodeTrackingTotal_{};
    std::size_t trackingGood_{};
    std::size_t trackingTotal_{};
    std::uint64_t stepsExecuted_{};
    std::uint64_t completedCampaigns_{};
    std::array<float, 2> episodeProgressMeters_{};
    std::array<float, 2> previousProgress_{};
    float survivingGuardFractionSum_{};
    float latestScore_{};
    float bestScore_{std::numeric_limits<float>::quiet_NaN()};
    float successRate_{};
    float survivalRate_{};
    float objectiveSurvivalRate_{};
    float trackingRate_{};
    float timeoutRate_{};
    float elapsedSeconds_{};
    bool combatMode_{};
    bool inProgress_{};
    bool episodeStarted_{};
    bool improved_{};
    bool hasLatest_{};
};

} // namespace ppo::examples::twovstwo::defenders
