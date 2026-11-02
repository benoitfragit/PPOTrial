/**
 * @file generalized_episode_outcome.hpp
 * @brief Shared episode termination contract for the runtime and regressions.
 */
#pragma once

#include "generalized_patrol_waypoint_sequencer.hpp"

#include <algorithm>
#include <cstddef>

namespace ppo::examples::bluedefense::defenders::generalized_episode_detail {

/** @brief Resolved mission flags passed unchanged to reward and PPO/GAE. */
struct EpisodeOutcome final
{
    bool noThreatRemains{};
    bool patrolCompleted{};
    bool missionFailed{};
    bool terminated{};
    bool truncated{};

    /** @brief Returns the same terminal classification for training and evaluation. */
    char const* DiagnosticCause() const noexcept
    {
        return missionFailed ? "failure"
            : terminated ? "success" : truncated ? "timeout" : "ongoing";
    }
};

/**
 * @brief Resolves success, failure and horizon precedence after one physics step.
 * @param state Valid resolved snapshot with immutable episode membership.
 * @param pendingThreats Whether a delayed Red still awaits activation.
 * @param sequencer Leader-owned waypoint progress for this episode.
 * @param lapCompleted Result of the independent physical lap validator.
 * @param settings No-threat horizon; ignored when any Red is configured.
 * @param dt Positive simulated duration of one step, in seconds.
 * @param episodeStep Number of resolved steps since the episode reset.
 * @param maximumSteps Existing training or evaluation horizon.
 * @return Terminal events take precedence over a coincident time-limit truncation.
 */
inline EpisodeOutcome EvaluateOutcome(
    GeneralizedRuntimeState const& state, bool const pendingThreats,
    GeneralizedPatrolWaypointSequencer const& sequencer,
    bool const lapCompleted, PatrolNoThreatSettings const& settings,
    float const dt, std::size_t const episodeStep,
    std::size_t const maximumSteps) noexcept
{
    bool const anyLivingBlue = std::any_of(state.blues.begin(),
        state.blues.begin() + state.counts.blueMobiles,
        [](Mobile const& mobile) { return mobile.available; });
    bool const anyLivingRed = std::any_of(state.reds.begin(),
        state.reds.begin() + state.counts.redMobiles,
        [](Mobile const& mobile) { return mobile.available; });
    EpisodeOutcome result{};
    result.noThreatRemains = state.counts.redMobiles > 0U
        && !anyLivingRed && anyLivingBlue && state.objectiveAvailable
        && !pendingThreats;
    result.patrolCompleted = state.counts.redMobiles == 0U
        && !pendingThreats && !state.failed && state.objectiveAvailable
        && sequencer.Completed() && sequencer.CircuitValid() && lapCompleted;
    // Preserve combat victory precedence over a simultaneous boundary exit.
    result.missionFailed = (state.failed && !result.noThreatRemains)
        || !state.objectiveAvailable || GeneralizedNoThreatTeamLost(state);
    result.terminated = result.missionFailed || result.noThreatRemains
        || result.patrolCompleted;
    std::size_t const limit = state.counts.redMobiles == 0U
        ? NoThreatEpisodeLimitSteps(settings, dt, maximumSteps) : maximumSteps;
    result.truncated = !result.terminated && episodeStep >= limit;
    return result;
}

} // namespace ppo::examples::bluedefense::defenders::generalized_episode_detail
