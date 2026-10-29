#pragma once

#include "blue_team_roles.hpp"
#include "generalized_team_dynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ppo::examples::twovstwo::defenders {

/** @brief Number of teammate formation slots at maximum blue capacity. */
inline constexpr std::size_t kMaximumFormationSlots =
    TeamCapacity::MaximumBlueMobiles - 1U;

/**
 * @brief Episode participation and current survival masks for both teams.
 *
 * Active masks are immutable episode membership. Alive masks may only lose
 * entries during a transition. This distinction prevents destroyed entities
 * from being confused with unused fixed-capacity storage.
 */
struct GeneralizedCombatStateMask final
{
    /** @brief Blue compact episode indices participating in the episode. */
    std::array<bool, TeamCapacity::MaximumBlueMobiles> blueActive{};
    /** @brief Surviving blue participants in compact episode order. */
    std::array<bool, TeamCapacity::MaximumBlueMobiles> blueAlive{};
    /** @brief Red physical indices participating in the episode. */
    std::array<bool, TeamCapacity::MaximumRedMobiles> redActive{};
    /** @brief Surviving red participants. */
    std::array<bool, TeamCapacity::MaximumRedMobiles> redAlive{};
    /** @brief Whether the fixed defended objective remains available. */
    bool objectiveAvailable{true};
};

/**
 * @brief Creates reward masks from the pure generalized blue-team state.
 * @param blue Fixed-capacity blue state and episode participation mask.
 * @param redActive Episode membership mask for red mobiles.
 * @param redAlive Current survival mask for red mobiles.
 * @param objectiveAvailable Current objective survival state.
 * Active generalized-state indices are compacted in increasing order so they
 * match BlueTeamRoles and BlueTeamActionBatch episode ordering.
 * @return Fixed-capacity combat mask without allocation or ECS dependency.
 */
inline GeneralizedCombatStateMask MakeGeneralizedCombatStateMask(
    GeneralizedBlueTeamState const& blue,
    std::array<bool, TeamCapacity::MaximumRedMobiles> const& redActive,
    std::array<bool, TeamCapacity::MaximumRedMobiles> const& redAlive,
    bool const objectiveAvailable) noexcept
{
    GeneralizedCombatStateMask result{};
    result.redActive = redActive;
    result.redAlive = redAlive;
    result.objectiveAvailable = objectiveAvailable;
    std::size_t compactIndex = 0U;
    for (std::size_t physicalIndex = 0U;
         physicalIndex < blue.active.size(); ++physicalIndex)
    {
        if (!blue.active[physicalIndex])
            continue;
        if (compactIndex >= result.blueActive.size())
            break;
        result.blueActive[compactIndex] = true;
        result.blueAlive[compactIndex] =
            blue.mobiles[physicalIndex].available;
        ++compactIndex;
    }
    return result;
}

/** @brief Explainable channels of a generalized cooperative reward. */
enum class GeneralizedRewardChannel : std::size_t
{
    RoleResponsibility,
    ControlEffort,
    BlueSurvival,
    RedNeutralization,
    ThreatContainment,
    ObjectiveSurvival,
    MissionOutcome,
    Timeout,
    Count
};

/** @brief Number of generalized reward channels. */
inline constexpr std::size_t kGeneralizedRewardChannelCount =
    static_cast<std::size_t>(GeneralizedRewardChannel::Count);

/** @brief Fixed reward-channel storage used by training and telemetry. */
using GeneralizedRewardValues =
    std::array<float, kGeneralizedRewardChannelCount>;

/**
 * @brief Weights and clamp used when reducing reward channels to a scalar.
 */
struct GeneralizedRewardSettings final
{
    /** @brief Per-channel multipliers in enum order. */
    GeneralizedRewardValues weights{1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F};
    /** @brief Symmetric scalar reward limit. */
    float clampMagnitude{100.F};
};

/**
 * @brief Dense responsibility signals computed by a future runtime adapter.
 *
 * No red target identity is represented. The leader owns the single patrol
 * signal and every teammate owns the signal of its immutable formation slot.
 */
struct GeneralizedResponsibilitySignals final
{
    /** @brief Leader patrol progress/quality contribution, normally [-1, 1]. */
    float leaderPatrol{};
    /** @brief Team-average reduction of objective danger, in [-1, 1]. */
    float threatContainment{};
    /** @brief Teammate contributions indexed by unique formation slot. */
    std::array<float, kMaximumFormationSlots> formationSlots{};
    /** @brief Non-negative normalized control effort by physical blue index. */
    std::array<float, TeamCapacity::MaximumBlueMobiles> controlEffort{};
};

/**
 * @brief Complete pure input for one cooperative reward transition.
 */
struct GeneralizedRewardTransition final
{
    /** @brief State immediately before physics and resolution. */
    GeneralizedCombatStateMask before{};
    /** @brief State after physics and resolution. */
    GeneralizedCombatStateMask after{};
    /** @brief Role-specific dense shaping signals. */
    GeneralizedResponsibilitySignals responsibilities{};
    /** @brief Sticky generalized mission failure from physics/world rules. */
    bool missionFailed{};
    /** @brief True while a delayed red participant remains scheduled. */
    bool pendingThreats{};
    /** @brief True when the episode exhausted its step budget. */
    bool timeout{};
};

/** @brief Weighted channel decomposition for one trajectory or the team. */
struct GeneralizedRewardBreakdown final
{
    /** @brief Unweighted, normalized channel values. */
    GeneralizedRewardValues values{};

    /**
     * @brief Reduces channels with finite settings.
     * @param settings Runtime weights and symmetric clamp.
     * @return Weighted scalar reward.
     */
    float Total(GeneralizedRewardSettings const& settings) const noexcept
    {
        float total = 0.F;
        for (std::size_t channel = 0U; channel < values.size(); ++channel)
            total += values[channel] * settings.weights[channel];
        return std::clamp(total, -settings.clampMagnitude,
                          settings.clampMagnitude);
    }
};

/**
 * @brief Allocation-free reward result for all blue trajectories and team.
 */
struct GeneralizedCooperativeRewards final
{
    /** @brief True for compact blue indices alive before this transition. */
    std::array<bool, TeamCapacity::MaximumBlueMobiles> active{};
    /** @brief Per-trajectory role and cooperative channels. */
    std::array<GeneralizedRewardBreakdown,
               TeamCapacity::MaximumBlueMobiles> perBlue{};
    /** @brief Mean role/control channels plus shared cooperative channels. */
    GeneralizedRewardBreakdown team{};
};

/** @brief Validation result for cooperative reward aggregation. */
enum class GeneralizedRewardStatus : std::uint8_t
{
    Applied,
    InvalidRoles,
    InvalidMasks,
    InvalidSignals,
    InvalidSettings
};

namespace generalized_reward_detail {

template<std::size_t Capacity>
inline std::size_t Count(std::array<bool, Capacity> const& mask) noexcept
{
    return static_cast<std::size_t>(std::count(mask.begin(), mask.end(), true));
}

template<std::size_t Capacity>
inline bool IsSubset(std::array<bool, Capacity> const& subset,
                     std::array<bool, Capacity> const& superset) noexcept
{
    for (std::size_t index = 0U; index < Capacity; ++index)
        if (subset[index] && !superset[index])
            return false;
    return true;
}

template<std::size_t Capacity>
inline bool Same(std::array<bool, Capacity> const& first,
                 std::array<bool, Capacity> const& second) noexcept
{
    return std::equal(first.begin(), first.end(), second.begin());
}

inline bool ValidMasks(GeneralizedCombatStateMask const& before,
                       GeneralizedCombatStateMask const& after,
                       std::size_t const blueCount) noexcept
{
    std::size_t const redCount = Count(before.redActive);
    if (!Valid(ActiveTeamCounts{blueCount, redCount})
        || Count(before.blueActive) != blueCount
        || !Same(before.blueActive, after.blueActive)
        || !Same(before.redActive, after.redActive)
        || !IsSubset(before.blueAlive, before.blueActive)
        || !IsSubset(after.blueAlive, after.blueActive)
        || !IsSubset(before.redAlive, before.redActive)
        || !IsSubset(after.redAlive, after.redActive)
        || !IsSubset(after.blueAlive, before.blueAlive)
        || (!before.objectiveAvailable && after.objectiveAvailable))
    {
        return false;
    }
    // BlueTeamRoles currently names dense physical indices [0, P). Reject a
    // sparse mask instead of silently associating a role with the wrong body.
    for (std::size_t index = 0U; index < before.blueActive.size(); ++index)
        if (before.blueActive[index] != (index < blueCount))
            return false;
    return true;
}

inline bool ValidSignals(GeneralizedResponsibilitySignals const& signals,
                         std::size_t const blueCount) noexcept
{
    if (!std::isfinite(signals.leaderPatrol)
        || !std::isfinite(signals.threatContainment)
        || std::abs(signals.threatContainment) > 1.F)
        return false;
    for (std::size_t slot = 0U; slot + 1U < blueCount; ++slot)
        if (!std::isfinite(signals.formationSlots[slot]))
            return false;
    for (std::size_t index = 0U; index < blueCount; ++index)
        if (!std::isfinite(signals.controlEffort[index])
            || signals.controlEffort[index] < 0.F)
        {
            return false;
        }
    return true;
}

inline bool ValidSettings(GeneralizedRewardSettings const& settings) noexcept
{
    if (!std::isfinite(settings.clampMagnitude)
        || settings.clampMagnitude <= 0.F)
    {
        return false;
    }
    return std::all_of(settings.weights.begin(), settings.weights.end(),
        [](float const weight) { return std::isfinite(weight); });
}

} // namespace generalized_reward_detail

/**
 * @brief Aggregates normalized role-specific and cooperative rewards.
 *
 * Red neutralization is divided by active N and blue loss by active P. Every
 * active blue receives the same cooperative event channels. Only the leader
 * receives patrol responsibility; each teammate receives only its unique slot
 * responsibility. The team decomposition averages role/control contributions
 * over P, making its scale independent of team size.
 *
 * @param roles Stable leader and formation-slot assignment for the episode.
 * @param transition Pre/post masks and pure shaping signals.
 * @param settings Finite weights and scalar clamp configuration.
 * @param result Output reset transactionally before validation.
 * @return Applied, or the rejected contract category.
 */
inline GeneralizedRewardStatus AggregateGeneralizedCooperativeRewards(
    BlueTeamRoles const& roles,
    GeneralizedRewardTransition const& transition,
    GeneralizedRewardSettings const& settings,
    GeneralizedCooperativeRewards& result) noexcept
{
    result = {};
    if (!roles.IsValid())
        return GeneralizedRewardStatus::InvalidRoles;
    std::size_t const blueCount = roles.ActiveBlueCount();
    if (!generalized_reward_detail::ValidMasks(
            transition.before, transition.after, blueCount))
    {
        return GeneralizedRewardStatus::InvalidMasks;
    }
    if (!generalized_reward_detail::ValidSignals(
            transition.responsibilities, blueCount))
    {
        return GeneralizedRewardStatus::InvalidSignals;
    }
    if (!generalized_reward_detail::ValidSettings(settings))
        return GeneralizedRewardStatus::InvalidSettings;

    std::size_t const redCount = generalized_reward_detail::Count(
        transition.before.redActive);
    std::size_t blueLosses = 0U;
    std::size_t redNeutralized = 0U;
    std::size_t remainingReds = 0U;
    for (std::size_t index = 0U; index < blueCount; ++index)
        blueLosses += transition.before.blueAlive[index]
            && !transition.after.blueAlive[index] ? 1U : 0U;
    for (std::size_t index = 0U;
         index < TeamCapacity::MaximumRedMobiles; ++index)
    {
        redNeutralized += transition.before.redAlive[index]
            && !transition.after.redAlive[index] ? 1U : 0U;
        remainingReds += transition.after.redAlive[index] ? 1U : 0U;
    }

    float const blueSurvival = -static_cast<float>(blueLosses)
        / static_cast<float>(blueCount);
    float const redNeutralization = redCount > 0U
        ? static_cast<float>(redNeutralized) / static_cast<float>(redCount)
        : 0.F;
    // Sparse loss avoids a positive per-step living bonus that would make
    // reward depend on episode duration.
    float const objectiveSurvival = transition.before.objectiveAvailable
        && !transition.after.objectiveAvailable ? -1.F : 0.F;
    float outcome = 0.F;
    if (transition.missionFailed || !transition.after.objectiveAvailable)
        outcome = -1.F;
    else if (redCount > 0U && remainingReds == 0U
             && !transition.pendingThreats)
        outcome = 1.F;
    float const timeout = transition.timeout
        && (remainingReds > 0U || transition.pendingThreats) ? -1.F : 0.F;

    auto const shared = [&](GeneralizedRewardBreakdown& breakdown) {
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::BlueSurvival)] = blueSurvival;
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::RedNeutralization)] = redNeutralization;
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::ThreatContainment)] =
                transition.responsibilities.threatContainment;
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::ObjectiveSurvival)] = objectiveSurvival;
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::MissionOutcome)] = outcome;
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::Timeout)] = timeout;
    };
    shared(result.team);

    float responsibilitySum = 0.F;
    float effortSum = 0.F;
    for (std::size_t physicalIndex = 0U;
         physicalIndex < blueCount; ++physicalIndex)
    {
        if (!transition.before.blueAlive[physicalIndex])
            continue;
        result.active[physicalIndex] = true;
        GeneralizedRewardBreakdown& breakdown = result.perBlue[physicalIndex];
        shared(breakdown);
        BlueRoleAssignment const assignment = *roles.Assignment(physicalIndex);
        float responsibility = transition.responsibilities.leaderPatrol;
        if (assignment.role == BlueRole::Teammate)
            responsibility = transition.responsibilities.formationSlots[
                assignment.formationSlot];
        responsibility = std::clamp(responsibility, -1.F, 1.F);
        float const effort = std::clamp(
            transition.responsibilities.controlEffort[physicalIndex], 0.F, 1.F);
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::RoleResponsibility)] = responsibility;
        breakdown.values[static_cast<std::size_t>(
            GeneralizedRewardChannel::ControlEffort)] = -effort;
        responsibilitySum += responsibility;
        effortSum += effort;
    }
    result.team.values[static_cast<std::size_t>(
        GeneralizedRewardChannel::RoleResponsibility)] =
            responsibilitySum / static_cast<float>(blueCount);
    result.team.values[static_cast<std::size_t>(
        GeneralizedRewardChannel::ControlEffort)] =
            -effortSum / static_cast<float>(blueCount);
    return GeneralizedRewardStatus::Applied;
}

static_assert(std::is_trivially_copyable_v<GeneralizedCombatStateMask>,
    "Generalized reward masks must remain fixed-capacity values");
static_assert(std::is_trivially_copyable_v<GeneralizedCooperativeRewards>,
    "Generalized cooperative rewards must not own dynamic storage");

} // namespace ppo::examples::twovstwo::defenders
