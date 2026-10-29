#pragma once

#include "patrol_core.hpp"
#include "team_capacity.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ppo::examples::bluedefense::defenders {

/** @brief Maximum number of unordered blue-mobile pairs. */
inline constexpr std::size_t kMaximumBlueCollisionPairs =
    TeamCapacity::MaximumBlueMobiles
    * (TeamCapacity::MaximumBlueMobiles - 1U) / 2U;

/**
 * @brief Fixed-capacity state of every blue mobile for one synchronized step.
 *
 * The active mask is authoritative and may be sparse. Active mobiles retain
 * their physical array index for the entire episode, including after their
 * availability mask is cleared on destruction.
 */
struct GeneralizedBlueTeamState final
{
    /** @brief Physical blue-mobile storage, indexed by stable episode index. */
    std::array<Mobile, TeamCapacity::MaximumBlueMobiles> mobiles{};
    /** @brief True for each physical index participating in the episode. */
    std::array<bool, TeamCapacity::MaximumBlueMobiles> active{};
    /** @brief Number of true entries in @ref active. */
    std::size_t activeCount{};
    /** @brief Shared world step, incremented once per synchronized update. */
    std::size_t step{};
    /** @brief True when at least one pair collided during the latest step. */
    bool collided{};
    /** @brief Sticky invalid-kinematics, boundary, or collision outcome. */
    bool failed{};
};

/**
 * @brief Compact actions for all active blue mobiles.
 *
 * Actions are ordered by increasing active physical index. `actionCount` must
 * equal the state's active count; inactive capacity never consumes an action.
 */
struct BlueTeamActionBatch final
{
    /** @brief Compact per-mobile commands using the legacy action semantics. */
    std::array<Action, TeamCapacity::MaximumBlueMobiles> actions{};
    /** @brief Number of valid commands in @ref actions. */
    std::size_t actionCount{};
};

/** @brief One applied command and its matching physical state transition. */
struct BlueMobileTransition final
{
    /** @brief Stable physical index in the generalized team state. */
    std::size_t physicalIndex{TeamCapacity::MaximumBlueMobiles};
    /** @brief Mobile state before the synchronized step. */
    Mobile before{};
    /** @brief Command applied to this mobile. */
    Action action{};
    /** @brief Mobile state after the synchronized step. */
    Mobile after{};
};

/** @brief Swept collision information for one unordered blue pair. */
struct BlueCollisionPair final
{
    /** @brief First physical index, always lower than @ref second. */
    std::size_t first{};
    /** @brief Second physical index. */
    std::size_t second{};
    /** @brief Minimum centre-to-centre distance over the complete step. */
    float minimumSeparation{};
};

/** @brief Allocation-free list of every collision during one team step. */
struct BlueCollisionReport final
{
    /** @brief Colliding pairs in deterministic lexicographic order. */
    std::array<BlueCollisionPair, kMaximumBlueCollisionPairs> pairs{};
    /** @brief Number of valid entries in @ref pairs. */
    std::size_t collisionCount{};
};

/** @brief Result of one valid synchronized generalized-team update. */
struct GeneralizedBlueTeamStep final
{
    /** @brief Team state after all active commands were applied. */
    GeneralizedBlueTeamState next{};
    /** @brief Exactly one compact transition per active blue mobile. */
    std::array<BlueMobileTransition,
               TeamCapacity::MaximumBlueMobiles> transitions{};
    /** @brief Number of valid transitions; equal to the input action count. */
    std::size_t transitionCount{};
    /** @brief Every swept collision found among active blue pairs. */
    BlueCollisionReport collisions{};
};

/** @brief Validation outcome of the generalized team-step boundary. */
enum class GeneralizedTeamStepStatus : std::uint8_t
{
    Applied,
    InvalidParameters,
    InvalidState,
    InvalidActionCount
};

/**
 * @brief Checks the mask, count, and participant kinematics of a team state.
 * @param state Fixed-capacity state to validate.
 * @return True when the count is in `[1, 9]`, matches the immutable episode
 * mask, and every participant has finite, non-negative kinematics. Destroyed
 * participants remain active with `available == false`.
 */
inline bool Valid(GeneralizedBlueTeamState const& state) noexcept
{
    if (!Valid(ActiveTeamCounts{state.activeCount, 0U}))
        return false;

    std::size_t observedCount = 0U;
    for (std::size_t physicalIndex = 0U;
         physicalIndex < TeamCapacity::MaximumBlueMobiles; ++physicalIndex)
    {
        if (!state.active[physicalIndex])
            continue;

        ++observedCount;
        Mobile const& mobile = state.mobiles[physicalIndex];
        if (!std::isfinite(mobile.north) || !std::isfinite(mobile.east)
            || !std::isfinite(mobile.heading)
            || !std::isfinite(mobile.speed)
            || !std::isfinite(mobile.turnRate) || mobile.speed < 0.F)
        {
            return false;
        }
    }
    return observedCount == state.activeCount;
}

/**
 * @brief Checks the subset of legacy parameters used by team dynamics.
 * @param parameters Legacy patrol parameters supplying SNA motion limits.
 * @return True when time, motion, collision, and boundary limits are valid.
 */
inline bool ValidGeneralizedTeamDynamicsParameters(
    Parameters const& parameters) noexcept
{
    return std::isfinite(parameters.dt) && parameters.dt > 0.F
        && std::isfinite(parameters.maxSpeed) && parameters.maxSpeed >= 0.F
        && std::isfinite(parameters.maxAcceleration)
        && parameters.maxAcceleration >= 0.F
        && std::isfinite(parameters.maxRotationRate)
        && parameters.maxRotationRate >= 0.F
        && std::isfinite(parameters.collisionDistance)
        && parameters.collisionDistance > 0.F
        && std::isfinite(parameters.halfNorth) && parameters.halfNorth > 0.F
        && std::isfinite(parameters.halfEast) && parameters.halfEast > 0.F
        && std::isfinite(parameters.hardBoundaryScale)
        && parameters.hardBoundaryScale > 0.F;
}

/**
 * @brief Computes the closest distance between two linearly swept mobiles.
 * @param firstBefore First mobile at the start of the step.
 * @param firstAfter First mobile at the end of the step.
 * @param secondBefore Second mobile at the start of the step.
 * @param secondAfter Second mobile at the end of the step.
 * @return Minimum simultaneous centre distance for normalized time `[0, 1]`.
 */
inline float MinimumSweptBlueSeparation(
    Mobile const& firstBefore, Mobile const& firstAfter,
    Mobile const& secondBefore, Mobile const& secondAfter) noexcept
{
    float const relativeNorth = firstBefore.north - secondBefore.north;
    float const relativeEast = firstBefore.east - secondBefore.east;
    float const relativeTravelNorth =
        (firstAfter.north - secondAfter.north) - relativeNorth;
    float const relativeTravelEast =
        (firstAfter.east - secondAfter.east) - relativeEast;
    float const relativeTravelSquared =
        relativeTravelNorth * relativeTravelNorth
        + relativeTravelEast * relativeTravelEast;
    float const closestTime = relativeTravelSquared > 1.0e-8F
        ? Clamp(-(relativeNorth * relativeTravelNorth
                  + relativeEast * relativeTravelEast)
                    / relativeTravelSquared,
                0.F, 1.F)
        : 0.F;
    return std::hypot(relativeNorth + closestTime * relativeTravelNorth,
                      relativeEast + closestTime * relativeTravelEast);
}

/**
 * @brief Detects swept collisions across every living active blue pair.
 * @param before Valid team state before movement.
 * @param after State after movement with the same active mask.
 * @param collisionDistance Inclusive collision radius in metres.
 * @return Allocation-free deterministic report containing all hits.
 */
inline BlueCollisionReport DetectSweptBlueCollisions(
    GeneralizedBlueTeamState const& before,
    GeneralizedBlueTeamState const& after,
    float const collisionDistance) noexcept
{
    BlueCollisionReport report{};
    for (std::size_t first = 0U;
         first < TeamCapacity::MaximumBlueMobiles; ++first)
    {
        if (!before.active[first] || !after.active[first]
            || !before.mobiles[first].available
            || !after.mobiles[first].available)
            continue;

        for (std::size_t second = first + 1U;
             second < TeamCapacity::MaximumBlueMobiles; ++second)
        {
            if (!before.active[second] || !after.active[second]
                || !before.mobiles[second].available
                || !after.mobiles[second].available)
                continue;

            float const separation = MinimumSweptBlueSeparation(
                before.mobiles[first], after.mobiles[first],
                before.mobiles[second], after.mobiles[second]);
            if (separation <= collisionDistance)
            {
                report.pairs[report.collisionCount++] =
                    {first, second, separation};
            }
        }
    }
    return report;
}

/**
 * @brief Applies exactly one action per active blue in one synchronized step.
 *
 * The function is transactional: on an invalid domain boundary, @p result is
 * reset and no partial transition is exposed. Critical loops use only fixed
 * storage and perform no allocation. Destroyed participants still consume
 * their stable compact action/transition slot, but their physical state is
 * preserved and they do not participate in motion, collision, or boundary
 * failure checks.
 *
 * @param parameters Legacy SNA-compatible motion and collision parameters.
 * @param previous Valid fixed-capacity team state.
 * @param commands Compact actions ordered by increasing active physical index.
 * @param result Output state, transitions, and all swept collision pairs.
 * @return Applied on success, otherwise the precise rejected contract.
 */
inline GeneralizedTeamStepStatus AdvanceGeneralizedBlueTeam(
    Parameters const& parameters,
    GeneralizedBlueTeamState const& previous,
    BlueTeamActionBatch const& commands,
    GeneralizedBlueTeamStep& result) noexcept
{
    result = {};
    if (!ValidGeneralizedTeamDynamicsParameters(parameters))
        return GeneralizedTeamStepStatus::InvalidParameters;
    if (!Valid(previous))
        return GeneralizedTeamStepStatus::InvalidState;
    if (commands.actionCount != previous.activeCount)
        return GeneralizedTeamStepStatus::InvalidActionCount;

    result.next = previous;
    std::size_t commandIndex = 0U;
    for (std::size_t physicalIndex = 0U;
         physicalIndex < TeamCapacity::MaximumBlueMobiles; ++physicalIndex)
    {
        if (!previous.active[physicalIndex])
            continue;

        Action const& action = commands.actions[commandIndex];
        Mobile const before = previous.mobiles[physicalIndex];
        Mobile const after = before.available
            ? Advance(parameters, before, action)
            : before;
        result.next.mobiles[physicalIndex] = after;
        result.transitions[commandIndex] =
            {physicalIndex, before, action, after};
        ++commandIndex;
    }
    result.transitionCount = commandIndex;
    ++result.next.step;
    result.collisions = DetectSweptBlueCollisions(
        previous, result.next, parameters.collisionDistance);
    result.next.collided = result.collisions.collisionCount != 0U;
    result.next.failed |= result.next.collided;

    for (std::size_t physicalIndex = 0U;
         physicalIndex < TeamCapacity::MaximumBlueMobiles; ++physicalIndex)
    {
        if (!result.next.active[physicalIndex]
            || !result.next.mobiles[physicalIndex].available)
            continue;

        Mobile const& mobile = result.next.mobiles[physicalIndex];
        if (!std::isfinite(mobile.north) || !std::isfinite(mobile.east)
            || !std::isfinite(mobile.heading)
            || !std::isfinite(mobile.speed)
            || !std::isfinite(mobile.turnRate)
            || std::abs(mobile.north)
                >= parameters.hardBoundaryScale * parameters.halfNorth
            || std::abs(mobile.east)
                >= parameters.hardBoundaryScale * parameters.halfEast)
        {
            result.next.failed = true;
        }
    }
    return GeneralizedTeamStepStatus::Applied;
}

static_assert(std::is_trivially_copyable_v<GeneralizedBlueTeamState>,
    "Generalized team state must remain suitable for fixed RLTools storage");
static_assert(std::is_trivially_copyable_v<GeneralizedBlueTeamStep>,
    "Generalized team transitions must not own dynamic storage");

} // namespace ppo::examples::bluedefense::defenders
