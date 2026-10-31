/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Core/blue_team_roles.hpp
 * @brief Désignation du leader et affectations stables des équipiers en wedge.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "team_capacity.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

namespace ppo::examples::bluedefense::defenders {

/** @brief Semantic responsibility of an active blue mobile. */
enum class BlueRole : std::uint8_t
{
    Leader,
    Teammate
};

/**
 * @brief Stable role data assigned to one physical blue mobile for an episode.
 *
 * The leader has no formation slot. Teammates own a unique slot in
 * `[0, activeBlueCount - 2]`. The normalized slot is in `[-1, 1]`; it is zero
 * when the team has only one teammate.
 */
struct BlueRoleAssignment final
{
    /** @brief Sentinel used by the leader, which has no teammate slot. */
    static constexpr std::size_t NoFormationSlot =
        TeamCapacity::MaximumBlueMobiles;

    /** @brief Semantic leader or teammate responsibility. */
    BlueRole role{BlueRole::Teammate};
    /** @brief Stable zero-based teammate slot or `NoFormationSlot`. */
    std::size_t formationSlot{NoFormationSlot};
    /** @brief Slot encoding in `[-1, 1]` for policy observations. */
    float normalizedFormationSlot{};
};

/**
 * @brief Leader-relative offset associated with a teammate formation slot.
 *
 * `longitudinalMeters` is positive ahead of the leader. `inwardMeters` is
 * positive towards the inside of the patrol curve and negative outside.
 */
struct FormationOffset final
{
    /** @brief Signed distance along the leader heading, in metres. */
    float longitudinalMeters{};
    /** @brief Signed distance towards the curve inside, in metres. */
    float inwardMeters{};
};

/**
 * @brief Returns a normalized scalar encoding for a teammate slot.
 * @param formationSlot Zero-based stable slot identifier.
 * @param teammateCount Number of teammates in the active team.
 * @return A value in `[-1, 1]`, unique when more than one slot exists.
 */
constexpr float NormalizeFormationSlot(
    std::size_t const formationSlot,
    std::size_t const teammateCount) noexcept
{
    if (teammateCount <= 1U)
        return 0.F;

    return -1.F + 2.F * static_cast<float>(formationSlot)
        / static_cast<float>(teammateCount - 1U);
}

/**
 * @brief Builds the distinct V-formation offset for a teammate slot.
 *
 * Slots alternate inside/outside while successive pairs move one additional
 * row behind the leader. Slot zero is the historical inside wingman position.
 *
 * @param formationSlot Zero-based stable slot identifier.
 * @param separationDistanceMeters Distance between consecutive rows, metres.
 * @param separationAngleRadians Formation angle from the leader rear axis.
 * @return Leader-relative longitudinal/inward offset in metres, or zero when
 * either input is invalid.
 */
inline FormationOffset EvaluateFormationOffset(
    std::size_t const formationSlot,
    float const separationDistanceMeters,
    float const separationAngleRadians) noexcept
{
    if (formationSlot >= TeamCapacity::MaximumBlueMobiles - 1U
        || !std::isfinite(separationDistanceMeters)
        || !std::isfinite(separationAngleRadians)
        || separationDistanceMeters <= 0.F
        || separationAngleRadians <= 0.F
        || separationAngleRadians >= 0.5F * 3.14159265358979323846F)
    {
        return {};
    }

    float const row = static_cast<float>(formationSlot / 2U + 1U);
    float const lateralSign = formationSlot % 2U == 0U ? 1.F : -1.F;
    return {
        -row * separationDistanceMeters * std::cos(separationAngleRadians),
        lateralSign * row * separationDistanceMeters
            * std::sin(separationAngleRadians)};
}

/**
 * @brief Immutable episode assignment from physical blue indices to roles.
 *
 * Construction guarantees exactly one leader and a unique stable formation
 * slot for every teammate. Physical indices use their deterministic input
 * order; changing the leader never changes that order.
 */
class BlueTeamRoles final
{
public:
    /**
     * @brief Creates a validated assignment with an explicit leader.
     * @param activeBlueCount Number of active blue mobiles in `[1, 9]`.
     * @param leaderPhysicalIndex Physical index of the episode leader.
     * @return Assignment, or `std::nullopt` when the inputs are invalid.
     */
    static std::optional<BlueTeamRoles> Create(
        std::size_t const activeBlueCount,
        std::size_t const leaderPhysicalIndex) noexcept
    {
        if (!Valid(ActiveTeamCounts{
                activeBlueCount, TeamCapacity::MinimumRedMobiles})
            || leaderPhysicalIndex >= activeBlueCount)
        {
            return std::nullopt;
        }

        BlueTeamRoles result;
        result.activeBlueCount_ = activeBlueCount;
        result.leaderPhysicalIndex_ = leaderPhysicalIndex;

        std::size_t nextSlot = 0U;
        std::size_t const teammateCount = activeBlueCount - 1U;
        for (std::size_t physicalIndex = 0U;
             physicalIndex < activeBlueCount; ++physicalIndex)
        {
            auto& assignment = result.assignments_[physicalIndex];
            if (physicalIndex == leaderPhysicalIndex)
            {
                assignment = {BlueRole::Leader,
                              BlueRoleAssignment::NoFormationSlot, 0.F};
                continue;
            }

            assignment = {BlueRole::Teammate, nextSlot,
                          NormalizeFormationSlot(nextSlot, teammateCount)};
            ++nextSlot;
        }
        return result;
    }

    /**
     * @brief Samples the physical leader reproducibly from an episode RNG.
     * @tparam RandomEngine Uniform random bit generator type.
     * @param activeBlueCount Number of active blue mobiles in `[1, 9]`.
     * @param randomEngine Episode reset random engine.
     * @return Valid sampled assignment, or `std::nullopt` for an invalid count.
     */
    template<typename RandomEngine>
/**
 * @brief Construit, transforme ou échantillonne les données produites par Sample.
 */
    static std::optional<BlueTeamRoles> Sample(
        std::size_t const activeBlueCount,
        RandomEngine& randomEngine)
    {
        if (!Valid(ActiveTeamCounts{
                activeBlueCount, TeamCapacity::MinimumRedMobiles}))
            return std::nullopt;

        std::uniform_int_distribution<std::size_t> distribution{
            0U, activeBlueCount - 1U};
        return Create(activeBlueCount, distribution(randomEngine));
    }

    /**
     * @brief Returns the number of active physical blue mobiles.
     * @return Validated episode-level blue count.
     */
    std::size_t ActiveBlueCount() const noexcept
    {
        return activeBlueCount_;
    }

    /**
     * @brief Returns the unique leader physical index.
     * @return Index in `[0, ActiveBlueCount())`.
     */
    std::size_t LeaderPhysicalIndex() const noexcept
    {
        return leaderPhysicalIndex_;
    }

    /**
     * @brief Looks up the assignment of an active physical blue mobile.
     * @param physicalIndex Physical index in deterministic episode order.
     * @return Assignment, or `std::nullopt` when the index is inactive.
     */
    std::optional<BlueRoleAssignment> Assignment(
        std::size_t const physicalIndex) const noexcept
    {
        if (physicalIndex >= activeBlueCount_)
            return std::nullopt;
        return assignments_[physicalIndex];
    }

    /**
     * @brief Verifies the exactly-one-leader and unique-slot invariants.
     * @return True when all active assignments satisfy the domain contract.
     */
    bool IsValid() const noexcept
    {
        if (!Valid(ActiveTeamCounts{
                activeBlueCount_, TeamCapacity::MinimumRedMobiles})
            || leaderPhysicalIndex_ >= activeBlueCount_)
        {
            return false;
        }

        std::size_t leaderCount = 0U;
        std::array<bool, TeamCapacity::MaximumBlueMobiles - 1U> usedSlots{};
        for (std::size_t physicalIndex = 0U;
             physicalIndex < activeBlueCount_; ++physicalIndex)
        {
            auto const& assignment = assignments_[physicalIndex];
            if (assignment.role == BlueRole::Leader)
            {
                ++leaderCount;
                if (physicalIndex != leaderPhysicalIndex_
                    || assignment.formationSlot
                        != BlueRoleAssignment::NoFormationSlot)
                {
                    return false;
                }
                continue;
            }

            if (assignment.formationSlot >= activeBlueCount_ - 1U
                || usedSlots[assignment.formationSlot])
            {
                return false;
            }
            usedSlots[assignment.formationSlot] = true;
            if (assignment.normalizedFormationSlot
                != NormalizeFormationSlot(
                    assignment.formationSlot, activeBlueCount_ - 1U))
            {
                return false;
            }
        }
        return leaderCount == 1U;
    }

private:
    BlueTeamRoles() = default;

    std::array<BlueRoleAssignment,
               TeamCapacity::MaximumBlueMobiles> assignments_{};
    std::size_t activeBlueCount_{};
    std::size_t leaderPhysicalIndex_{};
};

} // namespace ppo::examples::bluedefense::defenders
