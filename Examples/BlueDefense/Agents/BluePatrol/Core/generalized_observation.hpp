/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Core/generalized_observation.hpp
 * @brief Contrat versionné des observations décentralisées v8, dimensions, offsets et masques.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "blue_team_roles.hpp"
#include "patrol_core.hpp"
#include "team_capacity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Raw fixed-capacity observation contract for a generalized blue team.
 *
 * Entity tracks remain separate from the compact neural-network input. Every
 * track stores its availability mask in its final feature. This representation
 * therefore preserves identity-free, masked sets without padding values ever
 * being interpreted as real entities.
 */
struct GeneralizedObservationLayout final
{
    /** @brief New raw-observation contract version. */
    static constexpr std::uint32_t Version = 8U;
    /** @brief Historical own-platform features retained for v7 migration. */
    static constexpr std::size_t Own = ObservationLayout::Own;
    /** @brief Exact pseudo-LiDAR ray count. */
    static constexpr std::size_t Lidar = 61U;
    /** @brief Explicit role and normalized formation-slot features. */
    static constexpr std::size_t Role = 2U;
    /** @brief Fixed hostile-mobile track capacity. */
    static constexpr std::size_t RedTracks = TeamCapacity::MaximumRedMobiles;
    /** @brief Features per hostile track, including the final mask. */
    static constexpr std::size_t RedFeatures = ObservationLayout::RedFeatures;
    /** @brief Other-blue-mobile capacity seen by one controlled mobile. */
    static constexpr std::size_t BlueTracks =
        TeamCapacity::MaximumBlueMobiles - 1U;
    /** @brief Features per friendly track, including the final mask. */
    static constexpr std::size_t FriendlyFeatures =
        ObservationLayout::FriendlyFeatures;
    /** @brief Fixed objective capacity, deliberately separate from mobiles. */
    static constexpr std::size_t ObjectiveTracks = 1U;
    /** @brief Patrol features; the explicit role replaces legacy slot eight. */
    static constexpr std::size_t Patrol = ObservationLayout::Patrol;
    /** @brief Historical global features. */
    static constexpr std::size_t Global = ObservationLayout::Global;
    /** @brief Normalized active P and N counts. */
    static constexpr std::size_t TeamCounts = 2U;

    static constexpr std::size_t OwnOffset = 0U;
    static constexpr std::size_t LidarOffset = OwnOffset + Own;
    static constexpr std::size_t RoleOffset = LidarOffset + Lidar;
    static constexpr std::size_t RedOffset = RoleOffset + Role;
    static constexpr std::size_t BlueOffset =
        RedOffset + RedTracks * RedFeatures;
    static constexpr std::size_t ObjectiveOffset =
        BlueOffset + BlueTracks * FriendlyFeatures;
    static constexpr std::size_t PatrolOffset =
        ObjectiveOffset + ObjectiveTracks * FriendlyFeatures;
    static constexpr std::size_t GlobalOffset = PatrolOffset + Patrol;
    static constexpr std::size_t TeamCountsOffset = GlobalOffset + Global;
    static constexpr std::size_t Size = TeamCountsOffset + TeamCounts;
    static constexpr std::size_t ActionSize = ObservationLayout::ActionSize;
};

static_assert(GeneralizedObservationLayout::Lidar == 61U);
static_assert(GeneralizedObservationLayout::RedTracks == 10U);
static_assert(GeneralizedObservationLayout::BlueTracks == 8U);

/** @brief Flat raw values supplied by the generalized ECS observation bridge. */
using GeneralizedRawObservation =
    std::array<float, GeneralizedObservationLayout::Size>;

/**
 * @brief Metadata associated with one generalized raw observation.
 *
 * Counts are duplicated as normalized raw features so the actor can learn
 * team-size-dependent behavior while this integer form enables strict checks.
 */
struct GeneralizedObservationMetadata final
{
    /** @brief Episode blue mobiles P, including unavailable/destroyed slots. */
    std::size_t activeBlueMobiles{};
    /** @brief Episode red mobiles N, including unavailable/destroyed slots. */
    std::size_t activeRedMobiles{};
};

/** @brief Raw observation and its validated episode counts. */
struct GeneralizedObservation final
{
    GeneralizedRawObservation values{};
    GeneralizedObservationMetadata metadata{};
};

/**
 * @brief Returns the index of a final per-track availability mask.
 * @param offset First feature of the track set.
 * @param track Zero-based track index.
 * @param features Feature count per track.
 * @return Flat mask index in a generalized raw observation.
 */
constexpr std::size_t GeneralizedTrackMaskIndex(
    std::size_t const offset,
    std::size_t const track,
    std::size_t const features) noexcept
{
    return offset + track * features + features - 1U;
}

/**
 * @brief Writes the explicit semantic role and normalized formation slot.
 * @param observation Raw observation to update.
 * @param assignment Valid immutable episode role assignment.
 */
inline void EncodeGeneralizedRole(
    GeneralizedRawObservation& observation,
    BlueRoleAssignment const assignment) noexcept
{
    observation[GeneralizedObservationLayout::RoleOffset] =
        assignment.role == BlueRole::Leader ? -1.F : 1.F;
    observation[GeneralizedObservationLayout::RoleOffset + 1U] =
        assignment.normalizedFormationSlot;
}

/**
 * @brief Encodes the active team counts in their normalized raw slots.
 * @param observation Raw observation to update.
 * @param counts Valid episode mobile counts.
 */
inline void EncodeGeneralizedTeamCounts(
    GeneralizedRawObservation& observation,
    ActiveTeamCounts const counts) noexcept
{
    observation[GeneralizedObservationLayout::TeamCountsOffset] =
        static_cast<float>(counts.blueMobiles - TeamCapacity::MinimumBlueMobiles)
        / static_cast<float>(TeamCapacity::MaximumBlueMobiles
                             - TeamCapacity::MinimumBlueMobiles);
    observation[GeneralizedObservationLayout::TeamCountsOffset + 1U] =
        static_cast<float>(counts.redMobiles)
        / static_cast<float>(TeamCapacity::MaximumRedMobiles);
}

/**
 * @brief Validates finite values, masks and P/N capacity consistency.
 * @param observation Observation produced by the ECS bridge.
 * A destroyed entity clears its mask without changing the episode P/N
 * metadata, so the number of live masks may be lower than the episode count.
 * @return True when all raw values and mobile masks are coherent.
 */
inline bool Valid(GeneralizedObservation const& observation) noexcept
{
    ActiveTeamCounts const counts{observation.metadata.activeBlueMobiles,
                                  observation.metadata.activeRedMobiles};
    if (!Valid(counts)
        || !std::all_of(observation.values.begin(), observation.values.end(),
                       [](float const value) { return std::isfinite(value); }))
    {
        return false;
    }

    float const role =
        observation.values[GeneralizedObservationLayout::RoleOffset];
    float const formationSlot =
        observation.values[GeneralizedObservationLayout::RoleOffset + 1U];
    if ((role != -1.F && role != 1.F)
        || formationSlot < -1.F || formationSlot > 1.F
        || (role == -1.F && formationSlot != 0.F))
    {
        return false;
    }

    auto const countMasks = [&](std::size_t const offset,
                                std::size_t const tracks,
                                std::size_t const features) {
        std::size_t count = 0U;
        for (std::size_t track = 0U; track < tracks; ++track)
        {
            float const mask = observation.values[
                GeneralizedTrackMaskIndex(offset, track, features)];
            if (mask != 0.F && mask != 1.F)
                return tracks + 1U;
            count += mask > 0.5F ? 1U : 0U;
        }
        return count;
    };

    if (countMasks(GeneralizedObservationLayout::RedOffset,
                   GeneralizedObservationLayout::RedTracks,
                   GeneralizedObservationLayout::RedFeatures)
        > counts.redMobiles)
    {
        return false;
    }
    if (countMasks(GeneralizedObservationLayout::BlueOffset,
                   GeneralizedObservationLayout::BlueTracks,
                   GeneralizedObservationLayout::FriendlyFeatures)
        > counts.blueMobiles - 1U)
    {
        return false;
    }

    float const objectiveMask = observation.values[GeneralizedTrackMaskIndex(
        GeneralizedObservationLayout::ObjectiveOffset, 0U,
        GeneralizedObservationLayout::FriendlyFeatures)];
    if (objectiveMask != 0.F && objectiveMask != 1.F)
        return false;

    GeneralizedRawObservation expected = observation.values;
    EncodeGeneralizedTeamCounts(expected, counts);
    return observation.values[GeneralizedObservationLayout::TeamCountsOffset]
               == expected[GeneralizedObservationLayout::TeamCountsOffset]
        && observation.values[GeneralizedObservationLayout::TeamCountsOffset + 1U]
               == expected[GeneralizedObservationLayout::TeamCountsOffset + 1U];
}

/**
 * @brief Converts a historical 2v2 v7 observation to the new raw contract.
 *
 * The other blue mobile occupies the first blue slot; the objective remains a
 * distinct track. Padding is zero-masked. This adapter is intended only for
 * migration equivalence tests and initial v7 policy rollout validation.
 *
 * @param legacy Historical flattened v7 observation.
 * @return Valid P=2/N=2 generalized raw observation.
 */
inline GeneralizedObservation UpgradeLegacy2v2Observation(
    Observation const& legacy) noexcept
{
    GeneralizedObservation result{};
    result.metadata = {2U, 2U};
    auto& raw = result.values;

    std::copy_n(legacy.begin(), ObservationLayout::Own,
                raw.begin() + GeneralizedObservationLayout::OwnOffset);
    std::copy_n(legacy.begin() + ObservationLayout::Own,
                ObservationLayout::Lidar,
                raw.begin() + GeneralizedObservationLayout::LidarOffset);
    raw[GeneralizedObservationLayout::RoleOffset] =
        legacy[ObservationLayout::PatrolOffset + 8U];
    raw[GeneralizedObservationLayout::RoleOffset + 1U] =
        legacy[ObservationLayout::GlobalOffset + 1U];
    std::copy_n(legacy.begin() + ObservationLayout::RedOffset,
                ObservationLayout::RedTracks * ObservationLayout::RedFeatures,
                raw.begin() + GeneralizedObservationLayout::RedOffset);
    std::copy_n(legacy.begin() + ObservationLayout::FriendOffset,
                ObservationLayout::FriendlyFeatures,
                raw.begin() + GeneralizedObservationLayout::BlueOffset);
    std::copy_n(
        legacy.begin() + ObservationLayout::FriendOffset
            + ObservationLayout::FriendlyFeatures,
        ObservationLayout::FriendlyFeatures,
        raw.begin() + GeneralizedObservationLayout::ObjectiveOffset);
    std::copy_n(legacy.begin() + ObservationLayout::PatrolOffset,
                ObservationLayout::Patrol,
                raw.begin() + GeneralizedObservationLayout::PatrolOffset);
    std::copy_n(legacy.begin() + ObservationLayout::GlobalOffset,
                ObservationLayout::Global,
                raw.begin() + GeneralizedObservationLayout::GlobalOffset);
    EncodeGeneralizedTeamCounts(raw, {2U, 2U});
    return result;
}

} // namespace ppo::examples::bluedefense::defenders
