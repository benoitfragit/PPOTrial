#pragma once

#include "generalized_observation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Allocation-free ECS-neutral snapshot consumed by generalized MAPPO.
 *
 * Episode membership is encoded by @ref counts. Destroyed mobiles retain their
 * stable index and set `Mobile::available=false`; unused capacity stays zero.
 */
struct GeneralizedRuntimeState final
{
    /** @brief Validated episode-level blue and red mobile counts. */
    ActiveTeamCounts counts{};
    /** @brief Blue mobiles in stable blue_1..blue_P order. */
    std::array<Mobile, TeamCapacity::MaximumBlueMobiles> blues{};
    /** @brief Red mobiles in stable red_1..red_N order. */
    std::array<Mobile, TeamCapacity::MaximumRedMobiles> reds{};
    /** @brief ECS pseudo-LiDAR values for each participating blue mobile. */
    std::array<std::array<float, GeneralizedObservationLayout::Lidar>,
               TeamCapacity::MaximumBlueMobiles> lidar{};
    /** @brief True when the corresponding LiDAR row came from the ECS. */
    std::array<bool, TeamCapacity::MaximumBlueMobiles> lidarAvailable{};
    /** @brief Defended-objective position in metres. */
    float objectiveNorth{};
    float objectiveEast{};
    /** @brief True while the fixed objective survives. */
    bool objectiveAvailable{true};
    /** @brief Objective side, normalized to -1 or +1. */
    int side{1};
    /** @brief Synchronized team-step counter. */
    std::size_t step{};
    /** @brief Team failure: lost leader, no survivors, or a living blue outside. */
    bool failed{};
};

/**
 * @brief Team failure follows the assigned leader rather than any casualty.
 *
 * The episode ends if the assigned leader is destroyed, even while wingmen
 * survive. Losing a nonleader does not end the episode while the leader lives.
 * A living blue crossing the hard world boundary also ends the episode.
 * Other mission outcomes are evaluated separately by the MAPPO runtime.
 *
 * @param state Runtime state with valid episode counts.
 * @param parameters Valid world boundaries and hard-boundary scale.
 * @param leaderPhysicalIndex Immutable episode-assigned leader index.
 * @return True for leader loss, complete blue loss, or living blue out of bounds.
 */
inline bool GeneralizedBlueTeamFailed(
    GeneralizedRuntimeState const& state, Parameters const& parameters,
    std::size_t const leaderPhysicalIndex) noexcept
{
    if (leaderPhysicalIndex >= state.counts.blueMobiles
        || !state.blues[leaderPhysicalIndex].available)
    {
        return true;
    }
    bool anyLivingBlue = false;
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        Mobile const& blue = state.blues[index];
        if (!blue.available)
            continue;
        anyLivingBlue = true;
        if (std::abs(blue.north)
                >= parameters.hardBoundaryScale * parameters.halfNorth
            || std::abs(blue.east)
                >= parameters.hardBoundaryScale * parameters.halfEast)
        {
            return true;
        }
    }
    return !anyLivingBlue;
}

/**
 * @brief Validates runtime counts and finite participating entity kinematics.
 * @param state Candidate generalized runtime snapshot.
 * @return True when all participating storage is finite and capacity-safe.
 */
inline bool Valid(GeneralizedRuntimeState const& state) noexcept
{
    if (!Valid(state.counts) || !std::isfinite(state.objectiveNorth)
        || !std::isfinite(state.objectiveEast))
    {
        return false;
    }
    auto const finiteMobile = [](Mobile const& mobile) {
        return std::isfinite(mobile.north) && std::isfinite(mobile.east)
            && std::isfinite(mobile.heading) && std::isfinite(mobile.speed)
            && std::isfinite(mobile.turnRate)
            && (!mobile.available || mobile.speed >= 0.F);
    };
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        if (!finiteMobile(state.blues[index]))
            return false;
        if (state.lidarAvailable[index]
            && !std::all_of(state.lidar[index].begin(), state.lidar[index].end(),
                [](float const value) { return std::isfinite(value); }))
        {
            return false;
        }
    }
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
        if (!finiteMobile(state.reds[index]))
            return false;
    return true;
}

/**
 * @brief Average objective threat risk across the active red slots.
 *        Dead reds contribute zero; fixed active N avoids reward-scale drift.
 */
inline float ObjectiveThreatLevel(GeneralizedRuntimeState const& state,
                                  Parameters const& parameters) noexcept
{
    if (state.counts.redMobiles == 0U)
        return 0.F;
    DefenseTrack const objective{state.objectiveNorth, state.objectiveEast,
                                 0.F, 0.F, state.objectiveAvailable};
    float risk = 0.F;
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
    {
        Mobile const& red = state.reds[index];
        DefenseTrack const attacker{red.north, red.east, red.heading,
                                    red.speed, red.available};
        risk += ObjectiveThreatRisk(parameters.defense, attacker, objective);
    }
    return risk / static_cast<float>(state.counts.redMobiles);
}

namespace generalized_runtime_detail {

inline void WriteRelativeTrack(
    GeneralizedRawObservation& observation, std::size_t& offset,
    Mobile const& own, Mobile const& target, Parameters const& parameters,
    float const type) noexcept
{
    if (!target.available)
    {
        offset += GeneralizedObservationLayout::FriendlyFeatures;
        return;
    }
    float const diagonal = std::hypot(parameters.halfNorth,
                                      parameters.halfEast);
    float const north = target.north - own.north;
    float const east = target.east - own.east;
    float const relativeBearing = std::atan2(east, north) - own.heading;
    observation[offset++] = north / (2.F * parameters.halfNorth);
    observation[offset++] = east / (2.F * parameters.halfEast);
    observation[offset++] = (target.speed * std::cos(target.heading)
        - own.speed * std::cos(own.heading)) / parameters.maxSpeed;
    observation[offset++] = (target.speed * std::sin(target.heading)
        - own.speed * std::sin(own.heading)) / parameters.maxSpeed;
    observation[offset++] = std::hypot(north, east) / (2.F * diagonal);
    observation[offset++] = std::cos(relativeBearing);
    observation[offset++] = std::sin(relativeBearing);
    observation[offset++] = type;
    observation[offset++] = 1.F;
}

inline float ThreatPressure(GeneralizedRuntimeState const& state,
                            Parameters const& parameters) noexcept
{
    float pressure = 0.F;
    DefenseTrack const objective{state.objectiveNorth, state.objectiveEast,
                                 0.F, 0.F, state.objectiveAvailable};
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
    {
        Mobile const& red = state.reds[index];
        DefenseTrack const attacker{red.north, red.east, red.heading,
                                    red.speed, red.available};
        pressure = std::max(pressure, ObjectiveThreatRisk(
            parameters.defense, attacker, objective));
    }
    return pressure;
}

} // namespace generalized_runtime_detail

/**
 * @brief Builds exactly P decentralized generalized observations.
 *
 * Red and friendly mobiles are encoded as masked sets in stable scratch
 * storage. The attention model is permutation-invariant and no entity count
 * changes the semantic offsets of patrol, objective or count features.
 *
 * @param parameters Valid patrol/world normalization parameters.
 * @param state Finite fixed-capacity runtime snapshot.
 * @param roles Exactly-one-leader episode assignment.
 * @param observations Output array; only `[0, P)` is populated.
 * @return True on success, false when any contract is invalid.
 */
inline bool BuildGeneralizedObservations(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    BlueTeamRoles const& roles,
    std::array<GeneralizedObservation,
               TeamCapacity::MaximumBlueMobiles>& observations) noexcept
{
    observations = {};
    if (!Valid(parameters) || !Valid(state) || !roles.IsValid()
        || roles.ActiveBlueCount() != state.counts.blueMobiles)
    {
        return false;
    }

    float const diagonal = std::hypot(parameters.halfNorth,
                                      parameters.halfEast);
    std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
    Mobile const& leader = state.blues[leaderIndex];
    PatrolProjection const projection = ProjectOnPatrolCurve(
        parameters.geometry, leader.north, leader.east);
    float const pressure = generalized_runtime_detail::ThreatPressure(
        state, parameters);

    for (std::size_t physicalIndex = 0U;
         physicalIndex < state.counts.blueMobiles; ++physicalIndex)
    {
        GeneralizedObservation& wrapped = observations[physicalIndex];
        wrapped.metadata = {state.counts.blueMobiles,
                            state.counts.redMobiles};
        GeneralizedRawObservation& observation = wrapped.values;
        Mobile const& own = state.blues[physicalIndex];
        BlueRoleAssignment const assignment = *roles.Assignment(physicalIndex);

        std::size_t offset = GeneralizedObservationLayout::OwnOffset;
        observation[offset++] = own.north / parameters.halfNorth;
        observation[offset++] = own.east / parameters.halfEast;
        observation[offset++] = own.speed / parameters.maxSpeed;
        observation[offset++] = std::cos(own.heading);
        observation[offset++] = std::sin(own.heading);
        observation[offset++] = own.turnRate / parameters.maxRotationRate;
        observation[offset++] = own.available ? 1.F : 0.F;

        offset = GeneralizedObservationLayout::LidarOffset;
        for (std::size_t ray = 0U;
             ray < GeneralizedObservationLayout::Lidar; ++ray)
        {
            if (state.lidarAvailable[physicalIndex])
            {
                observation[offset++] = Clamp(
                    state.lidar[physicalIndex][ray], 0.F, 1.F);
            }
            else
            {
                float const angle = own.heading - kPi
                    + static_cast<float>(ray) * (kTwoPi / 60.F);
                float const distance = RayDistanceToBounds(
                    parameters, own, angle);
                observation[offset++] = distance < 0.F
                    ? 1.F : distance / (distance + diagonal);
            }
        }
        EncodeGeneralizedRole(observation, assignment);

        offset = GeneralizedObservationLayout::RedOffset;
        for (std::size_t redIndex = 0U;
             redIndex < state.counts.redMobiles; ++redIndex)
        {
            Mobile const& red = state.reds[redIndex];
            if (!red.available)
            {
                offset += GeneralizedObservationLayout::RedFeatures;
                continue;
            }
            float const north = red.north - own.north;
            float const east = red.east - own.east;
            float const relativeBearing = std::atan2(east, north) - own.heading;
            observation[offset++] = north / (2.F * parameters.halfNorth);
            observation[offset++] = east / (2.F * parameters.halfEast);
            observation[offset++] = (red.speed * std::cos(red.heading)
                - own.speed * std::cos(own.heading)) / parameters.maxSpeed;
            observation[offset++] = (red.speed * std::sin(red.heading)
                - own.speed * std::sin(own.heading)) / parameters.maxSpeed;
            observation[offset++] = std::hypot(north, east)
                / (2.F * diagonal);
            observation[offset++] = std::cos(relativeBearing);
            observation[offset++] = std::sin(relativeBearing);
            observation[offset++] = red.speed / parameters.maxSpeed;
            observation[offset++] = 1.F;
            observation[offset++] = red.north / parameters.halfNorth;
            observation[offset++] = red.east / parameters.halfEast;
            observation[offset++] = (std::abs(red.north) < parameters.halfNorth
                && std::abs(red.east) < parameters.halfEast) ? 1.F : 0.F;
            observation[offset++] = 1.F;
        }

        offset = GeneralizedObservationLayout::BlueOffset;
        for (std::size_t other = 0U;
             other < state.counts.blueMobiles; ++other)
        {
            if (other == physicalIndex)
                continue;
            generalized_runtime_detail::WriteRelativeTrack(
                observation, offset, own, state.blues[other], parameters, 0.F);
        }
        offset = GeneralizedObservationLayout::ObjectiveOffset;
        generalized_runtime_detail::WriteRelativeTrack(
            observation, offset, own,
            Mobile{state.objectiveNorth, state.objectiveEast,
                   0.F, 0.F, 0.F, state.objectiveAvailable},
            parameters, 1.F);

        offset = GeneralizedObservationLayout::PatrolOffset;
        if (assignment.role == BlueRole::Leader)
        {
            float const relativeTangent = projection.pose.heading - own.heading;
            float const phaseAngle = kTwoPi * projection.pose.phase;
            float const lookahead = std::max(
                3.F * own.speed, 0.25F * parameters.geometry.width);
            PatrolCurvePose const preview = EvaluatePatrolCurve(
                parameters.geometry, projection.pose.arclength + lookahead);
            float const minimumTurnRadius = parameters.cruiseSpeed
                / parameters.maxRotationRate;
            observation[offset++] = std::cos(relativeTangent);
            observation[offset++] = std::sin(relativeTangent);
            observation[offset++] = Clamp(
                projection.signedCrossTrackError
                    / std::max(0.5F * parameters.geometry.width, 1.F),
                -2.F, 2.F);
            observation[offset++] = std::sin(phaseAngle);
            observation[offset++] = std::cos(phaseAngle);
            observation[offset++] = Clamp(
                preview.curvature * minimumTurnRadius, -1.F, 1.F);
            offset += 2U;
        }
        else
        {
            FormationOffset const slot = EvaluateFormationOffset(
                assignment.formationSlot,
                parameters.geometry.separationDistance,
                parameters.geometry.separationAngle);
            float const cosHeading = std::cos(leader.heading);
            float const sinHeading = std::sin(leader.heading);
            float const inward = static_cast<float>(
                parameters.geometry.direction) * slot.inwardMeters;
            float const expectedNorth = leader.north
                + slot.longitudinalMeters * cosHeading
                - inward * sinHeading;
            float const expectedEast = leader.east
                + slot.longitudinalMeters * sinHeading
                + inward * cosHeading;
            float const errorNorth = own.north - expectedNorth;
            float const errorEast = own.east - expectedEast;
            offset += 6U;
            observation[offset++] = Clamp(
                (errorNorth * cosHeading + errorEast * sinHeading)
                    / parameters.geometry.separationDistance,
                -2.F, 2.F);
            observation[offset++] = Clamp(
                static_cast<float>(parameters.geometry.direction)
                    * (-errorNorth * sinHeading + errorEast * cosHeading)
                    / parameters.geometry.separationDistance,
                -2.F, 2.F);
        }
        observation[offset++] = assignment.role == BlueRole::Leader
            ? -1.F : 1.F;
        observation[offset++] = pressure;
        observation[GeneralizedObservationLayout::GlobalOffset] = 1.F;
        observation[GeneralizedObservationLayout::GlobalOffset + 1U] = 0.F;
        EncodeGeneralizedTeamCounts(observation, state.counts);
        if (!Valid(wrapped))
            return false;
    }
    return true;
}

static_assert(std::is_trivially_copyable_v<GeneralizedRuntimeState>,
    "Generalized runtime snapshots must remain allocation-free values");

} // namespace ppo::examples::bluedefense::defenders
