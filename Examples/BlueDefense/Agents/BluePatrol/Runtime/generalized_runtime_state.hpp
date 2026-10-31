/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_runtime_state.hpp
 * @brief État physique compact et construction des observations v8 depuis le registre ECS.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "generalized_observation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
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
    /** @brief Zone proximity from ECS; excluded from actor observations. */
    std::array<float, TeamCapacity::MaximumBlueMobiles> restrictedZoneRisk{};
    /** @brief Defended-objective position in metres. */
    float objectiveNorth{};
    float objectiveEast{};
    /** @brief True while the fixed objective survives. */
    bool objectiveAvailable{true};
    /** @brief Objective side, normalized to -1 or +1. */
    int side{1};
    /** @brief Synchronized team-step counter. */
    std::size_t step{};
    /** @brief Team failure: no survivors or a living blue outside. */
    bool failed{};
};

/**
 * @brief Team failure follows the survival of the team, not its leader.
 *
 * Losing an assigned leader does not cut off surviving teammates' actions.
 * The episode ends when all blues are gone or a living blue crosses the
 * hard world boundary; objective failure remains a separate runtime outcome.
 *
 * @param state Runtime state with valid episode counts.
 * @param parameters Valid world boundaries and hard-boundary scale.
 * @param leaderPhysicalIndex Immutable leader index, validated for safety.
 * @return True for invalid leader index, no survivors or world-boundary exit.
 */
inline bool GeneralizedBlueTeamFailed(
    GeneralizedRuntimeState const& state, Parameters const& parameters,
    std::size_t const leaderPhysicalIndex) noexcept
{
    if (leaderPhysicalIndex >= state.counts.blueMobiles)
        return true;
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
        if (!finiteMobile(state.blues[index])
            || !std::isfinite(state.restrictedZoneRisk[index])
            || state.restrictedZoneRisk[index] < 0.F
            || state.restrictedZoneRisk[index] > 1.F)
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
 * @brief First invalid kinematic field; destruction does not excuse NaN.
 */
inline char const* InvalidGeneralizedMobileField(Mobile const& mobile) noexcept
{
    if (!std::isfinite(mobile.north)) return "north";
    if (!std::isfinite(mobile.east)) return "east";
    if (!std::isfinite(mobile.heading)) return "heading";
    if (!std::isfinite(mobile.speed)) return "speed";
    if (!std::isfinite(mobile.turnRate)) return "turnRate";
    if (mobile.available && mobile.speed < 0.F) return "speed";
    return nullptr;
}

/**
 * @brief Detailed error built only on the failing validation path.
 * @return Entity, field, numerical values and availability (or count error).
 */
inline std::string DescribeInvalidGeneralizedRuntimeState(
    GeneralizedRuntimeState const& state)
{
    if (!Valid(state.counts))
        return "counts P=" + std::to_string(state.counts.blueMobiles)
            + " N=" + std::to_string(state.counts.redMobiles);
    if (!std::isfinite(state.objectiveNorth)) return "objective.north";
    if (!std::isfinite(state.objectiveEast)) return "objective.east";

    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        Mobile const& mobile = state.blues[index];
        if (!std::isfinite(state.restrictedZoneRisk[index])
            || state.restrictedZoneRisk[index] < 0.F
            || state.restrictedZoneRisk[index] > 1.F)
            return "blue_" + std::to_string(index + 1U)
                + ".restrictedZoneRisk";
        if (char const* field = InvalidGeneralizedMobileField(mobile))
            return "blue_" + std::to_string(index + 1U) + "." + field
                + " north=" + std::to_string(mobile.north)
                + " east=" + std::to_string(mobile.east)
                + " heading=" + std::to_string(mobile.heading)
                + " speed=" + std::to_string(mobile.speed)
                + " turnRate=" + std::to_string(mobile.turnRate)
                + " available=" + std::to_string(mobile.available);
        if (state.lidarAvailable[index])
        {
            for (std::size_t ray = 0U; ray < state.lidar[index].size(); ++ray)
                if (!std::isfinite(state.lidar[index][ray]))
                    return "blue_" + std::to_string(index + 1U)
                        + ".lidar[" + std::to_string(ray) + "]";
        }
    }
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
    {
        Mobile const& mobile = state.reds[index];
        if (char const* field = InvalidGeneralizedMobileField(mobile))
            return "red_" + std::to_string(index + 1U) + "." + field
                + " north=" + std::to_string(mobile.north)
                + " east=" + std::to_string(mobile.east)
                + " heading=" + std::to_string(mobile.heading)
                + " speed=" + std::to_string(mobile.speed)
                + " turnRate=" + std::to_string(mobile.turnRate)
                + " available=" + std::to_string(mobile.available);
    }
    return "unknown validation failure";
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

/**
 * @brief Sérialise ou écrit les données associées à WriteRelativeTrack selon les garanties implémentées.
 */
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

/**
 * @brief Réalise l'opération ThreatPressure; voir les préconditions et effets dans son corps C++.
 * @details Applique des bornes numériques.
 */
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
