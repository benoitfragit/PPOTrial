/**
 * @file flight_landing.hpp
 * @brief Ground-contact detection, touchdown limits and crash diagnostics.
 *
 * Role in the chain: JSBSim state -> **contact and crash rules** -> mission. These pure
 * functions decide whether a physics substep touched the ground and whether that contact
 * was safe. Every limit is checked on both snapshots around the substep (before and after),
 * because a hard contact can bounce within one 10 ms step.
 *
 * Contact detection: native JSBSim wheel and structure contacts first; only when the wheel
 * samples are missing, a geometric fallback compares the height above the deck (or the sea
 * outside it) with MissionSettings::contactHeightMeters.
 *
 * Not here: the episode state machine (flight_mission.cpp) and the reward values
 * (flight_reward.cpp).
 *
 * Read next: flight_landing.cpp, flight_mission.cpp.
 */
#pragma once

#include "runway_geometry.hpp"

namespace ppo::examples::flight {

/**
 * @brief Detects the first physical contact with the deck or the sea.
 * @param config Episode configuration and runway pose.
 * @param aircraft Aircraft state after a physics substep.
 * @return True for native wheel or structure contact, a belly contact with the gear not locked
 *     down, or a geometric fallback without wheel data.
 */
bool ContactDetected(Config const& config, AircraftState const& aircraft) noexcept;

/** @brief Whether every native wheel lies on the landable surface (true without wheel data). */
bool WheelsInside(Config const& config, AircraftState const& aircraft) noexcept;

/** @brief Whether every native wheel is in contact with the ground (weight on wheels without wheel data). */
bool AllWheelsDown(AircraftState const& aircraft) noexcept;

/**
 * @brief Checks the touchdown limits on both snapshots around the contact substep.
 * @param config Episode configuration and landing limits.
 * @param before Aircraft state before the substep.
 * @param after Aircraft state after the substep.
 * @return True when gear, sink, speed and attitude are all within limits.
 */
bool LandingSafe(Config const& config, AircraftState const& before, AircraftState const& after) noexcept;

/**
 * @brief Capture every failed crash condition using the same snapshots as the mission.
 * @param config Episode configuration and runway pose.
 * @param mission Mission state used by the crash checks: phase and route progress.
 * @param before Aircraft state before the substep.
 * @param after Aircraft state after the substep.
 * @param contact Whether the contact detector fired.
 * @return Fixed-capacity terminal diagnostic; angles are reported in degrees.
 */
CrashDiagnostic DiagnoseCrash(Config const& config, MissionState const& mission, AircraftState const& before,
    AircraftState const& after, bool contact) noexcept;

} // namespace ppo::examples::flight
