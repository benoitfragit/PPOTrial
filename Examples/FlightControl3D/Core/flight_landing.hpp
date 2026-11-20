#pragma once

#include "runway_geometry.hpp"

namespace ppo::examples::flight {

/**
 * @brief Detects the first physical contact with the deck or the sea.
 * @param config Episode configuration and runway pose.
 * @param aircraft Aircraft state after a physics substep.
 * @return True for native wheel or structure contact, or a geometric fallback without wheel data.
 */
bool ContactDetected(Config const& config, AircraftState const& aircraft) noexcept;

/** @brief Whether every native wheel lies on the landable surface. */
bool WheelsInside(Config const& config, AircraftState const& aircraft) noexcept;

/** @brief Whether every native wheel is in contact with the ground. */
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
 * @param phase Phase used by the crash checks.
 * @param before Aircraft state before the substep.
 * @param after Aircraft state after the substep.
 * @param contact Whether the contact detector fired.
 * @return Fixed-capacity terminal diagnostic.
 */
CrashDiagnostic DiagnoseCrash(Config const& config, Phase phase, AircraftState const& before,
    AircraftState const& after, bool contact) noexcept;

} // namespace ppo::examples::flight
