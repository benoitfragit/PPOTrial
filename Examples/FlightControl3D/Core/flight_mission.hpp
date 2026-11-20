#pragma once

#include "flight_config.hpp"

namespace ppo::examples::flight {

/**
 * @brief Advance the mission by one physics substep; landing limits also inspect the pre-impact state.
 * @param config Current episode settings and runway pose.
 * @param mission Mission state to update.
 * @param before Aircraft state before the physics substep.
 * @param after Aircraft state after the physics substep.
 * @param crashDiagnostic Optional terminal crash details; written only on crash.
 * @return Weighted rewards and episode boundary flags.
 */
StepResult AdvanceMission(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, CrashDiagnostic* crashDiagnostic = nullptr) noexcept;

} // namespace ppo::examples::flight
