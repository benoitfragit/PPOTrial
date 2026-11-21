/**
 * @file flight_mission.hpp
 * @brief Episode state machine advanced once per physics substep.
 *
 * Role in the chain: JSBSim substep -> **mission** -> rewards -> environment action.
 * AdvanceMission compares the aircraft state before and after one 10 ms physics substep,
 * detects contact, crash, touchdown, stop and timeout, and returns the weighted rewards of
 * that substep, including the shaping term gamma_sub * Phi(s') - Phi(s) of each potential
 * channel.
 *
 * State machine:
 *   Approach --first safe contact--> Rollout --stopped for stoppedHoldSeconds--> Stopped
 *   any phase --unsafe contact, invalid state, off-surface roll--> Crash
 *   any phase --time, distance or altitude limit--> Timeout (truncation, not termination)
 *
 * Not here: the action loop, the command filter and the discounting across substeps
 * (Simulator/flight_environment.cpp).
 *
 * Read next: flight_mission.cpp, flight_landing.hpp, flight_reward.hpp.
 */
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
 * @return Weighted rewards and episode boundary flags; after a boundary, only the flags.
 */
StepResult AdvanceMission(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, CrashDiagnostic* crashDiagnostic = nullptr) noexcept;

/**
 * @brief Advance the mission by one physics substep with the potentials of the state before it.
 *
 * Used when the deck moves during the substep: config holds the deck pose after the substep,
 * while previousPotentials were evaluated with the pose before it, so the shaping
 * gamma_sub * Phi(s') - Phi(s) still telescopes over the episode.
 * @param config Current episode settings and runway pose after the substep.
 * @param mission Mission state to update.
 * @param before Aircraft state before the physics substep.
 * @param after Aircraft state after the physics substep.
 * @param previousPotentials ShapingPotentials of the mission and aircraft before the substep.
 * @param crashDiagnostic Optional terminal crash details; written only on crash.
 * @return Weighted rewards and episode boundary flags; after a boundary, only the flags.
 */
StepResult AdvanceMission(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, std::array<double, RewardCount> const& previousPotentials,
    CrashDiagnostic* crashDiagnostic = nullptr) noexcept;

} // namespace ppo::examples::flight
