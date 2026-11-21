/**
 * @file flight_view.hpp
 * @brief Drawing entry points of the FlightControl3D desktop view.
 *
 * Role in the chain: session snapshot -> **renderer**. The desktop EnvironmentRenderer
 * calls these functions every frame with the latest FlightSnapshot. They only draw: no
 * function here changes the session, the physics or the learning.
 *
 * Read next: flight_view.cpp (scene and panels), flight_profile_view.cpp,
 * flight_reward_view.cpp, flight_display.hpp.
 */
#pragma once

#include "flight_session.hpp"
#include <raylib.h>

namespace ppo::examples::flight {

/** @brief Draw the 3D scene (sea, deck, runway, guidance, aircraft) into a render texture. */
void DrawFlightScene(FlightSnapshot const& state, RenderTexture2D const& target);
/** @brief Camera selection widgets (chase, overview, runway) and the chase distance. */
void DrawFlightDisplayControls();
/** @brief Instrument window: status, targets, controls, rewards, diagnostics and crash reasons. */
void DrawFlightInstruments(FlightSnapshot const& state);
/** @brief Draw the runway-relative vertical approach profile below the 3D scene. */
void DrawFlightApproachProfile(FlightSnapshot const& state);
/** @brief Show one signed episode total and the last PPO mean. */
void DrawFlightRewardBar(FlightSnapshot const& state, std::size_t channel);

} // namespace ppo::examples::flight
