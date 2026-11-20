#pragma once
#include "flight_session.hpp"
#include <raylib.h>

namespace ppo::examples::flight {
void DrawFlightScene(FlightSnapshot const& state, RenderTexture2D const& target);
void DrawFlightDisplayControls();
void DrawFlightInstruments(FlightSnapshot const& state);
/** @brief Draw the runway-relative vertical approach profile below the 3D scene. */
void DrawFlightApproachProfile(FlightSnapshot const& state);
/** @brief Show one signed episode total and the last PPO mean. */
void DrawFlightRewardBar(FlightSnapshot const& state,std::size_t channel);
} // namespace ppo::examples::flight
