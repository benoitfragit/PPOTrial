#pragma once
#include "flight_session.hpp"
#include <raylib.h>

namespace ppo::examples::flight {
void DrawFlightScene(FlightSnapshot const& state, RenderTexture2D const& target);
void DrawFlightDisplayControls();
void DrawFlightInstruments(FlightSnapshot const& state);
} // namespace ppo::examples::flight
