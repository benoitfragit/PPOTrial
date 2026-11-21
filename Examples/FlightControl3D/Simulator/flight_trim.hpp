/**
 * @file flight_trim.hpp
 * @brief Trim by simulated flight, for aircraft whose engine JSBSim's FGTrim cannot trim.
 *
 * Role in the chain: configuration tooling only (FlightControlHeadless --trim). It finds the
 * action reference of a new aircraft ("elevatorTrim" and "throttle"); it never runs during
 * training or inference, where the policy alone flies the aircraft.
 *
 * JSBSim's FGTrim solves the longitudinal equilibrium with integration suspended. A
 * turboprop such as the PC-7 PT6A then keeps the power of its spool state, so the throttle
 * has no effect and the trim fails. Here the aircraft flies the condition instead: the
 * elevator holds the flight path with pitch-rate damping, the throttle holds the airspeed,
 * both with proportional and integral terms, the ailerons keep the wings level, and the
 * commands are averaged once settled. On the T-6 it agrees with FGTrim within 0.005.
 *
 * Read next: jsbsim_aircraft.hpp (FGTrim), Runtime/flight_headless.cpp (--trim).
 */
#pragma once

#include "jsbsim_aircraft.hpp"

namespace ppo::examples::flight {

/**
 * @brief Flies a steady wings-level condition and returns the settled commands.
 * @param aircraft Simulator; it is reset to the condition and left at the end of the flight.
 * @param config Validated configuration (action reference used as the first guess).
 * @param condition Speed, flight path, flap, gear and height to hold.
 * @return Mean elevator, throttle and pitch over the settled window; converged is false when
 *     the mean speed or path errors, or the command spread, stay above their tolerances.
 */
TrimResult FlyTrim(JsbsimAircraft& aircraft, Config const& config, TrimCondition const& condition);

} // namespace ppo::examples::flight
