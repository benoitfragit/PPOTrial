/**
 * @file flight_reward.hpp
 * @brief Reward potentials, the direction cost and the touchdown quality.
 *
 * Role in the chain: mission -> **reward** -> PPO. AdvanceMission (flight_mission.cpp)
 * calls these functions once per physics substep and weights them with
 * RewardSettings::weights. The channels are:
 *
 * | Channel       | Kind                | Active phase       | Unweighted range           |
 * |---------------|---------------------|--------------------|----------------------------|
 * | Path          | potential Phi       | approach + rollout | (0, 1], 1 on the ground    |
 * | Direction     | running cost        | approach           | [-dt, 0] per substep       |
 * | VerticalSpeed | potential Phi       | approach           | [-1, 0], 0 on the ground   |
 * | Speed         | potential Phi       | approach + rollout | [-1, 0] airborne, [0, 1] ground |
 * | Touchdown     | one-shot event      | first safe contact | quality in [0, 1]          |
 * | Stop          | one-shot event      | confirmed stop     | landing-area factor [0, 1] |
 * | Crash         | one-shot event      | any                | 1 (weight is negative)     |
 *
 * Potential channels contribute gamma_sub * Phi(s') - Phi(s) per substep, see
 * flight_mission.cpp and FlightEnvironment::Step for the discounting.
 *
 * Not here: deciding when an event happens (flight_mission.cpp, flight_landing.cpp).
 *
 * Read next: flight_reward.cpp, flight_mission.cpp, Simulator/flight_environment.cpp.
 */
#pragma once

#include "flight_guidance.hpp"

namespace ppo::examples::flight {

/** @brief Vertical-speed error, in m/s, at which the VerticalSpeed potential saturates. */
inline constexpr double VerticalSpeedErrorScaleMps = 2.0;
/** @brief Airspeed error, in m/s, at which the airborne Speed potential saturates. */
inline constexpr double AirspeedErrorScaleMps = 10.0;
/** @brief Length of the touchdown zone past the threshold, in metres (ICAO touchdown zone). */
inline constexpr double TouchdownZoneLengthMeters = 900.0;

/** @brief Whether a channel is a potential-based shaping term, discounted inside each action. */
bool IsPotentialChannel(Reward channel) noexcept;

/**
 * @brief Distance from a point to the approach reference.
 * @param config Episode configuration and runway pose.
 * @param point Point in runway axes (along, cross, height above the deck).
 * @return Distance to the aim point before it; lateral offset and height once past it, in metres.
 */
double ApproachDistance(Config const& config, Gate const& point) noexcept;

/**
 * @brief Remaining flight distance used by the Path potential.
 *
 * Without remaining route gates it is ApproachDistance of the aircraft, unchanged for a
 * direct approach. With gates ahead it is the distance to the next gate, plus the legs
 * between the remaining gates, plus ApproachDistance of the last gate.
 * @param config Episode configuration and runway pose.
 * @param mission Route and progress.
 * @param aircraft Aircraft state.
 * @return Remaining distance, in metres.
 */
double PathDistance(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept;

/**
 * @brief Weighted shaping potentials of every channel, zero for non-potential channels.
 *
 * - Path: weight * tolerance / (tolerance + PathDistance) in flight, the full
 *   weight on the ground, so neither passing the threshold nor rolling is penalized.
 * - VerticalSpeed: -weight * min(|vz - vz_target| / 2 m/s, 1) in flight, zero on the ground.
 * - Speed: -weight * min(|CAS - target CAS| / 10 m/s, 1) in flight, then
 *   weight * (1 - min(ground speed / flare speed, 1)) on the ground, which is
 *   continuous at touchdown and rises while the aircraft brakes to a stop.
 *
 * Absorbing terminal states carry no potential; a truncated Timeout keeps it.
 * @param config Active episode configuration.
 * @param mission Episode phase and terminal state.
 * @param aircraft Physical aircraft state at the same instant.
 * @return Weighted potential per reward channel.
 */
std::array<double, RewardCount> ShapingPotentials(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept;

/**
 * @brief Unweighted running cost of flying away from the commanded direction.
 *
 * cost = -dt * (1 - cos(angle between velocity and command)) / 2, so it lies in [-dt, 0].
 * @param guidance Guidance after the physics substep.
 * @param dt Physics substep, in seconds.
 * @return Non-positive cost; a vanishing velocity receives the largest penalty.
 */
double DirectionCost(Guidance const& guidance, double dt) noexcept;

/**
 * @brief Quality of the first wheel contact, applied to the Touchdown reward.
 * @param config Episode configuration and touchdown limits.
 * @param mission Mission state holding the measured touchdown point, sink rate and speed.
 * @return Product of the landing-area, touchdown-zone, sink and speed factors, within [0, 1].
 */
double TouchdownQuality(Config const& config, MissionState const& mission) noexcept;

/** @brief Weight of one reward channel. */
inline double Weight(RewardSettings const& settings, Reward channel) noexcept
{
    return settings.weights[static_cast<std::size_t>(channel)];
}

} // namespace ppo::examples::flight
