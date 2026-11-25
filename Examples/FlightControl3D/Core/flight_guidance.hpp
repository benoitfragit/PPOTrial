/**
 * @file flight_guidance.hpp
 * @brief Approach, flare and rollout reference shared by rewards, observations and rendering.
 *
 * Role in the chain: between the aircraft state (JSBSim) and the learning signals. Guide()
 * turns the mission phase and the aircraft state into a commanded direction, a commanded
 * flight-path angle and a reference airspeed. The reward potentials, the observation and
 * the renderer all consume the same Guidance value, so the policy sees exactly what it is
 * rewarded for.
 *
 * Geometry: the glide plane crosses the threshold at Runway::thresholdHeightMeters and
 * meets the runway at the aim point, AimPointAlongMeters() past the threshold. Far out the
 * horizontal command intercepts the extended centreline InterceptLookaheadMeters ahead, at
 * most MaximumInterceptAngleDegrees off its axis; on short final it points at the aim
 * point; within CentrelineLookaheadMeters of it the target slides along the centreline.
 * The vertical command follows the glide plane. Below the flare height the descent blends
 * into a gentle contact sink. An episode may first fly a route of gates (a downwind, a
 * holding lap): holding uses the tangent of the active straight or circular leg plus
 * geometric position-error capture. Other routes point at the next gate or lead the track
 * by MaximumGateLeadDegrees during a turn. The approach starts once the last gate is passed.
 * On a moving deck every velocity is
 * deck-relative. All lengths are in metres, speeds in m/s and angles in radians.
 *
 * Not here: no controller. Guidance never commands a surface; it only defines the
 * reference the policy learns to follow.
 *
 * Read next: flight_guidance.cpp, flight_reward.hpp, flight_observation.hpp.
 */
#pragma once

#include "runway_geometry.hpp"

namespace ppo::examples::flight {

/**
 * @brief Shortest horizontal look-ahead, used near and past the aim point, in metres.
 *
 * Within this distance of the aim point the horizontal target slides along the
 * centreline so that a lateral offset is corrected smoothly instead of by a
 * sharp turn. The glide reference slides the same way.
 */
inline constexpr double CentrelineLookaheadMeters = 300.0;

/**
 * @brief Longest horizontal look-ahead, used far out to intercept the extended centreline, in metres.
 *
 * Following the command, a lateral offset decays roughly as
 * exp(-distance flown / look-ahead), so an aircraft started off the axis is
 * established on it long before the threshold.
 */
inline constexpr double InterceptLookaheadMeters = 1200.0;

/** @brief Largest angle between the commanded horizontal track and the runway axis, in degrees. */
inline constexpr double MaximumInterceptAngleDegrees = 30.0;

/**
 * @brief Largest horizontal angle between a gate command and the current track, in degrees.
 *
 * Toward a gate behind or abeam (the turns of a pattern) the command
 * leads the deck-relative track by at most this angle on the side of the gate, instead of
 * jumping by up to 180 degrees. Any turn rate then keeps the Direction cost below
 * weight * (1 - cos 10 deg) / 2 per second. With the T-6 scenario (Direction weight 2,
 * gamma 0.999 per 0.1 s action) that is 0.015/s, about 1.5 discounted over an unbounded
 * route, below the Crash penalty of 5: diving into the sea is never cheaper than turning,
 * as it was with a 90 to 180 degree command step at each turn.
 */
inline constexpr double MaximumGateLeadDegrees = 10.0;

/** @brief Horizontal radius within which a gate counts as passed, in metres. */
inline constexpr double GateRadiusMeters = 300.0;

/**
 * @brief Half-width of the corridor in which crossing a gate's plane also passes it, in metres.
 *
 * An aircraft that overshoots a gate a little still passes it once beyond the plane
 * normal to its inbound leg; farther off, the command turns it back toward the gate.
 */
inline constexpr double GateCorridorMeters = 2.0 * GateRadiusMeters;

/**
 * @brief Whether the aircraft has passed a route gate.
 * @param route Episode route.
 * @param index Gate index, below route.count.
 * @param along Aircraft distance past the threshold, in metres.
 * @param cross Aircraft offset right of the centreline, in metres.
 * @return True within GateRadiusMeters of the gate, or beyond the plane normal to its inbound
 *     leg and within GateCorridorMeters of the gate laterally.
 */
bool GatePassed(Route const& route, std::size_t index, double along, double cross) noexcept;

/** @brief World position of the runway threshold centre on the deck. */
Vec3 ThresholdPoint(Runway const& runway) noexcept;

/**
 * @brief Distance past the threshold where the approach plane meets the runway.
 * @param config Episode configuration: threshold crossing height and glide slope.
 * @return Aim-point distance, in metres: thresholdHeightMeters / tan(glide slope).
 */
double AimPointAlongMeters(Config const& config) noexcept;

/** @brief World position of the touchdown aim point on the runway centreline. */
Vec3 AimPoint(Config const& config) noexcept;

/**
 * @brief Height of the approach plane above the deck, for display and diagnostics.
 * @param config Episode configuration.
 * @param along Distance past the threshold, in metres.
 * @return Glide-plane height, in metres, zero at and beyond the aim point.
 */
double ApproachPathHeight(Config const& config, double along) noexcept;

/** @brief Target point and commanded unit direction in world axes. */
struct ApproachReference final
{
    /** @brief Centreline point the horizontal command converges toward, in world metres. */
    Vec3 target{};
    /** @brief Unit commanded direction, in world axes. */
    Vec3 direction{};
    /** @brief Commanded flight-path angle, in radians (atan of the vertical gradient). */
    double slope{};
};

/**
 * @brief Commanded flight direction shared by guidance, rewards, observations and rendering.
 * @param config Episode configuration and runway pose.
 * @param mission Current phase.
 * @param aircraft Current aircraft state.
 * @return While holding gates remain, tangent/capture direction on the active leg; for other
 *     routes, toward the next gate and its height; then toward the
 *     extended centreline far out, toward the aim point on short final, along the centreline
 *     near and past it, descending on the glide plane, with a gentle flare sink below the
 *     flare height and a level direction while rolling.
 */
ApproachReference ResolveApproachReference(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept;

/**
 * @brief Commanded vertical speed: the current horizontal speed along the commanded flight-path angle.
 * @param guidance Guidance holding the deck-relative velocity and the commanded slope.
 * @return Target vertical speed, in metres per second, negative when descending.
 */
double TargetVerticalSpeed(Guidance const& guidance) noexcept;

/**
 * @brief Resolves the active approach, flare or rollout reference.
 * @param config Episode configuration and runway pose.
 * @param mission Current phase and outcome.
 * @param aircraft Current finite aircraft state.
 * @return Value-only guidance consumed by observations, rewards and rendering.
 */
Guidance Guide(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept;

} // namespace ppo::examples::flight
