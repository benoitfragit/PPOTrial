/**
 * @file runway_geometry.hpp
 * @brief Vector algebra and conversions between the world, runway and body frames.
 *
 * Role in the chain: pure geometry used by guidance, landing checks, rewards,
 * observations, spawns, the JSBSim ground callback and the renderer.
 *
 * Frames:
 * - world: local tangent plane at the configured runway reference point, (north, east, up);
 * - runway axes: origin at the threshold centre on the deck, (along, right, up), level deck;
 * - body axes: (forward, right wing, top) of the aircraft, from roll, pitch and heading.
 *
 * Not here: no state, no configuration parsing, no physics. Every function is noexcept.
 *
 * Read next: flight_guidance.hpp (approach reference), flight_landing.hpp (contact rules).
 */
#pragma once

#include "flight_config.hpp"

namespace ppo::examples::flight {

/** @brief Component-wise sum. */
Vec3 Add(Vec3 first, Vec3 second) noexcept;
/** @brief Component-wise difference a - b. */
Vec3 Subtract(Vec3 first, Vec3 second) noexcept;
/** @brief Product of a vector by a scalar. */
Vec3 Scale(Vec3 vector, double scale) noexcept;
/** @brief Dot product. */
double Dot(Vec3 first, Vec3 second) noexcept;
/** @brief Cross product in the north-east-up component order used by the runway and body axes. */
Vec3 Cross(Vec3 first, Vec3 second) noexcept;
/** @brief Euclidean length of a vector. */
double Length(Vec3 vector) noexcept;
/** @brief Unit vector, or the zero vector for a vector shorter than 1e-6. */
Vec3 UnitDirection(Vec3 vector) noexcept;
/** @brief Horizontal unit vector along the runway, from the threshold toward the far end. */
Vec3 RunwayForward(Runway const& runway) noexcept;
/** @brief Horizontal unit vector to the right of the runway centreline. */
Vec3 RunwayRight(Runway const& runway) noexcept;
/** @brief Upward unit normal of the level deck. */
Vec3 RunwayNormal(Runway const& runway) noexcept;
/**
 * @brief Express a world point in runway axes.
 * @param runway Runway pose.
 * @param point World position, in metres.
 * @return Distance past the threshold (north), right of the centreline (east) and above the deck (up).
 */
Vec3 DeckCoordinates(Runway const& runway, Vec3 point) noexcept;
/** @brief Express a world direction or velocity in runway axes (no translation). */
Vec3 RunwayAxes(Runway const& runway, Vec3 vector) noexcept;
/**
 * @brief Velocity relative to the deck, in runway axes.
 * @param runway Runway pose and deck velocity.
 * @param velocity World velocity, in metres per second.
 * @return Deck-relative velocity (along, right, up); equal to RunwayAxes on a static deck.
 */
Vec3 DeckRelativeVelocity(Runway const& runway, Vec3 velocity) noexcept;
/**
 * @brief Pose of a moving deck some time after the episode start.
 *
 * The threshold advances along the runway heading at the deck speed and heaves vertically:
 * origin(t) = start + forward * speed * t + up * amplitude * sin(2 pi t / period + phase),
 * velocity(t) = forward * speed + up * amplitude * (2 pi / period) * cos(2 pi t / period + phase).
 * @param start Mean deck pose at the episode start (its velocity is ignored).
 * @param motion Deck speed and swell of the stage.
 * @param swellPhase Swell phase of the episode, in radians.
 * @param seconds Time since the episode start, in seconds.
 * @return Deck pose and velocity at that time; the start pose with zero velocity for a static deck.
 */
Runway MovedDeck(Runway const& start, DeckMotionSettings const& motion, double swellPhase,
    double seconds) noexcept;
/**
 * @brief Convert runway-frame coordinates to a world position.
 * @param runway Runway pose.
 * @param along Distance past the threshold, in metres.
 * @param cross Offset right of the centreline, in metres.
 * @param up Height above the deck, in metres.
 * @return World position, in metres.
 */
Vec3 FromRunway(Runway const& runway, double along, double cross, double up) noexcept;

/** @brief Aircraft body axes expressed in world coordinates. */
struct BodyAxes final
{
    /** @brief Unit nose direction. */
    Vec3 forward{};
    /** @brief Unit right-wing direction. */
    Vec3 right{};
    /** @brief Unit canopy (top) direction. */
    Vec3 up{};
};
/**
 * @brief Body axes from Euler angles.
 * @param roll Bank angle, positive right wing down, in radians.
 * @param pitch Nose-up pitch angle, in radians.
 * @param heading True heading, clockwise from north, in radians.
 * @return Orthonormal forward, right-wing and top axes.
 */
BodyAxes AircraftAxes(double roll, double pitch, double heading) noexcept;
/** @brief Express a world vector in body axes (forward, right, up). */
Vec3 InBodyAxes(BodyAxes const& axes, Vec3 vector) noexcept;
/**
 * @brief Check if runway-relative coordinates lie on the landable surface.
 * @param runway Physical runway and surrounding margin.
 * @param along Distance along the runway from its threshold, in metres.
 * @param cross Lateral distance to the runway centreline, in metres.
 * @return True inside the runway or its extended surface.
 */
bool OnLandingSurface(Runway const& runway, double along, double cross) noexcept;
/**
 * @brief Calculate the touchdown reward factor from runway-relative contact coordinates.
 * @param runway Physical runway and surrounding margin.
 * @param along Distance along the runway from its threshold, in metres.
 * @param cross Lateral distance to the runway centreline, in metres.
 * @return One on the runway, decreasing linearly to zero at the extended surface boundary.
 */
double LandingRewardFactor(Runway const& runway, double along, double cross) noexcept;

} // namespace ppo::examples::flight
