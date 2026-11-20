#pragma once
#include "flight_config.hpp"
#include <array>

namespace ppo::examples::flight {
Vec3 Add(Vec3 a, Vec3 b) noexcept;
Vec3 Subtract(Vec3 a, Vec3 b) noexcept;
Vec3 Scale(Vec3 a, double scale) noexcept;
double Dot(Vec3 a, Vec3 b) noexcept;
/** @brief Cross product in the north-east-up component order used by the runway and body axes. */
Vec3 Cross(Vec3 a, Vec3 b) noexcept;
/** @brief Euclidean length of a vector. */
double Length(Vec3 a) noexcept;
/** @brief Unit vector, or the zero vector for a vanishing input. */
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
 * @param point World position.
 * @return Distance past the threshold (north), right of the centreline (east) and above the deck (up).
 */
Vec3 DeckCoordinates(Runway const& runway, Vec3 point) noexcept;
/** @brief Express a world direction in runway axes. */
Vec3 RunwayAxes(Runway const& runway, Vec3 vector) noexcept;
/** @brief Convert runway-frame coordinates to a world position. */
Vec3 FromRunway(Runway const& runway, double along, double cross, double up) noexcept;

/** @brief Aircraft body axes expressed in world coordinates. */
struct BodyAxes final { Vec3 forward{}, right{}, up{}; };
/**
 * @brief Body axes from Euler angles.
 * @param roll Bank angle, positive right wing down, in radians.
 * @param pitch Nose-up pitch angle, in radians.
 * @param heading True heading, clockwise from north, in radians.
 * @return Orthonormal forward, right-wing and top axes.
 */
BodyAxes AircraftAxes(double roll, double pitch, double heading) noexcept;
/** @brief Express a world vector in body axes. */
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
