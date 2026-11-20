/**
 * @file runway_geometry.cpp
 * @brief Implementation of the frame conversions and landable-surface tests.
 *
 * The deck is level, so the runway normal is the world up axis and the runway
 * axes are a rotation about it by the runway heading. The body axes follow the
 * aerospace convention: heading, then pitch, then bank about the nose.
 *
 * Read next: runway_geometry.hpp for the frame definitions.
 */
#include "runway_geometry.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Length below which a vector has no meaningful direction. */
constexpr double MinimumDirectionLength = 1e-6;
} // namespace

Vec3 Add(Vec3 first, Vec3 second) noexcept
{
    return {first.north + second.north, first.east + second.east, first.up + second.up};
}

Vec3 Subtract(Vec3 first, Vec3 second) noexcept
{
    return {first.north - second.north, first.east - second.east, first.up - second.up};
}

Vec3 Scale(Vec3 vector, double scale) noexcept
{
    return {vector.north * scale, vector.east * scale, vector.up * scale};
}

double Dot(Vec3 first, Vec3 second) noexcept
{
    return first.north * second.north + first.east * second.east + first.up * second.up;
}

Vec3 Cross(Vec3 first, Vec3 second) noexcept
{
    return {first.east * second.up - first.up * second.east, first.up * second.north - first.north * second.up,
        first.north * second.east - first.east * second.north};
}

double Length(Vec3 vector) noexcept
{
    return std::sqrt(Dot(vector, vector));
}

Vec3 UnitDirection(Vec3 vector) noexcept
{
    double const length = Length(vector);
    if (length > MinimumDirectionLength)
    {
        return Scale(vector, 1.0 / length);
    }
    return Vec3{};
}

Vec3 RunwayForward(Runway const& runway) noexcept
{
    double const headingRadians = runway.headingDegrees * DegreesToRadians;
    return {std::cos(headingRadians), std::sin(headingRadians), 0};
}

Vec3 RunwayRight(Runway const& runway) noexcept
{
    double const headingRadians = runway.headingDegrees * DegreesToRadians;
    return {-std::sin(headingRadians), std::cos(headingRadians), 0};
}

Vec3 RunwayNormal(Runway const&) noexcept
{
    return {0, 0, 1};
}

Vec3 RunwayAxes(Runway const& runway, Vec3 vector) noexcept
{
    return {Dot(vector, RunwayForward(runway)), Dot(vector, RunwayRight(runway)), Dot(vector,
        RunwayNormal(runway))};
}

Vec3 DeckCoordinates(Runway const& runway, Vec3 point) noexcept
{
    return RunwayAxes(runway, Subtract(point, runway.origin));
}

Vec3 FromRunway(Runway const& runway, double along, double cross, double up) noexcept
{
    auto const offset = Add(Scale(RunwayForward(runway), along),
        Add(Scale(RunwayRight(runway), cross), Scale(RunwayNormal(runway), up)));
    return Add(runway.origin, offset);
}

BodyAxes AircraftAxes(double roll, double pitch, double heading) noexcept
{
    Vec3 const forward{std::cos(pitch) * std::cos(heading), std::cos(pitch) * std::sin(heading),
        std::sin(pitch)};
    // Level right-wing and pitched top axes, then bank about the forward axis:
    // right = cos(roll) * levelRight - sin(roll) * pitchedUp, up = forward x right.
    Vec3 const levelRight{-std::sin(heading), std::cos(heading), 0};
    Vec3 const pitchedUp{-std::sin(pitch) * std::cos(heading), -std::sin(pitch) * std::sin(heading),
        std::cos(pitch)};
    Vec3 const right = Subtract(Scale(levelRight, std::cos(roll)), Scale(pitchedUp, std::sin(roll)));
    return {forward, right, Cross(forward, right)};
}

Vec3 InBodyAxes(BodyAxes const& axes, Vec3 vector) noexcept
{
    return {Dot(vector, axes.forward), Dot(vector, axes.right), Dot(vector, axes.up)};
}

bool OnLandingSurface(Runway const& runway, double along, double cross) noexcept
{
    // The landable surface is the runway rectangle grown by the margin on every side.
    return along >= -runway.landingMarginMeters && along <= runway.lengthMeters + runway.landingMarginMeters
        && std::abs(cross) <= runway.widthMeters / 2 + runway.landingMarginMeters;
}

double LandingRewardFactor(Runway const& runway, double along, double cross) noexcept
{
    // Chebyshev distance outside the paved rectangle: zero on the runway,
    // then the largest of the longitudinal and lateral overshoots.
    double const alongDistance = std::max({0.0, -along, along - runway.lengthMeters});
    double const crossDistance = std::max(0.0, std::abs(cross) - runway.widthMeters / 2);
    double const distance = std::max(alongDistance, crossDistance);
    if (distance <= 0)
    {
        return 1.0;
    }
    if (runway.landingMarginMeters <= 0)
    {
        return 0.0;
    }
    // Linear fade from one at the runway edge to zero at the margin boundary.
    return std::clamp(1.0 - distance / runway.landingMarginMeters, 0.0, 1.0);
}
} // namespace ppo::examples::flight
