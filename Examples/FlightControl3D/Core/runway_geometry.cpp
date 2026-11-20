#include "runway_geometry.hpp"
#include <algorithm>

namespace ppo::examples::flight {
Vec3 Add(Vec3 a, Vec3 b) noexcept { return {a.north + b.north, a.east + b.east, a.up + b.up}; }
Vec3 Subtract(Vec3 a, Vec3 b) noexcept { return {a.north - b.north, a.east - b.east, a.up - b.up}; }
Vec3 Scale(Vec3 a, double scale) noexcept { return {a.north * scale, a.east * scale, a.up * scale}; }
double Dot(Vec3 a, Vec3 b) noexcept { return a.north * b.north + a.east * b.east + a.up * b.up; }
Vec3 Cross(Vec3 a, Vec3 b) noexcept
{
    return {a.east * b.up - a.up * b.east, a.up * b.north - a.north * b.up, a.north * b.east - a.east * b.north};
}
double Length(Vec3 a) noexcept { return std::sqrt(Dot(a, a)); }

Vec3 UnitDirection(Vec3 vector) noexcept
{
    constexpr double MinimumLength = 1e-6;
    double const length = Length(vector);
    return length > MinimumLength ? Scale(vector, 1.0 / length) : Vec3{};
}

Vec3 RunwayForward(Runway const& runway) noexcept
{
    double const heading = runway.headingDegrees * DegreesToRadians;
    return {std::cos(heading), std::sin(heading), 0};
}

Vec3 RunwayRight(Runway const& runway) noexcept
{
    double const heading = runway.headingDegrees * DegreesToRadians;
    return {-std::sin(heading), std::cos(heading), 0};
}

Vec3 RunwayNormal(Runway const&) noexcept { return {0, 0, 1}; }

Vec3 RunwayAxes(Runway const& runway, Vec3 vector) noexcept
{
    return {Dot(vector, RunwayForward(runway)), Dot(vector, RunwayRight(runway)), Dot(vector, RunwayNormal(runway))};
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
    Vec3 const forward{std::cos(pitch) * std::cos(heading), std::cos(pitch) * std::sin(heading), std::sin(pitch)};
    // Level right-wing and pitched top axes, then bank about the forward axis.
    Vec3 const levelRight{-std::sin(heading), std::cos(heading), 0};
    Vec3 const pitchedUp{-std::sin(pitch) * std::cos(heading), -std::sin(pitch) * std::sin(heading), std::cos(pitch)};
    Vec3 const right = Subtract(Scale(levelRight, std::cos(roll)), Scale(pitchedUp, std::sin(roll)));
    return {forward, right, Cross(forward, right)};
}

Vec3 InBodyAxes(BodyAxes const& axes, Vec3 vector) noexcept
{
    return {Dot(vector, axes.forward), Dot(vector, axes.right), Dot(vector, axes.up)};
}

bool OnLandingSurface(Runway const& runway, double along, double cross) noexcept
{
    return along >= -runway.landingMarginMeters && along <= runway.lengthMeters + runway.landingMarginMeters
        && std::abs(cross) <= runway.widthMeters / 2 + runway.landingMarginMeters;
}

double LandingRewardFactor(Runway const& runway, double along, double cross) noexcept
{
    double const alongDistance = std::max({0.0, -along, along - runway.lengthMeters});
    double const crossDistance = std::max(0.0, std::abs(cross) - runway.widthMeters / 2);
    double const distance = std::max(alongDistance, crossDistance);
    if (distance <= 0) return 1.0;
    if (runway.landingMarginMeters <= 0) return 0.0;
    return std::clamp(1.0 - distance / runway.landingMarginMeters, 0.0, 1.0);
}
} // namespace ppo::examples::flight
