/**
 * @file flight_display.hpp
 * @brief Pure display helpers of the renderer, kept header-only so they are unit-tested
 *        without a window (tests/display_tests.cpp, tests/core_tests.cpp).
 *
 * Role in the chain: snapshot -> **display helpers** -> renderer. No raylib or ImGui
 * dependency here.
 *
 * Read next: flight_view.cpp, flight_reward_view.cpp.
 */
#pragma once

#include "runway_geometry.hpp"
#include <algorithm>

namespace ppo::examples::flight {

/**
 * @brief Normalize a signed episode reward against Stop or the magnitude of Crash.
 * @param episode Signed accumulated channel reward.
 * @param stop Current Stop weight.
 * @param crash Current Crash weight.
 * @return A bounded bar fraction; if a reference is zero, use the other or one.
 */
inline float RewardProgressFraction(double episode, double stop, double crash) noexcept
{
    // Negative totals are compared with the crash penalty, positive ones with the stop reward.
    double reference = std::abs(stop);
    double alternate = std::abs(crash);
    if (episode < 0)
    {
        reference = std::abs(crash);
        alternate = std::abs(stop);
    }
    double scale = 1.0;
    if (reference > 0)
    {
        scale = reference;
    }
    else if (alternate > 0)
    {
        scale = alternate;
    }
    return static_cast<float>(std::clamp(std::abs(episode) / scale, 0.0, 1.0));
}

/**
 * @brief Positions the runway camera near the threshold, looking toward it.
 * @param runway Current runway pose and positive physical length.
 * @return Camera position in world coordinates, eight metres above the runway.
 */
inline Vec3 RunwayCameraPosition(Runway const& runway) noexcept
{
    constexpr double DistanceFromThresholdMeters = 300.0;
    constexpr double CameraHeightMeters = 8.0;
    return FromRunway(runway, std::min(DistanceFromThresholdMeters, runway.lengthMeters * 0.5),
        0, CameraHeightMeters);
}

/**
 * @brief Zooms the fixed runway camera toward a distant aircraft.
 *
 * The field of view frames a 100 m wide region around the aircraft: 2 atan(50 m / distance),
 * clamped to [0.4, 50] degrees.
 * @param camera Camera position in world metres.
 * @param aircraft Aircraft position in world metres.
 * @return Bounded vertical field of view in degrees.
 */
inline double RunwayCameraFieldOfView(Vec3 camera, Vec3 aircraft) noexcept
{
    constexpr double WidestFieldOfViewDegrees = 50.0;
    constexpr double NarrowestFieldOfViewDegrees = 0.4;
    constexpr double FramedHalfWidthMeters = 50.0;
    constexpr double MinimumDistanceMeters = 1.0;
    double const distance = std::hypot(std::hypot(aircraft.north - camera.north,
        aircraft.east - camera.east), aircraft.up - camera.up);
    if (!std::isfinite(distance) || distance < MinimumDistanceMeters)
    {
        return WidestFieldOfViewDegrees;
    }
    return std::clamp(2.0 * std::atan(FramedHalfWidthMeters / distance) / DegreesToRadians,
        NarrowestFieldOfViewDegrees, WidestFieldOfViewDegrees);
}

/** @brief Deck-relative flight-path angle, in degrees, positive when climbing. */
inline double DisplaySlopeDegrees(Guidance const& guidance) noexcept
{
    return std::atan2(guidance.relativeVelocity.up,
        std::hypot(guidance.relativeVelocity.north, guidance.relativeVelocity.east)) / DegreesToRadians;
}

} // namespace ppo::examples::flight
