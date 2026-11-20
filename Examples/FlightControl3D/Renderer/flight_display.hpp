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
inline float RewardProgressFraction(double episode,double stop,double crash) noexcept
{
    double const reference=episode<0 ? std::abs(crash) : std::abs(stop);
    double const alternate=episode<0 ? std::abs(stop) : std::abs(crash);
    double const scale=reference>0 ? reference : (alternate>0 ? alternate : 1.0);
    return static_cast<float>(std::clamp(std::abs(episode)/scale,0.0,1.0));
}

/**
 * @brief Positions the runway camera near the threshold, looking toward it.
 * @param runway Current runway pose and positive physical length.
 * @return Camera position in world coordinates, eight metres above the runway.
 */
inline Vec3 RunwayCameraPosition(Runway const& runway) noexcept
{
    constexpr double DistanceFromThresholdMeters=300.0;
    constexpr double CameraHeightMeters=8.0;
    return FromRunway(runway,std::min(DistanceFromThresholdMeters,runway.lengthMeters*0.5),
        0,CameraHeightMeters);
}

/**
 * @brief Zooms the fixed runway camera toward a distant aircraft.
 * @param camera Camera position in world metres.
 * @param aircraft Aircraft position in world metres.
 * @return Bounded vertical field of view in degrees.
 */
inline double RunwayCameraFieldOfView(Vec3 camera,Vec3 aircraft) noexcept
{
    double const distance=std::hypot(std::hypot(aircraft.north-camera.north,
        aircraft.east-camera.east),aircraft.up-camera.up);
    if(!std::isfinite(distance) || distance<1.0) return 50.0;
    return std::clamp(2.0*std::atan(50.0/distance)/DegreesToRadians,0.4,50.0);
}

/** Same deck-relative flight-path angle as the existing slope reward. */
inline double DisplaySlopeDegrees(Guidance const& guidance) noexcept
{
    return std::atan2(guidance.relativeVelocity.up,
        std::hypot(guidance.relativeVelocity.north,guidance.relativeVelocity.east))/DegreesToRadians;
}
} // namespace ppo::examples::flight
