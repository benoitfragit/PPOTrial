#pragma once
#include "flight_mission.hpp"

namespace ppo::examples::flight {

/** Render-only ILS references, anchored to the instantaneous runway frame. */
struct IlsDisplayGeometry final
{
    Vec3 localizerOrigin{}, localizerEnd{}, localizerLeft{}, localizerRight{};
    Vec3 glideOrigin{}, glideEnd{}, glideLow{}, glideHigh{};
};

inline IlsDisplayGeometry MakeIlsDisplayGeometry(Runway const& runway, double range,
    double glideDegrees, double localizerCaptureDegrees, double glideCaptureDegrees) noexcept
{
    IlsDisplayGeometry result{};
    // Lift the ground reference slightly to avoid depth fighting with the deck.
    constexpr double GroundLineHeight = 0.2;
    double const halfWidth = range*std::tan(localizerCaptureDegrees*DegreesToRadians);
    result.localizerOrigin = FromRunway(runway,0,0,GroundLineHeight);
    result.localizerEnd = FromRunway(runway,-range,0,GroundLineHeight);
    result.localizerLeft = FromRunway(runway,-range,-halfWidth,GroundLineHeight);
    result.localizerRight = FromRunway(runway,-range,halfWidth,GroundLineHeight);
    result.glideOrigin = FromRunway(runway,0,0,runway.thresholdHeightMeters);
    result.glideEnd = FromRunway(runway,-range,0,
        runway.thresholdHeightMeters+range*std::tan(glideDegrees*DegreesToRadians));
    result.glideLow = FromRunway(runway,-range,0,
        runway.thresholdHeightMeters+range*std::tan((glideDegrees-glideCaptureDegrees)*DegreesToRadians));
    result.glideHigh = FromRunway(runway,-range,0,
        runway.thresholdHeightMeters+range*std::tan((glideDegrees+glideCaptureDegrees)*DegreesToRadians));
    return result;
}

/** Same deck-relative flight-path angle as the existing slope reward. */
inline double DisplaySlopeDegrees(Guidance const& guidance) noexcept
{
    return std::atan2(guidance.relativeVelocity.up,
        std::hypot(guidance.relativeVelocity.north,guidance.relativeVelocity.east))/DegreesToRadians;
}
} // namespace ppo::examples::flight
