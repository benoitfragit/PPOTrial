#pragma once

// Deterministic reference follower for the visual patrol demonstration.
// Generates normalized SNA commands; physical integration remains owned by
// the existing Mobile/FlightModel systems (never teleports the vehicle).
#include "patrol_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ppo::examples::twovstwo::defenders {

inline Action PatrolReferenceAction(Parameters const& p,
                                    Mobile const& mobile,
                                    int side,
                                    std::size_t role,
                                    float arclength) noexcept
{
    if (!Valid(p) || role >= 2U
        || !std::isfinite(mobile.north) || !std::isfinite(mobile.east)
        || !std::isfinite(mobile.heading) || !std::isfinite(mobile.speed)
        || !std::isfinite(arclength))
        return {};

    // Pure pursuit: geometric lookahead and bicycle-like yaw-rate guidance.
    // At cruise speed the lookahead stays close to 3 seconds of travel.
    float const lookahead = 3.F * std::max(mobile.speed, p.cruiseSpeed);
    auto const target = PatrolReference(p, side, role, arclength + lookahead);
    float const north = target.north - mobile.north;
    float const east = target.east - mobile.east;
    float const distance = std::max(std::hypot(north, east), 1.F);
    float const relativeBearing = std::remainder(
        std::atan2(east, north) - mobile.heading, kTwoPi);

    float const yawRate = 2.F * std::max(mobile.speed, 0.F)
        * std::sin(relativeBearing) / distance;
    float const acceleration = (p.cruiseSpeed - mobile.speed) / (4.F * p.dt);

    return {
        Clamp(yawRate / p.maxRotationRate, -1.F, 1.F),
        Clamp(acceleration / p.maxAcceleration, -1.F, 1.F)
    };
}

} // namespace ppo::examples::twovstwo::defenders
