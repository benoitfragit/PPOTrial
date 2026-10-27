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

    // Pure pursuit for the optional non-PPO demonstration only. The target is
    // derived from the vehicle's analytical projection, never from a clock.
    // The MAPPO actor does not call this controller.
    (void)side;
    (void)arclength;
    float const lookahead = 3.F * std::max(mobile.speed, p.cruiseSpeed);
    auto const projection = ProjectOnPatrolCurve(
        p.geometry, role, mobile.north, mobile.east);
    auto const target = PatrolReference(
        p, 1, role, projection.pose.arclength + lookahead);
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
