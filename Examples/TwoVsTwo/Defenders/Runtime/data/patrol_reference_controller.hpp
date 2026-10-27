#pragma once

// Deterministic reference follower for the visual patrol demonstration.
// Generates normalized SNA commands; physical integration remains owned by
// the existing Mobile/FlightModel systems (never teleports the vehicle).
#include "patrol_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ppo::examples::twovstwo::defenders {

/**
 * @brief Computes a deterministic reference action for one semantic role.
 * @param p Valid patrol and vehicle parameters.
 * @param mobile Physical state of the controlled guard.
 * @param leader Current leader state; equal to mobile for role zero.
 * @param side Scenario side retained for interface compatibility.
 * @param role Zero for racetrack leader, one for wedge wingman.
 * @param arclength Retained deterministic reference parameter.
 * @return Normalized yaw-rate and acceleration commands.
 */
inline Action PatrolReferenceAction(Parameters const& p,
                                    Mobile const& mobile,
                                    Mobile const& leader,
                                    int side,
                                    std::size_t role,
                                    float arclength) noexcept
{
    if (!Valid(p) || role >= 2U
        || !std::isfinite(mobile.north) || !std::isfinite(mobile.east)
        || !std::isfinite(mobile.heading) || !std::isfinite(mobile.speed)
        || !std::isfinite(leader.north) || !std::isfinite(leader.east)
        || !std::isfinite(leader.heading) || !std::isfinite(leader.speed)
        || !std::isfinite(arclength))
        return {};

    // Pure pursuit for the optional non-PPO demonstration only. The target is
    // derived from the vehicle's analytical projection, never from a clock.
    // The MAPPO actor does not call this controller.
    (void)side;
    (void)arclength;
    float const leaderLookahead = 3.F * std::max(mobile.speed, p.cruiseSpeed);
    Reference target{};
    float targetSpeed = p.cruiseSpeed;
    if (role == 0U)
    {
        auto const projection = ProjectOnPatrolCurve(
            p.geometry, mobile.north, mobile.east);
        target = PatrolReference(
            p, 1, projection.pose.arclength + leaderLookahead);
    }
    else
    {
        Reference const leaderReference{
            leader.north, leader.east, leader.heading, 0.F, 0.F};
        Reference const desired = WingmanReference(p, leaderReference);
        target = desired;
        float const wingmanLookahead = 3.F
            * std::max(mobile.speed, p.cruiseSpeed);
        target.north += std::cos(leader.heading) * wingmanLookahead;
        target.east += std::sin(leader.heading) * wingmanLookahead;
        float const formationError = std::hypot(
            desired.north - mobile.north, desired.east - mobile.east);
        targetSpeed = Clamp(
            leader.speed + 0.05F * formationError, 0.F, p.maxSpeed);
    }
    float const north = target.north - mobile.north;
    float const east = target.east - mobile.east;
    float const distance = std::max(std::hypot(north, east), 1.F);
    float const relativeBearing = std::remainder(
        std::atan2(east, north) - mobile.heading, kTwoPi);

    float const yawRate = 2.F * std::max(mobile.speed, 0.F)
        * std::sin(relativeBearing) / distance;
    float const acceleration = (targetSpeed - mobile.speed) / (4.F * p.dt);

    return {
        Clamp(yawRate / p.maxRotationRate, -1.F, 1.F),
        Clamp(acceleration / p.maxAcceleration, -1.F, 1.F)
    };
}

} // namespace ppo::examples::twovstwo::defenders
