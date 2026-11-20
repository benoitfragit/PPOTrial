#include "flight_guidance.hpp"
#include "flight_t6_performance.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Steepest commanded descent or climb gradient, tan(6 degrees). */
double MaximumPathGradient() noexcept { return std::tan(6.0 * DegreesToRadians); }

/** @brief Cubic smoothstep of a fraction clamped to [0, 1]. */
double SmoothBlend(double fraction) noexcept
{
    double const t = std::clamp(fraction, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/** @brief Gentle sink rate commanded at wheel contact, well inside the touchdown limit. */
double ContactSinkMps(Config const& config) noexcept
{
    return std::min(0.5, 0.5 * config.mission.maxTouchdownSink);
}

/**
 * @brief Reference calibrated airspeed for the current phase and position.
 * @param config Episode configuration.
 * @param mission Current phase.
 * @param aircraft Current aircraft state.
 * @param along Distance past the threshold, in metres.
 * @return Approach speed far out, flare speed on short final, zero once rolling on the wheels.
 */
double TargetSpeed(Config const& config, MissionState const& mission, AircraftState const& aircraft,
    double along) noexcept
{
    if (mission.phase == Phase::Rollout && aircraft.weightOnWheels) return 0;
    // Decelerate smoothly from the approach speed to the flare speed between
    // 2.7 km and 2 km before the threshold, never below the stall reserve.
    constexpr double FlareSpeedReachedAlongMeters = -2000.0;
    constexpr double DecelerationLengthMeters = 700.0;
    auto const& m = config.mission;
    double const blend = SmoothBlend((FlareSpeedReachedAlongMeters - along) / DecelerationLengthMeters);
    double const requested = m.flareSpeed + (m.approachSpeed - m.flareSpeed) * blend;
    return std::max(requested, T6MinimumCalibratedSpeed(aircraft.flapPosition));
}

/**
 * @brief Vertical gradient commanded in flight: toward the target on the deck, flaring near the ground.
 * @param config Episode configuration.
 * @param height Height above the deck, in metres.
 * @param horizontalDistance Horizontal distance to the target point, in metres.
 * @param horizontalSpeed Current deck-relative horizontal speed, in metres per second.
 * @return dh/ds along the horizontal track.
 */
double AirborneGradient(Config const& config, double height, double horizontalDistance,
    double horizontalSpeed) noexcept
{
    double const limit = MaximumPathGradient();
    double const towardTarget = std::clamp(-height / horizontalDistance, -limit, limit);
    // Below the flare height the descent blends toward a gentle, strictly
    // negative contact sink, so the wheels still meet the deck.
    double const flareHeight = config.mission.flareHeightMeters;
    double const flare = SmoothBlend((flareHeight - height) / flareHeight);
    double const contactGradient = -ContactSinkMps(config) / std::max(horizontalSpeed, config.mission.flareSpeed);
    return towardTarget + (contactGradient - towardTarget) * flare;
}
} // namespace

Vec3 ThresholdPoint(Runway const& runway) noexcept
{
    return FromRunway(runway, 0, 0, 0);
}

double AimPointAlongMeters(Config const& config) noexcept
{
    return config.runway.thresholdHeightMeters / std::tan(config.mission.glideDegrees * DegreesToRadians);
}

Vec3 AimPoint(Config const& config) noexcept
{
    return FromRunway(config.runway, AimPointAlongMeters(config), 0, 0);
}

double ApproachPathHeight(Config const& config, double along) noexcept
{
    double const remaining = std::max(0.0, AimPointAlongMeters(config) - along);
    return remaining * std::tan(config.mission.glideDegrees * DegreesToRadians);
}

ApproachReference ResolveApproachReference(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept
{
    auto const& runway = config.runway;
    auto const local = DeckCoordinates(runway, aircraft.position);
    // Far out the horizontal target is the aim point itself; closer in it
    // slides along the centreline so the track converges without a sharp turn.
    double const targetAlong = std::max(AimPointAlongMeters(config), local.north + CentrelineLookaheadMeters);
    double const targetForward = targetAlong - local.north;
    double const targetRight = -local.east;
    double const horizontalDistance = std::hypot(targetForward, targetRight);
    double gradient = 0;
    if (mission.phase == Phase::Approach)
    {
        auto const velocity = RunwayAxes(runway, aircraft.velocity);
        gradient = AirborneGradient(config, local.up, horizontalDistance, std::hypot(velocity.north, velocity.east));
    }
    auto const runwayDirection = Add(Add(Scale(RunwayForward(runway), targetForward / horizontalDistance),
        Scale(RunwayRight(runway), targetRight / horizontalDistance)), Scale(RunwayNormal(runway), gradient));
    ApproachReference reference{};
    reference.target = FromRunway(runway, targetAlong, 0, 0);
    reference.direction = UnitDirection(runwayDirection);
    reference.slope = std::atan(gradient);
    return reference;
}

double TargetVerticalSpeed(Guidance const& guidance) noexcept
{
    double const horizontalSpeed = std::hypot(guidance.relativeVelocity.north, guidance.relativeVelocity.east);
    return horizontalSpeed * std::tan(guidance.targetSlope);
}

Guidance Guide(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept
{
    auto const& runway = config.runway;
    Guidance g{};
    auto const local = DeckCoordinates(runway, aircraft.position);
    g.along = local.north;
    g.cross = local.east;
    g.height = local.up;
    g.distance = std::hypot(g.along, g.cross);
    g.relativeVelocity = RunwayAxes(runway, aircraft.velocity);
    g.velocityDirection = UnitDirection(aircraft.velocity);
    g.targetSpeed = TargetSpeed(config, mission, aircraft, g.along);
    bool const lowOverRunway = g.height >= 0 && g.height <= config.mission.flareHeightMeters
        && OnLandingSurface(runway, g.along, g.cross);
    if (mission.phase == Phase::Rollout) g.mode = GuidanceMode::Rollout;
    else g.mode = lowOverRunway ? GuidanceMode::Flare : GuidanceMode::Approach;
    auto const reference = ResolveApproachReference(config, mission, aircraft);
    g.target = reference.target;
    g.targetValid = mission.outcome == Outcome::Flying;
    g.targetSlope = reference.slope;
    g.targetDirection = reference.direction;
    g.targetHeading = std::atan2(reference.direction.east, reference.direction.north);
    return g;
}
} // namespace ppo::examples::flight
