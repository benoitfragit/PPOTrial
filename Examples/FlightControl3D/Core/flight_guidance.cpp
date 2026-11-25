/**
 * @file flight_guidance.cpp
 * @brief Construction of the approach reference: aim point, lookahead, gradient, flare and speed.
 *
 * Summary of the guidance law (all quantities in runway axes):
 * - while holding gates remain: follow the straight legs and exact circular arcs using a
 *   tangent vector and position-error correction; other routes retain the gate lead law;
 *   vertical guidance is bounded to +/- tan(6 degrees); the rules below then apply;
 * - horizontal direction: toward the extended centreline a look-ahead L ahead,
 *   L = clamp(aim along - along, 300 m, 1.2 km), at most 30 degrees off the runway axis.
 *   Far out the aircraft intercepts the centreline 1.2 km ahead and is established on it
 *   well before the threshold; on short final it heads for the aim point; near and past
 *   the aim point the target slides 300 m ahead;
 * - vertical gradient toward the glide target on the deck (the aim point, or a point
 *   300 m ahead near and past it), -height / horizontal distance, bounded to
 *   +/- tan(6 degrees);
 * - below the flare height, a smoothstep blend toward the contact sink gradient
 *   -contactSink / max(horizontal speed, flare speed);
 * - reference airspeed: smoothstep deceleration from the approach speed to the flare speed
 *   between 2.7 km and 2 km before the threshold, never below 1.3 times the stall speed of
 *   the current flap position, and zero once rolling on the wheels.
 *
 * Read next: flight_reward.cpp (how the reference is rewarded), flight_aircraft_performance.hpp.
 */
#include "flight_guidance.hpp"
#include "flight_aircraft_performance.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Steepest commanded descent or climb angle, in degrees (twice a 3-degree glide). */
constexpr double MaximumPathAngleDegrees = 6.0;
/** @brief Upper bound of the contact sink rate, in metres per second (a gentle touchdown). */
constexpr double MaximumContactSinkMps = 0.5;
/** @brief Contact sink as a fraction of the structural touchdown sink limit. */
constexpr double ContactSinkFractionOfLimit = 0.5;
/** @brief Along-track position where the flare speed is reached, in metres (2 km before the threshold). */
constexpr double FlareSpeedReachedAlongMeters = -2000.0;
/** @brief Length of the smooth deceleration to the flare speed, in metres. */
constexpr double DecelerationLengthMeters = 700.0;

/** @brief Steepest commanded descent or climb gradient, tan(6 degrees). */
double MaximumPathGradient() noexcept
{
    return std::tan(MaximumPathAngleDegrees * DegreesToRadians);
}

/** @brief Cubic smoothstep 3t^2 - 2t^3 of a fraction clamped to [0, 1]. */
double SmoothBlend(double fraction) noexcept
{
    double const clampedFraction = std::clamp(fraction, 0.0, 1.0);
    return clampedFraction * clampedFraction * (3.0 - 2.0 * clampedFraction);
}

/** @brief Gentle sink rate commanded at wheel contact, well inside the touchdown limit, in m/s. */
double ContactSinkMps(Config const& config) noexcept
{
    return std::min(MaximumContactSinkMps, ContactSinkFractionOfLimit * config.mission.maxTouchdownSink);
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
    if (mission.phase == Phase::Rollout && aircraft.weightOnWheels)
    {
        return 0;
    }
    // Decelerate smoothly from the approach speed to the flare speed between
    // 2.7 km and 2 km before the threshold, never below the stall reserve.
    auto const& missionSettings = config.mission;
    double const blend = SmoothBlend((FlareSpeedReachedAlongMeters - along) / DecelerationLengthMeters);
    double const requested = missionSettings.flareSpeed
        + (missionSettings.approachSpeed - missionSettings.flareSpeed) * blend;
    return std::max(requested, MinimumCalibratedSpeed(config.performance, aircraft.flapPosition));
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
    double const contactGradient = -ContactSinkMps(config) / std::max(horizontalSpeed,
        config.mission.flareSpeed);
    return towardTarget + (contactGradient - towardTarget) * flare;
}

/** @brief Holding capture length: smaller values converge more aggressively to the path. */
constexpr double HoldingCaptureMeters = 500.0;
/** @brief Minimum usable length of a commanded horizontal vector. */
constexpr double MinimumHoldingDirectionLength = 1e-6;

/** @brief A point on the active holding leg and its forward unit tangent, in runway axes. */
struct HoldingProjection final
{
    Vec3 point{};
    Vec3 tangent{};
};

/**
 * @brief Project onto a straight holding leg or its exact circular arc.
 * @param route Holding geometry generated by SampleHoldingStage.
 * @param index Destination gate of the leg.
 * @param local Aircraft position in runway axes.
 * @return Closest point on the finite leg and its tangent in the direction of flight.
 */
HoldingProjection ProjectHoldingLeg(Route const& route, std::size_t index, Vec3 const& local) noexcept
{
    constexpr std::size_t OutboundGate = HoldingTurnSegments + 1;
    constexpr std::size_t FinalGate = HoldingRouteGateCount - 1;
    auto const& gate = route.gates[index];
    if (index == 0 || index == OutboundGate || index == FinalGate)
    {
        // Capture the inbound centreline, not the diagonal from the random spawn.
        double const from = index == 0 ? route.start.alongMeters : route.gates[index - 1].alongMeters;
        double const along = std::clamp(local.north, std::min(from, gate.alongMeters),
            std::max(from, gate.alongMeters));
        double const sign = index == OutboundGate ? -1.0 : 1.0;
        return {{along, gate.crossMeters, gate.heightMeters}, {sign, 0.0, 0.0}};
    }

    bool const firstTurn = index <= HoldingTurnSegments;
    std::size_t const entryIndex = firstTurn ? 0 : OutboundGate;
    auto const& entry = route.gates[entryIndex];
    auto const& exit = route.gates[entryIndex + HoldingTurnSegments];
    double const halfCross = (exit.crossMeters - entry.crossMeters) / 2.0;
    double const radius = std::abs(halfCross);
    double const centreCross = entry.crossMeters + halfCross;
    double const alongSign = firstTurn ? 1.0 : -1.0;
    double const crossSign = std::copysign(1.0, halfCross);
    double const u = alongSign * (local.north - entry.alongMeters);
    double const v = -crossSign * (local.east - centreCross);
    double const interval = Pi / static_cast<double>(HoldingTurnSegments);
    double const lower = static_cast<double>(index - entryIndex - 1) * interval;
    double const upper = lower + interval;
    double angle = std::atan2(u, v);
    if (angle < lower || angle > upper)
    {
        // Outside this arc, pick the nearer endpoint, including across atan2's branch cut.
        double const lowerDot = u * std::sin(lower) + v * std::cos(lower);
        double const upperDot = u * std::sin(upper) + v * std::cos(upper);
        angle = lowerDot >= upperDot ? lower : upper;
    }
    return {{entry.alongMeters + alongSign * radius * std::sin(angle),
        centreCross - crossSign * radius * std::cos(angle), gate.heightMeters},
        {alongSign * std::cos(angle), crossSign * std::sin(angle), 0.0}};
}

/**
 * @brief Geometric holding guidance, independent of the aircraft's instantaneous track.
 * @param config Episode configuration and runway pose.
 * @param mission Active ordered holding route.
 * @param aircraft Current aircraft state.
 * @return Active gate for display, continuous tangent/capture direction and bounded slope.
 */
ApproachReference HoldingReference(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept
{
    auto const& runway = config.runway;
    auto const local = DeckCoordinates(runway, aircraft.position);
    auto const& route = mission.route;
    auto const& gate = route.gates[mission.nextGate];
    auto projection = ProjectHoldingLeg(route, mission.nextGate, local);
    if (mission.nextGate > 0)
    {
        // Gate passage has a finite radius and can advance before the geometric junction.
        // Keep the preceding leg eligible for projection, never a future leg or another lap.
        auto const previous = ProjectHoldingLeg(route, mission.nextGate - 1, local);
        auto const previousError = Subtract(previous.point, local);
        auto const currentError = Subtract(projection.point, local);
        if (Dot(previousError, previousError) < Dot(currentError, currentError))
        {
            projection = previous;
        }
    }

    auto const error = Subtract(projection.point, local);
    auto horizontal = Add(projection.tangent,
        Scale(Vec3{error.north, error.east, 0.0}, 1.0 / HoldingCaptureMeters));
    if (Dot(horizontal, projection.tangent) <= 0.0)
    {
        // Far beyond a missed leg, the endpoint correction can cancel the tangent.
        // Recover toward the active gate instead of producing a zero or reversed field.
        horizontal = {gate.alongMeters - local.north, gate.crossMeters - local.east, 0.0};
    }
    if (Length(horizontal) < MinimumHoldingDirectionLength)
    {
        horizontal = projection.tangent;
    }
    horizontal = UnitDirection(horizontal);

    double const limit = MaximumPathGradient();
    double const gradient = std::clamp(error.up / HoldingCaptureMeters, -limit, limit);
    auto const world = Add(Add(Scale(RunwayForward(runway), horizontal.north),
        Scale(RunwayRight(runway), horizontal.east)), Scale(RunwayNormal(runway), gradient));
    return {FromRunway(runway, gate.alongMeters, gate.crossMeters, gate.heightMeters),
        UnitDirection(world), std::atan(gradient)};
}

/** @brief Horizontal distance below which the gate direction is not resolved, in metres. */
constexpr double MinimumGateDistanceMeters = 1.0;
/** @brief Deck-relative ground speed below which the track is undefined and the gate is pointed at directly. */
constexpr double MinimumTrackSpeedMps = 1.0;

/** @brief Horizontal unit direction in runway axes. */
struct HorizontalDirection final
{
    double forward{};
    double right{};
};

/**
 * @brief Horizontal command toward a gate, leading the current track by at most
 *        MaximumGateLeadDegrees.
 * @param runway Runway pose and deck velocity.
 * @param aircraft Current aircraft state.
 * @param gateForward Along-runway component of the unit direction to the gate.
 * @param gateRight Right component of the unit direction to the gate.
 * @return The gate direction when it lies within the lead of the track, otherwise the track
 *     rotated by the lead toward the gate.
 */
HorizontalDirection LeadTrack(Runway const& runway, AircraftState const& aircraft, double gateForward,
    double gateRight) noexcept
{
    HorizontalDirection const towardGate{gateForward, gateRight};
    // Gate bearings are in runway axes; the deck-relative track must use the same frame.
    auto const velocity = DeckRelativeVelocity(runway, aircraft.velocity);
    if (std::hypot(velocity.north, velocity.east) < MinimumTrackSpeedMps)
    {
        return towardGate;
    }
    double const track = std::atan2(velocity.east, velocity.north);
    double const offset = WrapAngle(std::atan2(gateRight, gateForward) - track);
    double const lead = MaximumGateLeadDegrees * DegreesToRadians;
    if (std::abs(offset) <= lead)
    {
        return towardGate;
    }
    // A gate behind or abeam is reached by a turn: lead the track on the side of the gate.
    double const heading = track + std::copysign(lead, offset);
    return {std::cos(heading), std::sin(heading)};
}

/**
 * @brief Command toward a route gate: at its horizontal position, or leading the track toward
 *        it during a turn (LeadTrack), climbing or descending toward its height at no more
 *        than the steepest path angle.
 * @param config Episode configuration and runway pose.
 * @param gate Next gate of the route.
 * @param aircraft Current aircraft state.
 * @return Gate point, unit direction and commanded flight-path angle.
 */
ApproachReference GateReference(Config const& config, Gate const& gate, AircraftState const& aircraft) noexcept
{
    auto const& runway = config.runway;
    auto const local = DeckCoordinates(runway, aircraft.position);
    double const forward = gate.alongMeters - local.north;
    double const right = gate.crossMeters - local.east;
    double const horizontalDistance = std::max(std::hypot(forward, right), MinimumGateDistanceMeters);
    double const limit = MaximumPathGradient();
    double const gradient = std::clamp((gate.heightMeters - local.up) / horizontalDistance, -limit, limit);
    auto const track = LeadTrack(runway, aircraft, forward / horizontalDistance, right / horizontalDistance);
    auto const horizontalForward = Scale(RunwayForward(runway), track.forward);
    auto const horizontalRight = Scale(RunwayRight(runway), track.right);
    ApproachReference reference{};
    reference.target = FromRunway(runway, gate.alongMeters, gate.crossMeters, gate.heightMeters);
    reference.direction = UnitDirection(Add(Add(horizontalForward, horizontalRight),
        Scale(RunwayNormal(runway), gradient)));
    reference.slope = std::atan(gradient);
    return reference;
}
} // namespace

bool GatePassed(Route const& route, std::size_t index, double along, double cross) noexcept
{
    auto const& gate = route.gates[index];
    double const toAircraftAlong = along - gate.alongMeters;
    double const toAircraftCross = cross - gate.crossMeters;
    if (std::hypot(toAircraftAlong, toAircraftCross) <= GateRadiusMeters)
    {
        return true;
    }
    // Inbound leg: from the previous gate, or from the route start for the first gate.
    Gate const& from = index == 0 ? route.start : route.gates[index - 1];
    double const legAlong = gate.alongMeters - from.alongMeters;
    double const legCross = gate.crossMeters - from.crossMeters;
    double const legLength = std::hypot(legAlong, legCross);
    if (legLength <= 0)
    {
        return false;
    }
    // Signed distance beyond the gate plane and lateral offset from the gate, along the leg axes.
    double const beyond = (toAircraftAlong * legAlong + toAircraftCross * legCross) / legLength;
    double const lateral = std::abs(toAircraftCross * legAlong - toAircraftAlong * legCross) / legLength;
    return beyond >= 0 && lateral <= GateCorridorMeters;
}

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
    // A route is flown gate by gate before the approach starts.
    if (mission.phase == Phase::Approach && mission.nextGate < mission.route.count)
    {
        if (mission.route.holding && mission.route.count == HoldingRouteGateCount)
        {
            return HoldingReference(config, mission, aircraft);
        }
        return GateReference(config, mission.route.gates[mission.nextGate], aircraft);
    }
    auto const& runway = config.runway;
    auto const local = DeckCoordinates(runway, aircraft.position);
    double const aimAlong = AimPointAlongMeters(config);
    // Horizontal reference: a point of the extended centreline a look-ahead
    // ahead. The look-ahead shrinks from 1.2 km far out to the aim point on short
    // final, then to 300 m near and past it, so a lateral offset decays well
    // before the threshold instead of being flown straight to the aim point.
    double const lookahead = std::clamp(aimAlong - local.north, CentrelineLookaheadMeters,
        InterceptLookaheadMeters);
    double const targetAlong = local.north + lookahead;
    // A large offset is intercepted at a bounded angle, as on a localizer.
    double const maximumRight = lookahead * std::tan(MaximumInterceptAngleDegrees * DegreesToRadians);
    double const targetRight = std::clamp(-local.east, -maximumRight, maximumRight);
    double const horizontalLength = std::hypot(lookahead, targetRight);
    // The rollout reference is level; only the airborne phase commands a gradient.
    double gradient = 0;
    if (mission.phase == Phase::Approach)
    {
        // The vertical reference stays on the glide plane: its distance is
        // measured to the aim point, or to the point sliding 300 m ahead near it.
        double const glideTargetAlong = std::max(aimAlong, local.north + CentrelineLookaheadMeters);
        double const glideDistance = std::hypot(glideTargetAlong - local.north, local.east);
        auto const velocity = DeckRelativeVelocity(runway, aircraft.velocity);
        gradient = AirborneGradient(config, local.up, glideDistance, std::hypot(velocity.north,
            velocity.east));
    }
    // Direction = horizontal unit vector toward the intercept plus the gradient
    // along the deck normal, then normalized.
    auto const horizontalForward = Scale(RunwayForward(runway), lookahead / horizontalLength);
    auto const horizontalRight = Scale(RunwayRight(runway), targetRight / horizontalLength);
    auto const runwayDirection = Add(Add(horizontalForward, horizontalRight), Scale(RunwayNormal(runway),
        gradient));
    ApproachReference reference{};
    reference.target = FromRunway(runway, targetAlong, 0, 0);
    reference.direction = UnitDirection(runwayDirection);
    reference.slope = std::atan(gradient);
    return reference;
}

double TargetVerticalSpeed(Guidance const& guidance) noexcept
{
    double const horizontalSpeed = std::hypot(guidance.relativeVelocity.north,
        guidance.relativeVelocity.east);
    return horizontalSpeed * std::tan(guidance.targetSlope);
}

Guidance Guide(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept
{
    auto const& runway = config.runway;
    Guidance guidance{};
    auto const local = DeckCoordinates(runway, aircraft.position);
    guidance.along = local.north;
    guidance.cross = local.east;
    guidance.height = local.up;
    guidance.distance = std::hypot(guidance.along, guidance.cross);
    guidance.relativeVelocity = DeckRelativeVelocity(runway, aircraft.velocity);
    guidance.velocityDirection = UnitDirection(Subtract(aircraft.velocity, runway.velocity));
    guidance.targetSpeed = TargetSpeed(config, mission, aircraft, guidance.along);
    // The mode is informative only: Flare means low over the landable surface.
    bool const lowOverRunway = guidance.height >= 0 && guidance.height <= config.mission.flareHeightMeters
        && OnLandingSurface(runway, guidance.along, guidance.cross);
    if (mission.phase == Phase::Rollout)
    {
        guidance.mode = GuidanceMode::Rollout;
    }
    else if (lowOverRunway)
    {
        guidance.mode = GuidanceMode::Flare;
    }
    else
    {
        guidance.mode = GuidanceMode::Approach;
    }
    auto const reference = ResolveApproachReference(config, mission, aircraft);
    guidance.target = reference.target;
    guidance.targetValid = mission.outcome == Outcome::Flying;
    guidance.targetSlope = reference.slope;
    guidance.targetDirection = reference.direction;
    guidance.targetHeading = std::atan2(reference.direction.east, reference.direction.north);
    return guidance;
}
} // namespace ppo::examples::flight
