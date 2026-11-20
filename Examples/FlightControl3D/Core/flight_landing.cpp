/**
 * @file flight_landing.cpp
 * @brief Measurements around the contact substep and the crash conditions built on them.
 *
 * Attitudes are measured relative to the deck: roll and pitch from the deck normal in body
 * axes, heading from the body forward axis in runway axes. Each touchdown measurement takes
 * the worst value of the before and after snapshots.
 *
 * Read next: flight_landing.hpp, flight_mission.cpp (Crashed, the authoritative rule).
 */
#include "flight_landing.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Roll, pitch and heading of the aircraft relative to the deck, in radians. */
struct DeckAttitude final
{
    double roll{};
    double pitch{};
    double heading{};
};

/**
 * @brief Attitude of the body axes relative to the level deck.
 *
 * roll = atan2(-right . n, up . n), pitch = asin(forward . n),
 * heading = atan2(forward . runwayRight, forward . runwayForward), with n the deck normal.
 */
DeckAttitude RelativeAttitude(Runway const& runway, AircraftState const& aircraft) noexcept
{
    auto const body = AircraftAxes(aircraft.roll, aircraft.pitch, aircraft.heading);
    auto const normal = RunwayNormal(runway);
    return {std::atan2(-Dot(body.right, normal), Dot(body.up, normal)),
        std::asin(std::clamp(Dot(body.forward, normal), -1.0, 1.0)),
        std::atan2(Dot(body.forward, RunwayRight(runway)), Dot(body.forward, RunwayForward(runway)))};
}

/** @brief Worst values of the touchdown measurements over the two contact snapshots. */
struct TouchdownMeasurements final
{
    /** @brief Largest sink rate, in m/s (positive downward). */
    double sink{};
    /** @brief Largest horizontal ground speed, in m/s. */
    double speed{};
    /** @brief Largest absolute deck-relative bank, in radians. */
    double roll{};
    /** @brief Lowest deck-relative pitch, in radians. */
    double pitchLow{};
    /** @brief Highest deck-relative pitch, in radians. */
    double pitchHigh{};
    /** @brief Largest absolute heading error relative to the runway, in radians. */
    double heading{};
};

TouchdownMeasurements Measure(Runway const& runway, AircraftState const& before,
    AircraftState const& after) noexcept
{
    auto const velocityBefore = RunwayAxes(runway, before.velocity);
    auto const velocityAfter = RunwayAxes(runway, after.velocity);
    auto const attitudeBefore = RelativeAttitude(runway, before);
    auto const attitudeAfter = RelativeAttitude(runway, after);
    TouchdownMeasurements result{};
    result.sink = std::max(-velocityBefore.up, -velocityAfter.up);
    result.speed = std::max(std::hypot(velocityBefore.north, velocityBefore.east),
        std::hypot(velocityAfter.north, velocityAfter.east));
    result.roll = std::max(std::abs(attitudeBefore.roll), std::abs(attitudeAfter.roll));
    result.pitchLow = std::min(attitudeBefore.pitch, attitudeAfter.pitch);
    result.pitchHigh = std::max(attitudeBefore.pitch, attitudeAfter.pitch);
    result.heading = std::max(std::abs(attitudeBefore.heading), std::abs(attitudeAfter.heading));
    return result;
}

void SetCrashCondition(CrashDiagnostic& report, CrashReason reason, bool failed, double value = 0,
    double limit = 0) noexcept
{
    report.conditions[static_cast<std::size_t>(reason)] = {failed, value, limit};
}

/** @brief Records the touchdown measurement limits; angles are reported in degrees. */
void DiagnoseTouchdownLimits(CrashDiagnostic& report, MissionSettings const& limits,
    TouchdownMeasurements const& measured) noexcept
{
    SetCrashCondition(report, CrashReason::Sink, measured.sink > limits.maxTouchdownSink, measured.sink,
        limits.maxTouchdownSink);
    SetCrashCondition(report, CrashReason::Speed, measured.speed > limits.maxTouchdownSpeed,
        measured.speed, limits.maxTouchdownSpeed);
    SetCrashCondition(report, CrashReason::Roll,
        measured.roll > limits.maxTouchdownRollDegrees * DegreesToRadians,
        measured.roll / DegreesToRadians, limits.maxTouchdownRollDegrees);
    SetCrashCondition(report, CrashReason::PitchLow,
        measured.pitchLow < limits.minTouchdownPitchDegrees * DegreesToRadians,
        measured.pitchLow / DegreesToRadians, limits.minTouchdownPitchDegrees);
    SetCrashCondition(report, CrashReason::PitchHigh,
        measured.pitchHigh > limits.maxTouchdownPitchDegrees * DegreesToRadians,
        measured.pitchHigh / DegreesToRadians, limits.maxTouchdownPitchDegrees);
    SetCrashCondition(report, CrashReason::Heading,
        measured.heading > limits.maxTouchdownHeadingDegrees * DegreesToRadians,
        measured.heading / DegreesToRadians, limits.maxTouchdownHeadingDegrees);
}
} // namespace

bool ContactDetected(Config const& config, AircraftState const& aircraft) noexcept
{
    if (aircraft.weightOnWheels || aircraft.structuralContact)
    {
        return true;
    }
    // Native JSBSim contacts use the finite deck and the sea outside it.
    // A tangent-plane height cannot override a valid physical contact sample.
    if (aircraft.wheelDataValid)
    {
        return false;
    }
    // Geometric fallback: height of the CG above the deck, or above the sea off the deck.
    auto const local = DeckCoordinates(config.runway, aircraft.position);
    double surfaceHeight = aircraft.position.up - config.runwayRandomization.seaHeightMeters;
    if (OnLandingSurface(config.runway, local.north, local.east))
    {
        surfaceHeight = local.up;
    }
    return surfaceHeight <= config.mission.contactHeightMeters;
}

bool WheelsInside(Config const& config, AircraftState const& aircraft) noexcept
{
    if (!aircraft.wheelDataValid)
    {
        return true;
    }
    for (auto const& wheel : aircraft.wheels)
    {
        auto const local = DeckCoordinates(config.runway, wheel.point);
        if (!OnLandingSurface(config.runway, local.north, local.east))
        {
            return false;
        }
    }
    return true;
}

bool AllWheelsDown(AircraftState const& aircraft) noexcept
{
    if (!aircraft.wheelDataValid)
    {
        return aircraft.weightOnWheels;
    }
    for (auto const& wheel : aircraft.wheels)
    {
        if (!wheel.onGround)
        {
            return false;
        }
    }
    return true;
}

bool LandingSafe(Config const& config, AircraftState const& before, AircraftState const& after) noexcept
{
    auto const& limits = config.mission;
    auto const measured = Measure(config.runway, before, after);
    return after.gearPosition >= limits.gearDownMinimum && !after.structuralContact
        && measured.sink <= limits.maxTouchdownSink && measured.speed <= limits.maxTouchdownSpeed
        && measured.roll <= limits.maxTouchdownRollDegrees * DegreesToRadians
        && measured.pitchLow >= limits.minTouchdownPitchDegrees * DegreesToRadians
        && measured.pitchHigh <= limits.maxTouchdownPitchDegrees * DegreesToRadians
        && measured.heading <= limits.maxTouchdownHeadingDegrees * DegreesToRadians;
}

CrashDiagnostic DiagnoseCrash(Config const& config, Phase phase, AircraftState const& before,
    AircraftState const& after, bool contact) noexcept
{
    auto const& limits = config.mission;
    CrashDiagnostic report{};
    bool const finite = IsFinite(before) && IsFinite(after);
    SetCrashCondition(report, CrashReason::InvalidState, !finite);
    if (!finite)
    {
        return report;
    }
    // Envelope limits, evaluated in every phase.
    auto const local = DeckCoordinates(config.runway, after.position);
    double const distance = std::hypot(local.north, local.east);
    SetCrashCondition(report, CrashReason::Distance, distance > limits.maxDistanceMeters, distance,
        limits.maxDistanceMeters);
    SetCrashCondition(report, CrashReason::Altitude, local.up > limits.maxAltitudeMeters, local.up,
        limits.maxAltitudeMeters);
    // Surface limits, once on the ground.
    if (contact || phase == Phase::Rollout)
    {
        SetCrashCondition(report, CrashReason::LandingArea,
            !OnLandingSurface(config.runway, local.north, local.east));
        SetCrashCondition(report, CrashReason::WheelsOutside, !WheelsInside(config, after));
    }
    if (!contact)
    {
        return report;
    }
    // Touchdown limits, only on a contact substep.
    SetCrashCondition(report, CrashReason::Gear, after.gearPosition < limits.gearDownMinimum,
        after.gearPosition, limits.gearDownMinimum);
    SetCrashCondition(report, CrashReason::StructuralContact, after.structuralContact);
    SetCrashCondition(report, CrashReason::NoWheelContact, !after.weightOnWheels);
    DiagnoseTouchdownLimits(report, limits, Measure(config.runway, before, after));
    return report;
}
} // namespace ppo::examples::flight
