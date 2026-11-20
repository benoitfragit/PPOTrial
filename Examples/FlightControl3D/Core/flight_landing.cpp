#include "flight_landing.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Roll, pitch and heading of the aircraft relative to the deck, in radians. */
struct DeckAttitude final { double roll{}, pitch{}, heading{}; };

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
    double sink{}, speed{}, roll{}, pitchLow{}, pitchHigh{}, heading{};
};

TouchdownMeasurements Measure(Runway const& runway, AircraftState const& before, AircraftState const& after) noexcept
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
} // namespace

bool ContactDetected(Config const& config, AircraftState const& aircraft) noexcept
{
    if (aircraft.weightOnWheels || aircraft.structuralContact) return true;
    // Native JSBSim contacts use the finite deck and the sea outside it.
    // A tangent-plane height cannot override a valid physical contact sample.
    if (aircraft.wheelDataValid) return false;
    auto const local = DeckCoordinates(config.runway, aircraft.position);
    double const height = OnLandingSurface(config.runway, local.north, local.east)
        ? local.up : aircraft.position.up - config.runwayRandomization.seaHeightMeters;
    return height <= config.mission.contactHeightMeters;
}

bool WheelsInside(Config const& config, AircraftState const& aircraft) noexcept
{
    if (!aircraft.wheelDataValid) return true;
    for (auto const& wheel : aircraft.wheels)
    {
        auto const local = DeckCoordinates(config.runway, wheel.point);
        if (!OnLandingSurface(config.runway, local.north, local.east)) return false;
    }
    return true;
}

bool AllWheelsDown(AircraftState const& aircraft) noexcept
{
    if (!aircraft.wheelDataValid) return aircraft.weightOnWheels;
    for (auto const& wheel : aircraft.wheels)
        if (!wheel.onGround) return false;
    return true;
}

bool LandingSafe(Config const& config, AircraftState const& before, AircraftState const& after) noexcept
{
    auto const& m = config.mission;
    auto const measured = Measure(config.runway, before, after);
    return after.gearPosition >= m.gearDownMinimum && !after.structuralContact
        && measured.sink <= m.maxTouchdownSink && measured.speed <= m.maxTouchdownSpeed
        && measured.roll <= m.maxTouchdownRollDegrees * DegreesToRadians
        && measured.pitchLow >= m.minTouchdownPitchDegrees * DegreesToRadians
        && measured.pitchHigh <= m.maxTouchdownPitchDegrees * DegreesToRadians
        && measured.heading <= m.maxTouchdownHeadingDegrees * DegreesToRadians;
}

CrashDiagnostic DiagnoseCrash(Config const& config, Phase phase, AircraftState const& before,
    AircraftState const& after, bool contact) noexcept
{
    auto const& m = config.mission;
    CrashDiagnostic report{};
    bool const finite = IsFinite(before) && IsFinite(after);
    SetCrashCondition(report, CrashReason::InvalidState, !finite);
    if (!finite) return report;
    auto const local = DeckCoordinates(config.runway, after.position);
    double const distance = std::hypot(local.north, local.east);
    SetCrashCondition(report, CrashReason::Distance, distance > m.maxDistanceMeters, distance, m.maxDistanceMeters);
    SetCrashCondition(report, CrashReason::Altitude, local.up > m.maxAltitudeMeters, local.up, m.maxAltitudeMeters);
    if (contact || phase == Phase::Rollout)
    {
        SetCrashCondition(report, CrashReason::LandingArea,
            !OnLandingSurface(config.runway, local.north, local.east));
        SetCrashCondition(report, CrashReason::WheelsOutside, !WheelsInside(config, after));
    }
    if (!contact) return report;
    SetCrashCondition(report, CrashReason::Gear, after.gearPosition < m.gearDownMinimum,
        after.gearPosition, m.gearDownMinimum);
    SetCrashCondition(report, CrashReason::StructuralContact, after.structuralContact);
    SetCrashCondition(report, CrashReason::NoWheelContact, !after.weightOnWheels);
    auto const measured = Measure(config.runway, before, after);
    SetCrashCondition(report, CrashReason::Sink, measured.sink > m.maxTouchdownSink, measured.sink, m.maxTouchdownSink);
    SetCrashCondition(report, CrashReason::Speed, measured.speed > m.maxTouchdownSpeed,
        measured.speed, m.maxTouchdownSpeed);
    SetCrashCondition(report, CrashReason::Roll, measured.roll > m.maxTouchdownRollDegrees * DegreesToRadians,
        measured.roll / DegreesToRadians, m.maxTouchdownRollDegrees);
    SetCrashCondition(report, CrashReason::PitchLow, measured.pitchLow < m.minTouchdownPitchDegrees * DegreesToRadians,
        measured.pitchLow / DegreesToRadians, m.minTouchdownPitchDegrees);
    SetCrashCondition(report, CrashReason::PitchHigh,
        measured.pitchHigh > m.maxTouchdownPitchDegrees * DegreesToRadians,
        measured.pitchHigh / DegreesToRadians, m.maxTouchdownPitchDegrees);
    SetCrashCondition(report, CrashReason::Heading,
        measured.heading > m.maxTouchdownHeadingDegrees * DegreesToRadians,
        measured.heading / DegreesToRadians, m.maxTouchdownHeadingDegrees);
    return report;
}
} // namespace ppo::examples::flight
