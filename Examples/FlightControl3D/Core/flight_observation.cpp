#include "flight_observation.hpp"
#include "flight_guidance.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
// Physical scales of the observation. Distances use a signed logarithm: the
// resolution is the size of a detail that still matters (a few metres near
// the deck, about a hundred metres far out) and the maximum saturates beyond
// the episode limits. Speeds and rates are divided by typical magnitudes.
constexpr double AlongResolutionMeters = 100.0, AlongMaximumMeters = 20000.0;
constexpr double CrossResolutionMeters = 10.0, CrossMaximumMeters = 5000.0;
constexpr double HeightResolutionMeters = 5.0, HeightMaximumMeters = 2000.0;
constexpr double GlideErrorResolutionMeters = 5.0, GlideErrorMaximumMeters = 500.0;
constexpr double ForwardSpeedScaleMps = 80.0;
constexpr double LateralSpeedScaleMps = 20.0;
constexpr double VerticalSpeedScaleMps = 10.0;
constexpr double AirspeedErrorScaleMps = 10.0;
constexpr double VerticalSpeedErrorScaleMps = 5.0;
constexpr double RateScaleRadiansPerSecond = 0.5;

/** @brief Clamp a finite value to [-1, 1]; a non-finite value becomes zero. */
float Unit(double value) noexcept
{
    return std::isfinite(value) ? static_cast<float>(std::clamp(value, -1.0, 1.0)) : 0.F;
}

void Set(Observation& observation, ObservationIndex index, float value) noexcept
{
    observation[Slot(index)] = value;
}
} // namespace

float SignedLog(double value, double resolution, double maximum) noexcept
{
    if (!std::isfinite(value)) return 0.F;
    double const magnitude = std::log1p(std::abs(value) / resolution) / std::log1p(maximum / resolution);
    return Unit(std::copysign(magnitude, value));
}

Observation Observe(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept
{
    if (!IsFinite(aircraft)) return {};
    // A time limit is a truncation: the critic must see the actual next state.
    auto observed = mission;
    if (observed.outcome == Outcome::Timeout) observed.outcome = Outcome::Flying;
    auto const g = Guide(config, observed, aircraft);
    auto const& runway = config.runway;
    double const heading = WrapAngle(aircraft.heading - runway.headingDegrees * DegreesToRadians);
    auto const body = AircraftAxes(aircraft.roll, aircraft.pitch, aircraft.heading);
    auto const targetBody = InBodyAxes(body, g.targetDirection);
    auto const velocityBody = InBodyAxes(body, g.velocityDirection);
    auto const& v = g.relativeVelocity;
    auto const& controls = mission.filteredControls;
    double const elapsed = static_cast<double>(mission.steps) * config.mission.physicsDt / config.mission.maxSeconds;

    Observation o{};
    Set(o, ObservationIndex::Along, SignedLog(g.along, AlongResolutionMeters, AlongMaximumMeters));
    Set(o, ObservationIndex::Cross, SignedLog(g.cross, CrossResolutionMeters, CrossMaximumMeters));
    Set(o, ObservationIndex::Height, SignedLog(g.height, HeightResolutionMeters, HeightMaximumMeters));
    Set(o, ObservationIndex::GlidePathError, SignedLog(g.height - ApproachPathHeight(config, g.along),
        GlideErrorResolutionMeters, GlideErrorMaximumMeters));
    Set(o, ObservationIndex::RunwayRemaining, Unit((runway.lengthMeters - g.along) / runway.lengthMeters));
    Set(o, ObservationIndex::VelocityForward, Unit(v.north / ForwardSpeedScaleMps));
    Set(o, ObservationIndex::VelocityRight, Unit(v.east / LateralSpeedScaleMps));
    Set(o, ObservationIndex::VelocityUp, Unit(v.up / VerticalSpeedScaleMps));
    Set(o, ObservationIndex::AirspeedError, Unit((aircraft.airspeed - g.targetSpeed) / AirspeedErrorScaleMps));
    Set(o, ObservationIndex::VerticalSpeedError, Unit((v.up - TargetVerticalSpeed(g)) / VerticalSpeedErrorScaleMps));
    Set(o, ObservationIndex::RollSin, Unit(std::sin(aircraft.roll)));
    Set(o, ObservationIndex::RollCos, Unit(std::cos(aircraft.roll)));
    Set(o, ObservationIndex::PitchSin, Unit(std::sin(aircraft.pitch)));
    Set(o, ObservationIndex::HeadingSin, Unit(std::sin(heading)));
    Set(o, ObservationIndex::HeadingCos, Unit(std::cos(heading)));
    Set(o, ObservationIndex::RollRate, Unit(aircraft.rollRate / RateScaleRadiansPerSecond));
    Set(o, ObservationIndex::PitchRate, Unit(aircraft.pitchRate / RateScaleRadiansPerSecond));
    Set(o, ObservationIndex::YawRate, Unit(aircraft.yawRate / RateScaleRadiansPerSecond));
    Set(o, ObservationIndex::TargetDirectionForward, Unit(targetBody.north));
    Set(o, ObservationIndex::TargetDirectionRight, Unit(targetBody.east));
    Set(o, ObservationIndex::TargetDirectionUp, Unit(targetBody.up));
    Set(o, ObservationIndex::VelocityDirectionForward, Unit(velocityBody.north));
    Set(o, ObservationIndex::VelocityDirectionRight, Unit(velocityBody.east));
    Set(o, ObservationIndex::VelocityDirectionUp, Unit(velocityBody.up));
    Set(o, ObservationIndex::WeightOnWheels, aircraft.weightOnWheels ? 1.F : 0.F);
    Set(o, ObservationIndex::Rollout, mission.phase == Phase::Rollout ? 1.F : 0.F);
    Set(o, ObservationIndex::StopProgress, Unit(mission.stoppedSeconds / config.mission.stoppedHoldSeconds));
    Set(o, ObservationIndex::Elapsed, Unit(elapsed));
    Set(o, ObservationIndex::Elevator, Unit(controls.elevator));
    Set(o, ObservationIndex::Aileron, Unit(controls.aileron));
    Set(o, ObservationIndex::Rudder, Unit(controls.rudder));
    Set(o, ObservationIndex::Throttle, Unit(controls.throttle));
    Set(o, ObservationIndex::Brake, Unit(controls.brake));
    return o;
}
} // namespace ppo::examples::flight
