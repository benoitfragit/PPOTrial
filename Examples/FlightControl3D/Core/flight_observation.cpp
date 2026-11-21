/**
 * @file flight_observation.cpp
 * @brief Fills the observation slots from the guidance, the attitude and the filtered controls.
 *
 * Normalization rules:
 * - distances: SignedLog with a resolution (size of a detail that still matters) and a
 *   maximum (beyond the episode limits);
 * - speeds and rates: divided by a typical magnitude, then clamped to [-1, 1];
 * - angles: sine and cosine, so no wrap-around discontinuity reaches the network;
 * - directions: unit vectors in body axes;
 * - flags and controls: already in [0, 1] or [-1, 1].
 *
 * Read next: flight_observation.hpp (slot list), Training/flight_learner.cpp (consumer).
 */
#include "flight_observation.hpp"
#include "flight_guidance.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
// Physical scales of the observation. Distances use a signed logarithm: the
// resolution is the size of a detail that still matters (a few metres near
// the deck, about a hundred metres far out) and the maximum saturates beyond
// the episode limits. Speeds and rates are divided by typical magnitudes.
constexpr double AlongResolutionMeters = 100.0;
constexpr double AlongMaximumMeters = 20000.0;
constexpr double CrossResolutionMeters = 10.0;
constexpr double CrossMaximumMeters = 5000.0;
constexpr double HeightResolutionMeters = 5.0;
constexpr double HeightMaximumMeters = 2000.0;
constexpr double GlideErrorResolutionMeters = 5.0;
constexpr double GlideErrorMaximumMeters = 500.0;
/** @brief About 1.4 times the 56 m/s approach speed. */
constexpr double ForwardSpeedScaleMps = 80.0;
/** @brief Lateral drift of a 20-degree track error at approach speed (56 * sin 20 deg = 19 m/s). */
constexpr double LateralSpeedScaleMps = 20.0;
/** @brief About 3.4 times the 3-degree descent rate at approach speed (2.9 m/s). */
constexpr double VerticalSpeedScaleMps = 10.0;
constexpr double AirspeedErrorScaleMps = 10.0;
constexpr double VerticalSpeedErrorScaleMps = 5.0;
/** @brief About 30 degrees per second. */
constexpr double RateScaleRadiansPerSecond = 0.5;

/** @brief Clamp a finite value to [-1, 1]; a non-finite value becomes zero. */
float Unit(double value) noexcept
{
    if (!std::isfinite(value))
    {
        return 0.F;
    }
    return static_cast<float>(std::clamp(value, -1.0, 1.0));
}

/** @brief Binary flag as 1 or 0. */
float Flag(bool value) noexcept
{
    return value ? 1.F : 0.F;
}

void Set(Observation& observation, ObservationIndex index, float value) noexcept
{
    observation[Slot(index)] = value;
}
} // namespace

float SignedLog(double value, double resolution, double maximum) noexcept
{
    if (!std::isfinite(value))
    {
        return 0.F;
    }
    double const magnitude = std::log1p(std::abs(value) / resolution) / std::log1p(maximum / resolution);
    return Unit(std::copysign(magnitude, value));
}

Observation Observe(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept
{
    if (!IsFinite(aircraft))
    {
        return {};
    }
    // A time limit is a truncation: the critic must see the actual next state.
    auto observed = mission;
    if (observed.outcome == Outcome::Timeout)
    {
        observed.outcome = Outcome::Flying;
    }
    auto const guidance = Guide(config, observed, aircraft);
    auto const& runway = config.runway;
    double const relativeHeading = WrapAngle(aircraft.heading - runway.headingDegrees * DegreesToRadians);
    auto const body = AircraftAxes(aircraft.roll, aircraft.pitch, aircraft.heading);
    auto const targetBody = InBodyAxes(body, guidance.targetDirection);
    auto const velocityBody = InBodyAxes(body, guidance.velocityDirection);
    auto const& velocity = guidance.relativeVelocity;
    auto const& controls = mission.filteredControls;
    double const elapsedFraction = static_cast<double>(mission.steps) * config.mission.physicsDt
        / config.mission.maxSeconds;
    double const glidePathErrorMeters = guidance.height - ApproachPathHeight(config, guidance.along);

    Observation observation{};
    // Runway-relative position.
    Set(observation, ObservationIndex::Along, SignedLog(guidance.along, AlongResolutionMeters,
        AlongMaximumMeters));
    Set(observation, ObservationIndex::Cross, SignedLog(guidance.cross, CrossResolutionMeters,
        CrossMaximumMeters));
    Set(observation, ObservationIndex::Height,
        SignedLog(guidance.height, HeightResolutionMeters, HeightMaximumMeters));
    Set(observation, ObservationIndex::GlidePathError,
        SignedLog(glidePathErrorMeters, GlideErrorResolutionMeters, GlideErrorMaximumMeters));
    Set(observation, ObservationIndex::RunwayRemaining,
        Unit((runway.lengthMeters - guidance.along) / runway.lengthMeters));
    // Deck-relative velocity and tracking errors.
    Set(observation, ObservationIndex::VelocityForward, Unit(velocity.north / ForwardSpeedScaleMps));
    Set(observation, ObservationIndex::VelocityRight, Unit(velocity.east / LateralSpeedScaleMps));
    Set(observation, ObservationIndex::VelocityUp, Unit(velocity.up / VerticalSpeedScaleMps));
    Set(observation, ObservationIndex::AirspeedError,
        Unit((aircraft.airspeed - guidance.targetSpeed) / AirspeedErrorScaleMps));
    Set(observation, ObservationIndex::VerticalSpeedError,
        Unit((velocity.up - TargetVerticalSpeed(guidance)) / VerticalSpeedErrorScaleMps));
    // Attitude relative to the runway and body rates.
    Set(observation, ObservationIndex::RollSin, Unit(std::sin(aircraft.roll)));
    Set(observation, ObservationIndex::RollCos, Unit(std::cos(aircraft.roll)));
    Set(observation, ObservationIndex::PitchSin, Unit(std::sin(aircraft.pitch)));
    Set(observation, ObservationIndex::HeadingSin, Unit(std::sin(relativeHeading)));
    Set(observation, ObservationIndex::HeadingCos, Unit(std::cos(relativeHeading)));
    Set(observation, ObservationIndex::RollRate, Unit(aircraft.rollRate / RateScaleRadiansPerSecond));
    Set(observation, ObservationIndex::PitchRate, Unit(aircraft.pitchRate / RateScaleRadiansPerSecond));
    Set(observation, ObservationIndex::YawRate, Unit(aircraft.yawRate / RateScaleRadiansPerSecond));
    // Commanded and actual directions in body axes.
    Set(observation, ObservationIndex::TargetDirectionForward, Unit(targetBody.north));
    Set(observation, ObservationIndex::TargetDirectionRight, Unit(targetBody.east));
    Set(observation, ObservationIndex::TargetDirectionUp, Unit(targetBody.up));
    Set(observation, ObservationIndex::VelocityDirectionForward, Unit(velocityBody.north));
    Set(observation, ObservationIndex::VelocityDirectionRight, Unit(velocityBody.east));
    Set(observation, ObservationIndex::VelocityDirectionUp, Unit(velocityBody.up));
    // Phase flags and episode progress.
    Set(observation, ObservationIndex::WeightOnWheels, Flag(aircraft.weightOnWheels));
    Set(observation, ObservationIndex::Rollout, Flag(mission.phase == Phase::Rollout));
    Set(observation, ObservationIndex::StopProgress,
        Unit(mission.stoppedSeconds / config.mission.stoppedHoldSeconds));
    Set(observation, ObservationIndex::Elapsed, Unit(elapsedFraction));
    // Gear extension and its latched command; the command flag also masks the gear action.
    Set(observation, ObservationIndex::GearPosition, Unit(aircraft.gearPosition));
    Set(observation, ObservationIndex::GearCommanded, Flag(mission.gearDownCommanded));
    // Filtered controls actually applied by the simulator.
    Set(observation, ObservationIndex::Elevator, Unit(controls.elevator));
    Set(observation, ObservationIndex::Aileron, Unit(controls.aileron));
    Set(observation, ObservationIndex::Rudder, Unit(controls.rudder));
    Set(observation, ObservationIndex::Throttle, Unit(controls.throttle));
    Set(observation, ObservationIndex::Brake, Unit(controls.brake));
    return observation;
}
} // namespace ppo::examples::flight
