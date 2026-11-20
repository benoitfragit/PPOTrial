#include "flight_spawn.hpp"
#include "flight_guidance.hpp"
#include <algorithm>
#include <cmath>

namespace ppo::examples::flight {
namespace {
double Draw(Range range, std::mt19937& rng)
{
    return std::uniform_real_distribution<double>(range.minimum, range.maximum)(rng);
}

/** @brief Draws a value uniformly in [-amplitude, amplitude]. */
double DrawSymmetric(double amplitude, std::mt19937& rng)
{
    return Draw({-amplitude, amplitude}, rng);
}

/** @brief Linear interpolation between two values. */
double Interpolate(double from, double to, double fraction) noexcept
{
    return from + (to - from) * fraction;
}

/** @brief Velocity of the given speed along a heading and a flight-path angle. */
Vec3 VelocityAlong(double speed, double heading, double slope) noexcept
{
    double const horizontal = speed * std::cos(slope);
    return {horizontal * std::cos(heading), horizontal * std::sin(heading), speed * std::sin(slope)};
}
} // namespace

EpisodeStart EvaluationStart(std::size_t episode, std::size_t episodes, double rollingShare,
    double maximumDifficulty) noexcept
{
    if (episodes == 0) return {};
    // Rolling starts come first; at least one airborne approach is always evaluated.
    double const share = std::clamp(rollingShare, 0.0, 1.0);
    auto const rounded = static_cast<std::size_t>(std::lround(share * static_cast<double>(episodes)));
    std::size_t const rollingEpisodes = std::min(rounded, episodes - 1);
    if (episode < rollingEpisodes) return {true, 0.0};
    // Airborne difficulties are stratified over (0, maximum]: the campaign
    // always reaches the curriculum frontier and rehearses easier cases.
    std::size_t const airborneEpisodes = episodes - rollingEpisodes;
    std::size_t const airborneIndex = std::min(episode - rollingEpisodes, airborneEpisodes - 1);
    double const fraction = static_cast<double>(airborneIndex + 1) / static_cast<double>(airborneEpisodes);
    return {false, std::clamp(maximumDifficulty, 0.0, 1.0) * fraction};
}

Runway SampleRunway(Config const& config, std::mt19937& rng)
{
    auto const& randomization = config.runwayRandomization;
    Runway runway = config.runway;
    runway.origin = {Draw(randomization.northMeters, rng), Draw(randomization.eastMeters, rng),
        Draw(randomization.deckHeightMeters, rng)};
    double const heading = config.runway.headingDegrees + Draw(randomization.headingOffsetDegrees, rng);
    runway.headingDegrees = std::fmod(heading + 360.0, 360.0);
    return runway;
}

AircraftState SampleApproachStart(Config const& config, Runway const& runway, ApproachSide side,
    double difficulty, std::mt19937& rng)
{
    auto const& initial = config.initial;
    auto const& mission = config.mission;
    double const d = std::clamp(difficulty, 0.0, 1.0);
    auto const& first = config.training.curriculum.firstDistanceMeters;
    Range const distanceRange{Interpolate(first.minimum, initial.distanceMeters.minimum, d),
        Interpolate(first.maximum, initial.distanceMeters.maximum, d)};
    double const distance = Draw(distanceRange, rng);
    double const sideSign = side == ApproachSide::Left ? -1.0 : 1.0;
    double const bearing = sideSign * d * Draw(initial.bearingDegrees, rng) * DegreesToRadians;
    double const along = -distance * std::cos(bearing);
    double const cross = distance * std::sin(bearing);
    // Height is drawn around the 3-degree plane through the threshold crossing height.
    double const height = ApproachPathHeight(config, along) + d * DrawSymmetric(initial.heightOffsetMeters, rng);

    AircraftState aircraft{};
    aircraft.position = FromRunway(runway, along, cross, height);
    aircraft.airspeed = mission.approachSpeed + DrawSymmetric(initial.speedToleranceMps, rng);
    double const headingOffset = d * Draw(initial.headingOffsetDegrees, rng);
    aircraft.heading = WrapAngle((runway.headingDegrees + headingOffset) * DegreesToRadians);
    aircraft.roll = DrawSymmetric(initial.rollDegrees, rng) * DegreesToRadians;
    aircraft.pitch = Draw(initial.pitchDegrees, rng) * DegreesToRadians;
    aircraft.gearPosition = 1.0;
    double const slope = (-mission.glideDegrees + DrawSymmetric(initial.slopeToleranceDegrees, rng)) * DegreesToRadians;
    aircraft.velocity = VelocityAlong(aircraft.airspeed, aircraft.heading, slope);
    return aircraft;
}

AircraftState SampleRolloutStart(Config const& config, Runway const& runway, std::mt19937& rng)
{
    // Start just above the deck; the simulator lowers the wheels onto it.
    constexpr double InitialWheelClearanceMeters = 2.5;
    auto const& rollout = config.initial.rolloutStart;
    double const along = Draw(rollout.alongMeters, rng);
    double const cross = Draw(rollout.crossMeters, rng);
    AircraftState aircraft{};
    aircraft.position = FromRunway(runway, along, cross, InitialWheelClearanceMeters);
    aircraft.airspeed = Draw(rollout.speedMps, rng);
    aircraft.heading = WrapAngle(runway.headingDegrees * DegreesToRadians);
    aircraft.gearPosition = 1.0;
    aircraft.velocity = Scale(RunwayForward(runway), aircraft.airspeed);
    return aircraft;
}
} // namespace ppo::examples::flight
