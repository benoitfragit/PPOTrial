/**
 * @file flight_spawn.cpp
 * @brief Samplers of the episode starts and the stratified evaluation plan.
 *
 * Every draw uses std::uniform_real_distribution on the caller's std::mt19937. The draws
 * happen in the order written below; changing that order changes every seeded episode.
 *
 * Read next: flight_spawn.hpp (difficulty semantics), Simulator/flight_environment.cpp.
 */
#include "flight_spawn.hpp"
#include "flight_guidance.hpp"
#include <algorithm>
#include <cmath>

namespace ppo::examples::flight {
namespace {
/** @brief Height above the deck of a rolling start before the simulator lowers the wheels, in metres. */
constexpr double InitialWheelClearanceMeters = 2.5;
/** @brief Gear position of every start: down and locked. */
constexpr double GearDown = 1.0;

/** @brief Draws a value uniformly in a range. */
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

/**
 * @brief Velocity of the given speed along a heading and a flight-path angle.
 * @param speed Speed, in metres per second.
 * @param heading Track azimuth, clockwise from north, in radians.
 * @param slope Flight-path angle, in radians (negative when descending).
 * @return World velocity (north, east, up), in metres per second.
 */
Vec3 VelocityAlong(double speed, double heading, double slope) noexcept
{
    double const horizontal = speed * std::cos(slope);
    return {horizontal * std::cos(heading), horizontal * std::sin(heading), speed * std::sin(slope)};
}
} // namespace

EpisodeStart EvaluationStart(std::size_t episode, std::size_t episodes, double rollingShare,
    double maximumDifficulty) noexcept
{
    if (episodes == 0)
    {
        return {};
    }
    // Rolling starts come first; at least one airborne approach is always evaluated.
    double const share = std::clamp(rollingShare, 0.0, 1.0);
    auto const rounded = static_cast<std::size_t>(std::lround(share * static_cast<double>(episodes)));
    std::size_t const rollingEpisodes = std::min(rounded, episodes - 1);
    if (episode < rollingEpisodes)
    {
        return {true, 0.0};
    }
    // Airborne difficulties are stratified over (0, maximum]: the campaign
    // always reaches the curriculum frontier and rehearses easier cases.
    // Airborne episode k of n uses difficulty maximum * (k + 1) / n.
    std::size_t const airborneEpisodes = episodes - rollingEpisodes;
    std::size_t const airborneIndex = std::min(episode - rollingEpisodes, airborneEpisodes - 1);
    double const fraction = static_cast<double>(airborneIndex + 1) / static_cast<double>(airborneEpisodes);
    return {false, std::clamp(maximumDifficulty, 0.0, 1.0) * fraction};
}

Runway SampleRunway(Config const& config, std::mt19937& rng)
{
    auto const& randomization = config.runwayRandomization;
    Runway runway = config.runway;
    // The three origin draws happen in braced-initializer order: north, east, deck height.
    runway.origin = {Draw(randomization.northMeters, rng), Draw(randomization.eastMeters, rng),
        Draw(randomization.deckHeightMeters, rng)};
    constexpr double FullTurnDegrees = 360.0;
    double const headingDegrees = config.runway.headingDegrees + Draw(randomization.headingOffsetDegrees,
        rng);
    runway.headingDegrees = std::fmod(headingDegrees + FullTurnDegrees, FullTurnDegrees);
    return runway;
}

AircraftState SampleApproachStart(Config const& config, Runway const& runway, ApproachSide side,
    double difficulty, std::mt19937& rng)
{
    auto const& initial = config.initial;
    auto const& mission = config.mission;
    double const clampedDifficulty = std::clamp(difficulty, 0.0, 1.0);
    // Distance range interpolated from the first curriculum range to the full envelope.
    auto const& firstDistance = config.training.curriculum.firstDistanceMeters;
    Range const distanceRange{
        Interpolate(firstDistance.minimum, initial.distanceMeters.minimum, clampedDifficulty),
        Interpolate(firstDistance.maximum, initial.distanceMeters.maximum, clampedDifficulty)};
    double const distanceMeters = Draw(distanceRange, rng);
    // Bearing from the extended centreline, scaled by the difficulty, signed by the side.
    double const sideSign = side == ApproachSide::Left ? -1.0 : 1.0;
    double const bearingRadians = sideSign * clampedDifficulty * Draw(initial.bearingDegrees, rng)
        * DegreesToRadians;
    double const alongMeters = -distanceMeters * std::cos(bearingRadians);
    double const crossMeters = distanceMeters * std::sin(bearingRadians);
    // Height is drawn around the 3-degree plane through the threshold crossing height.
    double const heightMeters = ApproachPathHeight(config, alongMeters)
        + clampedDifficulty * DrawSymmetric(initial.heightOffsetMeters, rng);

    AircraftState aircraft{};
    aircraft.position = FromRunway(runway, alongMeters, crossMeters, heightMeters);
    aircraft.airspeed = mission.approachSpeed + DrawSymmetric(initial.speedToleranceMps, rng);
    double const headingOffsetDegrees = clampedDifficulty * Draw(initial.headingOffsetDegrees, rng);
    aircraft.heading = WrapAngle((runway.headingDegrees + headingOffsetDegrees) * DegreesToRadians);
    aircraft.roll = DrawSymmetric(initial.rollDegrees, rng) * DegreesToRadians;
    aircraft.pitch = Draw(initial.pitchDegrees, rng) * DegreesToRadians;
    aircraft.gearPosition = GearDown;
    double const slopeRadians = (-mission.glideDegrees + DrawSymmetric(initial.slopeToleranceDegrees, rng))
        * DegreesToRadians;
    aircraft.velocity = VelocityAlong(aircraft.airspeed, aircraft.heading, slopeRadians);
    return aircraft;
}

AircraftState SampleRolloutStart(Config const& config, Runway const& runway, std::mt19937& rng)
{
    // Start just above the deck; the simulator lowers the wheels onto it.
    auto const& rollout = config.initial.rolloutStart;
    double const alongMeters = Draw(rollout.alongMeters, rng);
    double const crossMeters = Draw(rollout.crossMeters, rng);
    AircraftState aircraft{};
    aircraft.position = FromRunway(runway, alongMeters, crossMeters, InitialWheelClearanceMeters);
    aircraft.airspeed = Draw(rollout.speedMps, rng);
    aircraft.heading = WrapAngle(runway.headingDegrees * DegreesToRadians);
    aircraft.gearPosition = GearDown;
    aircraft.velocity = Scale(RunwayForward(runway), aircraft.airspeed);
    return aircraft;
}
} // namespace ppo::examples::flight
