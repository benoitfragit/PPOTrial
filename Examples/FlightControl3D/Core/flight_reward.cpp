#include "flight_reward.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Terminal states are absorbing and carry no potential; a truncated Timeout keeps it. */
bool Absorbing(MissionState const& mission) noexcept
{
    return mission.outcome == Outcome::Crash || mission.outcome == Outcome::Stopped;
}

/** @brief Magnitude of a ratio, saturated at one. */
double Saturated(double ratio) noexcept
{
    return std::min(std::abs(ratio), 1.0);
}

double PathValue(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept
{
    double const weight = Weight(config.reward, Reward::Path);
    // On the ground the approach is complete: a constant potential leaves the
    // rollout to the stop objective without a contradictory distance signal.
    if (mission.phase == Phase::Rollout) return weight;
    double const tolerance = config.reward.pathTolerance;
    return weight * tolerance / (tolerance + PathDistance(config, aircraft));
}

double VerticalSpeedValue(Config const& config, MissionState const& mission, Guidance const& guidance) noexcept
{
    // Only the airborne descent has a vertical-speed reference; the touchdown
    // sink itself is judged by the Touchdown quality.
    if (mission.phase == Phase::Rollout) return 0;
    double const error = guidance.relativeVelocity.up - TargetVerticalSpeed(guidance);
    return -Weight(config.reward, Reward::VerticalSpeed) * Saturated(error / VerticalSpeedErrorScaleMps);
}

double SpeedValue(Config const& config, MissionState const& mission, AircraftState const& aircraft,
    Guidance const& guidance) noexcept
{
    double const weight = Weight(config.reward, Reward::Speed);
    if (mission.phase == Phase::Approach)
        return -weight * Saturated((aircraft.airspeed - guidance.targetSpeed) / AirspeedErrorScaleMps);
    // On the ground the potential rises as the aircraft slows down. It is
    // zero at the flare speed, so it matches an on-speed airborne touchdown.
    double const groundSpeed = std::hypot(guidance.relativeVelocity.north, guidance.relativeVelocity.east);
    return weight * (1.0 - std::min(groundSpeed / config.mission.flareSpeed, 1.0));
}

/** @brief One past the touchdown zone, decreasing linearly to zero at mid-runway. */
double TouchdownZoneFactor(Runway const& runway, double along) noexcept
{
    double const zoneEnd = TouchdownZoneLengthMeters;
    double const fadeEnd = runway.lengthMeters / 2;
    if (along <= zoneEnd || fadeEnd <= zoneEnd) return 1.0;
    return std::clamp(1.0 - (along - zoneEnd) / (fadeEnd - zoneEnd), 0.0, 1.0);
}
} // namespace

bool IsPotentialChannel(Reward channel) noexcept
{
    return channel == Reward::Path || channel == Reward::VerticalSpeed || channel == Reward::Speed;
}

double PathDistance(Config const& config, AircraftState const& aircraft) noexcept
{
    auto const local = DeckCoordinates(config.runway, aircraft.position);
    double const aimAlong = AimPointAlongMeters(config);
    // Past the aim point only the lateral offset and the height remain: the
    // distance is continuous there and never penalizes rolling down the runway.
    double const remaining = std::max(0.0, aimAlong - local.north);
    return std::hypot(std::hypot(remaining, local.east), local.up);
}

std::array<double, RewardCount> ShapingPotentials(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept
{
    std::array<double, RewardCount> potentials{};
    if (Absorbing(mission) || !IsFinite(aircraft)) return potentials;
    auto const guidance = Guide(config, mission, aircraft);
    potentials[static_cast<std::size_t>(Reward::Path)] = PathValue(config, mission, aircraft);
    potentials[static_cast<std::size_t>(Reward::VerticalSpeed)] = VerticalSpeedValue(config, mission, guidance);
    potentials[static_cast<std::size_t>(Reward::Speed)] = SpeedValue(config, mission, aircraft, guidance);
    return potentials;
}

double DirectionCost(Guidance const& guidance, double dt) noexcept
{
    bool const moving = Dot(guidance.velocityDirection, guidance.velocityDirection) >= 0.5;
    double const alignment = moving
        ? std::clamp(Dot(guidance.velocityDirection, guidance.targetDirection), -1.0, 1.0) : -1.0;
    return -dt * 0.5 * (1.0 - alignment);
}

double TouchdownQuality(Config const& config, MissionState const& mission) noexcept
{
    auto const& m = config.mission;
    double const area = LandingRewardFactor(config.runway, mission.touchdownFromThreshold,
        mission.touchdownCrossMeters);
    double const zone = TouchdownZoneFactor(config.runway, mission.touchdownFromThreshold);
    // Quadratic in the sink rate: a firm but normal touchdown keeps most of
    // the reward, while one near the structural limit keeps almost nothing.
    double const sinkRatio = std::min(std::max(mission.touchdownSinkMps, 0.0) / m.maxTouchdownSink, 1.0);
    double const sink = 1.0 - sinkRatio * sinkRatio;
    double const speedMargin = m.maxTouchdownSpeed - m.flareSpeed;
    double const excess = speedMargin > 0
        ? std::clamp((mission.touchdownSpeedMps - m.flareSpeed) / speedMargin, 0.0, 1.0) : 0.0;
    double const speed = 1.0 - 0.5 * excess;
    return area * zone * sink * speed;
}
} // namespace ppo::examples::flight
