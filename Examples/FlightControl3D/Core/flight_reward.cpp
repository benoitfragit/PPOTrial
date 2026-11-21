/**
 * @file flight_reward.cpp
 * @brief Formulas of the potential channels, the direction cost and the touchdown quality.
 *
 * Potential-based shaping (Ng et al., 1999): adding F = gamma * Phi(s') - Phi(s) to the
 * reward leaves the optimal policy unchanged. Each potential below is bounded by its
 * weight and saturates far from the reference, so a distant start cannot dominate the
 * event rewards. The discounting per substep is done by the caller (flight_mission.cpp).
 *
 * Read next: flight_reward.hpp (channel table), flight_mission.cpp (event rewards).
 */
#include "flight_reward.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Squared length below which a unit-velocity direction is the zero vector (not moving). */
constexpr double MovingDirectionMinimumSquaredLength = 0.5;
/** @brief Alignment assigned to a vanishing velocity: as bad as flying backwards. */
constexpr double StationaryAlignment = -1.0;
/** @brief Maps 1 - cos(angle), in [0, 2], to a cost in [0, 1]. */
constexpr double DirectionCostScale = 0.5;
/** @brief Largest fraction of the Touchdown reward lost to an excess touchdown speed. */
constexpr double MaximumSpeedPenaltyFraction = 0.5;

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

/**
 * @brief Path potential: weight * tolerance / (tolerance + distance) in flight, weight on the ground.
 *
 * Half the weight at a distance equal to the tolerance (1.5 km in the scenario), the full
 * weight at the reference: a hyperbolic shape keeps a useful gradient from 15 km out.
 */
double PathValue(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept
{
    double const weight = Weight(config.reward, Reward::Path);
    // On the ground the approach is complete: a constant potential leaves the
    // rollout to the stop objective without a contradictory distance signal.
    if (mission.phase == Phase::Rollout)
    {
        return weight;
    }
    double const tolerance = config.reward.pathTolerance;
    return weight * tolerance / (tolerance + PathDistance(config, mission, aircraft));
}

/** @brief Straight-line distance between two points in runway axes, in metres. */
double GateDistance(Gate const& from, Gate const& to) noexcept
{
    return std::hypot(std::hypot(to.alongMeters - from.alongMeters, to.crossMeters - from.crossMeters),
        to.heightMeters - from.heightMeters);
}

/**
 * @brief VerticalSpeed potential: -weight * min(|vz - vz_target| / 2 m/s, 1) in flight, zero on
 *        the ground.
 */
double VerticalSpeedValue(Config const& config, MissionState const& mission,
    Guidance const& guidance) noexcept
{
    // Only the airborne descent has a vertical-speed reference; the touchdown
    // sink itself is judged by the Touchdown quality.
    if (mission.phase == Phase::Rollout)
    {
        return 0;
    }
    double const error = guidance.relativeVelocity.up - TargetVerticalSpeed(guidance);
    return -Weight(config.reward, Reward::VerticalSpeed) * Saturated(error / VerticalSpeedErrorScaleMps);
}

/**
 * @brief Speed potential: airspeed tracking in flight, deceleration progress on the ground.
 *
 * Airborne: -weight * min(|CAS - target| / 10 m/s, 1). Ground: weight * (1 - min(v / flare speed, 1)).
 */
double SpeedValue(Config const& config, MissionState const& mission, AircraftState const& aircraft,
    Guidance const& guidance) noexcept
{
    double const weight = Weight(config.reward, Reward::Speed);
    if (mission.phase == Phase::Approach)
    {
        return -weight * Saturated((aircraft.airspeed - guidance.targetSpeed) / AirspeedErrorScaleMps);
    }
    // On the ground the potential rises as the aircraft slows down. It is
    // zero at the flare speed, so it matches an on-speed airborne touchdown.
    double const groundSpeed = std::hypot(guidance.relativeVelocity.north, guidance.relativeVelocity.east);
    return weight * (1.0 - std::min(groundSpeed / config.mission.flareSpeed, 1.0));
}

/** @brief One within the touchdown zone, decreasing linearly to zero at mid-runway. */
double TouchdownZoneFactor(Runway const& runway, double along) noexcept
{
    double const zoneEnd = TouchdownZoneLengthMeters;
    double const fadeEnd = runway.lengthMeters / 2;
    if (along <= zoneEnd || fadeEnd <= zoneEnd)
    {
        return 1.0;
    }
    return std::clamp(1.0 - (along - zoneEnd) / (fadeEnd - zoneEnd), 0.0, 1.0);
}

/** @brief Sink factor 1 - (sink / max sink)^2, clamped: firm contacts keep most of the reward. */
double SinkFactor(MissionSettings const& limits, double sinkMps) noexcept
{
    // Quadratic in the sink rate: a firm but normal touchdown keeps most of
    // the reward, while one near the structural limit keeps almost nothing.
    double const sinkRatio = std::min(std::max(sinkMps, 0.0) / limits.maxTouchdownSink, 1.0);
    return 1.0 - sinkRatio * sinkRatio;
}

/** @brief Speed factor: one at or below the flare speed, half at the maximum touchdown speed. */
double SpeedFactor(MissionSettings const& limits, double speedMps) noexcept
{
    double const speedMargin = limits.maxTouchdownSpeed - limits.flareSpeed;
    double excess = 0.0;
    if (speedMargin > 0)
    {
        excess = std::clamp((speedMps - limits.flareSpeed) / speedMargin, 0.0, 1.0);
    }
    return 1.0 - MaximumSpeedPenaltyFraction * excess;
}
} // namespace

bool IsPotentialChannel(Reward channel) noexcept
{
    return channel == Reward::Path || channel == Reward::VerticalSpeed || channel == Reward::Speed;
}

double ApproachDistance(Config const& config, Gate const& point) noexcept
{
    // Past the aim point only the lateral offset and the height remain: the
    // distance is continuous there and never penalizes rolling down the runway.
    double const remaining = std::max(0.0, AimPointAlongMeters(config) - point.alongMeters);
    return std::hypot(std::hypot(remaining, point.crossMeters), point.heightMeters);
}

double PathDistance(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept
{
    auto const local = DeckCoordinates(config.runway, aircraft.position);
    Gate const position{local.north, local.east, local.up};
    if (mission.nextGate >= mission.route.count)
    {
        return ApproachDistance(config, position);
    }
    // Remaining route: to the next gate, along the legs still ahead, then the
    // approach from the last gate. Passing a gate never increases this length
    // (triangle inequality), so the potential cannot reward missing a gate.
    auto const& gates = mission.route.gates;
    double distance = GateDistance(position, gates[mission.nextGate]);
    for (std::size_t gate = mission.nextGate; gate + 1 < mission.route.count; ++gate)
    {
        distance += GateDistance(gates[gate], gates[gate + 1]);
    }
    return distance + ApproachDistance(config, gates[mission.route.count - 1]);
}

std::array<double, RewardCount> ShapingPotentials(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept
{
    std::array<double, RewardCount> potentials{};
    // Phi = 0 on absorbing states keeps the shaping policy-invariant; a
    // non-finite state cannot be evaluated and is about to crash anyway.
    if (Absorbing(mission) || !IsFinite(aircraft))
    {
        return potentials;
    }
    auto const guidance = Guide(config, mission, aircraft);
    potentials[static_cast<std::size_t>(Reward::Path)] = PathValue(config, mission, aircraft);
    potentials[static_cast<std::size_t>(Reward::VerticalSpeed)] = VerticalSpeedValue(config, mission,
        guidance);
    potentials[static_cast<std::size_t>(Reward::Speed)] = SpeedValue(config, mission, aircraft, guidance);
    return potentials;
}

double DirectionCost(Guidance const& guidance, double dt) noexcept
{
    // Unit vectors have a squared length of one; the zero vector marks no motion.
    bool const moving = Dot(guidance.velocityDirection, guidance.velocityDirection)
        >= MovingDirectionMinimumSquaredLength;
    double alignment = StationaryAlignment;
    if (moving)
    {
        alignment = std::clamp(Dot(guidance.velocityDirection, guidance.targetDirection), -1.0, 1.0);
    }
    return -dt * DirectionCostScale * (1.0 - alignment);
}

double TouchdownQuality(Config const& config, MissionState const& mission) noexcept
{
    auto const& limits = config.mission;
    double const area = LandingRewardFactor(config.runway, mission.touchdownFromThreshold,
        mission.touchdownCrossMeters);
    double const zone = TouchdownZoneFactor(config.runway, mission.touchdownFromThreshold);
    double const sink = SinkFactor(limits, mission.touchdownSinkMps);
    double const speed = SpeedFactor(limits, mission.touchdownSpeedMps);
    return area * zone * sink * speed;
}
} // namespace ppo::examples::flight
