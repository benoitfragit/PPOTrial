#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace ppo::examples::twovstwo::defenders {

/**
 * @brief Configurable, stateless assessment of hostile pressure.
 */
struct DefenseSettings final
{
    float activationRisk{0.25F};
    float deactivationRisk{0.10F};
    float activationDistanceMeters{90000.F};
    float deactivationDistanceMeters{110000.F};
    float protectedRadiusMeters{12000.F};
    float timeToIntrusionSeconds{300.F};
    float teammateThreatDistanceMeters{30000.F};
    float teammateVulnerabilityWeight{0.75F};
};

/**
 * @brief Minimal kinematic track consumed by stateless threat assessment.
 */
struct DefenseTrack final
{
    float north{};
    float east{};
    float heading{};
    float speed{};
    bool available{};
};


/**
 * @brief Validates finite risk thresholds and their inner/outer ordering.
 * @param settings Candidate settings.
 */
inline bool Valid(DefenseSettings const& settings) noexcept
{
    auto const finite = [](float value) { return std::isfinite(value); };
    return finite(settings.activationRisk)
        && finite(settings.deactivationRisk)
        && settings.activationRisk > settings.deactivationRisk
        && settings.activationRisk <= 1.F
        && settings.deactivationRisk >= 0.F
        && finite(settings.activationDistanceMeters)
        && finite(settings.deactivationDistanceMeters)
        && settings.activationDistanceMeters > 0.F
        && settings.deactivationDistanceMeters
            > settings.activationDistanceMeters
        && finite(settings.protectedRadiusMeters)
        && settings.protectedRadiusMeters > 0.F
        && settings.protectedRadiusMeters
            < settings.activationDistanceMeters
        && finite(settings.timeToIntrusionSeconds)
        && settings.timeToIntrusionSeconds > 0.F
        && finite(settings.teammateThreatDistanceMeters)
        && settings.teammateThreatDistanceMeters > 0.F
        && finite(settings.teammateVulnerabilityWeight)
        && settings.teammateVulnerabilityWeight >= 0.F
        && settings.teammateVulnerabilityWeight <= 1.F;
}

namespace patrol_defense_detail {

inline float Clamp01(float value) noexcept
{
    return std::max(0.F, std::min(value, 1.F));
}

inline float Distance(DefenseTrack const& first,
                      DefenseTrack const& second) noexcept
{
    return std::hypot(first.north - second.north,
                      first.east - second.east);
}

inline std::array<float, 2> Velocity(DefenseTrack const& track) noexcept
{
    return {track.speed * std::cos(track.heading),
            track.speed * std::sin(track.heading)};
}

inline float ClosingSpeed(DefenseTrack const& source,
                          DefenseTrack const& target) noexcept
{
    float const north = target.north - source.north;
    float const east = target.east - source.east;
    float const distance = std::max(std::hypot(north, east), 1.F);
    auto const sourceVelocity = Velocity(source);
    auto const targetVelocity = Velocity(target);
    return ((sourceVelocity[0] - targetVelocity[0]) * north
          + (sourceVelocity[1] - targetVelocity[1]) * east) / distance;
}


} // namespace patrol_defense_detail

/**
 * @brief Scores objective danger from one red mobile without assuming a
 *        fixed number of blue defenders.
 * @return Risk in [0, 1] including distance, closing velocity and
 *         time-to-intrusion inside the configured protection radius.
 */
inline float ObjectiveThreatRisk(DefenseSettings const& settings,
                                 DefenseTrack const& attacker,
                                 DefenseTrack const& objective) noexcept
{
    using namespace patrol_defense_detail;
    if (!attacker.available)
        return 0.F;
    float const objectiveDistance = Distance(attacker, objective);
    float const objectiveClosing = ClosingSpeed(attacker, objective);
    float const penetration = Clamp01(
        (settings.protectedRadiusMeters - objectiveDistance)
            / settings.protectedRadiusMeters);
    float distanceRisk = 0.F;
    if (objectiveDistance <= settings.activationDistanceMeters)
    {
        float const proximity = 1.F - Clamp01(
            (objectiveDistance - settings.protectedRadiusMeters)
                / (settings.activationDistanceMeters
                   - settings.protectedRadiusMeters));
        distanceRisk = settings.activationRisk
            + (1.F - settings.activationRisk) * proximity;
    }
    else if (objectiveDistance < settings.deactivationDistanceMeters)
    {
        float const outerProximity = 1.F - Clamp01(
            (objectiveDistance - settings.activationDistanceMeters)
                / (settings.deactivationDistanceMeters
                   - settings.activationDistanceMeters));
        distanceRisk = settings.deactivationRisk * outerProximity;
    }
    float const approachFactor = objectiveClosing > 0.F ? 1.F : 0.15F;
    float timeRisk = 0.F;
    if (objectiveClosing > 1.F)
    {
        float const timeToIntrusion = std::max(
            objectiveDistance - settings.protectedRadiusMeters, 0.F)
            / objectiveClosing;
        timeRisk = 1.F - Clamp01(
            timeToIntrusion / settings.timeToIntrusionSeconds);
    }
    float const objectiveRisk = std::max(
        penetration, std::max(distanceRisk * approachFactor, timeRisk));
    return objectiveDistance <= settings.protectedRadiusMeters
        ? 1.F : Clamp01(objectiveRisk);
}

/**
 * @brief Computes the instantaneous credibility of one hostile track.
 * @param settings Valid risk thresholds.
 * @param attacker Hostile kinematics.
 * @param objective Fixed defended objective.
 * @param guards Current blue-team kinematics.
 * @return Risk in [0, 1], including objective and teammate vulnerability.
 */
inline float ThreatRisk(DefenseSettings const& settings,
                        DefenseTrack const& attacker,
                        DefenseTrack const& objective,
                        std::array<DefenseTrack, 2> const& guards) noexcept
{
    using namespace patrol_defense_detail;
    if (!attacker.available)
        return 0.F;

    float const objectiveRisk = ObjectiveThreatRisk(settings, attacker, objective);
    float teammateRisk = 0.F;
    for (DefenseTrack const& guard : guards)
    {
        if (!guard.available)
            continue;
        float const distance = Distance(attacker, guard);
        float const closing = ClosingSpeed(attacker, guard);
        float const proximity = 1.F - Clamp01(
            distance / settings.teammateThreatDistanceMeters);
        float const pursuit = closing > 0.F ? 1.F : 0.2F;
        teammateRisk = std::max(teammateRisk, proximity * pursuit);
    }
    return Clamp01(std::max(
        objectiveRisk,
        settings.teammateVulnerabilityWeight * teammateRisk));
}


} // namespace ppo::examples::twovstwo::defenders
