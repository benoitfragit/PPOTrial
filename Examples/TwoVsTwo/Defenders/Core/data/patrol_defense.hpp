#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace ppo::examples::twovstwo::defenders {

/**
 * @brief Configurable threat classification, assignment, and recovery policy.
 */
struct DefenseSettings final
{
    float activationRisk{0.55F};
    float deactivationRisk{0.35F};
    float activationDistanceMeters{25000.F};
    float deactivationDistanceMeters{35000.F};
    float protectedRadiusMeters{12000.F};
    float timeToIntrusionSeconds{60.F};
    float teammateThreatDistanceMeters{12000.F};
    float teammateVulnerabilityWeight{0.75F};
    float minimumAssignmentSeconds{8.F};
    float deactivationHoldSeconds{5.F};
    float recoverySeconds{15.F};
    float switchImprovementFraction{0.15F};
};

/**
 * @brief Minimal kinematic track consumed by the defense coordinator.
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
 * @brief Stable tactical decision exposed to observations and rewards.
 */
struct DefenseDecision final
{
    static constexpr std::size_t Unassigned =
        std::numeric_limits<std::size_t>::max();

    std::array<std::size_t, 2> attackerForGuard{Unassigned, Unassigned};
    std::array<float, 2> threatRisk{};
    std::array<float, 2> defenseWeight{};
    std::size_t credibleThreats{};
};

/**
 * @brief Validates all defense thresholds and their hysteresis ordering.
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
        && settings.teammateVulnerabilityWeight <= 1.F
        && finite(settings.minimumAssignmentSeconds)
        && settings.minimumAssignmentSeconds >= 0.F
        && finite(settings.deactivationHoldSeconds)
        && settings.deactivationHoldSeconds >= 0.F
        && finite(settings.recoverySeconds)
        && settings.recoverySeconds > 0.F
        && finite(settings.switchImprovementFraction)
        && settings.switchImprovementFraction >= 0.F
        && settings.switchImprovementFraction < 1.F;
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

inline float InterceptCost(DefenseTrack const& guard,
                           DefenseTrack const& attacker) noexcept
{
    float const distance = Distance(guard, attacker);
    float const bearing = std::atan2(
        attacker.east - guard.east, attacker.north - guard.north);
    float const alignment = 0.5F * (1.F + std::cos(bearing - guard.heading));
    float const effectiveSpeed = std::max(guard.speed, 100.F);
    return distance / effectiveSpeed + 5.F * (1.F - alignment);
}

} // namespace patrol_defense_detail

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
    float objectiveRisk = std::max(
        penetration, std::max(distanceRisk * approachFactor, timeRisk));
    if (objectiveDistance <= settings.protectedRadiusMeters)
        objectiveRisk = 1.F;

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

/**
 * @brief Stateful hysteresis and one-to-one threat assignment coordinator.
 *
 * This class chooses tactical roles only. It never produces a vehicle command;
 * both steering and acceleration remain outputs of the decentralized Actor.
 */
class DefenseCoordinator final
{
public:
    /** @brief Clears all threat latches, assignments, and recovery weights. */
    void Reset() noexcept
    {
        credible_.fill(false);
        belowThresholdSteps_.fill(0U);
        assignment_.fill(DefenseDecision::Unassigned);
        defenseWeight_.fill(0.F);
        assignmentHeldSteps_ = 0U;
    }

    /**
     * @brief Updates risk latches and stable guard assignments once per ECS step.
     * @param settings Valid defense settings.
     * @param timeStep Simulation step duration in seconds.
     * @param guards Current blue tracks.
     * @param attackers Current hostile tracks.
     * @param objective Fixed defended objective.
     * @return Decision to expose to Actor observations and reward computation.
     */
    DefenseDecision Update(DefenseSettings const& settings,
                           float timeStep,
                           std::array<DefenseTrack, 2> const& guards,
                           std::array<DefenseTrack, 2> const& attackers,
                           DefenseTrack const& objective) noexcept
    {
        DefenseDecision result{};
        if (!Valid(settings) || !std::isfinite(timeStep) || timeStep <= 0.F)
            return result;

        std::size_t const releaseSteps = std::max<std::size_t>(1U,
            static_cast<std::size_t>(std::ceil(
                settings.deactivationHoldSeconds / timeStep)));
        std::array<std::size_t, 2> active{
            DefenseDecision::Unassigned, DefenseDecision::Unassigned};
        for (std::size_t threat = 0U; threat < attackers.size(); ++threat)
        {
            result.threatRisk[threat] = ThreatRisk(
                settings, attackers[threat], objective, guards);
            if (!attackers[threat].available)
            {
                credible_[threat] = false;
                belowThresholdSteps_[threat] = 0U;
                continue;
            }
            if (!credible_[threat]
                && result.threatRisk[threat] >= settings.activationRisk)
            {
                credible_[threat] = true;
                belowThresholdSteps_[threat] = 0U;
            }
            else if (credible_[threat]
                     && result.threatRisk[threat] <= settings.deactivationRisk)
            {
                ++belowThresholdSteps_[threat];
                if (belowThresholdSteps_[threat] >= releaseSteps)
                {
                    credible_[threat] = false;
                    belowThresholdSteps_[threat] = 0U;
                }
            }
            else
                belowThresholdSteps_[threat] = 0U;

            if (credible_[threat])
                active[result.credibleThreats++] = threat;
        }

        auto candidate = BestAssignment(guards, attackers, active,
                                        result.credibleThreats);
        bool const existingValid = AssignmentValid(
            assignment_, active, result.credibleThreats);
        std::size_t const minimumHoldSteps = static_cast<std::size_t>(std::ceil(
            settings.minimumAssignmentSeconds / timeStep));
        bool retain = existingValid
            && assignmentHeldSteps_ < minimumHoldSteps;
        if (existingValid && !retain)
        {
            float const currentCost = AssignmentCost(
                assignment_, guards, attackers);
            float const candidateCost = AssignmentCost(
                candidate, guards, attackers);
            retain = candidateCost >= currentCost
                * (1.F - settings.switchImprovementFraction);
        }

        if (!retain && candidate != assignment_)
        {
            assignment_ = candidate;
            assignmentHeldSteps_ = 0U;
        }
        else
            ++assignmentHeldSteps_;
        if (result.credibleThreats == 0U)
        {
            assignment_.fill(DefenseDecision::Unassigned);
            assignmentHeldSteps_ = 0U;
        }

        float const recoveryStep = timeStep / settings.recoverySeconds;
        for (std::size_t guard = 0U; guard < assignment_.size(); ++guard)
        {
            if (assignment_[guard] != DefenseDecision::Unassigned)
                defenseWeight_[guard] = 1.F;
            else
                defenseWeight_[guard] = defenseWeight_[guard]
                        <= recoveryStep + 1.e-6F
                    ? 0.F : defenseWeight_[guard] - recoveryStep;
        }
        result.attackerForGuard = assignment_;
        result.defenseWeight = defenseWeight_;
        return result;
    }

private:
    using Assignment = std::array<std::size_t, 2>;

    static Assignment BestAssignment(
        std::array<DefenseTrack, 2> const& guards,
        std::array<DefenseTrack, 2> const& attackers,
        std::array<std::size_t, 2> const& active,
        std::size_t count) noexcept
    {
        Assignment result{DefenseDecision::Unassigned,
                          DefenseDecision::Unassigned};
        if (count == 1U)
        {
            std::size_t const threat = active[0];
            std::size_t const guard =
                patrol_defense_detail::InterceptCost(guards[0], attackers[threat])
                <= patrol_defense_detail::InterceptCost(
                    guards[1], attackers[threat]) ? 0U : 1U;
            result[guard] = threat;
        }
        else if (count == 2U)
        {
            float const direct =
                patrol_defense_detail::InterceptCost(
                    guards[0], attackers[active[0]])
                + patrol_defense_detail::InterceptCost(
                    guards[1], attackers[active[1]]);
            float const crossed =
                patrol_defense_detail::InterceptCost(
                    guards[0], attackers[active[1]])
                + patrol_defense_detail::InterceptCost(
                    guards[1], attackers[active[0]]);
            bool const useDirect = direct <= crossed;
            result[0] = active[useDirect ? 0U : 1U];
            result[1] = active[useDirect ? 1U : 0U];
        }
        return result;
    }

    static bool AssignmentValid(
        Assignment const& assignment,
        std::array<std::size_t, 2> const& active,
        std::size_t count) noexcept
    {
        std::size_t assigned = 0U;
        std::array<bool, 2> covered{};
        for (std::size_t threat : assignment)
        {
            if (threat == DefenseDecision::Unassigned)
                continue;
            ++assigned;
            for (std::size_t index = 0U; index < count; ++index)
                if (threat == active[index])
                {
                    if (covered[index])
                        return false;
                    covered[index] = true;
                }
        }
        return assigned == count
            && std::all_of(covered.begin(), covered.begin() + count,
                           [](bool value) { return value; });
    }

    static float AssignmentCost(
        Assignment const& assignment,
        std::array<DefenseTrack, 2> const& guards,
        std::array<DefenseTrack, 2> const& attackers) noexcept
    {
        float result = 0.F;
        for (std::size_t guard = 0U; guard < assignment.size(); ++guard)
            if (assignment[guard] != DefenseDecision::Unassigned)
                result += patrol_defense_detail::InterceptCost(
                    guards[guard], attackers[assignment[guard]]);
        return result;
    }

    std::array<bool, 2> credible_{};
    std::array<std::size_t, 2> belowThresholdSteps_{};
    Assignment assignment_{DefenseDecision::Unassigned,
                           DefenseDecision::Unassigned};
    std::array<float, 2> defenseWeight_{};
    std::size_t assignmentHeldSteps_{};
};

} // namespace ppo::examples::twovstwo::defenders
