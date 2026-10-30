#pragma once

#include "blue_team_roles.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>

namespace ppo::examples::bluedefense::defenders {

inline constexpr float kPatrolPi = 3.14159265358979323846F;
inline constexpr float kPatrolTwoPi = 2.F * kPatrolPi;

/**
 * @brief Closed interval used by episode-level patrol randomization.
 */
struct PatrolRange final
{
    float minimum{};
    float maximum{};
};

/**
 * @brief User-configurable family of continuous racetrack geometries.
 *
 * Length and width are the overall outside dimensions of the stadium curve.
 * Heading is measured clockwise from geographic north. Formation angle is
 * measured from the leader's rear direction towards the inside of the loop.
 */
struct PatrolGeometrySettings final
{
    PatrolRange lengthMeters{50000.F, 60000.F};
    PatrolRange widthMeters{16000.F, 20000.F};
    PatrolRange headingRadians{1.3962634F, 1.7453293F};
    PatrolRange anchorNorthMeters{-1500.F, 1500.F};
    PatrolRange anchorEastMeters{-2000.F, 2000.F};
    PatrolRange separationDistanceMeters{6000.F, 9000.F};
    PatrolRange separationAngleRadians{0.17453293F, 0.43633232F};
    float boundaryMarginMeters{2000.F};
    std::size_t maximumSamplingAttempts{128U};
    bool clockwise{false};
};

/**
 * @brief One immutable racetrack and wedge sampled for an episode.
 */
struct PatrolGeometry final
{
    float anchorNorth{};
    float anchorEast{};
    float heading{0.5F * kPatrolPi};
    float length{55000.F};
    float width{11000.F};
    float separationDistance{7500.F};
    float separationAngle{0.30543262F};
    int direction{1};
};

/**
 * @brief Physical and map limits applied while sampling a patrol.
 */
struct PatrolSamplingConstraints final
{
    float halfNorth{};
    float halfEast{};
    float cruiseSpeed{};
    float maximumRotationRate{};
    float collisionDistance{};
};

/**
 * @brief Pose and local differential geometry of the analytical racetrack.
 */
struct PatrolCurvePose final
{
    float north{};
    float east{};
    float heading{};
    float arclength{};
    float phase{};
    float curvature{};
};

/**
 * @brief Nearest analytical curve location for an arbitrary mobile position.
 */
struct PatrolProjection final
{
    PatrolCurvePose pose{};
    float signedCrossTrackError{};
    float distance{};
};

namespace patrol_geometry_detail {

inline float Clamp(float value, float minimum, float maximum) noexcept
{
    return std::max(minimum, std::min(value, maximum));
}

inline float PositiveModulo(float value, float period) noexcept
{
    float const result = std::fmod(value, period);
    return result < 0.F ? result + period : result;
}

inline bool FiniteRange(PatrolRange const& range) noexcept
{
    return std::isfinite(range.minimum) && std::isfinite(range.maximum)
        && range.minimum <= range.maximum;
}

inline float Sample(PatrolRange const& range, std::mt19937& rng)
{
    if (range.minimum == range.maximum)
        return range.minimum;
    return std::uniform_real_distribution<float>{
        range.minimum, range.maximum}(rng);
}

struct LocalPose final
{
    float major{};
    float minor{};
    float tangentMajor{};
    float tangentMinor{};
    float curvature{};
};

inline float Radius(PatrolGeometry const& geometry) noexcept
{
    return 0.5F * geometry.width;
}

inline float StraightHalfLength(PatrolGeometry const& geometry) noexcept
{
    return 0.5F * (geometry.length - geometry.width);
}

inline float CircuitLength(PatrolGeometry const& geometry) noexcept
{
    return 4.F * StraightHalfLength(geometry)
        + kPatrolTwoPi * Radius(geometry);
}

inline LocalPose EvaluateCanonical(PatrolGeometry const& geometry,
                                   float canonicalArclength) noexcept
{
    float const half = StraightHalfLength(geometry);
    float const radius = Radius(geometry);
    float const straight = 2.F * half;
    float const arc = kPatrolPi * radius;
    float const s = PositiveModulo(
        canonicalArclength, 2.F * (straight + arc));

    LocalPose result{};
    if (s < straight)
    {
        result.major = -half + s;
        result.minor = -radius;
        result.tangentMajor = 1.F;
    }
    else if (s < straight + arc)
    {
        float const angle = -0.5F * kPatrolPi
            + (s - straight) / radius;
        result.major = half + radius * std::cos(angle);
        result.minor = radius * std::sin(angle);
        result.tangentMajor = -std::sin(angle);
        result.tangentMinor = std::cos(angle);
        result.curvature = 1.F / radius;
    }
    else if (s < 2.F * straight + arc)
    {
        float const onStraight = s - straight - arc;
        result.major = half - onStraight;
        result.minor = radius;
        result.tangentMajor = -1.F;
    }
    else
    {
        float const angle = 0.5F * kPatrolPi
            + (s - 2.F * straight - arc) / radius;
        result.major = -half + radius * std::cos(angle);
        result.minor = radius * std::sin(angle);
        result.tangentMajor = -std::sin(angle);
        result.tangentMinor = std::cos(angle);
        result.curvature = 1.F / radius;
    }
    return result;
}

inline std::array<float, 2> ToWorld(PatrolGeometry const& geometry,
                                    float major, float minor) noexcept
{
    float const axisNorth = std::cos(geometry.heading);
    float const axisEast = std::sin(geometry.heading);
    float const inwardNorth = -axisEast;
    float const inwardEast = axisNorth;
    return {
        geometry.anchorNorth + major * axisNorth + minor * inwardNorth,
        geometry.anchorEast + major * axisEast + minor * inwardEast
    };
}

inline std::array<float, 2> ToLocal(PatrolGeometry const& geometry,
                                    float north, float east) noexcept
{
    float const deltaNorth = north - geometry.anchorNorth;
    float const deltaEast = east - geometry.anchorEast;
    float const axisNorth = std::cos(geometry.heading);
    float const axisEast = std::sin(geometry.heading);
    return {
        deltaNorth * axisNorth + deltaEast * axisEast,
        -deltaNorth * axisEast + deltaEast * axisNorth
    };
}

inline float DistanceSquared(float firstMajor, float firstMinor,
                             float secondMajor, float secondMinor) noexcept
{
    float const major = firstMajor - secondMajor;
    float const minor = firstMinor - secondMinor;
    return major * major + minor * minor;
}

struct ProjectionCandidate final
{
    float canonicalArclength{};
    float distanceSquared{std::numeric_limits<float>::infinity()};
};

inline void KeepCloser(ProjectionCandidate candidate,
                       ProjectionCandidate& best) noexcept
{
    if (candidate.distanceSquared < best.distanceSquared)
        best = candidate;
}

} // namespace patrol_geometry_detail

/**
 * @brief Validates finite JSON ranges independently of a particular sample.
 * @param settings Candidate patrol family.
 * @return True when every interval and unit-domain constraint is valid.
 */
inline bool Valid(PatrolGeometrySettings const& settings) noexcept
{
    using patrol_geometry_detail::FiniteRange;
    return FiniteRange(settings.lengthMeters)
        && FiniteRange(settings.widthMeters)
        && FiniteRange(settings.headingRadians)
        && FiniteRange(settings.anchorNorthMeters)
        && FiniteRange(settings.anchorEastMeters)
        && FiniteRange(settings.separationDistanceMeters)
        && FiniteRange(settings.separationAngleRadians)
        && settings.lengthMeters.minimum > 0.F
        && settings.widthMeters.minimum > 0.F
        && settings.lengthMeters.maximum > settings.widthMeters.minimum
        && settings.separationDistanceMeters.minimum > 0.F
        && settings.separationAngleRadians.minimum > 0.F
        && settings.separationAngleRadians.maximum < 0.5F * kPatrolPi
        && std::isfinite(settings.boundaryMarginMeters)
        && settings.boundaryMarginMeters >= 0.F
        && settings.maximumSamplingAttempts > 0U;
}

/**
 * @brief Returns the centre-line circuit length in metres.
 * @param geometry Valid sampled geometry.
 */
inline float CircuitLength(PatrolGeometry const& geometry) noexcept
{
    return patrol_geometry_detail::CircuitLength(geometry);
}

/**
 * @brief Evaluates the leader's unique continuous patrol curve.
 * @param geometry Episode geometry.
 * @param teamArclength Continuous team progress in the configured direction.
 * @return Reference pose without any steering or acceleration command.
 */
inline PatrolCurvePose EvaluatePatrolCurve(PatrolGeometry const& geometry,
                                           float teamArclength) noexcept
{
    using namespace patrol_geometry_detail;
    float const length = patrol_geometry_detail::CircuitLength(geometry);
    float const canonicalArclength = geometry.direction >= 0
        ? PositiveModulo(teamArclength, length)
        : PositiveModulo(length - teamArclength, length);
    LocalPose local = EvaluateCanonical(geometry, canonicalArclength);
    local.tangentMajor *= static_cast<float>(geometry.direction);
    local.tangentMinor *= static_cast<float>(geometry.direction);
    local.curvature *= static_cast<float>(geometry.direction);

    auto const position = ToWorld(geometry, local.major, local.minor);
    float const tangentNorth = local.tangentMajor * std::cos(geometry.heading)
        - local.tangentMinor * std::sin(geometry.heading);
    float const tangentEast = local.tangentMajor * std::sin(geometry.heading)
        + local.tangentMinor * std::cos(geometry.heading);
    return {
        position[0], position[1], std::atan2(tangentEast, tangentNorth),
        PositiveModulo(teamArclength, length),
        PositiveModulo(teamArclength, length) / length,
        local.curvature
    };
}

/**
 * @brief Builds the desired wingman pose from the physical leader pose.
 * @param geometry Episode wedge geometry.
 * @param leader Current or predicted leader pose.
 * @return Rear/inward pose. It is not a second patrol curve.
 */
inline PatrolCurvePose EvaluateWedgePose(
    PatrolGeometry const& geometry, PatrolCurvePose const& leader) noexcept
{
    float const tangentNorth = std::cos(leader.heading);
    float const tangentEast = std::sin(leader.heading);
    float const inwardNorth = static_cast<float>(geometry.direction)
        * -tangentEast;
    float const inwardEast = static_cast<float>(geometry.direction)
        * tangentNorth;
    float const backward = geometry.separationDistance
        * std::cos(geometry.separationAngle);
    float const inward = geometry.separationDistance
        * std::sin(geometry.separationAngle);
    return {
        leader.north - backward * tangentNorth + inward * inwardNorth,
        leader.east - backward * tangentEast + inward * inwardEast,
        leader.heading, leader.arclength, leader.phase, leader.curvature
    };
}

/**
 * @brief Builds one generalized teammate pose from its stable formation slot.
 * @param geometry Episode patrol and V-formation geometry.
 * @param leader Current leader reference pose.
 * @param formationSlot Zero-based teammate slot in `[0, 7]`.
 * @return Leader-relative teammate pose, or the leader pose for an invalid
 * slot or invalid formation geometry.
 */
inline PatrolCurvePose EvaluateFormationPose(
    PatrolGeometry const& geometry,
    PatrolCurvePose const& leader,
    std::size_t const formationSlot) noexcept
{
    FormationOffset const offset = EvaluateFormationOffset(
        formationSlot, geometry.separationDistance,
        geometry.separationAngle);
    if (formationSlot >= TeamCapacity::MaximumBlueMobiles - 1U
        || (offset.longitudinalMeters == 0.F
            && offset.inwardMeters == 0.F))
    {
        return leader;
    }

    float const tangentNorth = std::cos(leader.heading);
    float const tangentEast = std::sin(leader.heading);
    float const inward = static_cast<float>(geometry.direction)
        * offset.inwardMeters;
    PatrolCurvePose result = leader;
    result.north += offset.longitudinalMeters * tangentNorth
        - inward * tangentEast;
    result.east += offset.longitudinalMeters * tangentEast
        + inward * tangentNorth;
    return result;
}

/**
 * @brief Checks every active generalized blue spawn against world bounds.
 * @param geometry Episode patrol and formation geometry.
 * @param settings Boundary margin configured for the patrol.
 * @param constraints Current world and vehicle constraints.
 * @param activeBlueCount Number of active blue mobiles in `[1, 9]`.
 * @param arclength Candidate leader arclength on the patrol circuit.
 * @return True when the leader and all active teammate slots are finite and
 * remain inside the configured world margin.
 */
inline bool GeneralizedFormationFitsBounds(
    PatrolGeometry const& geometry,
    PatrolGeometrySettings const& settings,
    PatrolSamplingConstraints const& constraints,
    std::size_t const activeBlueCount,
    float const arclength) noexcept
{
    if (activeBlueCount < TeamCapacity::MinimumBlueMobiles
        || activeBlueCount > TeamCapacity::MaximumBlueMobiles
        || !std::isfinite(arclength)
        || !std::isfinite(settings.boundaryMarginMeters)
        || settings.boundaryMarginMeters < 0.F
        || !std::isfinite(constraints.halfNorth)
        || !std::isfinite(constraints.halfEast)
        || constraints.halfNorth <= 0.F || constraints.halfEast <= 0.F)
    {
        return false;
    }

    auto const insideBounds = [&](PatrolCurvePose const& pose) noexcept {
        return std::isfinite(pose.north) && std::isfinite(pose.east)
            && std::abs(pose.north) + settings.boundaryMarginMeters
                <= constraints.halfNorth
            && std::abs(pose.east) + settings.boundaryMarginMeters
                <= constraints.halfEast;
    };

    PatrolCurvePose const leader = EvaluatePatrolCurve(
        geometry, arclength);
    if (!insideBounds(leader))
        return false;
    for (std::size_t formationSlot = 0U;
         formationSlot + 1U < activeBlueCount; ++formationSlot)
    {
        if (!insideBounds(EvaluateFormationPose(
                geometry, leader, formationSlot)))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Samples a bounded generalized-team spawn phase reproducibly.
 * @param geometry Already validated episode patrol geometry.
 * @param settings Patrol settings and bounded-attempt budget.
 * @param constraints Current world and vehicle constraints.
 * @param activeBlueCount Number of active blue mobiles in `[1, 9]`.
 * @param rng Explicit episode generator.
 * @return A phase for which all active blue poses fit, or `std::nullopt` when
 * the bounded attempt sequence is exhausted.
 */
inline std::optional<float> SampleGeneralizedFormationArclength(
    PatrolGeometry const& geometry,
    PatrolGeometrySettings const& settings,
    PatrolSamplingConstraints const& constraints,
    std::size_t const activeBlueCount,
    std::mt19937& rng)
{
    if (activeBlueCount < TeamCapacity::MinimumBlueMobiles
        || activeBlueCount > TeamCapacity::MaximumBlueMobiles
        || settings.maximumSamplingAttempts == 0U)
    {
        return std::nullopt;
    }

    float const circuit = CircuitLength(geometry);
    if (!std::isfinite(circuit) || circuit <= 0.F)
        return std::nullopt;
    std::uniform_real_distribution<float> phaseDistribution{0.F, circuit};
    for (std::size_t attempt = 0U;
         attempt < settings.maximumSamplingAttempts; ++attempt)
    {
        float const candidate = phaseDistribution(rng);
        if (GeneralizedFormationFitsBounds(
                geometry, settings, constraints, activeBlueCount, candidate))
        {
            return candidate;
        }
    }
    return std::nullopt;
}

/**
 * @brief Find a deterministic feasible evaluation phase on this episode's
 * newly sampled geometry, starting closest to the requested quarter-circuit.
 * The bounded grid search consumes no random numbers.
 */
inline std::optional<float> FindGeneralizedEvaluationArclength(
    PatrolGeometry const& geometry,
    PatrolGeometrySettings const& settings,
    PatrolSamplingConstraints const& constraints,
    std::size_t const activeBlueCount,
    float const preferredPhaseFraction) noexcept
{
    if (!std::isfinite(preferredPhaseFraction)
        || preferredPhaseFraction < 0.F || preferredPhaseFraction >= 1.F)
        return std::nullopt;
    float const circuit = CircuitLength(geometry);
    if (!std::isfinite(circuit) || circuit <= 0.F)
        return std::nullopt;

    constexpr std::size_t candidateCount = 256U;
    for (std::size_t attempt = 0U; attempt < candidateCount; ++attempt)
    {
        int const offset = attempt == 0U ? 0
            : (attempt % 2U != 0U
                ? static_cast<int>((attempt + 1U) / 2U)
                : -static_cast<int>(attempt / 2U));
        float const phase = PositiveModulo(
            preferredPhaseFraction + static_cast<float>(offset)
                / static_cast<float>(candidateCount), 1.F);
        float const candidate = circuit * phase;
        if (GeneralizedFormationFitsBounds(
                geometry, settings, constraints, activeBlueCount, candidate))
            return candidate;
    }
    return std::nullopt;
}

/**
 * @brief Projects a position onto the leader's unique patrol curve.
 * @param geometry Episode geometry.
 * @param north World north coordinate in metres.
 * @param east World east coordinate in metres.
 * @return Nearest curve pose and signed cross-track error.
 */
inline PatrolProjection ProjectOnPatrolCurve(PatrolGeometry const& geometry,
                                              float north,
                                              float east) noexcept
{
    using namespace patrol_geometry_detail;
    auto const local = ToLocal(geometry, north, east);
    float const major = local[0];
    float const minor = local[1];
    float const half = StraightHalfLength(geometry);
    float const centreRadius = Radius(geometry);
    float const radius = centreRadius;
    float const straight = 2.F * half;
    float const arc = kPatrolPi * centreRadius;

    ProjectionCandidate best{};
    float const straightMajor = Clamp(major, -half, half);
    KeepCloser({straightMajor + half,
                DistanceSquared(major, minor, straightMajor, -radius)}, best);
    KeepCloser({straight + arc + (half - straightMajor),
                DistanceSquared(major, minor, straightMajor, radius)}, best);

    float rightAngle = std::atan2(minor, major - half);
    rightAngle = Clamp(rightAngle, -0.5F * kPatrolPi, 0.5F * kPatrolPi);
    float const rightMajor = half + radius * std::cos(rightAngle);
    float const rightMinor = radius * std::sin(rightAngle);
    KeepCloser({straight + (rightAngle + 0.5F * kPatrolPi) * centreRadius,
                DistanceSquared(major, minor, rightMajor, rightMinor)}, best);

    float leftAngle = std::atan2(minor, major + half);
    if (leftAngle < 0.5F * kPatrolPi)
        leftAngle += kPatrolTwoPi;
    leftAngle = Clamp(leftAngle, 0.5F * kPatrolPi, 1.5F * kPatrolPi);
    float const leftMajor = -half + radius * std::cos(leftAngle);
    float const leftMinor = radius * std::sin(leftAngle);
    KeepCloser({2.F * straight + arc
                    + (leftAngle - 0.5F * kPatrolPi) * centreRadius,
                DistanceSquared(major, minor, leftMajor, leftMinor)}, best);

    float const length = patrol_geometry_detail::CircuitLength(geometry);
    float directedArclength = geometry.direction >= 0
        ? best.canonicalArclength
        : PositiveModulo(length - best.canonicalArclength, length);
    PatrolCurvePose const pose = EvaluatePatrolCurve(
        geometry, directedArclength);
    float const deltaNorth = north - pose.north;
    float const deltaEast = east - pose.east;
    float const leftNorth = -std::sin(pose.heading);
    float const leftEast = std::cos(pose.heading);
    return {pose,
            deltaNorth * leftNorth + deltaEast * leftEast,
            std::sqrt(best.distanceSquared)};
}

/**
 * @brief Validates one sampled geometry against vehicle and map constraints.
 * @param geometry Sample to validate.
 * @param settings Source settings containing the requested boundary margin.
 * @param constraints World, dynamics, and collision limits.
 */
inline bool Valid(PatrolGeometry const& geometry,
                  PatrolGeometrySettings const& settings,
                  PatrolSamplingConstraints const& constraints) noexcept
{
    using namespace patrol_geometry_detail;
    if (!std::isfinite(geometry.anchorNorth)
        || !std::isfinite(geometry.anchorEast)
        || !std::isfinite(geometry.heading)
        || !std::isfinite(geometry.length) || !std::isfinite(geometry.width)
        || !std::isfinite(geometry.separationDistance)
        || !std::isfinite(geometry.separationAngle)
        || geometry.length <= geometry.width || geometry.width <= 0.F
        || geometry.separationDistance <= 0.F
        || geometry.separationAngle <= 0.F
        || geometry.separationAngle >= 0.5F * kPatrolPi
        || std::abs(geometry.direction) != 1
        || !std::isfinite(constraints.halfNorth)
        || !std::isfinite(constraints.halfEast)
        || !std::isfinite(constraints.cruiseSpeed)
        || !std::isfinite(constraints.maximumRotationRate)
        || !std::isfinite(constraints.collisionDistance)
        || constraints.halfNorth <= 0.F || constraints.halfEast <= 0.F
        || constraints.cruiseSpeed <= 0.F
        || constraints.maximumRotationRate <= 0.F
        || constraints.collisionDistance <= 0.F)
        return false;

    float const minimumTurnRadius = constraints.cruiseSpeed
        / constraints.maximumRotationRate;
    if (Radius(geometry) < minimumTurnRadius
        || geometry.separationDistance <= constraints.collisionDistance)
        return false;

    float const halfLength = 0.5F * geometry.length;
    float const halfWidth = 0.5F * geometry.width;
    float const northExtent = std::abs(std::cos(geometry.heading)) * halfLength
        + std::abs(std::sin(geometry.heading)) * halfWidth;
    float const eastExtent = std::abs(std::sin(geometry.heading)) * halfLength
        + std::abs(std::cos(geometry.heading)) * halfWidth;
    if (std::abs(geometry.anchorNorth) + northExtent
            + settings.boundaryMarginMeters > constraints.halfNorth
        || std::abs(geometry.anchorEast) + eastExtent
            + settings.boundaryMarginMeters > constraints.halfEast)
        return false;

    float const circuit = patrol_geometry_detail::CircuitLength(geometry);
    for (std::size_t sample = 0U; sample < 64U; ++sample)
    {
        float const arclength = circuit * static_cast<float>(sample) / 64.F;
        auto const leader = EvaluatePatrolCurve(geometry, arclength);
        auto const wingman = EvaluateWedgePose(geometry, leader);
        auto const insideBounds = [&](PatrolCurvePose const& pose) {
            return std::abs(pose.north) + settings.boundaryMarginMeters
                    <= constraints.halfNorth
                && std::abs(pose.east) + settings.boundaryMarginMeters
                    <= constraints.halfEast;
        };
        if (!insideBounds(leader) || !insideBounds(wingman))
            return false;
    }
    return true;
}

/**
 * @brief Samples a bounded, reproducible episode geometry.
 * @param settings Valid configuration ranges.
 * @param constraints Current world and vehicle limits.
 * @param rng Explicit episode generator.
 * @return A feasible geometry, or std::nullopt after the configured attempts.
 */
inline std::optional<PatrolGeometry> SamplePatrolGeometry(
    PatrolGeometrySettings const& settings,
    PatrolSamplingConstraints const& constraints,
    std::mt19937& rng)
{
    if (!Valid(settings))
        return std::nullopt;
    for (std::size_t attempt = 0U;
         attempt < settings.maximumSamplingAttempts; ++attempt)
    {
        PatrolGeometry candidate{};
        candidate.anchorNorth = patrol_geometry_detail::Sample(
            settings.anchorNorthMeters, rng);
        candidate.anchorEast = patrol_geometry_detail::Sample(
            settings.anchorEastMeters, rng);
        candidate.heading = patrol_geometry_detail::Sample(
            settings.headingRadians, rng);
        candidate.length = patrol_geometry_detail::Sample(
            settings.lengthMeters, rng);
        candidate.width = patrol_geometry_detail::Sample(
            settings.widthMeters, rng);
        candidate.separationDistance = patrol_geometry_detail::Sample(
            settings.separationDistanceMeters, rng);
        candidate.separationAngle = patrol_geometry_detail::Sample(
            settings.separationAngleRadians, rng);
        candidate.direction = settings.clockwise ? -1 : 1;
        if (Valid(candidate, settings, constraints))
            return candidate;
    }
    return std::nullopt;
}

} // namespace ppo::examples::bluedefense::defenders
