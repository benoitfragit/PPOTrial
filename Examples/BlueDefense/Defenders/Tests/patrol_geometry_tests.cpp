#include "patrol_config.hpp"
#include "patrol_geometry.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

namespace patrol = ppo::examples::twovstwo::defenders;

namespace {

nlohmann::json ValidDocument()
{
    return {
        {"patrol", {
            {"racetrack", {
                {"lengthMeters", {{"min", 50000.F}, {"max", 60000.F}}},
                {"widthMeters", {{"min", 16000.F}, {"max", 20000.F}}},
                {"headingDegrees", {{"min", 80.F}, {"max", 100.F}}},
                {"anchorNorthMeters", {{"min", -1500.F}, {"max", 1500.F}}},
                {"anchorEastMeters", {{"min", -2000.F}, {"max", 2000.F}}},
                {"clockwise", false},
                {"boundaryMarginMeters", 2000.F},
                {"maximumSamplingAttempts", 32U}
            }},
            {"formation", {
                {"separationDistanceMeters", {{"min", 6000.F}, {"max", 9000.F}}},
                {"separationAngleDegrees", {{"min", 10.F}, {"max", 25.F}}}
            }}
        }}
    };
}

bool Same(patrol::PatrolGeometry const& first,
          patrol::PatrolGeometry const& second) noexcept
{
    return first.anchorNorth == second.anchorNorth
        && first.anchorEast == second.anchorEast
        && first.heading == second.heading
        && first.length == second.length
        && first.width == second.width
        && first.separationDistance == second.separationDistance
        && first.separationAngle == second.separationAngle
        && first.direction == second.direction;
}

} // namespace

int main()
{
    patrol::PatrolGeometrySettings const settings =
        patrol::ParsePatrolGeometrySettings(ValidDocument());
    patrol::PatrolSamplingConstraints const constraints{
        22329.375F, 35727.F, 420.F, 0.50F, 1000.F};

    std::mt19937 firstRng{17U};
    std::mt19937 repeatRng{17U};
    auto const first = patrol::SamplePatrolGeometry(
        settings, constraints, firstRng);
    auto const repeated = patrol::SamplePatrolGeometry(
        settings, constraints, repeatRng);
    if (!first || !repeated || !Same(*first, *repeated))
        return 1;

    std::mt19937 differentRng{18U};
    auto const different = patrol::SamplePatrolGeometry(
        settings, constraints, differentRng);
    if (!different || Same(*first, *different))
        return 2;

    auto fixed = settings;
    fixed.lengthMeters = {55000.F, 55000.F};
    fixed.widthMeters = {18000.F, 18000.F};
    fixed.headingRadians = {0.5F * patrol::kPatrolPi,
                            0.5F * patrol::kPatrolPi};
    fixed.anchorNorthMeters = {0.F, 0.F};
    fixed.anchorEastMeters = {0.F, 0.F};
    fixed.separationDistanceMeters = {7500.F, 7500.F};
    fixed.separationAngleRadians = {0.25F, 0.25F};
    std::mt19937 fixedRngA{1U};
    std::mt19937 fixedRngB{999U};
    auto const fixedA = patrol::SamplePatrolGeometry(
        fixed, constraints, fixedRngA);
    auto const fixedB = patrol::SamplePatrolGeometry(
        fixed, constraints, fixedRngB);
    if (!fixedA || !fixedB || !Same(*fixedA, *fixedB))
        return 3;

    if (patrol::GeneralizedFormationFitsBounds(
            *fixedA, fixed, constraints, 9U, 0.F))
    {
        return 8;
    }
    std::mt19937 formationRng{71U};
    auto const boundedFormation =
        patrol::SampleGeneralizedFormationArclength(
            *fixedA, fixed, constraints, 9U, formationRng);
    if (!boundedFormation
        || !patrol::GeneralizedFormationFitsBounds(
            *fixedA, fixed, constraints, 9U, *boundedFormation)
        || patrol::GeneralizedFormationFitsBounds(
            *fixedA, fixed, constraints, 0U, *boundedFormation)
        || patrol::GeneralizedFormationFitsBounds(
            *fixedA, fixed, constraints, 10U, *boundedFormation))
    {
        return 9;
    }

    auto const leader = patrol::EvaluatePatrolCurve(
        *fixedA, *boundedFormation);
    for (std::size_t slot = 0U; slot < 8U; ++slot)
    {
        auto const teammate = patrol::EvaluateFormationPose(
            *fixedA, leader, slot);
        if (std::abs(teammate.north) + fixed.boundaryMarginMeters
                > constraints.halfNorth
            || std::abs(teammate.east) + fixed.boundaryMarginMeters
                > constraints.halfEast)
        {
            return 10;
        }
    }

    float const circuit = patrol::CircuitLength(*first);
    for (std::size_t sample = 0U; sample < 128U; ++sample)
    {
        float const arclength = circuit
            * static_cast<float>(sample) / 128.F;
        auto const pose = patrol::EvaluatePatrolCurve(*first, arclength);
        auto const projection = patrol::ProjectOnPatrolCurve(
            *first, pose.north, pose.east);
        auto const wingman = patrol::EvaluateWedgePose(*first, pose);
        float const separation = std::hypot(
            pose.north - wingman.north, pose.east - wingman.east);
        if (!std::isfinite(pose.heading)
            || projection.distance > 0.05F
            || std::abs(projection.signedCrossTrackError) > 0.05F
            || std::abs(separation - first->separationDistance) > 0.05F)
            return 4;
    }

    auto impossible = settings;
    impossible.widthMeters = {500.F, 500.F};
    impossible.lengthMeters = {100000.F, 100000.F};
    impossible.maximumSamplingAttempts = 2U;
    std::mt19937 impossibleRng{0U};
    if (patrol::SamplePatrolGeometry(
            impossible, constraints, impossibleRng))
        return 5;

    auto reversed = ValidDocument();
    reversed["patrol"]["racetrack"]["lengthMeters"] = {
        {"min", 2.F}, {"max", 1.F}};
    try
    {
        (void)patrol::ParsePatrolGeometrySettings(reversed);
        return 6;
    }
    catch (std::invalid_argument const&)
    {
    }

    auto nonfinite = ValidDocument();
    nonfinite["patrol"]["racetrack"]["widthMeters"]["min"] =
        std::numeric_limits<float>::infinity();
    try
    {
        (void)patrol::ParsePatrolGeometrySettings(nonfinite);
        return 7;
    }
    catch (std::invalid_argument const&)
    {
    }

    return 0;
}
