/**
 * @file generalized_patrol_waypoint_sequencer_tests.cpp
 * @brief Deterministic RALLY/FOLLOW, waypoint, wedge and timeout regressions.
 */
#include "generalized_patrol_waypoint_sequencer.hpp"

#include <cmath>
#include <iostream>

namespace blue = ppo::examples::bluedefense::defenders;
namespace {

blue::Parameters TestParameters(int const direction = 1)
{
    blue::Parameters parameters{};
    parameters.dt = 0.2F;
    parameters.cruiseSpeed = 300.F;
    parameters.maxSpeed = 750.F;
    parameters.geometry.length = 40000.F;
    parameters.geometry.width = 16000.F;
    parameters.geometry.separationDistance = 2800.F;
    parameters.geometry.separationAngle = 0.55F;
    parameters.geometry.direction = direction;
    return parameters;
}

blue::GeneralizedRuntimeState StateAt(
    blue::Parameters const& parameters, blue::BlueTeamRoles const& roles,
    float const arclength)
{
    blue::GeneralizedRuntimeState state{};
    state.counts = {roles.ActiveBlueCount(), 0U};
    blue::PatrolCurvePose const leader = blue::EvaluatePatrolCurve(
        parameters.geometry, arclength);
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        blue::BlueRoleAssignment const assignment = *roles.Assignment(index);
        blue::PatrolFormationReference const reference =
            assignment.role == blue::BlueRole::Leader
            ? blue::PatrolFormationReference{leader, parameters.cruiseSpeed}
            : blue::EvaluateFormationReference(parameters.geometry, leader,
                assignment.formationSlot, parameters.cruiseSpeed);
        state.blues[index] = {reference.pose.north, reference.pose.east,
            reference.pose.heading, reference.speed, 0.F, true};
    }
    return state;
}

bool Near(float const first, float const second,
          float const tolerance = 0.05F)
{
    return std::abs(first - second) <= tolerance;
}

/** @brief Arms and confirms RALLY acquisition on two consecutive samples. */
bool EnterFollow(
    blue::GeneralizedPatrolWaypointSequencer& sequencer,
    blue::Parameters const& parameters,
    blue::GeneralizedRuntimeState const& state,
    blue::BlueTeamRoles const& roles,
    blue::PatrolNoThreatSettings const& settings)
{
    return sequencer.Observe(parameters, state, state, roles, settings)
            == blue::GeneralizedWaypointUpdate::None
        && !sequencer.Following()
        && sequencer.Observe(parameters, state, state, roles, settings)
            == blue::GeneralizedWaypointUpdate::EnteredFollow
        && sequencer.Following();
}

bool RallyAndGeneration()
{
    blue::Parameters parameters = TestParameters();
    auto const roles = *blue::BlueTeamRoles::Create(1U, 0U);
    blue::PatrolNoThreatSettings settings{};
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    auto far = StateAt(parameters, roles, 1700.F);
    float const heading = far.blues[0].heading;
    far.blues[0].north += 4000.F * -std::sin(heading);
    far.blues[0].east += 4000.F * std::cos(heading);
    if (sequencer.Observe(parameters, far, far, roles, settings)
            != blue::GeneralizedWaypointUpdate::None
        || sequencer.Mode() != blue::GeneralizedPatrolMode::Rally)
    {
        return false;
    }

    auto deadband = StateAt(parameters, roles, 1700.F);
    deadband.blues[0].north += 2700.F * -std::sin(heading);
    deadband.blues[0].east += 2700.F * std::cos(heading);
    if (sequencer.Observe(parameters, far, deadband, roles, settings)
            != blue::GeneralizedWaypointUpdate::None
        || sequencer.Mode() != blue::GeneralizedPatrolMode::Rally)
    {
        return false;
    }

    auto acquired = StateAt(parameters, roles, 1700.F);
    acquired.blues[0].north += 2490.F * -std::sin(heading);
    acquired.blues[0].east += 2490.F * std::cos(heading);
    if (sequencer.Observe(parameters, deadband, acquired, roles, settings)
            != blue::GeneralizedWaypointUpdate::None
        || sequencer.Following() || sequencer.ActiveWaypoint() != nullptr)
    {
        return false;
    }

    // Crossing the outer threshold cancels the first acquired sample. A
    // subsequent deadband sample alone must not confirm the cancelled latch.
    if (sequencer.Observe(parameters, acquired, far, roles, settings)
            != blue::GeneralizedWaypointUpdate::None
        || sequencer.Observe(parameters, far, deadband, roles, settings)
            != blue::GeneralizedWaypointUpdate::None
        || sequencer.Following())
    {
        return false;
    }

    // Reacquisition is retained in the deadband and confirmed on the next
    // resolved step, making both thresholds observable before FOLLOW.
    if (sequencer.Observe(parameters, deadband, acquired, roles, settings)
            != blue::GeneralizedWaypointUpdate::None
        || sequencer.Following()
        || sequencer.Observe(parameters, acquired, deadband, roles, settings)
            != blue::GeneralizedWaypointUpdate::EnteredFollow
        || !sequencer.Following()
        || sequencer.ActiveWaypointIndex() != 0U)
    {
        return false;
    }

    float const circuit = blue::CircuitLength(parameters.geometry);
    float const spacing = circuit
        / static_cast<float>(blue::GeneralizedPatrolWaypointSequencer::WaypointCount);
    for (std::size_t index = 0U;
         index < blue::GeneralizedPatrolWaypointSequencer::WaypointCount; ++index)
    {
        auto const& waypoint = sequencer.Waypoints()[index];
        auto const expected = blue::EvaluatePatrolCurve(
            parameters.geometry,
            1700.F + spacing * static_cast<float>(index + 1U));
        if (!Near(waypoint.pose.north, expected.north)
            || !Near(waypoint.pose.east, expected.east)
            || !Near(std::remainder(
                    waypoint.pose.heading - expected.heading,
                    blue::kPatrolTwoPi), 0.F)
            || !Near(waypoint.speed, parameters.cruiseSpeed)
            || !Near(waypoint.requiredDistanceMeters,
                spacing * static_cast<float>(index + 1U)))
        {
            return false;
        }
    }
    // Once confirmed, FOLLOW and its frozen waypoints cannot be reset by
    // leaving the outer corridor.
    return sequencer.Observe(parameters, far, far, roles, settings)
            == blue::GeneralizedWaypointUpdate::None
        && sequencer.Following()
        && sequencer.ActiveWaypointIndex() == 0U
        && Near(sequencer.Waypoints()[0].pose.arclength, 1700.F + spacing);
}

bool CompleteSequence(int const direction, std::size_t const blueCount,
                      bool const wingmanOffFormation = false)
{
    blue::Parameters parameters = TestParameters(direction);
    auto const roles = *blue::BlueTeamRoles::Create(
        blueCount, blueCount - 1U);
    blue::PatrolNoThreatSettings settings{};
    settings.minimumQualifiedFraction = 0.95F;
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    auto initial = StateAt(parameters, roles, 0.F);
    if (!EnterFollow(sequencer, parameters, initial, roles, settings))
    {
        return false;
    }

    constexpr std::size_t steps = 2000U;
    float const circuit = blue::CircuitLength(parameters.geometry);
    std::size_t previousCount = 0U;
    for (std::size_t step = 0U; step < steps; ++step)
    {
        auto const before = StateAt(
            parameters, roles, circuit * static_cast<float>(step) / steps);
        auto after = StateAt(parameters, roles,
            circuit * static_cast<float>(step + 1U) / steps);
        if (wingmanOffFormation && blueCount > 1U)
        {
            // Wingman violations must not block physically crossed leader gates.
            after.blues[0].heading += 1.F;
            after.blues[0].speed += 100.F;
        }
        sequencer.Observe(parameters, before, after, roles, settings);
        std::size_t const count = sequencer.ValidatedWaypointCount();
        if (count < previousCount || count > previousCount + 1U)
            return false;
        previousCount = count;
    }
    return sequencer.Completed() && sequencer.CircuitValid()
        && sequencer.ValidatedWaypointCount() == 10U
        && sequencer.ActiveWaypointIndex() == 10U;
}

bool NoPrematureAdvanceOrShortcut()
{
    blue::Parameters parameters = TestParameters();
    auto const roles = *blue::BlueTeamRoles::Create(1U, 0U);
    blue::PatrolNoThreatSettings settings{};
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    auto const start = StateAt(parameters, roles, 0.F);
    if (!EnterFollow(sequencer, parameters, start, roles, settings))
        return false;
    float const circuit = blue::CircuitLength(parameters.geometry);
    auto const oneStep = StateAt(parameters, roles, 60.F);
    sequencer.Observe(parameters, start, oneStep, roles, settings);
    if (sequencer.ValidatedWaypointCount() != 0U
        || !sequencer.CircuitValid())
    {
        return false;
    }

    blue::GeneralizedPatrolWaypointSequencer shortcut{};
    if (!EnterFollow(shortcut, parameters, start, roles, settings))
        return false;
    auto const halfway = StateAt(parameters, roles, 0.04F * circuit);
    shortcut.Observe(parameters, start, halfway, roles, settings);
    // A physically impossible branch-cut shortcut invalidates the lap and
    // cannot advance WPT1.
    if (shortcut.ValidatedWaypointCount() != 0U
        || shortcut.CircuitValid() || shortcut.Following()
        || shortcut.ActiveWaypoint() != nullptr)
        return false;
    // Recovery starts a new lap and never inherits the shortcut's progress.
    return EnterFollow(shortcut, parameters, halfway, roles, settings)
        && shortcut.CircuitValid() && shortcut.ValidatedWaypointCount() == 0U;
}

bool OrbitCannotAdvanceSequence()
{
    blue::Parameters parameters = TestParameters();
    auto const roles = *blue::BlueTeamRoles::Create(1U, 0U);
    blue::PatrolNoThreatSettings settings{};
    blue::GeneralizedPatrolWaypointSequencer sequencer{};
    auto previous = StateAt(parameters, roles, 0.F);
    if (!EnterFollow(sequencer, parameters, previous, roles, settings))
        return false;

    float const spacing = blue::CircuitLength(parameters.geometry)
        / static_cast<float>(
            blue::GeneralizedPatrolWaypointSequencer::WaypointCount);
    constexpr std::size_t approachSteps = 400U;
    for (std::size_t step = 1U; step <= approachSteps; ++step)
    {
        auto const current = StateAt(parameters, roles,
            spacing * static_cast<float>(step)
                / static_cast<float>(approachSteps));
        sequencer.Observe(parameters, previous, current, roles, settings);
        previous = current;
    }
    if (sequencer.ValidatedWaypointCount() != 1U)
        return false;

    blue::PatrolCurvePose const centre = sequencer.Waypoints()[0].pose;
    constexpr float orbitRadius = 100.F;
    constexpr std::size_t orbitSteps = 720U;
    for (std::size_t step = 0U; step < orbitSteps; ++step)
    {
        float const angle = blue::kPatrolTwoPi
            * static_cast<float>(step) / 120.F;
        auto current = previous;
        current.blues[0].north = centre.north
            + orbitRadius * std::cos(angle);
        current.blues[0].east = centre.east
            + orbitRadius * std::sin(angle);
        current.blues[0].heading = std::remainder(
            angle + 0.5F * blue::kPatrolPi, blue::kPatrolTwoPi);
        current.blues[0].speed = parameters.cruiseSpeed;
        sequencer.Observe(parameters, previous, current, roles, settings);
        previous = current;
    }
    // Recovery may restart at WPT1; local orbiting must never complete a lap.
    return !sequencer.Completed();
}

bool WedgeAndCombatCompatibility()
{
    blue::Parameters parameters = TestParameters(-1);
    blue::PatrolNoThreatSettings settings{};
    for (std::size_t count : {2U, 3U, 4U})
    {
        auto const roles = *blue::BlueTeamRoles::Create(count, count - 1U);
        auto state = StateAt(parameters, roles, 2500.F);
        blue::GeneralizedPatrolWaypointSequencer sequencer{};
        if (!EnterFollow(sequencer, parameters, state, roles, settings))
            return false;
        auto const targets = sequencer.Targets(
            parameters, state, roles, settings);
        auto const* active = sequencer.ActiveWaypoint();
        if (active == nullptr)
            return false;
        for (std::size_t index = 0U; index < count; ++index)
        {
            auto const assignment = *roles.Assignment(index);
            auto const expected = assignment.role == blue::BlueRole::Leader
                ? blue::PatrolFormationReference{
                    active->pose, parameters.cruiseSpeed}
                : blue::EvaluateFormationReference(
                    parameters.geometry, active->pose,
                    assignment.formationSlot, parameters.cruiseSpeed);
            if (!targets[index].valid
                || !Near(targets[index].guidancePose.north,
                    expected.pose.north)
                || !Near(targets[index].guidancePose.east,
                    expected.pose.east))
            {
                return false;
            }
        }

        for (std::size_t redCount : {1U, 2U})
        {
            state.counts.redMobiles = redCount;
            for (std::size_t red = 0U; red < redCount; ++red)
            {
                state.reds[red] = {100000.F + 1000.F * red,
                    100000.F, 0.F, 250.F, 0.F, true};
            }
            auto const combat = sequencer.Targets(
                parameters, state, roles, settings);
            auto const legacy = blue::ComputePatrolTargets(
                parameters, state, roles, settings.lookaheadSeconds);
            if (!Near(combat[roles.LeaderPhysicalIndex()].guidancePose.north,
                    legacy[roles.LeaderPhysicalIndex()].guidancePose.north)
                || !Near(combat[roles.LeaderPhysicalIndex()].guidancePose.east,
                    legacy[roles.LeaderPhysicalIndex()].guidancePose.east))
            {
                return false;
            }
        }
    }
    return true;
}

bool TimeoutConversion()
{
    blue::PatrolNoThreatSettings settings{};
    return blue::NoThreatEpisodeLimitSteps(settings, 0.2F, 4096U) == 2400U
        && blue::NoThreatEpisodeLimitSteps(settings, 0.3F, 4096U) == 1600U
        && blue::NoThreatEpisodeLimitSteps(settings, 0.2F, 1024U) == 1024U;
}

} // namespace

int main()
{
    if (!RallyAndGeneration()) { std::cerr << "rally/generation\n"; return 1; }
    for (int direction : {-1, 1})
        for (std::size_t count : {1U, 2U, 3U, 4U, 9U})
            if (!CompleteSequence(direction, count))
            {
                std::cerr << "sequence direction=" << direction
                          << " P=" << count << '\n';
                return 2;
            }
    for (int direction : {-1, 1})
        for (std::size_t count : {2U, 3U, 4U, 9U})
            if (!CompleteSequence(direction, count, true))
            {
                std::cerr << "wingman-independent waypoint direction="
                          << direction << " P=" << count << '\n';
                return 7;
            }
    if (!NoPrematureAdvanceOrShortcut()) { std::cerr << "shortcut\n"; return 3; }
    if (!OrbitCannotAdvanceSequence()) { std::cerr << "orbit\n"; return 4; }
    if (!WedgeAndCombatCompatibility()) { std::cerr << "wedge/combat\n"; return 5; }
    if (!TimeoutConversion()) { std::cerr << "timeout\n"; return 6; }
    return 0;
}
