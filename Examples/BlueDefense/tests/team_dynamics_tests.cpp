#include "generalized_team_dynamics.hpp"

#include <cmath>
#include <cstddef>

namespace blue = ppo::examples::bluedefense::defenders;

namespace {

constexpr float kPi = 3.14159265358979323846F;

blue::Parameters TestParameters() noexcept
{
    blue::Parameters parameters{};
    parameters.dt = 1.F;
    parameters.maxSpeed = 100.F;
    parameters.maxAcceleration = 100.F;
    parameters.maxRotationRate = 1.F;
    parameters.collisionDistance = 0.1F;
    parameters.halfNorth = 1000.F;
    parameters.halfEast = 1000.F;
    parameters.hardBoundaryScale = 2.F;
    return parameters;
}

blue::Mobile MobileAt(float const north, float const east,
                      float const heading, float const speed) noexcept
{
    return {north, east, heading, speed, 0.F, true};
}

bool Near(float const first, float const second,
          float const tolerance = 1.0e-4F) noexcept
{
    return std::abs(first - second) <= tolerance;
}

bool SingletonUsesExactlyOneActionAndTransition() noexcept
{
    blue::GeneralizedBlueTeamState state{};
    state.active[4U] = true;
    state.activeCount = 1U;
    state.step = 12U;
    state.mobiles[4U] = MobileAt(0.F, 0.F, 0.F, 10.F);

    blue::BlueTeamActionBatch commands{};
    commands.actionCount = 1U;
    commands.actions[0U] = {0.F, 0.5F};

    blue::GeneralizedBlueTeamStep result{};
    if (blue::AdvanceGeneralizedBlueTeam(
            TestParameters(), state, commands, result)
            != blue::GeneralizedTeamStepStatus::Applied
        || result.transitionCount != 1U
        || result.transitions[0U].physicalIndex != 4U
        || result.next.activeCount != 1U || !result.next.active[4U]
        || result.next.step != 13U || result.next.collided
        || result.collisions.collisionCount != 0U)
    {
        return false;
    }

    // acceleration=0.5 * 100 m/s2 for one second: 10 -> 60 m/s.
    if (!Near(result.next.mobiles[4U].north, 60.F)
        || !Near(result.next.mobiles[4U].east, 0.F)
        || !Near(result.transitions[0U].action[1U], 0.5F))
    {
        return false;
    }

    commands.actionCount = 0U;
    result.transitionCount = 9U;
    if (blue::AdvanceGeneralizedBlueTeam(
            TestParameters(), state, commands, result)
            != blue::GeneralizedTeamStepStatus::InvalidActionCount
        || result.transitionCount != 0U)
    {
        return false;
    }

    commands.actionCount = 2U;
    return blue::AdvanceGeneralizedBlueTeam(
               TestParameters(), state, commands, result)
        == blue::GeneralizedTeamStepStatus::InvalidActionCount;
}

bool CrossingPairIsDetectedBetweenEndpoints() noexcept
{
    blue::GeneralizedBlueTeamState state{};
    state.active[1U] = true;
    state.active[8U] = true;
    state.activeCount = 2U;
    state.mobiles[1U] = MobileAt(-10.F, 0.F, 0.F, 20.F);
    state.mobiles[8U] = MobileAt(10.F, 0.F, kPi, 20.F);

    blue::BlueTeamActionBatch commands{};
    commands.actionCount = 2U;
    blue::GeneralizedBlueTeamStep result{};
    return blue::AdvanceGeneralizedBlueTeam(
               TestParameters(), state, commands, result)
            == blue::GeneralizedTeamStepStatus::Applied
        && result.transitionCount == 2U
        && result.transitions[0U].physicalIndex == 1U
        && result.transitions[1U].physicalIndex == 8U
        && result.next.collided && result.next.failed
        && result.collisions.collisionCount == 1U
        && result.collisions.pairs[0U].first == 1U
        && result.collisions.pairs[0U].second == 8U
        && result.collisions.pairs[0U].minimumSeparation <= 0.1F
        && std::abs(result.next.mobiles[1U].north
                    - result.next.mobiles[8U].north) > 10.F;
}

bool ParallelPairDoesNotCollide() noexcept
{
    blue::GeneralizedBlueTeamState state{};
    state.active[0U] = true;
    state.active[1U] = true;
    state.activeCount = 2U;
    state.mobiles[0U] = MobileAt(-10.F, 0.F, 0.F, 20.F);
    state.mobiles[1U] = MobileAt(-10.F, 10.F, 0.F, 20.F);

    blue::BlueTeamActionBatch commands{};
    commands.actionCount = 2U;
    blue::GeneralizedBlueTeamStep result{};
    return blue::AdvanceGeneralizedBlueTeam(
               TestParameters(), state, commands, result)
            == blue::GeneralizedTeamStepStatus::Applied
        && result.transitionCount == 2U && !result.next.collided
        && !result.next.failed && result.collisions.collisionCount == 0U;
}

bool NineWayConvergenceChecksEveryPair() noexcept
{
    blue::GeneralizedBlueTeamState state{};
    state.active.fill(true);
    state.activeCount = blue::TeamCapacity::MaximumBlueMobiles;

    constexpr float radius = 10.F;
    for (std::size_t index = 0U;
         index < blue::TeamCapacity::MaximumBlueMobiles; ++index)
    {
        float const angle = 2.F * kPi * static_cast<float>(index)
            / static_cast<float>(blue::TeamCapacity::MaximumBlueMobiles);
        float const north = radius * std::cos(angle);
        float const east = radius * std::sin(angle);
        float const heading = std::atan2(-east, -north);
        state.mobiles[index] = MobileAt(north, east, heading, radius);
    }

    blue::BlueTeamActionBatch commands{};
    commands.actionCount = blue::TeamCapacity::MaximumBlueMobiles;
    blue::GeneralizedBlueTeamStep result{};
    if (blue::AdvanceGeneralizedBlueTeam(
            TestParameters(), state, commands, result)
            != blue::GeneralizedTeamStepStatus::Applied
        || result.transitionCount != blue::TeamCapacity::MaximumBlueMobiles
        || result.collisions.collisionCount
            != blue::kMaximumBlueCollisionPairs
        || !result.next.collided || !result.next.failed)
    {
        return false;
    }

    std::size_t observedPair = 0U;
    for (std::size_t first = 0U;
         first < blue::TeamCapacity::MaximumBlueMobiles; ++first)
    {
        if (result.transitions[first].physicalIndex != first)
            return false;
        for (std::size_t second = first + 1U;
             second < blue::TeamCapacity::MaximumBlueMobiles; ++second)
        {
            auto const& collision = result.collisions.pairs[observedPair++];
            if (collision.first != first || collision.second != second
                || collision.minimumSeparation > 0.1F)
            {
                return false;
            }
        }
    }
    return observedPair == blue::kMaximumBlueCollisionPairs;
}

bool DestroyedParticipantKeepsStableNoOpTransition() noexcept
{
    blue::GeneralizedBlueTeamState state{};
    state.active[3U] = true;
    state.activeCount = 1U;
    state.step = 7U;
    state.mobiles[3U] = MobileAt(3000.F, 3000.F, 0.F, 20.F);
    state.mobiles[3U].available = false;

    blue::BlueTeamActionBatch commands{};
    commands.actionCount = 1U;
    commands.actions[0U] = {1.F, 1.F};

    blue::GeneralizedBlueTeamStep result{};
    return blue::Valid(state)
        && blue::AdvanceGeneralizedBlueTeam(
               TestParameters(), state, commands, result)
            == blue::GeneralizedTeamStepStatus::Applied
        && result.transitionCount == 1U
        && result.transitions[0U].physicalIndex == 3U
        && !result.transitions[0U].before.available
        && !result.transitions[0U].after.available
        && Near(result.transitions[0U].after.north, 3000.F)
        && Near(result.transitions[0U].after.east, 3000.F)
        && Near(result.transitions[0U].after.heading, 0.F)
        && Near(result.transitions[0U].after.speed, 20.F)
        && result.next.step == 8U && !result.next.collided
        && !result.next.failed && result.collisions.collisionCount == 0U;
}

bool DestroyedParticipantIsExcludedFromSweptCollisions() noexcept
{
    blue::GeneralizedBlueTeamState state{};
    state.active[1U] = true;
    state.active[8U] = true;
    state.activeCount = 2U;
    state.mobiles[1U] = MobileAt(-10.F, 0.F, 0.F, 20.F);
    state.mobiles[1U].available = false;
    state.mobiles[8U] = MobileAt(10.F, 0.F, kPi, 20.F);

    blue::BlueTeamActionBatch commands{};
    commands.actionCount = 2U;
    commands.actions[0U] = {1.F, 1.F};
    blue::GeneralizedBlueTeamStep result{};
    return blue::AdvanceGeneralizedBlueTeam(
               TestParameters(), state, commands, result)
            == blue::GeneralizedTeamStepStatus::Applied
        && result.transitionCount == 2U
        && result.transitions[0U].physicalIndex == 1U
        && result.transitions[1U].physicalIndex == 8U
        && Near(result.next.mobiles[1U].north, -10.F)
        && Near(result.next.mobiles[8U].north, -10.F)
        && !result.next.collided && !result.next.failed
        && result.collisions.collisionCount == 0U;
}

} // namespace

int main()
{
    if (!SingletonUsesExactlyOneActionAndTransition())
        return 1;
    if (!CrossingPairIsDetectedBetweenEndpoints())
        return 2;
    if (!ParallelPairDoesNotCollide())
        return 3;
    if (!NineWayConvergenceChecksEveryPair())
        return 4;
    if (!DestroyedParticipantKeepsStableNoOpTransition())
        return 5;
    if (!DestroyedParticipantIsExcludedFromSweptCollisions())
        return 6;
    return 0;
}
