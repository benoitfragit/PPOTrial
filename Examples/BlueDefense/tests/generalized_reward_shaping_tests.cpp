#include "generalized_reward_shaping.hpp"
#include "generalized_cooperative_reward.hpp"

#include <cmath>
#include <cstddef>

namespace patrol = ppo::examples::bluedefense::defenders;

namespace {
bool Close(float const a, float const b, float const tolerance = 1.e-4F)
{
    return std::abs(a - b) <= tolerance;
}

patrol::GeneralizedRuntimeState MakeState(
    std::size_t const blues, std::size_t const reds)
{
    patrol::GeneralizedRuntimeState state{};
    state.counts = {blues, reds};
    for (std::size_t index = 0; index < blues; ++index)
        state.blues[index] = {0.F, 0.F, 0.F, 0.F, 0.F, true};
    for (std::size_t index = 0; index < reds; ++index)
        state.reds[index] = {90000.F, 0.F, 3.14159265F, 400.F, 0.F, true};
    state.objectiveNorth = 0.F;
    state.objectiveEast = 0.F;
    return state;
}

bool TestControlEffortPenalty()
{
    patrol::Parameters parameters{};
    parameters.controlEffortWeight = 0.002F;
    if (!Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{0.F, 0.F}), 0.F, 1.e-7F)
        || !Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{1.F, -1.F}), 0.002F, 1.e-7F)
        || !Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{0.5F, -0.5F}), 0.001F, 1.e-7F)
        || !Close(patrol::GeneralizedControlEffortPenalty(
            parameters, patrol::Action{2.F, -3.F}), 0.002F, 1.e-7F))
        return false;
    parameters.controlEffortWeight = 0.F;
    return Close(patrol::GeneralizedControlEffortPenalty(
        parameters, patrol::Action{1.F, 1.F}), 0.F, 1.e-7F);
}

bool TestTracking()
{
    patrol::Parameters parameters{};
    auto const roles = patrol::BlueTeamRoles::Create(1U, 0U);
    if (!roles)
        return false;
    auto before = MakeState(1U, 0U);
    auto const reference = patrol::EvaluatePatrolCurve(
        parameters.geometry, 3000.F);
    before.blues[0] = {reference.north, reference.east,
        reference.heading, parameters.cruiseSpeed, 0.F, true};
    auto onTrack = before;
    auto const advanced = patrol::EvaluatePatrolCurve(
        parameters.geometry, 3000.F + parameters.cruiseSpeed * parameters.dt);
    onTrack.blues[0].north = advanced.north;
    onTrack.blues[0].east = advanced.east;
    onTrack.blues[0].heading = advanced.heading;
    auto offTrack = onTrack;
    offTrack.blues[0].north += 5000.F;
    offTrack.blues[0].east += 5000.F;
    auto wrongHeading = onTrack;
    wrongHeading.blues[0].heading += 3.14159265F;
    float const good = patrol::GeneralizedLeaderTrackingReward(
        parameters, before, onTrack, *roles);
    return good > patrol::GeneralizedLeaderTrackingReward(
            parameters, before, offTrack, *roles)
        && good > patrol::GeneralizedLeaderTrackingReward(
            parameters, before, wrongHeading, *roles);
}

bool TestWedgeReference()
{
    patrol::Parameters parameters{};
    auto const roles = patrol::BlueTeamRoles::Create(2U, 0U);
    if (!roles)
        return false;
    auto state = MakeState(2U, 0U);
    state.blues[0] = {5000.F, -2000.F, 1.25F, 450.F, 0.F, true};
    auto const offset = patrol::EvaluateFormationOffset(
        0U, parameters.geometry.separationDistance,
        parameters.geometry.separationAngle);
    float const cosHeading = std::cos(state.blues[0].heading);
    float const sinHeading = std::sin(state.blues[0].heading);
    float const inside = static_cast<float>(parameters.geometry.direction)
        * offset.inwardMeters;
    state.blues[1].north = state.blues[0].north
        + offset.longitudinalMeters * cosHeading - inside * sinHeading;
    state.blues[1].east = state.blues[0].east
        + offset.longitudinalMeters * sinHeading + inside * cosHeading;
    std::array<patrol::GeneralizedObservation,
               patrol::TeamCapacity::MaximumBlueMobiles> observations{};
    if (!patrol::BuildGeneralizedObservations(
            parameters, state, *roles, observations))
        return false;
    std::size_t const offsetInObservation =
        patrol::GeneralizedObservationLayout::PatrolOffset;
    return Close(patrol::GeneralizedFormationQuality(
            parameters, state, *roles, 1U), 1.F)
        && Close(observations[1].values[offsetInObservation + 6U], 0.F)
        && Close(observations[1].values[offsetInObservation + 7U], 0.F);
}

bool TestPursuit()
{
    patrol::Parameters parameters{};
    auto before = MakeState(3U, 2U);
    before.reds[0].north = 75000.F;
    before.reds[1].north = 120000.F;
    auto approaching = before;
    approaching.blues[0].north += 80.F;
    approaching.blues[1].north += 80.F;
    approaching.blues[2].north += 80.F;
    auto withdrawing = before;
    withdrawing.blues[0].north -= 80.F;
    withdrawing.blues[1].north -= 80.F;
    withdrawing.blues[2].north -= 80.F;
    auto redOnly = before;
    redOnly.reds[0].north -= 80.F;
    redOnly.reds[1].north -= 80.F;
    return patrol::GeneralizedPursuitProgress(
            parameters, before, approaching) > 0.F
        && patrol::GeneralizedPursuitProgress(
            parameters, before, withdrawing) < 0.F
        && Close(patrol::GeneralizedPursuitProgress(
            parameters, before, redOnly), 0.F)
        && Close(patrol::GeneralizedPursuitProgress(
            parameters, MakeState(1U, 0U),
            MakeState(1U, 0U)), 0.F);
}

bool TestCooperation()
{
    patrol::Parameters parameters{};
    auto noRed = MakeState(2U, 0U);
    auto farRed = MakeState(2U, 1U);
    auto closeRed = farRed;
    closeRed.reds[0].north = 20000.F;
    return Close(patrol::GeneralizedCooperationWeight(
            parameters, noRed), 1.F)
        && Close(patrol::GeneralizedCooperationWeight(
            parameters, farRed), 0.F)
        && patrol::GeneralizedCooperationWeight(
            parameters, closeRed) > 0.F;
}
} // namespace

int main()
{
    if (!TestTracking()) return 1;
    if (!TestWedgeReference()) return 2;
    if (!TestPursuit()) return 3;
    if (!TestCooperation()) return 4;
    if (!TestControlEffortPenalty()) return 5;
    return 0;
}
