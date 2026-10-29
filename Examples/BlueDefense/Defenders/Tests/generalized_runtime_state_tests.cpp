#include "generalized_runtime_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace blue = ppo::examples::bluedefense::defenders;

namespace {

blue::GeneralizedRuntimeState StateFor(std::size_t const blueCount,
                                       std::size_t const redCount) noexcept
{
    blue::GeneralizedRuntimeState state{};
    state.counts = {blueCount, redCount};
    state.objectiveEast = 25000.F;
    for (std::size_t index = 0U; index < blueCount; ++index)
    {
        state.blues[index] = {
            1000.F * static_cast<float>(index),
            2000.F + 500.F * static_cast<float>(index),
            0.1F * static_cast<float>(index), 300.F, 0.F, true};
        state.lidarAvailable[index] = true;
        state.lidar[index].fill(0.25F + 0.01F * static_cast<float>(index));
    }
    for (std::size_t index = 0U; index < redCount; ++index)
    {
        state.reds[index] = {
            -10000.F + 750.F * static_cast<float>(index),
            -15000.F + 1000.F * static_cast<float>(index),
            0.2F, 220.F, 0.F, true};
    }
    return state;
}

std::size_t MaskCount(blue::GeneralizedRawObservation const& observation,
                      std::size_t const offset, std::size_t const tracks,
                      std::size_t const features) noexcept
{
    std::size_t result = 0U;
    for (std::size_t track = 0U; track < tracks; ++track)
    {
        result += observation[blue::GeneralizedTrackMaskIndex(
            offset, track, features)] > 0.5F ? 1U : 0U;
    }
    return result;
}

bool CheckCase(std::size_t const blueCount,
               std::size_t const redCount,
               std::size_t const leader) noexcept
{
    blue::Parameters parameters{};
    auto const roles = blue::BlueTeamRoles::Create(blueCount, leader);
    if (!roles)
        return false;
    auto const state = StateFor(blueCount, redCount);
    std::array<blue::GeneralizedObservation,
               blue::TeamCapacity::MaximumBlueMobiles> observations{};
    if (!blue::BuildGeneralizedObservations(
            parameters, state, *roles, observations))
    {
        return false;
    }
    for (std::size_t index = 0U; index < blueCount; ++index)
    {
        auto const& observation = observations[index];
        if (!blue::Valid(observation)
            || observation.metadata.activeBlueMobiles != blueCount
            || observation.metadata.activeRedMobiles != redCount
            || MaskCount(observation.values,
                   blue::GeneralizedObservationLayout::RedOffset,
                   blue::GeneralizedObservationLayout::RedTracks,
                   blue::GeneralizedObservationLayout::RedFeatures)
                != redCount
            || MaskCount(observation.values,
                   blue::GeneralizedObservationLayout::BlueOffset,
                   blue::GeneralizedObservationLayout::BlueTracks,
                   blue::GeneralizedObservationLayout::FriendlyFeatures)
                != blueCount - 1U
            || observation.values[
                   blue::GeneralizedObservationLayout::LidarOffset]
                != 0.25F + 0.01F * static_cast<float>(index))
        {
            return false;
        }
        auto const assignment = roles->Assignment(index);
        float const role = observation.values[
            blue::GeneralizedObservationLayout::RoleOffset];
        if (!assignment
            || (assignment->role == blue::BlueRole::Leader
                ? role != -1.F : role != 1.F))
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!CheckCase(1U, 0U, 0U))
        return 1;
    if (!CheckCase(2U, 2U, 1U))
        return 2;
    if (!CheckCase(9U, 10U, 4U))
        return 3;

    auto state = StateFor(9U, 10U);
    state.blues[8U].available = false;
    state.reds[9U].available = false;
    auto const roles = blue::BlueTeamRoles::Create(9U, 0U);
    std::array<blue::GeneralizedObservation,
               blue::TeamCapacity::MaximumBlueMobiles> observations{};
    if (!roles || !blue::BuildGeneralizedObservations(
            blue::Parameters{}, state, *roles, observations)
        || MaskCount(observations[0U].values,
               blue::GeneralizedObservationLayout::BlueOffset,
               blue::GeneralizedObservationLayout::BlueTracks,
               blue::GeneralizedObservationLayout::FriendlyFeatures) != 7U
        || MaskCount(observations[0U].values,
               blue::GeneralizedObservationLayout::RedOffset,
               blue::GeneralizedObservationLayout::RedTracks,
               blue::GeneralizedObservationLayout::RedFeatures) != 9U)
    {
        return 4;
    }

    blue::Parameters equivalentParameters{};
    equivalentParameters.geometry.direction = -1;
    auto equivalent = StateFor(2U, 2U);
    blue::Reference const leader = blue::PatrolReference(
        equivalentParameters, 1, 1234.F);
    equivalent.blues[0U] = blue::OnReference(equivalentParameters, leader);
    equivalent.blues[1U] = blue::OnReference(
        equivalentParameters,
        blue::WingmanReference(equivalentParameters, leader));
    auto const equivalentRoles = blue::BlueTeamRoles::Create(2U, 0U);
    if (!equivalentRoles || !blue::BuildGeneralizedObservations(
            equivalentParameters, equivalent, *equivalentRoles, observations)
        || std::abs(observations[1U].values[
               blue::GeneralizedObservationLayout::PatrolOffset + 6U])
               > 1.0e-5F
        || std::abs(observations[1U].values[
               blue::GeneralizedObservationLayout::PatrolOffset + 7U])
               > 1.0e-5F)
    {
        return 5;
    }
    // The dense defensive reward must prioritize a red with imminent risk
    // over one that cannot reach the objective in the threat horizon.
    auto riskState = StateFor(3U, 2U);
    riskState.objectiveNorth = 0.F;
    riskState.objectiveEast = 0.F;
    riskState.reds[0U] = {15000.F, 0.F, blue::kPi, 400.F, 0.F, true};
    riskState.reds[1U] = {120000.F, 0.F, blue::kPi, 400.F, 0.F, true};
    float const beforeRisk = blue::ObjectiveThreatLevel(
        riskState, blue::Parameters{});
    auto farNeutralized = riskState;
    farNeutralized.reds[1U].available = false;
    auto closeNeutralized = riskState;
    closeNeutralized.reds[0U].available = false;
    float const farBenefit = beforeRisk - blue::ObjectiveThreatLevel(
        farNeutralized, blue::Parameters{});
    float const closeBenefit = beforeRisk - blue::ObjectiveThreatLevel(
        closeNeutralized, blue::Parameters{});
    if (!(closeBenefit > farBenefit)
        || closeBenefit <= 0.F || farBenefit < 0.F)
        return 6;
    std::swap(riskState.reds[0U], riskState.reds[1U]);
    if (std::abs(blue::ObjectiveThreatLevel(
            riskState, blue::Parameters{}) - beforeRisk) > 1.0e-6F)
        return 7;
    auto noRed = StateFor(1U, 0U);
    if (blue::ObjectiveThreatLevel(noRed, blue::Parameters{}) != 0.F)
        return 8;

    // The flight model marks an aircraft destroyed after speed drops
    // below its minimum; its last finite speed can be negative.
    // Dead tracks must not abort the MAPPO feedback for surviving blues.
    auto destroyed = StateFor(2U, 1U);
    destroyed.blues[1U].available = false;
    destroyed.blues[1U].speed = -12.F;
    destroyed.reds[0U].available = false;
    destroyed.reds[0U].speed = -3.F;
    auto const destroyedRoles = blue::BlueTeamRoles::Create(2U, 0U);
    if (!destroyedRoles || !blue::Valid(destroyed)
        || !blue::BuildGeneralizedObservations(
            blue::Parameters{}, destroyed, *destroyedRoles, observations))
        return 9;

    auto invalidLiveBlue = destroyed;
    invalidLiveBlue.blues[1U].available = true;
    if (blue::Valid(invalidLiveBlue))
        return 10;

    auto invalidLiveRed = destroyed;
    invalidLiveRed.reds[0U].available = true;
    if (blue::Valid(invalidLiveRed))
        return 11;

    auto invalidDead = destroyed;
    invalidDead.reds[0U].speed =
        std::numeric_limits<float>::quiet_NaN();
    if (blue::Valid(invalidDead))
        return 12;
    return 0;
}
