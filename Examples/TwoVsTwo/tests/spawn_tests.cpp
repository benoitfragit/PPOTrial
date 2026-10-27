#include "spawn.hpp"
#include "spawn_config.hpp"
#include "defenderbackend.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <random>

namespace {
using ppo::examples::twovstwo::EpisodeSpawnLayout;
using ppo::examples::twovstwo::Spawn;
using ppo::examples::twovstwo::SpawnSettings;
using ppo::examples::twovstwo::WorldBounds;

constexpr float kHalfPi = 1.5707963267948966F;
constexpr float kTwoPi = 6.2831853071795865F;

bool Finite(Spawn const& spawn) noexcept
{
    return std::isfinite(spawn.north) && std::isfinite(spawn.east)
        && std::isfinite(spawn.heading) && std::isfinite(spawn.speed);
}

bool InWorld(Spawn const& spawn, WorldBounds const& world) noexcept
{
    return Finite(spawn)
        && std::fabs(spawn.north) < world.halfNorth
        && std::fabs(spawn.east) < world.halfEast;
}

float WorldScale(Spawn const& spawn, WorldBounds const& world) noexcept
{
    return std::max(std::fabs(spawn.north) / world.halfNorth,
                    std::fabs(spawn.east) / world.halfEast);
}

float Bearing(Spawn const& spawn, int side, WorldBounds const& world) noexcept
{
    float const north = spawn.north / world.halfNorth;
    float const oppositeObjectiveEast =
        -static_cast<float>(side) * spawn.east / world.halfEast;
    return std::atan2(north, oppositeObjectiveEast);
}

float AngleDifference(float first, float second) noexcept
{
    return std::remainder(first - second, kTwoPi);
}

bool Same(Spawn const& first, Spawn const& second) noexcept
{
    return first.north == second.north && first.east == second.east
        && first.heading == second.heading && first.speed == second.speed
        && first.destroyed == second.destroyed;
}

bool Same(EpisodeSpawnLayout const& first,
          EpisodeSpawnLayout const& second) noexcept
{
    return first.side == second.side && Same(first.red[0], second.red[0])
        && Same(first.red[1], second.red[1])
        && first.redActivationSteps == second.redActivationSteps
        && Same(first.restrictedZone, second.restrictedZone);
}
} // namespace

int main()
{
    static_assert(ppo::examples::twovstwo::defenders::kBackendAvailable);
    using ppo::examples::twovstwo::GenerateEpisodeSpawnLayout;
    using ppo::examples::twovstwo::HasPendingThreats;
    using ppo::examples::twovstwo::MakeSpawn;
    using ppo::examples::twovstwo::ParseSpawnSettings;
    using ppo::examples::twovstwo::Valid;

    SpawnSettings const settings{};
    if (!Valid(settings))
        return 1;

    for (WorldBounds const world : {WorldBounds{22329.375F, 35727.F},
                                    WorldBounds{15000.F, 22000.F},
                                    WorldBounds{30000.F, 45000.F}})
    {
        for (int side : {-1, 1})
        {
            bool sawFront = false;
            bool sawLateral = false;
            bool sawNorth = false;
            bool sawSouth = false;
            for (unsigned seed = 0U; seed < 256U; ++seed)
            {
                std::mt19937 rng{seed};
                auto const layout = GenerateEpisodeSpawnLayout(
                    side, settings, world, rng);
                std::mt19937 repeatRng{seed};
                auto const repeated = GenerateEpisodeSpawnLayout(
                    side, settings, world, repeatRng);
                if (!layout || !repeated || !Same(*layout, *repeated))
                    return 2;

                auto const objective = MakeSpawn(
                    "blue_objective", *layout, world, rng);
                auto const blue1 = MakeSpawn(
                    "blue_guard_1", *layout, world, rng);
                auto const blue2 = MakeSpawn(
                    "blue_guard_2", *layout, world, rng);
                auto const zone = MakeSpawn(
                    "blue_restricted_zone", *layout, world, rng);
                auto const red1 = MakeSpawn("red_1", *layout, world, rng);
                auto const red2 = MakeSpawn("red_2", *layout, world, rng);
                if (!objective || !blue1 || !blue2 || !zone || !red1 || !red2)
                    return 3;
                if (!InWorld(*objective, world) || !InWorld(*blue1, world)
                    || !InWorld(*blue2, world) || !InWorld(*zone, world))
                    return 4;
                if (objective->east * side <= blue1->east * side
                    || objective->east * side <= blue2->east * side
                    || blue1->north >= blue2->north)
                    return 5;

                float const separation = std::hypot(
                    red1->north - red2->north, red1->east - red2->east);
                if (separation + 1.e-3F < settings.red.minimumSeparation)
                    return 6;
                for (auto const& red : {*red1, *red2})
                {
                    float const scale = WorldScale(red, world);
                    float const bearing = Bearing(red, side, world);
                    if (!Finite(red)
                        || scale < settings.red.worldScale.minimum - 1.e-5F
                        || scale > settings.red.worldScale.maximum + 1.e-5F
                        || bearing < settings.red.bearingRadians.minimum - 1.e-5F
                        || bearing > settings.red.bearingRadians.maximum + 1.e-5F)
                        return 7;
                    float const expectedHeading = std::atan2(
                        objective->east - red.east,
                        objective->north - red.north);
                    if (std::fabs(AngleDifference(
                            red.heading, expectedHeading)) > 0.121F
                        || red.speed < 180.F || red.speed > 260.F)
                        return 8;
                    sawFront |= std::fabs(bearing) < 0.25F;
                    sawLateral |= std::fabs(bearing) > 1.2F;
                    sawNorth |= bearing > 0.5F;
                    sawSouth |= bearing < -0.5F;
                }
                if (red1->destroyed || !Same(*red1, layout->red[0])
                    || !Same(*red2, layout->red[1])
                    || !Same(*zone, layout->restrictedZone))
                    return 9;

                float const northFraction = zone->north / world.halfNorth;
                float const eastFraction = side * zone->east / world.halfEast;
                if (northFraction < settings.restrictedZone.northFraction.minimum
                    || northFraction > settings.restrictedZone.northFraction.maximum
                    || eastFraction < settings.restrictedZone.eastFraction.minimum
                    || eastFraction > settings.restrictedZone.eastFraction.maximum)
                    return 10;
            }
            if (!sawFront || !sawLateral || !sawNorth || !sawSouth)
                return 11;
        }
    }

    WorldBounds const world{22329.375F, 35727.F};
    std::mt19937 invalidRng{42U};
    if (GenerateEpisodeSpawnLayout(1, settings, WorldBounds{}, invalidRng))
        return 12;
    auto invalidSettings = settings;
    invalidSettings.red.worldScale.minimum = 1.F;
    if (Valid(invalidSettings))
        return 13;
    invalidSettings = settings;
    invalidSettings.red.minimumSeparation = 1.e9F;
    if (GenerateEpisodeSpawnLayout(1, invalidSettings, world, invalidRng))
        return 14;

    unsigned destroyedRed2 = 0U;
    unsigned disabledZones = 0U;
    for (unsigned seed = 0U; seed < 2000U; ++seed)
    {
        std::mt19937 probabilityRng{seed};
        auto const layout = GenerateEpisodeSpawnLayout(
            1, settings, world, probabilityRng);
        if (!layout)
            return 15;
        destroyedRed2 += layout->red[1].destroyed ? 1U : 0U;
        disabledZones += layout->restrictedZone.destroyed ? 1U : 0U;
    }
    if (destroyedRed2 < 240U || destroyedRed2 > 360U
        || disabledZones < 400U || disabledZones > 600U)
        return 16;

    auto noThreatSettings = settings;
    noThreatSettings.threatCurriculum.enabled = true;
    noThreatSettings.threatCurriculum.noThreatProbability = 1.F;
    noThreatSettings.threatCurriculum.singleThreatProbability = 0.F;
    std::mt19937 noThreatRng{1U};
    auto const noThreat = GenerateEpisodeSpawnLayout(
        1, noThreatSettings, world, noThreatRng);
    if (!noThreat || !noThreat->red[0].destroyed
        || !noThreat->red[1].destroyed
        || HasPendingThreats(*noThreat, 0U))
        return 19;

    auto delayedSingleSettings = settings;
    auto& delayedSingle = delayedSingleSettings.threatCurriculum;
    delayedSingle.enabled = true;
    delayedSingle.noThreatProbability = 0.F;
    delayedSingle.singleThreatProbability = 1.F;
    delayedSingle.delayedFirstThreatProbability = 1.F;
    delayedSingle.firstDelaySteps = {10U, 10U};
    std::mt19937 delayedSingleRng{2U};
    auto const single = GenerateEpisodeSpawnLayout(
        1, delayedSingleSettings, world, delayedSingleRng);
    if (!single
        || std::count(single->redActivationSteps.begin(),
                      single->redActivationSteps.end(), 10U) != 1
        || std::count(single->redActivationSteps.begin(),
                      single->redActivationSteps.end(),
                      EpisodeSpawnLayout::NeverActivate) != 1
        || !HasPendingThreats(*single, 0U)
        || HasPendingThreats(*single, 10U))
        return 20;

    auto staggeredSettings = settings;
    auto& staggered = staggeredSettings.threatCurriculum;
    staggered.enabled = true;
    staggered.noThreatProbability = 0.F;
    staggered.singleThreatProbability = 0.F;
    staggered.delayedFirstThreatProbability = 0.F;
    staggered.delayedSecondThreatProbability = 1.F;
    staggered.secondDelaySteps = {20U, 20U};
    std::mt19937 staggeredRng{3U};
    auto const pair = GenerateEpisodeSpawnLayout(
        1, staggeredSettings, world, staggeredRng);
    if (!pair
        || std::count(pair->redActivationSteps.begin(),
                      pair->redActivationSteps.end(), 0U) != 1
        || std::count(pair->redActivationSteps.begin(),
                      pair->redActivationSteps.end(), 20U) != 1
        || !HasPendingThreats(*pair, 0U)
        || HasPendingThreats(*pair, 20U))
        return 21;

    auto invalidCurriculum = settings;
    invalidCurriculum.threatCurriculum.noThreatProbability = 0.8F;
    invalidCurriculum.threatCurriculum.singleThreatProbability = 0.3F;
    if (Valid(invalidCurriculum))
        return 22;

    nlohmann::json const json = {
        {"red2InitiallyDestroyedProbability", 0.15},
        {"redSpawn", {
            {"minimumWorldScale", 1.05},
            {"maximumWorldScale", 1.35},
            {"minimumBearingDegrees", -90.0},
            {"maximumBearingDegrees", 90.0},
            {"minimumSeparation", 10000.0}}},
        {"threatCurriculum", {
            {"enabled", true},
            {"noThreatProbability", 0.10},
            {"singleThreatProbability", 0.30},
            {"delayedFirstThreatProbability", 0.20},
            {"delayedSecondThreatProbability", 0.45},
            {"minimumFirstDelaySteps", 256U},
            {"maximumFirstDelaySteps", 768U},
            {"minimumSecondDelaySteps", 384U},
            {"maximumSecondDelaySteps", 1024U}}},
        {"restrictedZoneSpawn", {
            {"enabledProbability", 0.75},
            {"minimumNorthFraction", -0.45},
            {"maximumNorthFraction", 0.45},
            {"minimumEastFraction", 0.15},
            {"maximumEastFraction", 0.65}}}
    };
    auto const parsed = ParseSpawnSettings(json);
    if (!parsed
        || std::fabs(parsed->red.bearingRadians.minimum + kHalfPi) > 1.e-6F
        || std::fabs(parsed->red.bearingRadians.maximum - kHalfPi) > 1.e-6F
        || !parsed->threatCurriculum.enabled
        || parsed->threatCurriculum.firstDelaySteps[0] != 256U
        || ParseSpawnSettings(nlohmann::json::object()))
        return 17;

    std::mt19937 unknownRng{7U};
    auto const layout = GenerateEpisodeSpawnLayout(1, settings, world, unknownRng);
    if (!layout || MakeSpawn("unknown", *layout, world, unknownRng))
        return 18;
    return 0;
}
