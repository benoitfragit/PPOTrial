#include "spawn.hpp"
#include "spawn_config.hpp"
#include "defenderbackend.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>

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
    if (first.side != second.side || first.red.size() != second.red.size()
        || first.redActivationSteps != second.redActivationSteps
        || !Same(first.friendlyRestrictedZone,
                 second.friendlyRestrictedZone)
        || first.restrictedZones.size() != second.restrictedZones.size())
        return false;
    for (std::size_t i = 0U; i < first.red.size(); ++i)
        if (!Same(first.red[i], second.red[i]))
            return false;
    for (std::size_t i = 0U; i < first.restrictedZones.size(); ++i)
        if (first.restrictedZones[i].id != second.restrictedZones[i].id
            || !Same(first.restrictedZones[i].spawn,
                     second.restrictedZones[i].spawn))
            return false;
    return true;
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

    SpawnSettings settings{};
    settings.restrictedZones.items = {
        {"zone_fixed", "restrictedzone", true, 1.F,
         ppo::examples::twovstwo::RestrictedZonePositionMode::Fixed,
         {1250.F, -2500.F}, {}, {}},
        {"zone_random", "restrictedzone", true, 0.75F,
         ppo::examples::twovstwo::RestrictedZonePositionMode::Random,
         {}, {-5000.F, 5000.F}, {-7000.F, 7000.F}}};
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
                auto const friendlyZone = MakeSpawn(
                    "blue_restricted_zone", *layout, world, rng);
                auto const fixedZone = MakeSpawn(
                    "zone_fixed", *layout, world, rng);
                auto const randomZone = MakeSpawn(
                    "zone_random", *layout, world, rng);
                auto const red1 = MakeSpawn("red_1", *layout, world, rng);
                auto const red2 = MakeSpawn("red_2", *layout, world, rng);
                if (!objective || !blue1 || !blue2 || !friendlyZone
                    || !fixedZone
                    || !randomZone || !red1 || !red2)
                    return 3;
                if (!InWorld(*objective, world) || !InWorld(*blue1, world)
                    || !InWorld(*blue2, world) || !InWorld(*fixedZone, world)
                    || !InWorld(*randomZone, world)
                    || !InWorld(*friendlyZone, world))
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
                    || !Same(*friendlyZone,
                             layout->friendlyRestrictedZone)
                    || !Same(*fixedZone, layout->restrictedZones[0].spawn)
                    || !Same(*randomZone, layout->restrictedZones[1].spawn))
                    return 9;

                if (fixedZone->north != 1250.F || fixedZone->east != -2500.F
                    || randomZone->north < -5000.F
                    || randomZone->north > 5000.F
                    || randomZone->east < -7000.F
                    || randomZone->east > 7000.F)
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
        disabledZones += layout->restrictedZones[1].spawn.destroyed ? 1U : 0U;
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
        || !HasPendingThreats(*single, 10U)
        || HasPendingThreats(*single, 11U))
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
        || !HasPendingThreats(*pair, 20U)
        || HasPendingThreats(*pair, 21U))
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
        {"restrictedZoneSpawn", {
            {"enabledProbability", 0.75},
            {"minimumNorthFraction", -0.45},
            {"maximumNorthFraction", 0.45},
            {"minimumEastFraction", 0.15},
            {"maximumEastFraction", 0.65}}},
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
        {"restrictedZones", {
            {"enabled", true},
            {"items", nlohmann::json::array({
                {
                    {"id", "fixed_zone"},
                    {"type", "RestrictedZone"},
                    {"platform", "restrictedzone"},
                    {"side", "Hostile"},
                    {"enabled", true},
                    {"enabledProbability", 1.0},
                    {"fixedPosition", {{"north", 10.0}, {"east", 20.0}}}
                },
                {
                    {"id", "random_zone"},
                    {"type", "RestrictedZone"},
                    {"platform", "restrictedzone"},
                    {"side", "Hostile"},
                    {"enabled", false},
                    {"enabledProbability", 0.0},
                    {"oppositeObjective", true},
                    {"randomPosition", {
                        {"north", {{"minimum", -4.0}, {"maximum", 5.0}}},
                        {"east", {{"minimum", 6.0}, {"maximum", 7.0}}}}}
                }
            })}
        }}
    };
    std::string diagnostic;
    auto const parsed = ParseSpawnSettings(json, &diagnostic);
    if (!parsed
        || std::fabs(parsed->red.bearingRadians.minimum + kHalfPi) > 1.e-6F
        || std::fabs(parsed->red.bearingRadians.maximum - kHalfPi) > 1.e-6F
        || !parsed->threatCurriculum.enabled
        || parsed->threatCurriculum.firstDelaySteps[0] != 256U
        || parsed->restrictedZones.items.size() != 2U
        || !parsed->restrictedZones.items[1].oppositeObjective
        || ParseSpawnSettings(nlohmann::json::object()))
        return 17;
    std::mt19937 parsedRng{11U};
    auto const parsedLayout = GenerateEpisodeSpawnLayout(
        1, *parsed, world, parsedRng);
    if (!parsedLayout || parsedLayout->restrictedZones.size() != 2U
        || parsedLayout->restrictedZones[0].spawn.north != 10.F
        || parsedLayout->restrictedZones[0].spawn.east != 20.F
        || parsedLayout->restrictedZones[0].spawn.destroyed
        || !parsedLayout->restrictedZones[1].spawn.destroyed
        || parsedLayout->restrictedZones[1].spawn.east >= 0.F)
        return 25;
    std::mt19937 mirroredZoneRng{11U};
    auto const mirroredZoneLayout = GenerateEpisodeSpawnLayout(
        -1, *parsed, world, mirroredZoneRng);
    if (!mirroredZoneLayout
        || mirroredZoneLayout->restrictedZones[1].spawn.east <= 0.F)
        return 42;
    auto globallyDisabled = *parsed;
    globallyDisabled.restrictedZones.enabled = false;
    std::mt19937 disabledRng{11U};
    auto const disabledLayout = GenerateEpisodeSpawnLayout(
        1, globallyDisabled, world, disabledRng);
    if (!disabledLayout || std::any_of(
            disabledLayout->restrictedZones.begin(),
            disabledLayout->restrictedZones.end(),
            [](auto const& zone) { return !zone.spawn.destroyed; }))
        return 26;

    auto boundedZoneSettings = settings;
    using ZoneMode = ppo::examples::twovstwo::RestrictedZonePositionMode;
    boundedZoneSettings.restrictedZones.items = {
        {"inside_north", "restrictedzone", true, 1.F, ZoneMode::Random,
         {}, {4000.F, 16000.F}, {8000.F, 22000.F}, true},
        {"inside_south", "restrictedzone", true, 1.F, ZoneMode::Random,
         {}, {-16000.F, -4000.F}, {8000.F, 22000.F}, true},
        {"outside_north", "restrictedzone", true, 1.F, ZoneMode::Random,
         {}, {3000.F, 15000.F}, {38000.F, 45000.F}, true},
        {"outside_south", "restrictedzone", true, 1.F, ZoneMode::Random,
         {}, {-15000.F, -3000.F}, {38000.F, 45000.F}, true}};
    for (int side : {-1, 1})
        for (unsigned seed = 0U; seed < 64U; ++seed)
        {
            std::mt19937 boundedZoneRng{seed};
            auto const boundedZones = GenerateEpisodeSpawnLayout(
                side, boundedZoneSettings, world, boundedZoneRng);
            if (!boundedZones || boundedZones->restrictedZones.size() != 4U)
                return 43;
            std::size_t insideCount = 0U;
            std::size_t outsideCount = 0U;
            float const objectiveEast = static_cast<float>(side)
                * 0.92F * world.halfEast;
            for (auto const& zone : boundedZones->restrictedZones)
            {
                Spawn const& spawn = zone.spawn;
                bool const inside = std::abs(spawn.north) < world.halfNorth
                    && std::abs(spawn.east) < world.halfEast;
                insideCount += inside ? 1U : 0U;
                outsideCount += inside ? 0U : 1U;
                if (spawn.east * static_cast<float>(side) >= 0.F
                    || std::hypot(spawn.north,
                                  spawn.east - objectiveEast) <= 7000.F)
                    return 44;
            }
            if (insideCount != 2U || outsideCount != 2U)
                return 45;
        }

    auto invalidZoneJson = json;
    invalidZoneJson["restrictedZones"]["items"][0]["randomPosition"] = {
        {"north", {{"minimum", 0.0}, {"maximum", 1.0}}},
        {"east", {{"minimum", 0.0}, {"maximum", 1.0}}}};
    if (ParseSpawnSettings(invalidZoneJson, &diagnostic)
        || diagnostic.find("exactly one") == std::string::npos)
        return 23;

    SpawnSettings noZones{};
    std::mt19937 noZoneRng{8U};
    auto const noZoneLayout = GenerateEpisodeSpawnLayout(
        1, noZones, world, noZoneRng);
    if (!noZoneLayout || !noZoneLayout->restrictedZones.empty()
        || MakeSpawn("zone_fixed", *noZoneLayout, world, noZoneRng))
        return 24;

    auto noRedSettings = settings;
    noRedSettings.redTeam =
        ppo::examples::twovstwo::RedTeamSpawnSettings{};
    noRedSettings.redTeam->activeCount = 0U;
    std::mt19937 noRedRng{31U};
    auto const noRed = GenerateEpisodeSpawnLayout(
        1, noRedSettings, world, noRedRng, &diagnostic);
    if (!noRed || !diagnostic.empty() || !noRed->red.empty()
        || !noRed->redActivationSteps.empty()
        || MakeSpawn("red_1", *noRed, world, noRedRng))
        return 27;

    auto maximumRedSettings = settings;
    maximumRedSettings.red.minimumSeparation = 1000.F;
    maximumRedSettings.redTeam =
        ppo::examples::twovstwo::RedTeamSpawnSettings{};
    auto& maximumTeam = *maximumRedSettings.redTeam;
    maximumTeam.activeCount =
        ppo::examples::twovstwo::defenders::TeamCapacity::MaximumRedMobiles;
    maximumTeam.maximumPlacementAttempts = 4096U;
    maximumTeam.delayedActivationProbability = 1.F;
    maximumTeam.activationDelaySteps = {17U, 17U};
    std::mt19937 maximumRng{32U};
    auto const maximum = GenerateEpisodeSpawnLayout(
        -1, maximumRedSettings, world, maximumRng, &diagnostic);
    std::mt19937 maximumRepeatRng{32U};
    auto const maximumRepeated = GenerateEpisodeSpawnLayout(
        -1, maximumRedSettings, world, maximumRepeatRng);
    if (!maximum || !maximumRepeated || !diagnostic.empty()
        || !Same(*maximum, *maximumRepeated)
        || maximum->red.size() != maximumTeam.activeCount
        || maximum->redActivationSteps.size() != maximumTeam.activeCount)
        return 28;
    for (std::size_t red = 0U; red < maximum->red.size(); ++red)
    {
        if (!maximum->red[red].destroyed
            || maximum->redActivationSteps[red] != 17U)
            return 29;
        for (std::size_t other = red + 1U;
             other < maximum->red.size(); ++other)
            if (std::hypot(
                    maximum->red[red].north - maximum->red[other].north,
                    maximum->red[red].east - maximum->red[other].east)
                + 1.e-3F < maximumRedSettings.red.minimumSeparation)
                return 30;
    }
    if (!MakeSpawn("red_10", *maximum, world, maximumRng)
        || MakeSpawn("red_11", *maximum, world, maximumRng)
        || MakeSpawn("red_01", *maximum, world, maximumRng))
        return 31;

    auto immediateSettings = maximumRedSettings;
    immediateSettings.redTeam->delayedActivationProbability = 0.F;
    std::mt19937 immediateRng{33U};
    auto const immediate = GenerateEpisodeSpawnLayout(
        1, immediateSettings, world, immediateRng);
    if (!immediate
        || std::any_of(immediate->red.begin(), immediate->red.end(),
                       [](Spawn const& red) { return red.destroyed; })
        || std::any_of(immediate->redActivationSteps.begin(),
                       immediate->redActivationSteps.end(),
                       [](std::size_t step) { return step != 0U; }))
        return 32;

    auto impossibleGeneralized = maximumRedSettings;
    impossibleGeneralized.red.minimumSeparation = 1.e9F;
    impossibleGeneralized.redTeam->maximumPlacementAttempts = 3U;
    std::mt19937 impossibleRng{34U};
    if (GenerateEpisodeSpawnLayout(
            1, impossibleGeneralized, world, impossibleRng, &diagnostic)
        || diagnostic.find("red_2") == std::string::npos
        || diagnostic.find("3 attempts") == std::string::npos)
        return 33;

    auto generalizedJson = json;
    generalizedJson.erase("red2InitiallyDestroyedProbability");
    generalizedJson.erase("threatCurriculum");
    generalizedJson["redTeam"] = {
        {"activeCount", 10U},
        {"maximumPlacementAttempts", 512U},
        {"delayedActivationProbability", 0.25},
        {"minimumActivationDelaySteps", 12U},
        {"maximumActivationDelaySteps", 24U}};
    auto const parsedGeneralized = ParseSpawnSettings(
        generalizedJson, &diagnostic);
    if (!parsedGeneralized || !parsedGeneralized->redTeam
        || parsedGeneralized->redTeam->activeCount != 10U
        || parsedGeneralized->red2DestroyedProbability != 0.F
        || !diagnostic.empty())
        return 34;
    generalizedJson["redTeam"]["activeCount"] = 11U;
    if (ParseSpawnSettings(generalizedJson, &diagnostic)
        || diagnostic.find("MaximumRedMobiles") == std::string::npos)
        return 35;

    generalizedJson["redTeam"]["activeCount"] = 2U;
    generalizedJson["threatCurriculum"] = json["threatCurriculum"];
    if (ParseSpawnSettings(generalizedJson, &diagnostic)
        || diagnostic.find("mutually exclusive") == std::string::npos)
        return 36;

    generalizedJson.erase("threatCurriculum");
    generalizedJson["redTeam"] = "invalid";
    if (ParseSpawnSettings(generalizedJson, &diagnostic)
        || diagnostic.find("must be an object") == std::string::npos)
        return 37;

    auto blueTeamJson = json;
    blueTeamJson["blueTeam"] = {{"activeCount", 1U}};
    auto parsedBlueTeam = ParseSpawnSettings(blueTeamJson, &diagnostic);
    if (!parsedBlueTeam || !parsedBlueTeam->blueTeam
        || parsedBlueTeam->blueTeam->activeCount != 1U
        || !diagnostic.empty())
        return 38;
    blueTeamJson["blueTeam"]["activeCount"] = 9U;
    parsedBlueTeam = ParseSpawnSettings(blueTeamJson, &diagnostic);
    if (!parsedBlueTeam || !parsedBlueTeam->blueTeam
        || parsedBlueTeam->blueTeam->activeCount != 9U)
        return 39;
    blueTeamJson["blueTeam"]["activeCount"] = 0U;
    if (ParseSpawnSettings(blueTeamJson, &diagnostic)
        || diagnostic.find("[1, 9]") == std::string::npos)
        return 40;
    blueTeamJson["blueTeam"] = {
        {"activeCount", 2U}, {"unexpected", true}};
    if (ParseSpawnSettings(blueTeamJson, &diagnostic)
        || diagnostic.find("unknown field") == std::string::npos)
        return 41;

    std::mt19937 unknownRng{7U};
    auto const layout = GenerateEpisodeSpawnLayout(1, settings, world, unknownRng);
    if (!layout || MakeSpawn("unknown", *layout, world, unknownRng))
        return 18;
    return 0;
}
