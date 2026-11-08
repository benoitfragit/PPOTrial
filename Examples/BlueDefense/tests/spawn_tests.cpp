/**
 * @file Examples/BlueDefense/tests/spawn_tests.cpp
 * @brief Tests et assertions déterministes pour spawn tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "spawn_config.hpp"
#include "team_capacity.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstddef>
#include <random>
#include <set>
#include <utility>
#include <string>
#include <vector>

namespace defense = ppo::examples::bluedefense;

namespace {

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
nlohmann::json ValidScenario()
{
    return {
        {"redSpawn", {
            {"minimumWorldScale", 1.15F},
            {"maximumWorldScale", 1.45F},
            {"minimumBearingDegrees", -90.F},
            {"maximumBearingDegrees", 90.F},
            {"minimumSeparation", 2000.F}
        }},
        {"restrictedZoneSpawn", {
            {"enabledProbability", 0.75F},
            {"minimumNorthFraction", -0.45F},
            {"maximumNorthFraction", 0.45F},
            {"minimumEastFraction", 0.15F},
            {"maximumEastFraction", 0.65F}
        }},
        {"blueTeam", {{"activeCount", 4U}}},
        {"redTeam", {
            {"activeCount", 4U},
            {"maximumPlacementAttempts", 1024U},
            {"delayedActivationProbability", 0.25F},
            {"minimumActivationDelaySteps", 256U},
            {"maximumActivationDelaySteps", 768U}
        }},
        {"restrictedZones", {
            {"enabled", true},
            {"items", nlohmann::json::array({
                {{"id", "red_zone"}, {"type", "RestrictedZone"},
                 {"platform", "restrictedzone"}, {"side", "Hostile"},
                 {"enabled", true}, {"enabledProbability", 1.F},
                 {"fixedPosition", {{"north", 1000.F}, {"east", -5000.F}}}}
            })}
        }}
    };
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool Same(defense::Spawn const& left, defense::Spawn const& right)
{
    return left.north == right.north && left.east == right.east
        && left.heading == right.heading && left.speed == right.speed
        && left.destroyed == right.destroyed;
}

bool Same(defense::EpisodeSpawnLayout const& left,
          defense::EpisodeSpawnLayout const& right)
{
    if (left.side != right.side
        || !Same(left.objective, right.objective)
        || left.red.size() != right.red.size()
        || left.redActivationSteps != right.redActivationSteps
        || !Same(left.friendlyRestrictedZone, right.friendlyRestrictedZone)
        || left.restrictedZones.size() != right.restrictedZones.size())
        return false;
    for (std::size_t index = 0U; index < left.red.size(); ++index)
        if (!Same(left.red[index], right.red[index]))
            return false;
    for (std::size_t index = 0U; index < left.restrictedZones.size(); ++index)
    {
        if (left.restrictedZones[index].id != right.restrictedZones[index].id
            || !Same(left.restrictedZones[index].spawn,
                     right.restrictedZones[index].spawn))
            return false;
    }
    return true;
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool CheckTeam(std::size_t const blueCount, std::size_t const redCount)
{
    auto document = ValidScenario();
    document["blueTeam"]["activeCount"] = blueCount;
    document["redTeam"]["activeCount"] = redCount;
    std::string diagnostic;
    auto const parsed = defense::ParseSpawnSettings(document, &diagnostic);
    if (!parsed || !diagnostic.empty() || !defense::Valid(*parsed)
        || !parsed->blueTeam || parsed->blueTeam->activeCount != blueCount
        || !parsed->redTeam || parsed->redTeam->activeCount != redCount)
        return false;

    defense::WorldBounds const world{22329.375F, 35727.F};
    for (int side : {-1, 1})
    {
        for (unsigned seed = 0U; seed < 16U; ++seed)
        {
            std::mt19937 firstRng{seed};
            std::mt19937 repeatedRng{seed};
            auto const first = defense::GenerateEpisodeSpawnLayout(
                side, *parsed, world, firstRng, &diagnostic);
            auto const repeated = defense::GenerateEpisodeSpawnLayout(
                side, *parsed, world, repeatedRng);
            if (!first || !repeated || !diagnostic.empty()
                || !Same(*first, *repeated)
                || first->red.size() != redCount
                || first->redActivationSteps.size() != redCount
                || first->restrictedZones.size() != 1U)
                return false;

            auto const objective = defense::MakeSpawn(
                "blue_objective", *first, world, firstRng);
            auto const friendlyZone = defense::MakeSpawn(
                "blue_restricted_zone", *first, world, firstRng);
            auto const hostileZone = defense::MakeSpawn(
                "red_zone", *first, world, firstRng);
            if (!objective || !friendlyZone || !hostileZone
                || objective->east * static_cast<float>(side) <= 0.F
                || defense::MakeSpawn("blue_guard_1", *first, world, firstRng)
                || defense::MakeSpawn("red_0", *first, world, firstRng)
                || defense::MakeSpawn("red_11", *first, world, firstRng))
                return false;

            for (std::size_t red = 0U; red < redCount; ++red)
            {
                auto const name = "red_" + std::to_string(red + 1U);
                auto const spawn = defense::MakeSpawn(
                    name, *first, world, firstRng);
                if (!spawn || !std::isfinite(spawn->north)
                    || !std::isfinite(spawn->east)
                    || !std::isfinite(spawn->heading)
                    || !std::isfinite(spawn->speed)
                    || spawn->speed < 0.F
                    || spawn->destroyed !=
                        (first->redActivationSteps[red] != 0U))
                    return false;
                for (std::size_t other = red + 1U; other < redCount; ++other)
                {
                    float const distance = std::hypot(
                        spawn->north - first->red[other].north,
                        spawn->east - first->red[other].east);
                    if (distance + 1.e-3F < parsed->red.minimumSeparation)
                        return false;
                }
            }
            if (redCount == 0U
                && (defense::HasPendingThreats(*first, 0U)
                    || defense::MakeSpawn("red_1", *first, world, firstRng)))
                return false;
        }
    }
    return true;
}

/**
 * @brief Réalise l'opération RejectInvalidSettings; voir les préconditions et effets dans son corps C++.
 */
bool RejectInvalidSettings()
{
    auto document = ValidScenario();
    std::string diagnostic;
    document.erase("blueTeam");
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document.erase("redTeam");
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document["blueTeam"]["activeCount"] = 0U;
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;
    document["blueTeam"]["activeCount"] = 10U;
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document["redTeam"]["activeCount"] = 11U;
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document["red2InitiallyDestroyedProbability"] = 0.25F;
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document["threatCurriculum"] = {{"enabled", false}};
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document["redTeam"]["maximumPlacementAttempts"] = 0U;
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document["restrictedZones"]["items"][0]["enabledProbability"] = 2.F;
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;

    document = ValidScenario();
    document["restrictedZones"]["items"][0]["randomPosition"] = {
        {"north", {{"minimum", 0.F}, {"maximum", 2.F}}},
        {"east", {{"minimum", 0.F}, {"maximum", 2.F}}}
    };
    if (defense::ParseSpawnSettings(document, &diagnostic))
        return false;
    return true;
}

bool CheckTriangularScenario(std::size_t blues)
{
    auto const fail = [](int line) {
        std::cerr << "[SPAWN_TEST_ERROR] CheckTriangularScenario line "
                  << line << '\\n';
        return false;
    };
    auto document = ValidScenario();
    document["blueTeam"]["activeCount"] = blues;
    defense::WorldBounds const world{22329.375F, 35727.F};
    std::vector<defense::Spawn> initialBlues;
    initialBlues.reserve(blues);
    for (std::size_t index = 0U; index < blues; ++index)
        initialBlues.push_back(
            {static_cast<float>(index) * 500.F, 0.F, 0.F, 0.F, false});

    for (std::size_t reds = 0U; reds <= blues; ++reds)
        for (bool const hostile : {false, true})
            for (bool const randomObjective : {false, true})
            {
                document["redTeam"]["activeCount"] = reds;
                document["restrictedZones"]["enabled"] = hostile;
                document["objectiveSpawn"]["mode"] =
                    randomObjective ? "randomized" : "fixed";
                auto parsed = defense::ParseSpawnSettings(document);
                if (!parsed)
                    return fail(__LINE__);
                for (auto& zone : parsed->restrictedZones.items)
                    zone.lethalRadiusMeters = 7000.F;
                for (int const side : {-1, 1})
                    for (unsigned seed = 0U; seed < 8U; ++seed)
                    {
                        std::mt19937 rng{seed};
                        std::mt19937 replay{seed};
                        auto const first = defense::GenerateEpisodeSpawnLayout(
                            side, *parsed, world, rng, nullptr, &initialBlues);
                        auto const second = defense::GenerateEpisodeSpawnLayout(
                            side, *parsed, world, replay, nullptr, &initialBlues);
                        if (!first || !second || !Same(*first, *second)
                            || first->red.size() != reds
                            || first->objective.destroyed
                            || !std::isfinite(first->objective.north)
                            || !std::isfinite(first->objective.east))
                            return fail(__LINE__);
                        if (!hostile && !first->restrictedZones.front().spawn.destroyed)
                            return fail(__LINE__);
                        if (hostile && randomObjective)
                            for (auto const& zone : first->restrictedZones)
                                if (std::hypot(
                                    first->objective.north - zone.spawn.north,
                                    first->objective.east - zone.spawn.east)
                                        <= 8000.F)
                                    return fail(__LINE__);
                    }
            }
    return true;
}

bool CheckRandomizedObjective()
{
    auto const fail = [](int line) {
        std::cerr << "[SPAWN_TEST_ERROR] CheckRandomizedObjective line "
                  << line << '\\n';
        return false;
    };
    auto document = ValidScenario();
    document["redTeam"]["activeCount"] = 2U;
    document["objectiveSpawn"] = {
        {"mode", "randomized"},
        {"northFraction", {{"minimum", -0.65F}, {"maximum", 0.65F}}},
        {"eastFraction", {{"minimum", 0.55F}, {"maximum", 0.85F}}},
        {"zoneSafetyMarginMeters", 2000.F},
        {"minimumBlueSeparationMeters", 2500.F}
    };
    document["restrictedZones"]["items"].push_back({
        {"id", "random_zone"}, {"type", "RestrictedZone"},
        {"platform", "restrictedzone"}, {"side", "Hostile"},
        {"enabled", true}, {"enabledProbability", 1.F},
        {"randomPosition", {
            {"north", {{"minimum", -12000.F}, {"maximum", 12000.F}}},
            {"east", {{"minimum", 18000.F}, {"maximum", 29000.F}}}
        }}
    });
    auto parsed = defense::ParseSpawnSettings(document);
    if (!parsed)
        return fail(__LINE__);
    for (auto& zone : parsed->restrictedZones.items)
        zone.lethalRadiusMeters = 7000.F;
    defense::WorldBounds const world{22329.375F, 35727.F};
    std::vector<defense::Spawn> const blueSpawns{
        {0.F, 0.F, 0.F, 0.F, false}, {1000.F, 1000.F, 0.F, 0.F, false}};
    std::set<std::pair<float, float>> samples;
    for (int side : {-1, 1})
        for (unsigned seed = 0U; seed < 128U; ++seed)
        {
            std::mt19937 rng{seed};
            std::mt19937 replay{seed};
            std::string diagnostic;
            auto const layout = defense::GenerateEpisodeSpawnLayout(
                side, *parsed, world, rng, &diagnostic, &blueSpawns);
            auto const repeated = defense::GenerateEpisodeSpawnLayout(
                side, *parsed, world, replay, nullptr, &blueSpawns);
            if (!layout || !repeated || !Same(*layout, *repeated)
                || !diagnostic.empty() || layout->objective.destroyed)
                return fail(__LINE__);
            samples.emplace(layout->objective.north, layout->objective.east);
            auto const objective = defense::MakeSpawn(
                "blue_objective", *layout, world, rng);
            if (!objective || !Same(*objective, layout->objective)
                || objective->east * static_cast<float>(side) <= 0.F
                || std::fabs(objective->north) >= world.halfNorth - 1000.F
                || std::fabs(objective->east) >= world.halfEast - 1000.F)
                return fail(__LINE__);
            for (auto const& zone : layout->restrictedZones)
                if (std::hypot(objective->north - zone.spawn.north,
                               objective->east - zone.spawn.east) <= 9000.F)
                    return fail(__LINE__);
            for (auto const& red : layout->red)
            {
                float const desired = std::atan2(
                    objective->east - red.east, objective->north - red.north);
                if (std::fabs(std::remainder(red.heading - desired,
                                            6.283185307179586F)) > 0.1201F)
                    return fail(__LINE__);
            }
        }
    if (samples.size() < 16U)
        return fail(__LINE__);

    // Deterministic impossible geometry must fail without unsafe fallback.
    parsed->objective.northFraction = {0.F, 0.F};
    parsed->objective.eastFraction = {0.6F, 0.6F};
    parsed->objective.maximumPlacementAttempts = 3U;
    parsed->restrictedZones.items[0].fixedPosition = {
        0.F, 0.6F * world.halfEast};
    parsed->restrictedZones.items[1].enabled = false;
    std::mt19937 impossible{17U};
    std::string diagnostic;
    if (defense::GenerateEpisodeSpawnLayout(
            1, *parsed, world, impossible, &diagnostic, &blueSpawns)
        || diagnostic.find("cannot place objective") == std::string::npos)
        return fail(__LINE__);
    document["objectiveSpawn"]["mode"] = "unknown";
    return !defense::ParseSpawnSettings(document);
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    if (!CheckTeam(1U, 0U)) { std::cerr << "[SPAWN_TEST_ERROR] 1v0\n"; return 1; }
    if (!CheckTeam(2U, 2U)) { std::cerr << "[SPAWN_TEST_ERROR] 2v2\n"; return 2; }
    if (!CheckTeam(4U, 4U)) { std::cerr << "[SPAWN_TEST_ERROR] 4v4\n"; return 3; }
    if (!CheckTeam(9U, 10U)) { std::cerr << "[SPAWN_TEST_ERROR] 9v10\n"; return 4; }
    if (!RejectInvalidSettings()) { std::cerr << "[SPAWN_TEST_ERROR] invalid settings\n"; return 5; }
    if (!CheckRandomizedObjective()) return 6;
    if (!CheckTriangularScenario(3U)) return 7;
    if (!CheckTriangularScenario(4U)) return 8;
    return 0;
}
