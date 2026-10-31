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
#include <cstddef>
#include <random>
#include <string>

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
    if (left.side != right.side || left.red.size() != right.red.size()
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

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    if (!CheckTeam(1U, 0U)) return 1;
    if (!CheckTeam(2U, 2U)) return 2;
    if (!CheckTeam(4U, 4U)) return 3;
    if (!CheckTeam(9U, 10U)) return 4;
    if (!RejectInvalidSettings()) return 5;
    return 0;
}
