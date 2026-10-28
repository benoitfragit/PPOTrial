#include "spawn_config.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_set>

namespace ppo::examples::twovstwo {
namespace {

constexpr float kDegreesToRadians = 0.017453292519943295F;

std::optional<float> Number(nlohmann::json const& object,
                            std::string_view key)
{
    auto const item = object.find(std::string{key});
    if (item == object.end() || !item->is_number())
        return std::nullopt;
    float const value = item->get<float>();
    return std::isfinite(value) ? std::optional<float>{value} : std::nullopt;
}

std::optional<std::size_t> Step(nlohmann::json const& object,
                                std::string_view key)
{
    auto const item = object.find(std::string{key});
    if (item == object.end() || !item->is_number_unsigned())
        return std::nullopt;
    return item->get<std::size_t>();
}

bool OnlyKeys(nlohmann::json const& object,
              std::initializer_list<std::string_view> allowed,
              std::string_view path,
              std::string& diagnostic)
{
    for (auto const& [key, value] : object.items())
    {
        (void)value;
        if (std::find(allowed.begin(), allowed.end(), std::string_view{key})
            == allowed.end())
        {
            diagnostic = std::string{path} + ": unknown field '" + key + "'";
            return false;
        }
    }
    return true;
}

std::optional<SpawnRange> ParseRange(nlohmann::json const& object,
                                     std::string_view path,
                                     std::string& diagnostic)
{
    if (!object.is_object()
        || !OnlyKeys(object, {"minimum", "maximum"}, path, diagnostic))
        return std::nullopt;
    auto const minimum = Number(object, "minimum");
    auto const maximum = Number(object, "maximum");
    if (!minimum || !maximum || *minimum > *maximum)
    {
        diagnostic = std::string{path}
            + ": minimum and maximum must be finite and ordered";
        return std::nullopt;
    }
    return SpawnRange{*minimum, *maximum};
}

bool ParseRestrictedZones(nlohmann::json const& settings,
                          SpawnSettings& result,
                          std::string& diagnostic)
{
    auto const collection = settings.find("restrictedZones");
    if (collection == settings.end())
        return true;
    if (!collection->is_object()
        || !OnlyKeys(*collection, {"enabled", "items"},
                     "restrictedZones", diagnostic))
        return false;
    auto const globalEnabled = collection->find("enabled");
    auto const items = collection->find("items");
    if (globalEnabled == collection->end() || !globalEnabled->is_boolean()
        || items == collection->end() || !items->is_array())
    {
        diagnostic = "restrictedZones: enabled(bool) and items(array) are required";
        return false;
    }

    result.restrictedZones.enabled = globalEnabled->get<bool>();
    result.restrictedZones.items.reserve(items->size());
    std::unordered_set<std::string> ids;
    std::size_t index = 0U;
    for (auto const& item : *items)
    {
        std::string const path = "restrictedZones.items["
            + std::to_string(index) + "]";
        ++index;
        if (!item.is_object()
            || !OnlyKeys(item,
                {"id", "type", "platform", "side", "enabled",
                 "enabledProbability", "fixedPosition", "randomPosition"},
                path, diagnostic))
            return false;
        auto const id = item.find("id");
        auto const type = item.find("type");
        auto const platform = item.find("platform");
        auto const side = item.find("side");
        auto const enabled = item.find("enabled");
        auto const probability = Number(item, "enabledProbability");
        if (id == item.end() || !id->is_string() || id->get<std::string>().empty()
            || type == item.end() || !type->is_string()
            || type->get<std::string>() != "RestrictedZone"
            || platform == item.end() || !platform->is_string()
            || platform->get<std::string>().empty()
            || side == item.end() || !side->is_string()
            || side->get<std::string>() != "Hostile"
            || enabled == item.end() || !enabled->is_boolean()
            || !probability || *probability < 0.F || *probability > 1.F)
        {
            diagnostic = path
                + ": invalid id/type/platform/side/enabled/enabledProbability";
            return false;
        }
        std::string const zoneId = id->get<std::string>();
        if (!ids.emplace(zoneId).second)
        {
            diagnostic = path + ": duplicate id '" + zoneId + "'";
            return false;
        }

        auto const fixed = item.find("fixedPosition");
        auto const random = item.find("randomPosition");
        if ((fixed == item.end()) == (random == item.end()))
        {
            diagnostic = path
                + ": exactly one of fixedPosition or randomPosition is required";
            return false;
        }

        RestrictedZoneSpawnSettings zone{};
        zone.id = zoneId;
        zone.platform = platform->get<std::string>();
        zone.enabled = enabled->get<bool>();
        zone.enabledProbability = *probability;
        if (fixed != item.end())
        {
            if (!fixed->is_object()
                || !OnlyKeys(*fixed, {"north", "east"},
                             path + ".fixedPosition", diagnostic))
                return false;
            auto const north = Number(*fixed, "north");
            auto const east = Number(*fixed, "east");
            if (!north || !east)
            {
                diagnostic = path
                    + ".fixedPosition: finite north/east values are required";
                return false;
            }
            zone.positionMode = RestrictedZonePositionMode::Fixed;
            zone.fixedPosition = {*north, *east};
        }
        else
        {
            if (!random->is_object()
                || !OnlyKeys(*random, {"north", "east"},
                             path + ".randomPosition", diagnostic)
                || !random->contains("north") || !random->contains("east"))
                return false;
            auto const north = ParseRange(
                random->at("north"), path + ".randomPosition.north",
                diagnostic);
            auto const east = ParseRange(
                random->at("east"), path + ".randomPosition.east",
                diagnostic);
            if (!north || !east)
                return false;
            zone.positionMode = RestrictedZonePositionMode::Random;
            zone.randomNorth = *north;
            zone.randomEast = *east;
        }
        result.restrictedZones.items.push_back(std::move(zone));
    }
    return true;
}

bool ParseRedTeam(nlohmann::json const& settings,
                  SpawnSettings& result,
                  std::string& diagnostic)
{
    auto const item = settings.find("redTeam");
    if (item == settings.end())
        return true;
    if (!item->is_object())
    {
        diagnostic = "redTeam must be an object";
        return false;
    }
    if (!OnlyKeys(*item,
            {"activeCount", "maximumPlacementAttempts",
             "delayedActivationProbability",
             "minimumActivationDelaySteps",
             "maximumActivationDelaySteps"},
            "redTeam", diagnostic))
        return false;

    auto const count = Step(*item, "activeCount");
    auto const attempts = Step(*item, "maximumPlacementAttempts");
    auto const delayed = Number(*item, "delayedActivationProbability");
    auto const minimumDelay = Step(*item, "minimumActivationDelaySteps");
    auto const maximumDelay = Step(*item, "maximumActivationDelaySteps");
    if (!count)
    {
        diagnostic = "redTeam.activeCount must be an unsigned integer in [0, "
            + std::to_string(defenders::TeamCapacity::MaximumRedMobiles)
            + "]";
        return false;
    }
    if (*count > defenders::TeamCapacity::MaximumRedMobiles)
    {
        diagnostic = "redTeam.activeCount exceeds MaximumRedMobiles="
            + std::to_string(defenders::TeamCapacity::MaximumRedMobiles);
        return false;
    }
    if (!attempts || *attempts == 0U
        || *attempts > RedTeamSpawnSettings::MaximumPlacementAttempts)
    {
        diagnostic = "redTeam.maximumPlacementAttempts must be in [1, "
            + std::to_string(
                RedTeamSpawnSettings::MaximumPlacementAttempts) + "]";
        return false;
    }
    if (!delayed || *delayed < 0.F || *delayed > 1.F)
    {
        diagnostic = "redTeam.delayedActivationProbability must be in [0, 1]";
        return false;
    }
    if (!minimumDelay || !maximumDelay || *minimumDelay == 0U
        || *minimumDelay > *maximumDelay
        || *maximumDelay >= EpisodeSpawnLayout::NeverActivate / 2U)
    {
        diagnostic = "redTeam activation delay steps must be positive, finite and ordered";
        return false;
    }

    result.redTeam = RedTeamSpawnSettings{
        *count, *attempts, *delayed, {*minimumDelay, *maximumDelay}};
    return true;
}

bool ParseBlueTeam(nlohmann::json const& settings,
                   SpawnSettings& result,
                   std::string& diagnostic)
{
    auto const item = settings.find("blueTeam");
    if (item == settings.end())
        return true;
    if (!item->is_object()
        || !OnlyKeys(*item, {"activeCount"}, "blueTeam", diagnostic))
    {
        if (diagnostic.empty())
            diagnostic = "blueTeam must be an object";
        return false;
    }

    auto const count = Step(*item, "activeCount");
    if (!count
        || *count < defenders::TeamCapacity::MinimumBlueMobiles
        || *count > defenders::TeamCapacity::MaximumBlueMobiles)
    {
        diagnostic = "blueTeam.activeCount must be an unsigned integer in ["
            + std::to_string(defenders::TeamCapacity::MinimumBlueMobiles)
            + ", "
            + std::to_string(defenders::TeamCapacity::MaximumBlueMobiles)
            + "]";
        return false;
    }
    result.blueTeam = BlueTeamSpawnSettings{*count};
    return true;
}

} // namespace

std::optional<SpawnSettings> ParseSpawnSettings(
    nlohmann::json const& settings,
    std::string* diagnostic)
{
    std::string error;
    auto fail = [&](std::string message) -> std::optional<SpawnSettings> {
        if (diagnostic != nullptr)
            *diagnostic = std::move(message);
        return std::nullopt;
    };
    if (!settings.is_object())
        return fail("twoVsTwo must be an object");
    auto const red2Probability = Number(
        settings, "red2InitiallyDestroyedProbability");
    auto const red2Item = settings.find(
        "red2InitiallyDestroyedProbability");
    auto const redItem = settings.find("redSpawn");
    auto const friendlyZoneItem = settings.find("restrictedZoneSpawn");
    bool const hasGeneralizedRedTeam = settings.contains("redTeam");
    if (red2Item != settings.end() && !red2Probability)
        return fail("red2InitiallyDestroyedProbability must be finite and numeric");
    if ((!hasGeneralizedRedTeam && !red2Probability)
        || redItem == settings.end() || !redItem->is_object()
        || friendlyZoneItem == settings.end()
        || !friendlyZoneItem->is_object())
        return fail(hasGeneralizedRedTeam
            ? "twoVsTwo requires redSpawn and restrictedZoneSpawn"
            : "archived 2-vs-2 requires red2InitiallyDestroyedProbability, redSpawn and restrictedZoneSpawn");

    auto const minimumScale = Number(*redItem, "minimumWorldScale");
    auto const maximumScale = Number(*redItem, "maximumWorldScale");
    auto const minimumBearing = Number(*redItem, "minimumBearingDegrees");
    auto const maximumBearing = Number(*redItem, "maximumBearingDegrees");
    auto const minimumSeparation = Number(*redItem, "minimumSeparation");
    auto const zoneProbability = Number(
        *friendlyZoneItem, "enabledProbability");
    auto const minimumNorth = Number(
        *friendlyZoneItem, "minimumNorthFraction");
    auto const maximumNorth = Number(
        *friendlyZoneItem, "maximumNorthFraction");
    auto const minimumEast = Number(
        *friendlyZoneItem, "minimumEastFraction");
    auto const maximumEast = Number(
        *friendlyZoneItem, "maximumEastFraction");
    if (!minimumScale || !maximumScale || !minimumBearing || !maximumBearing
        || !minimumSeparation || !zoneProbability || !minimumNorth
        || !maximumNorth || !minimumEast || !maximumEast)
        return fail("redSpawn or restrictedZoneSpawn contains a missing or non-finite numeric value");

    SpawnSettings result{};
    result.red2DestroyedProbability = red2Probability.value_or(0.F);
    result.red.worldScale = {*minimumScale, *maximumScale};
    result.red.bearingRadians = {
        *minimumBearing * kDegreesToRadians,
        *maximumBearing * kDegreesToRadians
    };
    result.red.minimumSeparation = *minimumSeparation;
    result.friendlyRestrictedZone.enabledProbability = *zoneProbability;
    result.friendlyRestrictedZone.northFraction = {
        *minimumNorth, *maximumNorth};
    result.friendlyRestrictedZone.eastFraction = {
        *minimumEast, *maximumEast};
    if (!ParseRedTeam(settings, result, error))
        return fail(error);
    if (!ParseBlueTeam(settings, result, error))
        return fail(error);
    if (!ParseRestrictedZones(settings, result, error))
        return fail(error);

    auto const curriculumItem = settings.find("threatCurriculum");
    if (curriculumItem != settings.end())
    {
        if (!curriculumItem->is_object())
            return fail("threatCurriculum must be an object");
        auto const enabled = curriculumItem->find("enabled");
        auto const noThreat = Number(*curriculumItem, "noThreatProbability");
        auto const singleThreat = Number(
            *curriculumItem, "singleThreatProbability");
        auto const delayedFirst = Number(
            *curriculumItem, "delayedFirstThreatProbability");
        auto const delayedSecond = Number(
            *curriculumItem, "delayedSecondThreatProbability");
        auto const minimumFirst = Step(
            *curriculumItem, "minimumFirstDelaySteps");
        auto const maximumFirst = Step(
            *curriculumItem, "maximumFirstDelaySteps");
        auto const minimumSecond = Step(
            *curriculumItem, "minimumSecondDelaySteps");
        auto const maximumSecond = Step(
            *curriculumItem, "maximumSecondDelaySteps");
        if (enabled == curriculumItem->end() || !enabled->is_boolean()
            || !noThreat || !singleThreat || !delayedFirst || !delayedSecond
            || !minimumFirst || !maximumFirst
            || !minimumSecond || !maximumSecond)
            return fail("threatCurriculum contains a missing or invalid field");

        auto& curriculum = result.threatCurriculum;
        curriculum.enabled = enabled->get<bool>();
        curriculum.noThreatProbability = *noThreat;
        curriculum.singleThreatProbability = *singleThreat;
        curriculum.delayedFirstThreatProbability = *delayedFirst;
        curriculum.delayedSecondThreatProbability = *delayedSecond;
        curriculum.firstDelaySteps = {*minimumFirst, *maximumFirst};
        curriculum.secondDelaySteps = {*minimumSecond, *maximumSecond};
    }
    if (result.redTeam && result.threatCurriculum.enabled)
        return fail("redTeam and enabled threatCurriculum are mutually exclusive");
    if (!Valid(result))
        return fail("twoVsTwo spawn settings are internally inconsistent");
    if (diagnostic != nullptr)
        diagnostic->clear();
    return result;
}

} // namespace ppo::examples::twovstwo
