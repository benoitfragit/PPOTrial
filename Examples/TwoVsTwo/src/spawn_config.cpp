#include "spawn_config.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <string>
#include <string_view>

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

} // namespace

std::optional<SpawnSettings> ParseSpawnSettings(
    nlohmann::json const& settings)
{
    if (!settings.is_object())
        return std::nullopt;
    auto const red2Probability = Number(
        settings, "red2InitiallyDestroyedProbability");
    auto const redItem = settings.find("redSpawn");
    auto const zoneItem = settings.find("restrictedZoneSpawn");
    if (!red2Probability || redItem == settings.end() || !redItem->is_object()
        || zoneItem == settings.end() || !zoneItem->is_object())
        return std::nullopt;

    auto const minimumScale = Number(*redItem, "minimumWorldScale");
    auto const maximumScale = Number(*redItem, "maximumWorldScale");
    auto const minimumBearing = Number(*redItem, "minimumBearingDegrees");
    auto const maximumBearing = Number(*redItem, "maximumBearingDegrees");
    auto const minimumSeparation = Number(*redItem, "minimumSeparation");
    auto const zoneProbability = Number(*zoneItem, "enabledProbability");
    auto const minimumNorth = Number(*zoneItem, "minimumNorthFraction");
    auto const maximumNorth = Number(*zoneItem, "maximumNorthFraction");
    auto const minimumEast = Number(*zoneItem, "minimumEastFraction");
    auto const maximumEast = Number(*zoneItem, "maximumEastFraction");
    if (!minimumScale || !maximumScale || !minimumBearing || !maximumBearing
        || !minimumSeparation || !zoneProbability || !minimumNorth
        || !maximumNorth || !minimumEast || !maximumEast)
        return std::nullopt;

    SpawnSettings result{};
    result.red2DestroyedProbability = *red2Probability;
    result.red.worldScale = {*minimumScale, *maximumScale};
    result.red.bearingRadians = {
        *minimumBearing * kDegreesToRadians,
        *maximumBearing * kDegreesToRadians
    };
    result.red.minimumSeparation = *minimumSeparation;
    result.restrictedZone.enabledProbability = *zoneProbability;
    result.restrictedZone.northFraction = {*minimumNorth, *maximumNorth};
    result.restrictedZone.eastFraction = {*minimumEast, *maximumEast};

    auto const curriculumItem = settings.find("threatCurriculum");
    if (curriculumItem != settings.end())
    {
        if (!curriculumItem->is_object())
            return std::nullopt;
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
            return std::nullopt;

        auto& curriculum = result.threatCurriculum;
        curriculum.enabled = enabled->get<bool>();
        curriculum.noThreatProbability = *noThreat;
        curriculum.singleThreatProbability = *singleThreat;
        curriculum.delayedFirstThreatProbability = *delayedFirst;
        curriculum.delayedSecondThreatProbability = *delayedSecond;
        curriculum.firstDelaySteps = {*minimumFirst, *maximumFirst};
        curriculum.secondDelaySteps = {*minimumSecond, *maximumSecond};
    }
    return Valid(result) ? std::optional<SpawnSettings>{result} : std::nullopt;
}

} // namespace ppo::examples::twovstwo
