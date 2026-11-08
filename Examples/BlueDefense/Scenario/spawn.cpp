/**
 * @file Examples/BlueDefense/Scenario/spawn.cpp
 * @brief Placement physique des Bleus, Rouges et zones selon le scénario.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "spawn.hpp"
#include "restrictedzonegeometry.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <string>

namespace ppo::examples::bluedefense {
namespace {

constexpr float kHalfPi = 1.5707963267948966F;
constexpr float kHeadingJitter = 0.12F;
constexpr float kNominalRedSpeed = 220.F;
constexpr float kRedSpeedJitter = 40.F;

/**
 * @brief Réalise l'opération FiniteRange; voir les préconditions et effets dans son corps C++.
 */
bool FiniteRange(SpawnRange const& range) noexcept
{
    return std::isfinite(range.minimum) && std::isfinite(range.maximum)
        && range.minimum <= range.maximum;
}

bool ValidWorld(WorldBounds const& world) noexcept
{
    return std::isfinite(world.halfNorth) && std::isfinite(world.halfEast)
        && world.halfNorth > 0.F && world.halfEast > 0.F;
}

/**
 * @brief Construit, transforme ou échantillonne les données produites par Sample.
 */
float Sample(SpawnRange const& range, std::mt19937& rng)
{
    return std::uniform_real_distribution<float>{
        range.minimum, range.maximum}(rng);
}

/**
 * @brief Construit, transforme ou échantillonne les données produites par SampleStep.
 */
std::size_t SampleStep(std::array<std::size_t, 2> const& range,
                       std::mt19937& rng)
{
    return std::uniform_int_distribution<std::size_t>{
        range[0], range[1]}(rng);
}

/**
 * @brief Réalise l'opération Probability; voir les préconditions et effets dans son corps C++.
 * @details Applique des bornes numériques.
 */
bool Probability(float value) noexcept
{
    return std::isfinite(value) && value >= 0.F && value <= 1.F;
}

bool ValidDelayRange(std::array<std::size_t, 2> const& range) noexcept
{
    return range[0] > 0U && range[0] <= range[1]
        && range[1] < EpisodeSpawnLayout::NeverActivate / 2U;
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Applique des bornes numériques.
 */
bool ValidRestrictedZone(RestrictedZoneSpawnSettings const& zone) noexcept
{
    if (zone.id.empty() || zone.platform.empty()
        || !Probability(zone.enabledProbability)
        || !std::isfinite(zone.lethalRadiusMeters)
        || zone.lethalRadiusMeters < 0.F)
        return false;
    if (zone.positionMode == RestrictedZonePositionMode::Fixed)
        return std::isfinite(zone.fixedPosition[0])
            && std::isfinite(zone.fixedPosition[1]);
    return zone.positionMode == RestrictedZonePositionMode::Random
        && FiniteRange(zone.randomNorth) && FiniteRange(zone.randomEast);
}

/** @brief Déclare le type Position, utilisé par Placement physique des Bleus, Rouges et zones selon le scénario. */
struct Position final
{
    float north{};
    float east{};
};

/**
 * @brief Réalise l'opération PositionOnExpandedBoundary; voir les préconditions et effets dans son corps C++.
 * @details Applique des bornes numériques.
 */
Position PositionOnExpandedBoundary(float bearing, float scale, float side,
                                    WorldBounds const& world) noexcept
{
    float const normalizedNorth = std::sin(bearing);
    float const normalizedEast = -side * std::cos(bearing);
    float const boundaryFactor = 1.F / std::max(
        std::fabs(normalizedNorth), std::fabs(normalizedEast));
    return {
        scale * boundaryFactor * normalizedNorth * world.halfNorth,
        scale * boundaryFactor * normalizedEast * world.halfEast
    };
}

/**
 * @brief Construit, transforme ou échantillonne les données produites par SampleRedPosition.
 */
Position SampleRedPosition(RedSpawnSettings const& settings, float side,
                           WorldBounds const& world, std::mt19937& rng)
{
    return PositionOnExpandedBoundary(
        Sample(settings.bearingRadians, rng),
        Sample(settings.worldScale, rng), side, world);
}

float Separation(Position const& first, Position const& second) noexcept
{
    return std::hypot(first.north - second.north,
                      first.east - second.east);
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool IsSeparatedFromAll(Position const& candidate,
                        std::vector<Position> const& accepted,
                        float minimumSeparation) noexcept
{
    return std::all_of(
        accepted.begin(), accepted.end(),
        [&](Position const& other) {
            return Separation(candidate, other) >= minimumSeparation;
        });
}

/**
 * @brief Réalise l'opération RedIndex; voir les préconditions et effets dans son corps C++.
 */
std::optional<std::size_t> RedIndex(std::string_view name) noexcept
{
    constexpr std::string_view prefix{"red_"};
    if (name.size() <= prefix.size()
        || name.substr(0U, prefix.size()) != prefix)
        return std::nullopt;

    std::string_view const suffix = name.substr(prefix.size());
    if (suffix.front() == '0')
        return std::nullopt;
    std::size_t oneBasedIndex = 0U;
    auto const result = std::from_chars(
        suffix.data(), suffix.data() + suffix.size(), oneBasedIndex);
    if (result.ec != std::errc{} || result.ptr != suffix.data() + suffix.size()
        || oneBasedIndex == 0U)
        return std::nullopt;
    return oneBasedIndex - 1U;
}

/**
 * @brief Construit, transforme ou échantillonne les données produites par MakeRedSpawn.
 */
Spawn MakeRedSpawn(Position const& position, Position const& objective,
                   std::mt19937& rng, bool destroyed)
{
    float const headingToObjective = std::atan2(
        objective.east - position.east, objective.north - position.north);
    float const heading = headingToObjective
        + std::uniform_real_distribution<float>{
            -kHeadingJitter, kHeadingJitter}(rng);
    float const speed = kNominalRedSpeed
        + std::uniform_real_distribution<float>{
            -kRedSpeedJitter, kRedSpeedJitter}(rng);
    return {position.north, position.east, heading, speed, destroyed};
}

/** @brief Sample all hostile zones exactly once in scenario declaration order. */
void SampleRestrictedZones(SpawnSettings const& settings, float normalizedSide,
                           std::mt19937& rng, EpisodeSpawnLayout& result)
{
    result.restrictedZones.reserve(settings.restrictedZones.items.size());
    for (auto const& zone : settings.restrictedZones.items)
    {
        bool const active = settings.restrictedZones.enabled && zone.enabled
            && std::bernoulli_distribution{
                static_cast<double>(zone.enabledProbability)}(rng);
        float north = zone.fixedPosition[0];
        float east = zone.fixedPosition[1];
        if (zone.positionMode == RestrictedZonePositionMode::Random)
        {
            north = Sample(zone.randomNorth, rng);
            east = Sample(zone.randomEast, rng);
        }
        if (zone.oppositeObjective)
            east = -normalizedSide * std::abs(east);
        result.restrictedZones.push_back(
            RestrictedZoneEpisodeSpawn{zone.id,
                Spawn{north, east, 0.F, 0.F, !active}});
    }
}

/** @brief Enforce world, blue/red separation and the full hostile-zone radius. */
bool SafeObjective(Position const& candidate,
                   ObjectiveSpawnSettings const& settings,
                   RestrictedZonesSpawnSettings const& zones,
                   EpisodeSpawnLayout const& layout,
                   std::vector<Position> const& reds,
                   std::vector<Spawn> const* initialBlues,
                   WorldBounds const& world) noexcept
{
    if (std::fabs(candidate.north) + settings.worldMarginMeters
            >= world.halfNorth
        || std::fabs(candidate.east) + settings.worldMarginMeters
            >= world.halfEast)
        return false;
    if (!IsSeparatedFromAll(candidate, reds, settings.minimumRedSeparationMeters))
        return false;
    if (initialBlues != nullptr)
        for (Spawn const& blue : *initialBlues)
            if (!std::isfinite(blue.north) || !std::isfinite(blue.east)
                || Separation(candidate, {blue.north, blue.east})
                    <= settings.minimumBlueSeparationMeters)
                return false;
    if (zones.enabled)
        for (std::size_t index = 0U; index < zones.items.size(); ++index)
        {
            auto const& zone = zones.items[index];
            if (!zone.enabled || zone.enabledProbability == 0.F)
                continue;
            float const lethal = zone.lethalRadiusMeters;
            // Check even currently inactive but potentially active zones.
            if (!std::isfinite(lethal) || lethal <= 0.F)
                return false;
            float const exclusion = std::max(
                ppo::model::kRestrictedZoneWarningRadiusScale * lethal,
                lethal + settings.zoneSafetyMarginMeters);
            auto const& centre = layout.restrictedZones[index].spawn;
            if (Separation(candidate, {centre.north, centre.east}) <= exclusion)
                return false;
        }
    return true;
}

} // namespace

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool Valid(SpawnSettings const& settings) noexcept
{
    auto const fractionRange = [](SpawnRange const& range) {
        return FiniteRange(range)
            && range.minimum >= -1.F && range.maximum <= 1.F;
    };
    if (!settings.blueTeam || !settings.redTeam
        || !defenders::Valid(defenders::ActiveTeamCounts{
            settings.blueTeam->activeCount, settings.redTeam->activeCount}))
        return false;

    bool const baseValid = FiniteRange(settings.red.worldScale)
        && settings.red.worldScale.minimum > 1.F
        && FiniteRange(settings.red.bearingRadians)
        && settings.red.bearingRadians.minimum >= -kHalfPi
        && settings.red.bearingRadians.maximum <= kHalfPi
        && std::isfinite(settings.red.minimumSeparation)
        && settings.red.minimumSeparation > 0.F
        && Probability(settings.friendlyRestrictedZone.enabledProbability)
        && fractionRange(settings.friendlyRestrictedZone.northFraction)
        && fractionRange(settings.friendlyRestrictedZone.eastFraction);
    if (!baseValid)
        return false;

    auto const& team = *settings.redTeam;
    if (team.maximumPlacementAttempts == 0U
        || team.maximumPlacementAttempts
            > RedTeamSpawnSettings::MaximumPlacementAttempts
        || !Probability(team.delayedActivationProbability)
        || !ValidDelayRange(team.activationDelaySteps))
        return false;

    auto const& objective = settings.objective;
    auto const fractionRange = [](SpawnRange const& range) {
        return FiniteRange(range) && range.minimum >= -1.F
            && range.maximum <= 1.F;
    };
    if (objective.mode != ObjectiveSpawnMode::Fixed
        && objective.mode != ObjectiveSpawnMode::Randomized)
        return false;
    if (!fractionRange(objective.northFraction)
        || !FiniteRange(objective.eastFraction)
        || objective.eastFraction.minimum < 0.F
        || objective.eastFraction.maximum > 1.F
        || !std::isfinite(objective.worldMarginMeters)
        || !std::isfinite(objective.zoneSafetyMarginMeters)
        || !std::isfinite(objective.minimumBlueSeparationMeters)
        || !std::isfinite(objective.minimumRedSeparationMeters)
        || objective.worldMarginMeters < 0.F
        || objective.zoneSafetyMarginMeters < 0.F
        || objective.minimumBlueSeparationMeters < 0.F
        || objective.minimumRedSeparationMeters < 0.F
        || objective.maximumPlacementAttempts == 0U
        || objective.maximumPlacementAttempts
            > ObjectiveSpawnSettings::MaximumPlacementAttempts)
        return false;
    auto const& zones = settings.restrictedZones.items;
    for (std::size_t index = 0U; index < zones.size(); ++index)
    {
        if (!ValidRestrictedZone(zones[index]))
            return false;
        for (std::size_t other = index + 1U; other < zones.size(); ++other)
            if (zones[index].id == zones[other].id)
                return false;
    }
    return true;
}

/**
 * @brief Construit, transforme ou échantillonne les données produites par GenerateEpisodeSpawnLayout.
 */
std::optional<EpisodeSpawnLayout> GenerateEpisodeSpawnLayout(
    int side, SpawnSettings const& settings, WorldBounds const& world,
    std::mt19937& rng, std::string* diagnostic,
    std::vector<Spawn> const* initialBlues)
{
    auto const fail = [diagnostic](std::string message)
        -> std::optional<EpisodeSpawnLayout> {
        if (diagnostic != nullptr)
            *diagnostic = std::move(message);
        return std::nullopt;
    };
    if (!Valid(settings))
        return fail("blueDefense spawn settings are internally inconsistent");
    if (!ValidWorld(world))
        return fail("world half extents must be finite and strictly positive");

    float const normalizedSide = side < 0 ? -1.F : 1.F;
    EpisodeSpawnLayout result{};
    result.side = normalizedSide < 0.F ? -1 : 1;
    result.objective = Spawn{0.F, normalizedSide * 0.92F * world.halfEast,
                             0.F, 0.F, false};
    bool const randomized =
        settings.objective.mode == ObjectiveSpawnMode::Randomized;

    std::vector<Position> positions;
    if (settings.redTeam)
    {
        auto const& team = *settings.redTeam;
        positions.reserve(team.activeCount);
        for (std::size_t red = 0U; red < team.activeCount; ++red)
        {
            bool placed = false;
            for (std::size_t attempt = 0U;
                 attempt < team.maximumPlacementAttempts; ++attempt)
            {
                Position const candidate = SampleRedPosition(
                    settings.red, normalizedSide, world, rng);
                if (!IsSeparatedFromAll(candidate, positions,
                                        settings.red.minimumSeparation))
                    continue;
                positions.push_back(candidate);
                placed = true;
                break;
            }
            if (!placed)
                return fail("redTeam: unable to place red_"
                    + std::to_string(red + 1U) + " after "
                    + std::to_string(team.maximumPlacementAttempts)
                    + " attempts while enforcing minimumSeparation="
                    + std::to_string(settings.red.minimumSeparation));
        }
    }

    if (randomized)
    {
        // Sample hostile-zone centres before the defended objective: never
        // accept an overlap, even if both distributions are randomized.
        SampleRestrictedZones(settings, normalizedSide, rng, result);
        if (initialBlues != nullptr && settings.blueTeam
            && initialBlues->size() != settings.blueTeam->activeCount)
            return fail("objectiveSpawn: initial Blue count mismatch");
        if (settings.restrictedZones.enabled)
            for (auto const& zone : settings.restrictedZones.items)
                if (zone.enabled && zone.enabledProbability > 0.F
                    && (!std::isfinite(zone.lethalRadiusMeters)
                        || zone.lethalRadiusMeters <= 0.F))
                    return fail("objectiveSpawn: missing physical lethal radius for " + zone.id);

        bool placed = false;
        for (std::size_t attempt = 0U;
             attempt < settings.objective.maximumPlacementAttempts; ++attempt)
        {
            Position const candidate{
                Sample(settings.objective.northFraction, rng) * world.halfNorth,
                normalizedSide * Sample(settings.objective.eastFraction, rng)
                    * world.halfEast};
            if (!SafeObjective(candidate, settings.objective,
                               settings.restrictedZones, result, positions,
                               initialBlues, world))
                continue;
            result.objective.north = candidate.north;
            result.objective.east = candidate.east;
            placed = true;
            break;
        }
        if (!placed)
            return fail("objectiveSpawn: cannot place objective outside world, "
                        "Blue/Red separation and hostile zone exclusion after "
                + std::to_string(settings.objective.maximumPlacementAttempts)
                + " attempts");
    }

    if (settings.redTeam)
    {
        auto const& team = *settings.redTeam;
        result.red.reserve(team.activeCount);
        result.redActivationSteps.reserve(team.activeCount);
        Position const objective{result.objective.north, result.objective.east};
        for (Position const& position : positions)
        {
            std::size_t const activationStep = std::bernoulli_distribution{
                static_cast<double>(team.delayedActivationProbability)}(rng)
                ? SampleStep(team.activationDelaySteps, rng) : 0U;
            result.red.push_back(MakeRedSpawn(
                position, objective, rng, activationStep != 0U));
            result.redActivationSteps.push_back(activationStep);
        }
    }
    bool const friendlyZoneEnabled = std::bernoulli_distribution{
        static_cast<double>(
            settings.friendlyRestrictedZone.enabledProbability)}(rng);
    result.friendlyRestrictedZone = {
        Sample(settings.friendlyRestrictedZone.northFraction, rng)
            * world.halfNorth,
        normalizedSide
            * Sample(settings.friendlyRestrictedZone.eastFraction, rng)
            * world.halfEast,
        0.F, 0.F, !friendlyZoneEnabled
    };
    // Preserve the fixed-mode RNG sequence byte-for-byte: Reds, friendly zone,
    // then hostile zones, exactly as before objective randomization existed.
    if (!randomized)
        SampleRestrictedZones(settings, normalizedSide, rng, result);
    if (diagnostic != nullptr)
        diagnostic->clear();
    return result;
}

/**
 * @brief Construit, transforme ou échantillonne les données produites par MakeSpawn.
 */
std::optional<Spawn> MakeSpawn(std::string_view name,
                               EpisodeSpawnLayout const& layout,
                               WorldBounds const& world,
                               std::mt19937&)
{
    if (!ValidWorld(world))
        return std::nullopt;

    if (name == "blue_objective")
        return layout.objective;
    if (name == "blue_restricted_zone")
        return layout.friendlyRestrictedZone;
    if (auto const red = RedIndex(name); red && *red < layout.red.size())
        return layout.red[*red];
    auto const zone = std::find_if(
        layout.restrictedZones.begin(), layout.restrictedZones.end(),
        [name](RestrictedZoneEpisodeSpawn const& item) {
            return item.id == name;
        });
    if (zone != layout.restrictedZones.end())
        return zone->spawn;
    return std::nullopt;
}

} // namespace ppo::examples::bluedefense
