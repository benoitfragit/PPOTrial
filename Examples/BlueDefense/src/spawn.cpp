#include "spawn.hpp"

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
constexpr unsigned kPairSamplingAttempts = 128U;

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

float Sample(SpawnRange const& range, std::mt19937& rng)
{
    return std::uniform_real_distribution<float>{
        range.minimum, range.maximum}(rng);
}

std::size_t SampleStep(std::array<std::size_t, 2> const& range,
                       std::mt19937& rng)
{
    return std::uniform_int_distribution<std::size_t>{
        range[0], range[1]}(rng);
}

bool Probability(float value) noexcept
{
    return std::isfinite(value) && value >= 0.F && value <= 1.F;
}

bool ValidDelayRange(std::array<std::size_t, 2> const& range) noexcept
{
    return range[0] > 0U && range[0] <= range[1]
        && range[1] < EpisodeSpawnLayout::NeverActivate / 2U;
}

bool ValidRestrictedZone(RestrictedZoneSpawnSettings const& zone) noexcept
{
    if (zone.id.empty() || zone.platform.empty()
        || !Probability(zone.enabledProbability))
        return false;
    if (zone.positionMode == RestrictedZonePositionMode::Fixed)
        return std::isfinite(zone.fixedPosition[0])
            && std::isfinite(zone.fixedPosition[1]);
    return zone.positionMode == RestrictedZonePositionMode::Random
        && FiniteRange(zone.randomNorth) && FiniteRange(zone.randomEast);
}

void GenerateThreatSchedule(ThreatCurriculumSettings const& settings,
                            EpisodeSpawnLayout& layout, std::mt19937& rng)
{
    if (!settings.enabled)
        return;

    std::fill(layout.redActivationSteps.begin(),
              layout.redActivationSteps.end(),
              EpisodeSpawnLayout::NeverActivate);
    float const countSample = std::uniform_real_distribution<float>{0.F, 1.F}(rng);
    std::size_t const threatCount = countSample < settings.noThreatProbability
        ? 0U
        : countSample < settings.noThreatProbability
                            + settings.singleThreatProbability
            ? 1U : 2U;
    if (threatCount == 0U)
        return;

    std::array<std::size_t, 2> activationOrder{0U, 1U};
    std::shuffle(activationOrder.begin(), activationOrder.end(), rng);
    std::size_t const firstStep = std::bernoulli_distribution{
        static_cast<double>(settings.delayedFirstThreatProbability)}(rng)
        ? SampleStep(settings.firstDelaySteps, rng) : 0U;
    layout.redActivationSteps[activationOrder[0]] = firstStep;

    if (threatCount == 2U)
    {
        std::size_t secondStep = firstStep;
        if (std::bernoulli_distribution{
                static_cast<double>(settings.delayedSecondThreatProbability)}(rng))
            secondStep += SampleStep(settings.secondDelaySteps, rng);
        layout.redActivationSteps[activationOrder[1]] = secondStep;
    }
}

struct Position final
{
    float north{};
    float east{};
};

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

std::optional<std::array<Position, 2>> FallbackRedPair(
    RedSpawnSettings const& settings, float side,
    WorldBounds const& world) noexcept
{
    std::array<Position, 4> const candidates{
        PositionOnExpandedBoundary(settings.bearingRadians.minimum,
            settings.worldScale.minimum, side, world),
        PositionOnExpandedBoundary(settings.bearingRadians.minimum,
            settings.worldScale.maximum, side, world),
        PositionOnExpandedBoundary(settings.bearingRadians.maximum,
            settings.worldScale.minimum, side, world),
        PositionOnExpandedBoundary(settings.bearingRadians.maximum,
            settings.worldScale.maximum, side, world)
    };

    float bestDistance = -std::numeric_limits<float>::infinity();
    std::array<Position, 2> best{};
    for (std::size_t first = 0U; first < candidates.size(); ++first)
        for (std::size_t second = first + 1U;
             second < candidates.size(); ++second)
        {
            float const distance = Separation(
                candidates[first], candidates[second]);
            if (distance > bestDistance)
            {
                bestDistance = distance;
                best = {candidates[first], candidates[second]};
            }
        }
    return bestDistance >= settings.minimumSeparation
        ? std::optional<std::array<Position, 2>>{best}
        : std::nullopt;
}

Spawn MakeRedSpawn(Position const& position, float objectiveEast,
                   std::mt19937& rng, bool destroyed)
{
    float const headingToObjective = std::atan2(
        objectiveEast - position.east, -position.north);
    float const heading = headingToObjective
        + std::uniform_real_distribution<float>{
            -kHeadingJitter, kHeadingJitter}(rng);
    float const speed = kNominalRedSpeed
        + std::uniform_real_distribution<float>{
            -kRedSpeedJitter, kRedSpeedJitter}(rng);
    return {position.north, position.east, heading, speed, destroyed};
}

} // namespace

bool Valid(SpawnSettings const& settings) noexcept
{
    auto const fractionRange = [](SpawnRange const& range) {
        return FiniteRange(range)
            && range.minimum >= -1.F && range.maximum <= 1.F;
    };
    auto const& curriculum = settings.threatCurriculum;
    bool const baseValid = Probability(settings.red2DestroyedProbability)
        && FiniteRange(settings.red.worldScale)
        && settings.red.worldScale.minimum > 1.F
        && FiniteRange(settings.red.bearingRadians)
        && settings.red.bearingRadians.minimum >= -kHalfPi
        && settings.red.bearingRadians.maximum <= kHalfPi
        && std::isfinite(settings.red.minimumSeparation)
        && settings.red.minimumSeparation > 0.F
        && Probability(settings.friendlyRestrictedZone.enabledProbability)
        && fractionRange(settings.friendlyRestrictedZone.northFraction)
        && fractionRange(settings.friendlyRestrictedZone.eastFraction)
        && Probability(curriculum.noThreatProbability)
        && Probability(curriculum.singleThreatProbability)
        && curriculum.noThreatProbability
             + curriculum.singleThreatProbability <= 1.F
        && Probability(curriculum.delayedFirstThreatProbability)
        && Probability(curriculum.delayedSecondThreatProbability)
        && ValidDelayRange(curriculum.firstDelaySteps)
        && ValidDelayRange(curriculum.secondDelaySteps);
    if (!baseValid)
        return false;
    if (settings.blueTeam
        && !defenders::Valid(defenders::ActiveTeamCounts{
            settings.blueTeam->activeCount,
            defenders::TeamCapacity::MinimumRedMobiles}))
    {
        return false;
    }
    if (settings.redTeam)
    {
        auto const& team = *settings.redTeam;
        if (team.activeCount
                > defenders::TeamCapacity::MaximumRedMobiles
            || team.maximumPlacementAttempts == 0U
            || team.maximumPlacementAttempts
                > RedTeamSpawnSettings::MaximumPlacementAttempts
            || !Probability(team.delayedActivationProbability)
            || !ValidDelayRange(team.activationDelaySteps)
            || curriculum.enabled)
            return false;
    }
    auto const& zones = settings.restrictedZones.items;
    for (std::size_t i = 0U; i < zones.size(); ++i)
    {
        if (!ValidRestrictedZone(zones[i]))
            return false;
        for (std::size_t j = i + 1U; j < zones.size(); ++j)
            if (zones[i].id == zones[j].id)
                return false;
    }
    return true;
}

std::optional<EpisodeSpawnLayout> GenerateEpisodeSpawnLayout(
    int side, SpawnSettings const& settings, WorldBounds const& world,
    std::mt19937& rng, std::string* diagnostic)
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
    float const objectiveEast = normalizedSide * 0.92F * world.halfEast;
    if (settings.redTeam)
    {
        auto const& team = *settings.redTeam;
        std::vector<Position> positions;
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

        result.red.reserve(team.activeCount);
        result.redActivationSteps.reserve(team.activeCount);
        for (Position const& position : positions)
        {
            std::size_t const activationStep = std::bernoulli_distribution{
                static_cast<double>(team.delayedActivationProbability)}(rng)
                ? SampleStep(team.activationDelaySteps, rng) : 0U;
            result.red.push_back(MakeRedSpawn(
                position, objectiveEast, rng, activationStep != 0U));
            result.redActivationSteps.push_back(activationStep);
        }
    }
    else
    {
        auto const fallback = FallbackRedPair(
            settings.red, normalizedSide, world);
        if (!fallback)
            return fail("archived 2-vs-2 red pair cannot satisfy minimumSeparation");
        std::optional<std::array<Position, 2>> positions;
        for (unsigned attempt = 0U; attempt < kPairSamplingAttempts; ++attempt)
        {
            std::array<Position, 2> const candidate{
                SampleRedPosition(settings.red, normalizedSide, world, rng),
                SampleRedPosition(settings.red, normalizedSide, world, rng)
            };
            if (Separation(candidate[0], candidate[1])
                >= settings.red.minimumSeparation)
            {
                positions = candidate;
                break;
            }
        }
        if (!positions)
            positions = fallback;

        result.red.reserve(2U);
        result.red.push_back(MakeRedSpawn(
            (*positions)[0], objectiveEast, rng, false));
        result.red.push_back(MakeRedSpawn(
            (*positions)[1], objectiveEast, rng, false));
        result.redActivationSteps.assign(2U, 0U);

        if (settings.threatCurriculum.enabled)
            GenerateThreatSchedule(settings.threatCurriculum, result, rng);
        else if (std::bernoulli_distribution{
                    static_cast<double>(
                        settings.red2DestroyedProbability)}(rng))
            result.redActivationSteps[1] = EpisodeSpawnLayout::NeverActivate;
        for (std::size_t red = 0U; red < result.red.size(); ++red)
            result.red[red].destroyed = result.redActivationSteps[red] != 0U;
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
    if (diagnostic != nullptr)
        diagnostic->clear();
    return result;
}

std::optional<Spawn> MakeSpawn(std::string_view name,
                               EpisodeSpawnLayout const& layout,
                               WorldBounds const& world,
                               std::mt19937& rng)
{
    if (!ValidWorld(world))
        return std::nullopt;

    auto offset = [&](float radius) {
        return std::uniform_real_distribution<float>{-radius, radius}(rng);
    };
    float const side = layout.side < 0 ? -1.F : 1.F;
    float const guardNorth = 0.18F * world.halfNorth;
    float const guardEast = 0.70F * world.halfEast;
    float const guardEastJitter = 0.02F * world.halfEast;

    if (name == "blue_objective")
        return Spawn{0.F, side * (0.92F * world.halfEast), 0.F, 0.F, false};
    if (name == "blue_guard_1")
        return Spawn{-guardNorth + offset(std::min(500.F, 0.025F * world.halfNorth)),
                     side * (guardEast + offset(guardEastJitter)),
                     -side * kHalfPi, 0.F, false};
    if (name == "blue_guard_2")
        return Spawn{guardNorth + offset(std::min(500.F, 0.025F * world.halfNorth)),
                     side * (guardEast + offset(guardEastJitter)),
                     -side * kHalfPi, 0.F, false};
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
