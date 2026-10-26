#include "spawn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ppo::examples::twovstwo {
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

void GenerateThreatSchedule(ThreatCurriculumSettings const& settings,
                            EpisodeSpawnLayout& layout, std::mt19937& rng)
{
    if (!settings.enabled)
        return;

    layout.redActivationSteps.fill(EpisodeSpawnLayout::NeverActivate);
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
    return Probability(settings.red2DestroyedProbability)
        && FiniteRange(settings.red.worldScale)
        && settings.red.worldScale.minimum > 1.F
        && FiniteRange(settings.red.bearingRadians)
        && settings.red.bearingRadians.minimum >= -kHalfPi
        && settings.red.bearingRadians.maximum <= kHalfPi
        && std::isfinite(settings.red.minimumSeparation)
        && settings.red.minimumSeparation > 0.F
        && Probability(settings.restrictedZone.enabledProbability)
        && fractionRange(settings.restrictedZone.northFraction)
        && fractionRange(settings.restrictedZone.eastFraction)
        && Probability(curriculum.noThreatProbability)
        && Probability(curriculum.singleThreatProbability)
        && curriculum.noThreatProbability
             + curriculum.singleThreatProbability <= 1.F
        && Probability(curriculum.delayedFirstThreatProbability)
        && Probability(curriculum.delayedSecondThreatProbability)
        && ValidDelayRange(curriculum.firstDelaySteps)
        && ValidDelayRange(curriculum.secondDelaySteps);
}

std::optional<EpisodeSpawnLayout> GenerateEpisodeSpawnLayout(
    int side, SpawnSettings const& settings, WorldBounds const& world,
    std::mt19937& rng)
{
    if (!Valid(settings) || !ValidWorld(world))
        return std::nullopt;

    float const normalizedSide = side < 0 ? -1.F : 1.F;
    auto const fallback = FallbackRedPair(
        settings.red, normalizedSide, world);
    if (!fallback)
        return std::nullopt;
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

    EpisodeSpawnLayout result{};
    result.side = normalizedSide < 0.F ? -1 : 1;
    float const objectiveEast = normalizedSide * 0.92F * world.halfEast;
    result.red[0] = MakeRedSpawn((*positions)[0], objectiveEast, rng, false);
    result.red[1] = MakeRedSpawn((*positions)[1], objectiveEast, rng, false);

    if (settings.threatCurriculum.enabled)
        GenerateThreatSchedule(settings.threatCurriculum, result, rng);
    else if (std::bernoulli_distribution{
                 static_cast<double>(settings.red2DestroyedProbability)}(rng))
        result.redActivationSteps[1] = EpisodeSpawnLayout::NeverActivate;
    for (std::size_t red = 0U; red < result.red.size(); ++red)
        result.red[red].destroyed = result.redActivationSteps[red] != 0U;

    bool const zoneEnabled = std::bernoulli_distribution{
        static_cast<double>(settings.restrictedZone.enabledProbability)}(rng);
    result.restrictedZone = {
        Sample(settings.restrictedZone.northFraction, rng) * world.halfNorth,
        normalizedSide * Sample(settings.restrictedZone.eastFraction, rng)
            * world.halfEast,
        0.F, 0.F, !zoneEnabled
    };
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
        return layout.restrictedZone;
    if (name == "red_1")
        return layout.red[0];
    if (name == "red_2")
        return layout.red[1];
    return std::nullopt;
}

} // namespace ppo::examples::twovstwo
