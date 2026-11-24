/**
 * @file flight_spawn.cpp
 * @brief Samplers of the episode starts and the stratified evaluation plan.
 *
 * Every draw uses std::uniform_real_distribution on the caller's std::mt19937. The draws
 * happen in the order written below; changing that order changes every seeded episode.
 *
 * Read next: flight_spawn.hpp (difficulty semantics), Simulator/flight_environment.cpp.
 */
#include "flight_spawn.hpp"
#include "flight_guidance.hpp"
#include <algorithm>
#include <cmath>

namespace ppo::examples::flight {
namespace {
/** @brief Height above the deck of a rolling start before the simulator lowers the wheels, in metres. */
constexpr double InitialWheelClearanceMeters = 2.5;
/** @brief Gear position of every start: down and locked. */
constexpr double GearDown = 1.0;

/** @brief Draws a value uniformly in a range. */
double Draw(Range range, std::mt19937& rng)
{
    return std::uniform_real_distribution<double>(range.minimum, range.maximum)(rng);
}

/** @brief Draws a value uniformly in [-amplitude, amplitude]. */
double DrawSymmetric(double amplitude, std::mt19937& rng)
{
    return Draw({-amplitude, amplitude}, rng);
}

/** @brief Linear interpolation between two values. */
double Interpolate(double from, double to, double fraction) noexcept
{
    return from + (to - from) * fraction;
}

/**
 * @brief Velocity of the given speed along a heading and a flight-path angle.
 * @param speed Speed, in metres per second.
 * @param heading Track azimuth, clockwise from north, in radians.
 * @param slope Flight-path angle, in radians (negative when descending).
 * @return World velocity (north, east, up), in metres per second.
 */
Vec3 VelocityAlong(double speed, double heading, double slope) noexcept
{
    double const horizontal = speed * std::cos(slope);
    return {horizontal * std::cos(heading), horizontal * std::sin(heading), speed * std::sin(slope)};
}
} // namespace

EpisodeStart EvaluationStart(std::size_t episode, std::size_t episodes, double rollingShare,
    double maximumDifficulty) noexcept
{
    if (episodes == 0)
    {
        return {};
    }
    // Rolling starts come first; at least one airborne approach is always evaluated.
    double const share = std::clamp(rollingShare, 0.0, 1.0);
    auto const rounded = static_cast<std::size_t>(std::lround(share * static_cast<double>(episodes)));
    std::size_t const rollingEpisodes = std::min(rounded, episodes - 1);
    if (episode < rollingEpisodes)
    {
        return {true, 0.0};
    }
    // Airborne difficulties are stratified over (0, maximum]: the campaign
    // always reaches the curriculum frontier and rehearses easier cases.
    // Airborne episode k of n uses difficulty maximum * (k + 1) / n.
    std::size_t const airborneEpisodes = episodes - rollingEpisodes;
    std::size_t const airborneIndex = std::min(episode - rollingEpisodes, airborneEpisodes - 1);
    double const fraction = static_cast<double>(airborneIndex + 1) / static_cast<double>(airborneEpisodes);
    return {false, std::clamp(maximumDifficulty, 0.0, 1.0) * fraction};
}

Runway SampleRunway(Config const& config, std::mt19937& rng)
{
    auto const& randomization = config.runwayRandomization;
    Runway runway = config.runway;
    // The three origin draws happen in braced-initializer order: north, east, deck height.
    runway.origin = {Draw(randomization.northMeters, rng), Draw(randomization.eastMeters, rng),
        Draw(randomization.deckHeightMeters, rng)};
    constexpr double FullTurnDegrees = 360.0;
    double const headingDegrees = config.runway.headingDegrees + Draw(randomization.headingOffsetDegrees,
        rng);
    runway.headingDegrees = std::fmod(headingDegrees + FullTurnDegrees, FullTurnDegrees);
    return runway;
}

AircraftState SampleApproachStart(Config const& config, Runway const& runway, ApproachSide side,
    double difficulty, std::mt19937& rng)
{
    auto const& initial = config.initial;
    auto const& mission = config.mission;
    double const clampedDifficulty = std::clamp(difficulty, 0.0, 1.0);
    // Distance range interpolated from the first curriculum range to the full envelope.
    auto const& firstDistance = config.training.curriculum.firstDistanceMeters;
    Range const distanceRange{
        Interpolate(firstDistance.minimum, initial.distanceMeters.minimum, clampedDifficulty),
        Interpolate(firstDistance.maximum, initial.distanceMeters.maximum, clampedDifficulty)};
    double const distanceMeters = Draw(distanceRange, rng);
    // Bearing from the extended centreline, scaled by the difficulty, signed by the side.
    double const sideSign = side == ApproachSide::Left ? -1.0 : 1.0;
    double const bearingRadians = sideSign * clampedDifficulty * Draw(initial.bearingDegrees, rng)
        * DegreesToRadians;
    double const alongMeters = -distanceMeters * std::cos(bearingRadians);
    double const crossMeters = distanceMeters * std::sin(bearingRadians);
    // Height is drawn around the 3-degree plane through the threshold crossing height.
    double const heightMeters = ApproachPathHeight(config, alongMeters)
        + clampedDifficulty * DrawSymmetric(initial.heightOffsetMeters, rng);

    AircraftState aircraft{};
    aircraft.position = FromRunway(runway, alongMeters, crossMeters, heightMeters);
    aircraft.airspeed = mission.approachSpeed + DrawSymmetric(initial.speedToleranceMps, rng);
    double const headingOffsetDegrees = clampedDifficulty * Draw(initial.headingOffsetDegrees, rng);
    aircraft.heading = WrapAngle((runway.headingDegrees + headingOffsetDegrees) * DegreesToRadians);
    aircraft.roll = DrawSymmetric(initial.rollDegrees, rng) * DegreesToRadians;
    aircraft.pitch = Draw(initial.pitchDegrees, rng) * DegreesToRadians;
    aircraft.gearPosition = GearDown;
    double const slopeRadians = (-mission.glideDegrees + DrawSymmetric(initial.slopeToleranceDegrees, rng))
        * DegreesToRadians;
    aircraft.velocity = VelocityAlong(aircraft.airspeed, aircraft.heading, slopeRadians);
    return aircraft;
}

CurriculumPosition ResolveCurriculum(CurriculumSettings const& settings, double difficulty) noexcept
{
    CurriculumPosition position{};
    std::size_t const levels = CurriculumLevels(settings);
    if (levels <= 1)
    {
        return position;
    }
    double const scale = std::clamp(difficulty, 0.0, 1.0) * static_cast<double>(levels - 1);
    double const approachSteps = static_cast<double>(settings.levels - 1);
    if (scale <= approachSteps)
    {
        // A single approach level is the full approach envelope.
        if (approachSteps > 0)
        {
            position.approachDifficulty = scale / approachSteps;
        }
        return position;
    }
    // Stage k covers the scale interval (approachSteps + k, approachSteps + k + 1].
    double const stageScale = std::ceil(scale - approachSteps) - 1.0;
    position.staged = true;
    position.stage = std::min(static_cast<std::size_t>(std::max(stageScale, 0.0)), settings.stages.size() - 1);
    return position;
}

namespace {
/** @brief Level start of a pattern or holding stage at the stage height. */
AircraftState LevelStart(Config const& config, Runway const& runway, Gate const& start, double headingDegrees,
    std::mt19937& rng)
{
    auto const& initial = config.initial;
    AircraftState aircraft{};
    aircraft.position = FromRunway(runway, start.alongMeters, start.crossMeters, start.heightMeters);
    aircraft.airspeed = config.mission.approachSpeed + DrawSymmetric(initial.speedToleranceMps, rng);
    aircraft.heading = WrapAngle(headingDegrees * DegreesToRadians);
    aircraft.roll = DrawSymmetric(initial.rollDegrees, rng) * DegreesToRadians;
    aircraft.pitch = Draw(initial.pitchDegrees, rng) * DegreesToRadians;
    aircraft.gearPosition = GearDown;
    aircraft.velocity = VelocityAlong(aircraft.airspeed, aircraft.heading, 0.0);
    return aircraft;
}

/** @brief Direct approach from the stage envelope, descending along the glide slope. */
StageStart SampleApproachStage(Config const& config, Runway const& runway, StageSettings const& stage,
    double sideSign, std::mt19937& rng)
{
    auto const& initial = config.initial;
    double const distanceMeters = Draw(stage.distanceMeters, rng);
    double const bearingRadians = sideSign * Draw(stage.bearingDegrees, rng) * DegreesToRadians;
    // A positive heading offset turns toward the extended centreline: the start lies on the
    // side of sideSign, so the centreline is reached by turning against that sign.
    double const headingOffsetDegrees = -sideSign * Draw(stage.headingOffsetDegrees, rng);
    double const alongMeters = -distanceMeters * std::cos(bearingRadians);
    double const crossMeters = distanceMeters * std::sin(bearingRadians);
    double const heightMeters = ApproachPathHeight(config, alongMeters) + Draw(stage.heightMeters, rng);
    StageStart start{};
    auto& aircraft = start.aircraft;
    aircraft.position = FromRunway(runway, alongMeters, crossMeters, heightMeters);
    aircraft.airspeed = config.mission.approachSpeed + DrawSymmetric(initial.speedToleranceMps, rng);
    aircraft.heading = WrapAngle((runway.headingDegrees + headingOffsetDegrees) * DegreesToRadians);
    aircraft.roll = DrawSymmetric(initial.rollDegrees, rng) * DegreesToRadians;
    aircraft.pitch = Draw(initial.pitchDegrees, rng) * DegreesToRadians;
    aircraft.gearPosition = GearDown;
    double const slopeRadians = (-config.mission.glideDegrees + DrawSymmetric(initial.slopeToleranceDegrees, rng))
        * DegreesToRadians;
    aircraft.velocity = VelocityAlong(aircraft.airspeed, aircraft.heading, slopeRadians);
    return start;
}

/** @brief Downwind start abeam the runway, then downwind end, base and final gates. */
StageStart SamplePatternStage(Config const& config, Runway const& runway, StageSettings const& stage,
    double sideSign, std::mt19937& rng)
{
    double const alongMeters = Draw(stage.startAlongMeters, rng);
    double const downwindOffset = Draw(stage.lateralMeters, rng);
    double const headingOffsetDegrees = Draw(stage.headingOffsetDegrees, rng);
    double const heightMeters = Draw(stage.heightMeters, rng);
    double const crossMeters = sideSign * downwindOffset;
    constexpr double HalfTurnDegrees = 180.0;
    StageStart start{};
    start.route.start = {alongMeters, crossMeters, heightMeters};
    start.route.gates[0] = {PatternDownwindEndAlongMeters, crossMeters, heightMeters};
    start.route.gates[1] = {PatternBaseAlongMeters, crossMeters * PatternBaseOffsetFraction, heightMeters};
    start.route.gates[2] = {PatternFinalAlongMeters, 0.0, heightMeters};
    start.route.count = 3;
    // Downwind is flown opposite the runway heading.
    start.aircraft = LevelStart(config, runway, start.route.start,
        runway.headingDegrees + HalfTurnDegrees + headingOffsetDegrees, rng);
    return start;
}

/**
 * @brief Append a semicircle between the parallel holding legs.
 * @param route Route whose last point is the tangent entry of the turn.
 * @param exit Tangent exit at the same along coordinate and opposite lateral edge.
 * @param alongSign +1 for the turn beyond the fix, -1 for the far turn.
 *
 * The radius is half the separation of the legs. Intermediate gates sample
 * the circle every 45 degrees; the endpoint is kept exact.
 */
void AppendHoldingTurn(Route& route, Gate const& exit, double alongSign) noexcept
{
    Gate const entry = route.gates[route.count - 1];
    double const halfCross = (exit.crossMeters - entry.crossMeters) / 2.0;
    double const radius = std::abs(halfCross);
    for (std::size_t segment = 1; segment < HoldingTurnSegments; ++segment)
    {
        double const angle = Pi * static_cast<double>(segment) / static_cast<double>(HoldingTurnSegments);
        route.gates[route.count++] = {entry.alongMeters + alongSign * radius * std::sin(angle),
            entry.crossMeters + halfCross * (1.0 - std::cos(angle)), entry.heightMeters};
    }
    route.gates[route.count++] = exit;
}

/** @brief Inbound start toward the fix, one racetrack lap on the given side, back over the fix. */
StageStart SampleHoldingStage(Config const& config, Runway const& runway, StageSettings const& stage,
    double sideSign, std::mt19937& rng)
{
    double const alongMeters = Draw(stage.startAlongMeters, rng);
    double const crossMeters = sideSign * Draw(stage.lateralMeters, rng);
    double const headingOffsetDegrees = Draw(stage.headingOffsetDegrees, rng);
    double const heightMeters = Draw(stage.heightMeters, rng);
    double const fix = stage.fixAlongMeters;
    double const outboundEnd = fix - stage.outboundMeters;
    double const width = sideSign * stage.holdingWidthMeters;
    StageStart start{};
    start.route.start = {alongMeters, crossMeters, heightMeters};
    // The two straight legs are joined by semicircles tangent at their endpoints.
    start.route.holding = true;
    start.route.gates[start.route.count++] = {fix, 0.0, heightMeters};
    AppendHoldingTurn(start.route, {fix, width, heightMeters}, 1.0);
    start.route.gates[start.route.count++] = {outboundEnd, width, heightMeters};
    AppendHoldingTurn(start.route, {outboundEnd, 0.0, heightMeters}, -1.0);
    start.route.gates[start.route.count++] = {fix, 0.0, heightMeters};
    start.aircraft = LevelStart(config, runway, start.route.start, runway.headingDegrees + headingOffsetDegrees,
        rng);
    return start;
}
} // namespace

StageStart SampleStageStart(Config const& config, Runway const& runway, StageSettings const& stage,
    ApproachSide side, std::mt19937& rng)
{
    double const sideSign = side == ApproachSide::Left ? -1.0 : 1.0;
    if (stage.kind == StageKind::Approach)
    {
        return SampleApproachStage(config, runway, stage, sideSign, rng);
    }
    if (stage.kind == StageKind::Pattern)
    {
        return SamplePatternStage(config, runway, stage, sideSign, rng);
    }
    return SampleHoldingStage(config, runway, stage, sideSign, rng);
}

AircraftState SampleRolloutStart(Config const& config, Runway const& runway, std::mt19937& rng)
{
    // Start just above the deck; the simulator lowers the wheels onto it.
    auto const& rollout = config.initial.rolloutStart;
    double const alongMeters = Draw(rollout.alongMeters, rng);
    double const crossMeters = Draw(rollout.crossMeters, rng);
    AircraftState aircraft{};
    aircraft.position = FromRunway(runway, alongMeters, crossMeters, InitialWheelClearanceMeters);
    aircraft.airspeed = Draw(rollout.speedMps, rng);
    aircraft.heading = WrapAngle(runway.headingDegrees * DegreesToRadians);
    aircraft.gearPosition = GearDown;
    aircraft.velocity = Scale(RunwayForward(runway), aircraft.airspeed);
    return aircraft;
}
} // namespace ppo::examples::flight
