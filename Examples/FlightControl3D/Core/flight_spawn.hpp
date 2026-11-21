/**
 * @file flight_spawn.hpp
 * @brief Episode starts: runway pose, airborne approaches, rolling starts and evaluation plans.
 *
 * Role in the chain: first step of every episode, before JSBSim is reset. The samplers
 * draw from a std::mt19937 owned by the environment; the number and order of the draws are
 * part of the reproducibility contract (seeded validation episodes, golden trace).
 *
 * Difficulty d in [0, 1] interpolates the airborne envelope: the start distance range moves
 * linearly from curriculum.firstDistanceMeters (d = 0) to initial.distanceMeters (d = 1),
 * while the bearing, heading and height offsets are scaled by d. The side of the extended
 * centreline is chosen by the caller.
 *
 * Not here: the training difficulty draw (sqrt density, FlightEnvironment::Reset) and the
 * curriculum promotion (Training/flight_curriculum.cpp).
 *
 * Read next: flight_spawn.cpp, Simulator/flight_environment.cpp.
 */
#pragma once

#include "runway_geometry.hpp"
#include <random>

namespace ppo::examples::flight {

/** @brief Side of the extended centreline on which an airborne start is placed. */
enum class ApproachSide : std::uint8_t { Left, Right };

/** @brief Kind and difficulty of an independent episode start. */
struct EpisodeStart final
{
    /** @brief True for an already-landed rolling start, false for an airborne approach. */
    bool rolling{};
    /** @brief Airborne start difficulty in [0, 1]; a rolling start ignores it. */
    double difficulty{1.0};
    /** @brief Airborne start with the gear retracted: the policy must command its extension. */
    bool gearUp{};
};

/**
 * @brief Start of one episode of a stratified evaluation campaign.
 *
 * The campaign mirrors the training mix at a given curriculum difficulty: the
 * rolling share of the episodes (rounded, at least one airborne episode left)
 * starts on the runway, then the airborne episodes use evenly spaced
 * difficulties up to the maximum, from a short aligned final to the most
 * distant and offset starts of that level.
 * @param episode Zero-based episode index within the campaign.
 * @param episodes Number of episodes of the campaign, at least one.
 * @param rollingShare Share of rolling starts, in [0, 1).
 * @param maximumDifficulty Hardest airborne difficulty of the campaign, in [0, 1].
 * @return Start of the requested episode; the last airborne episode uses the maximum difficulty.
 */
EpisodeStart EvaluationStart(std::size_t episode, std::size_t episodes, double rollingShare,
    double maximumDifficulty) noexcept;

/**
 * @brief Draws the episode runway pose from the configured randomization.
 *
 * Draw order: north, east, deck height, heading offset (four uniform draws).
 * @param config Validated configuration.
 * @param rng Episode random generator.
 * @return Level runway with a random origin, deck height and heading in [0, 360) degrees.
 */
Runway SampleRunway(Config const& config, std::mt19937& rng);

/**
 * @brief Samples an airborne start around the glide plane.
 *
 * Difficulty 0 is a short, aligned final in the first curriculum distance
 * range; difficulty 1 is the full configured cone. In between, the distance
 * range is interpolated and the bearing, heading and height offsets are scaled.
 * @param config Validated configuration.
 * @param runway Episode runway pose.
 * @param side Side of the centreline used for the bearing.
 * @param difficulty Start difficulty in [0, 1].
 * @param rng Episode random generator.
 * @return Aircraft state around the approach plane, heading roughly toward the runway.
 */
AircraftState SampleApproachStart(Config const& config, Runway const& runway, ApproachSide side,
    double difficulty, std::mt19937& rng);

/** @brief Position of a difficulty on the curriculum scale: an approach level or a stage. */
struct CurriculumPosition final
{
    /** @brief True for a stage start, false for an approach start. */
    bool staged{};
    /** @brief Approach difficulty in [0, 1], used when staged is false. */
    double approachDifficulty{1.0};
    /** @brief Stage index, used when staged is true. */
    std::size_t stage{};
};

/**
 * @brief Maps a curriculum difficulty to an approach difficulty or a stage.
 *
 * The scale has CurriculumLevels() - 1 unit steps: the first levels - 1 steps interpolate
 * the approach difficulty from 0 to 1, then each further step (k, k + 1] is stage k.
 * Difficulty 1 is therefore the last stage, or the full approach without stages.
 * @param settings Curriculum settings.
 * @param difficulty Curriculum difficulty in [0, 1].
 * @return Approach difficulty or stage index.
 */
CurriculumPosition ResolveCurriculum(CurriculumSettings const& settings, double difficulty) noexcept;

/** @brief Airborne start of a curriculum stage: the aircraft and the route of gates to fly first. */
struct StageStart final
{
    AircraftState aircraft{};
    Route route{};
};

/**
 * @brief Samples the airborne start of a curriculum stage.
 *
 * - approach: direct approach from the stage envelope (distance, bearing signed by the side,
 *   heading offset, height above the glide plane), no route;
 * - pattern: level flight downwind, abeam the runway on the given side, heading opposite the
 *   runway; gates at the downwind end, on base and on the extended centreline 7 km out;
 * - holding: level flight inbound toward a fix on the extended centreline; gates around one
 *   racetrack lap on the given side, back over the fix, then the approach.
 *
 * Draw order: two position draws (distance and bearing, start along and lateral offset), the
 * heading offset, the height, then the speed, roll and pitch, and for an approach the slope.
 * @param config Validated configuration.
 * @param runway Episode runway pose.
 * @param stage Stage settings.
 * @param side Side of the centreline of the start and of the route.
 * @param rng Episode random generator.
 * @return Initial aircraft state (gear down) and the route of the episode.
 */
StageStart SampleStageStart(Config const& config, Runway const& runway, StageSettings const& stage,
    ApproachSide side, std::mt19937& rng);

/**
 * @brief Samples an already-landed rolling start; the simulator settles its wheels on the deck.
 * @param config Validated configuration.
 * @param runway Episode runway pose.
 * @param rng Episode random generator.
 * @return Aligned rolling state slightly above the runway surface.
 */
AircraftState SampleRolloutStart(Config const& config, Runway const& runway, std::mt19937& rng);

} // namespace ppo::examples::flight
