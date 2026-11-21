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

/**
 * @brief Samples an already-landed rolling start; the simulator settles its wheels on the deck.
 * @param config Validated configuration.
 * @param runway Episode runway pose.
 * @param rng Episode random generator.
 * @return Aligned rolling state slightly above the runway surface.
 */
AircraftState SampleRolloutStart(Config const& config, Runway const& runway, std::mt19937& rng);

} // namespace ppo::examples::flight
