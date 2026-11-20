#pragma once

#include "runway_geometry.hpp"

namespace ppo::examples::flight {

/**
 * @brief Horizontal look-ahead used to converge onto the centreline near and past the aim point.
 *
 * Far out, the commanded direction points straight at the aim point. Within
 * this distance of it, the horizontal target slides along the centreline so
 * that a lateral offset is corrected smoothly instead of by a sharp turn.
 */
inline constexpr double CentrelineLookaheadMeters = 300.0;

/** @brief World position of the runway threshold centre on the deck. */
Vec3 ThresholdPoint(Runway const& runway) noexcept;

/**
 * @brief Distance past the threshold where the approach plane meets the runway.
 * @param config Episode configuration: threshold crossing height and glide slope.
 * @return Aim-point distance, so that the plane crosses the threshold at its crossing height.
 */
double AimPointAlongMeters(Config const& config) noexcept;

/** @brief World position of the touchdown aim point on the runway centreline. */
Vec3 AimPoint(Config const& config) noexcept;

/**
 * @brief Height of the approach plane above the deck, for display and diagnostics.
 * @param config Episode configuration.
 * @param along Distance past the threshold, in metres.
 * @return Glide-plane height, zero at and beyond the aim point.
 */
double ApproachPathHeight(Config const& config, double along) noexcept;

/** @brief Target point and commanded unit direction in world axes. */
struct ApproachReference final
{
    Vec3 target{}, direction{};
    double slope{};
};

/**
 * @brief Commanded flight direction shared by guidance, rewards, observations and rendering.
 * @param config Episode configuration and runway pose.
 * @param mission Current phase.
 * @param aircraft Current aircraft state.
 * @return Toward the aim point while far out, along the centreline near and past it, with a
 *     gentle flare sink below the flare height and a level direction while rolling.
 */
ApproachReference ResolveApproachReference(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept;

/**
 * @brief Commanded vertical speed: the current horizontal speed along the commanded flight-path angle.
 * @param guidance Guidance holding the deck-relative velocity and the commanded slope.
 * @return Target vertical speed, in metres per second, negative when descending.
 */
double TargetVerticalSpeed(Guidance const& guidance) noexcept;

/**
 * @brief Resolves the active approach, flare or rollout reference.
 * @param config Episode configuration and runway pose.
 * @param mission Current phase and outcome.
 * @param aircraft Current finite aircraft state.
 * @return Value-only guidance consumed by observations, rewards and rendering.
 */
Guidance Guide(Config const& config, MissionState const& mission, AircraftState const& aircraft) noexcept;

} // namespace ppo::examples::flight
