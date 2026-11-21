/**
 * @file flight_observation.hpp
 * @brief Layout and normalization of the 35-slot policy observation.
 *
 * Role in the chain: mission + guidance -> **observation** -> PPO actor and critic.
 * The slot order, count, scales and bounds are part of the checkpoint contract: a trained
 * network reads slot i as the quantity named ObservationNames[i]. Never reorder slots or
 * change a scale without a new contract version (flight_config.cpp, ContractVersion).
 *
 * Frames: positions and velocities are expressed in runway axes, attitudes relative to the
 * runway heading, and direction vectors in body axes, so the observation does not depend on
 * the randomized runway pose. Every slot lies in [-1, 1].
 *
 * Not here: diagnostics of the observation statistics (Runtime/flight_session.cpp).
 *
 * Read next: flight_observation.cpp (scales and resolutions), flight_guidance.hpp.
 */
#pragma once

#include "flight_config.hpp"

namespace ppo::examples::flight {

/**
 * @brief Named slots of the policy observation.
 *
 * Positions and velocities are expressed in runway axes, so the inputs do not
 * depend on the randomized runway heading. Every slot is bounded to [-1, 1].
 */
enum class ObservationIndex : std::size_t
{
    Along,
    Cross,
    Height,
    GlidePathError,
    RunwayRemaining,
    VelocityForward,
    VelocityRight,
    VelocityUp,
    AirspeedError,
    VerticalSpeedError,
    RollSin,
    RollCos,
    PitchSin,
    HeadingSin,
    HeadingCos,
    RollRate,
    PitchRate,
    YawRate,
    TargetDirectionForward,
    TargetDirectionRight,
    TargetDirectionUp,
    VelocityDirectionForward,
    VelocityDirectionRight,
    VelocityDirectionUp,
    WeightOnWheels,
    Rollout,
    StopProgress,
    Elapsed,
    GearPosition,
    GearCommanded,
    Elevator,
    Aileron,
    Rudder,
    Throttle,
    Brake,
    Count
};
static_assert(static_cast<std::size_t>(ObservationIndex::Count) == ObservationSize,
    "ObservationSize must match the named observation slots");
static_assert(static_cast<std::size_t>(ObservationIndex::Rollout) == RolloutObservation,
    "RolloutObservation must name the rollout flag");
static_assert(static_cast<std::size_t>(ObservationIndex::GearCommanded) == GearCommandObservation,
    "GearCommandObservation must name the gear command flag");

/** @brief Index of a named observation slot. */
inline constexpr std::size_t Slot(ObservationIndex index) noexcept
{
    return static_cast<std::size_t>(index);
}

/** @brief Log and diagnostic labels of the observation slots, in ObservationIndex order. */
inline constexpr std::array<char const*, ObservationSize> ObservationNames{
    "along", "cross", "height", "glide_path_error", "runway_remaining",
    "velocity_forward", "velocity_right", "velocity_up", "airspeed_error", "vertical_speed_error",
    "roll_sin", "roll_cos", "pitch_sin", "heading_sin", "heading_cos",
    "roll_rate", "pitch_rate", "yaw_rate",
    "target_direction_body_forward", "target_direction_body_right", "target_direction_body_up",
    "velocity_direction_body_forward", "velocity_direction_body_right", "velocity_direction_body_up",
    "weight_on_wheels", "rollout", "stop_progress", "elapsed", "gear_position", "gear_commanded",
    "filtered_elevator", "filtered_aileron", "filtered_rudder", "filtered_throttle", "filtered_brake"};

/**
 * @brief Signed logarithmic compression of a distance into [-1, 1].
 *
 * f(x) = sign(x) * log(1 + |x| / resolution) / log(1 + maximum / resolution), clamped.
 * @param value Signed distance, in metres.
 * @param resolution Scale below which the response is nearly linear, in metres.
 * @param maximum Distance mapped to one; larger magnitudes saturate.
 * @return Bounded feature with fine resolution near zero and coarse resolution far away;
 *     zero for a non-finite value.
 */
float SignedLog(double value, double resolution, double maximum) noexcept;

/**
 * @brief Builds the normalized policy observation.
 * @param config Episode configuration and runway pose.
 * @param mission Current mission state; a Timeout is observed as the flying state it truncates.
 * @param aircraft Current aircraft state.
 * @return Bounded features, or zeros for a non-finite aircraft state.
 */
Observation Observe(Config const& config, MissionState const& mission,
    AircraftState const& aircraft) noexcept;

} // namespace ppo::examples::flight
