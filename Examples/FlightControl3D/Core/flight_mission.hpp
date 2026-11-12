#pragma once

#include "flight_config.hpp"

namespace ppo::examples::flight {

struct FlightPlan final
{
    std::array<Vec3, MaximumWaypoints> points{};
    std::size_t count{};
};

Vec3 FromRunway(Runway const& runway, double along, double cross, double up) noexcept;
std::array<double, 2> ToRunway(Runway const& runway, Vec3 const& point) noexcept;
FlightPlan MakeFlightPlan(Config const& config) noexcept;
/** Endpoint of the closest 3D segment, used for initial rally and missed-gate recovery. */
std::size_t ClosestPatrolWaypoint(FlightPlan const& plan,Vec3 position) noexcept;
Guidance Guide(Config const& config, FlightPlan const& plan,
    MissionState const& mission, AircraftState const& aircraft) noexcept;
/** Called after each FDM substep; landing limits also inspect the pre-impact state. */
StepResult AdvanceMission(Config const& config, FlightPlan const& plan, MissionState& mission,
    AircraftState const& before, AircraftState const& after, Runway const* previousRunway = nullptr) noexcept;
Observation Observe(Config const& config, FlightPlan const& plan, MissionState const& mission,
    AircraftState const& aircraft) noexcept;

inline constexpr std::array<char const*, ObservationSize> ObservationNames{
    "target_forward", "target_right", "target_up", "runway_along", "runway_cross", "runway_distance",
    "height", "airspeed", "velocity_north", "velocity_east", "velocity_up",
    "roll_sin", "roll_cos", "pitch_sin", "pitch_cos", "heading_error_sin", "heading_error_cos",
    "roll_rate", "pitch_rate", "yaw_rate", "slope_sin", "slope_cos", "localizer_error", "glide_error",
    "ils_valid", "target_speed", "target_slope", "course_error_sin", "course_error_cos", "path_distance",
    "gear_position", "weight_on_wheels", "patrol", "approach", "rollout", "elapsed",
    "localizer_captured", "glide_captured", "touchdown", "touchdown_distance", "remaining_runway", "stopped_hold",
    "deck_velocity_north", "deck_velocity_east", "deck_velocity_up", "deck_normal_north", "deck_normal_east", "deck_normal_up",
    "deck_angular_north", "deck_angular_east", "deck_angular_up",
    "lap_progress", "completed_laps", "path_qualified", "localizer_hold", "glide_hold"};

} // namespace ppo::examples::flight
