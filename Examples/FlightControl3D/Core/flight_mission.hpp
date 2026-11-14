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
/**
 * @brief Resolves the active racetrack, LOC, glide, flare or rollout reference.
 * @param config Validated episode configuration and instantaneous runway pose.
 * @param plan Closed racetrack bound to the current runway pose.
 * @param mission Current phase, capture state and ordered waypoint progress.
 * @param aircraft Current finite aircraft state.
 * @param runway Optional previous runway snapshot for time-consistent reward evaluation.
 * @return Value-only guidance consumed by observations, rewards and rendering.
 */
Guidance Guide(Config const& config, FlightPlan const& plan,
    MissionState const& mission, AircraftState const& aircraft, Runway const* runway = nullptr) noexcept;
/** Which channels are potential-based and therefore discounted within one PPO action. */
inline constexpr bool IsPotentialReward(Reward channel) noexcept
{
    return channel == Reward::Localizer || channel == Reward::Glide
        || channel == Reward::Speed || channel == Reward::Slope;
}
/**
 * Weighted state potentials. Physical rewards use gamma^(1/actionSubsteps) per
 * JSBSim step and are compounded back into one gamma-discounted PPO action.
 * Capture/crash/stopped are absorbing (zero potential); Timeout is truncated.
 */
std::array<double, RewardCount> ShapingPotentials(Config const&,FlightPlan const&,
    MissionState const&,AircraftState const&,Runway const* runway=nullptr) noexcept;
/** Independent failed constraints for one LOC/GLIDE capture sample. */
enum class CaptureGateFailure : std::uint16_t
{
    State=1u<<0,Localizer=1u<<1,Heading=1u<<2,Route=1u<<3,
    Speed=1u<<4,Roll=1u<<5,Rates=1u<<6,Glide=1u<<7
};
inline constexpr std::array<char const*,8> CaptureGateFailureNames{
    "state","localizer","heading","route","speed","roll","rates","glide"};
inline constexpr std::uint16_t GateBit(CaptureGateFailure failure) noexcept
{ return static_cast<std::uint16_t>(failure); }
/** Diagnoses all simultaneous failed constraints instead of only the first. */
std::uint16_t CaptureGateFailureMask(Config const&,CurriculumStage,MissionState const&,
    AircraftState const&,Guidance const&) noexcept;
/** Confirm actual beam geometry and stable physical flight before a lesson can end. */
bool CaptureGateQualified(Config const&,CurriculumStage,MissionState const&,
    AircraftState const&,Guidance const&) noexcept;
/** Called after each FDM substep; landing limits also inspect the pre-impact state. */
StepResult AdvanceMission(Config const& config, FlightPlan const& plan, MissionState& mission,
    AircraftState const& before, AircraftState const& after, Runway const* previousRunway = nullptr) noexcept;
Observation Observe(Config const& config, FlightPlan const& plan, MissionState const& mission,
    AircraftState const& aircraft) noexcept;

inline constexpr std::array<char const*, ObservationSize> ObservationNames{
    "target_forward", "target_right", "target_up", "runway_along", "runway_cross", "runway_distance",
    "height", "airspeed", "velocity_north", "velocity_east", "velocity_up",
    "roll_sin", "roll_cos", "pitch_sin", "pitch_cos", "heading_error_sin", "heading_error_cos",
    "roll_rate", "pitch_rate", "yaw_rate", "slope_sin", "slope_cos", "localizer_sin", "glide_error",
    "ils_valid", "target_speed", "target_slope", "course_error_sin", "course_error_cos", "path_distance",
    "gear_position", "weight_on_wheels", "patrol", "approach", "rollout", "elapsed",
    "localizer_captured", "glide_captured", "touchdown", "touchdown_distance", "remaining_runway", "completion_hold",
    "deck_velocity_north", "deck_velocity_east", "deck_velocity_up", "deck_normal_north", "deck_normal_east", "deck_normal_up",
    "deck_angular_north", "deck_angular_east", "deck_angular_up",
    "lap_progress", "completed_laps", "path_qualified", "localizer_hold", "glide_hold"};

} // namespace ppo::examples::flight
