/**
 * @file flight_types.hpp
 * @brief Plain value types shared by every stage of the FlightControl3D chain.
 *
 * Role in the chain JSBSim -> mission -> reward -> observation -> PPO -> session -> render:
 * these structs carry the data exchanged between the stages (aircraft state, controls,
 * mission state, guidance, step results). They hold no behaviour beyond trivial helpers.
 *
 * Frames and units: SI units everywhere (metres, metres per second, radians, seconds).
 * World vectors use a local tangent frame (north, east, up) centred on the configured
 * runway reference point; runway-relative vectors reuse Vec3 with north = along the
 * runway, east = right of the centreline and up = above the deck.
 *
 * Not here: JSBSim access (Simulator/), reward formulas (flight_reward.cpp), the
 * observation layout (flight_observation.hpp) and configuration (flight_config.hpp).
 *
 * Read next: flight_config.hpp, runway_geometry.hpp, flight_mission.hpp.
 */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ppo::examples::flight {

/** @brief The mathematical constant pi, to double precision. */
inline constexpr double Pi = 3.14159265358979323846;
/** @brief Exact international foot, in metres (JSBSim works in feet). */
inline constexpr double FeetToMeters = 0.3048;
/** @brief Multiply degrees by this factor to obtain radians. */
inline constexpr double DegreesToRadians = Pi / 180.0;
/** @brief Continuous actor outputs: elevator, aileron, rudder, throttle and brake. */
inline constexpr std::size_t ContinuousActions = 5;
/** @brief Number of named observation slots, see ObservationIndex. */
inline constexpr std::size_t ObservationSize = 35;
/** @brief Observation slot that flags the ground-roll phase and enables the brake action. */
inline constexpr std::size_t RolloutObservation = 25;
/** @brief Observation slot that flags a latched gear extension and disables the gear action. */
inline constexpr std::size_t GearCommandObservation = 29;
/** @brief Every normalized observation lies within [-ObservationBound, ObservationBound]. */
inline constexpr float ObservationBound = 1.F;

/**
 * @brief Three-component vector in (north, east, up) order.
 *
 * In the world frame the components are local-tangent north, east and up, in metres or
 * metres per second. In runway axes the same fields mean along, right and up.
 */
struct Vec3 final
{
    double north{};
    double east{};
    double up{};
};

/** @brief Native JSBSim wheel sample, written by JsbsimAircraft::Read. */
struct WheelContact final
{
    /** @brief Wheel contact point in the world frame, in metres. */
    Vec3 point{};
    /** @brief JSBSim weight-on-wheel flag of this gear unit. */
    bool onGround{};
};

/**
 * @brief SI snapshot of the aircraft, in the local tangent frame at the runway reference point.
 *
 * Written by JsbsimAircraft (Reset, Advance) and by the spawn samplers (requested initial
 * state); read by the mission, rewards, guidance, observations and renderer.
 */
struct AircraftState final
{
    /** @brief Centre-of-gravity position (north, east, up), in metres. */
    Vec3 position{};
    /** @brief Inertial velocity (north, east, up), in metres per second. */
    Vec3 velocity{};
    /** @brief Calibrated airspeed, in metres per second. */
    double airspeed{};
    /** @brief Bank angle, positive right wing down, in radians. */
    double roll{};
    /** @brief Nose-up pitch angle, in radians. */
    double pitch{};
    /** @brief True heading, clockwise from north, in radians. */
    double heading{};
    /** @brief Body roll rate p, in radians per second. */
    double rollRate{};
    /** @brief Body pitch rate q, in radians per second. */
    double pitchRate{};
    /** @brief Body yaw rate r, in radians per second. */
    double yawRate{};
    /** @brief Normalized landing-gear extension, in [0, 1] (1 = down and locked). */
    double gearPosition{};
    /** @brief Normalized flap position, in [0, 1] (1 = 30 degrees on the T-6). */
    double flapPosition{};
    /** @brief True when any wheel reports weight on wheels. */
    bool weightOnWheels{};
    /** @brief True when a non-wheel (structural) contact point touches the ground. */
    bool structuralContact{};
    /** @brief False when JSBSim failed to advance or produced a non-finite value. */
    bool finite{true};
    /** @brief Native wheel samples (nose, left and right main gear). */
    std::array<WheelContact, 3> wheels{};
    /** @brief True when all three wheel samples were read from JSBSim. */
    bool wheelDataValid{};
};

/**
 * @brief Physical control inputs.
 *
 * Written by ToControls (policy mapping) and the environment command filter; read by
 * JsbsimAircraft::Advance. Elevator, aileron and rudder lie in [-1, 1]; throttle, brake,
 * flap and gear in [0, 1].
 */
struct Controls final
{
    /** @brief Elevator command, in [-1, 1], with the JSBSim fcs/elevator-cmd-norm sign convention. */
    double elevator{};
    /** @brief Aileron command, in [-1, 1]. */
    double aileron{};
    /** @brief Rudder pedal command, in [-1, 1]; also steers the nose wheel on the ground. */
    double rudder{};
    /** @brief Throttle lever, in [0, 1]. */
    double throttle{};
    /** @brief Symmetric wheel brake, in [0, 1]. */
    double brake{};
    /** @brief Flap command, in [0, 1], selected by the mission phase. */
    double flap{};
    /**
     * @brief Gear lever, 0 (up) or 1 (down).
     *
     * From the policy it is an extension request; the environment latches the first
     * request, so the gear is never retracted during an episode.
     */
    double gear{};
};

/** @brief Mission phase: airborne approach until the first safe contact, then ground roll. */
enum class Phase : std::uint8_t { Approach, Rollout };
/** @brief Episode outcome; Stopped and Crash terminate, Timeout truncates. */
enum class Outcome : std::uint8_t { Flying, Stopped, Crash, Timeout };

/** @brief Independent conditions that can make a flight end in a crash. */
enum class CrashReason : std::size_t
{
    InvalidState,
    Distance,
    Altitude,
    LandingArea,
    WheelsOutside,
    Gear,
    StructuralContact,
    Sink,
    Speed,
    Roll,
    PitchLow,
    PitchHigh,
    Heading,
    NoWheelContact,
    RouteIncomplete,
    Count
};
/** @brief Number of CrashReason values. */
inline constexpr std::size_t CrashReasonCount = static_cast<std::size_t>(CrashReason::Count);

/** @brief One failed condition, with the measured value and its configured limit when applicable. */
struct CrashCondition final
{
    /** @brief True when this condition caused or accompanied the crash. */
    bool failed{};
    /** @brief Measured value; angles are reported in degrees for display. */
    double value{};
    /** @brief Configured limit, in the same unit as value. */
    double limit{};
};

/** @brief Diagnostic captured at the terminal physics substep, independent of episode resets. */
struct CrashDiagnostic final
{
    /** @brief One entry per CrashReason, indexed by its value. */
    std::array<CrashCondition, CrashReasonCount> conditions{};
};

/** @brief Active navigation reference exposed to the policy snapshot and renderer. */
enum class GuidanceMode : std::uint8_t { Approach, Flare, Rollout };

/**
 * @brief Value-only guidance computed by Guide() from the mission and the aircraft state.
 *
 * Read by the rewards, the observation and the renderer; never stored by the simulator.
 */
struct Guidance final
{
    /** @brief Distance past the runway threshold, in metres (negative before it). */
    double along{};
    /** @brief Lateral offset right of the centreline, in metres. */
    double cross{};
    /** @brief Horizontal distance from the threshold centre, in metres. */
    double distance{};
    /** @brief Height above the deck, in metres. */
    double height{};
    /** @brief Reference calibrated airspeed, in metres per second (zero while rolling on wheels). */
    double targetSpeed{};
    /** @brief Commanded flight-path angle, in radians (negative when descending). */
    double targetSlope{};
    /** @brief Absolute commanded track azimuth, in radians, for diagnostics and the renderer. */
    double targetHeading{};
    /** @brief Geometric target point on the centreline, in world metres. */
    Vec3 target{};
    /** @brief Unit commanded direction in the world frame. */
    Vec3 targetDirection{};
    /** @brief Unit deck-relative velocity direction in the world frame (zero when not moving). */
    Vec3 velocityDirection{};
    /** @brief Aircraft velocity relative to the deck, expressed in runway axes, in metres per second. */
    Vec3 relativeVelocity{};
    /** @brief Active reference mode, for display only. */
    GuidanceMode mode{GuidanceMode::Approach};
    /** @brief True only when the geometric target point is an active control reference. */
    bool targetValid{};
};

/**
 * @brief Reward channels. Path, VerticalSpeed and Speed are potential-based
 * shaping terms; Direction is a running cost; the others are one-shot events.
 *
 * The order and names are part of the JSON schema, the checkpoint signature and the
 * logs parsed by tools/flight/curriculum_ci.py: never reorder or rename them.
 */
enum class Reward : std::size_t
{
    Path,
    Direction,
    VerticalSpeed,
    Speed,
    Touchdown,
    Stop,
    Crash,
    Count
};
/** @brief Number of reward channels. */
inline constexpr std::size_t RewardCount = static_cast<std::size_t>(Reward::Count);
/** @brief JSON keys and log labels of the reward channels, in Reward order. */
inline constexpr std::array<char const*, RewardCount> RewardNames{
    "Path", "Direction", "VerticalSpeed", "Speed", "Touchdown", "Stop", "Crash"};

/** @brief Largest number of gates of one episode route (a holding lap and its joining legs). */
inline constexpr std::size_t MaximumGates = 11;
/** @brief Number of equal angular intervals in each semicircular holding turn. */
inline constexpr std::size_t HoldingTurnSegments = 4;
/** @brief Inbound, two semicircles, outbound and final inbound. */
inline constexpr std::size_t HoldingRouteGateCount = 2 * HoldingTurnSegments + 3;
static_assert(HoldingRouteGateCount <= MaximumGates, "Holding route exceeds gate capacity");

/** @brief Point of a route, in runway axes: along past the threshold, cross right of the centreline, up. */
struct Gate final
{
    double alongMeters{};
    double crossMeters{};
    double heightMeters{};
};

/**
 * @brief Fixed-capacity sequence of gates flown before the final approach.
 *
 * Empty for a direct approach. The start point is kept so that the first leg has a
 * direction, like every later leg between two gates.
 */
struct Route final
{
    std::array<Gate, MaximumGates> gates{};
    /** @brief Number of gates in use, at most MaximumGates. */
    std::size_t count{};
    /** @brief Horizontal start of the route, in runway axes (heightMeters unused). */
    Gate start{};
    /** @brief True for the two tangent semicircles and straight legs of a holding lap. */
    bool holding{};
};

/**
 * @brief Mutable episode state, written only by AdvanceMission and the environment resets.
 *
 * Read by the rewards, the observation, the session statistics and the renderer.
 */
struct MissionState final
{
    /** @brief Current phase. */
    Phase phase{Phase::Approach};
    /** @brief Current outcome; Flying until an episode boundary. */
    Outcome outcome{Outcome::Flying};
    /** @brief Physics substeps advanced in this episode. */
    std::uint64_t steps{};
    /** @brief Consecutive time spent stopped on all wheels, in seconds. */
    double stoppedSeconds{};
    /** @brief Last filtered physical controls, included in the policy observation. */
    Controls filteredControls{};
    /** @brief True once the gear extension is commanded; it stays latched until the episode ends. */
    bool gearDownCommanded{true};
    /** @brief Gates to fly before the final approach; empty for a direct approach. */
    Route route{};
    /** @brief Index of the next gate; equal to route.count once the route is complete. */
    std::size_t nextGate{};
    /** @brief True after the first safe touchdown (or for a rolling start). */
    bool touchedDown{};
    /** @brief True once the first ground contact was recorded. */
    bool contactAttempted{};
    /** @brief True when the first contact involved a wheel. */
    bool firstWheelContact{};
    /** @brief Lateral offset of the first contact point, in metres. */
    double touchdownCrossMeters{};
    /** @brief Worst sink rate around the first contact, in metres per second (positive down). */
    double touchdownSinkMps{};
    /** @brief Ground speed at the first contact, in metres per second. */
    double touchdownSpeedMps{};
    /** @brief Gear extension at the first contact, in [0, 1]. */
    double touchdownGearPosition{};
    /** @brief Touchdown factor from location, sink rate and speed, applied to the Touchdown reward. */
    double touchdownQuality{1.0};
    /** @brief Landing-surface factor of the touchdown point, applied to the Stop reward. */
    double landingAreaFactor{1.0};
    /** @brief Distance of the first contact past the threshold, in metres. */
    double touchdownFromThreshold{};
    /** @brief Distance of the stop point past the threshold, in metres. */
    double stopFromThreshold{};
    /** @brief Horizontal distance rolled on the ground, in metres. */
    double groundRoll{};
};

/** @brief Result of one physics substep (AdvanceMission) or one action (FlightEnvironment::Step). */
struct StepResult final
{
    /** @brief Weighted reward of each channel, in Reward order. */
    std::array<double, RewardCount> rewards{};
    /** @brief Sum of the channel rewards. */
    double reward{};
    /** @brief True at an absorbing end (Stopped or Crash). */
    bool terminated{};
    /** @brief True at a time or envelope limit (Timeout); the critic bootstraps. */
    bool truncated{};
};

/** @brief Normalized policy input, see ObservationIndex for the slot meanings. */
using Observation = std::array<float, ObservationSize>;

/**
 * @brief Wraps an angle to [-pi, pi].
 * @param angle Angle in radians.
 * @return Equivalent angle in [-pi, pi].
 */
inline double WrapAngle(double angle) noexcept
{
    return std::remainder(angle, 2.0 * Pi);
}

/**
 * @brief Checks that every physical value of a state is finite.
 * @param state Aircraft state to inspect.
 * @return False for any NaN or infinity, or when the simulator flagged the state as invalid.
 */
inline bool IsFinite(AircraftState const& state) noexcept
{
    std::array<double, 15> const values{
        state.position.north, state.position.east, state.position.up,
        state.velocity.north, state.velocity.east, state.velocity.up,
        state.airspeed, state.roll, state.pitch, state.heading,
        state.rollRate, state.pitchRate, state.yawRate, state.gearPosition, state.flapPosition};
    for (double const value : values)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    if (state.wheelDataValid)
    {
        for (auto const& wheel : state.wheels)
        {
            if (!std::isfinite(wheel.point.north) || !std::isfinite(wheel.point.east)
                || !std::isfinite(wheel.point.up))
            {
                return false;
            }
        }
    }
    return state.finite;
}

} // namespace ppo::examples::flight
