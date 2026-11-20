#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ppo::examples::flight {

inline constexpr double Pi = 3.14159265358979323846;
inline constexpr double FeetToMeters = 0.3048;
inline constexpr double DegreesToRadians = Pi / 180.0;
/** @brief Continuous actor outputs: elevator, aileron, rudder, throttle and brake. */
inline constexpr std::size_t ContinuousActions = 5;
/** @brief Number of named observation slots, see ObservationIndex. */
inline constexpr std::size_t ObservationSize = 33;
/** @brief Observation slot that flags the ground-roll phase and enables the brake action. */
inline constexpr std::size_t RolloutObservation = 25;
/** @brief Every normalized observation lies within [-ObservationBound, ObservationBound]. */
inline constexpr float ObservationBound = 1.F;

struct Vec3 final { double north{}, east{}, up{}; };
struct WheelContact final { Vec3 point{}; bool onGround{}; };

/** SI snapshot: geodetic local tangent frame at the runway reference point. */
struct AircraftState final
{
    Vec3 position{}, velocity{};
    double airspeed{}, roll{}, pitch{}, heading{};
    double rollRate{}, pitchRate{}, yawRate{}, gearPosition{}, flapPosition{};
    bool weightOnWheels{}, structuralContact{}, finite{true};
    std::array<WheelContact, 3> wheels{};
    bool wheelDataValid{};
};

/** @brief Physical control inputs; the landing gear always stays extended. */
struct Controls final
{
    double elevator{}, aileron{}, rudder{}, throttle{}, brake{};
    double flap{};
};

enum class Phase : std::uint8_t { Approach, Rollout };
enum class Outcome : std::uint8_t { Flying, Stopped, Crash, Timeout };

/** @brief Independent conditions that can make a flight end in a crash. */
enum class CrashReason : std::size_t
{
    InvalidState, Distance, Altitude, LandingArea, WheelsOutside, Gear, StructuralContact,
    Sink, Speed, Roll, PitchLow, PitchHigh, Heading, NoWheelContact, Count
};
inline constexpr std::size_t CrashReasonCount = static_cast<std::size_t>(CrashReason::Count);
/** @brief One failed condition, with the measured value and its configured limit when applicable. */
struct CrashCondition final { bool failed{}; double value{}, limit{}; };
/** @brief Diagnostic captured at the terminal physics substep, independent of episode resets. */
struct CrashDiagnostic final
{
    std::array<CrashCondition, CrashReasonCount> conditions{};
};

/** @brief Active navigation reference exposed to the policy snapshot and renderer. */
enum class GuidanceMode : std::uint8_t { Approach, Flare, Rollout };

struct Guidance final
{
    /** @brief Runway-frame position: distance from the threshold, lateral offset and height. */
    double along{}, cross{}, distance{}, height{};
    double targetSpeed{}, targetSlope{};
    /** @brief Absolute commanded track azimuth, for diagnostics and the renderer. */
    double targetHeading{};
    Vec3 target{};
    /** @brief Unit target and actual deck-relative velocity directions in the world frame. */
    Vec3 targetDirection{}, velocityDirection{};
    /** @brief Aircraft velocity relative to the deck, expressed in runway axes. */
    Vec3 relativeVelocity{};
    GuidanceMode mode{GuidanceMode::Approach};
    /** @brief True only when the geometric target point is an active control reference. */
    bool targetValid{};
};

/**
 * @brief Reward channels. Path, VerticalSpeed and Speed are potential-based
 * shaping terms; Direction is a running cost; the others are one-shot events.
 */
enum class Reward : std::size_t
{
    Path, Direction, VerticalSpeed, Speed, Touchdown, Stop, Crash, Count
};
inline constexpr std::size_t RewardCount = static_cast<std::size_t>(Reward::Count);
inline constexpr std::array<char const*, RewardCount> RewardNames{
    "Path", "Direction", "VerticalSpeed", "Speed", "Touchdown", "Stop", "Crash"};

struct MissionState final
{
    Phase phase{Phase::Approach};
    Outcome outcome{Outcome::Flying};
    std::uint64_t steps{};
    double stoppedSeconds{};
    /** @brief Last filtered physical controls, included in the policy observation. */
    Controls filteredControls{};
    bool touchedDown{};
    bool contactAttempted{}, firstWheelContact{};
    double touchdownCrossMeters{}, touchdownSinkMps{}, touchdownSpeedMps{}, touchdownGearPosition{};
    /** @brief Touchdown factor from location, sink rate and speed, applied to the Touchdown reward. */
    double touchdownQuality{1.0};
    /** @brief Landing-surface factor of the touchdown point, applied to the Stop reward. */
    double landingAreaFactor{1.0};
    double touchdownFromThreshold{}, stopFromThreshold{}, groundRoll{};
};

struct StepResult final
{
    std::array<double, RewardCount> rewards{};
    double reward{};
    bool terminated{}, truncated{};
};
using Observation = std::array<float, ObservationSize>;

inline double WrapAngle(double angle) noexcept
{
    return std::remainder(angle, 2.0 * Pi);
}

inline bool IsFinite(AircraftState const& state) noexcept
{
    std::array<double, 15> const values{
        state.position.north, state.position.east, state.position.up,
        state.velocity.north, state.velocity.east, state.velocity.up,
        state.airspeed, state.roll, state.pitch, state.heading,
        state.rollRate, state.pitchRate, state.yawRate, state.gearPosition, state.flapPosition};
    for (double const value : values)
        if (!std::isfinite(value)) return false;
    if (state.wheelDataValid)
        for (auto const& wheel : state.wheels)
            if (!std::isfinite(wheel.point.north) || !std::isfinite(wheel.point.east)
                || !std::isfinite(wheel.point.up)) return false;
    return state.finite;
}

} // namespace ppo::examples::flight
