#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ppo::examples::flight {

inline constexpr double Pi = 3.14159265358979323846;
inline constexpr double FeetToMeters = 0.3048;
inline constexpr double DegreesToRadians = Pi / 180.0;
inline constexpr std::size_t ContinuousActions = 5; // elevator, aileron, rudder, throttle, brake
inline constexpr std::size_t PolicyOutputs = 2 * ContinuousActions + 1;
inline constexpr std::size_t ObservationSize = 56;
inline constexpr std::size_t RolloutObservation = 34;
inline constexpr std::size_t MaximumWaypoints = 128;

struct Vec3 final { double north{}, east{}, up{}; };
struct WheelContact final { Vec3 point{};bool onGround{}; };

/** SI snapshot: geodetic local tangent frame at the runway threshold. */
struct AircraftState final
{
    Vec3 position{}, velocity{};
    double airspeed{}, roll{}, pitch{}, heading{};
    double rollRate{}, pitchRate{}, yawRate{}, gearPosition{}, flapPosition{};
    bool weightOnWheels{}, structuralContact{}, finite{true};
    std::array<WheelContact,3> wheels{};
    bool wheelDataValid{};
};

struct Controls final
{
    double elevator{}, aileron{}, rudder{}, throttle{}, brake{};
    bool gearDown{};
    double flap{};
};

enum class Phase : std::uint8_t { Patrol, Approach, Rollout };
enum class Outcome : std::uint8_t { Flying, Stopped, Crash, Timeout, CaptureCompleted };
/** One policy, three independent training scenarios. The desktop defaults to Joint. */
enum class CurriculumStage : std::uint8_t { Loc, Glide, Joint };
inline constexpr char const* StageName(CurriculumStage stage) noexcept
{
    return stage==CurriculumStage::Loc ? "loc" : stage==CurriculumStage::Glide ? "glide" : "joint";
}
/** @brief Active navigation reference exposed to the policy snapshot and renderer. */
enum class GuidanceMode : std::uint8_t { Patrol, Localizer, Glide, Flare, Rollout };

struct Guidance final
{
    double along{}, cross{}, distance{}, localizer{}, glideError{};
    double targetSpeed{}, targetSlope{}, pathDistance{}, courseError{};
    double height{};
    Vec3 target{}, relativeVelocity{};
    /** @brief Source that owns the current target point and speed/slope commands. */
    GuidanceMode mode{GuidanceMode::Patrol};
    /** @brief True only when target/course-error fields are an active control reference. */
    bool targetValid{};
    bool ilsValid{};
};

enum class Reward : std::size_t
{
    Path, Localizer, Glide, Speed, Slope, EnterLocalizer, EnterGlide,
    Touchdown, Stop, Crash, Count
};
inline constexpr std::size_t RewardCount = static_cast<std::size_t>(Reward::Count);
inline constexpr std::array<char const*, RewardCount> RewardNames{
    "Path", "Localizer", "Glide", "Speed", "Slope", "EnterLocalizer", "EnterGlide",
    "Touchdown", "Stop", "Crash"};

struct MissionState final
{
    Phase phase{Phase::Patrol};
    Outcome outcome{Outcome::Flying};
    std::size_t waypoint{1};
    std::uint64_t steps{};
    double stoppedSeconds{};
    /** @brief Continuous stabilized-capture hold; terminal only for LOC/Glide lessons. */
    double captureGateSeconds{};
    double localizerHoldSeconds{},glideHoldSeconds{};
    /** @brief Continuous time outside each beam's release envelope. */
    double localizerLossSeconds{},glideLossSeconds{};
    /** @brief Delay before racetrack guidance returns after a confirmed ILS loss. */
    double approachRecoverySeconds{};
    std::size_t waypointsPassed{},completedLaps{};
    /** @brief First in-corridor RALLY sample, confirmed on the next sample. */
    bool rallyCorridorLatched{};
    bool pathQualified{};
    /** @brief Active captures; sustained pre-threshold loss returns to patrol and clears both. */
    bool capturedLocalizer{}, capturedGlide{};
    /** @brief True while approach-only guidance gets a short beam-recapture grace period. */
    bool approachRecoveryPending{};
    /** @brief True when the grace period must restore GLIDE as well as retain LOC. */
    bool approachRecoveryRequiresGlide{};
    /** @brief Preserve the existing one-shot entry bonuses across later reacquisitions. */
    bool rewardedLocalizerCapture{},rewardedGlideCapture{};
    bool touchedDown{};
    bool contactAttempted{},firstWheelContact{};
    double touchdownCrossMeters{},touchdownSinkMps{},touchdownSpeedMps{},touchdownGearPosition{};
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
    std::array<double, 18> const values{
        state.position.north, state.position.east, state.position.up,
        state.velocity.north, state.velocity.east, state.velocity.up,
        state.airspeed, state.roll, state.pitch, state.heading,
        state.rollRate, state.pitchRate, state.yawRate, state.gearPosition, state.flapPosition,
        std::hypot(state.velocity.north, state.velocity.east),
        std::hypot(state.position.north, state.position.east), state.velocity.up};
    for (double const value : values)
        if (!std::isfinite(value)) return false;
    if(state.wheelDataValid) for(auto const& wheel:state.wheels)
        if(!std::isfinite(wheel.point.north) || !std::isfinite(wheel.point.east) || !std::isfinite(wheel.point.up)) return false;
    return state.finite;
}

} // namespace ppo::examples::flight
