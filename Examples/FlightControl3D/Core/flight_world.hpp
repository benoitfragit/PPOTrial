/**
 * @file Examples/FlightControl3D/Core/flight_world.hpp
 * @brief Layout of the FlightControl3D world row (task.json "world"): what the environment
 *        DLL writes after each step and the Visualiseur reads, group by group.
 * @details The groups follow each other in this order, as in projects/T6/task.json; each
 * enum lists the values of one group, in the order of its task.json "fields". Everything
 * the desktop renderer read from a FlightSnapshot that comes from the simulation is here:
 * aircraft, runway and deck, guidance, mission, landing, crash, episode rewards and the
 * convergence diagnostics of the observations and actions.
 *
 * Read next: Environment/flight_environment_library.cpp (writer),
 * Visualiser/flight_visualiser.cpp (reader).
 */
#pragma once

#include "flight_types.hpp"

#include <cstddef>

namespace ppo::examples::flight::world {

/** @brief "aircraft": pose of the aircraft (north, east, up in metres; angles in radians). */
enum Aircraft : std::size_t
{
    AircraftNorth,
    AircraftEast,
    AircraftUp,
    AircraftRoll,
    AircraftPitch,
    AircraftHeading,
    AircraftGear,
    AircraftAirspeed,
    AircraftSize
};

/** @brief "airfield": the runway of the episode (threshold centre, heading in radians, dimensions). */
enum Airfield : std::size_t
{
    RunwayNorth,
    RunwayEast,
    RunwayUp,
    RunwayHeading,
    RunwayLength,
    RunwayWidth,
    AirfieldSize
};

/** @brief "deck": glide references of the runway, the sea and the motion of the deck. */
enum Deck : std::size_t
{
    ThresholdHeight,
    LandingMargin,
    Elevation,
    SeaHeight,
    DeckSpeed,
    DeckVerticalSpeed,
    SwellAmplitude,
    SwellPeriod,
    Heave,
    SwellAngle,
    DeckSize
};

/** @brief "guidance": the reference the policy and the rewards follow (Guidance), in world metres. */
enum GuidanceValues : std::size_t
{
    GuidanceMode,
    TargetValid,
    TargetSpeed,
    TargetSlope,
    TargetHeading,
    TargetNorth,
    TargetEast,
    TargetUp,
    DirectionNorth,
    DirectionEast,
    DirectionUp,
    VelocityDirectionNorth,
    VelocityDirectionEast,
    VelocityDirectionUp,
    RelativeAlong,
    RelativeRight,
    RelativeUp,
    Along,
    Cross,
    Distance,
    Height,
    AimAlong,
    GlideDegrees,
    GuidanceSize
};

/** @brief How the run uses the episodes: what ppokit-run said (set_learning) and the campaign resets. */
enum RunMode : std::size_t
{
    PlayMode,
    TrainMode,
    EvaluationMode
};

/**
 * @brief "mission": phase and outcome of the episode, its clock, the last episode, the
 *        curriculum, the filtered controls of the last action and the episode counters.
 */
enum Mission : std::size_t
{
    Phase,
    Outcome,
    ElapsedSeconds,
    MaximumSeconds,
    LastOutcome,
    LastSeconds,
    WeightOnWheels,
    AircraftType,
    Mode,
    StartedAirborne,
    CurriculumLevel,
    CurriculumLevels,
    WindowStops,
    WindowEpisodes,
    Difficulty,
    AppliedElevator,
    AppliedAileron,
    AppliedRudder,
    AppliedThrottle,
    AppliedBrake,
    AppliedFlap,
    AppliedGear,
    /** @brief Episodes this environment finished, and how they ended (the desktop's counters). */
    Episodes,
    Touchdowns,
    Stops,
    Crashes,
    Timeouts,
    MissionSize
};

/** @brief Aircraft types of the "aircraft" of a scenario (Mission AircraftType). */
enum AircraftKind : std::size_t
{
    T6Texan2,
    Pc7
};

/** @brief "landing": the touchdown and stop of the current episode once it touched down, else of the last one. */
enum Landing : std::size_t
{
    FromCurrentEpisode,
    TouchedDown,
    TouchdownFromThreshold,
    GroundRoll,
    TouchdownSink,
    TouchdownSpeed,
    TouchdownQuality,
    LandingAreaFactor,
    ResultOutcome,
    StopFromThreshold,
    LandingSize
};

/** @brief Values of one crash condition: failed (0 or 1), measured value, limit. */
inline constexpr std::size_t CrashConditionValues = 3;

/**
 * @brief "crash": the latest crash, kept across episodes: its conditions (CrashReason order),
 *        then the pose of the aircraft and of the runway at the crash.
 */
enum Crash : std::size_t
{
    HasCrash,
    CrashEvent,
    CrashConditions,
    CrashAircraftNorth = CrashConditions + CrashReasonCount * CrashConditionValues,
    CrashAircraftEast,
    CrashAircraftUp,
    CrashAircraftRoll,
    CrashAircraftPitch,
    CrashAircraftHeading,
    CrashAircraftGear,
    CrashRunwayNorth,
    CrashRunwayEast,
    CrashRunwayUp,
    CrashRunwayHeading,
    CrashSize
};

/** @brief "episode_rewards": weighted sum of each channel over the episode, then the active weights. */
enum EpisodeRewards : std::size_t
{
    EpisodeSums,
    ActiveWeights = EpisodeSums + RewardCount,
    EpisodeRewardsSize = ActiveWeights + RewardCount
};

/**
 * @brief "diagnostics": the convergence diagnostics of FlightSession, over the last complete
 *        episode (the window): the live observation and raw action of the last step, then per
 *        observation slot its mean |x|, maximum |x| and saturations, per action axis its raw
 *        and applied mean |x| and saturations, and the mean of each reward channel.
 */
enum Diagnostics : std::size_t
{
    DiagnosticsEnabled,
    SaturationThreshold,
    WindowSamples,
    LiveObservation,
    LiveRawAction = LiveObservation + ObservationSize,
    ObservationMeanAbsolute = LiveRawAction + ContinuousActions,
    ObservationMaximumAbsolute = ObservationMeanAbsolute + ObservationSize,
    ObservationSaturations = ObservationMaximumAbsolute + ObservationSize,
    RawMeanAbsolute = ObservationSaturations + ObservationSize,
    AppliedMeanAbsolute = RawMeanAbsolute + ContinuousActions,
    RawSaturations = AppliedMeanAbsolute + ContinuousActions,
    AppliedSaturations = RawSaturations + ContinuousActions,
    RewardMeans = AppliedSaturations + ContinuousActions,
    DiagnosticsSize = RewardMeans + RewardCount
};

/** @brief Offset of each group in the row. */
inline constexpr std::size_t AircraftAt = 0;
inline constexpr std::size_t AirfieldAt = AircraftAt + AircraftSize;
inline constexpr std::size_t DeckAt = AirfieldAt + AirfieldSize;
inline constexpr std::size_t GuidanceAt = DeckAt + DeckSize;
inline constexpr std::size_t MissionAt = GuidanceAt + GuidanceSize;
inline constexpr std::size_t LandingAt = MissionAt + MissionSize;
inline constexpr std::size_t CrashAt = LandingAt + LandingSize;
inline constexpr std::size_t EpisodeRewardsAt = CrashAt + CrashSize;
inline constexpr std::size_t DiagnosticsAt = EpisodeRewardsAt + EpisodeRewardsSize;
/** @brief Values of one world row. */
inline constexpr std::size_t Size = DiagnosticsAt + DiagnosticsSize;

// projects/T6/task.json "world" declares these sizes (checked by the repository test too).
static_assert(AircraftSize == 8 && AirfieldSize == 6 && DeckSize == 10 && GuidanceSize == 23, "task.json world");
static_assert(MissionSize == 27 && LandingSize == 10 && CrashSize == 58 && EpisodeRewardsSize == 14,
    "task.json world");
static_assert(DiagnosticsSize == 175 && Size == 331, "task.json world");

} // namespace ppo::examples::flight::world
