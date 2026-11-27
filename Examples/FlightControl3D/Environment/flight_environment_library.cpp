/**
 * @file Examples/FlightControl3D/Environment/flight_environment_library.cpp
 * @brief The FlightControl3D simulation as an environment DLL for ppokit-run.
 * @details Everything of the example except the learner: JSBSim, mission, rewards,
 * observation, the training stream of starts and its curriculum. ppokit-run decides and
 * learns through the project's network DLL.
 *
 * - Action: the task's six values (five Gaussian controls, then the gear Bernoulli draw),
 *   mapped to physical controls by ToControls. The brake acts only while rolling and the
 *   gear only until its extension is latched, as in the example's own learner.
 * - Rewards: the weighted value of each channel of the step (StepResult::rewards).
 * - World (task.json "world", Core/flight_world.hpp): what the desktop renderer read of the
 *   simulation, for the Visualiseur: the aircraft, the runway and the deck, the guidance,
 *   the mission and the last episode, the landing, the latest crash (kept across episodes,
 *   as FlightSession kept it), the episode's reward sums and the convergence diagnostics of
 *   FlightSession over the last complete episode.
 * - Curriculum: airborne training episodes promote the start difficulty as in
 *   FlightSession. Only training moves it: while ppokit-run plays or evaluates
 *   (set_learning 0) it stays, as the desktop's inference kept it.
 * - Training state (TrainingState / RestoreTrainingState): the curriculum level, its
 *   current window and the training stream (random engine, next approach side). ppokit-run
 *   saves it with each checkpoint of the run and gives it back when the run resumes, so the
 *   weights, the curriculum and the stream of starts resume together; a fresh run gets the
 *   empty state (level 0, empty window). checkpoints/ppokit_curriculum_level.txt, which
 *   earlier versions wrote at each promotion, is only read when the environment is created,
 *   for a checkpoint saved without a state.
 * - Evaluation (the suites of eval/): an episode of a campaign starts as a FlightSession
 *   validation episode, Reset(seed, EvaluationStart(episode, episodes, rolling share,
 *   curriculum difficulty)), and never moves the curriculum. Its metrics (task.json
 *   "metrics", in this order): stopped, touched_down, crashed, timed_out (flags), stop_m
 *   and touchdown_m past the threshold (NaN when they did not happen), and the curriculum
 *   difficulty of the campaign, as ValidationMetrics counts them.
 */
#include "flight_curriculum.hpp"
#include "flight_environment.hpp"
#include "flight_guidance.hpp"
#include "flight_policy.hpp"
#include "flight_reward_tuning.hpp"
#include "flight_world.hpp"

#include "ppokit/artifacts/random_state.hpp"
#include "ppokit/environment/sdk.hpp"
#include "ppokit/environment/run_profile.h"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace flight = ppo::examples::flight;

constexpr std::size_t ActionSize = flight::ContinuousActions + 1;
constexpr std::size_t GearAction = flight::ContinuousActions;
namespace world = flight::world;
/** @brief Schema of the training state text. */
constexpr char const* TrainingStateSchema = "flight.training_state/1";
/** @brief A binary observation flag is set above this value. */
constexpr float FlagThreshold = 0.5F;

double Radians(double degrees) noexcept
{
    return degrees * flight::Pi / 180.0;
}

/** @brief The scenario of a project (project.json "settings"). */
std::filesystem::path SettingsOf(std::filesystem::path const& project)
{
    std::optional<std::filesystem::path> const settings = ppokit::schema::LoadProject(project).SettingsFile();
    if (!settings)
    {
        throw std::runtime_error(project.string() + "/project.json names no \"settings\" (the scenario JSON)");
    }
    return *settings;
}

/** @brief A float of the world row from a double, a count or a flag. */
float Value(double value) noexcept
{
    return static_cast<float>(value);
}

/** @brief Aircraft type of a scenario's JSBSim model name (t6texan2 or pc7, as LoadConfig validates). */
world::AircraftKind KindOf(std::string const& aircraft) noexcept
{
    if (aircraft == "pc7")
    {
        return world::Pc7;
    }
    return world::T6Texan2;
}

/**
 * @brief The convergence diagnostics of FlightSession over one window of actions (here an
 *        episode): absolute sums, maxima and saturation counts of the observations and actions,
 *        and the reward sums.
 */
struct DiagnosticWindow final
{
    std::uint64_t samples{};
    std::array<double, flight::ObservationSize> observationAbsoluteSums{};
    std::array<double, flight::ObservationSize> observationMaximumAbsolute{};
    std::array<std::uint64_t, flight::ObservationSize> observationSaturations{};
    std::array<double, flight::ContinuousActions> rawAbsoluteSums{};
    std::array<double, flight::ContinuousActions> appliedAbsoluteSums{};
    std::array<std::uint64_t, flight::ContinuousActions> rawSaturations{};
    std::array<std::uint64_t, flight::ContinuousActions> appliedSaturations{};
    std::array<double, flight::RewardCount> rewardSums{};

    /**
     * @brief Adds one action as FlightSession's RecordDiagnostics did.
     * @param observation Observation the action was chosen on.
     * @param action Raw action and its active axes.
     * @param applied Filtered physical controls after the action.
     * @param step Rewards of the action.
     * @param threshold Fraction of the bound counted as saturated.
     */
    void Record(flight::Observation const& observation, flight::PolicyAction const& action,
        flight::Controls const& applied, flight::StepResult const& step, double threshold) noexcept
    {
        ++samples;
        for (std::size_t slot = 0; slot < flight::ObservationSize; ++slot)
        {
            double const absolute = std::abs(static_cast<double>(observation[slot]));
            observationAbsoluteSums[slot] += absolute;
            observationMaximumAbsolute[slot] = std::max(observationMaximumAbsolute[slot], absolute);
            if (absolute >= threshold * flight::ObservationBound)
            {
                ++observationSaturations[slot];
            }
        }
        std::array<double, flight::ContinuousActions> const physical = flight::Applied(applied);
        for (std::size_t axis = 0; axis < flight::ContinuousActions; ++axis)
        {
            rawAbsoluteSums[axis] += std::abs(action.raw[axis]);
            appliedAbsoluteSums[axis] += std::abs(physical[axis]);
            bool const active = flight::ActiveAxis(axis, action.brakeActive);
            if (active && flight::RawSaturated(axis, action.raw[axis], threshold))
            {
                ++rawSaturations[axis];
            }
            if (active && flight::AppliedSaturated(axis, physical[axis], threshold))
            {
                ++appliedSaturations[axis];
            }
        }
        for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
        {
            rewardSums[channel] += step.rewards[channel];
        }
    }
};

/** @brief Episodes finished by one environment, and how they ended. */
struct EpisodeCounters final
{
    std::uint64_t episodes{};
    std::uint64_t touchdowns{};
    std::uint64_t stops{};
    std::uint64_t crashes{};
    std::uint64_t timeouts{};
};

/** @brief Indices of the episode metrics, in the order of task.json "metrics". */
enum Metric : std::size_t
{
    StoppedMetric,
    TouchedDownMetric,
    CrashedMetric,
    TimedOutMetric,
    StopMetersMetric,
    TouchdownMetersMetric,
    DifficultyMetric,
    MetricTotal,
};

double Flag(bool value) noexcept
{
    return value ? 1.0 : 0.0;
}

/** @brief A distance that applies only when its event happened, else NaN. */
double DistanceIf(bool happened, double meters) noexcept
{
    if (happened)
    {
        return meters;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief The configuration, once task.json's "contract" is checked against it: the task
 *        names the aircraft and contract its checkpoints are trained in.
 * @throws std::runtime_error when task.json names another one.
 */
flight::Config CheckedContract(std::filesystem::path const& project, flight::Config config)
{
    std::string const contract = flight::EnvironmentContract(config);
    ppokit::schema::TaskSchema const task = ppokit::schema::LoadTaskSchema(project / "task.json");
    if (task.contract != contract)
    {
        throw std::runtime_error(project.string() + ": task.json \"contract\" is '" + task.contract
            + "', the scenario's aircraft and contract are '" + contract + "'");
    }
    return config;
}

class FlightEnvironmentLibrary final
{
public:
    static constexpr std::size_t MetricCount = MetricTotal;

    FlightEnvironmentLibrary(std::filesystem::path const& project, std::uint64_t seed)
        : config_(CheckedContract(project, flight::LoadConfig(SettingsOf(project)))),
          environment_(config_, flight::DefaultDataRoot(), static_cast<std::uint32_t>(seed)),
          curriculum_(config_.training.curriculum)
    {
        // The level file of an earlier version, for a checkpoint saved without a training state.
        if (std::optional<std::size_t> const level =
                flight::LoadCurriculumLevel(project / "checkpoints" / "ppokit_curriculum_level.txt"))
        {
            curriculum_.Restore(*level);
        }
        environment_.SetDifficulty(curriculum_.Difficulty());
    }

    void Reset(float* observation, float* world)
    {
        evaluating_ = false;
        current_ = environment_.Reset();
        StartEpisode();
        std::copy(current_.begin(), current_.end(), observation);
        WriteWorld(world);
    }

    /** @brief Training moves the curriculum; playing and evaluating leave it as it is. */
    void SetLearning(bool learning) noexcept { learning_ = learning; }

    /** @brief The curriculum (level, current window) and the training stream, as JSON. */
    std::string TrainingState() const
    {
        flight::FlightEnvironment::TrainingStream const stream = environment_.SavedTrainingStream();
        nlohmann::json state;
        state["schema"] = TrainingStateSchema;
        state["level"] = curriculum_.Level();
        state["window"] = {{"episodes", curriculum_.WindowEpisodes()}, {"stops", curriculum_.WindowStops()}};
        state["next_side"] = SideName(stream.nextSide);
        state["random"] = ppokit::artifacts::StandardRandomText(stream.random);
        return state.dump();
    }

    /**
     * @brief Resumes a state of TrainingState; the empty state (a fresh run) is level 0 with
     *        an empty window, the training stream staying as the seed started it.
     * @throws std::runtime_error for a state of another schema or out of the curriculum.
     */
    void RestoreTrainingState(std::string const& text)
    {
        if (text.empty())
        {
            curriculum_.Restore(0);
        }
        else
        {
            ResumeTrainingState(text);
        }
        environment_.SetDifficulty(curriculum_.Difficulty());
    }

    /** @brief A validation episode of FlightSession: seeded, its start spread over the campaign. */
    void ResetEvaluation(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes, float* observation,
        float* world)
    {
        evaluating_ = true;
        ResetSeeded(seed, episode, episodes, curriculum_.Difficulty(), observation, world);
    }

    /** @brief The full-task mix of FlightSession::ResetInferenceEpisode, including early checkpoints. */
    void ResetInference(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes, float* observation,
        float* world)
    {
        evaluating_ = false;
        ResetSeeded(seed, episode, episodes, flight::FullTaskDifficulty, observation, world);
    }

    void EpisodeMetrics(double* metrics) const
    {
        flight::MissionState const& mission = environment_.Mission();
        bool const stopped = mission.outcome == flight::Outcome::Stopped;
        metrics[StoppedMetric] = Flag(stopped);
        metrics[TouchedDownMetric] = Flag(mission.touchedDown);
        metrics[CrashedMetric] = Flag(mission.outcome == flight::Outcome::Crash);
        metrics[TimedOutMetric] = Flag(mission.outcome == flight::Outcome::Timeout);
        metrics[StopMetersMetric] = DistanceIf(stopped, mission.stopFromThreshold);
        metrics[TouchdownMetersMetric] = DistanceIf(mission.touchedDown, mission.touchdownFromThreshold);
        metrics[DifficultyMetric] = curriculum_.Difficulty();
    }

    ppokit::runtime::StepEnd Step(float const* action, float* observation, float* world, double* rewardChannels)
    {
        flight::PolicyAction const policy = ToPolicyAction(action);
        flight::StepResult const step = environment_.Step(flight::ToControls(policy, config_.initial));
        RecordStep(policy, step);
        current_ = environment_.ObservationNow();
        std::copy(current_.begin(), current_.end(), observation);
        std::copy(step.rewards.begin(), step.rewards.end(), rewardChannels);
        bool const ended = step.terminated || step.truncated;
        if (ended)
        {
            FinishEpisode();
        }
        if (ended && learning_ && !evaluating_)
        {
            AdvanceCurriculum();
        }
        WriteWorld(world);
        if (step.terminated)
        {
            return ppokit::runtime::StepEnd::Terminated;
        }
        return step.truncated ? ppokit::runtime::StepEnd::Truncated : ppokit::runtime::StepEnd::Running;
    }

    /** @brief Validates every weight first: applied whole or not at all. */
    void SetWeights(double const* weights)
    {
        flight::Config next = config_;
        for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
        {
            next = flight::WithRewardTuning(next, channel, weights[channel]);
        }
        config_ = next;
        environment_.SetRewardSettings(config_.reward);
    }

    void Weights(double* weights) const
    {
        std::copy(config_.reward.weights.begin(), config_.reward.weights.end(), weights);
    }

private:
    /** @brief Both campaign modes use the same native start geometry and JSBSim instance. */
    void ResetSeeded(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes, double frontier,
        float* observation, float* world)
    {
        flight::EpisodeStart const start = flight::EvaluationStart(
            episode, episodes, config_.training.curriculum.rolloutStartShare, frontier);
        environment_.Reset(seed, start);
        current_ = environment_.ObservationNow();
        StartEpisode();
        std::copy(current_.begin(), current_.end(), observation);
        WriteWorld(world);
    }

    /** @brief The task's action as the example's policy action, with the axes active in this state. */
    flight::PolicyAction ToPolicyAction(float const* action) const noexcept
    {
        flight::PolicyAction policy;
        for (std::size_t axis = 0; axis < flight::ContinuousActions; ++axis)
        {
            policy.raw[axis] = action[axis];
        }
        policy.brakeActive = current_[flight::RolloutObservation] > FlagThreshold;
        policy.gearActive = current_[flight::GearCommandObservation] <= FlagThreshold;
        policy.gearDown = action[GearAction] > 0.5F;
        return policy;
    }

    /** @brief Clears what belongs to one episode: its reward sums and its diagnostic window. */
    void StartEpisode() noexcept
    {
        episodeRewardSums_.fill(0.0);
        window_ = DiagnosticWindow{};
    }

    /** @brief Adds an action to the episode's reward sums and diagnostics; keeps it as the live one. */
    void RecordStep(flight::PolicyAction const& policy, flight::StepResult const& step) noexcept
    {
        for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
        {
            episodeRewardSums_[channel] += step.rewards[channel];
        }
        liveObservation_ = current_;
        liveRaw_ = policy.raw;
        window_.Record(current_, policy, environment_.Mission().filteredControls, step,
            config_.diagnostics.saturationThreshold);
    }

    /** @brief The episode ended: it becomes the last one, its window the one shown; a crash is kept. */
    void FinishEpisode() noexcept
    {
        lastEpisode_ = environment_.Mission();
        lastWindow_ = window_;
        CountEpisode(lastEpisode_);
        if (lastEpisode_.outcome != flight::Outcome::Crash)
        {
            return;
        }
        crash_ = environment_.CrashDetails();
        hasCrash_ = true;
        // A non-finite crash keeps its reasons but has no pose to show, as in FlightSession.
        if (!flight::IsFinite(environment_.Aircraft()))
        {
            return;
        }
        crashAircraft_ = environment_.Aircraft();
        crashRunway_ = environment_.Settings().runway;
        ++crashEvent_;
    }

    /** @brief Counts a finished episode and how it ended, as FlightSession did. */
    void CountEpisode(flight::MissionState const& episode) noexcept
    {
        ++counters_.episodes;
        counters_.touchdowns += episode.touchedDown ? 1U : 0U;
        counters_.stops += episode.outcome == flight::Outcome::Stopped ? 1U : 0U;
        counters_.crashes += episode.outcome == flight::Outcome::Crash ? 1U : 0U;
        counters_.timeouts += episode.outcome == flight::Outcome::Timeout ? 1U : 0U;
    }

    /** @brief How the run uses this episode (world::RunMode). */
    world::RunMode Mode() const noexcept
    {
        if (evaluating_)
        {
            return world::EvaluationMode;
        }
        return learning_ ? world::TrainMode : world::PlayMode;
    }

    void WriteWorld(float* row) const noexcept
    {
        if (row == nullptr)
        {
            return;
        }
        WriteAircraft(row + world::AircraftAt);
        WriteRunway(row);
        WriteGuidance(row + world::GuidanceAt);
        WriteMission(row + world::MissionAt);
        WriteLanding(row + world::LandingAt);
        WriteCrash(row + world::CrashAt);
        WriteRewards(row + world::EpisodeRewardsAt);
        WriteDiagnostics(row + world::DiagnosticsAt);
    }

    void WriteAircraft(float* values) const noexcept
    {
        flight::AircraftState const& aircraft = environment_.Aircraft();
        values[world::AircraftNorth] = Value(aircraft.position.north);
        values[world::AircraftEast] = Value(aircraft.position.east);
        values[world::AircraftUp] = Value(aircraft.position.up);
        values[world::AircraftRoll] = Value(aircraft.roll);
        values[world::AircraftPitch] = Value(aircraft.pitch);
        values[world::AircraftHeading] = Value(aircraft.heading);
        values[world::AircraftGear] = Value(aircraft.gearPosition);
        values[world::AircraftAirspeed] = Value(aircraft.airspeed);
    }

    /** @brief The "airfield" and "deck" groups: the runway of the episode, the sea and the deck motion. */
    void WriteRunway(float* row) const noexcept
    {
        flight::Config const& settings = environment_.Settings();
        flight::Runway const& runway = settings.runway;
        float* const airfield = row + world::AirfieldAt;
        airfield[world::RunwayNorth] = Value(runway.origin.north);
        airfield[world::RunwayEast] = Value(runway.origin.east);
        airfield[world::RunwayUp] = Value(runway.origin.up);
        airfield[world::RunwayHeading] = Value(Radians(runway.headingDegrees));
        airfield[world::RunwayLength] = Value(runway.lengthMeters);
        airfield[world::RunwayWidth] = Value(runway.widthMeters);
        flight::DeckState const deck = environment_.CurrentDeck();
        float* const values = row + world::DeckAt;
        values[world::ThresholdHeight] = Value(runway.thresholdHeightMeters);
        values[world::LandingMargin] = Value(runway.landingMarginMeters);
        values[world::Elevation] = Value(runway.elevationMeters);
        values[world::SeaHeight] = Value(settings.runwayRandomization.seaHeightMeters);
        values[world::DeckSpeed] = Value(deck.motion.speedMps);
        values[world::DeckVerticalSpeed] = Value(runway.velocity.up);
        values[world::SwellAmplitude] = Value(deck.motion.swellAmplitudeMeters);
        values[world::SwellPeriod] = Value(deck.motion.swellPeriodSeconds);
        values[world::Heave] = Value(deck.heaveMeters);
        values[world::SwellAngle] = Value(deck.swellAngle);
    }

    void WriteGuidance(float* values) const noexcept
    {
        flight::Guidance const guidance = environment_.CurrentGuidance();
        values[world::GuidanceMode] = Value(static_cast<double>(guidance.mode));
        values[world::TargetValid] = Value(Flag(guidance.targetValid));
        values[world::TargetSpeed] = Value(guidance.targetSpeed);
        values[world::TargetSlope] = Value(guidance.targetSlope);
        values[world::TargetHeading] = Value(guidance.targetHeading);
        values[world::TargetNorth] = Value(guidance.target.north);
        values[world::TargetEast] = Value(guidance.target.east);
        values[world::TargetUp] = Value(guidance.target.up);
        values[world::DirectionNorth] = Value(guidance.targetDirection.north);
        values[world::DirectionEast] = Value(guidance.targetDirection.east);
        values[world::DirectionUp] = Value(guidance.targetDirection.up);
        values[world::VelocityDirectionNorth] = Value(guidance.velocityDirection.north);
        values[world::VelocityDirectionEast] = Value(guidance.velocityDirection.east);
        values[world::VelocityDirectionUp] = Value(guidance.velocityDirection.up);
        values[world::RelativeAlong] = Value(guidance.relativeVelocity.north);
        values[world::RelativeRight] = Value(guidance.relativeVelocity.east);
        values[world::RelativeUp] = Value(guidance.relativeVelocity.up);
        values[world::Along] = Value(guidance.along);
        values[world::Cross] = Value(guidance.cross);
        values[world::Distance] = Value(guidance.distance);
        values[world::Height] = Value(guidance.height);
        values[world::AimAlong] = Value(flight::AimPointAlongMeters(environment_.Settings()));
        values[world::GlideDegrees] = Value(config_.mission.glideDegrees);
    }

    void WriteMission(float* values) const noexcept
    {
        flight::MissionState const& mission = environment_.Mission();
        flight::Config const& settings = environment_.Settings();
        values[world::Phase] = Value(static_cast<double>(mission.phase));
        values[world::Outcome] = Value(static_cast<double>(mission.outcome));
        values[world::ElapsedSeconds] = Value(static_cast<double>(mission.steps) * settings.mission.physicsDt);
        values[world::MaximumSeconds] = Value(settings.mission.maxSeconds);
        values[world::LastOutcome] = Value(static_cast<double>(lastEpisode_.outcome));
        values[world::LastSeconds] = Value(static_cast<double>(lastEpisode_.steps) * config_.mission.physicsDt);
        values[world::WeightOnWheels] = Value(Flag(environment_.Aircraft().weightOnWheels));
        values[world::AircraftType] = Value(static_cast<double>(KindOf(config_.aircraft)));
        values[world::Mode] = Value(static_cast<double>(Mode()));
        values[world::StartedAirborne] = Value(Flag(environment_.StartedAirborne()));
        values[world::CurriculumLevel] = Value(static_cast<double>(curriculum_.Level()));
        values[world::CurriculumLevels] = Value(static_cast<double>(curriculum_.Levels()));
        values[world::WindowStops] = Value(static_cast<double>(curriculum_.WindowStops()));
        values[world::WindowEpisodes] = Value(static_cast<double>(curriculum_.WindowEpisodes()));
        values[world::Difficulty] = Value(curriculum_.Difficulty());
        flight::Controls const& applied = mission.filteredControls;
        values[world::AppliedElevator] = Value(applied.elevator);
        values[world::AppliedAileron] = Value(applied.aileron);
        values[world::AppliedRudder] = Value(applied.rudder);
        values[world::AppliedThrottle] = Value(applied.throttle);
        values[world::AppliedBrake] = Value(applied.brake);
        values[world::AppliedFlap] = Value(applied.flap);
        values[world::AppliedGear] = Value(applied.gear);
        values[world::Episodes] = Value(static_cast<double>(counters_.episodes));
        values[world::Touchdowns] = Value(static_cast<double>(counters_.touchdowns));
        values[world::Stops] = Value(static_cast<double>(counters_.stops));
        values[world::Crashes] = Value(static_cast<double>(counters_.crashes));
        values[world::Timeouts] = Value(static_cast<double>(counters_.timeouts));
    }

    /** @brief The touchdown of the current episode once it touched down, else of the last one. */
    void WriteLanding(float* values) const noexcept
    {
        flight::MissionState const& mission = environment_.Mission();
        bool const current = mission.touchedDown;
        flight::MissionState const& result = current ? mission : lastEpisode_;
        values[world::FromCurrentEpisode] = Value(Flag(current));
        values[world::TouchedDown] = Value(Flag(result.touchedDown));
        values[world::TouchdownFromThreshold] = Value(result.touchdownFromThreshold);
        values[world::GroundRoll] = Value(result.groundRoll);
        values[world::TouchdownSink] = Value(result.touchdownSinkMps);
        values[world::TouchdownSpeed] = Value(result.touchdownSpeedMps);
        values[world::TouchdownQuality] = Value(result.touchdownQuality);
        values[world::LandingAreaFactor] = Value(result.landingAreaFactor);
        values[world::ResultOutcome] = Value(static_cast<double>(result.outcome));
        values[world::StopFromThreshold] = Value(result.stopFromThreshold);
    }

    void WriteCrash(float* values) const noexcept
    {
        values[world::HasCrash] = Value(Flag(hasCrash_));
        values[world::CrashEvent] = Value(static_cast<double>(crashEvent_));
        for (std::size_t reason = 0; reason < flight::CrashReasonCount; ++reason)
        {
            flight::CrashCondition const& condition = crash_.conditions[reason];
            float* const entry = values + world::CrashConditions + reason * world::CrashConditionValues;
            entry[0] = Value(Flag(condition.failed));
            entry[1] = Value(condition.value);
            entry[2] = Value(condition.limit);
        }
        values[world::CrashAircraftNorth] = Value(crashAircraft_.position.north);
        values[world::CrashAircraftEast] = Value(crashAircraft_.position.east);
        values[world::CrashAircraftUp] = Value(crashAircraft_.position.up);
        values[world::CrashAircraftRoll] = Value(crashAircraft_.roll);
        values[world::CrashAircraftPitch] = Value(crashAircraft_.pitch);
        values[world::CrashAircraftHeading] = Value(crashAircraft_.heading);
        values[world::CrashAircraftGear] = Value(crashAircraft_.gearPosition);
        values[world::CrashRunwayNorth] = Value(crashRunway_.origin.north);
        values[world::CrashRunwayEast] = Value(crashRunway_.origin.east);
        values[world::CrashRunwayUp] = Value(crashRunway_.origin.up);
        values[world::CrashRunwayHeading] = Value(Radians(crashRunway_.headingDegrees));
    }

    void WriteRewards(float* values) const noexcept
    {
        for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
        {
            values[world::EpisodeSums + channel] = Value(episodeRewardSums_[channel]);
            values[world::ActiveWeights + channel] = Value(config_.reward.weights[channel]);
        }
    }

    /** @brief The live values of the last action and the window of the last complete episode. */
    void WriteDiagnostics(float* values) const noexcept
    {
        values[world::DiagnosticsEnabled] = Value(Flag(config_.diagnostics.enabled));
        values[world::SaturationThreshold] = Value(config_.diagnostics.saturationThreshold);
        values[world::WindowSamples] = Value(static_cast<double>(lastWindow_.samples));
        double const samples = static_cast<double>(std::max<std::uint64_t>(lastWindow_.samples, 1));
        for (std::size_t slot = 0; slot < flight::ObservationSize; ++slot)
        {
            values[world::LiveObservation + slot] = liveObservation_[slot];
            values[world::ObservationMeanAbsolute + slot] = Value(lastWindow_.observationAbsoluteSums[slot] / samples);
            values[world::ObservationMaximumAbsolute + slot] = Value(lastWindow_.observationMaximumAbsolute[slot]);
            values[world::ObservationSaturations + slot] =
                Value(static_cast<double>(lastWindow_.observationSaturations[slot]));
        }
        for (std::size_t axis = 0; axis < flight::ContinuousActions; ++axis)
        {
            values[world::LiveRawAction + axis] = Value(liveRaw_[axis]);
            values[world::RawMeanAbsolute + axis] = Value(lastWindow_.rawAbsoluteSums[axis] / samples);
            values[world::AppliedMeanAbsolute + axis] = Value(lastWindow_.appliedAbsoluteSums[axis] / samples);
            values[world::RawSaturations + axis] = Value(static_cast<double>(lastWindow_.rawSaturations[axis]));
            values[world::AppliedSaturations + axis] = Value(static_cast<double>(lastWindow_.appliedSaturations[axis]));
        }
        for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
        {
            values[world::RewardMeans + channel] = Value(lastWindow_.rewardSums[channel] / samples);
        }
    }

    /** @brief Only airborne episodes measure the curriculum difficulty, as in FlightSession. */
    /** @brief Counts an airborne training episode; a promotion raises the start difficulty. */
    void AdvanceCurriculum()
    {
        if (!environment_.StartedAirborne())
        {
            return;
        }
        if (curriculum_.Record(environment_.Mission().outcome == flight::Outcome::Stopped))
        {
            environment_.SetDifficulty(curriculum_.Difficulty());
        }
    }

    static char const* SideName(flight::ApproachSide side) noexcept
    {
        if (side == flight::ApproachSide::Right)
        {
            return "right";
        }
        return "left";
    }

    static flight::ApproachSide ParseSide(std::string const& name)
    {
        if (name == "left")
        {
            return flight::ApproachSide::Left;
        }
        if (name == "right")
        {
            return flight::ApproachSide::Right;
        }
        throw std::runtime_error("FlightControl3D training state: unknown approach side '" + name + "'");
    }

    /** @brief Reads a TrainingState text: curriculum first (validated), then the stream. */
    void ResumeTrainingState(std::string const& text)
    {
        nlohmann::json state;
        try
        {
            state = nlohmann::json::parse(text);
        }
        catch (nlohmann::json::exception const& error)
        {
            throw std::runtime_error(std::string("FlightControl3D training state: ") + error.what());
        }
        auto const schema = state.value("schema", std::string());
        // A packaged actor has an evaluation difficulty, but owns no training RNG or promotion window.
        if (!learning_ && schema == "flight.evaluation_level/1")
        {
            auto const& level = state.at("level");
            if (!level.is_number_unsigned() || level.get<std::uint64_t>() >= curriculum_.Levels())
                throw std::runtime_error("FlightControl3D evaluation level exceeds the scenario curriculum");
            curriculum_.Restore(level.get<std::size_t>());
            return;
        }
        if (schema != TrainingStateSchema)
        {
            throw std::runtime_error("FlightControl3D training state: not a " + std::string(TrainingStateSchema));
        }
        flight::FlightEnvironment::TrainingStream stream;
        stream.nextSide = ParseSide(state.at("next_side").get<std::string>());
        stream.random = ppokit::artifacts::ParseRandomText(
            state.at("random").get<std::string>(), "FlightControl3D training state");
        curriculum_.Resume(state.at("level").get<std::size_t>(), state.at("window").at("episodes").get<std::size_t>(),
            state.at("window").at("stops").get<std::size_t>());
        environment_.ResumeTrainingStream(stream);
    }

    flight::Config config_;
    flight::FlightEnvironment environment_;
    flight::FlightCurriculum curriculum_;
    flight::Observation current_{};
    /** @brief The episode belongs to an evaluation campaign: it never moves the curriculum. */
    bool evaluating_ = false;
    /** @brief ppokit-run trains (set_learning 1, the default) or plays: only training moves the curriculum. */
    bool learning_ = true;
    /** @brief Weighted reward of each channel since the episode started. */
    std::array<double, flight::RewardCount> episodeRewardSums_{};
    /** @brief The last finished episode (Flying before the first ends). */
    flight::MissionState lastEpisode_{};
    /** @brief Diagnostics of the episode in progress, and of the last complete one (shown). */
    DiagnosticWindow window_{};
    DiagnosticWindow lastWindow_{};
    /** @brief Observation and raw action of the last action. */
    flight::Observation liveObservation_{};
    std::array<double, flight::ContinuousActions> liveRaw_{};
    EpisodeCounters counters_{};
    /** @brief The latest crash: its conditions, and its pose when it was finite (counted by crashEvent_). */
    bool hasCrash_ = false;
    std::uint64_t crashEvent_ = 0;
    flight::CrashDiagnostic crash_{};
    flight::AircraftState crashAircraft_{};
    flight::Runway crashRunway_{};
};

} // namespace

PPOKIT_ENVIRONMENT(FlightEnvironmentLibrary, "flight_control_3d", ppo::examples::flight::ObservationSize, ActionSize,
    ppo::examples::flight::RewardCount, ppo::examples::flight::world::Size)

/** @brief Resolve runner defaults from the same validated scenario used by the simulator. */
extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_run_profile_v1(char const* project, std::uint64_t update,
    PpokitRunProfileV1* profile, char* error, std::size_t errorSize)
{
    try
    {
        auto const config = CheckedContract(project, flight::LoadConfig(SettingsOf(project)));
        auto const& training = config.training;
        *profile = {training.maxUpdates, training.checkpointEvery, flight::TrainingSeed(training.seed, update),
            training.randomizeEnvironmentSeed, training.resume, config.evaluation.seed,
            static_cast<std::uint32_t>(config.evaluation.episodes)};
        return 0;
    }
    catch (std::exception const& failure)
    {
        ppokit::environment::sdk::CopyError(failure.what(), error, errorSize);
    }
    catch (...)
    {
        ppokit::environment::sdk::CopyError("unknown FlightControl3D run profile error", error, errorSize);
    }
    return 1;
}

/** @brief Declare the existing flight scenario editor's allowed fields in the provider. */
extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_scenario_editable_v1(char const* pointer)
{
    std::string_view const key(pointer);
    for (std::string_view const prefix : {"/initial/", "/mission/", "/runway/", "/runwayRandomization/"})
        if (key.substr(0, prefix.size()) == prefix) return 1;
    constexpr std::string_view curriculum = "/training/curriculum/";
    return key.substr(0, curriculum.size()) == curriculum && key != "/training/curriculum/levels" ? 1 : 0;
}

/** @brief Validate using exactly the runtime configuration parser and checkpoint contract. */
extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_validate_scenario_v1(
    char const* project, char const* candidate, char* error, std::size_t errorSize)
{
    try
    {
        (void)CheckedContract(project, flight::LoadConfig(candidate));
        return 0;
    }
    catch (std::exception const& failure)
    {
        ppokit::environment::sdk::CopyError(failure.what(), error, errorSize);
    }
    catch (...)
    {
        ppokit::environment::sdk::CopyError("unknown scenario validation error", error, errorSize);
    }
    return 1;
}

/** @brief Apply the inference start without counting it as a training or validation episode. */
extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_reset_inference_v1(PpokitEnvironment* handle,
    std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes, float* observations,
    std::uint8_t* available, float* world)
{
    return ppokit::environment::sdk::Guarded<FlightEnvironmentLibrary>(handle, [&](FlightEnvironmentLibrary& env) {
        env.ResetInference(seed, episode, episodes, observations, world);
        available[0] = 1;
    });
}
